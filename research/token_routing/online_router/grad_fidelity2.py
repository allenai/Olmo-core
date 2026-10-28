"""Gradient fidelity v2 (2026-10-28): does a truncated pass's parameter gradient predict TRAINING outcomes?

grad_fidelity.py measures one draw on one row: cos(g_drop, g_full). In training, random selection nearly ties the
router and keep 10% trains like keep 30%, so a one-row, one-draw cosine is not what training sees: training averages
over the batch's rows and over steps. This probe separates the two error sources of a stochastic rule:

  per row (K independent draws g_1..g_K of the rule, g_full the full pass, gbar = mean_k g_k)
    bias          cos(gbar, g_full); rel_bias = ||gbar - g_full|| / ||g_full||, both DEBIASED for the finite K
                  (E||gbar - g_full||^2 = bias^2 + var / K, so var_hat / K is subtracted; also the raw values)
    variance      var = mean_k ||g_k - gbar||^2 / ||g_full||^2 (unbiased, K - 1)
    single draw   mean_k cos(g_k, g_full) -- grad_fidelity.py's number
  deterministic rules (router top-k, oracle, first/last) have zero variance: bias only.
  batch level     cos(sum over a 16-row batch of g_rule, sum of g_full), per draw; and the same over 8 batches
                  (128 rows) accumulated -- the update training actually applies.

Settings (the online router's routing structure; keep = share of each document's body tokens that survives):
  2st_7_16 (7:0.2, 16:0.07 -- the recipe), 1st_7_10 (7:0.1), early drops 1st_{1,3,5}_10 (split after block 1/3/5
  at 0.1), 2st_{3,5}_16 (first stage at block 3 / 5, 0.2, then 16:0.07).
Rules: router (per-stage ridge on h_L + h_L*q features, target log ||dL/dh_L||, refit offline per checkpoint on
  training rows; stage 2 fit on the full pass), router_explore (the same with 10% of each stage's budget uniform
  random first, as in training), random, oracle (the true full-pass target at that layer), first_last.
Gradient views (fixed random coordinate samples, so per-draw / per-batch vectors fit in memory; cosines over millions
  of coordinates): "trained" = the parameters the recipe trains (LoRA A / B + the LM head for a LoRA checkpoint; all
  parameters for a dense / full-FT / base one) and "all" = every parameter.

    # one worker (one GPU):
    python grad_fidelity2.py --ckpt <dir> --tag recipe --shard <dir> --fit-offset 0 --row-offset 1464 \\
        --rows-per-task 32 --settings 2st_7_16,1st_7_10 --out part.json
    # driver: one worker per GPU over a plan, then merge
    python grad_fidelity2.py --plan plan.json --parallel 8 --out-dir /results
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_probe as OP  # noqa: E402

SETTINGS = {
    "2st_7_16": [(7, 0.2), (16, 0.07)],
    "1st_7_10": [(7, 0.1)],
    "1st_1_10": [(1, 0.1)],
    "1st_3_10": [(3, 0.1)],
    "1st_5_10": [(5, 0.1)],
    "2st_3_16": [(3, 0.2), (16, 0.07)],
    "2st_5_16": [(5, 0.2), (16, 0.07)],
}
DET_RULES = ("router", "oracle", "first_last")
STO_RULES = ("random", "router_explore")
EXPLORE = 0.1
BATCH, ACCUM = 16, 8


def log(m):
    print(f"[grad-fid2] {m}", flush=True)


# ---------------------------------------------------------------- model
def load_model(ckpt):
    """Qwen3.5-4B from a model_and_optim dir; a LoRA export (config.json early_lora block, or adapter keys in the
    checkpoint) gets its adapters installed before the load. Returns (model, lora_block or None)."""
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune
    from olmo_core.nn.early_lora import enable_from_config_block, lora_block_from_checkpoint
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig

    mo = ckpt if ckpt.rstrip("/").endswith("model_and_optim") else os.path.join(ckpt, "model_and_optim")
    root = os.path.dirname(mo.rstrip("/"))
    cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320, attn_backend=AttentionBackendName("flash_2"))
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    block = None
    cj = os.path.join(root, "config.json")
    if os.path.exists(cj):
        block = json.load(open(cj)).get("early_lora")
    if block is None:
        block = lora_block_from_checkpoint(root)
    if block:
        enable_from_config_block(model, block)
    t0 = time.time()
    load_model_and_optim_state(mo, model)
    log(f"loaded {mo} (lora={block}) in {time.time() - t0:.0f}s")
    model = model.cuda().to(torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    freeze_fla_length_autotune()
    return model, block


class Views:
    """Fixed random coordinate samples of the parameter gradient: 'trained' and 'all'."""

    def __init__(self, model, lora: bool, n_trained=8_000_000, n_all=16_000_000, seed=0):
        named = [(n, p) for n, p in model.named_parameters()]
        if lora:
            tr = [(n, p) for n, p in named if "lora_" in n or n.startswith("lm_head.")]
        else:
            tr = named
        self.spec = {}
        g = torch.Generator().manual_seed(seed)
        for view, ps, budget in (("trained", tr, n_trained), ("all", named, n_all)):
            tot = sum(p.numel() for _, p in ps)
            frac = min(1.0, budget / max(1, tot))
            items = []
            for n, p in ps:
                m = max(1, int(round(p.numel() * frac)))
                idx = (torch.randperm(p.numel(), generator=g)[:m] if frac >= 1.0 or p.numel() < 4 * m
                       else torch.randint(p.numel(), (m,), generator=g))
                items.append((p, idx.cuda()))
            self.spec[view] = items
            log(f"view {view}: {len(ps)} tensors, {tot / 1e6:.1f}M params, {sum(i.numel() for _, i in items) / 1e6:.1f}M sampled")
        self.params = [p for _, p in named]

    def take(self):
        out = {}
        for view, items in self.spec.items():
            parts = []
            for p, idx in items:
                if p.grad is None:
                    parts.append(torch.zeros(idx.numel(), device="cuda"))
                else:
                    parts.append(p.grad.view(-1)[idx].float())
            out[view] = torch.cat(parts)
        for p in self.params:
            p.grad = None
        return out


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-30))


# ---------------------------------------------------------------- passes
def grad_pass(model, row, stages):
    """Answer-CE forward + backward with parameter gradients. ``stages``: [(layer, kept positions (sorted, original
    indexing, nested))]; before block ``layer`` the hidden state is gathered to that stage's positions (original RoPE
    positions kept)."""
    from torch.utils.checkpoint import checkpoint

    dev = "cuda"
    ids = torch.from_numpy(row["ids"]).to(dev)
    lm = torch.from_numpy(row["lmask"]).to(dev)
    T = ids.numel()
    pred_full = torch.nonzero(lm[1:], as_tuple=True)[0]
    tgt = ids[pred_full + 1]
    fin = stages[-1][1] if stages else torch.arange(T, device=dev)
    pred = torch.searchsorted(fin, pred_full)
    assert bool((fin[pred] == pred_full).all()), "an answer-predicting position was dropped"
    x = ids[None]
    _, _, abk, pbk, lmk = model._prepare_inputs(x, None, logits_to_keep=pred[None], position_ids=None)
    at = {L: p for L, p in stages}
    cur = torch.arange(T, device=dev)
    h = model.embeddings(x)
    for key, block in model.blocks.items():
        i = int(key)
        if i in at:
            idx = torch.searchsorted(cur, at[i])
            h = h[:, idx]
            cur = at[i]
            abk = {**abk, "position_ids": cur[None]}
        kw = {**abk, **pbk.get(i, {})}
        h = checkpoint(lambda hh, b=block, k=kw: b(hh, **k), h, use_reentrant=False)
    ce = F.cross_entropy(model.lm_head(h, **lmk)[0].float(), tgt)
    ce.backward()
    return float(ce)


def doc_topk_subset(score, doc, cand, rel):
    """Among candidate body tokens (bool mask), each document keeps its top ceil(rel * |doc & cand|) by score."""
    out = torch.zeros_like(cand)
    ci = torch.nonzero(cand, as_tuple=True)[0]
    if ci.numel() == 0:
        return out
    s, d = score[ci], doc[ci]
    o1 = torch.argsort(-s)
    o2 = o1[torch.argsort(d[o1], stable=True)]
    ds = d[o2]
    cnt = torch.bincount(d, minlength=int(doc.max()) + 1)
    starts = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(ci.numel(), device=score.device) - starts[ds]
    keep = rank < torch.ceil(rel * cnt.float()).long()[ds]
    out[ci[o2[keep]]] = True
    return out


def select(rule, setting, ctx, gen):
    """Stage sets (original positions) for one draw of ``rule`` under ``setting``."""
    b, body_pos, doc, n = ctx["b"], ctx["body_pos"], ctx["doc"], ctx["n"]
    nonbody = torch.nonzero(~b, as_tuple=True)[0]
    surv = torch.ones(n, dtype=torch.bool, device=b.device)
    prev, sets = 1.0, []
    for L, k in SETTINGS[setting]:
        rel = min(1.0, k / prev)
        if rule in ("router", "router_explore"):
            sc = ctx["router"][L].clone()
        elif rule == "oracle":
            sc = ctx["oracle"][L].clone()
        elif rule == "random":
            sc = torch.rand(n, device=b.device, generator=gen).double()
        else:
            sc = ctx["first_last"].clone()
        if rule == "router_explore":
            ns = int(surv.sum())
            k_e = int(round(EXPLORE * math.ceil(rel * ns)))
            if k_e > 0:
                u = torch.rand(n, device=b.device, generator=gen)
                u = torch.where(surv, u, torch.full_like(u, 2.0))
                ex = torch.zeros_like(surv)
                ex[torch.argsort(u)[:k_e]] = True
                sc = torch.where(ex, torch.full_like(sc, float("inf")), sc)
        surv = doc_topk_subset(sc.float(), doc, surv, rel)
        keep = torch.sort(torch.cat([nonbody, body_pos[surv]])).values
        sets.append((L, keep))
        prev = k
    return sets, float(surv.float().mean())


# ---------------------------------------------------------------- worker
def worker(a):
    settings = [s for s in a.settings.split(",") if s]
    layers = sorted({L for s in settings for L, _ in SETTINGS[s]})
    OP.HIDDEN_LAYERS = tuple(sorted(set(OP.HIDDEN_LAYERS) | set(layers)))
    OP.TARGET_LAYERS = {**OP.TARGET_LAYERS, **{f"gnorm@L{L}": L for L in layers}}
    OP.FEATURE_SETS = {**OP.FEATURE_SETS, **{f"L{L}+q": (L, True, False) for L in layers}}
    torch.manual_seed(0)
    model, lora = load_model(a.ckpt)
    fit = OP.load_rows(a.shard, a.fit_rows, a.fit_offset, None)
    test = OP.load_rows(a.shard, a.rows_per_task, a.row_offset, None)
    # interleave tasks so every 16-row batch mixes them (training batches do)
    by = {}
    for r in test:
        by.setdefault(r["task"], []).append(r)
    test = [r for grp in zip(*by.values()) for r in grp]
    # ---- per-stage routers: ridge on h_L + h_L*q, target log ||dL/dh_L|| (row-centred), on training rows
    stats = {L: [0, 0] for L in layers}
    t0 = time.time()
    for i, row in enumerate(fit):
        o = OP.run_row(model, row)
        tg = OP.targets_of(o)
        parts = OP.feature_parts(row, o)
        bb = parts["body"]
        for L in layers:
            X = OP.design(parts, f"L{L}+q")
            y = tg[f"gnorm@L{L}"][bb].double()
            y = y - y.mean()
            stats[L][0] = stats[L][0] + X.T @ X
            stats[L][1] = stats[L][1] + X.T @ y
    W = {}
    for L, (G, bvec) in stats.items():
        A = G.clone()
        A.diagonal().add_(a.ridge * float(torch.diagonal(G).mean()))
        W[L] = torch.linalg.solve(A, bvec)
    log(f"fit {len(fit)} rows for layers {layers} in {time.time() - t0:.0f}s")
    del stats
    views = Views(model, lora=bool(lora))
    rules = list(DET_RULES) + list(STO_RULES)
    KB = min(a.k, a.kb)

    def zeros():
        return {v: torch.zeros(sum(i.numel() for _, i in it), device="cuda") for v, it in views.spec.items()}

    def nb(rule):
        return KB if rule in STO_RULES else 1

    bsum = {(s, r, k): zeros() for s in settings for r in rules for k in range(nb(r))}
    asum = {(s, r, k): zeros() for s in settings for r in rules for k in range(nb(r))}
    bfull, afull = zeros(), zeros()
    rows_out, batch_out, accum_out = [], [], []
    nb_done = 0
    t0 = time.time()
    for i, row in enumerate(test):
        for p in model.parameters():
            p.requires_grad_(False)
        o = OP.run_row(model, row)
        tg = OP.targets_of(o)
        parts = OP.feature_parts(row, o)
        bm = parts["body"]
        body_pos = torch.nonzero(bm, as_tuple=True)[0]
        doc = parts["cid"][bm].long()
        doc = doc - doc.min()
        n = body_pos.numel()
        cnt = torch.bincount(doc)
        starts = torch.cumsum(cnt, 0) - cnt
        j = torch.arange(n, device=bm.device) - starts[doc]
        r_ = cnt[doc] - 1 - j
        ctx = dict(b=bm, body_pos=body_pos, doc=doc, n=n,
                   router={L: OP.design(parts, f"L{L}+q") @ W[L] for L in layers},
                   oracle={L: tg[f"gnorm@L{L}"][bm].double() for L in layers},
                   first_last=(-torch.minimum(j, r_).double() + 1e-6 * torch.rand(n, device=bm.device).double()))
        for p in model.parameters():
            p.requires_grad_(True)
        ce_full = grad_pass(model, row, [])
        gF = views.take()
        for v in gF:
            bfull[v] += gF[v]
            afull[v] += gF[v]
        base = dict(task=row["task"], row=i, ce_full=ce_full, n_body=n, T=int(row["ids"].shape[0]), tag=a.tag)
        for s in settings:
            for rule in rules:
                K = a.k if rule in STO_RULES else 1
                gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"{a.seed}/{a.tag}/{i}/{s}/{rule}".encode()))
                gsum = {v: None for v in views.spec}
                sq = {v: 0.0 for v in views.spec}
                cos1 = {v: [] for v in views.spec}
                ces, kept = [], []
                for k in range(K):
                    sets, kf = select(rule, s, ctx, gen)
                    ces.append(grad_pass(model, row, sets) - ce_full)
                    kept.append(kf)
                    g = views.take()
                    for v in g:
                        gsum[v] = g[v].clone() if gsum[v] is None else gsum[v] + g[v]
                        sq[v] += float(g[v] @ g[v])
                        cos1[v].append(cos(g[v], gF[v]))
                        if k < nb(rule):
                            bsum[(s, rule, k)][v] += g[v]
                rec = dict(base, setting=s, rule=rule, K=K, dce=float(np.mean(ces)), kept_frac=float(np.mean(kept)))
                for v in views.spec:
                    gb = gsum[v] / K
                    f2 = float(gF[v] @ gF[v])
                    var = (sq[v] / K - float(gb @ gb)) * K / max(1, K - 1) if K > 1 else 0.0
                    dev2 = float((gb - gF[v]) @ (gb - gF[v]))
                    gb2_db = max(float(gb @ gb) - var / K, 1e-30)
                    rec.update({
                        f"{v}_cos1": float(np.mean(cos1[v])),
                        f"{v}_cos_bar_raw": cos(gb, gF[v]),
                        f"{v}_cos_bar": float(gb @ gF[v]) / math.sqrt(gb2_db * f2),
                        f"{v}_relbias_raw": math.sqrt(dev2 / f2),
                        f"{v}_relbias": math.sqrt(max(dev2 - var / K, 0.0) / f2),
                        f"{v}_var": var / f2,
                        f"{v}_norm_ratio": math.sqrt(float(gb @ gb) / f2),
                    })
                rows_out.append(rec)
        # batch boundaries
        if (i + 1) % BATCH == 0:
            for (s, rule, k), d in bsum.items():
                rec = dict(tag=a.tag, setting=s, rule=rule, draw=k, batch=nb_done)
                for v in d:
                    rec[f"{v}_cos"] = cos(d[v], bfull[v])
                    asum[(s, rule, k)][v] += d[v]
                    d[v].zero_()
                batch_out.append(rec)
            for v in bfull:
                bfull[v].zero_()
            nb_done += 1
            if nb_done % ACCUM == 0:
                for (s, rule, k), d in asum.items():
                    rec = dict(tag=a.tag, setting=s, rule=rule, draw=k, group=nb_done // ACCUM - 1)
                    for v in d:
                        rec[f"{v}_cos"] = cos(d[v], afull[v])
                        d[v].zero_()
                    accum_out.append(rec)
                for v in afull:
                    afull[v].zero_()
        el = time.time() - t0
        log(f"{a.tag} row {i + 1}/{len(test)} ({row['task']}, T={base['T']}) {el:.0f}s, "
            f"ETA {el / (i + 1) * (len(test) - i - 1):.0f}s")
    if nb_done % ACCUM:  # a partial accumulation group (fewer than 8 batches)
        for (s, rule, k), d in asum.items():
            rec = dict(tag=a.tag, setting=s, rule=rule, draw=k, group=nb_done // ACCUM, n_batches=nb_done % ACCUM)
            for v in d:
                rec[f"{v}_cos"] = cos(d[v], afull[v])
            accum_out.append(rec)
    json.dump(dict(tag=a.tag, ckpt=a.ckpt, shard=a.shard, lora=lora, settings=settings, K=a.k, KB=KB,
                   row_offset=a.row_offset, rows_per_task=a.rows_per_task, fit_offset=a.fit_offset, fit_rows=a.fit_rows,
                   rows=rows_out, batches=batch_out, accum=accum_out), open(a.out, "w"))
    log(f"wrote {a.out}")


# ---------------------------------------------------------------- driver + merge
def summarize(parts):
    out = []
    rows = [r for p in parts for r in p["rows"]]
    keys = sorted({(r["tag"], r["setting"], r["rule"]) for r in rows})
    med = lambda v: float(np.median(v)) if v else float("nan")  # noqa: E731
    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    for tag, s, rule in keys:
        rr = [r for r in rows if (r["tag"], r["setting"], r["rule"]) == (tag, s, rule)]
        bb = [b for p in parts for b in p["batches"] if (b["tag"], b["setting"], b["rule"]) == (tag, s, rule)]
        aa = [x for p in parts for x in p["accum"] if (x["tag"], x["setting"], x["rule"]) == (tag, s, rule)]
        rec = dict(tag=tag, setting=s, rule=rule, eval_size=len(rr), n_batch_draws=len(bb), n_accum_draws=len(aa),
                   dce=med([r["dce"] for r in rr]), kept_frac=mean([r["kept_frac"] for r in rr]))
        for v in ("trained", "all"):
            for m in ("cos1", "cos_bar", "cos_bar_raw", "relbias", "relbias_raw", "var", "norm_ratio"):
                k = f"{v}_{m}"
                if rr and k in rr[0]:
                    rec[k] = med([r[k] for r in rr])
            rec[f"{v}_batch_cos"] = mean([b[f"{v}_cos"] for b in bb if f"{v}_cos" in b])
            rec[f"{v}_accum_cos"] = mean([x[f"{v}_cos"] for x in aa if f"{v}_cos" in x])
        out.append(rec)
    return out


def driver(a):
    plan = json.load(open(a.plan))
    os.makedirs(a.out_dir, exist_ok=True)
    G = a.parallel
    lanes = [[] for _ in range(G)]
    for jdx, job in enumerate(plan["jobs"]):
        lanes[job.get("gpu", jdx % G)].append((jdx, job))
    procs = []
    for gi, lane in enumerate(lanes):
        if not lane:
            continue
        cmds = []
        for jdx, job in lane:
            out = os.path.join(a.out_dir, f"part_{jdx:02d}_{job['tag']}.json")
            args = [sys.executable, "-u", os.path.abspath(__file__), "--ckpt", job["ckpt"], "--tag", job["tag"],
                    "--shard", job.get("shard", plan["shard"]), "--settings", job["settings"],
                    "--row-offset", str(job["row_offset"]), "--rows-per-task", str(job["rows_per_task"]),
                    "--fit-offset", str(job.get("fit_offset", plan.get("fit_offset", 0))),
                    "--fit-rows", str(job.get("fit_rows", plan.get("fit_rows", 32))),
                    "--k", str(job.get("k", plan.get("k", 8))), "--out", out]
            cmds.append(" ".join(f"'{x}'" for x in args) + f" > {a.out_dir}/log_{jdx:02d}.txt 2>&1")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi))
        procs.append(subprocess.Popen(["bash", "-c", " ; ".join(cmds)], env=env))
        log(f"gpu {gi}: {len(lane)} job(s)")
    for p in procs:
        p.wait()
    parts = [json.load(open(os.path.join(a.out_dir, f))) for f in sorted(os.listdir(a.out_dir))
             if f.startswith("part_") and f.endswith(".json")]
    summ = summarize(parts)
    json.dump(dict(plan=plan, summary=summ), open(os.path.join(a.out_dir, "merged.json"), "w"), indent=1)
    for s in summ:
        log(f"{s['tag']:10s} {s['setting']:9s} {s['rule']:15s} n={s['eval_size']:3d} "
            f"cos1={s.get('trained_cos1', float('nan')):+.3f} cos_bar={s.get('trained_cos_bar', float('nan')):+.3f} "
            f"var={s.get('trained_var', float('nan')):.3f} batch={s['trained_batch_cos']:+.3f} "
            f"accum={s['trained_accum_cos']:+.3f} | all: cos1={s.get('all_cos1', float('nan')):+.3f} "
            f"cos_bar={s.get('all_cos_bar', float('nan')):+.3f} accum={s['all_accum_cos']:+.3f} dCE={s['dce']:+.3f}")
    for f in sorted(os.listdir(a.out_dir)):
        if f.startswith("log_"):
            tail = open(os.path.join(a.out_dir, f)).read().splitlines()[-3:]
            log(f"{f}: " + " | ".join(tail))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--out-dir", default="/results")
    ap.add_argument("--ls", default="", help="comma-separated dirs to list before anything else (smoke)")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tag", default="ckpt")
    ap.add_argument("--shard", default="")
    ap.add_argument("--settings", default="2st_7_16,1st_7_10")
    ap.add_argument("--row-offset", type=int, default=1464)
    ap.add_argument("--rows-per-task", type=int, default=32)
    ap.add_argument("--fit-offset", type=int, default=0)
    ap.add_argument("--fit-rows", type=int, default=32)
    ap.add_argument("--k", type=int, default=8, help="draws per stochastic rule and row")
    ap.add_argument("--kb", type=int, default=4, help="draws kept for the batch / 8-batch sums")
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="part.json")
    a = ap.parse_args()
    for d in [x for x in a.ls.split(",") if x]:
        try:
            log(f"ls {d}: {sorted(os.listdir(d))[:400]}")
        except OSError as exc:
            log(f"ls {d}: {exc}")
    if a.plan:
        driver(a)
    elif a.ckpt:
        worker(a)


if __name__ == "__main__":
    main()
