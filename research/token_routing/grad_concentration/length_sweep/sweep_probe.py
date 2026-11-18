"""Gradient concentration vs context length: the concentration-table protocol of ``../probe.py`` at one row length.

What is measured (unchanged from ``probe.py`` / ``analyze.py``, the source of the hub table "Global oracle / per-row
quota / per-document quota / random"):
  * gradient = the summed-answer-CE weight gradient of every ``nn.Linear`` in blocks 7-31 (teacher forced, no-cot,
    frozen checkpoint); token t's term in a linear is rank 1 (delta_t x_t^T), energy e_t = |delta_t|^2 |x_t|^2.
  * score of a token = sum of e_t over the linears of blocks >= 7 (``down7``); only document-body tokens are ranked
    (``body`` view), every other token (prompt, markers, query, answer) is always kept.
  * one step = 16 rows (4 per task, interleaved exactly as probe.py); four selections per step:
      down7_body  global oracle: top-k by score across all 16 rows
      rowq7_body  per-row quota: each row keeps its own top-k share
      docq7_body  per-document quota: each document of each row keeps its own top-k share
      rand7_body  random
  * update view: tokens' rank-1 terms go into TensorSketch accumulators per rank bin (dim 32768, the same hashes,
    seeds 1009 + linear index over the model's full linear list); cos(G_S(k), G) summed over the (block, mixer|mlp)
    groups of blocks 7-31; k for cos 0.95 / 0.9 by linear interpolation on the realized kept-body share (analyze.py's
    ``kfrac`` axis).
Changes, none of which alter the quantity:
  * k grid extended below 0.05% (0.01 / 0.02 / 0.03%): the global oracle at long rows needs fewer than 0.05%.
  * only the four rankings above (scope 16); blocks 0-6 run without autograd (their linears do not enter any
    blocks-7-31 quantity; dL/d(output) of a block >= 7 linear does not depend on blocks < 7 having grads).
  * extra: per-row self accumulators for the per-row quota bins: cos of a row's own top-k update with that row's own
    full update (row-level sparsity per row, hence per task).
  * blocks can be spread over several GPUs (``--devices``) so one row longer than one GPU's memory fits; YaRN
    (``--yarn``) for rows past the base model's 262,144 positions.
Results are written per step (``<out>/step_XXXX.npz`` + ``rows_XXXX.json``), so a preempted job resumes.

    python sweep_probe.py --ckpt <model_and_optim> --rows-from tok --tok-root /data/prasann/gradlen/tok/both \
        --rung 32k --rows 128 --out /data/prasann/gradlen/results/<label> [--steps 0:8] [--devices 0,1]
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
GC = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(GC, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, GC)

import probe as P  # noqa: E402  (data helpers, body mask, checkpoint finder: reused unchanged)

KS = (0.0001, 0.0002, 0.0003) + P.KS
NB = 32
SPLIT = 7
RANKS = ("down7_body", "rowq7_body", "docq7_body", "rand7_body")
STEP_ROWS = 16


def log(m):
    print(f"[gsl] {m}", flush=True)


# ---------------------------------------------------------------------------------------- rankings (probe.py, new KS)
def rank_bins(score, elig):
    n = int(elig.sum())
    bins = torch.zeros(score.numel(), dtype=torch.long)
    idx = torch.nonzero(elig, as_tuple=True)[0]
    order = torch.argsort(score[idx], descending=True)
    frac = (torch.arange(n, dtype=torch.float64) + 1) / max(n, 1)
    bins[idx[order]] = torch.searchsorted(torch.tensor(KS, dtype=torch.float64), frac) + 1
    return bins


def rank_bins_grouped(score, elig, grp):
    bins = torch.zeros(score.numel(), dtype=torch.long)
    idx = torch.nonzero(elig, as_tuple=True)[0]
    s, g = score[idx].double(), torch.unique(grp[idx], return_inverse=True)[1]
    o = torch.argsort(-s)
    o = o[torch.argsort(g[o], stable=True)]
    gs = g[o]
    cnt = torch.bincount(gs)
    start = torch.cumsum(cnt, 0) - cnt
    frac = (torch.arange(len(o), dtype=torch.float64) - start[gs] + 1) / cnt[gs]
    bins[idx[o]] = torch.searchsorted(torch.tensor(KS, dtype=torch.float64), frac) + 1
    return bins


def energy_curve(score, elig):
    s = torch.sort(score[elig].double(), descending=True).values
    if s.numel() == 0:
        return np.zeros(len(KS))
    c = torch.cumsum(s, 0)
    i = torch.tensor([max(1, int(np.ceil(k * s.numel()))) - 1 for k in KS])
    return (c[i] / c[-1].clamp(min=1e-300)).numpy()


# --------------------------------------------------------------------------------------------------------- model
def set_yarn(cfg, factor, old_len):
    from olmo_core.nn.rope import RoPEConfig, YaRNRoPEScalingConfig

    n = 0
    seen = set()

    def walk(o):
        nonlocal n
        if id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, RoPEConfig):
            o.scaling = YaRNRoPEScalingConfig(factor=factor, old_context_len=old_len)
            n += 1
            return
        if dataclasses.is_dataclass(o):
            for f in dataclasses.fields(o):
                walk(getattr(o, f.name))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(cfg)
    return n


def build_model(ckpt, devices, yarn=0.0, yarn_old=262144):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320, attn_backend=AttentionBackendName("flash_2"))
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    if yarn:
        log(f"YaRN factor {yarn} (old_context_len {yarn_old}) on {set_yarn(cfg, yarn, yarn_old)} RoPE configs")
    model = cfg.build(init_device="cpu")
    t0 = time.time()
    load_model_and_optim_state(ckpt, model)
    log(f"loaded {ckpt} in {time.time() - t0:.0f}s")
    model = model.to(torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    devs = [torch.device(f"cuda:{d}") for d in devices]
    place = {}
    keys = list(model.blocks.keys())
    # blocks < 7 are cheap (no autograd) -> put them with the first share; spread 7..31 evenly
    late = [k for k in keys if int(k) >= SPLIT]
    for k in keys:
        if int(k) < SPLIT:
            place[k] = devs[0]
    for i, k in enumerate(late):
        place[k] = devs[i * len(devs) // len(late)]
    model.embeddings.to(devs[0])
    for k, b in model.blocks.items():
        b.to(place[k])
    model.lm_head.to(devs[-1])
    model._gsl_place = place
    model._gsl_dev0 = devs[0]
    for d in devs:
        with torch.cuda.device(d):
            torch.cuda.reset_peak_memory_stats()
    log(f"froze FLA length autotune on {freeze_fla_length_autotune()} kernels; blocks on "
        f"{sorted(set(str(v) for v in place.values()))}")
    return model


def to_dev(o, d):
    if torch.is_tensor(o):
        return o.to(d, non_blocking=True)
    if isinstance(o, dict):
        return {k: to_dev(v, d) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(to_dev(v, d) for v in o)
    return o


def row_loss(model, row):
    """probe.row_loss with blocks placed on several devices and blocks < 7 outside autograd. Blocks >= 7 run under
    non-reentrant checkpointing with early stop off (every linear re-runs in the recompute, as in probe.py)."""
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    d0 = model._gsl_dev0
    x = torch.from_numpy(row["ids"]).to(d0)
    lm = torch.from_numpy(row["lmask"]).to(d0)
    pred = torch.nonzero(lm[1:], as_tuple=True)[0]
    tgt = x[pred + 1]
    with torch.cuda.device(d0):
        _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    kw_cache = {}
    with torch.no_grad(), torch.cuda.device(d0):
        h = model.embeddings(x[None])
    with set_checkpoint_early_stop(False):
        for key, block in model.blocks.items():
            d = model._gsl_place[key]
            if d not in kw_cache:
                kw_cache[d] = to_dev(abk, d)
            kw = {**kw_cache[d], **to_dev(pbk.get(int(key), {}), d)}
            h = h.to(d)
            with torch.cuda.device(d):
                if int(key) < SPLIT:
                    with torch.no_grad():
                        h = block(h, **kw)
                    if int(key) == SPLIT - 1:
                        h = h.detach().requires_grad_(True)
                else:
                    h = checkpoint(lambda hh, b=block, k=kw: b(hh, **k), h, use_reentrant=False)
    dl = next(model.lm_head.parameters()).device
    with torch.cuda.device(dl):
        logits = model.lm_head(h.to(dl), **to_dev(lmk, dl))[0].float()
        loss = F.cross_entropy(logits, tgt.to(dl), reduction="sum")
    return loss, int(pred.numel())


# ----------------------------------------------------------------------------------------------------------- tap
class Tap(P.Tap):
    """probe.Tap (same hashes; recompute pairing of inputs and output grads) for the linears of blocks >= 7 only, on
    their own devices. ``mode``: 'energy' -> self.E[idx] = e_t (T,) on CPU; 'sketch' -> rank-1 terms into the
    accumulators of ``self.active[idx]`` = [(acc, bins on that device)]."""

    def __init__(self, lins, m, chunk):
        self.lins, self.m, self.chunk = lins, m, chunk
        self.mode = None
        self.E = {}
        self.fired = 0
        self.active = {}
        self.exact, self.exact_names, self.check = {}, [], {}
        self.ckpt, self.recompute = True, False
        self.xs, self.gs = {}, {}
        self.hash = {}
        for ln in lins:
            dev = ln.mod.weight.device
            g = torch.Generator(device="cpu").manual_seed(1009 + ln.idx)  # idx over the FULL linear list: as probe.py
            o, i = ln.mod.out_features, ln.mod.in_features
            self.hash[ln.idx] = (torch.randint(0, m, (o,), generator=g).to(dev),
                                 (torch.randint(0, 2, (o,), generator=g) * 2 - 1).float().to(dev),
                                 torch.randint(0, m, (i,), generator=g).to(dev),
                                 (torch.randint(0, 2, (i,), generator=g) * 2 - 1).float().to(dev))
            ln.mod.register_forward_hook(self._fwd(ln.idx))

    def _grad(self, i, x, g):
        self.fired += 1
        x = x.reshape(-1, x.shape[-1])
        g = g.detach().reshape(-1, g.shape[-1])
        if self.mode == "energy":  # chunked over tokens: no full-length fp32 copies of g / x at long rows
            self.E[i] = torch.cat([(g[c0:c0 + self.chunk].float().square().sum(-1) *
                                    x[c0:c0 + self.chunk].float().square().sum(-1)).cpu()
                                   for c0 in range(0, x.shape[0], self.chunk)])
            return
        T = x.shape[0]
        h1, s1, h2, s2 = self.hash[i]
        m = self.m
        torch.backends.cuda.matmul.allow_tf32 = True
        for c0 in range(0, T, self.chunk):
            c1 = min(T, c0 + self.chunk)
            gc, xc = g[c0:c1].float(), x[c0:c1].float()
            cs = torch.zeros(c1 - c0, m, device=g.device).index_add_(1, h1, gc * s1)
            Pf = torch.fft.rfft(cs, dim=1)
            cs.zero_().index_add_(1, h2, xc * s2)
            Pf *= torch.fft.rfft(cs, dim=1)
            del cs
            Pr = torch.view_as_real(Pf).reshape(c1 - c0, -1)
            for acc, bins in self.active.get(i, ()):
                oh = F.one_hot(bins[c0:c1], acc.shape[0]).T.float()
                acc.addmm_(oh, Pr)
            del Pf, Pr
        torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------------------------------------------------------- data
def load_tok_rows(root, tasks, rung, n, max_len, task_rungs=None):
    """``n`` rows per task from converter shard dirs ``<root>/<task>_<rung>`` (``task_rungs`` overrides the rung per
    task), in file order."""
    rows = {}
    for t in tasks:
        d = f"{root}/{t}_{(task_rungs or {}).get(t, rung)}"
        tok, msk, st, en = P.read_parts(d, "token_ids_part_*.npy", "labels_mask_*.npy")
        out = []
        for k in range(len(st)):
            s, e = int(st[k]), int(en[k]) + 1
            if e - s > max_len or not msk[s:e].any():
                continue
            out.append(dict(task=t, k=k, ids=tok[s:e].astype(np.int64), lmask=msk[s:e].copy(),
                            md5=hashlib.md5(tok[s:e].tobytes()).hexdigest()))
            if len(out) == n:
                break
        rows[t] = out
        log(f"{t}@{rung}: {len(out)} rows from {d} ({len(st)} in shard)")
    return rows


# ---------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default="contradiction,nq,outlier,oolong")
    ap.add_argument("--rows-from", choices=("fresh", "tok"), default="tok",
                    help="fresh = probe.py's <task>_fresh pools past --row-offset (the original table's rows); "
                         "tok = gen_rows.py converter shards")
    ap.add_argument("--shards", default="/data/prasann/gradconc/shards")
    ap.add_argument("--seen-shards", default="mt4f7808,mt4s1952")
    ap.add_argument("--row-offset", type=int, default=7808)
    ap.add_argument("--tok-root", default="/data/prasann/gradlen/tok/both")
    ap.add_argument("--rung", default="8k")
    ap.add_argument("--task-rungs", default="", help="per-task rung override, e.g. contradiction=448k,outlier=448k")
    ap.add_argument("--rows", type=int, default=128, help="rows per task")
    ap.add_argument("--max-len", type=int, default=2_000_000)
    ap.add_argument("--steps", default="", help="a:b = only steps a..b-1 (sharding); default all")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--yarn", type=float, default=0.0, help="YaRN factor over 262,144 positions (0 = native RoPE)")
    ap.add_argument("--sketch-dim", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--no-row-self", action="store_true")
    ap.add_argument("--memprobe", type=int, default=0,
                    help="memory probe: one synthetic row of this many tokens (a real row tiled), pass 1 + pass 2, "
                         "report peak memory per device, exit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    tasks = a.tasks.split(",")
    os.makedirs(a.out, exist_ok=True)
    devices = [int(d) for d in a.devices.split(",")]

    if a.rows_from == "fresh":
        seen = [os.path.join(a.shards, s) for s in a.seen_shards.split(",") if s]
        rows_by_task, _ = P.load_rows(a.shards, tasks, a.rows, a.row_offset, a.max_len, seen)
    else:
        tr = dict(kv.split("=") for kv in a.task_rungs.split(",") if kv)
        rows_by_task = load_tok_rows(a.tok_root, tasks, a.rung, a.rows, a.max_len, tr)
    n_per = min(len(v) for v in rows_by_task.values())
    per = STEP_ROWS // len(tasks)
    nsteps = n_per // per
    order = []
    for j in range(nsteps):
        for t in tasks:
            order += rows_by_task[t][j * per:(j + 1) * per]
    s0, s1 = (0, nsteps)
    if a.steps:
        s0, s1 = (int(v) for v in a.steps.split(":"))
        s1 = min(s1, nsteps)
    if order:
        log(f"{len(order)} rows = {nsteps} steps of {STEP_ROWS}; this job: steps {s0}..{s1 - 1}; lengths "
            f"{min(len(r['ids']) for r in order)}-{max(len(r['ids']) for r in order)}")

    model = build_model(P.find_ckpt(a.ckpt), devices, a.yarn)
    lins_all = P.list_linears(model)
    lins = [ln for ln in lins_all if ln.block >= SPLIT]
    tap = Tap(lins, a.sketch_dim, a.chunk)
    nm = a.sketch_dim // 2 + 1
    nb = len(KS) + 1
    groups = sorted({ln.group for ln in lins})
    acc = {r: {g: None for g in groups} for r in RANKS}
    gdev = {ln.group: ln.mod.weight.device for ln in lins}
    for r in RANKS:
        for g in groups:
            acc[r][g] = torch.zeros(nb, 2 * nm, device=gdev[g])
    rself = None if a.no_row_self else {
        q: {g: torch.zeros(nb, 2 * nm, device=gdev[g]) for g in groups} for q in range(STEP_ROWS)}
    log(f"{len(lins)} linears in blocks >= {SPLIT}, {len(groups)} groups; accumulators "
        f"{(len(RANKS) + (0 if rself is None else STEP_ROWS)) * len(groups) * nb * 2 * nm * 4 / 2**30:.1f} GiB")

    def peaks():
        return {str(d): round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in devices}

    if a.memprobe:
        base = next(iter(rows_by_task.values()))[0]
        reps = int(np.ceil(a.memprobe / len(base["ids"])))
        ids = np.tile(base["ids"][:-1], reps)[:a.memprobe - len(base["ids"])]
        row = dict(ids=np.concatenate([ids, base["ids"]]),
                   lmask=np.concatenate([np.zeros(len(ids), bool), base["lmask"]]))
        res = {"T": int(len(row["ids"])), "devices": devices, "yarn": a.yarn}
        t0 = time.time()
        try:
            tap.mode = "energy"
            loss, _ = row_loss(model, row)
            tap.backward(loss)
            res["pass1_s"], res["pass1_peak_gib"] = round(time.time() - t0, 1), peaks()
            bins = torch.ones(len(row["ids"]), dtype=torch.long)
            tap.mode = "sketch"
            tap.active = {ln.idx: [(acc[r][ln.group], bins.to(gdev[ln.group])) for r in RANKS] +
                          ([] if rself is None else [(rself[0][ln.group], bins.to(gdev[ln.group]))]) for ln in lins}
            t1 = time.time()
            loss, _ = row_loss(model, row)
            tap.backward(loss)
            res["pass2_s"], res["peak_gib"], res["ok"] = round(time.time() - t1, 1), peaks(), True
        except torch.OutOfMemoryError as exc:
            res.update(ok=False, error=str(exc)[:400], peak_gib=peaks())
        log(f"MEMPROBE {json.dumps(res)}")
        json.dump(res, open(os.path.join(a.out, f"memprobe_{a.memprobe}.json"), "w"), indent=1)
        return

    rng = torch.Generator().manual_seed(1)
    t0 = time.time()
    done_rows = 0
    for J in range(s0, s1):
        fn = os.path.join(a.out, f"step_{J:04d}.npz")
        if os.path.exists(fn):
            log(f"step {J} exists, skipping")
            continue
        batch = order[J * STEP_ROWS:(J + 1) * STEP_ROWS]
        # ---- pass 1: energies -> down7 score per token
        tap.mode = "energy"
        S, body, cids, meta = [], [], [], []
        for ri, row in enumerate(batch):
            tap.fired, tap.E = 0, {}
            loss, nans = row_loss(model, row)
            tap.backward(loss)
            if tap.fired != len(lins):
                raise SystemExit(f"hooks fired {tap.fired} times for {len(lins)} linears")
            x = torch.from_numpy(row["ids"])
            bm, cid = P.body_mask(x.cuda(devices[0]))
            bm, cid = bm.cpu(), cid.cpu()
            sc = torch.stack([tap.E[ln.idx] for ln in lins]).sum(0)
            S.append(sc)
            body.append(bm)
            cids.append(cid)
            ndoc = int((x == P.DOC_START).sum())
            meta.append(dict(task=row["task"], k=int(row["k"]), T=int(len(row["ids"])), n_body=int(bm.sum()),
                             n_docs=ndoc, n_answer=nans, ce=float(loss) / max(nans, 1), md5=row["md5"],
                             body_energy_share=float(sc[bm].sum() / sc.sum()),
                             energy_curve=energy_curve(sc, bm).tolist()))
            done_rows += 1
            if done_rows in (1, 2, 5, 10) or done_rows % 16 == 0:
                el = time.time() - t0
                left = (s1 - J) * STEP_ROWS * 2 - (ri + 1)
                log(f"[pass1] step {J} row {ri} ({row['task']} T={len(row['ids'])} CE={meta[-1]['ce']:.3f} "
                    f"body={float(bm.float().mean()):.2f} docs={ndoc}) {el:.0f}s; peak {peaks()}; "
                    f"ETA ~{el / done_rows / 2.5 * left:.0f}s")
        # ---- bins
        lens = [len(r["ids"]) for r in batch]
        Scat, Bcat = torch.cat(S), torch.cat(body)
        rowid = torch.cat([torch.full((n_,), q) for q, n_ in enumerate(lens)])
        docid = torch.unique(rowid * 10_000_000 + torch.cat(cids).clamp(min=0), return_inverse=True)[1]
        offs = np.cumsum([0] + lens)
        out = {}
        bins_by = {}
        for r in RANKS:
            if r.startswith("rand"):
                sc = torch.rand(Bcat.numel(), generator=rng)
                bins = rank_bins(sc, Bcat)
                en = np.zeros(len(KS))
            elif r.startswith(("rowq", "docq")):
                sc = Scat
                bins = rank_bins_grouped(sc, Bcat, rowid if r.startswith("rowq") else docid)
                se = sc[Bcat].double()
                en = np.array([float(se[bins[Bcat] <= j + 1].sum() / se.sum()) for j in range(len(KS))])
            else:
                sc = Scat
                bins = rank_bins(sc, Bcat)
                en = energy_curve(sc, Bcat)
            be = bins[Bcat]
            out[f"{r}|kfrac"] = np.array([float((be <= j + 1).float().mean()) for j in range(len(KS))])
            out[f"{r}|eshare"] = en
            out[f"{r}|nelig"] = np.array([int(Bcat.sum()), Bcat.numel()])
            bins_by[r] = bins
        # per-row kept-body share at each rowq bin (x axis of the per-row self curves)
        rk = []
        for q in range(len(batch)):
            bq = bins_by["rowq7_body"][offs[q]:offs[q + 1]][body[q]]
            rk.append([float((bq <= j + 1).float().mean()) for j in range(len(KS))])
        # ---- pass 2: sketches
        tap.mode = "sketch"
        for ri, row in enumerate(batch):
            sl = slice(offs[ri], offs[ri + 1])
            bd = {}
            act = {}
            for ln in lins:
                d = gdev[ln.group]
                for r in RANKS:
                    if (r, d) not in bd:
                        bd[(r, d)] = bins_by[r][sl].to(d)
                    act.setdefault(ln.idx, []).append((acc[r][ln.group], bd[(r, d)]))
                if rself is not None:
                    act[ln.idx].append((rself[ri][ln.group], bd[("rowq7_body", d)]))
            tap.active = act
            loss, _ = row_loss(model, row)
            tap.backward(loss)
            tap.active = {}
        # ---- finalize: cumulative dot / sq / full per group (analyze.py's layout), then zero
        def fin(A):
            Sx = torch.fft.irfft(torch.view_as_complex(A.view(nb, nm, 2)), n=a.sketch_dim, dim=1).double()
            C = torch.cumsum(Sx, 0)
            return (C @ C[-1]).cpu().numpy(), (C * C).sum(1).cpu().numpy(), float(C[-1] @ C[-1])

        for r in RANKS:
            dot, sq, full = np.zeros((2 * NB, nb)), np.zeros((2 * NB, nb)), np.zeros(2 * NB)
            for g in groups:
                dot[g], sq[g], full[g] = fin(acc[r][g])
                acc[r][g].zero_()
            out[f"{r}|dot"], out[f"{r}|sq"], out[f"{r}|full"] = dot, sq, full
        if rself is not None:
            rd, rs, rf = np.zeros((len(batch), nb)), np.zeros((len(batch), nb)), np.zeros(len(batch))
            for q in range(len(batch)):
                for g in groups:
                    d_, s_, f_ = fin(rself[q][g])
                    rd[q] += d_
                    rs[q] += s_
                    rf[q] += f_
                    rself[q][g].zero_()
            out["rowself|dot"], out["rowself|sq"], out["rowself|full"] = rd, rs, rf
            out["rowself|kfrac"] = np.array(rk)
        np.savez_compressed(fn + ".tmp.npz", **out)
        os.replace(fn + ".tmp.npz", fn)
        json.dump(dict(step=J, rows=meta, ks=KS, peak_gib=peaks(), seconds=time.time() - t0, yarn=a.yarn,
                       ckpt=a.ckpt, rung=a.rung, rows_from=a.rows_from, tok_root=a.tok_root),
                  open(os.path.join(a.out, f"rows_{J:04d}.json"), "w"))
        log(f"step {J} done ({time.time() - t0:.0f}s); peak {peaks()}")
    log(f"GSL_DONE {a.out}")


if __name__ == "__main__":
    main()
