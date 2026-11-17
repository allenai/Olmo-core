"""Offline check for the online token router: can a linear probe on early-layer features predict which
document tokens carry large loss gradient?

For training rows of a fast8k mix shard (mt4s488: contradiction / nq / outlier / oolong at 8k), one full-context
forward + backward of the answer CE (teacher forced, no-cot) per row gives per-token targets:

  gnorm@emb   log ||dCE/de_t||          at the input embedding (what the online router fits in training)
  gxh@emb     log |<dCE/de_t, e_t>|     first-order cost of zeroing the token's embedding
  gnorm@L8 / gnorm@L16                   log ||dCE/dh_t|| at block 8 / 16 outputs

and the router's feature parts (``olmo_core.nn.online_token_router``: structure, lexical query match, gold flag,
RMS-normalised hidden state h at layer 0 / 4 / 8 and its query interaction h*q). Ridge fits on FIT rows (stats
accumulated, nothing stored), then the HELD-OUT rows are scored:

  * per-row Spearman(pred, target) over body tokens;
  * top-10% recall: share of the target's top-10% body tokens in the prediction's top 10%;
  * mass@rho: share of sum_t exp(2 * target) (squared gradient norm) inside the predicted top rho, rho = 10/30%,
    vs random (= rho) and the oracle (target's own top rho);
  * gold AUC of each target (do gold-document tokens carry larger gradient?).

Plus the ONLINE question: targets are only observable on kept tokens of a COMPACTED row. For held-out rows, a
compacted forward keeping a random 30% of body tokens (slot-less, original RoPE positions) gives gnorm@emb on the
kept tokens; Spearman with the full-context gnorm@emb on the same tokens says how much the online target tracks
the full-context one.

    python research/token_routing/online_router/offline_probe.py --ckpt <distcp model_and_optim> \
        --shard <mt4s488 dir> --fit-rows 24 --test-rows 12 --out probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens  # noqa: E402
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row  # noqa: E402
from olmo_core.nn.online_token_router import (  # noqa: E402
    N_STRUCT,
    query_mask,
    structural_features,
)

EOS, DOC_START, DOC_END = 248044, 248049, 248050
TASKS = ["contradiction", "nq", "outlier", "oolong"]  # mt4 part order t0..t3 (metadata.json merged_from)
HIDDEN_LAYERS = (0, 4, 8)
TARGET_LAYERS = {"gnorm@L8": 8, "gnorm@L16": 16}
TARGETS = ["gnorm@emb", "gxh@emb", "gnorm@L8", "gnorm@L16"]
FEATURE_SETS = {
    # name: (hidden layer or None, use query interaction, use gold)
    "pos": (None, False, False),
    "pos+lex": (None, False, False),
    "pos+lex+gold": (None, False, True),
    "emb": (0, False, False),
    "emb+q": (0, True, False),
    "L4+q": (4, True, False),
    "L8+q": (8, True, False),
    "L8+q+gold": (8, True, True),
}


def log(m: str) -> None:
    print(f"[otr-probe] {m}", flush=True)


def shard_tasks(shard: str) -> List[str]:
    """Task names in part order (t0, t1, ...) from the shard's metadata (``merged_from`` source basenames)."""
    import ast

    m = json.load(open(os.path.join(shard, "metadata.json")))
    mf = m.get("merged_from")
    if isinstance(mf, str):
        mf = ast.literal_eval(mf)
    return [x["src"].rstrip("/").split("/")[-1] for x in mf] if mf else list(TASKS)


def load_rows(shard: str, n_per_task: int, offset: int, tasks: Optional[List[str]] = None) -> List[Dict]:
    """``tasks``: task names to load (default: the mt4 ``TASKS``); part indices come from the shard's metadata."""
    gold_table = json.load(open(os.path.join(shard, "gold_fingerprints.json")))
    rows = []
    order = shard_tasks(shard) if tasks is not None else list(TASKS)
    for task in tasks if tasks is not None else TASKS:
        ti = order.index(task)
        ids = np.fromfile(os.path.join(shard, f"token_ids_part_t{ti}_000.npy"), dtype=np.uint32)
        lm = np.fromfile(os.path.join(shard, f"labels_mask_t{ti}_000.npy"), dtype=np.bool_)
        ends = np.nonzero(ids == EOS)[0]
        starts = np.concatenate([[0], ends[:-1] + 1])
        for k in range(offset, offset + n_per_task):
            s, e = int(starts[k]), int(ends[k]) + 1
            x = ids[s:e].astype(np.int64)
            gold = gold_table.get(content_fingerprint_from_row(x.tolist(), EOS))
            if gold and isinstance(gold[0], (list, tuple)):
                gold = [a for p in gold for a in p]
            rows.append(dict(task=task, ids=x, lmask=lm[s:e].copy(), gold=sorted(set(int(g) for g in (gold or [])))))
    return rows


def build_model(ckpt: str):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320, attn_backend=AttentionBackendName("flash_2"))
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    t0 = time.time()
    load_model_and_optim_state(ckpt, model)
    log(f"loaded {ckpt} in {time.time() - t0:.0f}s")
    model = model.cuda().to(torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"froze FLA length autotune on {freeze_fla_length_autotune()} kernels")
    return model


def run_row(model, row, keep_pos=None):
    """Forward+backward of the answer CE. ``keep_pos`` (sorted original positions) -> compacted row.
    Returns dict of per-position tensors over the (possibly compacted) row."""
    from torch.utils.checkpoint import checkpoint

    dev = "cuda"
    ids = torch.from_numpy(row["ids"]).to(dev)
    lmask = torch.from_numpy(row["lmask"]).to(dev)
    T = ids.numel()
    pos = torch.arange(T, device=dev) if keep_pos is None else keep_pos
    x = ids[pos][None]
    lm = lmask[pos]
    # shifted targets: logits at i predict token i+1 when it is an answer token
    pred = torch.nonzero(lm[1:], as_tuple=True)[0]
    tgt = x[0, pred + 1]
    _, _, abk, pbk, lmk = model._prepare_inputs(
        x, None, logits_to_keep=pred[None], position_ids=(None if keep_pos is None else pos[None])
    )
    e = model.embeddings(x).detach().requires_grad_(True)
    h = e
    caps = {}
    want = set(HIDDEN_LAYERS) | set(TARGET_LAYERS.values())
    for key, block in model.blocks.items():
        i = int(key)
        if i in want and i > 0:
            caps[i] = h
            h.retain_grad()
        kw = {**abk, **pbk.get(i, {})}
        h = checkpoint(lambda hh, b=block, k=kw: b(hh, **k), h, use_reentrant=False)
    lg = model.lm_head(h, **lmk)[0].float()
    ce = F.cross_entropy(lg, tgt)
    ce.backward()
    out = {"ce": float(ce), "pos": pos, "x": x[0], "e": e.detach()[0], "ge": e.grad.detach()[0]}
    for i, t in caps.items():
        out[f"h{i}"] = t.detach()[0]
        out[f"g{i}"] = t.grad.detach()[0]
    return out


def targets_of(o) -> Dict[str, torch.Tensor]:
    ge, e = o["ge"].float(), o["e"].float()
    t = {
        "gnorm@emb": torch.log(ge.norm(dim=-1) + 1e-12),
        "gxh@emb": torch.log((ge * e).sum(-1).abs() + 1e-12),
    }
    for name, L in TARGET_LAYERS.items():
        t[name] = torch.log(o[f"g{L}"].float().norm(dim=-1) + 1e-12)
    return t


def feature_parts(row, o):
    """Router feature parts for the full-context row (same functions the trainer uses)."""
    x = o["x"][None]
    cid = build_chunk_ids_from_tokens(x, DOC_START, DOC_END, EOS, mode="chunked")
    struct, body = structural_features(x, cid, doc_start_id=DOC_START, doc_end_id=DOC_END)
    lm = torch.from_numpy(row["lmask"]).to(x.device)
    labels = torch.full_like(x, -100)
    labels[0, :-1] = torch.where(lm[1:], x[0, 1:], torch.full_like(x[0, 1:], -100))
    qm = query_mask(cid, labels)
    B, T = x.shape
    V = int(x.max()) + 1
    key = x[0].long()
    qset = torch.zeros(V, dtype=torch.bool, device=x.device)
    qset[key[qm[0]]] = True
    in_q = qset[key].float()
    cnt = torch.zeros(V, device=x.device).index_add_(0, key[body[0]], torch.ones(int(body.sum()), device=x.device))
    rep = torch.log1p(cnt[key]) / 5.0
    gflag = torch.zeros(T, device=x.device)
    c = cid[0].long()
    for g in row["gold"]:
        gflag[c == g] = 1.0
    b = body[0]
    parts = {"struct": struct[0], "lex": torch.stack([in_q, rep], -1), "gold": gflag[:, None], "body": b, "cid": c}
    for L in HIDDEN_LAYERS:
        h = (o["e"] if L == 0 else o[f"h{L}"]).float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        q = h[qm[0]].mean(0)
        q = q * torch.rsqrt(q.pow(2).mean() + 1e-6)
        parts[f"h{L}"] = h
        parts[f"q{L}"] = q
    return parts


def design(parts, fs: str) -> torch.Tensor:
    L, use_q, use_gold = FEATURE_SETS[fs]
    b = parts["body"]
    cols = [parts["struct"][b]]
    if fs != "pos":
        cols.append(parts["lex"][b])
    if use_gold:
        cols.append(parts["gold"][b])
    if L is not None:
        h = parts[f"h{L}"][b]
        cols.append(h)
        if use_q:
            hq = h * parts[f"q{L}"][None]
            cols += [hq.mean(-1, keepdim=True), hq]
    return torch.cat(cols, -1).double()


def rank(a: torch.Tensor) -> torch.Tensor:
    return a.argsort().argsort().double()


def spearman(a, b) -> float:
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm() + 1e-12))


def topk_metrics(pred, tgt, rhos=(0.1, 0.3)):
    n = pred.numel()
    out = {}
    k10 = max(1, int(round(0.1 * n)))
    top_t = set(tgt.argsort(descending=True)[:k10].tolist())
    top_p = set(pred.argsort(descending=True)[:k10].tolist())
    out["recall@10"] = len(top_t & top_p) / k10
    w = torch.exp(2 * (tgt - tgt.max()))
    for rho in rhos:
        k = max(1, int(round(rho * n)))
        out[f"mass@{int(rho * 100)}"] = float(w[pred.argsort(descending=True)[:k]].sum() / w.sum())
        out[f"oracle_mass@{int(rho * 100)}"] = float(w[tgt.argsort(descending=True)[:k]].sum() / w.sum())
    return out


def auc(score, pos) -> float:
    pos = pos.bool()
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank(score) + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--fit-rows", type=int, default=24, help="per task")
    ap.add_argument("--test-rows", type=int, default=12, help="per task, disjoint (taken after the fit rows)")
    ap.add_argument("--ridge", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--compact-keep", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    model = build_model(a.ckpt)
    fit = load_rows(a.shard, a.fit_rows, 0)
    test = load_rows(a.shard, a.test_rows, a.fit_rows)
    log(f"{len(fit)} fit rows, {len(test)} test rows; lengths {min(len(r['ids']) for r in fit + test)}-"
        f"{max(len(r['ids']) for r in fit + test)}")

    # ---- pass 1: ridge statistics on fit rows (per task and pooled)
    stats: Dict = {}
    ce_fit = []
    t0 = time.time()
    for i, row in enumerate(fit):
        o = run_row(model, row)
        ce_fit.append((row["task"], o["ce"]))
        tg = targets_of(o)
        parts = feature_parts(row, o)
        b = parts["body"]
        Y = torch.stack([tg[t][b] - tg[t][b].mean() for t in TARGETS], -1).double()
        for fs in FEATURE_SETS:
            X = design(parts, fs)
            G, Bm = X.T @ X, X.T @ Y
            for scope in ("pooled", row["task"]):
                s = stats.setdefault((fs, scope), [0, 0])
                s[0] = s[0] + G
                s[1] = s[1] + Bm
        if i < 3 or (i + 1) % 10 == 0:
            el = time.time() - t0
            log(f"fit row {i + 1}/{len(fit)} ({row['task']}, T={len(row['ids'])}, CE={o['ce']:.3f}) "
                f"{el:.0f}s, ETA {el / (i + 1) * (len(fit) + len(test) - i - 1):.0f}s")
    W = {}
    for (fs, scope), (G, Bm) in stats.items():
        for lam in a.ridge:
            A = G.clone()
            A.diagonal().add_(lam * float(torch.diagonal(G).mean()))
            W[(fs, scope, lam)] = torch.linalg.solve(A, Bm)
    del stats

    # ---- pass 2: held-out rows
    res: Dict = {}
    agree = []
    gold_auc: Dict = {}
    ce_test = []
    baseline: Dict = {}
    for i, row in enumerate(test):
        o = run_row(model, row)
        ce_test.append((row["task"], o["ce"]))
        tg = targets_of(o)
        parts = feature_parts(row, o)
        b = parts["body"]
        T_b = {t: tg[t][b] - tg[t][b].mean() for t in TARGETS}
        gb = parts["gold"][b, 0]
        if row["gold"]:
            for t in TARGETS:
                gold_auc.setdefault((row["task"], t), []).append(auc(T_b[t], gb))
        for fs in FEATURE_SETS:
            X = design(parts, fs)
            for scope in ("pooled", "task"):
                sk = "pooled" if scope == "pooled" else row["task"]
                for lam in a.ridge:
                    P = X @ W[(fs, sk, lam)]
                    for ti, t in enumerate(TARGETS):
                        m = {"spearman": spearman(P[:, ti], T_b[t]), **topk_metrics(P[:, ti], T_b[t])}
                        for k, v in m.items():
                            res.setdefault((fs, scope, lam, t, row["task"], k), []).append(v)
        # random baseline
        rnd = torch.rand(int(b.sum()), device=b.device, dtype=torch.float64)
        for t in TARGETS:
            m = {"spearman": spearman(rnd, T_b[t]), **topk_metrics(rnd, T_b[t])}
            for k, v in m.items():
                baseline.setdefault((t, row["task"], k), []).append(v)
        # online target agreement: random-30% compacted forward
        body_pos = torch.nonzero(b, as_tuple=True)[0]
        nb = body_pos.numel()
        kk = max(1, int(round(a.compact_keep * nb)))
        kept_body = body_pos[torch.randperm(nb, device=b.device)[:kk]]
        keep = ~b.clone()
        keep[kept_body] = True
        keep_pos = torch.nonzero(keep, as_tuple=True)[0]
        oc = run_row(model, row, keep_pos=keep_pos)
        gc = torch.log(oc["ge"].float().norm(dim=-1) + 1e-12)
        isb = b[keep_pos]
        agree.append(dict(task=row["task"], spearman=spearman(gc[isb], tg["gnorm@emb"][keep_pos][isb]),
                          ce_full=o["ce"], ce_compact=oc["ce"]))
        if i < 3 or (i + 1) % 10 == 0:
            log(f"test row {i + 1}/{len(test)} ({row['task']}) CE full {o['ce']:.3f} compact30 {oc['ce']:.3f}")

    def mean(v):
        v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    summary = {"ckpt": a.ckpt, "shard": a.shard, "fit_rows_per_task": a.fit_rows, "eval_size_rows_per_task": a.test_rows,
               "ce_fit": {t: mean([c for tt, c in ce_fit if tt == t]) for t in TASKS},
               "ce_test": {t: mean([c for tt, c in ce_test if tt == t]) for t in TASKS},
               "results": [], "random": [], "gold_auc": [], "online_agreement": []}
    for (fs, scope, lam, t, task, k), v in sorted(res.items()):
        summary["results"].append(dict(features=fs, scope=scope, ridge=lam, target=t, task=task, metric=k, value=mean(v), n=len(v)))
    for (t, task, k), v in sorted(baseline.items()):
        summary["random"].append(dict(target=t, task=task, metric=k, value=mean(v)))
    for (task, t), v in sorted(gold_auc.items()):
        summary["gold_auc"].append(dict(task=task, target=t, value=mean(v), n=len(v)))
    for task in TASKS:
        rr = [x for x in agree if x["task"] == task]
        summary["online_agreement"].append(dict(task=task, spearman=mean([x["spearman"] for x in rr]),
                                                ce_full=mean([x["ce_full"] for x in rr]),
                                                ce_compact30=mean([x["ce_compact"] for x in rr]), n=len(rr)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    # compact print: best ridge per (features, scope, target) by pooled-over-task spearman
    log("=== held-out (mean over tasks of per-row means); best ridge per row ===")
    for t in TARGETS:
        for scope in ("pooled", "task"):
            for fs in FEATURE_SETS:
                best = None
                for lam in a.ridge:
                    sp = mean([r["value"] for r in summary["results"] if r["features"] == fs and r["scope"] == scope
                               and r["ridge"] == lam and r["target"] == t and r["metric"] == "spearman"])
                    if best is None or sp > best[1]:
                        best = (lam, sp)
                lam = best[0]
                per = {task: {k: next((r["value"] for r in summary["results"] if r["features"] == fs and r["scope"] == scope
                                       and r["ridge"] == lam and r["target"] == t and r["task"] == task and r["metric"] == k), float("nan"))
                              for k in ("spearman", "recall@10", "mass@30")} for task in TASKS}
                log(f"{t:10s} {scope:6s} {fs:13s} lam={lam:<5} " + "  ".join(
                    f"{task[:6]} sp={per[task]['spearman']:+.2f} r10={per[task]['recall@10']:.2f} m30={per[task]['mass@30']:.2f}"
                    for task in TASKS))
        rnd = {task: {k: next(r["value"] for r in summary["random"] if r["target"] == t and r["task"] == task and r["metric"] == k)
                      for k in ("recall@10", "mass@30", "oracle_mass@30")} for task in TASKS}
        log(f"{t:10s} random/oracle        " + "  ".join(
            f"{task[:6]} r10={rnd[task]['recall@10']:.2f} m30={rnd[task]['mass@30']:.2f} oracle_m30={rnd[task]['oracle_mass@30']:.2f}"
            for task in TASKS))
    log("gold AUC of targets: " + "  ".join(f"{r['task']}/{r['target']}={r['value']:.2f}" for r in summary["gold_auc"]))
    log("online agreement (compact30 gnorm vs full gnorm on kept tokens): " + "  ".join(
        f"{r['task']} sp={r['spearman']:+.2f} CE {r['ce_full']:.3f}->{r['ce_compact30']:.3f}" for r in summary["online_agreement"]))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
