"""How many tokens carry the summed (minibatch) parameter update, on a frozen checkpoint.

Question (Prasann, 2026-10-29): how many tokens are needed to capture 70 / 90 / 95% of the gradient across the whole
update, at different stages of dense training; does the learning signal spread out later in training, and does that
cap train-time token-drop speedups?

For every ``nn.Linear`` inside the 32 blocks, the batch gradient is G = sum_t delta_t x_t^T (delta_t = dL/d(output) at
token t, x_t = the layer's input). Each token's term is rank 1, with energy e_t = ||delta_t||^2 ||x_t||^2.
Loss = the training loss up to scale: summed answer CE over the batch (teacher forced, no-cot), parameters frozen.

Two passes per 128-row batch:
  1. forward + backward, hooks record e_t per (linear, token). From these, oracle rankings per batch scope
     (16 rows = one optimizer step; 128 rows = 8 steps), token-global within the batch:
       down<L>_<view>  score = sum of e_t over the linears of blocks >= L (L in SPLITS); covers those blocks
       own_<lin>_<view>  score = that linear's own e_t (linears of REP_BLOCKS)
       rand<L>_body    random score (reference)
       recipe_oracle / recipe_random   the recipe's two-stage shape: keep 20% of body tokens past block 7 and 7%
                       past block 16 (nested), by down7 / down16 score (or random); non-body tokens always kept
     view ``all``: every token is ranked, k = share of all tokens. View ``body``: only document-body tokens are ranked,
     non-body tokens (prompt, markers, query, answer) are always kept, k = share of body tokens.
     The ENERGY view (share of sum e_t held by the top k) comes straight from these scores.
  2. the same forward + backward again; each token's rank-1 term is added into the accumulator of its rank bin
     (bins = KS quantiles), for every ranking. Then for every k, G_S(k) = sum of bins up to k, and
     cos(G_S, G), ||G_S|| / ||G|| and <G_S, G> / ||G||^2 are read off (the UPDATE view).
     Accumulators are TensorSketches (CountSketch of delta (x) x via FFT, dim --sketch-dim): an unbiased random
     projection of the gradient matrix; inner products are preserved in expectation. Groups (block, mixer|mlp) share
     one sketch (independent hashes per linear, so cross terms cancel in expectation). Two linears also get EXACT
     fp32 accumulators (--exact) to validate the sketch.

    python research/token_routing/grad_concentration/probe.py --ckpt <dir> --label base --out /results/x.json
"""
from __future__ import annotations

import argparse
import ast
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
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens  # noqa: E402
from olmo_core.nn.online_token_router import structural_features  # noqa: E402

EOS, DOC_START, DOC_END = 248044, 248049, 248050
NB = 32
KS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SPLITS = (0, 3, 7, 16, 24)
REP_BLOCKS = (0, 3, 7, 16, 24, 31)
RECIPE = ((7, 0.2), (16, 0.07))
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
SHARDS = f"{W}/ds64/fast8k/shards"
CHUNK = 1024


def log(m):
    print(f"[grad-conc] {m}", flush=True)


def find_ckpt(path):
    if os.path.isdir(os.path.join(path, "model_and_optim")):
        return os.path.join(path, "model_and_optim")
    if path.rstrip("/").endswith("model_and_optim"):
        return path
    c = sorted(glob.glob(f"{path}/step*/model_and_optim"), key=lambda p: int(p.split("/step")[-1].split("/")[0]))
    if not c:
        raise SystemExit(f"no model_and_optim under {path}: "
                         f"{sorted(os.listdir(path))[:40] if os.path.isdir(path) else 'missing'}")
    return c[-1]


# ----------------------------------------------------------------------------------------------------------- data
def read_parts(shard, pattern_tok, pattern_mask):
    meta = json.load(open(os.path.join(shard, "metadata.json")))
    toks, masks = sorted(glob.glob(os.path.join(shard, pattern_tok))), sorted(glob.glob(os.path.join(shard, pattern_mask)))
    tok = np.concatenate([np.fromfile(t, dtype=meta.get("dtype", "uint32")) for t in toks])
    msk = np.concatenate([np.fromfile(m, dtype=meta.get("mask_dtype", "bool")) for m in masks])
    ends = np.flatnonzero(tok == EOS)
    return tok, msk, np.concatenate([[0], ends[:-1] + 1]), ends


def merged_task_index(shard, task):
    m = json.load(open(os.path.join(shard, "metadata.json")))
    mf = m.get("merged_from")
    if isinstance(mf, str):
        mf = ast.literal_eval(mf)
    names = [x["src"].rstrip("/").split("/")[-1].split("_g16M")[0].replace("_fresh", "") for x in mf]
    return names.index(task)


def load_rows(shards, tasks, n, offset, max_len, seen_shards):
    """``n`` rows per task from the single-task fresh pools, starting at ``offset`` (rows past the largest fresh run's
    prefix), dropping any row whose md5 appears in ``seen_shards`` (merged training shards)."""
    seen = {}
    rows = {}
    for task in tasks:
        tok, msk, st, en = read_parts(f"{shards}/{task}_fresh", "token_ids_part_*.npy", "labels_mask_*.npy")
        hs = set()
        for sh in seen_shards:
            try:
                i = merged_task_index(sh, task)
                t2, _, s2, e2 = read_parts(sh, f"token_ids_part_t{i}_*.npy", f"labels_mask_t{i}_*.npy")
                hs |= {hashlib.md5(t2[s:e + 1].tobytes()).hexdigest() for s, e in zip(s2, e2)}
            except Exception as exc:  # noqa: BLE001
                log(f"seen-check {sh}/{task} failed: {exc}")
        out, dropped = [], 0
        for k in range(offset, len(st)):
            s, e = int(st[k]), int(en[k]) + 1
            h = hashlib.md5(tok[s:e].tobytes()).hexdigest()
            if h in hs:
                dropped += 1
                continue
            if e - s > max_len or not msk[s:e].any():
                continue
            out.append(dict(task=task, k=k, ids=tok[s:e].astype(np.int64), lmask=msk[s:e].copy(), md5=h))
            if len(out) == n:
                break
        seen[task] = dropped
        rows[task] = out
        log(f"{task}: {len(out)} rows from {task}_fresh offset {offset} ({len(st)} rows in pool; "
            f"{dropped} skipped as seen in training shards)")
    return rows, seen


def body_mask(x):
    cid = build_chunk_ids_from_tokens(x[None], DOC_START, DOC_END, EOS, mode="chunked")
    _, body = structural_features(x[None], cid, doc_start_id=DOC_START, doc_end_id=DOC_END, offsets={})
    return body[0].bool()


# ---------------------------------------------------------------------------------------------------------- model
def build_model(ckpt):
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


def row_loss(model, row, ckpt=True):
    """Summed answer CE of one row; the embedding output is a leaf so every block's linears get output grads.
    ``ckpt``: every block runs under non-reentrant activation checkpointing (fits a 40 GB A100); the Tap then takes
    each linear's input from the backward-time recompute."""
    from torch.utils.checkpoint import checkpoint

    x = torch.from_numpy(row["ids"]).cuda()
    lm = torch.from_numpy(row["lmask"]).cuda()
    pred = torch.nonzero(lm[1:], as_tuple=True)[0]
    tgt = x[pred + 1]
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    h = model.embeddings(x[None]).detach().requires_grad_(True)
    for key, block in model.blocks.items():
        kw = {**abk, **pbk.get(int(key), {})}
        h = checkpoint(lambda hh, b=block, k=kw: b(hh, **k), h, use_reentrant=False) if ckpt else block(h, **kw)
    return F.cross_entropy(model.lm_head(h, **lmk)[0].float(), tgt, reduction="sum"), int(pred.numel())


class Lin:
    def __init__(self, idx, name, block, part, mod):
        self.idx, self.name, self.block, self.part, self.mod = idx, name, block, part, mod
        self.group = 2 * block + part


def list_linears(model):
    lins = []
    for key, block in model.blocks.items():
        b = int(key)
        for n, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                part = 1 if n.startswith("feed_forward") else 0
                lins.append(Lin(len(lins), f"b{b}.{n}", b, part, mod))
    return lins


# ------------------------------------------------------------------------------------------------------- rankings
class Ranking:
    """One token ranking within one batch scope. ``lins``: the linears it covers. ``key_of(lin)``: accumulator key
    (group id, or the linear's name for per-linear rankings). ``nbins``: bins incl. bin 0 = always kept."""

    def __init__(self, name, lins, per_linear=False, nbins=len(KS) + 1):
        self.name, self.lins, self.per_linear, self.nbins = name, lins, per_linear, nbins
        self.bins = {}  # row id -> (T,) long on GPU, or dict stage -> tensor (recipe)

    def key_of(self, lin):
        return lin.name if self.per_linear else lin.group

    def bins_for(self, rid, lin):
        b = self.bins[rid]
        if isinstance(b, dict):  # recipe: stage by block
            return b[1] if lin.block >= RECIPE[1][0] else b[0]
        return b


def rank_bins(score, elig):
    """bins (N,) long: 0 for non-eligible (always kept), 1..K for eligible tokens by descending score quantile."""
    n = int(elig.sum())
    bins = torch.zeros(score.numel(), dtype=torch.long)
    idx = torch.nonzero(elig, as_tuple=True)[0]
    order = torch.argsort(score[idx], descending=True)
    frac = (torch.arange(n, dtype=torch.float64) + 1) / max(n, 1)
    bins[idx[order]] = torch.searchsorted(torch.tensor(KS, dtype=torch.float64), frac) + 1
    return bins


def energy_curve(score, elig):
    """share of the eligible tokens' summed score held by their top-k (k in KS)."""
    s = torch.sort(score[elig].double(), descending=True).values
    if s.numel() == 0:
        return np.zeros(len(KS))
    c = torch.cumsum(s, 0)
    i = torch.tensor([max(1, int(np.ceil(k * s.numel()))) - 1 for k in KS])
    return (c[i] / c[-1].clamp(min=1e-300)).numpy()


# ------------------------------------------------------------------------------------------------------------- tap
class Tap:
    def __init__(self, lins, m, exact_names):
        self.lins, self.m = lins, m
        self.mode = None
        self.E = {}
        self.fired = 0
        self.active = {}  # lin idx -> [(acc tensor, bins tensor)]
        self.exact = {}  # lin idx -> [(acc (nb, out, in) fp32, bins)]
        self.exact_names = exact_names
        self.check = {}
        self.ckpt = True
        self.recompute = False  # True during backward: forward hooks then come from the checkpoint recompute
        self.xs, self.gs = {}, {}  # linear inputs / output grads waiting for their partner (either may come first)
        dev = torch.device("cuda")
        self.hash = []
        for ln in lins:
            g = torch.Generator(device="cpu").manual_seed(1009 + ln.idx)
            o, i = ln.mod.out_features, ln.mod.in_features
            self.hash.append((torch.randint(0, m, (o,), generator=g).to(dev),
                              (torch.randint(0, 2, (o,), generator=g) * 2 - 1).float().to(dev),
                              torch.randint(0, m, (i,), generator=g).to(dev),
                              (torch.randint(0, 2, (i,), generator=g) * 2 - 1).float().to(dev)))
            ln.mod.register_forward_hook(self._fwd(ln.idx))

    def _fwd(self, i):
        def hook(mod, inp, out):
            if self.mode is None:
                return
            if self.recompute:
                x = inp[0].detach()
                if i in self.gs:
                    self._grad(i, x, self.gs.pop(i))
                else:
                    self.xs[i] = x
                return
            if not out.requires_grad:
                return
            if self.ckpt:
                out.register_hook(lambda g, i=i: self._on_g(i, g))
            else:
                x = inp[0].detach()
                out.register_hook(lambda g, i=i, x=x: self._grad(i, x, g))
        return hook

    def _on_g(self, i, g):
        if i in self.xs:
            self._grad(i, self.xs.pop(i), g)
        else:
            self.gs[i] = g.detach()

    def backward(self, loss):
        self.recompute = self.ckpt
        loss.backward()
        self.recompute = False
        if self.xs or self.gs:
            raise SystemExit(f"unpaired linear inputs/grads after backward: {len(self.xs)} / {len(self.gs)}")

    def _grad(self, i, x, g):
        self.fired += 1
        x = x.reshape(-1, x.shape[-1])
        g = g.detach().reshape(-1, g.shape[-1])
        if self.mode == "energy":
            self.E[i] = (g.float().square().sum(-1) * x.float().square().sum(-1)).cpu()
            if self.lins[i].name in self.check:
                self.check[self.lins[i].name] = g.float().T @ x.float()
            return
        T = x.shape[0]
        h1, s1, h2, s2 = self.hash[i]
        m = self.m
        for c0 in range(0, T, CHUNK):
            c1 = min(T, c0 + CHUNK)
            gc, xc = g[c0:c1].float(), x[c0:c1].float()
            cs = torch.zeros(c1 - c0, m, device=g.device).index_add_(1, h1, gc * s1)
            P = torch.fft.rfft(cs, dim=1)
            cs.zero_().index_add_(1, h2, xc * s2)
            P *= torch.fft.rfft(cs, dim=1)
            del cs
            Pr = torch.view_as_real(P).reshape(c1 - c0, -1)
            # TF32 for the sketch bins (one-hot x sketch rows: rounding is uncorrelated and far below the sketch
            # noise); exact accumulators stay full fp32
            torch.backends.cuda.matmul.allow_tf32 = True
            for acc, bins in self.active.get(i, ()):
                oh = F.one_hot(bins[c0:c1], acc.shape[0]).T.float()
                acc.addmm_(oh, Pr)
            torch.backends.cuda.matmul.allow_tf32 = False
            for acc, bins in self.exact.get(i, ()):
                bc = bins[c0:c1]
                for j in range(acc.shape[0]):
                    sel = bc == j
                    if sel.any():
                        acc[j].addmm_(gc[sel].T, xc[sel])
            del P, Pr


# ---------------------------------------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", default="contradiction,nq,outlier,oolong")
    ap.add_argument("--rows", type=int, default=128, help="rows per task")
    ap.add_argument("--row-offset", type=int, default=7808, help="first fresh-pool row (mt4f7808 uses rows < 7808)")
    ap.add_argument("--shards", default=SHARDS, help="shard root (weka on Beaker; a node-local copy on slurm)")
    ap.add_argument("--seen-shards", default="mt4f7808,mt4s1952", help="merged training shards (under --shards)")
    ap.add_argument("--scopes", default="16,128", help="batch sizes (rows); must nest, largest = pass size")
    ap.add_argument("--sketch-dim", type=int, default=32768)
    ap.add_argument("--exact", default="b7.feed_forward.w2,b16.attention.w_out",
                    help="linears that also get exact fp32 accumulators (smallest scope only)")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--no-ckpt", action="store_true", help="no per-block activation checkpointing (needs ~60 GB)")
    ap.add_argument("--max-batches", type=int, default=0, help="debug: stop after this many large batches")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    tasks = a.tasks.split(",")
    scopes = [int(s) for s in a.scopes.split(",")]
    big = max(scopes)
    assert all(big % s == 0 and s % len(tasks) == 0 for s in scopes)
    seen_shards = [os.path.join(a.shards, s) for s in a.seen_shards.split(",") if s]
    missing = [s for s in seen_shards if not os.path.isdir(s)]
    if missing:
        raise SystemExit(f"seen-check shards missing: {missing}")
    rows_by_task, seen = load_rows(a.shards, tasks, a.rows, a.row_offset, a.max_len, seen_shards)
    n_per = min(len(v) for v in rows_by_task.values())
    nbig = n_per * len(tasks) // big
    if a.max_batches:
        nbig = min(nbig, a.max_batches)
    # batches: big batch j = rows [j*big/ntask, (j+1)*big/ntask) of every task, ordered so the small scopes nest
    small = min(scopes)
    order = []
    per_small = small // len(tasks)
    for j in range(nbig * big // small):
        for t in tasks:
            order += rows_by_task[t][j * per_small:(j + 1) * per_small]
    log(f"{len(order)} rows in {nbig} batches of {big}; scopes {scopes}; lengths "
        f"{min(len(r['ids']) for r in order)}-{max(len(r['ids']) for r in order)}")

    model = build_model(find_ckpt(a.ckpt))
    lins = list_linears(model)
    exact_names = [s for s in a.exact.split(",") if s]
    tap = Tap(lins, a.sketch_dim, exact_names)
    tap.ckpt = not a.no_ckpt
    by_name = {ln.name: ln for ln in lins}
    rep = [ln for ln in lins if ln.block in REP_BLOCKS]
    log(f"{len(lins)} linears ({sum(ln.mod.weight.numel() for ln in lins) / 1e9:.2f}B params); block 3: "
        f"{[ln.name for ln in lins if ln.block == 3]}; block 4: {[ln.name for ln in lins if ln.block == 4]}")
    nm = a.sketch_dim // 2 + 1

    def make_rankings():
        R = []
        for L in SPLITS:
            for v in ("all", "body"):
                R.append(Ranking(f"down{L}_{v}", [ln for ln in lins if ln.block >= L]))
        for L in (7, 16):
            R.append(Ranking(f"rand{L}_body", [ln for ln in lins if ln.block >= L]))
        for ln in rep:
            for v in ("all", "body"):
                R.append(Ranking(f"own_{ln.name}_{v}", [ln], per_linear=True))
        for nmr in ("recipe_oracle", "recipe_random"):
            R.append(Ranking(nmr, [ln for ln in lins if ln.block >= RECIPE[0][0]], nbins=3))
        return R

    rankings = {s: make_rankings() for s in scopes}
    acc = {s: {r.name: {} for r in rankings[s]} for s in scopes}
    for s in scopes:
        for r in rankings[s]:
            for ln in r.lins:
                k = r.key_of(ln)
                if k not in acc[s][r.name]:
                    acc[s][r.name][k] = torch.zeros(r.nbins, 2 * nm, device="cuda")
    # per-linear sketches of the validation linears under down7_body (smallest scope), to compare with exact
    val_sk = {nme: torch.zeros(len(KS) + 1, 2 * nm, device="cuda") for nme in exact_names}
    exact_acc = {(nme, rn): torch.zeros(len(KS) + 1, by_name[nme].mod.out_features, by_name[nme].mod.in_features,
                                        device="cuda") for nme in exact_names for rn in ("own_all", "down7_body")}
    nacc = sum(t.numel() for s in scopes for d in acc[s].values() for t in d.values())
    log(f"sketch dim {a.sketch_dim}: {nacc * 4 / 2**30:.1f} GiB of accumulators; exact "
        f"{sum(t.numel() for t in exact_acc.values()) * 4 / 2**30:.1f} GiB; "
        f"GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    res = {}  # (scope, ranking) -> lists over batches
    exact_res = {}
    perrow = []

    def put(d, key, val):
        d.setdefault(key, []).append(val)

    def finalize(s, bi):
        for r in rankings[s]:
            G = len(r.lins) if r.per_linear else 2 * NB
            dot = np.zeros((G, r.nbins))
            sq = np.zeros((G, r.nbins))
            full = np.zeros(G)
            for k, A in acc[s][r.name].items():
                gi = 0 if r.per_linear else k
                S = torch.fft.irfft(torch.view_as_complex(A.view(r.nbins, nm, 2)), n=a.sketch_dim, dim=1).double()
                C = torch.cumsum(S, 0)
                dot[gi] = (C @ C[-1]).cpu().numpy()
                sq[gi] = (C * C).sum(1).cpu().numpy()
                full[gi] = float(C[-1] @ C[-1])
                A.zero_()
            put(res, (s, r.name, "dot"), dot)
            put(res, (s, r.name, "sq"), sq)
            put(res, (s, r.name, "full"), full)
        if s == small:
            for nme, A in val_sk.items():
                S = torch.fft.irfft(torch.view_as_complex(A.view(-1, nm, 2)), n=a.sketch_dim, dim=1).double()
                C = torch.cumsum(S, 0)
                put(exact_res, ("sketch", nme, "down7_body", "dot"), (C @ C[-1]).cpu().numpy())
                put(exact_res, ("sketch", nme, "down7_body", "sq"), (C * C).sum(1).cpu().numpy())
                A.zero_()
            for (nme, rn), A in exact_acc.items():
                full_ = A.sum(0)
                c = torch.zeros_like(full_)
                dots, sqs = [], []
                for j in range(A.shape[0]):
                    c += A[j]
                    dots.append(float((c.double() * full_.double()).sum()))
                    sqs.append(float(c.double().square().sum()))
                put(exact_res, ("exact", nme, rn, "dot"), np.array(dots))
                put(exact_res, ("exact", nme, rn, "sq"), np.array(sqs))
                A.zero_()
        log(f"finalized scope {s} batch {bi}")

    t0 = time.time()
    rng = torch.Generator().manual_seed(1)
    for J in range(nbig):
        batch = order[J * big:(J + 1) * big]
        # ---- pass 1: energies
        tap.mode = "energy"
        E, body = [], []
        for ri, row in enumerate(batch):
            if J == 0 and ri == 0:  # validate delta x^T against autograd's weight gradient
                for nme in exact_names:
                    by_name[nme].mod.weight.requires_grad_(True)
                    tap.check[nme] = None
            tap.fired = 0
            loss, nans = row_loss(model, row, tap.ckpt)
            tap.backward(loss)
            if tap.fired != len(lins):
                raise SystemExit(f"hooks fired {tap.fired} times for {len(lins)} linears")
            if tap.check:
                for nme in exact_names:
                    w = by_name[nme].mod.weight
                    err = float((tap.check[nme] - w.grad.float()).norm() / w.grad.float().norm())
                    log(f"check {nme}: |sum delta x^T - autograd grad| / |grad| = {err:.2e}")
                    w.grad = None
                    w.requires_grad_(False)
                tap.check = {}
            x = torch.from_numpy(row["ids"]).cuda()
            bm = body_mask(x).cpu()
            E.append(torch.stack([tap.E[ln.idx] for ln in lins]))  # (n_lin, T)
            body.append(bm)
            row.update(ce=float(loss) / max(nans, 1), n_answer=nans)
            n = J * big + ri + 1
            if n in (1, 2, 5, 10) or n % 64 == 0:
                el = time.time() - t0
                log(f"[pass1] row {n} ({row['task']} T={len(row['ids'])} CE={row['ce']:.3f} body={float(bm.float().mean()):.2f}) "
                    f"{el:.0f}s; ETA ~{el / n * (len(order) - n) * 3.5:.0f}s (pass 2 ~2.5x pass 1)")
        # per-row energy curves (batch of one)
        blk = torch.tensor([ln.block for ln in lins])
        for ri, row in enumerate(batch):
            ec = []
            for L in SPLITS:
                sc = E[ri][blk >= L].sum(0)
                ec.append([energy_curve(sc, torch.ones_like(body[ri])), energy_curve(sc, body[ri])])
            perrow.append(dict(task=row["task"], k=row["k"], T=len(row["ids"]), n_body=int(body[ri].sum()),
                               n_answer=row["n_answer"], ce=row["ce"], md5=row["md5"],
                               body_energy_share=[float(E[ri][blk >= L].sum(0)[body[ri]].sum() / E[ri][blk >= L].sum())
                                                  for L in SPLITS],
                               energy_curves=np.array(ec).tolist()))
        # ---- bins per scope
        lens = [len(r["ids"]) for r in batch]
        for s in scopes:
            for bi in range(big // s):
                sl = slice(bi * s, (bi + 1) * s)
                Ecat = torch.cat(E[sl], 1)
                Bcat = torch.cat(body[sl])
                offs = np.cumsum([0] + lens[sl])
                nrows = range(bi * s, (bi + 1) * s)
                for r in rankings[s]:
                    if r.name.startswith("recipe"):
                        if r.name == "recipe_oracle":
                            s1 = Ecat[blk >= RECIPE[0][0]].sum(0)
                            s2 = Ecat[blk >= RECIPE[1][0]].sum(0)
                        else:
                            s1 = torch.rand(Bcat.numel(), generator=rng)
                            s2 = torch.rand(Bcat.numel(), generator=rng)
                        nbody = int(Bcat.sum())
                        keep1 = ~Bcat.clone()
                        bidx = torch.nonzero(Bcat, as_tuple=True)[0]
                        k1 = bidx[torch.argsort(s1[bidx], descending=True)[:int(round(RECIPE[0][1] * nbody))]]
                        keep1[k1] = True
                        keep2 = ~Bcat.clone()
                        k2 = k1[torch.argsort(s2[k1], descending=True)[:int(round(RECIPE[1][1] * nbody))]]
                        keep2[k2] = True
                        b1 = torch.where(keep1, 1, 2)
                        b2 = torch.where(keep2, 1, 2)
                        bins = None
                        put(res, (s, r.name, "kfrac"), np.array([float(keep1.float().mean()), float(keep2.float().mean())]))
                    else:
                        if r.name.startswith("rand"):
                            sc = torch.rand(Bcat.numel(), generator=rng)
                        elif r.name.startswith("own_"):
                            sc = Ecat[r.lins[0].idx]
                        else:
                            sc = Ecat[blk >= int(r.name[4:].split("_")[0])].sum(0)
                        elig = Bcat if r.name.endswith("_body") else torch.ones_like(Bcat)
                        bins = rank_bins(sc, elig)
                        en = energy_curve(sc, elig) if not r.name.startswith("rand") else np.zeros(len(KS))
                        put(res, (s, r.name, "eshare"), en)
                        put(res, (s, r.name, "nelig"), np.array([int(elig.sum()), Bcat.numel()]))
                        if not r.name.startswith("rand"):
                            put(res, (s, r.name, "elig_energy_share"), np.array([float(sc[elig].sum() / sc.sum())]))
                    for q, rid in enumerate(nrows):
                        a_, b_ = offs[q], offs[q + 1]
                        r.bins[rid] = ({0: b1[a_:b_].cuda(), 1: b2[a_:b_].cuda()} if bins is None
                                       else bins[a_:b_].cuda())
        # ---- pass 2: sketches
        tap.mode = "sketch"
        for ri, row in enumerate(batch):
            act = {}
            for s in scopes:
                for r in rankings[s]:
                    for ln in r.lins:
                        act.setdefault(ln.idx, []).append((acc[s][r.name][r.key_of(ln)], r.bins_for(ri, ln)))
            ex = {}
            sm = next(r for r in rankings[small] if r.name == "down7_body")
            for nme in exact_names:
                ln = by_name[nme]
                own = next(r for r in rankings[small] if r.name == f"own_{nme}_all")
                ex[ln.idx] = [(exact_acc[(nme, "own_all")], own.bins_for(ri, ln)),
                              (exact_acc[(nme, "down7_body")], sm.bins_for(ri, ln))]
                act[ln.idx].append((val_sk[nme], sm.bins_for(ri, ln)))
            tap.active, tap.exact = act, ex
            loss, _ = row_loss(model, row, tap.ckpt)
            tap.backward(loss)
            n = J * big + ri + 1
            if n in (1, 2, 5, 10) or n % 32 == 0:
                el = time.time() - t0
                log(f"[pass2] row {n}/{len(order)} {el:.0f}s; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            for s in scopes:
                if (ri + 1) % s == 0:
                    finalize(s, (J * big + ri + 1) // s - 1)
        for s in scopes:
            for r in rankings[s]:
                r.bins.clear()
        tap.active, tap.exact = {}, {}
        log(f"batch {J + 1}/{nbig} done, {time.time() - t0:.0f}s")

    out_base = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    arrs = {"|".join(map(str, k)): np.stack(v) for k, v in res.items()}
    arrs.update({"|".join(map(str, k)): np.stack(v) for k, v in exact_res.items()})
    np.savez_compressed(f"{out_base}_arrays.npz", **arrs)
    json.dump(dict(ckpt=a.ckpt, ckpt_resolved=find_ckpt(a.ckpt), label=a.label, tasks=tasks, row_offset=a.row_offset,
                   rows_per_task=n_per, scopes=scopes, n_big=nbig, ks=KS, splits=SPLITS, rep_blocks=REP_BLOCKS,
                   recipe=RECIPE, sketch_dim=a.sketch_dim, exact=exact_names, seen_skipped=seen,
                   lins=[dict(name=ln.name, block=ln.block, part=ln.part, out=ln.mod.out_features,
                              inp=ln.mod.in_features) for ln in lins],
                   group_layout="group = 2*block + part (0 mixer, 1 mlp)",
                   bins="bin 0 = always kept (non-body tokens in the body view); bins 1..K = rank quantiles KS; "
                        "dot/sq are CUMULATIVE over bins (k = KS[j-1] at bin j); recipe: bin 1 kept, 2 dropped",
                   rows=perrow, seconds=time.time() - t0), open(a.out, "w"))
    log(f"wrote {a.out} and {out_base}_arrays.npz ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
