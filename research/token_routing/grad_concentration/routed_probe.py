"""Where does the gradient-sparsity -> routing-sparsity link break? (Prasann, 2026-10-30; experiment tokdrop_lr.)

probe.py measures attribution under the DENSE forward: each token's rank-1 term delta_t x_t^T comes from the full-context
forward/backward, and the top-k terms reach cos 0.95+ with the batch update. Routing removes the dropped tokens from the
forward in every block past the split, so the kept tokens' activations and gradients change. This probe measures, on the
same frozen checkpoints and unseen fresh-pool rows, for matched selections S (keep fraction k at a block-7 / block-16 split,
and the recipe's nested 7:0.2 -> 16:0.07 shape):

  D      the dense batch gradient (every token, full context)
  A_S    the attribution partial sum: the dense terms of the kept tokens (+ every non-body token; blocks before the split
         get every token's term), i.e. probe.py's top-k update
  R_S    the TRUE routed gradient: the gradient of the routed forward (dropped tokens gathered out at the split, kept
         tokens at their original RoPE positions -- the online router's mid_compact path, batch of one row), with
         the same answer CE

for four selectors: ``og`` oracle, batch-global top-k by probe.py's down<L> score (sum over blocks >= L of each token's
rank-1 energy ||delta_t||^2 ||x_t||^2); ``od`` the same oracle score allocated per document like the router
(doc_share 1: each document keeps its own top ceil(k |doc|)); ``rnd`` uniform, per-document allocation (the training
random control); ``rtr`` a ridge router emulating the online router (features: structural + lexical scalars, cos(h, q),
RMS-normed h and h*q at the split; target: row-centred log ||dL/dh|| at the split from the dense pass; fit on earlier
batches + the other half of this batch, 10% exploration, per-document allocation; the recipe's stage 2 scores the ROUTED
h16 in-pass).

Gradients are TensorSketches (dim --sketch-dim per (block, mixer|mlp) group; hashes fixed, so every vector lives in the
same sketch space and cross-batch inner products are valid). With --adam-ckpt / --adam-factors every vector also gets an
Adam-preconditioned copy: D_ij = (v_ij + eps^2)^-1/2 from the checkpoint's exp_avg_sq, approximated per linear by a
rank-1 (Adafactor-style) factorisation exp(alpha_i + beta_j) so each token's term stays rank 1.

Per 16-row batch (one optimizer step: 4 rows per task) saved to --save-dir as <label>/b<j>.npz:
  tot_*  task-summed vectors at full sketch dim (fp16 + per-(vector, group) fp32 scale): D, A_<sel>, R_<sel> (+ adam_)
  tsk_*  per-task vectors folded to --fold-dim: D, A_<sel>, R_<sel>
  typ_*  per-task x token-type vectors (answer / prompt / gold body / other body) folded, for the FOCUS selections:
         dense split into kept/dropped by type, routed by type
  rows   per-row CE (dense and every routed pass), keep stats, gold parse
analyze_gap.py turns these into the waterfall.

    python research/token_routing/grad_concentration/routed_probe.py --ckpt <dir> --label f1952 \
        --num-shards 8 --shard 0 --save-dir /weka/.../gradgap --out /results/routed_probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe as P  # noqa: E402  (load_rows, body_mask, build_model, list_linears, find_ckpt; adds src/ to the path)

from olmo_core.nn.online_token_router import query_mask, structural_features  # noqa: E402

NB = 32
NG = 2 * NB
DIGIT0 = 15  # Qwen3.5 tokenizer: '0'..'9' are the single-digit tokens 15..24
TYPES = ("answer", "prompt", "gold", "body")
SPLIT_KS = {7: (0.03, 0.07, 0.2), 16: (0.01, 0.03, 0.07)}
SELECTORS = ("og", "od", "rnd", "rtr")
RECIPE = ((7, 0.2), (16, 0.07))
EXPLORE = 0.1
SAMPLES = 2048
CHUNK = 1024


def log(m):
    print(f"[routed-probe] {m}", flush=True)


def sel_names():
    out = []
    for L, ks in SPLIT_KS.items():
        for k in ks:
            for s in SELECTORS:
                out.append(f"{s}_L{L}_k{k:g}")
    for s in SELECTORS:
        out.append(f"{s}_recipe")
    return out


FOCUS = [f"{s}_recipe" for s in SELECTORS]


# ------------------------------------------------------------------------------------------------------------ rows
def digits_runs(ids):
    """integers spelled by runs of single-digit tokens."""
    out, cur = [], ""
    for t in ids.tolist():
        if DIGIT0 <= t < DIGIT0 + 10:
            cur += str(t - DIGIT0)
        elif cur:
            out.append(int(cur))
            cur = ""
    if cur:
        out.append(int(cur))
    return out


def prep_row(row, task):
    """GPU tensors + token types. Gold documents: the document whose header number ("Claim N" / "Document [N]") is in
    the answer (contradiction / nq / outlier); oolong has none (counting task)."""
    x = torch.from_numpy(row["ids"]).cuda()
    lm = torch.from_numpy(row["lmask"]).cuda()
    T = x.numel()
    pred = torch.nonzero(lm[1:], as_tuple=True)[0]
    body, cid = P.body_mask(x)
    body, cid = body.cuda(), cid.cuda()
    labels = torch.full((T,), -100, dtype=torch.long, device=x.device)
    labels[pred] = x[pred + 1]
    qm = query_mask(cid[None], labels[None])[0]
    ty = torch.ones(T, dtype=torch.long, device=x.device)  # prompt / markers / query
    ty[body] = 3
    ty[lm] = 0
    ty[pred] = 0
    gold_docs, nums = [], []
    if task != "oolong":
        ans = set(digits_runs(row["ids"][row["lmask"]]))
        nd = int(cid.max().item()) + 1 if bool((cid >= 0).any()) else 0
        cc = cid.cpu().numpy()
        for d in range(nd):
            pos = np.flatnonzero(cc == d)[:16]
            r = digits_runs(row["ids"][pos])
            n = r[0] if r else -1
            nums.append(n)
            if n in ans:
                gold_docs.append(d)
        if gold_docs:
            g = torch.zeros(nd, dtype=torch.bool, device=x.device)
            g[gold_docs] = True
            isg = body & (cid >= 0) & g[cid.clamp(min=0)]
            ty[isg] = 2
    numbered_ok = bool(nums) and nums == list(range(1, len(nums) + 1))
    return dict(x=x, pred=pred, tgt=x[pred + 1], body=body, cid=cid.long(), qm=qm, ty=ty, T=T,
                gold_docs=gold_docs, numbered_ok=numbered_ok)


# ------------------------------------------------------------------------------------------------------------- tap
class Tap:
    """Per-token TensorSketch of every block linear's rank-1 weight-gradient term, accumulated into per-(task, group)
    bins (``specs``); or per-block energies ``||delta||^2 ||x||^2`` (mode ``energy``). Handles compacted rows: block b
    ran on the original positions ``cur[b]``; a spec's bins are given over original positions."""

    def __init__(self, lins, m, adam=None, exact=()):
        self.lins, self.m = lins, m
        self.mode = None
        self.ckpt = True
        self.recompute = False
        self.xs, self.gs = {}, {}
        self.cur = {}
        self.specs = []  # (acc (n_task, NG, nbins, 2nm) , bins_fn(block) -> (T,) long or None, adam: bool)
        self.task = 0
        self.E = None
        self.fired = 0
        self.adam = adam or {}
        self.exact_names = set(exact)
        self.exact = {}  # name -> list of (acc (out, in) fp32, mask_fn(block) -> (T,) bool)
        self.lin_sk = {}  # name -> list of (acc (2nm,), mask_fn)
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

    def _ts(self, i, gc, xc):
        h1, s1, h2, s2 = self.hash[i]
        cs = torch.zeros(gc.shape[0], self.m, device=gc.device).index_add_(1, h1, gc * s1)
        Pf = torch.fft.rfft(cs, dim=1)
        cs.zero_().index_add_(1, h2, xc * s2)
        Pf *= torch.fft.rfft(cs, dim=1)
        return torch.view_as_real(Pf).reshape(gc.shape[0], -1)

    def _grad(self, i, x, g):
        self.fired += 1
        ln = self.lins[i]
        x = x.reshape(-1, x.shape[-1])
        g = g.detach().reshape(-1, g.shape[-1])
        cur = self.cur[ln.block]
        if self.mode == "energy":
            self.E[ln.block].index_add_(0, cur, g.float().square().sum(-1) * x.float().square().sum(-1))
            return
        T = x.shape[0]
        specs = [(acc, fn(ln.block), ad) for acc, fn, ad in self.specs]
        specs = [(acc[self.task, ln.group], b[cur], ad) for acc, b, ad in specs if b is not None]
        adam = self.adam.get(ln.name) if any(ad for _, _, ad in specs) else None
        ex = [(A, fn(ln.block)[cur]) for A, fn in self.exact.get(ln.name, ())]
        lsk = [(A, fn(ln.block)[cur]) for A, fn in self.lin_sk.get(ln.name, ())]
        torch.backends.cuda.matmul.allow_tf32 = True
        for c0 in range(0, T, CHUNK):
            c1 = min(T, c0 + CHUNK)
            gc, xc = g[c0:c1].float(), x[c0:c1].float()
            Pr = self._ts(i, gc, xc)
            Pa = self._ts(i, gc * adam[0], xc * adam[1]) if adam is not None else None
            for acc, bins, ad in specs:
                if ad and Pa is None:
                    continue
                oh = F.one_hot(bins[c0:c1], acc.shape[0]).T.float()
                acc.addmm_(oh, Pa if ad else Pr)
            for A, msk in lsk:
                A.add_((Pr * msk[c0:c1, None]).sum(0))
            if ex:
                torch.backends.cuda.matmul.allow_tf32 = False
                for A, msk in ex:
                    mm = msk[c0:c1]
                    if bool(mm.any()):
                        A.addmm_(gc[mm].T, xc[mm])
                torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False


# ----------------------------------------------------------------------------------------------------------- model
def run_row(model, tap, r, stages=None, cap=(), ckpt=True):
    """Answer CE (summed) of one row through the block stack. ``stages``: {block L: fn(h, cur) -> keep (bool over the
    current columns)}: at block L the row is gathered to the kept columns, which keep their original RoPE positions
    (the online router's mid_compact for a batch of one row). ``cap``: blocks whose INPUT hidden state is returned with
    its gradient retained (dense pass only)."""
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    x, pred, tgt, T = r["x"], r["pred"], r["tgt"], r["T"]
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    h = model.embeddings(x[None]).detach().requires_grad_(True)
    cur = torch.arange(T, device=x.device)
    stages = stages or {}
    capt = {}
    with set_checkpoint_early_stop(False):
        for key, block in model.blocks.items():
            b = int(key)
            if b in stages:
                with torch.no_grad():
                    keep = stages[b](h, cur)
                idx = torch.nonzero(keep, as_tuple=True)[0]
                h = h[:, idx]
                cur = cur[idx]
                abk = {**abk, "position_ids": cur[None]}
            if b in cap:
                h.retain_grad()
                capt[b] = h
            tap.cur[b] = cur
            kw = {**abk, **pbk.get(b, {})}
            h = checkpoint(lambda hh, bl=block, k=kw: bl(hh, **k), h, use_reentrant=False) if ckpt else block(h, **kw)
    newcol = torch.full((T,), -1, dtype=torch.long, device=x.device)
    newcol[cur] = torch.arange(cur.numel(), device=x.device)
    lt = newcol[pred]
    if bool((lt < 0).any()):
        raise RuntimeError("an answer position was dropped")
    logits = model.lm_head(h, **{**lmk, "logits_to_keep": lt[None]})[0]
    return F.cross_entropy(logits.float(), tgt, reduction="sum"), capt, cur


# ---------------------------------------------------------------------------------------------------------- router
def rms(h):
    h = h.float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)


def scalar_feats(r):
    """(T, nS) structural + lexical scalars (online_token_router.token_features, minus cos_h_q)."""
    x, cid, body = r["x"], r["cid"], r["body"]
    struct, _ = structural_features(x[None], cid[None], doc_start_id=P.DOC_START, doc_end_id=P.DOC_END, offsets={})
    V = int(x.max().item()) + 1
    qset = torch.zeros(V, dtype=torch.bool, device=x.device)
    qset[x[r["qm"]]] = True
    in_q = qset[x].float()
    cnt = torch.zeros(V, device=x.device).index_add_(0, x[body], torch.ones(int(body.sum()), device=x.device))
    rep = torch.log1p(cnt[x]) / 5.0
    return torch.cat([struct[0], (in_q * body)[:, None], (rep * body)[:, None]], dim=-1)


def feats(hn, q, s):
    """router feature rows: [scalars, cos(h, q), h, h*q] (h RMS-normed, q = RMS-normed mean over query tokens)."""
    cos = (hn * q[None]).mean(-1, keepdim=True)
    return torch.cat([s, cos, hn, hn * q[None]], dim=-1)


def query_vec(hn, qm):
    q = hn[qm].mean(0) if bool(qm.any()) else hn.mean(0)
    return q * torch.rsqrt(q.pow(2).mean() + 1e-6)


class Ridge:
    def __init__(self, nf, decay=0.9, lam=0.1):
        self.XtX = torch.zeros(nf, nf, device="cuda", dtype=torch.float64)
        self.Xty = torch.zeros(nf, device="cuda", dtype=torch.float64)
        self.decay, self.lam = decay, lam

    def stats(self, X, y):
        X = X.double()
        return X.T @ X, X.T @ y.double()

    def solve(self, extra):
        A = self.XtX.clone()
        bb = self.Xty.clone()
        for a_, b_ in extra:
            A += a_
            bb += b_
        lam = self.lam * float(A.diagonal().mean())
        A.diagonal().add_(lam)
        return torch.linalg.solve(A, bb).float()

    def push(self, stats):
        self.XtX.mul_(self.decay)
        self.Xty.mul_(self.decay)
        for a_, b_ in stats:
            self.XtX += a_
            self.Xty += b_


# ------------------------------------------------------------------------------------------------------- selection
def doc_topk(score, body, cid, frac, k_row=None, prio=None):
    """per-document allocation (doc_share 1): every document keeps its own top ceil(frac |doc body|) by ``score``
    (``prio`` tokens first), then the rest of the row budget ceil(frac |body|) (or ``k_row``) by score over the row."""
    T = score.numel()
    keep = torch.zeros(T, dtype=torch.bool, device=score.device)
    idx = torch.nonzero(body, as_tuple=True)[0]
    if idx.numel() == 0:
        return keep
    sc = score[idx].double()
    if prio is not None:
        sc = torch.where(prio[idx], torch.full_like(sc, 1e30), sc)
    d = cid[idx]
    o = torch.argsort(-sc)
    o = o[torch.argsort(d[o], stable=True)]
    ds = d[o]
    _, inv, cnt = torch.unique_consecutive(ds, return_inverse=True, return_counts=True)
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(o.numel(), device=score.device) - start[inv]
    kd = torch.ceil(cnt.double() * frac).long()
    keep[idx[o[rank < kd[inv]]]] = True
    kr = int(math.ceil(frac * idx.numel())) if k_row is None else int(k_row)
    left = kr - int(keep.sum())
    if left > 0:
        rest = idx[~keep[idx]]
        if rest.numel():
            top = torch.argsort(-sc[~keep[idx]])[:left]
            keep[rest[top]] = True
    return keep


def global_topk(scores, bodies, frac):
    """batch-global top round(frac * total body) by score (probe.py's ranking); list of per-row keeps."""
    allb = torch.cat([s[b] for s, b in zip(scores, bodies)])
    n = int(round(frac * allb.numel()))
    thr = torch.topk(allb.double(), max(n, 1)).values[-1] if n > 0 else float("inf")
    out = []
    for s, b in zip(scores, bodies):
        out.append(b & (s.double() >= thr))
    return out


def explore_mask(body, frac_budget, gen):
    nb = int(body.sum())
    ke = int(round(math.ceil(nb * frac_budget) * EXPLORE))
    m = torch.zeros_like(body)
    if ke > 0:
        idx = torch.nonzero(body, as_tuple=True)[0]
        m[idx[torch.randperm(idx.numel(), generator=gen, device="cpu")[:ke].to(idx.device)]] = True
    return m


# -------------------------------------------------------------------------------------------------------- lora
def merge_lora(model, lins, ckpt_dir, scale):
    """A LoRA checkpoint (the recipe: ``lora_A`` / ``lora_B`` on every block linear, early_lora.py) loads its base
    weights through the tolerant base load; merge the adapters into them (W += scale * B A). Returns #merged."""
    from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

    keys = [k for k in get_checkpoint_metadata(ckpt_dir).state_dict_metadata if k.endswith("lora_A")]
    if not keys:
        return 0
    n = 0
    for ln in lins:
        pname = f"blocks.{ln.block}.{ln.name.split('.', 1)[1]}"
        ka = [k for k in keys if k.endswith(f"{pname}.lora_A")]
        if len(ka) != 1:
            continue
        A, Bm = load_keys(ckpt_dir, [ka[0], ka[0][:-1] + "B"])
        with torch.no_grad():
            W = ln.mod.weight
            W.copy_((W.float() + scale * (Bm.float().cuda() @ A.float().cuda())).to(W.dtype))
        n += 1
    log(f"merged {n} LoRA adapters (scale {scale}) of {len(keys)} in the checkpoint")
    if n != len(keys):
        raise SystemExit(f"LoRA merge matched {n} of {len(keys)} adapters")
    return n


# -------------------------------------------------------------------------------------------------------- adam
def adam_factors(ckpt_dir, lins, eps=1e-8):
    """Rank-1 log-additive fit of the Adam preconditioner D = (v + eps^2)^-1/2 per linear: log D_ij ~ a_i + b_j.
    Returns {name: (exp(a) (out,), exp(b) (in,))} and per-linear fit diagnostics."""
    from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

    import glob as _glob
    run = ckpt_dir[:-len("/model_and_optim")] if ckpt_dir.rstrip("/").endswith("model_and_optim") else ckpt_dir
    cands = [ckpt_dir] + sorted(_glob.glob(f"{run}/step*/model_and_optim"),
                                key=lambda q: -int(q.split("/step")[-1].split("/")[0]))
    keys = []
    for c in cands:
        try:
            md = get_checkpoint_metadata(c)
        except Exception as exc:  # noqa: BLE001
            log(f"adam: {c}: {exc}")
            continue
        allk = list(md.state_dict_metadata)
        keys = [k for k in allk if k.endswith("exp_avg_sq")]
        log(f"adam: {c}: {len(allk)} keys, {len(keys)} exp_avg_sq (e.g. {(keys or allk)[:2]})")
        if keys:
            ckpt_dir = c
            break
    if not keys:
        return None, {"source": "none", "searched": cands}
    out, diag = {}, {}
    for ln in lins:
        pname = f"blocks.{ln.block}.{ln.name.split('.', 1)[1]}.weight"
        cand = [k for k in keys if k.endswith(f"{pname}.exp_avg_sq")]
        if len(cand) != 1:
            raise SystemExit(f"adam: {pname}: {cand[:3]}")
        (v,) = load_keys(ckpt_dir, cand)
        v = v.float().cuda()
        L = -0.5 * torch.log(v + eps * eps)
        a = L.mean(1)
        b = L.mean(0) - L.mean()
        fit = a[:, None] + b[None, :]
        r2 = 1.0 - float((L - fit).var() / L.var().clamp(min=1e-30))
        D, Dh = torch.exp(L), torch.exp(fit)
        diag[ln.name] = dict(r2_log=r2, cos_D=float((D * Dh).sum() / (D.norm() * Dh.norm())))
        out[ln.name] = (torch.exp(a).cpu(), torch.exp(b).cpu())
        del v, L, fit, D, Dh
    diag["_source"] = {"source": "checkpoint exp_avg_sq", "dir": ckpt_dir}
    return out, diag


def estimate_adam_factors(model, lins, batches, eps=1e-8):
    """No optimizer state saved: estimate Adam's second moment from the probe's own dense minibatch gradients
    (16-row batches, mean answer CE per step as in training), Adafactor-style: row / column sums of E[G^2] give
    v_ij ~ R_i C_j / sum(R), so D ~ R_i^-1/2 C_j^-1/2 (global scale irrelevant to cosines)."""
    Rs = {ln.name: torch.zeros(ln.mod.out_features, device="cuda", dtype=torch.float64) for ln in lins}
    Cs = {ln.name: torch.zeros(ln.mod.in_features, device="cuda", dtype=torch.float64) for ln in lins}
    for ln in lins:
        ln.mod.weight.requires_grad_(True)
    tap = argparse.Namespace(cur={})
    nb = 0
    for bi, R in enumerate(batches):
        n_ans = sum(int(r["pred"].numel()) for r in R)
        for r in R:
            loss, _, _ = run_row(model, tap, r, ckpt=True)
            (loss / n_ans).backward()
        for ln in lins:
            g = ln.mod.weight.grad.double().square()
            Rs[ln.name] += g.sum(1)
            Cs[ln.name] += g.sum(0)
            ln.mod.weight.grad = None
        nb += 1
        log(f"[adam-est] batch {bi + 1}/{len(batches)}")
    for ln in lins:
        ln.mod.weight.requires_grad_(False)
    out, diag = {}, {}
    for ln in lins:
        Rv, Cv = Rs[ln.name] / nb, Cs[ln.name] / nb
        tot = Rv.sum()
        a = (Rv + eps * eps * Cv.numel()).rsqrt()
        b = (Cv / tot.clamp(min=1e-300) + eps * eps).rsqrt()
        out[ln.name] = (a.float().cpu(), b.float().cpu())
        diag[ln.name] = dict(r2_log=float("nan"), cos_D=float("nan"))
    diag["_source"] = {"source": f"estimated from {nb} dense 16-row batches (Adafactor row/col sums of E[G^2])"}
    return out, diag


# ---------------------------------------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", default="contradiction,nq,outlier,oolong")
    ap.add_argument("--rows", type=int, default=128, help="rows per task in total (over all shards)")
    ap.add_argument("--row-offset", type=int, default=7808)
    ap.add_argument("--shards", default=P.SHARDS)
    ap.add_argument("--seen-shards", default="mt4f7808,mt4s1952")
    ap.add_argument("--batch", type=int, default=16, help="rows per optimizer step (4 per task)")
    ap.add_argument("--num-shards", type=int, default=1, help="jobs splitting the batches")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--sketch-dim", type=int, default=8192)
    ap.add_argument("--fold-dim", type=int, default=2048)
    ap.add_argument("--adam-ckpt", default="", help="checkpoint dir with optimizer state (exp_avg_sq)")
    ap.add_argument("--adam-factors", default="", help="precomputed factors (.pt from --only-adam-factors)")
    ap.add_argument("--only-adam-factors", default="", help="write the Adam factors to this .pt and exit")
    ap.add_argument("--exact", default="b16.attention.w_out,b20.feed_forward.w2")
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--lora-scale", type=float, default=32.0, help="LoRA checkpoints (recipe): W += scale * B A")
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--sels", default="", help="debug: comma list of selections (default all)")
    ap.add_argument("--save-dir", required=False, default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    t00 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    model = P.build_model(P.find_ckpt(a.ckpt), a.attn_backend)
    lins = P.list_linears(model)
    merged = merge_lora(model, lins, P.find_ckpt(a.ckpt), a.lora_scale)
    adam, adam_diag = None, {}
    if a.adam_factors:
        blob = torch.load(a.adam_factors)
        adam, adam_diag = blob["factors"], blob["diag"]
    elif a.adam_ckpt:
        adam, adam_diag = adam_factors(P.find_ckpt(a.adam_ckpt), lins)
        if adam is None:
            log(f"adam: no optimizer state found ({adam_diag})")
            if not a.only_adam_factors:
                raise SystemExit("no optimizer state: run --only-adam-factors (estimates from batches) first")
    if adam is None and a.only_adam_factors:
        tasks_ = a.tasks.split(",")
        seen_ = [os.path.join(a.shards, s_) for s_ in a.seen_shards.split(",") if s_]
        rbt, _ = P.load_rows(a.shards, tasks_, a.rows, a.row_offset, 12288, seen_)
        per_ = a.batch // len(tasks_)
        nb_ = min(len(v) for v in rbt.values()) // per_
        if a.max_batches:
            nb_ = min(nb_, a.max_batches)
        batches = [[prep_row(rr, t) for t in tasks_ for rr in rbt[t][J * per_:(J + 1) * per_]] for J in range(nb_)]
        adam, adam_diag = estimate_adam_factors(model, lins, batches)
    if adam_diag and any(isinstance(d, dict) and "r2_log" in d for d in adam_diag.values()):
        r2 = [d["r2_log"] for k, d in adam_diag.items() if not k.startswith("_")]
        cd = [d["cos_D"] for k, d in adam_diag.items() if not k.startswith("_")]
        log(f"adam factor fit: log-R2 median {np.median(r2):.3f} (min {min(r2):.3f}); cos(D, fit) median "
            f"{np.median(cd):.4f} (min {min(cd):.4f})")
    if a.only_adam_factors:
        os.makedirs(os.path.dirname(os.path.abspath(a.only_adam_factors)), exist_ok=True)
        torch.save({"factors": adam, "diag": adam_diag}, a.only_adam_factors)
        json.dump({"adam_diag": adam_diag}, open(a.out, "w"))
        log(f"wrote {a.only_adam_factors}")
        return
    if adam is not None:
        adam = {k: (u.cuda().float(), w.cuda().float()) for k, (u, w) in adam.items()}

    tasks = a.tasks.split(",")
    seen = [os.path.join(a.shards, s) for s in a.seen_shards.split(",") if s]
    rows_by_task, _ = P.load_rows(a.shards, tasks, a.rows, a.row_offset, 12288, seen)
    n_per = min(len(v) for v in rows_by_task.values())
    per = a.batch // len(tasks)
    nbatch = n_per // per
    mine = [j for j in range(nbatch) if j % a.num_shards == a.shard]
    if a.max_batches:
        mine = mine[:a.max_batches]
    log(f"{nbatch} batches of {a.batch} rows; shard {a.shard}/{a.num_shards} runs {mine}")

    exact_names = [s for s in a.exact.split(",") if s]
    tap = Tap(lins, a.sketch_dim, adam=adam, exact=exact_names)
    tap.ckpt = not a.no_ckpt
    nm = a.sketch_dim // 2 + 1
    sels = [s for s in sel_names() if not a.sels or s in a.sels.split(",")]
    focus = [s for s in FOCUS if s in sels]
    nT = len(tasks)
    dev = torch.device("cuda")
    nS = None  # set from the first row
    ridge = {}
    gen = torch.Generator(device="cpu").manual_seed(1234 + a.shard)

    def new_acc(nb):
        return torch.zeros(nT, NG, nb, 2 * nm, device=dev)

    acc_A = {s: new_acc(8) for s in sels}  # dense pass: bins type + 4 * dropped
    acc_R = {s: new_acc(4) for s in sels}  # routed pass: bins type
    acc_Aa = {s: new_acc(2) for s in sels} if adam else {}  # adam: kept / dropped
    acc_Ra = {s: new_acc(1) for s in sels} if adam else {}
    log(f"{len(sels)} selections; accumulators "
        f"{sum(t.numel() for d in (acc_A, acc_R, acc_Aa, acc_Ra) for t in d.values()) * 4 / 2**30:.1f} GiB; "
        f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    by_name = {ln.name: ln for ln in lins}
    for nme in exact_names:
        if nme not in by_name:
            raise SystemExit(f"--exact {nme}: no such linear; e.g. {[ln.name for ln in lins if ln.block == 16]}")

    def to_real(A):
        """(..., 2nm) accumulated FFT products -> (..., m) real sketch."""
        sh = A.shape[:-1]
        return torch.fft.irfft(torch.view_as_complex(A.reshape(-1, nm, 2).contiguous()), n=a.sketch_dim,
                               dim=1).reshape(*sh, a.sketch_dim)

    def fold(S, m2):
        return S.reshape(*S.shape[:-1], S.shape[-1] // m2, m2).sum(-2)

    def q16(S):
        """fp16 + fp32 scale per leading index set (all but the last dim)."""
        sc = S.abs().amax(-1, keepdim=True).clamp(min=1e-30)
        return (S / sc).half().cpu().numpy(), sc[..., 0].float().cpu().numpy()

    out_rows = []
    exact_log = []
    save_dir = os.path.join(a.save_dir, a.label) if a.save_dir else ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    blk = torch.arange(NB)
    t0 = time.time()
    n_done = 0
    for bj, J in enumerate(mine):
        batch = []
        for t in tasks:
            for rr in rows_by_task[t][J * per:(J + 1) * per]:
                batch.append((t, rr))
        R = [prep_row(rr, t) for t, rr in batch]
        # ---------------------------------------------------------------- pass 0: dense energies, router features
        tap.mode = "energy"
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            tap.E = torch.zeros(NB, r["T"], device=dev)
            tap.fired = 0
            loss, capt, _ = run_row(model, tap, r, cap=(7, 16), ckpt=tap.ckpt)
            tap.backward(loss)
            if tap.fired != len(lins):
                raise SystemExit(f"hooks fired {tap.fired} for {len(lins)} linears")
            E = tap.E
            r["down"] = {L: E[L:].sum(0) for L in (7, 16)}
            r["energy_type"] = {L: [float(E[L:].sum(0)[r["ty"] == k].sum() / E[L:].sum()) for k in range(4)]
                                for L in (7, 16)}
            r["ce_dense"] = float(loss)
            s = scalar_feats(r)
            r["s"] = s
            nS = s.shape[1]
            for L in (7, 16):
                hn = rms(capt[L][0].detach())
                g = capt[L].grad[0].float()
                q = query_vec(hn, r["qm"])
                r[f"hn{L}"] = hn.half()
                r[f"q{L}"] = q
                tg = torch.log(g.norm(dim=-1) + 1e-12)
                bidx = torch.nonzero(r["body"], as_tuple=True)[0]
                smp = bidx[torch.randperm(bidx.numel(), generator=gen)[:SAMPLES].to(dev)]
                X = feats(hn[smp], q, s[smp])
                y = tg[smp] - tg[smp].mean()
                r[f"fit{L}"] = (X, y)
            del capt
            tap.E = None
            n = n_done + i + 1
            if n in (1, 2, 5, 10) or n % 16 == 0:
                log(f"[p0] batch {J} row {i} {t} T={r['T']} CE={r['ce_dense']:.3f} gold_docs={r['gold_docs']} "
                    f"numbered={r['numbered_ok']} {time.time() - t0:.0f}s")
        # ---------------------------------------------------------------- router fits (2 folds) and scores
        nf = nS + 1 + 2 * R[0]["hn7"].shape[1]
        for L in (7, 16):
            if L not in ridge:
                ridge[L] = Ridge(nf)
            st = []
            for fo in (0, 1):  # summed stats per fold (rows i % 2 == fo)
                parts = [ridge[L].stats(*r[f"fit{L}"]) for i, r in enumerate(R) if i % 2 == fo]
                st.append((sum(p_[0] for p_ in parts), sum(p_[1] for p_ in parts)))
                del parts
            for fo in (0, 1):
                w = ridge[L].solve([st[1 - fo]])
                for i, r in enumerate(R):
                    if i % 2 == fo:
                        r[f"w{L}"] = w
                        r[f"rscore{L}"] = feats(r[f"hn{L}"].float(), r[f"q{L}"], r["s"]) @ w
            # router fidelity on this batch: spearman of the score vs the target over the sampled tokens
            ridge[L].push(st)
        for r in R:
            for L in (7, 16):
                X, y = r[f"fit{L}"]
                p = X @ r[f"w{L}"]
                r[f"rfid{L}"] = float(torch.corrcoef(torch.stack([p.argsort().argsort().float(),
                                                                   y.argsort().argsort().float()]))[0, 1])
                del r[f"fit{L}"]
        # ---------------------------------------------------------------- selections (keep masks, original positions)
        bodies = [r["body"] for r in R]
        K = {}  # sel -> list over rows of {L: keep mask}
        for L, ks in SPLIT_KS.items():
            for k in ks:
                og = global_topk([r["down"][L] for r in R], bodies, k)
                for s in SELECTORS:
                    lst = []
                    for i, r in enumerate(R):
                        if s == "og":
                            kp = og[i]
                        elif s == "od":
                            kp = doc_topk(r["down"][L], r["body"], r["cid"], k)
                        elif s == "rnd":
                            kp = doc_topk(torch.rand(r["T"], device=dev), r["body"], r["cid"], k)
                        else:
                            kp = doc_topk(r[f"rscore{L}"], r["body"], r["cid"], k,
                                          prio=explore_mask(r["body"], k, gen))
                        lst.append({L: kp})
                    K[f"{s}_L{L}_k{k:g}"] = lst
        (L1, k1), (L2, k2) = RECIPE
        f2 = k2 / k1
        og1 = global_topk([r["down"][L1] for r in R], bodies, k1)
        og2 = global_topk([r["down"][L2] for r in R], [b & m for b, m in zip(bodies, og1)],
                          k2 * sum(int(b.sum()) for b in bodies) / max(1, sum(int((b & m).sum()) for b, m in zip(bodies, og1))))
        for s in SELECTORS:
            lst = []
            for i, r in enumerate(R):
                b = r["body"]
                if s == "og":
                    m1, m2 = og1[i], og2[i]
                elif s == "od":
                    m1 = doc_topk(r["down"][L1], b, r["cid"], k1)
                    m2 = doc_topk(r["down"][L2], b & m1, r["cid"], f2)
                elif s == "rnd":
                    m1 = doc_topk(torch.rand(r["T"], device=dev), b, r["cid"], k1)
                    m2 = doc_topk(torch.rand(r["T"], device=dev), b & m1, r["cid"], f2)
                else:
                    m1 = doc_topk(r[f"rscore{L1}"], b, r["cid"], k1, prio=explore_mask(b, k1, gen))
                    # attribution: stage 2 scored on the DENSE h16 (the routed pass re-scores its own h16)
                    m2 = doc_topk(r[f"rscore{L2}"], b & m1, r["cid"], f2, prio=explore_mask(b & m1, f2, gen))
                lst.append({L1: m1, L2: m2})
            K[f"{s}_recipe"] = lst

        def drop_fn(sel, i):
            ks_ = K[sel][i]
            body = R[i]["body"]
            Ls = sorted(ks_)
            cache = {}

            def fn(bk):
                Lh = max([L for L in Ls if L <= bk], default=None)
                if Lh is None:
                    return 0
                if Lh not in cache:
                    cache[Lh] = (body & ~ks_[Lh]).long()
                return cache[Lh]
            return fn

        # ---------------------------------------------------------------- pass 1: dense sketch (attribution)
        tap.mode = "sketch"
        ex_acc = {}
        for nme in exact_names:
            ln = by_name[nme]
            ex_acc[nme] = {v: torch.zeros(ln.mod.out_features, ln.mod.in_features, device=dev)
                           for v in ("D", "A_og_recipe", "R_og_recipe", "R_rnd_recipe")}
        lsk_acc = {nme: {v: torch.zeros(2 * nm, device=dev) for v in ("D", "A_og_recipe", "R_og_recipe",
                                                                       "R_rnd_recipe")} for nme in exact_names}
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            tap.task = tasks.index(t)
            ty = r["ty"]
            specs = []
            for s in sels:
                dfn = drop_fn(s, i)
                specs.append((acc_A[s], (lambda bk, f=dfn, ty=ty: ty + 4 * f(bk)), False))
                if adam:
                    specs.append((acc_Aa[s], (lambda bk, f=dfn, ty=ty: (f(bk) * torch.ones_like(ty))), True))
            tap.specs = specs
            ones = torch.ones(r["T"], dtype=torch.bool, device=dev)
            tap.exact, tap.lin_sk = {}, {}
            if "og_recipe" in sels:
                dfn = drop_fn("og_recipe", i)
                for nme in exact_names:
                    kept = (lambda bk, f=dfn: (1 - (f(bk) * torch.ones_like(ty))).bool())
                    tap.exact[nme] = [(ex_acc[nme]["D"], lambda bk: ones), (ex_acc[nme]["A_og_recipe"], kept)]
                    tap.lin_sk[nme] = [(lsk_acc[nme]["D"], lambda bk: ones), (lsk_acc[nme]["A_og_recipe"], kept)]
            tap.fired = 0
            loss, _, _ = run_row(model, tap, r, ckpt=tap.ckpt)
            tap.backward(loss)
            out_rows.append(dict(batch=J, task=t, k=rr["k"], md5=rr["md5"], T=r["T"], n_body=int(r["body"].sum()),
                                 n_answer=int(r["pred"].numel()), gold_docs=r["gold_docs"],
                                 numbered_ok=r["numbered_ok"], ce_dense=r["ce_dense"],
                                 type_tokens=[int((r["ty"] == k).sum()) for k in range(4)],
                                 energy_type=r["energy_type"], router_spearman={L: r[f"rfid{L}"] for L in (7, 16)},
                                 ce_routed={}, keep_frac={}, rtr2_overlap=None))
        # ---------------------------------------------------------------- routed passes
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            tap.task = tasks.index(t)
            ty = r["ty"]
            body, cid = r["body"], r["cid"]
            orow = out_rows[-len(R) + i]
            for s in sels:
                ks_ = K[s][i]
                stages = {}
                for L, kp in ks_.items():
                    if s == "rtr_recipe" and L == L2:
                        def fn2(h, cur, r=r, m2=kp):
                            hn = rms(h[0])
                            bc = r["body"][cur]
                            q = query_vec(hn, r["qm"][cur])
                            sc = feats(hn, q, r["s"][cur]) @ r[f"w{L2}"]
                            kk = doc_topk(sc, bc, r["cid"][cur], f2, prio=explore_mask(bc, f2, gen))
                            orow["rtr2_overlap"] = float((kk & m2[cur]).sum() / max(1, int(m2.sum())))
                            r["_rtr_m2"] = torch.zeros_like(r["body"])
                            r["_rtr_m2"][cur[kk]] = True
                            return kk | ~bc
                        stages[L] = fn2
                    else:
                        stages[L] = (lambda h, cur, kp=kp, body=body: (kp | ~body)[cur])
                specs = [(acc_R[s], (lambda bk, ty=ty: ty), False)]
                if adam:
                    specs.append((acc_Ra[s], (lambda bk, ty=ty: torch.zeros_like(ty)), True))
                tap.specs = specs
                tap.exact, tap.lin_sk = {}, {}
                if s in ("og_recipe", "rnd_recipe"):
                    for nme in exact_names:
                        tap.exact[nme] = [(ex_acc[nme][f"R_{s}"], lambda bk, T=r["T"]: torch.ones(T, dtype=torch.bool,
                                                                                                  device=dev))]
                        tap.lin_sk[nme] = [(lsk_acc[nme][f"R_{s}"], lambda bk, T=r["T"]: torch.ones(
                            T, dtype=torch.bool, device=dev))]
                tap.fired = 0
                loss, _, cur = run_row(model, tap, r, stages=stages, ckpt=tap.ckpt)
                tap.backward(loss)
                orow["ce_routed"][s] = float(loss)
                orow["keep_frac"][s] = {L: float((kp & body).sum() / max(1, int(body.sum()))) for L, kp in ks_.items()}
                if s == "rtr_recipe" and "_rtr_m2" in r:
                    orow["keep_frac"][s]["16_routed"] = float(r["_rtr_m2"].sum() / max(1, int(body.sum())))
                    del r["_rtr_m2"]
            n_done += 1
            if n_done in (1, 2, 5, 10) or n_done % 8 == 0:
                el = time.time() - t0
                tot = len(mine) * a.batch
                log(f"[routed] row {n_done}/{tot} ({t}) CE dense {r['ce_dense']:.3f} "
                    f"og_recipe {orow['ce_routed'].get('og_recipe', float('nan')):.3f} "
                    f"rnd_recipe {orow['ce_routed'].get('rnd_recipe', float('nan')):.3f}; {el:.0f}s, "
                    f"ETA {el / n_done * (tot - n_done):.0f}s; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        # ---------------------------------------------------------------- finalize batch
        ex = {}
        for nme in exact_names:
            E_ = {k: v.double() for k, v in ex_acc[nme].items()}
            S_ = {k: torch.fft.irfft(torch.view_as_complex(v.view(nm, 2).contiguous()), n=a.sketch_dim).double()
                  for k, v in lsk_acc[nme].items()}

            def c(d, u, v):
                return float((d[u] * d[v]).sum() / (d[u].norm() * d[v].norm()).clamp(min=1e-300))
            ex[nme] = {f"{u}~{v}": (c(E_, u, v), c(S_, u, v)) for u, v in (("A_og_recipe", "D"), ("R_og_recipe", "D"),
                                                                             ("R_rnd_recipe", "D"),
                                                                             ("R_og_recipe", "A_og_recipe"))}
            log(f"exact check batch {J} {nme}: " + ", ".join(f"{k} exact {v[0]:.4f} sketch {v[1]:.4f}"
                                                             for k, v in ex[nme].items()))
        exact_log.append(dict(batch=J, **ex))
        if save_dir:
            arr = {}
            first = sels[0]
            SA = {s: to_real(acc_A[s]) for s in sels}  # (nT, NG, 8, m)
            SR = {s: to_real(acc_R[s]) for s in sels}  # (nT, NG, 4, m)
            Dt = SA[first].sum(2)  # (nT, NG, m)
            arr["tot_D"], arr["tot_D_sc"] = q16(Dt.sum(0))
            arr["tsk_D"], arr["tsk_D_sc"] = q16(fold(Dt, a.fold_dim))
            for s in sels:
                At = SA[s][:, :, :4].sum(2)
                Rt = SR[s].sum(2)
                arr[f"tot_A_{s}"], arr[f"tot_A_{s}_sc"] = q16(At.sum(0))
                arr[f"tot_R_{s}"], arr[f"tot_R_{s}_sc"] = q16(Rt.sum(0))
                arr[f"tsk_A_{s}"], arr[f"tsk_A_{s}_sc"] = q16(fold(At, a.fold_dim))
                arr[f"tsk_R_{s}"], arr[f"tsk_R_{s}_sc"] = q16(fold(Rt, a.fold_dim))
                if s in focus:
                    arr[f"typ_A8_{s}"], arr[f"typ_A8_{s}_sc"] = q16(fold(SA[s], a.fold_dim))  # (nT, NG, 8, mf)
                    arr[f"typ_R_{s}"], arr[f"typ_R_{s}_sc"] = q16(fold(SR[s], a.fold_dim))
            if adam:
                SAa = {s: to_real(acc_Aa[s]) for s in sels}
                SRa = {s: to_real(acc_Ra[s]) for s in sels}
                arr["adam_tot_D"], arr["adam_tot_D_sc"] = q16(SAa[first].sum((0, 2)))
                for s in sels:
                    arr[f"adam_tot_A_{s}"], arr[f"adam_tot_A_{s}_sc"] = q16(SAa[s][:, :, 0].sum(0))
                    arr[f"adam_tot_R_{s}"], arr[f"adam_tot_R_{s}_sc"] = q16(SRa[s][:, :, 0].sum(0))
            np.savez(os.path.join(save_dir, f"b{J:03d}.npz"), **arr)
            del SA, SR
            log(f"saved {save_dir}/b{J:03d}.npz ({sum(v.nbytes for v in arr.values()) / 2**20:.0f} MiB)")
        for d in (acc_A, acc_R, acc_Aa, acc_Ra):
            for v in d.values():
                v.zero_()
        del R
        log(f"batch {J} done ({bj + 1}/{len(mine)}), {time.time() - t0:.0f}s")
        json.dump(dict(label=a.label, ckpt=a.ckpt, shard=a.shard, num_shards=a.num_shards, batches=mine[:bj + 1],
                       tasks=tasks, sels=sels, focus=focus, types=TYPES, split_ks={str(k): v for k, v in SPLIT_KS.items()},
                       recipe=RECIPE, sketch_dim=a.sketch_dim, fold_dim=a.fold_dim, adam=bool(adam), adam_diag=adam_diag,
                       exact=exact_log, rows=out_rows, seconds=time.time() - t00,
                       lins=[dict(name=ln.name, block=ln.block, part=ln.part) for ln in lins]),
                  open(a.out, "w"))
        if save_dir:
            json.dump(json.load(open(a.out)), open(os.path.join(save_dir, f"meta_shard{a.shard}.json"), "w"))
    log(f"done {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
