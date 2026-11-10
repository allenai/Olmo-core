"""Context-fidelity probe: can cheap stand-ins for dropped document tokens keep the kept tokens' hidden states and
gradients dense-like? (Prasann, 2026-11-10; experiment tokdrop_lr.) No training.

Setup (follows records/grad-routing-gap.md and reuses its code in ../grad_concentration/): a frozen Qwen3.5-4B
checkpoint (32 blocks; full attention at blocks 3, 7, ..., 31, Gated DeltaNet elsewhere), 16-row batches (4 rows per
task = one optimizer step) of unseen fresh-pool rows. One cut at the INPUT of block 7 (blocks 0-6 see every token, as
in the recipe), keeping 20% of each document's body tokens; prompt / markers / query / answer always kept. Kept tokens
keep their original RoPE positions. Selections (per document, doc_share 1):
  rtr  the ridge emulation of the online router (routed_probe.py; the recipe checkpoint saved no router state)
  rnd  uniform random
  od   gradient-attribution top-k (dense down7 energy, sum over blocks >= 7 of ||delta_t||^2 ||x_t||^2)
Variants for the dropped tokens above the cut (DeltaNet blocks always see the kept tokens only):
  drop       plain drop (the reference; the earlier probe's <sel>_L7_k0.2)
  frozen     dropped tokens stay as keys/values of every full-attention block >= 7 at their block-7 INPUT state
             (projected by that block's own norm / K / V), constants: no update, no gradient
  refresh15  frozen, but the dropped tokens also run block 15 (no grad) once; blocks 19-31 see the refreshed state
  kvfresh    diagnostic upper bound of the frozen family: the dropped tokens' DENSE input state at each full-attention
             block (forward-only dense pass for them; no gradient)
  mean1      one stand-in per document = mean of its dropped tokens' block-7 states, logit bias log(count), placed at
             the document's last dropped position (visible to strictly later queries)
  span64     one stand-in per 64-token span of each document, same construction (spanN: N-token spans; span1 is a
             correctness check: each dropped token its own stand-in, count 1, must reproduce frozen)
  lin7       plain drop + a fixed linear correction added to the kept tokens after block 15:
             delta_hat = W [h7; 1], W ridge-fitted on held-out rows to delta = h16_dense - h16_drop
  lin7r      the same with inputs [h7; h16_drop; 1]
  frozen_all diagnostic: frozen context in EVERY block >= 7, DeltaNet included (the dropped tokens' stale block-7
             state is fed to each mixer as constant input; their outputs are discarded)
  staleNN    frozen_all with only a random NN% of each document's dropped tokens as stale context (stale20, stale50;
             per-row seed, so the selections are unchanged); the rest are dropped
  fwdonly    diagnostic bound for any no-gradient context: the exact dense forward (hidden states and CE = dense), the
             dropped tokens detached at every block input >= 7 and excluded from the gradient
Gradients keep only the kept tokens' weight terms (dropped rows binned out; they are zero except in fwdonly /
frozen_all, where the dropped rows' own projections run).
Per row x selection x variant: answer CE (summed), relative hidden-state error vs dense at the kept positions after
blocks 15 / 23 / 31 (mean over kept tokens of ||h - h_dense|| / ||h_dense||; also over answer positions), the analytic
step cost (flop_meter.py convention: 6 x params per token run with grad, 12 n_heads head_dim per attended pair,
2 x params per token run forward-only, LM head per token run) vs the dense step, and the gradient as a TensorSketch
(raw and Adam-preconditioned, per task folded) for cos / expected cos / norm-ratio analysis in analyze_context.py.

    python research/token_routing/context_fidelity/context_probe.py --ckpt <dir> --label f1952 --shards <dir> \
        --adam-factors adam_f1952.pt --num-shards 8 --shard 0 --save-dir /data/.../ctxfid --out x.json
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
sys.path.insert(0, os.path.join(HERE, "..", "grad_concentration"))
import probe as P  # noqa: E402
import routed_probe as RP  # noqa: E402

NB = 32
NG = 2 * NB
CUT = 7
KEEP = 0.2
ATTN = tuple(b for b in range(NB) if b % 4 == 3)
ATTN_UP = tuple(b for b in ATTN if b >= CUT)
ERR_BLOCKS = (15, 23, 31)
REFRESH = 15
CORR_AT = 15
SPAN = 64
SELECTORS = ("rtr", "rnd", "od")
VARIANTS = ("drop", "frozen", "refresh15", "kvfresh", "mean1", "span64", "lin7", "lin7r", "frozen_all", "fwdonly",
            "stale20", "stale50")
LAMBDAS = (1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def log(m):
    print(f"[ctx-probe] {m}", flush=True)


def ck(block, h, kw, use_ckpt):
    from torch.utils.checkpoint import checkpoint
    if use_ckpt:
        return checkpoint(lambda hh, bl=block, k=kw: bl(hh, **k), h, use_reentrant=False)
    return block(h, **kw)


# ------------------------------------------------------------------------------------------------------ forwards
def dense_row(model, tap, r, use_ckpt=True):
    """dense forward; captures (detached) the input of every full-attention block >= CUT and the outputs of
    ERR_BLOCKS / CORR_AT, each (T, d)."""
    from torch.utils.checkpoint import set_checkpoint_early_stop

    x, pred, tgt, T = r["x"], r["pred"], r["tgt"], r["T"]
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    h = model.embeddings(x[None]).detach().requires_grad_(True)
    cur = torch.arange(T, device=x.device)
    caps = {}
    with set_checkpoint_early_stop(False):
        for key, block in model.blocks.items():
            b = int(key)
            if b == CUT or b in ATTN_UP:
                caps[f"in{b}"] = h.detach()[0]
            tap.cur[b] = cur
            h = ck(block, h, {**abk, **pbk.get(b, {})}, use_ckpt)
            if b in ERR_BLOCKS or b == CORR_AT:
                caps[f"out{b}"] = h.detach()[0]
    logits = model.lm_head(h, **lmk)[0]
    return F.cross_entropy(logits.float(), tgt, reduction="sum"), caps


def union_ctx(ctx, kidx, cidx):
    """kept + context rows in original-position order (the union the context blocks run on)."""
    upos = torch.sort(torch.cat([kidx, cidx])).values
    ukeep = torch.zeros(upos.numel(), dtype=torch.bool, device=upos.device)
    kin = torch.searchsorted(upos, kidx)
    ukeep[kin] = True
    return dict(ctx, cidx=cidx, upos=upos, kin=kin, cin=torch.searchsorted(upos, cidx), ukeep=ukeep)


def make_ctx(var, hcut, r, keep, kidx, didx):
    """context for the full-attention blocks >= CUT. hcut: (T, d) detached block-7 input."""
    if var in ("frozen", "refresh15", "frozen_all") or var.startswith("stale"):
        cidx = didx
        if var.startswith("stale"):  # staleNN: a random NN% of each document's dropped tokens stay as stale context
            g = torch.Generator(device="cpu").manual_seed(int(r["x"].sum()) * 7919 + r["T"])
            sc = torch.rand(r["T"], generator=g).to(hcut.device)
            cidx = torch.nonzero(RP.doc_topk(sc, ~keep, r["cid"], int(var[5:]) / 100.0), as_tuple=True)[0]
        return union_ctx(dict(kind="full", state=hcut[cidx], all=var == "frozen_all" or var.startswith("stale")),
                         kidx, cidx)
    if var == "fwdonly":
        return dict(kind="fwdonly")
    if var == "kvfresh":
        return union_ctx(dict(kind="full", state=None, all=False), kidx, didx)
    if var == "mean1" or var.startswith("span"):
        dev = hcut.device
        cid = r["cid"]
        T = hcut.shape[0]
        if didx.numel() == 0:
            return None
        dc = cid[didx]
        if bool((dc < 0).any()):
            raise RuntimeError("a dropped token has no document id")
        if var == "mean1":
            key = dc
        else:
            first = torch.full((int(cid.max()) + 1,), T, dtype=torch.long, device=dev)
            bpos = torch.nonzero(r["body"], as_tuple=True)[0]
            first.scatter_reduce_(0, cid[bpos], bpos, reduce="amin")
            key = dc * 100000 + (didx - first[dc]) // int(var[4:])
        uk, inv = torch.unique(key, return_inverse=True)
        S = uk.numel()
        cnt = torch.bincount(inv, minlength=S).float()
        sm = torch.zeros(S, hcut.shape[1], device=dev, dtype=torch.float32).index_add_(0, inv, hcut[didx].float())
        st = (sm / cnt[:, None]).to(hcut.dtype)
        spos = torch.full((S,), -1, dtype=torch.long, device=dev).scatter_reduce_(0, inv, didx, reduce="amax")
        o = torch.argsort(spos)
        st, spos, cnt = st[o], spos[o], cnt[o]
        Tk = kidx.numel()
        pos = torch.cat([kidx, spos])
        is_s = torch.cat([torch.zeros(Tk, dtype=torch.bool, device=dev), torch.ones(S, dtype=torch.bool, device=dev)])
        pq, pk = pos[:, None], pos[None, :]
        allowed = torch.where(is_s[None, :], pk < pq, pk <= pq) | torch.eye(Tk + S, dtype=torch.bool, device=dev)
        colb = torch.cat([torch.zeros(Tk, device=dev), torch.log(cnt)])
        bias = torch.where(allowed, colb[None, :].expand(Tk + S, -1), torch.full((), float("-inf"), device=dev))
        kvm = ~is_s
        return dict(kind="standin", state=st, pos=pos, bias=bias[None, None], kvm=kvm[None], S=S, cnt=cnt)
    return None


def variant_row(model, tap, r, keep, var, dcap=None, corr=None, use_ckpt=True, want=None):
    """routed forward for one variant. ``keep`` (T,) bool over original positions. Returns (loss, outs, info):
    outs {b: (Tk, d) detached} for ERR_BLOCKS / CORR_AT; info: kidx and the context object."""
    from torch.utils.checkpoint import set_checkpoint_early_stop

    x, pred, tgt, T = r["x"], r["pred"], r["tgt"], r["T"]
    dev = x.device
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    h = model.embeddings(x[None]).detach().requires_grad_(True)
    allpos = torch.arange(T, device=dev)
    kidx = torch.nonzero(keep, as_tuple=True)[0]
    didx = torch.nonzero(~keep, as_tuple=True)[0]
    outs, ctx, hcut_k = {}, None, None
    want = want if want is not None else set(ERR_BLOCKS) | {CORR_AT}
    with set_checkpoint_early_stop(False):
        for key, block in model.blocks.items():
            b = int(key)
            kw = {**abk, **pbk.get(b, {})}
            if b == CUT:
                ctx = make_ctx(var, h.detach()[0], r, keep, kidx, didx)
                hcut_k = h[:, kidx]
                if ctx is None or ctx["kind"] != "fwdonly":
                    h = hcut_k
            fwdo = ctx is not None and ctx["kind"] == "fwdonly"
            if b < CUT or (fwdo and b >= CUT):
                tap.cur[b] = allpos
                if fwdo:
                    h = torch.where(keep[None, :, None], h, h.detach())
                    kw = {**kw, "kv_grad_mask": keep[None]} if b in ATTN else kw
                h = ck(block, h, kw, use_ckpt)
            elif ctx is None or didx.numel() == 0 or (b not in ATTN_UP and not ctx.get("all")):
                tap.cur[b] = kidx
                h = ck(block, h, {**kw, "position_ids": kidx[None]}, use_ckpt)
            elif ctx["kind"] == "full":
                C = ctx["state"] if ctx["state"] is not None else dcap[f"in{b}"][ctx["cidx"]]
                upos = ctx["upos"]
                U = h.new_zeros(upos.numel(), h.shape[-1]).index_copy(0, ctx["kin"], h[0]) \
                    .index_copy(0, ctx["cin"], C.to(h.dtype))[None]
                tap.cur[b] = upos
                kwu = {**kw, "position_ids": upos[None]}
                if b in ATTN:
                    kwu["kv_grad_mask"] = ctx["ukeep"][None]
                out = ck(block, U, kwu, use_ckpt)
                h = out[:, ctx["kin"]]
                if var == "refresh15" and b == REFRESH:
                    ctx["state"] = out[0, ctx["cin"]].detach()
            else:  # stand-ins appended after the kept tokens; position-causal masked SDPA with log-count bias
                U = torch.cat([h, ctx["state"][None].to(h.dtype)], dim=1)
                tap.cur[b] = ctx["pos"]
                out = ck(block, U, {**kw, "position_ids": ctx["pos"][None], "attn_bias": ctx["bias"],
                                    "kv_grad_mask": ctx["kvm"]}, use_ckpt)
                h = out[:, :kidx.numel()]
            if b == CORR_AT and corr is not None:
                h = h + corr(hcut_k[0], h[0])[None].to(h.dtype)
            if b in want:
                outs[b] = h.detach()[0][kidx] if fwdo else h.detach()[0]
    if ctx is not None and ctx["kind"] == "fwdonly":
        logits = model.lm_head(h, **lmk)[0]
        return F.cross_entropy(logits.float(), tgt, reduction="sum"), outs, dict(kidx=kidx, didx=didx, ctx=ctx)
    newcol = torch.full((T,), -1, dtype=torch.long, device=dev)
    newcol[kidx] = torch.arange(kidx.numel(), device=dev)
    lt = newcol[pred]
    if bool((lt < 0).any()):
        raise RuntimeError("an answer position was dropped")
    logits = model.lm_head(h, **{**lmk, "logits_to_keep": lt[None]})[0]
    return F.cross_entropy(logits.float(), tgt, reduction="sum"), outs, dict(kidx=kidx, didx=didx, ctx=ctx)


# ---------------------------------------------------------------------------------------------------------- flops
class Flops:
    """analytic training FLOPs of one row (flop_meter.py / token_exit_example_flops convention)."""

    def __init__(self, model):
        bl = list(model.blocks.values())
        self.P = [int(b.num_flops_per_token(0)) for b in bl]  # 6 x params (+ GDN conv / recurrence), per token
        self.Pmix = [int(b.attention.num_flops_per_token(0)) for b in bl]
        a = bl[CUT].attention
        self.pair = 12 * a.n_heads * a.head_dim  # fwd+bwd per attended (query, key) pair
        self.pair_f = 4 * a.n_heads * a.head_dim  # forward only
        self.kv = {b: 2 * (bl[b].attention.w_k.weight.numel() + bl[b].attention.w_v.weight.numel()) for b in ATTN}
        self.lm = int(model.lm_head.num_flops_per_token(0))
        self.d = int(model.d_model) if hasattr(model, "d_model") else 2560

    def dense(self, T):
        return float(T * (sum(self.P) + self.lm) + len(ATTN) * self.pair * T * (T + 1) / 2)

    def row(self, T, kidx, didx, var, ctx):
        Tk, nd = kidx.numel(), didx.numel()
        pre = sum(self.P[:CUT]) * T + sum(1 for b in ATTN if b < CUT) * self.pair * T * (T + 1) / 2
        post = Tk * (sum(self.P[CUT:]) + self.lm) + len(ATTN_UP) * self.pair * Tk * (Tk + 1) / 2
        extra = 0.0
        if nd and (var in ("frozen", "refresh15", "kvfresh", "frozen_all", "fwdonly") or var.startswith("stale")):
            if ctx is not None and "cidx" in ctx:  # context rows (all dropped tokens, or a subset for staleNN)
                didx = ctx["cidx"]
                nd = didx.numel()
            # kept queries x context keys (charged fwd+bwd), plus the K/V projections of the context tokens (fwd only)
            pairs = float(torch.searchsorted(didx, kidx).sum())
            extra += len(ATTN_UP) * (self.pair * pairs) + nd * sum(self.kv[b] for b in ATTN_UP)
            dq_pairs = float((didx + 1).sum())  # dropped queries over every earlier key
            if var == "refresh15":
                extra += nd * self.P[REFRESH] / 3 + self.pair_f * dq_pairs
            if var == "frozen_all" or var.startswith("stale"):  # stale context through every DeltaNet mixer (fwd only)
                extra += nd * sum(self.Pmix[b] for b in range(CUT, NB) if b not in ATTN) / 3
            if var in ("kvfresh", "fwdonly"):  # forward-only dense pass of the dropped tokens through blocks CUT .. 31
                extra += nd * sum(self.P[CUT:]) / 3 + len(ATTN_UP) * self.pair_f * dq_pairs
        if nd and (var == "mean1" or var.startswith("span")) and ctx is not None:
            S = ctx["S"]
            spos = ctx["pos"][Tk:]
            pairs = float(torch.searchsorted(spos, kidx).sum())
            extra += len(ATTN_UP) * (self.pair * pairs) + S * sum(self.kv[b] for b in ATTN_UP) + 2 * nd * self.d
        if var == "lin7":
            extra += 6 * (self.d + 1) * self.d * Tk
        if var == "lin7r":
            extra += 6 * (2 * self.d + 1) * self.d * Tk
        return float(pre + post + extra)


# ------------------------------------------------------------------------------------------------- linear correction
class CorrStats:
    """ridge stats for delta = h16_dense - h16_drop at kept positions; two halves (row parity) for picking lambda."""

    def __init__(self, d, with_routed):
        self.din = (2 * d if with_routed else d) + 1
        self.with_routed = with_routed
        z = lambda *s: torch.zeros(*s, device="cuda", dtype=torch.float64)  # noqa: E731
        self.A = [z(self.din, self.din), z(self.din, self.din)]
        self.B = [z(self.din, d), z(self.din, d)]
        self.Y = [z(()), z(())]
        self.YY0 = [z(()), z(())]  # ||delta||^2 (error with no correction)
        self.n = [0, 0]

    def feats(self, h7, h16r):
        parts = [h7.double()] + ([h16r.double()] if self.with_routed else []) + \
            [torch.ones(h7.shape[0], 1, device=h7.device, dtype=torch.float64)]
        return torch.cat(parts, dim=1)

    def add(self, half, h7, h16r, delta):
        X = self.feats(h7, h16r)
        Yd = delta.double()
        self.A[half] += X.T @ X
        self.B[half] += X.T @ Yd
        self.Y[half] += (Yd * Yd).sum()
        self.n[half] += X.shape[0]

    def solve(self, A, B, n, lam):
        # scaled ridge: penalty lam * n * var_j on each non-intercept column (= ridge on standardized features)
        mu = A[-1, :-1] / n
        var = (A.diagonal()[:-1] / n - mu * mu).clamp(min=1e-12)
        R = A.clone()
        R.diagonal()[:-1] += lam * n * var
        return torch.linalg.solve(R, B)

    def sse(self, W, h):
        return float((W * (self.A[h] @ W)).sum() - 2 * (W * self.B[h]).sum() + self.Y[h])

    def fit(self):
        res = {}
        for lam in LAMBDAS:
            e = 0.0
            for h in (0, 1):
                W = self.solve(self.A[1 - h], self.B[1 - h], self.n[1 - h], lam)
                e += self.sse(W, h)
            res[lam] = e / float(self.Y[0] + self.Y[1])
        best = min(res, key=res.get)
        W = self.solve(self.A[0] + self.A[1], self.B[0] + self.B[1], self.n[0] + self.n[1], best)
        A, B, Y = self.A[0] + self.A[1], self.B[0] + self.B[1], self.Y[0] + self.Y[1]
        r2_in = 1.0 - float((W * (A @ W)).sum() - 2 * (W * B).sum() + Y) / float(Y)
        W32 = W.float()
        wr = self.with_routed

        def corr(h7, h16r):
            X = torch.cat([h7.float()] + ([h16r.float()] if wr else []) +
                          [torch.ones(h7.shape[0], 1, device=h7.device)], dim=1)
            return X @ W32
        return corr, dict(lambda_cv_rel_sse={str(k): v for k, v in res.items()}, best_lambda=best,
                          heldout_rel_sse=res[best], train_r2=r2_in, n_tokens=sum(self.n))


# ---------------------------------------------------------------------------------------------------------- driver
def select(R, gen, ridge, use_explore=True):
    """pass-0 products (energies, router fits) -> per-row keep masks {sel: keep (T,) bool incl. non-body}."""
    nS = R[0]["s"].shape[1]
    nf = nS + 1 + 2 * R[0]["hn7"].shape[1]
    L = CUT
    if L not in ridge:
        ridge[L] = RP.Ridge(nf)
    st = []
    for fo in (0, 1):
        parts = [ridge[L].stats(*r[f"fit{L}"]) for i, r in enumerate(R) if i % 2 == fo]
        st.append((sum(p_[0] for p_ in parts), sum(p_[1] for p_ in parts)))
    for fo in (0, 1):
        w = ridge[L].solve([st[1 - fo]])
        for i, r in enumerate(R):
            if i % 2 == fo:
                r["rscore"] = RP.feats(r[f"hn{L}"].float(), r[f"q{L}"], r["s"]) @ w
                r["w"] = w
    ridge[L].push(st)
    for r in R:
        X, y = r[f"fit{L}"]
        p = X @ r.pop("w")
        r["rfid"] = float(torch.corrcoef(torch.stack([p.argsort().argsort().float(), y.argsort().argsort().float()]))[0, 1])
        del r[f"fit{L}"]
    K = []
    for r in R:
        b, cid = r["body"], r["cid"]
        ks = {}
        for s in SELECTORS:
            if s == "od":
                kp = RP.doc_topk(r["down"], b, cid, KEEP)
            elif s == "rnd":
                kp = RP.doc_topk(torch.rand(r["T"], device=b.device), b, cid, KEEP)
            else:
                kp = RP.doc_topk(r["rscore"], b, cid, KEEP,
                                 prio=RP.explore_mask(b, KEEP, gen) if use_explore else None)
            ks[s] = kp | ~b
        K.append(ks)
    return K


def pass0(model, tap, R, gen, t0, tag):
    """dense energies (od score) and router features / targets (rtr), as routed_probe.py pass 0 (block 7 only)."""
    dev = torch.device("cuda")
    tap.mode = "energy"
    tap.specs = []
    for i, r in enumerate(R):
        tap.E = torch.zeros(NB, r["T"], device=dev)
        loss, capt, _ = RP.run_row(model, tap, r, cap=(CUT,), ckpt=tap.ckpt)
        tap.backward(loss)
        r["down"] = tap.E[CUT:].sum(0)
        r["ce_dense0"] = float(loss)
        s = RP.scalar_feats(r)
        r["s"] = s
        hn = RP.rms(capt[CUT][0].detach())
        g = capt[CUT].grad[0].float()
        q = RP.query_vec(hn, r["qm"])
        r[f"hn{CUT}"], r[f"q{CUT}"] = hn.half(), q
        tg = torch.log(g.norm(dim=-1) + 1e-12)
        bidx = torch.nonzero(r["body"], as_tuple=True)[0]
        smp = bidx[torch.randperm(bidx.numel(), generator=gen)[:RP.SAMPLES].to(dev)]
        r[f"fit{CUT}"] = (RP.feats(hn[smp], q, s[smp]), tg[smp] - tg[smp].mean())
        del capt
        tap.E = None
        if i in (0, 1, 4, 9) or (i + 1) % 16 == 0:
            log(f"[p0 {tag}] row {i + 1}/{len(R)} T={r['T']} CE={r['ce_dense0']:.3f} {time.time() - t0:.0f}s")
    tap.mode = None


def relerr(hv, hd):
    """mean over tokens of ||hv - hd|| / ||hd|| (fp32)."""
    hv, hd = hv.float(), hd.float()
    return ((hv - hd).norm(dim=-1) / hd.norm(dim=-1).clamp(min=1e-6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", default="contradiction,nq,outlier,oolong")
    ap.add_argument("--rows", type=int, default=128, help="probe rows per task (over all shards)")
    ap.add_argument("--fit-rows", type=int, default=16, help="held-out rows per task for the linear correction")
    ap.add_argument("--row-offset", type=int, default=7808)
    ap.add_argument("--shards", default=P.SHARDS)
    ap.add_argument("--seen-shards", default="mt4f7808,mt4s1952")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--sketch-dim", type=int, default=8192)
    ap.add_argument("--fold-dim", type=int, default=2048)
    ap.add_argument("--adam-factors", default="")
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--lora-scale", type=float, default=32.0)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    t00 = time.time()
    variants = [v for v in a.variants.split(",") if v]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    model = P.build_model(P.find_ckpt(a.ckpt), a.attn_backend)
    lins = P.list_linears(model)
    merged = RP.merge_lora(model, lins, P.find_ckpt(a.ckpt), a.lora_scale)
    adam = None
    if a.adam_factors:
        blob = torch.load(a.adam_factors)
        adam = {k: (u.cuda().float(), w.cuda().float()) for k, (u, w) in blob["factors"].items()}
    fl = Flops(model)
    d = fl.d

    tasks = a.tasks.split(",")
    seen = [os.path.join(a.shards, s) for s in a.seen_shards.split(",") if s]
    rows_by_task, _ = P.load_rows(a.shards, tasks, a.rows + a.fit_rows, a.row_offset, 12288, seen)
    probe_rows = {t: v[:a.rows] for t, v in rows_by_task.items()}
    fit_rows = {t: v[a.rows:a.rows + a.fit_rows] for t, v in rows_by_task.items()}
    per = a.batch // len(tasks)
    nbatch = min(len(v) for v in probe_rows.values()) // per
    mine = [j for j in range(nbatch) if j % a.num_shards == a.shard]
    if a.max_batches:
        mine = mine[:a.max_batches]
    log(f"{nbatch} batches of {a.batch}; shard {a.shard}/{a.num_shards} runs {mine}; fit rows/task "
        f"{min(len(v) for v in fit_rows.values())}; variants {variants}; LoRA merged {merged}")

    tap = RP.Tap(lins, a.sketch_dim, adam=adam, exact=())
    tap.ckpt = True
    nm = a.sketch_dim // 2 + 1
    nT = len(tasks)
    dev = torch.device("cuda")

    # ------------------------------------------------------------ linear corrections (fit rows, fixed seed)
    corrs, corr_diag = {}, {}
    need_lin = [v for v in variants if v in ("lin7", "lin7r")]
    if need_lin and a.fit_rows:
        genf = torch.Generator(device="cpu").manual_seed(4321)
        ridge_f = {}
        stats = {(s, v): CorrStats(d, v == "lin7r") for s in SELECTORS for v in need_lin}
        nfit = min(len(v) for v in fit_rows.values()) // per
        for J in range(nfit):
            R = [RP.prep_row(rr, t) for t in tasks for rr in fit_rows[t][J * per:(J + 1) * per]]
            pass0(model, tap, R, genf, t00, f"fit{J}")
            K = select(R, genf, ridge_f)
            with torch.no_grad():
                for i, r in enumerate(R):
                    _, dcap = dense_row(model, tap, r, use_ckpt=False)
                    for s in SELECTORS:
                        kp = K[i][s]
                        _, outs, info = variant_row(model, tap, r, kp, "drop", use_ckpt=False, want={CORR_AT})
                        kidx = info["kidx"]
                        h7 = dcap[f"in{CUT}"][kidx]
                        hr = outs[CORR_AT]
                        delta = dcap[f"out{CORR_AT}"][kidx].float() - hr.float()
                        for v in need_lin:
                            stats[(s, v)].add(i % 2, h7, hr, delta)
                    del dcap
            del R
            log(f"[fit] batch {J + 1}/{nfit} {time.time() - t00:.0f}s")
        for (s, v), st in stats.items():
            corrs[(s, v)], corr_diag[f"{s}_{v}"] = st.fit()
            log(f"[fit] {s} {v}: {corr_diag[f'{s}_{v}']}")
        del stats
        torch.cuda.empty_cache()

    # ------------------------------------------------------------ probe
    combos = [(s, v) for s in SELECTORS for v in variants]

    def new_acc(nb=1):
        return torch.zeros(nT, NG, nb, 2 * nm, device=dev)

    acc_D, acc_Da = new_acc(), (new_acc() if adam else None)
    acc_R = {c: new_acc(2) for c in combos}  # bins: kept / dropped rows (only bin 0 is the routed gradient)
    acc_Ra = {c: new_acc(2) for c in combos} if adam else {}
    log(f"accumulators {(2 + 2 * len(combos)) * acc_D.numel() * 4 / 2**30:.2f} GiB")

    def to_real(A):
        sh = A.shape[:-1]
        return torch.fft.irfft(torch.view_as_complex(A.reshape(-1, nm, 2).contiguous()), n=a.sketch_dim,
                               dim=1).reshape(*sh, a.sketch_dim)

    def fold(S, m2):
        return S.reshape(*S.shape[:-1], S.shape[-1] // m2, m2).sum(-2)

    def q16(S):
        sc = S.abs().amax(-1, keepdim=True).clamp(min=1e-30)
        return (S / sc).half().cpu().numpy(), sc[..., 0].float().cpu().numpy()

    gen = torch.Generator(device="cpu").manual_seed(1234 + a.shard)
    ridge = {}
    save_dir = os.path.join(a.save_dir, a.label) if a.save_dir else ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    out_rows = []
    t0 = time.time()
    n_done = 0
    tot_rows = len(mine) * a.batch
    for bj, J in enumerate(mine):
        batch = [(t, rr) for t in tasks for rr in probe_rows[t][J * per:(J + 1) * per]]
        R = [RP.prep_row(rr, t) for t, rr in batch]
        pass0(model, tap, R, gen, t0, f"b{J}")
        K = select(R, gen, ridge)
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            tap.task = tasks.index(t)
            zeros = torch.zeros(r["T"], dtype=torch.long, device=dev)
            # dense sketch pass + captures
            tap.mode = "sketch"
            tap.specs = [(acc_D, lambda bk, z=zeros: z, False)] + ([(acc_Da, lambda bk, z=zeros: z, True)] if adam else [])
            tap.fired = 0
            loss, dcap = dense_row(model, tap, r)
            tap.backward(loss)
            if tap.fired != len(lins):
                raise SystemExit(f"hooks fired {tap.fired} for {len(lins)} linears")
            nb = int(r["body"].sum())
            orow = dict(batch=J, task=t, k=rr["k"], md5=rr["md5"], T=r["T"], n_body=nb, n_answer=int(r["pred"].numel()),
                        ce_dense=float(loss), ce_dense0=r["ce_dense0"], router_spearman=r["rfid"],
                        flops_dense=fl.dense(r["T"]), v={})
            pred = r["pred"]
            for s in SELECTORS:
                kp = K[i][s]
                for v in variants:
                    if v in ("lin7", "lin7r") and (s, v) not in corrs:
                        continue
                    dbin = (~kp).long()
                    tap.specs = [(acc_R[(s, v)], lambda bk, z=dbin: z, False)] + \
                        ([(acc_Ra[(s, v)], lambda bk, z=dbin: z, True)] if adam else [])
                    tap.fired = 0
                    loss_v, outs, info = variant_row(model, tap, r, kp, v, dcap=dcap, corr=corrs.get((s, v)))
                    tap.backward(loss_v)
                    kidx, didx = info["kidx"], info["didx"]
                    kmask = kidx > 0  # position 0 is identical in every variant
                    pos_in_k = torch.full((r["T"],), -1, dtype=torch.long, device=dev)
                    pos_in_k[kidx] = torch.arange(kidx.numel(), device=dev)
                    acol = pos_in_k[pred]
                    err = {}
                    for bk in ERR_BLOCKS:
                        e = relerr(outs[bk], dcap[f"out{bk}"][kidx])
                        err[str(bk)] = [float(e[kmask].mean()), float(e[acol].mean())]
                    ctx = info["ctx"]
                    orow["v"][f"{s}_{v}"] = dict(ce=float(loss_v), err=err,
                                                 flops=fl.row(r["T"], kidx, didx, v, ctx),
                                                 n_kept=int(kidx.numel()), n_drop=int(didx.numel()),
                                                 keep_body=float((kp & r["body"]).sum() / max(1, nb)),
                                                 n_standin=int(ctx["S"]) if ctx is not None and "S" in ctx else 0,
                                                 n_ctx=int(ctx["cidx"].numel()) if ctx is not None and "cidx" in ctx else 0)
                    del outs
            del dcap
            out_rows.append(orow)
            n_done += 1
            if n_done in (1, 2, 5, 10) or n_done % 8 == 0:
                el = time.time() - t0
                vv = orow["v"]
                log(f"[probe] row {n_done}/{tot_rows} ({t}, T={r['T']}) CE dense {orow['ce_dense']:.3f} | rtr: " +
                    " ".join(f"{v} {vv[f'rtr_{v}']['ce']:.3f}" for v in variants if f"rtr_{v}" in vv) +
                    f" | {el:.0f}s, ETA {el / n_done * (tot_rows - n_done):.0f}s; "
                    f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        tap.mode = None
        if save_dir:
            arr = {}
            Dt = to_real(acc_D)[:, :, 0]  # (nT, NG, m)
            arr["tot_D"], arr["tot_D_sc"] = q16(Dt.sum(0))
            arr["tsk_D"], arr["tsk_D_sc"] = q16(fold(Dt, a.fold_dim))
            if adam:
                arr["adam_tot_D"], arr["adam_tot_D_sc"] = q16(to_real(acc_Da)[:, :, 0].sum(0))
            for (s, v) in combos:
                if not bool(acc_R[(s, v)].any()):
                    continue
                Rt = to_real(acc_R[(s, v)])[:, :, 0]
                arr[f"tot_R_{s}_{v}"], arr[f"tot_R_{s}_{v}_sc"] = q16(Rt.sum(0))
                arr[f"tsk_R_{s}_{v}"], arr[f"tsk_R_{s}_{v}_sc"] = q16(fold(Rt, a.fold_dim))
                if adam:
                    arr[f"adam_tot_R_{s}_{v}"], arr[f"adam_tot_R_{s}_{v}_sc"] = q16(to_real(acc_Ra[(s, v)])[:, :, 0].sum(0))
            np.savez(os.path.join(save_dir, f"b{J:03d}.npz"), **arr)
            log(f"saved {save_dir}/b{J:03d}.npz ({sum(x.nbytes for x in arr.values()) / 2**20:.0f} MiB)")
        for A in [acc_D, acc_Da, *acc_R.values(), *acc_Ra.values()]:
            if A is not None:
                A.zero_()
        del R
        meta = dict(label=a.label, ckpt=a.ckpt, shard=a.shard, num_shards=a.num_shards, batches=mine[:bj + 1],
                    tasks=tasks, selectors=SELECTORS, variants=variants, cut=CUT, keep=KEEP, refresh=REFRESH,
                    corr_at=CORR_AT, span=SPAN, err_blocks=ERR_BLOCKS, sketch_dim=a.sketch_dim, fold_dim=a.fold_dim,
                    adam=bool(adam), adam_factors=a.adam_factors, lora_merged=merged, corr_diag=corr_diag,
                    fit_rows_per_task=a.fit_rows, rows=out_rows, seconds=time.time() - t00)
        json.dump(meta, open(a.out, "w"))
        if save_dir:
            json.dump(meta, open(os.path.join(save_dir, f"meta_shard{a.shard}.json"), "w"))
        log(f"batch {J} done ({bj + 1}/{len(mine)}), {time.time() - t0:.0f}s")
    log(f"done {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
