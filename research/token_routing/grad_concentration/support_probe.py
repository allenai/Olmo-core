"""High-gradient vs supporting tokens in the TRUE routed gradient (Prasann, 2026-11-02; experiment tokdrop_lr).

routed_probe.py showed that keeping the oracle top-gradient tokens past a split and removing the rest from the forward
keeps only a small part of the dense gradient, and that >95% of the loss is the KEPT tokens' own terms changing once
their context is gone. This probe asks which tokens a selection needs, and whether a removal score (the keep-gate
gradient) finds them. No training; frozen checkpoints, the same 128 unseen fresh-pool rows per task, 16-row batches
(one optimizer step each).

Scores (dense pass, per row):
  en     each body token's own rank-1 weight-gradient energy summed over blocks >= L (probe.py's down<L>)
  g05    |dCE/dg| with every body token's keep gate at g = 0.5 in blocks >= L (soft_keep: attention +log g key bias,
         GDN beta / decay scaled by g, removal-aware conv; grad_importance.py's best removal score, a0.5_abs)
  g1     the same at g = 1 (cheap first-order deletion estimate)
  rnd    uniform
  attn   attention mass a set of query tokens puts on each key (heads + full-attention layers >= L, dense forward)
Allocations (rank fraction r = rank / unit size within the unit; keep r < k): glob (the 16-row batch), row, doc.

Parts (--parts):
  q1     q1_<alloc>_L<L>: the smallest en set (fraction grid GRID) whose ATTRIBUTION cosine (dense terms of the kept
         tokens + every token before the split + every non-body token, vs the dense gradient) reaches 0.95 this step.
  gs     <score>_<alloc>_L<L>_k<k>: fixed keep fraction k (default: en and g05 x 3 allocations, g1 per row, rnd per
         document; splits 0/3/7/16; k 0.5-20%; override with --gs-*).
  joint  per allocation and split: jt_union_* = (smallest g05 set holding 95% of the batch's gate mass) + the
         smallest en prefix that lifts the attribution cosine to 0.95; jt_comb_* = the smallest prefix of the
         combined ranking min(r_en, r_g05) reaching BOTH 95% gate mass and attribution cosine 0.95.
  hh     hh_<rule>_L<L>_k<k> (per row, splits 7/16): enhalf (top-en B/2 only), enattn (B/2 top en + B/2 by attention
         mass from those tokens), engate (B/2 top en + B/2 by g05).
  q3     (stretch) token x layer cells at split 7 through the masked block_keep path.
Also per batch (no extra passes): gate-mass and energy-mass concentration curves over GRID (plain and squared) and
Jaccard(g05 set, en set) at each k, per allocation and split.

Vectors (TensorSketch, routed_probe.Tap hashes, raw + Adam-preconditioned): D, A_s (attribution partial sum), R_s (TRUE
routed gradient: compacted at the split with original RoPE positions; q3 masked). Saved per batch in analyze_gap.py's
format; per-step cos(R, D) / cos(A, D) / norm ratio also logged and written to the JSON (batch_stats[].step).

    python research/token_routing/grad_concentration/support_probe.py --ckpt <dir> --label f1952 \
        --adam-factors <adam_f1952.pt> --num-shards 16 --shard 0 --save-dir <dir> --out <json>
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
import probe as P  # noqa: E402
import routed_probe as RP  # noqa: E402

NB, NG = RP.NB, RP.NG
GRID = (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.014, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1,
        0.14, 0.2, 0.3, 0.4, 0.5, 0.7)
TARGET = 0.95
GATE_MASS = 0.95
SPLITS = (0, 3, 7, 16)
ALLOCS = ("glob", "row", "doc")
GS_KS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2)
GS_DEFAULT = (("en", ALLOCS), ("g05", ALLOCS), ("g1", ("row",)), ("rnd", ("doc",)))
HH_KS = {7: (0.03, 0.1, 0.3), 16: (0.01, 0.03, 0.1)}
HH_RULES = ("enhalf", "enattn", "engate")
Q3_L = 7
Q3_KS = (0.03, 0.1)
Q3_RULES = ("tok_en", "cell_en", "tok_cg", "cell_cg")
CHUNK = RP.CHUNK


def log(m):
    print(f"[support-probe] {m}", flush=True)


def split_of(s):
    return Q3_L if s.startswith("q3_") else int(s.split("_L")[1].split("_")[0])


# --------------------------------------------------------------------------------------------------------- tap
class MTap(RP.Tap):
    """routed_probe.Tap with two cheaper modes. ``multi``: every token's sketch is added to the bins it belongs to via
    one multi-hot matmul per chunk (``memb(block) -> (nbins, T)`` over the row's positions: dense pass only).
    ``sum``: the whole pass is summed into ``accS`` (routed passes)."""

    def _grad(self, i, x, g):
        if self.mode not in ("multi", "sum"):
            return super()._grad(i, x, g)
        self.fired += 1
        ln = self.lins[i]
        x = x.reshape(-1, x.shape[-1])
        g = g.detach().reshape(-1, g.shape[-1])
        adam = self.adam.get(ln.name)
        T = x.shape[0]
        M = self.memb(ln.block) if self.mode == "multi" else None
        torch.backends.cuda.matmul.allow_tf32 = True
        for c0 in range(0, T, CHUNK):
            c1 = min(T, c0 + CHUNK)
            gc, xc = g[c0:c1].float(), x[c0:c1].float()
            Pr = self._ts(i, gc, xc)
            Pa = self._ts(i, gc * adam[0], xc * adam[1]) if adam is not None else None
            if M is not None:
                self.accM[0][ln.group].addmm_(M[:, c0:c1], Pr)
                if Pa is not None:
                    self.accM[1][ln.group].addmm_(M[:, c0:c1], Pa)
            else:
                self.accS[0][ln.group] += Pr.sum(0)
                if Pa is not None:
                    self.accS[1][ln.group] += Pa.sum(0)
        torch.backends.cuda.matmul.allow_tf32 = False


class QKCap:
    """Captures post-RoPE q, k of every full-attention block while ``on`` (wraps ``_prepare_qkv``)."""

    def __init__(self, model):
        self.on, self.store, self.blocks, self.scale = False, {}, [], {}
        for key, block in model.blocks.items():
            att = block.attention
            if type(att).__name__ == "GatedDeltaNet" or not hasattr(att, "_prepare_qkv"):
                continue
            b = int(key)
            orig = att._prepare_qkv

            def wrapped(x, *args, _o=orig, _b=b, **kw):
                q, k, v = _o(x, *args, **kw)
                if self.on:
                    self.store[_b] = (q.detach()[0], k.detach()[0])
                return q, k, v
            att._prepare_qkv = wrapped
            self.blocks.append(b)
            sc = getattr(getattr(att, "backend", None), "scale", None)
            self.scale[b] = sc if sc is not None else att.head_dim ** -0.5

    def mass(self, qidx, L, T):
        """(T,) attention mass the queries ``qidx`` put on every key, summed over heads and full-attention blocks >= L."""
        out = torch.zeros(T, device=qidx.device)
        if qidx.numel() == 0:
            return out
        pos = torch.arange(T, device=qidx.device)
        for b in self.blocks:
            if b < L:
                continue
            q, k = self.store[b]
            rep = q.shape[1] // k.shape[1]
            kk = k.float().repeat_interleave(rep, dim=1)  # (T, H, D)
            for c0 in range(0, qidx.numel(), 256):
                qi = qidx[c0:c0 + 256]
                s = torch.einsum("qhd,thd->hqt", q[qi].float(), kk) * self.scale[b]
                s.masked_fill_(pos[None, None, :] > qi[None, :, None], float("-inf"))
                out += s.softmax(-1).sum((0, 1))
        return out


# ----------------------------------------------------------------------------------------------------- forwards
def math_sdpa():
    """soft_keep attention runs SDPA with a dense additive bias; the memory-efficient kernel's backward fails on these
    shapes ("LSE is not correctly aligned (strideH)"), so gate passes use the math kernel (forward AND the
    checkpoint recompute in backward)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.MATH])


def gate_grad(model, r, L, gval, cell=False):
    """|dCE/dg| per body token (T,) (zeros off body) with every body gate at ``gval`` in blocks >= L; cell=True: per
    (block >= L, body token) gates (NB - L, n_body) that also mix the token's own block update. Returns (score, CE)."""
    bidx = torch.nonzero(r["body"], as_tuple=True)[0]
    shape = (NB - L, bidx.numel()) if cell else (bidx.numel(),)
    z = torch.full(shape, float(gval), device=bidx.device, requires_grad=True)
    with math_sdpa():
        ce = run_soft(model, r, L, z, cell)
        ce.backward()
    if cell:
        return z.grad.detach(), float(ce)
    out = torch.zeros(r["T"], device=bidx.device)
    out[bidx] = z.grad.abs()
    return out, float(ce)


def run_soft(model, r, L, z, cell):
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    x, pred, tgt, T = r["x"], r["pred"], r["tgt"], r["T"]
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    bidx = torch.nonzero(r["body"], as_tuple=True)[0]
    with torch.no_grad():
        h = model.embeddings(x[None])
    with set_checkpoint_early_stop(False):
        for key, block in model.blocks.items():
            b = int(key)
            kw = {**abk, **pbk.get(b, {})}
            if b < L:
                with torch.no_grad():
                    h = block(h, **kw)
                continue
            zz = z[b - L] if cell else z
            sk = torch.ones(T, device=x.device, dtype=torch.float32).index_put((bidx,), zz)

            def fn(hh, s, bl=block, k=kw):
                out = bl(hh, soft_keep=s[None], **k)
                if not cell:
                    return out
                sw = s[None, :, None].to(out.dtype)
                return hh * (1 - sw) + out * sw
            h = checkpoint(fn, h, sk, use_reentrant=False)
    logits = model.lm_head(h, **lmk)[0]
    return F.cross_entropy(logits.float(), tgt, reduction="sum")


def run_masked(model, tap, r, L, keeps):
    """Answer CE through the masked per-(block, token) path: in block b >= L only tokens with ``keeps[b - L]`` run the
    block (attention key mask / GDN no state write via ``block_keep``; absent tokens' block outputs are discarded)."""
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    x, pred, tgt, T = r["x"], r["pred"], r["tgt"], r["T"]
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    h = model.embeddings(x[None]).detach().requires_grad_(True)
    cur = torch.arange(T, device=x.device)
    with set_checkpoint_early_stop(False):
        for key, block in model.blocks.items():
            b = int(key)
            tap.cur[b] = cur
            kw = {**abk, **pbk.get(b, {})}
            if b >= L:
                kp = keeps[b - L]

                def fn(hh, bl=block, k=kw, kp=kp):
                    return torch.where(kp[None, :, None], bl(hh, block_keep=kp[None], **k), hh)
            else:
                def fn(hh, bl=block, k=kw):
                    return bl(hh, **k)
            h = checkpoint(fn, h, use_reentrant=False)
    logits = model.lm_head(h, **lmk)[0]
    return F.cross_entropy(logits.float(), tgt, reduction="sum")


# ---------------------------------------------------------------------------------------------------- selection
def rank_frac(score, grp):
    """rank within group / group size (0 = best) for every element of ``score`` (N,), groups ``grp`` (N,) long."""
    o = torch.argsort(-score.double())
    o = o[torch.argsort(grp[o], stable=True)]
    _, inv, cnt = torch.unique_consecutive(grp[o], return_inverse=True, return_counts=True)
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(o.numel(), device=score.device) - start[inv]
    out = torch.empty(score.numel(), dtype=torch.float64, device=score.device)
    out[o] = rank.double() / cnt[inv].double()
    return out


def topk_mask(score, elig, n):
    m = torch.zeros_like(elig)
    idx = torch.nonzero(elig, as_tuple=True)[0]
    n = min(int(n), idx.numel())
    if n > 0:
        m[idx[torch.topk(score[idx].double(), n).indices]] = True
    return m


def spearman_t(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def first_reaching(vals, thr):
    for j, v in enumerate(vals):
        if v >= thr:
            return j
    return None


# ---------------------------------------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", default="contradiction,nq,outlier,oolong")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--row-offset", type=int, default=7808)
    ap.add_argument("--shards", default=P.SHARDS)
    ap.add_argument("--seen-shards", default="mt4f7808,mt4s1952")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--sketch-dim", type=int, default=8192)
    ap.add_argument("--adam-factors", default="")
    ap.add_argument("--adam-estimate", action="store_true")
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--lora-scale", type=float, default=32.0)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--parts", default="q1,gs,joint,hh", help="subset of q1,gs,joint,hh,q3")
    ap.add_argument("--splits", default=",".join(map(str, SPLITS)), help="q1 / gs / joint splits")
    ap.add_argument("--gs-ks", default=",".join(f"{k:g}" for k in GS_KS))
    ap.add_argument("--gs-rules", default="", help="cross product rules x --gs-allocs (default: GS_DEFAULT)")
    ap.add_argument("--gs-allocs", default="glob,row,doc")
    ap.add_argument("--selfcheck", action="store_true", help="also run an unrouted pass in 'sum' mode (== D)")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    t00 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    parts = set(a.parts.split(","))
    splits = tuple(int(x) for x in a.splits.split(","))
    gs_ks = tuple(float(x) for x in a.gs_ks.split(","))
    gs_spec = (tuple((r_, tuple(a.gs_allocs.split(","))) for r_ in a.gs_rules.split(",")) if a.gs_rules
               else GS_DEFAULT)

    model = P.build_model(P.find_ckpt(a.ckpt), a.attn_backend)
    lins = P.list_linears(model)
    RP.merge_lora(model, lins, P.find_ckpt(a.ckpt), a.lora_scale)
    tasks = a.tasks.split(",")
    seen = [os.path.join(a.shards, s) for s in a.seen_shards.split(",") if s]
    rows_by_task, _ = P.load_rows(a.shards, tasks, a.rows, a.row_offset, 12288, seen)
    per = a.batch // len(tasks)
    nbatch = min(len(v) for v in rows_by_task.values()) // per
    adam, adam_diag = None, {}
    if a.adam_factors:
        blob = torch.load(a.adam_factors)
        adam, adam_diag = blob["factors"], blob["diag"]
    elif a.adam_estimate:
        bs = [[RP.prep_row(rr, t) for t in tasks for rr in rows_by_task[t][J * per:(J + 1) * per]] for J in range(nbatch)]
        adam, adam_diag = RP.estimate_adam_factors(model, lins, bs)
        del bs
    if adam is not None:
        adam = {k: (u.cuda().float(), w.cuda().float()) for k, (u, w) in adam.items()}
    sp = 2 if adam is not None else 1
    mine = [j for j in range(nbatch) if j % a.num_shards == a.shard]
    if a.max_batches:
        mine = mine[:a.max_batches]
    log(f"{nbatch} batches of {a.batch} rows; shard {a.shard}/{a.num_shards} runs {mine}; adam={adam is not None}")

    tap = MTap(lins, a.sketch_dim, adam=adam, exact=())
    tap.ckpt = True
    qk = QKCap(model)
    log(f"full-attention blocks: {qk.blocks}")
    nm = a.sketch_dim // 2 + 1
    dev = torch.device("cuda")

    # ------------------------------------------------------------------------------------ selection names
    s_gs = ([f"{rule}_{al}_L{L}_k{k:g}" for L in splits for k in gs_ks for rule, als in gs_spec for al in als]
            if "gs" in parts else [])
    s_hh = ([f"hh_{rule}_L{L}_k{k:g}" for L, ks in HH_KS.items() for k in ks for rule in HH_RULES]
            if "hh" in parts else [])
    s_q3 = [f"q3_{rule}_k{k:g}" for k in Q3_KS for rule in Q3_RULES] if "q3" in parts else []
    named = s_gs + s_hh + s_q3  # known before the dense sketch pass
    grid_kinds = (["en"] if ("q1" in parts or "joint" in parts) else []) + (["comb", "union"] if "joint" in parts else [])
    grid_keys = [(kind, al, L) for L in splits for al in ALLOCS for kind in grid_kinds]
    s_grid = ([f"q1_{al}_L{L}" for L in splits for al in ALLOCS] if "q1" in parts else []) + \
             ([f"jt_{kind}_{al}_L{L}" for L in splits for al in ALLOCS for kind in ("union", "comb")]
              if "joint" in parts else [])
    allsels = s_grid + named
    ng = len(GRID)
    nbins = 1 + len(grid_keys) * ng + len(named)
    # scores needed: (score, L) pairs
    gate_need = set()
    for s in s_gs:
        rule, L = s.split("_")[0], split_of(s)
        if rule in ("g05", "g1"):
            gate_need.add((rule, L))
    if "joint" in parts:
        gate_need |= {("g05", L) for L in splits}
    if s_hh:
        gate_need |= {("g05", L) for L in HH_KS}
    log(f"selections: {len(s_grid)} grid-chosen + {len(s_gs)} gs + {len(s_hh)} hh + {len(s_q3)} q3 = {len(allsels)}; "
        f"gate passes/row {len(gate_need)}; dense bins {nbins}; accumulators "
        f"{(sp * NG * nbins + sp * NG * len(allsels)) * 2 * nm * 4 / 2**30:.1f} GiB")

    def to_real(A):
        sh = A.shape[:-1]
        return torch.fft.irfft(torch.view_as_complex(A.reshape(-1, nm, 2).contiguous()), n=a.sketch_dim,
                               dim=1).reshape(*sh, a.sketch_dim)

    def q16(S):
        sc = S.abs().amax(-1, keepdim=True).clamp(min=1e-30)
        return (S / sc).half().cpu().numpy(), sc[..., 0].float().cpu().numpy()

    def cosv(x_, y_):
        x_, y_ = x_.double(), y_.double()
        return float((x_ * y_).sum() / (x_.norm() * y_.norm()).clamp(min=1e-300))

    save_dir = os.path.join(a.save_dir, a.label) if a.save_dir else ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    out_rows, out_batches = [], []
    t0 = time.time()
    n_done = 0
    grid_t = torch.tensor(GRID, dtype=torch.float64, device=dev)
    for bj, J in enumerate(mine):
        batch = [(t, rr) for t in tasks for rr in rows_by_task[t][J * per:(J + 1) * per]]
        R = [RP.prep_row(rr, t) for t, rr in batch]
        gen = torch.Generator(device="cuda").manual_seed(4321 + J)
        # ------------------------------------------------------------ pass 0: dense energies, q/k, gate gradients
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            tb = time.time()
            tap.mode = "energy"
            tap.E = torch.zeros(NB, r["T"], device=dev)
            tap.fired = 0
            qk.on = bool(s_hh)
            loss, _, _ = RP.run_row(model, tap, r, ckpt=True)
            tap.backward(loss)
            qk.on = False
            if tap.fired != len(lins):
                raise SystemExit(f"hooks fired {tap.fired} for {len(lins)} linears")
            r["E"] = tap.E
            tap.E = None
            tap.mode = None
            r["ce_dense"] = float(loss.detach())
            body = r["body"]
            nb_ = int(body.sum())
            r["sc"] = {("en", L): r["E"][L:].sum(0) for L in set(splits) | set(HH_KS) | {Q3_L}}
            r["ce_gate"] = {}
            for rule, L in sorted(gate_need):
                r["sc"][(rule, L)], r["ce_gate"][f"{rule}_L{L}"] = gate_grad(model, r, L, 0.5 if rule == "g05" else 1.0)
            for L in set(splits) | set(HH_KS):
                r["sc"][("rnd", L)] = torch.rand(r["T"], device=dev, generator=gen)
            # half + half rules (per row, need this row's q/k)
            K = {}
            for s in s_hh:
                rule, L, k = s.split("_")[1], split_of(s), float(s.split("_k")[1])
                kr = max(1, math.ceil(k * nb_))
                ka = kr // 2
                m_half = topk_mask(r["sc"][("en", L)], body, ka)
                rest = body & ~m_half
                if rule == "enhalf":
                    K[s] = m_half
                elif rule == "enattn":
                    mass = qk.mass(torch.nonzero(m_half, as_tuple=True)[0], L, r["T"])
                    K[s] = m_half | topk_mask(mass, rest, kr - ka)
                else:
                    K[s] = m_half | topk_mask(r["sc"][("g05", L)], rest, kr - ka)
            qk.store.clear()
            # token x layer cells (stretch)
            if s_q3:
                bidx = torch.nonzero(body, as_tuple=True)[0]
                gc, r["ce_gate"]["cell"] = gate_grad(model, r, Q3_L, 0.5, cell=True)
                Eb = r["E"][Q3_L:][:, bidx]
                tok_cg = torch.zeros(r["T"], device=dev)
                tok_cg[bidx] = gc.sum(0).abs()
                nonb = ~body
                for k in Q3_KS:
                    kr = max(1, math.ceil(k * nb_))
                    ncell = kr * (NB - Q3_L)

                    def cells(sc):
                        flat = torch.zeros(sc.numel(), dtype=torch.bool, device=dev)
                        flat[torch.topk(sc.reshape(-1).double(), min(ncell, sc.numel())).indices] = True
                        kp = torch.zeros(NB - Q3_L, r["T"], dtype=torch.bool, device=dev)
                        kp[:, bidx] = flat.view(sc.shape)
                        return kp | nonb[None]
                    tok = lambda m: (m | nonb)[None].expand(NB - Q3_L, -1).contiguous()  # noqa: E731
                    K[f"q3_tok_en_k{k:g}"] = tok(topk_mask(r["sc"][("en", Q3_L)], body, kr))
                    K[f"q3_cell_en_k{k:g}"] = cells(Eb)
                    K[f"q3_tok_cg_k{k:g}"] = tok(topk_mask(tok_cg, body, kr))
                    K[f"q3_cell_cg_k{k:g}"] = cells(gc.abs())
            r["K"] = K
            n = n_done + i + 1
            if n in (1, 2, 5, 10) or n % 16 == 0:
                log(f"[p0] batch {J} row {i} {t} T={r['T']} body={nb_} CE={r['ce_dense']:.3f} gate CE "
                    + " ".join(f"{k_}={v_:.2f}" for k_, v_ in r["ce_gate"].items())
                    + f" {time.time() - tb:.0f}s/row, {time.time() - t0:.0f}s")
        # ------------------------------------------------------------ batch-level rank fractions
        bodies = [r["body"] for r in R]
        nbody = sum(int(b.sum()) for b in bodies)

        def rf_of(score, al, L, _c={}):
            key = (J, score, al, L)
            if key not in _c:
                sc = torch.cat([r["sc"][(score, L)][b] for r, b in zip(R, bodies)])
                if al == "glob":
                    grp = torch.zeros(sc.numel(), dtype=torch.long, device=dev)
                elif al == "row":
                    grp = torch.cat([torch.full((int(b.sum()),), i, dtype=torch.long, device=dev)
                                     for i, b in enumerate(bodies)])
                else:
                    grp = torch.cat([i * 100000 + r["cid"][b] for i, (r, b) in enumerate(zip(R, bodies))])
                rf = rank_frac(sc, grp)
                lst, o = [], 0
                for r, b in zip(R, bodies):
                    v = torch.full((r["T"],), float("inf"), dtype=torch.float64, device=dev)
                    v[b] = rf[o:o + int(b.sum())]
                    o += int(b.sum())
                    lst.append(v)
                _c.clear() if len(_c) > 400 else None
                _c[key] = lst
            return _c[key]

        bres = {"batch": J, "grid": {}, "conc": {}, "jaccard": {}, "step": {}}
        # concentration curves (no passes): share of the batch's score mass held by the top-f selection
        conc_scores = ["en"] + (["g05"] if any(rule == "g05" for rule, _ in gate_need) else []) + \
                      (["g1"] if any(rule == "g1" for rule, _ in gate_need) else [])
        for L in splits:
            for scn in conc_scores:
                if (scn, L) not in R[0]["sc"]:
                    continue
                for al in ALLOCS:
                    rfl = rf_of(scn, al, L)
                    vals = torch.cat([r["sc"][(scn, L)][b].double() for r, b in zip(R, bodies)])
                    rr_ = torch.cat([v[b] for v, b in zip(rfl, bodies)])
                    plain = [float(vals[rr_ < f].sum() / vals.sum()) for f in GRID]
                    sq = [float(vals[rr_ < f].pow(2).sum() / vals.pow(2).sum()) for f in GRID]
                    bres["conc"][f"{scn}_{al}_L{L}"] = dict(plain=plain, sq=sq)
        # Jaccard(g05 set, en set) per allocation / split / k
        for L in splits:
            if ("g05", L) not in R[0]["sc"]:
                continue
            for al in ALLOCS:
                re_, rg_ = rf_of("en", al, L), rf_of("g05", al, L)
                jac = []
                for k in gs_ks:
                    inter = sum(int(((e < k) & (g < k)).sum()) for e, g in zip(re_, rg_))
                    uni = sum(int(((e < k) | (g < k)).sum()) for e, g in zip(re_, rg_))
                    jac.append(inter / max(1, uni))
                bres["jaccard"][f"{al}_L{L}"] = jac
        # gs keep masks
        for s in s_gs:
            rule, al = s.split("_")[0], s.split("_")[1]
            L, k = split_of(s), float(s.split("_k")[1])
            rfl = rf_of(rule, al, L)
            for i, r in enumerate(R):
                r["K"][s] = (rfl[i] < k) & r["body"]
        # joint: gate-95 set per (alloc, L): smallest grid fraction whose g05 selection holds >= 95% of gate mass
        g95 = {}
        if "joint" in parts:
            for L in splits:
                for al in ALLOCS:
                    cur = bres["conc"][f"g05_{al}_L{L}"]["plain"]
                    j = first_reaching(cur, GATE_MASS)
                    f = GRID[j] if j is not None else 1.0
                    g95[(al, L)] = f
        # grid rows: per (kind, al, L) -> list over rows of (T,) rank values (keep r < f)
        def grid_rank(kind, al, L):
            re_ = rf_of("en", al, L)
            if kind == "en":
                return re_
            rg_ = rf_of("g05", al, L)
            if kind == "comb":
                return [torch.minimum(e, g) for e, g in zip(re_, rg_)]
            f = g95[(al, L)]  # union: the gate-95 set always in, then the en prefix
            return [torch.where(g < f, torch.zeros_like(e), e) for e, g in zip(re_, rg_)]
        GR = {key: grid_rank(*key) for key in grid_keys}
        # ------------------------------------------------------------ pass 1: dense sketch, multi-hot bins
        tap.accM = [torch.zeros(NG, nbins, 2 * nm, device=dev) for _ in range(sp)]
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            body = r["body"]
            T = r["T"]
            cache = {}
            rows_tok = {key: (GR[key][i][None, :] < grid_t[:, None]) | ~body[None] for key in grid_keys}
            named_k = [(split_of(s), r["K"][s] if s.startswith("q3_") else (r["K"][s] | ~body)) for s in named]

            def memb(bk, cache=cache, rows_tok=rows_tok, named_k=named_k, T=T):
                if bk in cache:
                    return cache[bk]
                M = torch.ones(nbins, T, device=dev)
                o = 1
                for key in grid_keys:
                    if bk >= key[2]:
                        M[o:o + ng] = rows_tok[key].float()
                    o += ng
                for L, kp in named_k:
                    if bk >= L:
                        M[o] = (kp[bk - L] if kp.dim() == 2 else kp).float()
                    o += 1
                cache[bk] = M
                return M
            tap.memb = memb
            tap.mode = "multi"
            tap.fired = 0
            loss, _, _ = RP.run_row(model, tap, r, ckpt=True)
            tap.backward(loss)
            tap.mode = None
            del cache, rows_tok
        # ------------------------------------------------------------ grid-chosen budgets
        SD = [to_real(acc[:, 0]) for acc in tap.accM]
        A_vec = {}
        o = 1
        for kind, al, L in grid_keys:
            C = [to_real(tap.accM[spi][:, o:o + ng]) for spi in range(sp)]  # (NG, ng, m)
            curves = []
            for spi in range(sp):
                ip = torch.einsum("gjm,gm->j", C[spi].double(), SD[spi].double())
                nn_ = C[spi].double().pow(2).sum((0, 2))
                curves.append((ip / torch.sqrt(nn_ * SD[spi].double().pow(2).sum())).tolist())
            if kind == "en":
                ok = [c >= TARGET for c in curves[0]]
                name = f"q1_{al}_L{L}"
            else:
                gm = []
                rg_ = rf_of("g05", al, L)
                gv = torch.cat([r["sc"][("g05", L)][b].double() for r, b in zip(R, bodies)])
                rk = torch.cat([v[b] for v, b in zip(GR[(kind, al, L)], bodies)])
                gm = [float(gv[rk < f].sum() / gv.sum()) for f in GRID]
                ok = [c >= TARGET and m_ >= GATE_MASS for c, m_ in zip(curves[0], gm)]
                name = f"jt_{kind}_{al}_L{L}"
            js = first_reaching([1.0 if x_ else 0.0 for x_ in ok], 0.5)
            fstar = GRID[js] if js is not None else 1.0
            rk_rows = GR[(kind, al, L)]
            kept = sum(int(((rk_rows[i] < fstar) & R[i]["body"]).sum()) for i in range(len(R)))
            bres["grid"][f"{kind}_{al}_L{L}"] = dict(
                fstar=fstar, keep_frac=kept / max(1, nbody), attr_cos=curves[0][js] if js is not None else 1.0,
                curve_raw=curves[0], curve_adam=curves[1] if sp > 1 else None,
                g95_frac=g95.get((al, L)), gate_mass=(gm if kind != "en" else None))
            if name in s_grid:
                A_vec[name] = [(C[spi][:, js].clone() if js is not None else SD[spi].clone()) for spi in range(sp)]
                for i, r in enumerate(R):
                    r["K"][name] = (rk_rows[i] < fstar) & r["body"]
            del C
            o += ng
        for s in named:
            A_vec[s] = [to_real(tap.accM[spi][:, o]) for spi in range(sp)]
            o += 1
        del tap.accM
        log(f"batch {J} grid budgets: " + ", ".join(f"{k}: k={v['keep_frac']:.4f} attr {v['attr_cos']:.3f}"
                                                    for k, v in bres["grid"].items()))
        # ------------------------------------------------------------ routed passes
        accR = {s: torch.zeros(sp, NG, 2 * nm, device=dev) for s in allsels}
        if a.selfcheck:
            accR["_dense"] = torch.zeros(sp, NG, 2 * nm, device=dev)
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            body = r["body"]
            nb_ = int(body.sum())
            orow = dict(batch=J, task=t, k=rr["k"], md5=rr["md5"], T=r["T"], n_body=nb_, n_answer=int(r["pred"].numel()),
                        gold_docs=r["gold_docs"], numbered_ok=r["numbered_ok"], ce_dense=r["ce_dense"],
                        ce_gate=r["ce_gate"], ce_routed={}, keep={}, energy_share={}, gold_share={},
                        spearman_gate_en={})
            for (rule, L) in sorted(gate_need):
                orow["spearman_gate_en"][f"{rule}_L{L}"] = spearman_t(r["sc"][(rule, L)][body], r["sc"][("en", L)][body])
            gold = r["ty"] == 2
            for s in allsels + (["_dense"] if a.selfcheck else []):
                tap.accS = [accR[s][spi] for spi in range(sp)]
                tap.mode = "sum"
                tap.fired = 0
                if s == "_dense":
                    loss, _, _ = RP.run_row(model, tap, r, ckpt=True)
                elif s.startswith("q3_"):
                    loss = run_masked(model, tap, r, Q3_L, r["K"][s])
                else:
                    kp = r["K"][s]
                    loss, _, _ = RP.run_row(model, tap, r, stages={split_of(s): (
                        lambda h, cur, kp=kp, body=body: (kp | ~body)[cur])}, ckpt=True)
                tap.backward(loss)
                tap.mode = None
                if tap.fired != len(lins):
                    raise SystemExit(f"{s}: hooks fired {tap.fired} for {len(lins)} linears")
                orow["ce_routed"][s] = float(loss.detach())
                if s == "_dense":
                    continue
                if s.startswith("q3_"):
                    kp = r["K"][s][:, body]
                    orow["keep"][s] = float(kp.float().mean())
                    Eb = r["E"][Q3_L:][:, body]
                    orow["energy_share"][s] = float((Eb * kp).sum() / Eb.sum().clamp(min=1e-30))
                    orow["gold_share"][s] = float(kp[:, gold[body]].sum() / kp.sum().clamp(min=1))
                    continue
                kb = r["K"][s] & body
                L = split_of(s)
                en = r["sc"][("en", L)]
                orow["keep"][s] = float(kb.sum() / max(1, nb_))
                orow["energy_share"][s] = float(en[kb].sum() / en[body].sum().clamp(min=1e-30))
                orow["gold_share"][s] = float((kb & gold).sum() / max(1, int(kb.sum())))
            out_rows.append(orow)
            n_done += 1
            if n_done in (1, 2, 5, 10) or n_done % 8 == 0:
                el = time.time() - t0
                tot = len(mine) * a.batch
                show = [s for s in allsels if s.endswith("L7_k0.07") or s.startswith("q1_glob_L7")][:6]
                log(f"[routed] row {n_done}/{tot} ({t}) CE dense {r['ce_dense']:.3f} "
                    + " ".join(f"{s} {orow['ce_routed'][s]:.3f}" for s in show)
                    + f"; {el:.0f}s, ETA {el / n_done * (tot - n_done):.0f}s; GPU peak "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            for kk in ("E", "K", "sc"):
                r.pop(kk, None)
        # ------------------------------------------------------------ per-step cosines, save
        SR = {s: [to_real(accR[s][spi]) for spi in range(sp)] for s in accR}
        if a.selfcheck:
            for spi in range(sp):
                bres[f"selfcheck_{spi}"] = cosv(SR["_dense"][spi], SD[spi])
                log(f"selfcheck space {spi}: cos(D_sum, D_multi) = {bres[f'selfcheck_{spi}']:.6f}")
        nrow = {s: [r_ for r_ in out_rows if r_["batch"] == J] for s in ("x",)}["x"]
        for s in allsels:
            st = {}
            for spi, sn in enumerate(("raw", "adam")[:sp]):
                st[sn] = dict(routed_cos=cosv(SR[s][spi], SD[spi]), attr_cos=cosv(A_vec[s][spi], SD[spi]),
                              routed_vs_attr=cosv(SR[s][spi], A_vec[s][spi]),
                              norm_ratio=float(SR[s][spi].double().norm() / SD[spi].double().norm()))
            st["keep"] = float(np.mean([r_["keep"][s] for r_ in nrow]))
            st["ce_routed"] = float(np.sum([r_["ce_routed"][s] for r_ in nrow]) / np.sum([r_["n_answer"] for r_ in nrow]))
            bres["step"][s] = st
        bres["ce_dense"] = float(np.sum([r_["ce_dense"] for r_ in nrow]) / np.sum([r_["n_answer"] for r_ in nrow]))
        log(f"batch {J} per-step (dense answer CE {bres['ce_dense']:.3f}): sel | keep | cos(A,D) | cos(R,D) raw | "
            "cos(R,D) adam | |R|/|D| | routed CE")
        for s in allsels:
            st = bres["step"][s]
            log(f"  {s:24s} {st['keep']:.4f} | {st['raw']['attr_cos']:.3f} | {st['raw']['routed_cos']:.3f} | "
                f"{st.get('adam', {}).get('routed_cos', float('nan')):.3f} | {st['raw']['norm_ratio']:.2f} | "
                f"{st['ce_routed']:.3f}")
        if save_dir:
            arr = {}
            arr["tot_D"], arr["tot_D_sc"] = q16(SD[0])
            if sp > 1:
                arr["adam_tot_D"], arr["adam_tot_D_sc"] = q16(SD[1])
            for s in allsels:
                for spi, pre in enumerate(("tot_", "adam_tot_")[:sp]):
                    arr[f"{pre}A_{s}"], arr[f"{pre}A_{s}_sc"] = q16(A_vec[s][spi])
                    arr[f"{pre}R_{s}"], arr[f"{pre}R_{s}_sc"] = q16(SR[s][spi])
            np.savez(os.path.join(save_dir, f"b{J:03d}.npz"), **arr)
            log(f"saved {save_dir}/b{J:03d}.npz ({sum(v.nbytes for v in arr.values()) / 2**20:.0f} MiB)")
        out_batches.append(bres)
        del SR, SD, A_vec, accR, R
        log(f"batch {J} done ({bj + 1}/{len(mine)}), {time.time() - t0:.0f}s")
        meta = dict(label=a.label, ckpt=a.ckpt, shard=a.shard, num_shards=a.num_shards, batches=mine[:bj + 1],
                    tasks=tasks, sels=allsels, s_grid=s_grid, s_gs=s_gs, s_hh=s_hh, s_q3=s_q3, splits=splits,
                    gs_ks=gs_ks, grid=GRID, target=TARGET, gate_mass=GATE_MASS, hh_ks=HH_KS, q3_ks=Q3_KS,
                    sketch_dim=a.sketch_dim, adam=adam is not None, adam_source=(adam_diag or {}).get("_source"),
                    batch_stats=out_batches, rows=out_rows, seconds=time.time() - t00)
        json.dump(meta, open(a.out, "w"))
        if save_dir:
            json.dump(meta, open(os.path.join(save_dir, f"meta_shard{a.shard}.json"), "w"))
    log(f"done {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
