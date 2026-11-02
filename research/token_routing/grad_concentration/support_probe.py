"""High-gradient vs supporting tokens in the TRUE routed gradient (Prasann, 2026-11-02; experiment tokdrop_lr).

routed_probe.py showed that keeping the oracle top-gradient tokens past a split and removing the rest from the forward
keeps only a small part of the dense gradient, and that >95% of the loss is the KEPT tokens' own terms changing once
their context is gone. This probe asks which context they need. No training; frozen checkpoints, the same 128 unseen
fresh-pool rows per task, 16-row batches (one optimizer step each).

Scores (dense pass, per row):
  energy  each body token's own rank-1 gradient energy summed over blocks >= L (probe.py's down<L>; E[b, t] per block)
  gate    |dCE/dg| with every body token's keep gate at g = 0.5 in blocks >= L (soft_keep: attention +log g key bias,
          GDN beta / decay scaled by g, removal-aware conv; grad_importance.py's best removal score, a0.5_abs)
  attn    attention mass a set of query tokens puts on each key, summed over heads and the full-attention layers >= L
          (dense forward, causal softmax)
  cgate   cell gates: one gate per (block >= L, token) that scales both its key / state-write presence (soft_keep) and
          its own residual update; |dCE/dg_bt| at g = 0.5

Selections (keep = these body tokens; prompt / query / answer / markers are always kept):
  Q1 q1_<alloc>_L<L>   the smallest energy set whose ATTRIBUTION cosine (dense terms of the kept tokens + every token
                       before the split + every non-body token, vs the dense gradient) reaches 0.95 on this step;
                       alloc glob (batch-wide ranking) / row (each row the same fraction) / doc (each document the same
                       fraction); L in 0, 3, 7, 16. The budget is searched on a fraction grid per step (GRID).
  Q2 <rule>_L<L>_k<B>  fixed per-row budget B at split L: en (top energy), enhalf (top energy, B/2 only), gate (top
                       gate score), enattn (B/2 top energy + B/2 by attention mass FROM those tokens), engate (B/2 top
                       energy + B/2 by gate score), rnd (uniform)
  Q3 q3_<rule>_k<B>    split 7, cell budget B x n_body x 25 blocks, per row, through the MASKED path (block_keep: an
                       absent cell is neither a key nor a state write and its own block update is discarded): tok_en
                       (token-level, energy), cell_en (top cells by per-block energy E[b, t]), tok_cg (token-level,
                       |sum_b dCE/dg_bt|), cell_cg (top cells by |dCE/dg_bt|)

Vectors (TensorSketch, routed_probe.Tap hashes, raw + Adam-preconditioned copies): D (dense), A_s (attribution partial
sum of the kept tokens), R_s (TRUE routed gradient: Q1/Q2 compacted at the split with original RoPE positions, Q3
masked). Saved per batch to --save-dir/<label>/b<j>.npz in analyze_gap.py's format (tot_*, adam_tot_*);
analyze_support.py computes per-step / expected cosines, norm ratios.

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
Q1_SPLITS = (0, 3, 7, 16)
ALLOCS = ("glob", "row", "doc")
Q2_KS = {7: (0.01, 0.03, 0.1, 0.3), 16: (0.01, 0.03, 0.1)}
Q2_RULES = ("en", "enhalf", "gate", "enattn", "engate", "rnd")
Q3_L = 7
Q3_KS = (0.03, 0.1)
Q3_RULES = ("tok_en", "cell_en", "tok_cg", "cell_cg")
GATE = 0.5
CHUNK = RP.CHUNK


def log(m):
    print(f"[support-probe] {m}", flush=True)


def q1_names():
    return [f"q1_{a}_L{L}" for L in Q1_SPLITS for a in ALLOCS]


def q2_names():
    return [f"{s}_L{L}_k{k:g}" for L, ks in Q2_KS.items() for k in ks for s in Q2_RULES]


def q3_names():
    return [f"q3_{s}_k{k:g}" for k in Q3_KS for s in Q3_RULES]


# --------------------------------------------------------------------------------------------------------- tap
class MTap(RP.Tap):
    """routed_probe.Tap with two cheaper modes. ``multi``: every token's sketch is added to the bins it belongs to via
    one multi-hot matmul per chunk (membership ``memb(block) -> (nbins, T)`` float over original positions, which here
    are the row's positions: dense pass only). ``sum``: the whole pass is summed into ``accS`` (routed passes)."""

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
def run_soft(model, r, L, z, cell):
    """Answer CE (summed) with body keep gates in blocks >= L: token gates ``z`` (n_body,) shared by every block, or
    cell gates ``z`` (NB - L, n_body). Gates are soft_keep (key / state-write presence); cell gates also mix the
    token's own block update (h' = (1 - g) h + g block(h))."""
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
    ap.add_argument("--parts", default="q1,q2,q3", help="debug: subset of q1,q2,q3")
    ap.add_argument("--selfcheck", action="store_true", help="also run an unrouted pass in 'sum' mode (== D)")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    t00 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    parts = set(a.parts.split(","))

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
    gen = torch.Generator(device="cpu").manual_seed(4321 + a.shard)

    s_q1 = q1_names() if "q1" in parts else []
    s_q2 = q2_names() if "q2" in parts else []
    s_q3 = q3_names() if "q3" in parts else []
    named = s_q2 + s_q3  # selections known before the dense sketch pass
    allsels = s_q1 + named
    ng = len(GRID)
    q1_keys = [(al, L) for L in Q1_SPLITS for al in ALLOCS] if s_q1 else []
    nbins = 1 + len(q1_keys) * ng + len(named)  # [D] + grid rows + named kept rows
    log(f"selections: {len(s_q1)} q1 + {len(s_q2)} q2 + {len(s_q3)} q3; dense bins {nbins}; "
        f"accumulators {(sp * NG * nbins + sp * NG * len(allsels)) * 2 * nm * 4 / 2**30:.1f} GiB")

    def to_real(A):
        sh = A.shape[:-1]
        return torch.fft.irfft(torch.view_as_complex(A.reshape(-1, nm, 2).contiguous()), n=a.sketch_dim,
                               dim=1).reshape(*sh, a.sketch_dim)

    def q16(S):
        sc = S.abs().amax(-1, keepdim=True).clamp(min=1e-30)
        return (S / sc).half().cpu().numpy(), sc[..., 0].float().cpu().numpy()

    save_dir = os.path.join(a.save_dir, a.label) if a.save_dir else ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    out_rows, out_batches = [], []
    t0 = time.time()
    n_done = 0
    for bj, J in enumerate(mine):
        batch = [(t, rr) for t in tasks for rr in rows_by_task[t][J * per:(J + 1) * per]]
        R = [RP.prep_row(rr, t) for t, rr in batch]
        # ------------------------------------------------------------ pass 0: dense energies, attention q/k, gates
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            tb = time.time()
            tap.mode = "energy"
            tap.E = torch.zeros(NB, r["T"], device=dev)
            tap.fired = 0
            qk.on = bool(s_q2)
            loss, _, _ = RP.run_row(model, tap, r, ckpt=True)
            tap.backward(loss)
            qk.on = False
            if tap.fired != len(lins):
                raise SystemExit(f"hooks fired {tap.fired} for {len(lins)} linears")
            r["E"] = tap.E
            tap.E = None
            tap.mode = None
            r["ce_dense"] = float(loss)
            body = r["body"]
            nb_ = int(body.sum())
            r["down"] = {L: r["E"][L:].sum(0) for L in set(Q1_SPLITS) | set(Q2_KS) | {Q3_L}}
            # Q2 selections that need this row's q/k (attention mass from the top-energy tokens)
            K2 = {}
            if s_q2:
                for L, ks in Q2_KS.items():
                    bidx = torch.nonzero(body, as_tuple=True)[0]
                    z = torch.full((bidx.numel(),), GATE, device=dev, requires_grad=True)
                    ce_g = run_soft(model, r, L, z, cell=False)
                    ce_g.backward()
                    gs = torch.zeros(r["T"], device=dev)
                    gs[bidx] = z.grad.abs()
                    r[f"gate{L}"] = gs
                    r[f"ce_gate{L}"] = float(ce_g)
                    en = r["down"][L]
                    for k in ks:
                        kr = max(1, int(round(k * nb_)))
                        ka = kr // 2
                        m_en = topk_mask(en, body, kr)
                        m_half = topk_mask(en, body, ka)
                        rest = body & ~m_half
                        mass = qk.mass(torch.nonzero(m_half, as_tuple=True)[0], L, r["T"])
                        K2[f"en_L{L}_k{k:g}"] = m_en
                        K2[f"enhalf_L{L}_k{k:g}"] = m_half
                        K2[f"gate_L{L}_k{k:g}"] = topk_mask(gs, body, kr)
                        K2[f"enattn_L{L}_k{k:g}"] = m_half | topk_mask(mass, rest, kr - ka)
                        K2[f"engate_L{L}_k{k:g}"] = m_half | topk_mask(gs, rest, kr - ka)
                        K2[f"rnd_L{L}_k{k:g}"] = topk_mask(torch.rand(r["T"], generator=None, device=dev), body, kr)
            qk.store.clear()
            r["K2"] = K2
            # Q3 cell gates at split 7
            K3 = {}
            if s_q3:
                bidx = torch.nonzero(body, as_tuple=True)[0]
                z = torch.full((NB - Q3_L, bidx.numel()), GATE, device=dev, requires_grad=True)
                ce_c = run_soft(model, r, Q3_L, z, cell=True)
                ce_c.backward()
                gc = z.grad.detach()  # (NB - L, n_body)
                r["ce_cgate"] = float(ce_c)
                Eb = r["E"][Q3_L:][:, bidx]  # (NB - L, n_body)
                tok_cg = torch.zeros(r["T"], device=dev)
                tok_cg[bidx] = gc.sum(0).abs()
                r["tok_cg"] = tok_cg
                nonb = ~body
                for k in Q3_KS:
                    kr = max(1, int(round(k * nb_)))
                    ncell = kr * (NB - Q3_L)

                    def cells(sc):
                        flat = torch.zeros(sc.numel(), dtype=torch.bool, device=dev)
                        flat[torch.topk(sc.reshape(-1).double(), min(ncell, sc.numel())).indices] = True
                        kp = torch.zeros(NB - Q3_L, r["T"], dtype=torch.bool, device=dev)
                        kp[:, bidx] = flat.view(sc.shape)
                        return kp | nonb[None]
                    tok = lambda m: (m | nonb)[None].expand(NB - Q3_L, -1).contiguous()  # noqa: E731
                    K3[f"q3_tok_en_k{k:g}"] = tok(topk_mask(r["down"][Q3_L], body, kr))
                    K3[f"q3_cell_en_k{k:g}"] = cells(Eb)
                    K3[f"q3_tok_cg_k{k:g}"] = tok(topk_mask(tok_cg, body, kr))
                    K3[f"q3_cell_cg_k{k:g}"] = cells(gc.abs())
            r["K3"] = K3
            n = n_done + i + 1
            if n in (1, 2, 5, 10) or n % 16 == 0:
                log(f"[p0] batch {J} row {i} {t} T={r['T']} body={nb_} CE={r['ce_dense']:.3f} "
                    f"gate CE7={r.get('ce_gate7', float('nan')):.2f} cell CE={r.get('ce_cgate', float('nan')):.2f} "
                    f"{time.time() - tb:.0f}s/row, {time.time() - t0:.0f}s")
        # ------------------------------------------------------------ Q1 rank fractions (per allocation and split)
        bodies = [r["body"] for r in R]
        RF = {}  # (alloc, L) -> list of (T,) float64, +inf off body
        for al, L in q1_keys:
            sc = torch.cat([r["down"][L][b] for r, b in zip(R, bodies)])
            if al == "glob":
                grp = torch.zeros(sc.numel(), dtype=torch.long, device=dev)
            elif al == "row":
                grp = torch.cat([torch.full((int(b.sum()),), i, dtype=torch.long, device=dev) for i, b in enumerate(bodies)])
            else:
                grp = torch.cat([i * 100000 + r["cid"][b] for i, (r, b) in enumerate(zip(R, bodies))])
            rf = rank_frac(sc, grp)
            lst, o = [], 0
            for r, b in zip(R, bodies):
                nb_ = int(b.sum())
                v = torch.full((r["T"],), float("inf"), dtype=torch.float64, device=dev)
                v[b] = rf[o:o + nb_]
                o += nb_
                lst.append(v)
            RF[(al, L)] = lst
        # ------------------------------------------------------------ pass 1: dense sketch, multi-hot bins
        tap.accM = [torch.zeros(NG, nbins, 2 * nm, device=dev) for _ in range(sp)]
        grid_t = torch.tensor(GRID, dtype=torch.float64, device=dev)
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            body = r["body"]
            T = r["T"]
            cache = {}
            rows_tok = {}
            for qi_, (al, L) in enumerate(q1_keys):
                rows_tok[(al, L)] = (RF[(al, L)][i][None, :] < grid_t[:, None]) | ~body[None]  # (ng, T)
            named_k = []
            for s in named:
                if s in r["K2"]:
                    L = int(s.split("_L")[1].split("_")[0])
                    named_k.append((L, (r["K2"][s] | ~body)))
                else:
                    named_k.append((Q3_L, r["K3"][s]))  # (NB - L, T)

            def memb(bk, cache=cache, rows_tok=rows_tok, named_k=named_k, T=T):
                if bk in cache:
                    return cache[bk]
                M = torch.ones(nbins, T, device=dev)
                o = 1
                for al, L in q1_keys:
                    if bk >= L:
                        M[o:o + ng] = rows_tok[(al, L)].float()
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
            del cache
        # ------------------------------------------------------------ Q1 budgets: smallest grid fraction reaching 0.95
        SD = [to_real(acc[:, 0]) for acc in tap.accM]  # (NG, m) per space
        bres = {"batch": J, "q1": {}}
        A_vec = {}  # sel -> [raw, adam] (NG, m)
        o = 1
        for al, L in q1_keys:
            curves = []
            for spi in range(sp):
                C = to_real(tap.accM[spi][:, o:o + ng])  # (NG, ng, m)
                ip = torch.einsum("gjm,gm->j", C.double(), SD[spi].double())
                nn_ = C.double().pow(2).sum((0, 2))
                curves.append((ip / torch.sqrt(nn_ * SD[spi].double().pow(2).sum())).tolist())
                if spi == 0:
                    ok = [j for j, c in enumerate(curves[0]) if c >= TARGET]
                    js = ok[0] if ok else None
                if js is not None:
                    A_vec.setdefault(f"q1_{al}_L{L}", []).append(C[:, js].clone())
                else:
                    A_vec.setdefault(f"q1_{al}_L{L}", []).append(SD[spi].clone())
                del C
            fstar = GRID[js] if js is not None else 1.0
            kept = sum(int(((RF[(al, L)][i] < fstar) & R[i]["body"]).sum()) for i in range(len(R)))
            nbody = sum(int(r["body"].sum()) for r in R)
            bres["q1"][f"{al}_L{L}"] = dict(fstar=fstar, keep_frac=kept / max(1, nbody), attr_cos=curves[0][js]
                                             if js is not None else 1.0, curve_raw=curves[0],
                                             curve_adam=curves[1] if sp > 1 else None)
            for i, r in enumerate(R):
                r.setdefault("K1", {})[f"q1_{al}_L{L}"] = (L, (RF[(al, L)][i] < fstar) & r["body"])
            o += ng
        for s in named:
            A_vec[s] = [to_real(tap.accM[spi][:, o]) for spi in range(sp)]
            o += 1
        del tap.accM
        log(f"batch {J} q1 budgets: " + ", ".join(f"{k} f*={v['fstar']:g} keep={v['keep_frac']:.4f}"
                                                 for k, v in bres["q1"].items()))
        # ------------------------------------------------------------ routed passes
        accR = {s: torch.zeros(sp, NG, 2 * nm, device=dev) for s in allsels}
        if a.selfcheck:
            accR["_dense"] = torch.zeros(sp, NG, 2 * nm, device=dev)
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            body = r["body"]
            nb_ = int(body.sum())
            orow = dict(batch=J, task=t, k=rr["k"], md5=rr["md5"], T=r["T"], n_body=nb_, n_answer=int(r["pred"].numel()),
                        gold_docs=r["gold_docs"], numbered_ok=r["numbered_ok"], ce_dense=r["ce_dense"],
                        ce_gate={L: r.get(f"ce_gate{L}") for L in Q2_KS}, ce_cgate=r.get("ce_cgate"),
                        ce_routed={}, keep={}, energy_share={}, gold_share={}, overlap_en={}, spearman_gate_en={})
            for L in Q2_KS:
                if f"gate{L}" in r:
                    orow["spearman_gate_en"][L] = spearman_t(r[f"gate{L}"][body], r["down"][L][body])
            gold = r["ty"] == 2
            for s in allsels + (["_dense"] if a.selfcheck else []):
                tap.accS = [accR[s][spi] for spi in range(sp)]
                tap.mode = "sum"
                tap.fired = 0
                if s == "_dense":
                    loss, _, _ = RP.run_row(model, tap, r, ckpt=True)
                elif s in r["K3"]:
                    loss = run_masked(model, tap, r, Q3_L, r["K3"][s])
                else:
                    L, kp = r["K1"][s] if s.startswith("q1_") else (int(s.split("_L")[1].split("_")[0]), r["K2"][s])
                    loss, _, _ = RP.run_row(model, tap, r, stages={L: (lambda h, cur, kp=kp, body=body: (kp | ~body)[cur])},
                                            ckpt=True)
                tap.backward(loss)
                tap.mode = None
                if tap.fired != len(lins):
                    raise SystemExit(f"{s}: hooks fired {tap.fired} for {len(lins)} linears")
                orow["ce_routed"][s] = float(loss)
                if s == "_dense":
                    continue
                if s in r["K3"]:
                    kp = r["K3"][s][:, body]
                    orow["keep"][s] = float(kp.float().mean())
                    Eb = r["E"][Q3_L:][:, body]
                    orow["energy_share"][s] = float((Eb * kp).sum() / Eb.sum().clamp(min=1e-30))
                    orow["gold_share"][s] = float(kp[:, gold[body]].sum() / kp.sum().clamp(min=1))
                    continue
                kb = (r["K1"][s][1] if s.startswith("q1_") else r["K2"][s]) & body
                L = int(s.split("_L")[1].split("_")[0])
                orow["keep"][s] = float(kb.sum() / max(1, nb_))
                orow["energy_share"][s] = float(r["down"][L][kb].sum() / r["down"][L][body].sum().clamp(min=1e-30))
                orow["gold_share"][s] = float((kb & gold).sum() / max(1, int(kb.sum())))
                if not s.startswith("q1_"):
                    en = r["K2"]["en_" + s.split("_", 1)[1]] & body
                    orow["overlap_en"][s] = float((kb & en).sum() / max(1, int((kb | en).sum())))
            out_rows.append(orow)
            n_done += 1
            if n_done in (1, 2, 5, 10) or n_done % 8 == 0:
                el = time.time() - t0
                tot = len(mine) * a.batch
                log(f"[routed] row {n_done}/{tot} ({t}) CE dense {r['ce_dense']:.3f} "
                    + " ".join(f"{s} {orow['ce_routed'].get(s, float('nan')):.3f}" for s in
                               ("q1_doc_L7", "en_L7_k0.1", "gate_L7_k0.1", "enattn_L7_k0.1", "q3_cell_en_k0.1"))
                    + f"; {el:.0f}s, ETA {el / n_done * (tot - n_done):.0f}s; GPU peak "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            for kk in ("E", "K2", "K3", "K1", "down", "tok_cg") + tuple(f"gate{L}" for L in Q2_KS):
                r.pop(kk, None)
        # ------------------------------------------------------------ save
        SR = {s: [to_real(accR[s][spi]) for spi in range(sp)] for s in accR}
        if a.selfcheck:
            for spi in range(sp):
                x_, y_ = SR["_dense"][spi].double(), SD[spi].double()
                log(f"selfcheck space {spi}: cos(D_sum, D_multi) = {float((x_ * y_).sum() / (x_.norm() * y_.norm())):.6f}"
                    f" norm ratio {float(x_.norm() / y_.norm()):.6f}")
                bres[f"selfcheck_{spi}"] = float((x_ * y_).sum() / (x_.norm() * y_.norm()))
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
                    tasks=tasks, sels=allsels, q1=s_q1, q2=s_q2, q3=s_q3, grid=GRID, target=TARGET, q2_ks=Q2_KS,
                    q3_ks=Q3_KS, gate=GATE, sketch_dim=a.sketch_dim, adam=adam is not None,
                    adam_source=(adam_diag or {}).get("_source"), batch_stats=out_batches, rows=out_rows,
                    seconds=time.time() - t00)
        json.dump(meta, open(a.out, "w"))
        if save_dir:
            json.dump(meta, open(os.path.join(save_dir, f"meta_shard{a.shard}.json"), "w"))
    log(f"done {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
