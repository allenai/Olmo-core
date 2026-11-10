"""How much of the CPT gradient comes from short- vs long-range dependencies: an exact additive decomposition of the
full-loss parameter gradient by the distance between a target and the token whose computation receives its credit.

The CPT loss is a sum over targets, so its gradient is G = sum_c grad L_c over target chunks c of W tokens (L_c = the
chunk's summed CE / the row's total target count, so sum_c L_c = the full mean loss). For each chunk c, one dense
forward + backward of L_c on the row's prefix x[: (c+1) W + 1] (identical to the full forward on those positions:
attention is causal). Every token's rank-1 gradient terms are TensorSketched per block band (0-6, 7-15, 16-23, 24-31),
and the terms of tokens in chunk c' are credited to gap g = c - c'. Gap 0 is split further with one K/V-detached pass
of the full loss (each token's own-label term = distance 0; the rest of gap 0 = other tokens of the same chunk).
"Through any path": credit at token j from target i includes multi-hop relays through intermediate tokens.

Per row writes the band x gap sketch sums and the own-label sketch; the summed parameter grads of the chunk passes
(= the exact full gradient) give a per-band self-check against the sketch of sum_g. cpt_distance_table.py reads them.

    python research/token_routing/grad_concentration/cpt_distance_decomp.py --ckpt <ckpt> --yarn-factor 4 \
        --dev /data/prasann/olmo3_cptdrop/data/dev32k_1024 --rows 1,2,3,4 --chunk 1024 --out-dir /data/.../cpt_distance/cpt_32k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cpt_gradmap import NB, CE_CHUNK, ParamGrads  # noqa: E402
from cpt_probe import Capture  # noqa: E402
from eval_devce import build_cfg, find_distcp, load_model, load_shard, parse_yarn  # noqa: E402

BANDS = ((0, 7), (7, 16), (16, 24), (24, 32))


def log(m):
    print(f"[cpt-dist] {m}", flush=True)


def band_of(b):
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= b < hi)


class BandSketch:
    """Per-token TensorSketch of every block linear's rank-1 term, one sketch space of m dims per block band."""

    def __init__(self, model, m, chunk=1024):
        self.on, self.phase, self.fired, self.n = False, "fwd", 0, 0
        self.m, self.chunk = m, chunk
        self.xs, self.gs, self.meta = {}, {}, {}
        self.S = None  # (n_bands, T, m)
        for key, block in model.blocks.items():
            b = int(key)
            for _, mod in block.named_modules():
                if isinstance(mod, nn.Linear):
                    i = self.n
                    self.n += 1
                    g = torch.Generator().manual_seed(7001 + i)
                    hx = torch.randint(0, m, (mod.in_features,), generator=g)
                    sx = torch.randint(0, 2, (mod.in_features,), generator=g).float() * 2 - 1
                    hy = torch.randint(0, m, (mod.out_features,), generator=g)
                    sy = torch.randint(0, 2, (mod.out_features,), generator=g).float() * 2 - 1
                    self.meta[i] = (band_of(b), hx.cuda(), sx.cuda(), hy.cuda(), sy.cuda())
                    mod.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(mod, inp, out):
            if not self.on:
                return
            if self.phase == "fwd":
                if out.requires_grad:
                    out.register_hook(lambda g, i=i: self._on_g(i, g.detach()[0]))
                return
            x = inp[0].detach()[0]
            if i in self.gs:
                self._do(i, x, self.gs.pop(i))
            else:
                self.xs[i] = x
        return hook

    def _on_g(self, i, g):
        if i in self.xs:
            self._do(i, self.xs.pop(i), g)
        else:
            self.gs[i] = g

    @torch.no_grad()
    def _do(self, i, x, g):
        bd, hx, sx, hy, sy = self.meta[i]
        T, m = x.shape[0], self.m
        for s in range(0, T, self.chunk):
            e = min(T, s + self.chunk)
            cx = torch.zeros(e - s, m, device=x.device).index_add_(1, hx, x[s:e].float() * sx)
            cy = torch.zeros(e - s, m, device=x.device).index_add_(1, hy, g[s:e].float() * sy)
            self.S[bd, s:e] += torch.fft.irfft(torch.fft.rfft(cx) * torch.fft.rfft(cy), n=m)
        self.fired += 1


def run_pass(model, sk, x, tmask, denom):
    """Forward + backward of sum(CE over tmask) / denom (block loop as in cpt_gradmap.run_pass)."""
    from torch.utils.checkpoint import checkpoint

    T = x.numel()
    _, _, abk, pbk, _ = model._prepare_inputs(x[None], None)
    h = model.embeddings(x[None]).detach().requires_grad_(True)
    sk.on, sk.phase, sk.fired = True, "fwd", 0
    for key, block in model.blocks.items():
        b = int(key)
        kw = {**abk, **pbk.get(b, {})}
        m = checkpoint(lambda hh, bl=block, k=kw: bl.attention_residual_stream(hh, bl.attention_norm(bl.attention(hh, **k))),
                       h, use_reentrant=False)
        h = checkpoint(lambda mm, bl=block: bl.feed_forward_residual_stream(mm, bl.feed_forward_norm(bl.feed_forward(mm))),
                       m, use_reentrant=False)
    loss = 0.0
    for s in range(0, T - 1, CE_CHUNK):
        e_ = min(T - 1, s + CE_CHUNK)
        ok = tmask[s:e_]
        if not bool(ok.any()):
            continue

        def chunk_ce(hh, s=s, e_=e_):
            return F.cross_entropy(model.lm_head(hh)[0].float(), x[s + 1:e_ + 1], reduction="none")

        ce = checkpoint(chunk_ce, h[:, s:e_], use_reentrant=False)
        loss = loss + (ce * ok).sum() / denom
    sk.phase = "bwd"
    loss.backward()
    sk.on = False
    if sk.fired != sk.n or sk.xs or sk.gs:
        raise SystemExit(f"sketch hooks: fired {sk.fired}/{sk.n}, unpaired {len(sk.xs)} x {len(sk.gs)} g")
    return float(loss)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--yarn-factor", type=parse_yarn, required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--m", type=int, default=8192, help="sketch dims per block band")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out", default=None, help="alternative to --out-dir: <out-dir>/<label> (run_cpt_oracle.sbatch)")
    a = ap.parse_args()
    if a.out_dir is None:
        a.out_dir = os.path.dirname(a.out)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows_idx = [int(r) for r in a.rows.split(",")]
    allrows, masks, _ = load_shard(a.dev, max(rows_idx) + 1, "olmo3_7b", 0)
    L = allrows.shape[1]
    model, load_s = load_model(a.ckpt, build_cfg("olmo3_7b", a.yarn_factor), "cuda", L)
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded {find_distcp(a.ckpt)} in {load_s:.0f}s; rows {rows_idx} x {L}, chunk {a.chunk}")
    cap = Capture(model)
    sk = BandSketch(model, a.m)
    pg = ParamGrads(model)
    dev = torch.device("cuda")
    out = os.path.join(a.out_dir, a.label)
    os.makedirs(out, exist_ok=True)
    nb = len(BANDS)
    band_rows = [[r for r in range(NB) if lo <= r < hi] for lo, hi in BANDS]
    t0 = time.time()
    for k, ri in enumerate(rows_idx):
        if os.path.exists(os.path.join(out, f"r{ri:04d}.npz")):
            log(f"row {ri} already done, skipping")
            continue
        tr = time.time()
        x = torch.from_numpy(allrows[ri].astype(np.int64)).cuda()
        lm = torch.from_numpy(masks[ri]).cuda()
        T = x.numel()
        W = a.chunk
        nch = (T + W - 1) // W
        tgt_all = lm[1:].clone()
        N = int(tgt_all.sum())
        bins = torch.zeros(nb, nch, a.m, device=dev, dtype=torch.float64)
        pg.enable(True)
        losses = []
        for c in range(nch):
            end = min(T, (c + 1) * W + 1)  # chunk c's targets are logits c W .. (c+1) W - 1, labels up to (c+1) W
            xs = x[:end]
            pos = torch.arange(end - 1, device=dev)
            tm = tgt_all[:end - 1] & (pos >= c * W) & (pos < (c + 1) * W)
            sk.S = torch.zeros(nb, end, a.m, device=dev)
            losses.append(run_pass(model, sk, xs, tm, N))
            for g in range(c + 1):
                c2 = c - g
                bins[:, g] += sk.S[:, c2 * W:min(end, (c2 + 1) * W)].double().sum(1)
            sk.S = None
        pgr = pg.take()  # summed over the chunk passes = the exact full-loss gradient
        # own-label pass: full loss with keys / values detached
        cap.detach_kv = True
        sk.S = torch.zeros(nb, T, a.m, device=dev)
        loss_own = run_pass(model, sk, x, tgt_all, N)
        cap.detach_kv = False
        pg.take()
        pg.enable(False)
        own = sk.S.double().sum(1)
        sk.S = None
        exact = np.array([pgr[r, :2].sum() for r in range(NB)])
        exact_band = np.array([exact[rs].sum() for rs in band_rows])
        sk_band = bins.sum(1).pow(2).sum(-1).cpu().numpy()
        np.savez_compressed(os.path.join(out, f"r{ri:04d}.npz"), bins=bins.float().cpu().numpy(),
                            own=own.float().cpu().numpy(), exact_band=exact_band, chunk_losses=np.array(losses))
        log(f"row {k + 1}/{len(rows_idx)} (dev row {ri}) full loss {sum(losses):.3f} (own pass {loss_own:.3f}); "
            f"sketch/exact norm^2 per band {np.round(sk_band / exact_band, 3).tolist()}; {time.time() - tr:.0f}s this row, "
            f"ETA {(time.time() - t0) / (k + 1) * (len(rows_idx) - k - 1):.0f}s; GPU peak "
            f"{torch.cuda.max_memory_allocated() / 2**30:.0f} GiB")
    json.dump(dict(label=a.label, ckpt=a.ckpt, dev=a.dev, rows=rows_idx, T=L, chunk=a.chunk, m=a.m,
                   bands=[list(b) for b in BANDS]), open(os.path.join(out, f"meta_{rows_idx[0]}.json"), "w"))
    log(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
