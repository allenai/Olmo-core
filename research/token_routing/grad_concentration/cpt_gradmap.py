"""Per-token x per-block gradient maps for long-context CONTINUED PRETRAINING (OLMo 3 7B, Longmino dev rows).

The CPT counterpart of ``gradmap.py`` (SFT answer loss). Reuses ``grad_composition/cpt_probe.py``'s K/V-detach hook
(``Capture``) and its block loop (two checkpointed halves per block). Per row, on one GPU (bf16, frozen weights):

  pass full   loss = mean next-token CE over every target of the row. Per (block b, token t):
                G[b, t] = ||dL/dh_b[t]||  (b = 0..32; h_b = block b's input, h_32 = final hidden state)
                E[b, t] = sum over block b's linears of ||dL/dy_t||^2 ||x_t||^2  (rank-1 weight-gradient energy,
                          the same attribution quantity as the SFT maps' E)
              plus exact per-block parameter-gradient energy ||grad W||^2 (attention / FFN / norms; LM head) and the
              per-token target CE.
  pass own    the same loss with every attention layer's keys / values DETACHED: a token's hidden-state gradient
              then comes only from its own label. G_own[b, t]; the context part is g_full - g_own, stored as
              C[b, t] = ||g_full - g_own||^2 (the gradient token t receives for serving as key / value to later tokens).
  pass late   loss = mean CE over the targets in the last --span tokens only: G_late, E_late, exact param energy.
  pass mid    loss = mean CE over the targets in [T/2, T/2 + span): G_mid, E_mid, exact param energy.

g_full is offloaded to host memory (bf16) between the full and own passes, so the row length is bounded by the 33
retained block inputs (33 x T x 4096 bf16 = 35 GB at 128k).

Writes <out-dir>/<label>/r<row>.npz (float32 maps; ids, ce) and <out-dir>/<label>/rows.json.

    python research/token_routing/grad_concentration/cpt_gradmap.py --ckpt /data/prasann/olmo3_cptdrop/runs/o3cd-Dlr20-480 \
        --yarn-factor 4 --dev /data/prasann/olmo3_cptdrop/data/dev32k_1024 --rows 1,7,10 --label dlr20-480_32k \
        --out-dir /data/prasann/grad_composition/cpt_gradmap
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
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "olmo3_cptdrop"))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "grad_composition"))
from cpt_probe import Capture, ParamGrads as _PG  # noqa: E402
from eval_devce import build_cfg, find_distcp, load_model, load_shard, parse_yarn  # noqa: E402

NB = 32
CE_CHUNK = 4096


def log(m):
    print(f"[cpt-gradmap] {m}", flush=True)


class ParamGrads(_PG):
    """cpt_probe's per-block parameter-gradient energy, without its fp32 running sum (29 GB on the GPU)."""

    @torch.no_grad()
    def take(self):
        e = torch.zeros(NB + 1, 3, dtype=torch.float64)
        for p, i, part in self.groups:
            if p.grad is not None:
                e[i if i >= 0 else NB, part] += float(p.grad.float().pow(2).sum())
                p.grad = None
        return e.numpy()


class Energy:
    """Per-token rank-1 weight-gradient energy of every block linear, E[b, t] += ||dy_t||^2 ||x_t||^2.

    ||x_t||^2 is taken in the first forward; the tensor hook on that forward's output fires in backward (non-reentrant
    checkpointing backpropagates through the original graph nodes; the recompute only refills saved tensors, and the
    forward hooks it re-triggers are ignored via ``phase``)."""

    def __init__(self, model):
        self.on = False
        self.phase = "fwd"
        self.E = None
        self.fired = 0
        self.n = 0
        for key, block in model.blocks.items():
            b = int(key)
            for _, mod in block.named_modules():
                if isinstance(mod, nn.Linear):
                    mod.register_forward_hook(self._hook(b))
                    self.n += 1

    def _hook(self, b):
        def hook(mod, inp, out):
            if not self.on or self.phase != "fwd" or not out.requires_grad:
                return
            xn = inp[0].detach()[0].float().pow(2).sum(-1)

            def on_g(g, xn=xn):
                self.E[b].add_(xn * g.detach()[0].float().pow(2).sum(-1))
                self.fired += 1
            out.register_hook(on_g)
        return hook


def run_pass(model, cap, en, x, tmask, energy, on_grad):
    """Forward + backward of the mean CE over targets ``tmask`` (bool over positions 1..T-1 of the row: tmask[i] = the
    logit at i is scored against token i+1). ``on_grad(b, g)`` receives dL/dh_b (1, T, d) for b = 0..32.
    Returns (loss, per-token CE (T,) with NaN off-target)."""
    from torch.utils.checkpoint import checkpoint

    T = x.numel()
    _, _, abk, pbk, _ = model._prepare_inputs(x[None], None)
    e = model.embeddings(x[None]).detach().requires_grad_(True)
    e.register_hook(lambda g: on_grad(0, g))
    h = e
    en.on, en.phase, en.fired = energy, "fwd", 0
    if energy:
        en.E = torch.zeros(NB, T, device=x.device)
    for key, block in model.blocks.items():
        b = int(key)
        kw = {**abk, **pbk.get(b, {})}
        m = checkpoint(lambda hh, bl=block, k=kw: bl.attention_residual_stream(hh, bl.attention_norm(bl.attention(hh, **k))),
                       h, use_reentrant=False)
        h = checkpoint(lambda mm, bl=block: bl.feed_forward_residual_stream(mm, bl.feed_forward_norm(bl.feed_forward(mm))),
                       m, use_reentrant=False)
        h.register_hook(lambda g, b=b: on_grad(b + 1, g))
        del m
    n_tgt = int(tmask.sum())
    tok_ce = torch.full((T,), float("nan"), device=x.device)
    loss = 0.0
    for s in range(0, T - 1, CE_CHUNK):
        e_ = min(T - 1, s + CE_CHUNK)
        ok = tmask[s:e_]
        if not bool(ok.any()):
            continue

        def chunk_ce(hh, s=s, e_=e_):
            lg = model.lm_head(hh)[0].float()
            return F.cross_entropy(lg, x[s + 1:e_ + 1], reduction="none")

        ce = checkpoint(chunk_ce, h[:, s:e_], use_reentrant=False)
        loss = loss + (ce * ok).sum() / n_tgt
        tok_ce[s + 1:e_ + 1] = torch.where(ok, ce.detach(), torch.full_like(ce, float("nan")))
    en.phase = "bwd"
    loss.backward()
    en.on = False
    if energy and en.fired != en.n:
        raise SystemExit(f"energy hooks fired {en.fired} times for {en.n} linears")
    return float(loss), tok_ce


def f32(t):
    return t.detach().float().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--yarn-factor", type=parse_yarn, required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--rows", required=True, help="comma-separated row indices of the dev shard")
    ap.add_argument("--span", type=int, default=1024, help="target-span length for the late / mid passes")
    ap.add_argument("--no-own", action="store_true", help="skip the K/V-detached pass")
    ap.add_argument("--smoke-len", type=int, default=0,
                    help="memory test only: concatenate consecutive rows into rows of this length")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows_idx = [int(r) for r in a.rows.split(",")]
    allrows, masks, _ = load_shard(a.dev, max(rows_idx) + 1 + max(0, a.smoke_len) // 4096, "olmo3_7b", 0)
    if a.smoke_len:
        allrows = allrows.reshape(-1)[None, :a.smoke_len].repeat(max(rows_idx) + 1, 0)
        masks = np.ones_like(allrows, dtype=bool)
        log(f"SMOKE: rows are the first {a.smoke_len} tokens of the concatenated shard")
    L = allrows.shape[1]
    cfg = build_cfg("olmo3_7b", a.yarn_factor)
    model, load_s = load_model(a.ckpt, cfg, "cuda", L)
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded {find_distcp(a.ckpt)} in {load_s:.0f}s; rows {rows_idx} x {L} from {a.dev}")
    cap = Capture(model)
    if len(cap.full) != 8:
        raise SystemExit(f"expected 8 full-attention blocks, found {cap.full}")
    en = Energy(model)
    pg = ParamGrads(model)
    out = os.path.join(a.out_dir, a.label)
    os.makedirs(out, exist_ok=True)
    meta = []
    t0 = time.time()
    for k, ri in enumerate(rows_idx):
        tr = time.time()
        x = torch.from_numpy(allrows[ri].astype(np.int64)).cuda()
        lm = torch.from_numpy(masks[ri]).cuda()
        T = x.numel()
        tgt = lm[1:].clone()  # logits at i predict token i+1
        pos = torch.arange(T - 1, device=x.device)
        s_late = T - 1 - a.span
        s_mid = T // 2
        spans = {"late": (pos >= s_late), "mid": (pos >= s_mid) & (pos < s_mid + a.span)}
        arr, item = {}, dict(row=ri, T=T, span=a.span, s_late=int(s_late), s_mid=int(s_mid))
        # ---- full
        G = torch.zeros(NB + 1, T, device=x.device)
        host = [None] * (NB + 1)

        def on_full(b, g):
            gg = g.detach()[0]
            G[b] = gg.float().norm(dim=-1)
            if not a.no_own:
                host[b] = gg.to("cpu")
        pg.enable(True)
        loss, ce = run_pass(model, cap, en, x, tgt, True, on_full)
        arr.update(G=f32(G), E=f32(en.E), ce=f32(ce), ids=allrows[ri].astype(np.int32))
        item.update(loss=loss, pgrad=pg.take().tolist())
        # ---- own (K/V detached)
        if not a.no_own:
            Go = torch.zeros(NB + 1, T, device=x.device)
            C = torch.zeros(NB + 1, T, device=x.device)

            def on_own(b, g):
                go = g.detach()[0].float()
                gf = host[b].to(x.device, non_blocking=False).float()
                Go[b] = go.norm(dim=-1)
                C[b] = (gf - go).pow(2).sum(-1)
                host[b] = None
            pg.enable(False)
            cap.detach_kv = True
            loss_o, _ = run_pass(model, cap, en, x, tgt, False, on_own)
            cap.detach_kv = False
            arr.update(G_own=f32(Go), C=f32(C))
            item["loss_own_pass"] = loss_o
            del Go, C
        host = None
        # ---- target spans
        for nm, sm in spans.items():
            Gs = torch.zeros(NB + 1, T, device=x.device)

            def on_span(b, g, Gs=Gs):
                Gs[b] = g.detach()[0].float().norm(dim=-1)
            pg.enable(True)
            ls, _ = run_pass(model, cap, en, x, tgt & sm, True, on_span)
            arr[f"G_{nm}"], arr[f"E_{nm}"] = f32(Gs), f32(en.E)
            item[f"loss_{nm}"] = ls
            item[f"pgrad_{nm}"] = pg.take().tolist()
            del Gs
        pg.enable(False)
        en.E = None
        np.savez_compressed(os.path.join(out, f"r{ri:04d}.npz"), **arr)
        meta.append(item)
        json.dump(dict(label=a.label, ckpt=a.ckpt, ckpt_resolved=find_distcp(a.ckpt), dev=a.dev, yarn_factor=a.yarn_factor,
                       T=L, span=a.span, eval_size_rows=len(meta), full_blocks=cap.full,
                       pgrad_layout="(33, 3): rows 0-31 blocks, 32 lm_head; cols attention, ffn, norms", rows=meta),
                  open(os.path.join(out, "rows.json"), "w"), indent=1)
        el = time.time() - t0
        n = k + 1
        log(f"row {n}/{len(rows_idx)} (dev row {ri}) loss {loss:.3f} late {item['loss_late']:.3f} mid {item['loss_mid']:.3f} "
            f"own-pass {item.get('loss_own_pass', float('nan')):.3f}; {time.time() - tr:.0f}s this row, ETA "
            f"{el / n * (len(rows_idx) - n):.0f}s; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB")
    log(f"done {len(meta)} rows in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
