"""Gradient composition of the LM loss in long-context CONTINUED PRETRAINING: OLMo 3 7B on 32k Longmino dev rows.

The CPT counterpart of ``probe.py`` (records/grad-composition.md, addendum). Every token is a label, so there are no
documents-vs-query categories; the structure is position in the row, distance to the document start (EOS-delimited)
and label vs context. Per row, on one GPU (bf16, frozen):

  pass 1 (full)    loss = mean next-token CE over the labelled targets. Block = two checkpointed halves (OLMo 3 is
                   reordered-norm: h + norm(attn(h)), then + norm(ffn(.))). Per token t, block b: G = ||dL/dh_b,t||
                   (b = 0..32), the residual update norm, attention mass received at the 8 full-attention blocks
                   (from all queries, and from queries >= 4096 tokens away = beyond every sliding window), the
                   per-token target CE; exact per-block parameter-gradient energy (attention / FFN / norms; LM head).
  pass 2 (own)     same loss with every attention layer's keys and values DETACHED: a token's hidden-state gradient
                   then only flows from its own label (its own query path and residual stream). G_own; the context
                   part is g_full - g_own (what the token receives for serving as keys / values to later tokens).

Per-row statistics -> <out>_stats.npz; per-row token CE + ids -> <out>_ce.npz (long-context gain analysis);
full float16 G / G_own for the first --dump-rows rows -> <out>_dump_<i>.npz.

    python cpt_probe.py --ckpt /data/prasann/olmo3_cptdrop/runs/o3cd-Dlr20-480 --yarn-factor 4 \
        --dev /data/prasann/olmo3_cptdrop/data/dev32k_1024 --rows 48 --out /data/.../dlr20-480/cpt.json
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
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "olmo3_cptdrop"))
from eval_devce import build_cfg, find_distcp, load_model, load_shard, parse_yarn  # noqa: E402

EOS = 100257
NB = 32
KS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
POS_EDGES = (0, 4096, 8192, 16384, 32768)
DOC_EDGES = (0, 64, 512, 4096, 16384, 1 << 30)
PYR = (0, 3, 7, 8, 12, 16, 20, 24, 28, 31, 32)
PYR_KA = (0.5, 0.3, 0.1)
PYR_SETS = ("nest_3_7_16", "one_7_3pct", "one_16_3pct", "one_7_10pct", "one_7_30pct", "random_3pct", "random_10pct")
PYR_TGTS = ("g16", "g24", "g31", "g32")
FAR = 4096
CE_CHUNK = 4096


def log(m):
    print(f"[cpt-probe] {m}", flush=True)


class Capture:
    def __init__(self, model):
        from olmo_core.nn.attention import Attention

        self.on = False
        self.detach_kv = False
        self.am = {}
        self.full = []
        for key, block in model.blocks.items():
            att = block.attention
            assert isinstance(att, Attention), type(att)
            i = int(key)
            if tuple(getattr(att.backend, "window_size", (-1, -1))) == (-1, -1):
                self.full.append(i)
            self._wrap(att, i)

    def _wrap(self, att, i):
        orig = att._prepare_qkv

        def wrapped(*args, **kw):
            q, k, v = orig(*args, **kw)
            if self.on and i in self.full:
                sc = getattr(att.backend, "scale", None) or q.shape[-1] ** -0.5
                self.am[i] = attn_mass(q.detach(), k.detach(), sc)
            if self.detach_kv:
                k, v = k.detach(), v.detach()
            return q, k, v

        att._prepare_qkv = wrapped


@torch.no_grad()
def attn_mass(q, k, scale, chunk=256):
    """Head-mean causal attention mass received by each key: from all queries, and from queries >= FAR away."""
    q, k = q[0], k[0]
    T, Hq, _ = q.shape
    k = k.repeat_interleave(Hq // k.shape[1], dim=1)
    tot = torch.zeros(T, device=q.device)
    far = torch.zeros(T, device=q.device)
    ar = torch.arange(T, device=q.device)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        S = torch.einsum("qhd,khd->hqk", q[s:e].float(), k[:e].float()) * scale
        S.masked_fill_((ar[None, :e] > ar[s:e, None])[None], float("-inf"))
        P = torch.softmax(S, dim=-1).mean(0)  # (c, e)
        tot[:e] += P.sum(0)
        dist_ok = (ar[s:e, None] - ar[None, :e]) >= FAR
        far[:e] += (P * dist_ok).sum(0)
    return tot, far


class ParamGrads:
    def __init__(self, model):
        self.groups = []
        for n, p in model.named_parameters():
            if n.startswith("blocks."):
                i = int(n.split(".")[1])
                part = 0 if ".attention." in n else 1 if ".feed_forward." in n else 2
            elif n.startswith("lm_head"):
                i, part = -1, 0
            else:
                continue
            self.groups.append((p, i, part))
        self.acc = None

    def enable(self, on):
        for p, _, _ in self.groups:
            p.requires_grad_(on)

    @torch.no_grad()
    def take(self):
        e = torch.zeros(NB + 1, 3, dtype=torch.float64)
        if self.acc is None:
            self.acc = [torch.zeros_like(p, dtype=torch.float32) for p, _, _ in self.groups]
        for (p, i, part), a in zip(self.groups, self.acc):
            if p.grad is None:
                continue
            e[i if i >= 0 else NB, part] += float(p.grad.float().pow(2).sum())
            a.add_(p.grad.float())
            p.grad = None
        return e.numpy()

    @torch.no_grad()
    def summed(self):
        e = torch.zeros(NB + 1, 3, dtype=torch.float64)
        for (p, i, part), a in zip(self.groups, self.acc or []):
            e[i if i >= 0 else NB, part] += float(a.pow(2).sum())
        return e.numpy()


def run_pass(model, cap, x, lmask, record):
    """Forward + backward of the mean target CE. record=True: also capture attention mass and per-token CE."""
    from torch.utils.checkpoint import checkpoint

    T = x.numel()
    tgt_ok = lmask[1:]  # logits at i predict token i+1
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None)
    e = model.embeddings(x[None]).detach().requires_grad_(True)
    hs, h = [e], e
    cap.on = record
    cap.am.clear()
    for key, block in model.blocks.items():
        kw = {**abk, **pbk.get(int(key), {})}
        m = checkpoint(lambda hh, b=block, k=kw: b.attention_residual_stream(hh, b.attention_norm(b.attention(hh, **k))),
                       h, use_reentrant=False)
        h = checkpoint(lambda mm, b=block: b.feed_forward_residual_stream(mm, b.feed_forward_norm(b.feed_forward(mm))),
                       m, use_reentrant=False)
        h.retain_grad()
        hs.append(h)
        del m
    cap.on = False
    n_tgt = int(tgt_ok.sum())
    tok_ce = torch.full((T,), float("nan"), device=x.device)
    loss = 0.0
    for s in range(0, T - 1, CE_CHUNK):
        e_ = min(T - 1, s + CE_CHUNK)

        def chunk_ce(hh, s=s, e_=e_):
            lg = model.lm_head(hh)[0].float()
            return F.cross_entropy(lg, x[s + 1:e_ + 1], reduction="none")

        ce = checkpoint(chunk_ce, h[:, s:e_], use_reentrant=False)
        ok = tgt_ok[s:e_]
        loss = loss + (ce * ok).sum() / n_tgt
        if record:
            tok_ce[s + 1:e_ + 1] = torch.where(ok, ce.detach(), torch.full_like(ce, float("nan")))
    loss.backward()
    f = lambda t: t.detach()[0].float()  # noqa: E731
    G = torch.stack([f(t.grad).norm(dim=-1) for t in hs])  # (33, T)
    gvec = hs  # keep references for the own/ctx split
    out = {"loss": float(loss), "G": G}
    if record:
        out["tok_ce"] = tok_ce
        out["dh"] = torch.stack([(f(hs[b + 1]) - f(hs[b])).norm(dim=-1) for b in range(NB)])
        out["AMall"] = torch.stack([cap.am[l][0] for l in cap.full])
        out["AMfar"] = torch.stack([cap.am[l][1] for l in cap.full])
    out["_hs"] = gvec
    return out


def ranks(x):
    r = x.argsort(-1).argsort(-1).float()
    r = r - r.mean(-1, keepdim=True)
    return r / (r.norm(dim=-1, keepdim=True) + 1e-12)


def topk_mask(score, rho):
    k = max(1, int(math.ceil(rho * score.numel())))
    m = torch.zeros_like(score, dtype=torch.bool)
    m[score.topk(k).indices] = True
    return m


def row_stats(full, own_G, cos_own, e_own, e_ctx, x, lmask):
    G = full["G"]
    T = G.shape[1]
    dev = G.device
    pos = torch.arange(T, device=dev)
    is_eos = x == EOS
    # distance to the current document's start (EOS-delimited; the token after an EOS starts a document)
    last_eos = torch.where(is_eos, pos, torch.full_like(pos, -1)).cummax(0).values
    dstart = pos - last_eos - 1
    dstart[last_eos < 0] = pos[last_eos < 0]
    S = {}
    m2 = G ** 2
    tot = m2.sum(-1, keepdim=True).clamp(min=1e-30)
    srt = m2.sort(-1, descending=True).values.cumsum(-1)
    S["conc"] = torch.stack([srt[:, max(1, int(round(k * T))) - 1] for k in KS], -1) / tot
    pb = torch.bucketize(pos, torch.tensor(POS_EDGES[1:-1], device=dev), right=True)
    db = torch.bucketize(dstart, torch.tensor(DOC_EDGES[1:-1], device=dev), right=True)
    S["pos_share"] = torch.zeros(m2.shape[0], len(POS_EDGES) - 1, device=dev).index_add_(1, pb, m2) / tot
    S["doc_share"] = torch.zeros(m2.shape[0], len(DOC_EDGES) - 1, device=dev).index_add_(1, db, m2) / tot
    S["doc_cnt"] = torch.bincount(db, minlength=len(DOC_EDGES) - 1).float()
    R = ranks(torch.log(G + 1e-12))
    S["xcorr"] = R @ R.T
    PY = list(PYR)
    cm = torch.zeros(len(PY), len(PY), len(PYR_KA))
    for ai, a in enumerate(PY):
        for ki, ka in enumerate(PYR_KA):
            keep = topk_mask(G[a], ka)
            cm[ai, :, ki] = (m2[PY][:, keep].sum(-1) / m2[PY].sum(-1).clamp(min=1e-30)).cpu()
    S["contain_mass"] = cm

    def nested(stages):
        alive = torch.ones(T, dtype=torch.bool, device=dev)
        for L, rho in stages:
            sc = torch.where(alive, G[L], torch.full_like(G[L], -1.0))
            alive = topk_mask(sc, rho) & alive
        return alive
    gen = torch.Generator(device=dev).manual_seed(0)
    rnd = torch.rand(T, device=dev, generator=gen)
    sets = {"nest_3_7_16": nested([(3, 0.5), (7, 0.1), (16, 0.03)]), "one_7_3pct": topk_mask(G[7], 0.03),
            "one_16_3pct": topk_mask(G[16], 0.03), "one_7_10pct": topk_mask(G[7], 0.1), "one_7_30pct": topk_mask(G[7], 0.3),
            "random_3pct": topk_mask(rnd, 0.03), "random_10pct": topk_mask(rnd, 0.1)}
    tg = {"g16": m2[16], "g24": m2[24], "g31": m2[31], "g32": m2[32]}
    S["pyr_nested"] = torch.tensor([[float(t[s].sum() / t.sum().clamp(min=1e-30)) for t in tg.values()] for s in sets.values()])
    # label vs context: per block, energy of the own-label part and of the context part relative to the full gradient
    S["own_share"] = e_own  # (33,)  sum ||g_own||^2 / sum ||g_full||^2
    S["ctx_share"] = e_ctx  # (33,)  sum ||g_full - g_own||^2 / sum ||g_full||^2
    S["own_cos"] = cos_own  # (33,)  energy-weighted mean cos(g_own, g_full)
    S["own_rank_corr"] = (ranks(torch.log(own_G + 1e-12)) * R).sum(-1)  # Spearman(G_own_b, G_full_b)
    # per position bin, the context share of the gradient energy (computed in main and passed through e_ctx_pos)
    # attention mass from far queries (>= 4096 back) by key position bin, and its share of all mass received
    AMf, AMa = full["AMfar"], full["AMall"]
    S["am_far_frac"] = AMf.sum(-1) / AMa.sum(-1).clamp(min=1e-30)  # (8,)
    S["am_far_pos"] = torch.zeros(AMf.shape[0], len(POS_EDGES) - 1, device=dev).index_add_(1, pb, AMf) / AMf.sum(-1, keepdim=True).clamp(min=1e-30)
    S["am_far_doc"] = torch.zeros(AMf.shape[0], len(DOC_EDGES) - 1, device=dev).index_add_(1, db, AMf) / AMf.sum(-1, keepdim=True).clamp(min=1e-30)
    Rf = ranks(AMf)
    S["agree_g_amfar"] = (ranks(torch.log(G[[l for l in range(3, 32, 4)]] + 1e-12)) * Rf).sum(-1)
    S["agree_g_dh"] = (ranks(torch.log(G[:NB] + 1e-12)) * ranks(full["dh"])).sum(-1)
    ce = full["tok_ce"]
    ok = ~torch.isnan(ce)
    S["ce_pos"] = torch.stack([ce[ok & (pb == i)].mean() if (ok & (pb == i)).any() else torch.tensor(float("nan"), device=dev)
                               for i in range(len(POS_EDGES) - 1)])
    return {k: (v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.float32)) for k, v in S.items()}, dstart


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--yarn-factor", type=parse_yarn, required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--row-start", type=int, default=0)
    ap.add_argument("--dump-rows", type=int, default=2)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    rows, masks, meta = load_shard(a.dev, a.rows, "olmo3_7b", a.row_start)
    n, L = rows.shape
    log(f"{n} rows x {L} from {a.dev}")
    cfg = build_cfg("olmo3_7b", a.yarn_factor)
    model, load_s = load_model(a.ckpt, cfg, "cuda", L)
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded {find_distcp(a.ckpt)} in {load_s:.0f}s")
    cap = Capture(model)
    log(f"full-attention blocks: {cap.full}")
    if len(cap.full) != 8:
        raise SystemExit(f"expected 8 full-attention blocks, found {cap.full}")
    pg = ParamGrads(model)
    base = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    stats, meta_rows, ces, ids = {}, [], [], []
    t0 = time.time()
    for i in range(n):
        x = torch.from_numpy(rows[i]).cuda()
        lm = torch.from_numpy(masks[i]).cuda()
        # pass 1: full (parameter gradients on)
        pg.enable(True)
        full = run_pass(model, cap, x, lm, record=True)
        pgr = pg.take()
        hs = full.pop("_hs")
        gfull = [t.grad.detach()[0] for t in hs]  # bf16; upcast per block below
        del hs
        # pass 2: keys / values detached, parameters frozen
        pg.enable(False)
        cap.detach_kv = True
        own = run_pass(model, cap, x, lm, record=False)
        cap.detach_kv = False
        hs2 = own.pop("_hs")
        e_own, e_ctx, cos_o = [], [], []
        T = x.numel()
        pos = torch.arange(T, device=x.device)
        pb = torch.bucketize(pos, torch.tensor(POS_EDGES[1:-1], device=x.device), right=True)
        ctx_pos = torch.zeros(NB + 1, len(POS_EDGES) - 1)
        for b in range(NB + 1):
            go = hs2[b].grad.detach()[0].float()
            gf = gfull[b].float()
            ef = gf.pow(2).sum(-1)
            den = ef.sum().clamp(min=1e-30)
            ec = (gf - go).pow(2).sum(-1)
            e_own.append(float(go.pow(2).sum() / den))
            e_ctx.append(float(ec.sum() / den))
            cs = (go * gf).sum(-1) / (go.norm(dim=-1) * gf.norm(dim=-1) + 1e-30)
            cos_o.append(float((cs * ef).sum() / den))
            ctx_pos[b] = (torch.zeros(len(POS_EDGES) - 1, device=x.device).index_add_(0, pb, ec)
                          / torch.zeros(len(POS_EDGES) - 1, device=x.device).index_add_(0, pb, ef).clamp(min=1e-30)).cpu()
        own_G = own["G"]
        del hs2, gfull
        st, dstart = row_stats(full, own_G, torch.tensor(cos_o), torch.tensor(e_own), torch.tensor(e_ctx), x, lm)
        st["ctx_share_pos"] = ctx_pos.numpy()
        st["pgrad"] = pgr
        for k, v in st.items():
            stats.setdefault(k, []).append(v)
        ce = full["tok_ce"].cpu().numpy().astype(np.float32)
        ces.append(ce.astype(np.float16))
        ids.append(rows[i].astype(np.int32))
        meta_rows.append(dict(row=a.row_start + i, loss=full["loss"], loss_own=own["loss"],
                              n_docs=int((rows[i] == EOS).sum())))
        if i < a.dump_rows:
            np.savez_compressed(f"{base}_dump_{i}.npz", G=full["G"].half().cpu().numpy(), G_own=own_G.half().cpu().numpy(),
                                dh=full["dh"].half().cpu().numpy(), AMall=full["AMall"].half().cpu().numpy(),
                                AMfar=full["AMfar"].half().cpu().numpy(), ce=ce.astype(np.float16), ids=rows[i].astype(np.int32),
                                dstart=dstart.cpu().numpy().astype(np.int32))
        del full, own
        k = i + 1
        if k in (1, 2, 5, 10) or k % 8 == 0 or k == n:
            el = time.time() - t0
            log(f"row {k}/{n} loss {meta_rows[-1]['loss']:.3f} (own-pass {meta_rows[-1]['loss_own']:.3f}) "
                f"blocks0-6 pgrad share {pgr[:7].sum() / pgr.sum():.3f} ctx share@b16 {e_ctx[16]:.2f} "
                f"{el:.0f}s, {el / k:.1f}s/row, ETA {el / k * (n - k):.0f}s; peak mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB")
    np.savez_compressed(f"{base}_stats.npz", **{k: np.stack(v) for k, v in stats.items()})
    np.savez_compressed(f"{base}_ce.npz", ce=np.stack(ces), ids=np.stack(ids))
    json.dump(dict(ckpt=a.ckpt, ckpt_resolved=find_distcp(a.ckpt), label=a.label, dev=a.dev, row_start=a.row_start,
                   eval_size_rows=n, yarn_factor=a.yarn_factor, full_blocks=cap.full, ks=KS, pos_edges=POS_EDGES,
                   doc_edges=DOC_EDGES[:-1], pyr=PYR, pyr_ka=PYR_KA, pyr_sets=PYR_SETS, pyr_tgts=PYR_TGTS, far=FAR,
                   rows=meta_rows, summed_pgrad=pg.summed().tolist(),
                   pgrad_layout="(33, 3): rows 0-31 blocks, 32 lm_head; cols attention, ffn, norms"),
              open(a.out, "w"), indent=1)
    log(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
