"""Per-token, per-block gradient and activation composition of the answer loss on a frozen checkpoint.

Question (parent session, 2026-10-22): how does the gradient of the answer CE distribute over tokens and blocks of
Qwen3.5-4B, how does that change over CTC fine-tuning, and what does it say about where a train-time token router
should split, whether one decision can serve every layer, and gradient vs activation targets.

Per held-out row: one full-context forward + backward (teacher forced, no-cot, parameters frozen). Every block is
run as two checkpointed halves (mixer, FFN) so the residual stream is observed at the block input h_b, between
the halves m_b and at the output h_{b+1}. Per token t and block b:

  G[b]      ||dL/dh_b||, b = 0..32 (0 = embedding output, 32 = final block output)
  GM[b]     ||dL/dm_b|| (gradient arriving at the mixer's output projection)
  H[b]      ||h_b||;  UA[b] = ||m_b - h_b|| (mixer update), UF[b] = ||h_{b+1} - m_b|| (FFN update)
  attrA/F   <dL/dm_b, m_b - h_b>, <dL/dh_{b+1}, h_{b+1} - m_b>: first-order loss change from removing block b's
            mixer / FFN update at token t (per-(token, layer) skip importance)
  XA/XF     input norm of the mixer's w_out / the FFN's w2; pwA = GM * XA, pwF = G[b+1] * XF are the Frobenius
            norms of token t's rank-1 share of dL/dW_out and dL/dW2 (the token's share of the PARAMETER gradient)
  AM*[l]    attention mass received at the full-attention blocks (3, 7, ..., 31), head mean: from the
            answer-predicting positions (AMans), from the query tokens (AMq), summed over all queries (AMall)

Token categories: prompt, marker, header (first 8 body tokens), first (next 32), mid, last (last 32 body tokens),
each of the body ones split gold / non-gold (``gold_fingerprints.json``), query, answer, other.

Per-row statistics (computed here, ``<out>.npz``): category mass and top-decile composition per layer, concentration
(share of mass in the top k% body tokens), cross-layer Spearman of token rankings, gradient-vs-activation rank
agreement per layer, split analysis (share of the downstream parameter-gradient mass that a per-document top-10%
keep by the layer-L ranking captures, for L = 0..24), and a mid-stack keep-10% ablation (CE of dropping all but
each document's top 10% before block 7 under several rankings).

    python research/token_routing/grad_composition/probe.py --ckpt <dir> --shard <mt11s960> --out /results/x.json
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
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
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "online_router"))

from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens  # noqa: E402
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row  # noqa: E402
from olmo_core.nn.online_token_router import query_mask, structural_features  # noqa: E402

EOS, DOC_START, DOC_END = 248044, 248049, 248050
NB = 32
CATS = ["prompt", "marker", "hdr_g", "hdr_n", "first_g", "first_n", "mid_g", "mid_n", "last_g", "last_n",
        "query", "answer", "other"]
BODY_CATS = list(range(2, 10))
KS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
SPLITS = tuple(range(0, 17)) + (20, 24)
ABL_L = 7
ABL_RHO = 0.1
ABL_RULES = ("g7", "dh7", "amlate", "pwdown", "attrdown", "random", "first_last")
PYR = (0, 3, 7, 8, 12, 16, 20, 24, 28, 32)
PYR_KA = (0.5, 0.3, 0.1)
PYR_KB = (0.1, 0.03)
PYR_SETS = ("nest_3_7_16", "one_7_3pct", "one_16_3pct", "one_7_10pct", "random_3pct")
PYR_TGTS = ("g16", "g24", "g32", "pw16")
DROP_CONC_COLS = [f"{q}_{c}" for q in ("g7", "pwdown7") for c in ("dropped_share", "top10_of_dropped", "top1_of_dropped",
                                                                  "eff_n_frac")] + ["lag1_corr_logG7"]
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"


def log(m):
    print(f"[grad-comp] {m}", flush=True)


def find_ckpt(path):
    if os.path.isdir(os.path.join(path, "model_and_optim")):
        return os.path.join(path, "model_and_optim")
    if path.rstrip("/").endswith("model_and_optim"):
        return path
    c = sorted(glob.glob(f"{path}/step*/model_and_optim"), key=lambda p: int(p.split("/step")[-1].split("/")[0]))
    if not c:
        raise SystemExit(f"no model_and_optim under {path}: {sorted(os.listdir(path))[:40] if os.path.isdir(path) else 'missing'}")
    return c[-1]


def shard_tasks(shard):
    m = json.load(open(os.path.join(shard, "metadata.json")))
    mf = m.get("merged_from")
    if isinstance(mf, str):
        mf = ast.literal_eval(mf)
    return [x["src"].rstrip("/").split("/")[-1].split("_g16M")[0] for x in mf]


def shard_rows(shard, task):
    """(ids uint32, labels mask bool, row starts, row ends) of one task's part."""
    ti = shard_tasks(shard).index(task)
    ids = np.fromfile(os.path.join(shard, f"token_ids_part_t{ti}_000.npy"), dtype=np.uint32)
    lm = np.fromfile(os.path.join(shard, f"labels_mask_t{ti}_000.npy"), dtype=np.bool_)
    ends = np.nonzero(ids == EOS)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    return ids, lm, starts, ends


def row_hashes(shard, task):
    ids, _, st, en = shard_rows(shard, task)
    return {hashlib.md5(ids[s:e + 1].tobytes()).hexdigest() for s, e in zip(st, en)}


def load_rows(shard, tasks, n, offset):
    gold_table = json.load(open(os.path.join(shard, "gold_fingerprints.json")))
    rows = []
    for task in tasks:
        ids, lm, st, en = shard_rows(shard, task)
        for k in range(offset, min(offset + n, len(st))):
            s, e = int(st[k]), int(en[k]) + 1
            x = ids[s:e].astype(np.int64)
            gold = gold_table.get(content_fingerprint_from_row(x.tolist(), EOS))
            if gold and isinstance(gold[0], (list, tuple)):
                gold = [a for p in gold for a in p]
            rows.append(dict(task=task, k=k, ids=x, lmask=lm[s:e].copy(), md5=hashlib.md5(ids[s:e].tobytes()).hexdigest(),
                             gold=sorted(set(int(g) for g in (gold or [])))))
    return rows


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


class ParamGrads:
    """Parameter-gradient energy per block (mixer / FFN / norms) and the LM head, per row, plus the per-task SUM of
    row gradients (the minibatch gradient the optimizer sees, up to scale). The embedding is excluded: the probe
    detaches the embedding output to read dL/de (its per-token gradient is G[0])."""

    def __init__(self, model):
        self.groups = []  # (param, block index or -1 for the head, part 0 mixer / 1 ffn / 2 norms)
        for n, p in model.named_parameters():
            if n.startswith("blocks."):
                i = int(n.split(".")[1])
                part = 0 if ".attention." in n else 1 if ".feed_forward." in n else 2
            elif n.startswith("lm_head"):
                i, part = -1, 0
            else:
                continue
            p.requires_grad_(True)
            self.groups.append((p, i, part))
        self.acc = None

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
    def flush(self):
        """Energy of the accumulated (summed over rows) gradient per group; resets the accumulator."""
        e = torch.zeros(NB + 1, 3, dtype=torch.float64)
        for (p, i, part), a in zip(self.groups, self.acc or []):
            e[i if i >= 0 else NB, part] += float(a.pow(2).sum())
            a.zero_()
        return e.numpy()


class Capture:
    """Forward-time captures: output-projection input norms (every block) and attention mass (full attention)."""

    def __init__(self, model):
        from olmo_core.nn.attention import Attention

        self.on = False
        self.xa, self.xf, self.am = {}, {}, {}
        self.pred = self.isq = None
        self.full = []
        for key, block in model.blocks.items():
            i = int(key)
            block.attention.w_out.register_forward_pre_hook(self._norm_hook(self.xa, i))
            block.feed_forward.w2.register_forward_pre_hook(self._norm_hook(self.xf, i))
            if isinstance(block.attention, Attention):
                self.full.append(i)
                self._wrap_qkv(block.attention, i)

    def _norm_hook(self, store, i):
        def hook(mod, args):
            if self.on:
                store[i] = args[0].detach()[0].float().norm(dim=-1)
        return hook

    def _wrap_qkv(self, att, i):
        orig = att._prepare_qkv

        def wrapped(*args, **kw):
            q, k, v = orig(*args, **kw)
            if self.on:
                sc = getattr(att.backend, "scale", None) or q.shape[-1] ** -0.5
                self.am[i] = attn_mass(q.detach(), k.detach(), self.pred, self.isq, sc)
            return q, k, v

        att._prepare_qkv = wrapped


@torch.no_grad()
def attn_mass(q, k, pred, isq, scale, chunk=512):
    """Head-mean causal attention probability received by each key: summed over all queries, averaged over the
    answer-predicting positions, averaged over the query tokens."""
    q, k = q[0].float(), k[0].float()
    T, Hq, _ = q.shape
    k = k.repeat_interleave(Hq // k.shape[1], dim=1)
    tot = torch.zeros(T, device=q.device)
    ans = torch.zeros(T, device=q.device)
    qq = torch.zeros(T, device=q.device)
    ar = torch.arange(T, device=q.device)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        S = torch.einsum("qhd,khd->hqk", q[s:e], k[:e]) * scale
        S.masked_fill_((ar[None, :e] > ar[s:e, None])[None], float("-inf"))
        P = torch.softmax(S, dim=-1).mean(0)
        tot[:e] += P.sum(0)
        if pred[s:e].any():
            ans[:e] += P[pred[s:e]].sum(0)
        if isq[s:e].any():
            qq[:e] += P[isq[s:e]].sum(0)
    return tot, ans / max(1, int(pred.sum())), qq / max(1, int(isq.sum()))


def categories(row, x):
    """Per-token category codes, body mask, doc ids, in-doc offsets, gold flag, query mask, answer-predicting mask."""
    cid = build_chunk_ids_from_tokens(x[None], DOC_START, DOC_END, EOS, mode="chunked")
    offs = {}
    _, body = structural_features(x[None], cid, doc_start_id=DOC_START, doc_end_id=DOC_END, offsets=offs)
    lm = torch.from_numpy(row["lmask"]).to(x.device)
    labels = torch.full_like(x, -100)
    labels[:-1] = torch.where(lm[1:], x[1:], torch.full_like(x[1:], -100))
    isq = query_mask(cid, labels[None])[0]
    c, b, j, r = cid[0].long(), body[0], offs["j"][0], offs["r"][0]
    gold = torch.zeros_like(b)
    for g in row["gold"]:
        gold |= c == g
    cat = torch.full_like(c, CATS.index("prompt"))
    cat[(c >= 0) & ~b] = CATS.index("marker")
    sub = torch.where(j < 8, 0, torch.where(r < 32, 3, torch.where(j < 40, 1, 2)))  # hdr, first, mid, last
    cat = torch.where(b, 2 + 2 * sub + (~gold).long(), cat)
    cat[isq] = CATS.index("query")
    cat[lm] = CATS.index("answer")
    first_ans = int(torch.nonzero(lm)[0]) if lm.any() else len(x)
    cat[(torch.arange(len(x), device=x.device) > first_ans) & ~lm & (c < 0)] = CATS.index("other")
    pred = torch.zeros_like(b)
    pred[:-1] = lm[1:]
    return dict(cat=cat, body=b, cid=c, j=j, r=r, gold=gold & b, isq=isq, pred=pred)


def full_pass(model, cap, row, cats):
    from torch.utils.checkpoint import checkpoint

    x = torch.from_numpy(row["ids"]).cuda()
    pred_pos = torch.nonzero(cats["pred"], as_tuple=True)[0]
    tgt = x[pred_pos + 1]
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred_pos[None])
    e = model.embeddings(x[None]).detach().requires_grad_(True)
    hs, ms = [e], []
    h = e
    cap.on, cap.pred, cap.isq = True, cats["pred"], cats["isq"]
    cap.xa.clear(), cap.xf.clear(), cap.am.clear()
    for key, block in model.blocks.items():
        kw = {**abk, **pbk.get(int(key), {})}
        m = checkpoint(lambda hh, b=block, k=kw: b.attention_residual_stream(hh, b.attention(b.attention_norm(hh), **k)),
                       h, use_reentrant=False)
        h = checkpoint(lambda mm, b=block: b.feed_forward_residual_stream(mm, b.feed_forward(b.feed_forward_norm(mm))),
                       m, use_reentrant=False)
        m.retain_grad()
        h.retain_grad()
        ms.append(m)
        hs.append(h)
    cap.on = False
    ce = F.cross_entropy(model.lm_head(h, **lmk)[0].float(), tgt)
    ce.backward()
    f = lambda t: t.detach()[0].float()  # noqa: E731
    o = {k: [] for k in ("G", "GM", "H", "UA", "UF", "attrA", "attrF", "pwA", "pwF", "DH0")}
    h0 = f(hs[0])
    for b in range(NB + 1):
        hb, gb = f(hs[b]), f(hs[b].grad)
        o["G"].append(gb.norm(dim=-1))
        o["H"].append(hb.norm(dim=-1))
        o["DH0"].append((hb - h0).norm(dim=-1))
        if b == NB:
            break
        mb, gmb = f(ms[b]), f(ms[b].grad)
        hn, gn = f(hs[b + 1]), f(hs[b + 1].grad)
        o["GM"].append(gmb.norm(dim=-1))
        o["UA"].append((mb - hb).norm(dim=-1))
        o["UF"].append((hn - mb).norm(dim=-1))
        o["attrA"].append((gmb * (mb - hb)).sum(-1))
        o["attrF"].append((gn * (hn - mb)).sum(-1))
        o["pwA"].append(o["GM"][-1] * cap.xa[b])
        o["pwF"].append(gn.norm(dim=-1) * cap.xf[b])
    o = {k: torch.stack(v) for k, v in o.items()}
    for nm, i in (("AMall", 0), ("AMans", 1), ("AMq", 2)):
        o[nm] = torch.stack([cap.am[l][i] for l in cap.full])
    del hs, ms
    return float(ce), o


@torch.no_grad()
def mid_ce(model, row, keep_pos, L):
    """Answer CE with every token through blocks [0, L) and only ``keep_pos`` through [L, 32) (the mid-stack drop)."""
    x = torch.from_numpy(row["ids"]).cuda()
    lm = torch.from_numpy(row["lmask"]).cuda()
    pred_full = torch.nonzero(lm[1:], as_tuple=True)[0]
    tgt = x[pred_full + 1]
    pred = torch.searchsorted(keep_pos, pred_full)
    assert bool((keep_pos[pred] == pred_full).all())
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    h = model.embeddings(x[None])
    for key, block in model.blocks.items():
        i = int(key)
        if i == L:
            h = h[:, keep_pos]
            abk = {**abk, "position_ids": keep_pos[None]}
        h = block(h, **{**abk, **pbk.get(i, {})})
    return float(F.cross_entropy(model.lm_head(h, **lmk)[0].float(), tgt))


def doc_topk(score, doc, rho):
    n = score.numel()
    o1 = torch.argsort(-score)
    o2 = o1[torch.argsort(doc[o1], stable=True)]
    cnt = torch.bincount(doc)
    starts = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(n, device=score.device) - starts[doc[o2]]
    keep = torch.zeros(n, dtype=torch.bool, device=score.device)
    keep[o2] = rank < torch.ceil(rho * cnt.float()).long()[doc[o2]]
    return keep


def ranks(x):
    """Row-wise ranks (last dim), standardized for Spearman via a dot product."""
    r = x.argsort(-1).argsort(-1).float()
    r = r - r.mean(-1, keepdim=True)
    return r / (r.norm(dim=-1, keepdim=True) + 1e-12)


def row_stats(o, cats, ce, model, row):
    b = cats["body"]
    cat = cats["cat"]
    doc = cats["cid"][b]
    doc = doc - doc.min()
    nb = int(b.sum())
    pw = o["pwA"] ** 2 + o["pwF"] ** 2  # (32, T) token's parameter-gradient energy at block b's output projections
    attr = o["attrA"].abs() + o["attrF"].abs()
    upd = (o["UA"] ** 2 + o["UF"] ** 2).sqrt()
    mass = {"g2": o["G"] ** 2, "pw": pw, "attr": attr, "dh": upd ** 2, "am": o["AMans"], "amall": o["AMall"]}
    ncat = len(CATS)
    S = {}
    for q, M in mass.items():
        Lq = M.shape[0]
        cs = torch.zeros(Lq, ncat, device=M.device).index_add_(1, cat, M)
        S[f"cat_sum/{q}"] = cs
        Mb = M[:, b]
        k10 = max(1, int(round(0.1 * nb)))
        top = Mb.topk(k10, dim=-1).indices
        S[f"top10_cat/{q}"] = torch.stack([torch.bincount(cat[b][top[l]], minlength=ncat).float() for l in range(Lq)])
        srt = Mb.sort(-1, descending=True).values.cumsum(-1)
        tot = srt[:, -1:].clamp(min=1e-30)
        S[f"conc/{q}"] = torch.stack([srt[:, max(1, int(round(kk * nb))) - 1] for kk in KS], -1) / tot
        S[f"docconc10/{q}"] = torch.stack([Mb[l][doc_topk(Mb[l], doc, 0.1)].sum() for l in range(Lq)]) / tot[:, 0]
    S["cat_cnt"] = torch.bincount(cat, minlength=ncat).float()
    # cross-layer rank correlation of body-token importance
    for q, M in (("g", o["G"]), ("pw", pw), ("attr", attr), ("dh", upd), ("am", o["AMans"])):
        R = ranks(M[:, b])
        S[f"xcorr/{q}"] = R @ R.T
    k10 = max(1, int(round(0.1 * nb)))
    topg = torch.zeros(NB + 1, nb, device=b.device)
    topg.scatter_(1, o["G"][:, b].topk(k10, dim=-1).indices, 1.0)
    S["xjacc/g"] = (topg @ topg.T) / k10
    # gradient vs activation agreement per layer (body tokens)
    Rg = ranks(o["G"][:NB, b])
    S["agree/g_dh"] = (Rg * ranks(upd[:, b])).sum(-1)
    S["agree/g_attr"] = (Rg * ranks(attr[:, b])).sum(-1)
    S["agree/g_pw"] = (Rg * ranks(pw[:, b])).sum(-1)
    S["agree/g_h"] = (Rg * ranks(o["H"][:NB, b])).sum(-1)
    S["agree/g_dh0"] = (Rg * ranks(o["DH0"][:NB, b])).sum(-1)
    full = torch.tensor([3 + 4 * i for i in range(8)], device=b.device)
    Ram = ranks(o["AMans"][:, b])
    S["agree/g_am"] = (ranks(o["G"][full][:, b]) * Ram).sum(-1)
    S["agree/dh_am"] = (ranks(upd[full][:, b]) * Ram).sum(-1)
    S["agree/pw_am"] = (ranks(pw[full][:, b]) * Ram).sum(-1)
    # split analysis: per-document top-10% of body tokens by the layer-L score; non-body tokens always kept
    E = pw.sum()
    rnd = torch.rand(nb, device=b.device)
    sp = []
    for L in SPLITS:
        down = pw[L:].sum(0)
        dsum = down.sum().clamp(min=1e-30)
        base = down[~b].sum()
        cols = []
        for sc in (o["G"][L, b], o["DH0"][L, b], down[b], rnd, o["AMans"][(full >= L).nonzero()[:, 0]].mean(0)[b]
                   if (full >= L).any() else rnd):
            cols.append(float((base + down[b][doc_topk(sc, doc, 0.1)].sum()) / dsum))
        cols.append(float(pw[:L].sum() / E))
        sp.append(cols)
    S["split"] = torch.tensor(sp)  # (len(SPLITS), [gL, dh0L, oracle, random, am>=L, upstream share])
    # mid-stack keep-10% ablation before block ABL_L
    late = full >= ABL_L
    nbpos = torch.nonzero(b, as_tuple=True)[0]
    cnt = torch.bincount(doc)
    starts = torch.cumsum(cnt, 0) - cnt
    jj = torch.arange(nb, device=b.device) - starts[doc]
    rr = cnt[doc] - 1 - jj
    scores = {"g7": o["G"][ABL_L, b], "dh7": o["DH0"][ABL_L, b], "amlate": o["AMans"][late][:, b].mean(0),
              "pwdown": pw[ABL_L:, b].sum(0), "attrdown": attr[ABL_L:, b].sum(0), "random": rnd,
              "first_last": -torch.minimum(jj, rr).float() + 1e-3 * rnd}
    abl = []
    keeps = {}
    for nm in ABL_RULES:
        kb = doc_topk(scores[nm].float(), doc, ABL_RHO)
        keeps[nm] = kb
        keep = ~b.clone()
        keep[nbpos[kb]] = True
        abl.append(mid_ce(model, row, torch.nonzero(keep, as_tuple=True)[0], ABL_L) - ce)
    S["abl_dce"] = torch.tensor(abl)
    # overlap of the router's keep (g7) with the other rankings' keeps (share of g7's kept set)
    S["abl_overlap_g7"] = torch.tensor([float((keeps["g7"] & keeps[nm]).sum() / keeps["g7"].sum().clamp(min=1))
                                        for nm in ABL_RULES])
    # attention-important tokens the gradient ranking misses: of each doc's top 10% by late answer attention,
    # the share NOT in the g7 keep, and their mean g7 percentile
    S["am_missed_by_g7"] = torch.tensor([float((keeps["amlate"] & ~keeps["g7"]).sum() / keeps["amlate"].sum().clamp(min=1)),
                                         float((o["G"][ABL_L, b].argsort().argsort().float()[keeps["amlate"]] / nb).mean())])
    # pyramid: per-document keep at layer a (top ka by G_a) vs layer b's per-document top kb (by G_b)
    G = o["G"]
    PY = torch.tensor(PYR, device=b.device)
    Gb = G[PY][:, b]
    cnt_c = torch.zeros(len(PYR), len(PYR), len(PYR_KA), len(PYR_KB))
    mass_c = torch.zeros(len(PYR), len(PYR), len(PYR_KA))
    keep_a = {(ai, ki): doc_topk(Gb[ai], doc, ka) for ai in range(len(PYR)) for ki, ka in enumerate(PYR_KA)}
    keep_b = {(bi, ki): doc_topk(Gb[bi], doc, kb) for bi in range(len(PYR)) for ki, kb in enumerate(PYR_KB)}
    for ai in range(len(PYR)):
        for ki in range(len(PYR_KA)):
            ka_set = keep_a[(ai, ki)]
            for bi in range(len(PYR)):
                m2 = Gb[bi] ** 2
                mass_c[ai, bi, ki] = float(m2[ka_set].sum() / m2.sum().clamp(min=1e-30))
                for kj in range(len(PYR_KB)):
                    kb_set = keep_b[(bi, kj)]
                    cnt_c[ai, bi, ki, kj] = float((ka_set & kb_set).sum() / kb_set.sum().clamp(min=1))
    S["pyr_contain_cnt"] = cnt_c  # share of b's top-kb tokens inside a's top-ka keep
    S["pyr_contain_mass"] = mass_c  # share of G_b^2 body mass inside a's top-ka keep
    # nested 50% @3 -> 10% @7 -> 3% @16 (each stage ranks only the survivors, by its own full-context G), vs one-shot
    def nested(stages):
        alive = torch.ones(nb, dtype=torch.bool, device=b.device)
        for L, rho in stages:
            sc = torch.where(alive, G[L, b].float(), torch.full_like(G[L, b].float(), -1.0))
            alive = doc_topk(sc, doc, rho) & alive
        return alive
    pyr_sets = {"nest_3_7_16": nested([(3, 0.5), (7, 0.1), (16, 0.03)]), "one_7_3pct": doc_topk(G[7, b], doc, 0.03),
                "one_16_3pct": doc_topk(G[16, b], doc, 0.03), "one_7_10pct": keeps["g7"],
                "random_3pct": doc_topk(rnd, doc, 0.03)}
    tgt_mass = {f"g{L}": G[L, b] ** 2 for L in (16, 24, 32)}
    tgt_mass["pw16"] = pw[16:, b].sum(0)
    S["pyr_nested"] = torch.tensor([[float(m[ks].sum() / m.sum().clamp(min=1e-30)) for m in tgt_mass.values()]
                                    for ks in pyr_sets.values()])  # rows PYR_SETS, cols PYR_TGTS
    # within-document concentration of what a 10% keep at block 7 DROPS (merge vs drop)
    dc = []
    for nm, m in (("g7", G[7, b] ** 2), ("pwdown7", pw[7:, b].sum(0))):
        dropped = ~doc_topk(G[7, b], doc, 0.1)
        tot_m = m.sum().clamp(min=1e-30)
        dshare = float(m[dropped].sum() / tot_m)
        # per document: share of the dropped mass held by the top 10% / top 1 of the dropped tokens; effective
        # number of dropped tokens (exp entropy) as a fraction of the dropped count; mass-weighted over documents
        t10 = t1 = eff = wsum = 0.0
        for d in range(int(doc.max()) + 1):
            md = m[(doc == d) & dropped]
            if md.numel() < 10 or float(md.sum()) <= 0:
                continue
            w = float(md.sum())
            srt = md.sort(descending=True).values
            p_ = srt / srt.sum()
            t10 += w * float(srt[:max(1, md.numel() // 10)].sum() / srt.sum())
            t1 += w * float(srt[0] / srt.sum())
            eff += w * float(torch.exp(-(p_ * torch.log(p_ + 1e-30)).sum()) / md.numel())
            wsum += w
        dc += [dshare, t10 / max(wsum, 1e-30), t1 / max(wsum, 1e-30), eff / max(wsum, 1e-30)]
    # adjacency: lag-1 correlation of log G7 between neighbouring body tokens of the same document
    lg = torch.log(G[7, b].float() + 1e-12)
    same = doc[1:] == doc[:-1]
    a1, a2 = lg[:-1][same], lg[1:][same]
    a1, a2 = a1 - a1.mean(), a2 - a2.mean()
    dc.append(float((a1 * a2).sum() / (a1.norm() * a2.norm() + 1e-12)))
    S["drop_conc"] = torch.tensor(dc)  # DROP_CONC_COLS
    return {k: v.detach().float().cpu().numpy() for k, v in S.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True, help="checkpoint label (dump dir name)")
    ap.add_argument("--shard", default=f"{W}/ds64/fast8k/shards/mt11s960")
    ap.add_argument("--tasks", default="contradiction,nq,outlier,oolong,strmatch,grouping,hotpotqa,reorder,xabsence,rerank,qdmatch_nq")
    ap.add_argument("--rows", type=int, default=32, help="rows per task")
    ap.add_argument("--row-offset", type=int, default=720, help="first row per task (mt11s960 rows >= 720 are unseen by "
                    "every mt11 run up to s720; mt11 rows are disjoint from every mt4 shard)")
    ap.add_argument("--example-rows", type=int, default=2, help="rows per task whose full token x layer arrays go to /results")
    ap.add_argument("--dump-dir", default=f"{W}/grad_composition/dumps", help="full per-row arrays (float16 npz); 'none' = off")
    ap.add_argument("--tokenizer", default=f"{W}/hf_tokenizers/Qwen3.5-0.8B-Base")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    tasks = a.tasks.split(",")
    rows = [r for r in load_rows(a.shard, tasks, a.rows, a.row_offset) if len(r["ids"]) <= a.max_len and r["lmask"].any()]
    log(f"{len(rows)} rows from {a.shard} offset {a.row_offset}; lengths {min(len(r['ids']) for r in rows)}-"
        f"{max(len(r['ids']) for r in rows)}")
    # which training shards contain the probe rows (should be none)
    seen = {}
    shards_dir = os.path.dirname(a.shard.rstrip("/"))
    for r in rows:
        r["seen_in"] = []
    for sh in ("mt11s240", "mt11s480", "mt11s720", "mt11s960", "mt4s488", "mt4s976", "mt4s1464", "mt4s1952"):
        p = os.path.join(shards_dir, sh)
        if not os.path.isdir(p) or os.path.abspath(p) == os.path.abspath(a.shard):
            continue
        names = shard_tasks(p)
        for t in tasks:
            if t in names:
                hs = row_hashes(p, t)
                seen[f"{sh}/{t}"] = sum(r["md5"] in hs for r in rows if r["task"] == t)
                for r in rows:
                    if r["task"] == t and r["md5"] in hs:
                        r["seen_in"].append(sh)
    log(f"probe rows found in training shards: {seen}")
    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    except Exception as exc:  # decoding is only for the example heatmaps
        log(f"no tokenizer ({exc}); example rows keep ids only")

    model = build_model(find_ckpt(a.ckpt))
    cap = Capture(model)
    pg = ParamGrads(model)
    task_pg = {}
    log(f"full-attention blocks: {cap.full}")
    out_base = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    dump = os.path.join(a.dump_dir, a.label) if a.dump_dir not in ("", "none") else ""
    if dump:
        os.makedirs(dump, exist_ok=True)
    stats, meta, nex = {}, [], {}
    t0 = time.time()
    for i, row in enumerate(rows):
        x = torch.from_numpy(row["ids"]).cuda()
        cats = categories(row, x)
        ce, o = full_pass(model, cap, row, cats)
        st = row_stats(o, cats, ce, model, row)
        st["pgrad"] = pg.take()
        if i + 1 == len(rows) or rows[i + 1]["task"] != row["task"]:
            task_pg[row["task"]] = pg.flush().tolist()
        for k, v in st.items():
            stats.setdefault(k, []).append(v)
        meta.append(dict(task=row["task"], k=row["k"], T=len(row["ids"]), ce=ce, n_body=int(cats["body"].sum()),
                         n_docs=int(cats["cid"].max()) + 1, n_gold_docs=len(row["gold"]),
                         n_answer=int(cats["pred"].sum()), seen_in=row["seen_in"]))
        arrs = {k: v.half().cpu().numpy() for k, v in o.items()}
        arrs.update(cat=cats["cat"].cpu().numpy().astype(np.int8), cid=cats["cid"].cpu().numpy().astype(np.int16),
                    ids=row["ids"].astype(np.int32))
        if dump:
            np.savez(os.path.join(dump, f"{row['task']}_{row['k']}.npz"), **arrs)
        if nex.get(row["task"], 0) < a.example_rows:
            nex[row["task"]] = nex.get(row["task"], 0) + 1
            np.savez_compressed(f"{out_base}_ex_{row['task']}_{row['k']}.npz", **arrs)
            if tok is not None:
                json.dump([tok.decode([int(t)]) for t in row["ids"]], open(f"{out_base}_ex_{row['task']}_{row['k']}.tok.json", "w"))
        n = i + 1
        if n in (1, 2, 5, 10) or n % 25 == 0 or n == len(rows):
            el = time.time() - t0
            log(f"row {n}/{len(rows)} ({row['task']} T={len(row['ids'])} CE={ce:.3f} abl dCE g7={st['abl_dce'][0]:+.3f} "
                f"rnd={st['abl_dce'][5]:+.3f}) {el:.0f}s, ETA {el / n * (len(rows) - n):.0f}s")
    np.savez_compressed(f"{out_base}_stats.npz", **{k: np.stack(v) for k, v in stats.items()})
    json.dump(dict(ckpt=a.ckpt, ckpt_resolved=find_ckpt(a.ckpt), label=a.label, shard=a.shard, row_offset=a.row_offset,
                   rows_per_task=a.rows, tasks=tasks, cats=CATS, ks=KS, splits=SPLITS, split_cols=
                   ["gL", "dh0L", "oracle", "random", "am_ge_L", "upstream_share"], abl_rules=ABL_RULES, abl_L=ABL_L,
                   abl_rho=ABL_RHO, full_blocks=cap.full, pyr=PYR, pyr_ka=PYR_KA,
                   pyr_kb=PYR_KB, pyr_sets=PYR_SETS, pyr_tgts=PYR_TGTS, drop_conc_cols=DROP_CONC_COLS,
                   task_summed_pgrad=task_pg, pgrad_layout="(33, 3): rows 0-31 blocks, 32 lm_head; cols mixer, ffn, norms", seen_in_training=seen, rows=meta, dump=dump),
              open(a.out, "w"), indent=1)
    log(f"wrote {a.out} and {out_base}_stats.npz ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
