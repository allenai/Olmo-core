"""Attention-edge gradient analysis on Qwen3.5-4B + LoRA r64 (one GPU per process, rows sharded over processes).

Per row (B = 1, unpadded, answer CE summed over answer tokens = the training loss up to normalisation):

Q1  one ordinary forward + backward with every softmax-attention layer routed through ``edge_attn`` (exact,
    streamed in query chunks). In the backward: per layer and head, (a) |dL/ds_ij|, (b) a_ij |<g_i, v_j>|,
    (c) a_ij, aggregated to (query segment, key segment) blocks, distance bins and raw-edge histograms.
Q2  the LoRA gradient of that pass is the reference. For each edge pattern:
    DETACH: extra backward passes over the SAME forward graph (retain_graph), keeping only the kept edges' terms of
            the attention gradient (forward = full attention);
    MASK:   a fresh forward + backward with the edges removed in both.
    Per pattern: kept edge fraction per layer (= attention FLOP fraction), per-block dot / norms against the
    reference (32 blocks), a CountSketch of the global LoRA gradient (for batch-level cosines), MASK loss.

    python probe_edges.py --base BASE/model_and_optim --adapter fresh|<arm_stepN.pt> --heldout X.npz \
        --set lc16k [--task contradiction] --rows 32 --shard 0 --nshards 4 --out DIR [--drop 0.05,0.5]
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
sys.path.insert(0, HERE)

import edge_attn as E  # noqa: E402
import rows as R  # noqa: E402

SKETCH_M = 1 << 15
FRACS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5)
ATTN_FRACS = (0.02, 0.05, 0.1, 0.25)
GRID = np.unique(np.concatenate([np.logspace(-4, 0, 41), np.array(FRACS)]))
MASS_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
ALL_PATTERNS = ([f"oracle{int(f * 100)}" for f in FRACS] + [f"random{int(f * 100)}" for f in FRACS]
                + [f"attn{int(f * 100)}" for f in ATTN_FRACS]
                + ["qa_only", "selfdoc_qa", "selfdoc_hubs_qa", "g4", "g4sym", "local256", "local1024", "local4096",
                   "selfdoc_blindhubs_qa", "topk4", "topk16"])


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------------------- model
def build_model(base: str, adapter: str, seed: int):
    import torch.distributed as dist
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.early_lora import install_early_lora
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(20000 + os.getpid() % 20000))
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320, attn_backend=AttentionBackendName("flash_2"))
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    t0 = time.time()
    load_model_and_optim_state(base, model)
    log(f"loaded base {base} in {time.time() - t0:.0f}s")
    torch.manual_seed(seed)
    mods = install_early_lora(model, n_blocks=len(model.blocks), rank=64, scale=32.0, check_calls=False)
    info = {"adapter": adapter, "lora_modules": len(mods)}
    if adapter != "fresh":
        sd = torch.load(adapter, map_location="cpu")
        meta = sd.pop("__meta__", {})
        named = dict(model.named_parameters())
        loaded, skipped = 0, []
        for k, v in sd.items():
            name = k[len("model."):] if k.startswith("model.") else k
            if name in named:
                with torch.no_grad():
                    named[name].copy_(v.to(named[name].dtype))
                loaded += 1
            else:
                skipped.append(name)
        n_lora = sum(1 for m in mods for _ in (m.lora_A, m.lora_B))
        n_lora_loaded = sum(1 for k in sd if "lora_" in k and (k[6:] if k.startswith("model.") else k) in named)
        if n_lora_loaded != n_lora:
            raise RuntimeError(f"adapter {adapter}: {n_lora_loaded} of {n_lora} LoRA tensors loaded")
        info.update(loaded=loaded, skipped=skipped[:20], n_skipped=len(skipped), meta_ckpt=meta.get("ckpt"))
        log(f"adapter {adapter}: loaded {loaded} tensors ({n_lora_loaded} LoRA), skipped {len(skipped)}")
    model = model.cuda().to(torch.bfloat16).eval()
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    blocks = []
    for key, block in model.blocks.items():
        blocks.append([p for n, p in block.named_parameters() if "lora_" in n])
    try:
        from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

        freeze_fla_length_autotune()
    except Exception as exc:  # noqa: BLE001
        log(f"fla autotune freeze skipped: {exc}")
    return model, blocks, info


def row_loss(model, ids: torch.Tensor, lm: torch.Tensor):
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    pred = torch.nonzero(lm[1:], as_tuple=True)[0]
    tgt = ids[pred + 1]
    _, _, abk, pbk, lmk = model._prepare_inputs(ids[None], None, logits_to_keep=pred[None])
    h = model.embeddings(ids[None])
    with set_checkpoint_early_stop(False):
        for key, block in model.blocks.items():
            kw = {**abk, **pbk.get(int(key), {})}
            h = checkpoint(lambda hh, b=block, k=kw: b(hh, **k), h, use_reentrant=False)
    return F.cross_entropy(model.lm_head(h, **lmk)[0].float(), tgt, reduction="sum"), int(pred.numel())


# ------------------------------------------------------------------------------------------------------- grads
class GradTap:
    def __init__(self, blocks, m=SKETCH_M, seed=1234):
        self.blocks = blocks
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.h, self.sg = [], []
        for ps in blocks:
            n = sum(p.numel() for p in ps)
            self.h.append(torch.randint(0, m, (n,), generator=g, device="cuda", dtype=torch.int64).to(torch.int32))
            self.sg.append((torch.randint(0, 2, (n,), generator=g, device="cuda", dtype=torch.int8) * 2 - 1))
        self.m = m
        self.n_params = sum(p.numel() for ps in blocks for p in ps)

    def zero(self):
        for ps in self.blocks:
            for p in ps:
                p.grad = None

    def flat(self):
        out = []
        for ps in self.blocks:
            out.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten()
                                  for p in ps]))
        return out

    def sketch(self, flat):
        s = torch.zeros(self.m, device="cuda", dtype=torch.float64)
        for f, h, sg in zip(flat, self.h, self.sg):
            s.index_add_(0, h.long(), (f * sg).double())
        return s.float()


def compare(ref, cur):
    dots = [float(a @ b) for a, b in zip(ref, cur)]
    n1 = [float(a @ a) for a in ref]
    n2 = [float(b @ b) for b in cur]
    return dots, n1, n2


# ------------------------------------------------------------------------------------------------------- Q1
def summarize_q1(stats: E.EdgeStats, info, layers, save_blocks: bool):
    cls = info["cls"]
    Eb = R.block_edges(info["seg_len"])
    cat = R.category_index(cls)
    ncat = len(R.CLS_NAMES) * len(R.KEY_NAMES)
    cat_t = torch.tensor(cat, device="cuda").flatten()
    Et = torch.tensor(Eb, device="cuda")
    out = {"cat_edges": np.bincount(cat.flatten(), weights=Eb.flatten(), minlength=ncat).tolist(), "layers": {}}
    Zsum = {}
    pooled = {q: torch.zeros_like(Et) for q in E.EdgeStats.QTY}
    for L in layers:
        lo = {}
        Zsum[L] = {}
        for q in E.EdgeStats.QTY:
            Zh = stats.blocks[L][q]
            if Zh.dim() == 3:
                H = Zh.shape[0]
                catm = torch.zeros(H, ncat, device="cuda", dtype=torch.float64).index_add_(1, cat_t, Zh.reshape(H, -1))
                Z = Zh.sum(0)
            else:
                catm = torch.zeros(1, ncat, device="cuda", dtype=torch.float64).index_add_(1, cat_t, Zh.reshape(1, -1))
                Z = Zh
            Zsum[L][q] = Z
            pooled[q] += Z / Z.sum().clamp_min(1e-300)
            lo[q] = {"cat_mass_per_head": catm.cpu().numpy().tolist(), "total": float(Z.sum()),
                     "conc": concentration(Z, Et), "dist_mass": stats.dist[L][q].view(-1, E.NDIST).cpu().numpy().tolist()}
            if stats.hist:
                lo[q]["raw_hist"] = stats.hist[L][q].cpu().numpy().tolist()
        out["layers"][str(L)] = lo
    out["pooled_conc"] = {q: concentration(pooled[q], Et) for q in E.EdgeStats.QTY}
    out["dist_cnt"] = stats.dist_cnt.cpu().numpy().tolist()
    blocks_np = None
    if save_blocks:
        blocks_np = {q: torch.stack([Zsum[L][q] for L in layers]).float().cpu().numpy() for q in E.EdgeStats.QTY}
    return out, Zsum, Et, blocks_np


def concentration(Z: torch.Tensor, Et: torch.Tensor):
    """greedy by density (mass per edge): mass captured at edge-fraction grid points and the edge fraction needed for
    each mass level; greedy by mass: fraction of (causal, non-empty) blocks needed for each mass level."""
    valid = Et > 0
    z, e = Z[valid].double(), Et[valid].double()
    tot, etot = float(z.sum()), float(e.sum())
    if tot <= 0:
        return None
    o = torch.argsort(z / e, descending=True)
    ce = (torch.cumsum(e[o], 0) / etot).cpu().numpy()
    cm = (torch.cumsum(z[o], 0) / tot).cpu().numpy()
    ce0, cm0 = np.concatenate([[0.0], ce]), np.concatenate([[0.0], cm])
    mass_at = np.interp(GRID, ce0, cm0).tolist()
    edge_for = [float(np.interp(l, cm0, ce0)) for l in MASS_LEVELS]
    zs = torch.sort(z, descending=True).values
    cmb = (torch.cumsum(zs, 0) / tot).cpu().numpy()
    nb = len(cmb)
    blocks_for = [float((np.searchsorted(cmb, l) + 1) / nb) for l in MASS_LEVELS]
    return {"mass_at_edge_frac": mass_at, "edge_frac_for_mass": edge_for, "block_frac_for_mass": blocks_for,
            "n_blocks": nb}


# ------------------------------------------------------------------------------------------------------- patterns
def greedy_K(order: torch.Tensor, Et: torch.Tensor, frac: float, K0: torch.Tensor) -> torch.Tensor:
    """add blocks in ``order`` (flat indices) on top of K0 while the kept causal edges stay <= frac of all."""
    K = K0.clone()
    etot = float(Et.sum())
    used = float(Et[K0].sum())
    flatE = Et.flatten()
    o = order[~K0.flatten()[order] & (flatE[order] > 0)]
    ce = torch.cumsum(flatE[o], 0) + used
    take = o[ce <= frac * etot]
    K.view(-1)[take] = True
    return K


def build_patterns(names, info, Zsum, Et, layers, seed, dev="cuda"):
    nseg, cls = info["nseg"], torch.tensor(info["cls"], device=dev)
    seg = info["seg_t"]
    T = info["T"]
    qa_tok = (cls[seg] == R.QUES) | (cls[seg] == R.ANS)
    pre_tok = cls[seg] == R.PRE
    is_doc = (cls == R.GOLD) | (cls == R.DIST)
    K0 = torch.zeros(nseg, nseg, dtype=torch.bool, device=dev)
    K0[:, 0] = True  # preamble keys always kept (attention sink; ~20-130 tokens)
    eye = torch.eye(nseg, dtype=torch.bool, device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    gold = torch.nonzero(cls == R.GOLD).flatten()
    dist = torch.nonzero(cls == R.DIST).flatten()
    ng = max(1, gold.numel())
    decoys = dist[torch.randperm(dist.numel(), generator=g)[:ng].to(dev)]
    hubs = torch.zeros(nseg, dtype=torch.bool, device=dev)
    hubs[gold] = True
    hubs[decoys] = True
    # gold-blind hubs: top documents by answer/question-row attention mass, summed over layers
    qa_seg = (cls == R.QUES) | (cls == R.ANS)
    att_tot = sum(Zsum[L]["attn"] for L in layers)
    doc_score = att_tot[qa_seg].sum(0).clone()
    doc_score[~is_doc] = -1
    bhubs = torch.zeros(nseg, dtype=torch.bool, device=dev)
    bhubs[torch.topk(doc_score, min(2 * ng, int(is_doc.sum()))).indices] = True
    out = []
    for name in names:
        kw = dict(keyall=pre_tok)
        if name.startswith("oracle") or name.startswith("attn") or name.startswith("random"):
            frac = int(name.lstrip("oraclentdm")) / 100.0
            Ks = {}
            for L in layers:
                if name.startswith("oracle"):
                    sal = Zsum[L]["logit"] / Zsum[L]["logit"].sum() + Zsum[L]["value"] / Zsum[L]["value"].sum()
                elif name.startswith("attn"):
                    sal = Zsum[L]["attn"]
                else:
                    sal = None
                if sal is None:
                    order = torch.randperm(nseg * nseg, generator=g).to(dev)
                else:
                    dens = torch.where(Et > 0, sal / Et.clamp_min(1), torch.full_like(sal, -1.0))
                    order = torch.argsort(dens.flatten(), descending=True)
                Ks[L] = greedy_K(order, Et, frac, K0)
                if name.startswith("random"):
                    Ks = {LL: Ks[L] for LL in layers}  # one random pattern shared by every layer
                    break
            p = E.Pattern(name, seg, K=Ks, **kw)
        elif name == "qa_only":
            p = E.Pattern(name, seg, K=None, rowall=qa_tok, **kw)
        elif name == "selfdoc_qa":
            p = E.Pattern(name, seg, K=eye, rowall=qa_tok, **kw)
        elif name in ("selfdoc_hubs_qa", "selfdoc_blindhubs_qa"):
            hb = hubs if name == "selfdoc_hubs_qa" else bhubs
            K = eye | (is_doc[:, None] & hb[None, :])
            p = E.Pattern(name, seg, K=K, rowall=qa_tok, **kw)
        elif name in ("g4", "g4sym"):
            G = 4
            grp = torch.full((nseg,), -1, dtype=torch.long)
            rest = torch.nonzero(is_doc & ~hubs).flatten().cpu()
            grp[rest] = torch.randint(0, G, (rest.numel(),), generator=g)
            grp = grp.to(dev)
            hub_all = hubs.clone()
            hub_all[0] = True  # preamble is a hub
            K = (grp[:, None] == grp[None, :]) & (grp[:, None] >= 0)  # group -> own group
            K |= (grp[:, None] >= 0) & hub_all[None, :]  # group -> hubs
            K |= hub_all[:, None] & hub_all[None, :]  # hub -> hubs
            if name == "g4sym":
                hg = torch.randint(0, G, (nseg,), generator=g).to(dev)
                K |= hubs[:, None] & (grp[None, :] == hg[:, None])  # hub doc -> one random group
            p = E.Pattern(name, seg, K=K | eye, rowall=qa_tok, **kw)
        elif name.startswith("local"):
            p = E.Pattern(name, seg, K=None, rowall=qa_tok, window=int(name[5:]), **kw)
        elif name.startswith("topk"):
            k = int(name[4:])
            Ks = {}
            for L in layers:
                A = Zsum[L]["attn"].clone()
                A[~is_doc[:, None].expand_as(A)] = -1
                A[:, ~is_doc] = -1
                A[eye] = -1
                A[Et <= 0] = -1
                kk = min(k, nseg)
                top = torch.topk(A, kk, dim=1)
                K = torch.zeros_like(eye)
                K.scatter_(1, top.indices, top.values > 0)
                Ks[L] = K | eye
            p = E.Pattern(name, seg, K=Ks, rowall=qa_tok, **kw)
        else:
            raise ValueError(name)
        out.append(p)
    return out


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="fresh")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--task", default="")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--row-offset", type=int, default=0)
    ap.add_argument("--drop", default="", help="lo,hi: gold-document drop of each row (keep share of distractors)")
    ap.add_argument("--patterns", default="all")
    ap.add_argument("--modes", default="detach,mask")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-head-max-seg", type=int, default=1500, help="per-head block matrices when nseg <= this")
    ap.add_argument("--raw-hist-max-T", type=int, default=70000)
    ap.add_argument("--save-blocks-rows", type=int, default=2)
    ap.add_argument("--chunk-elems", type=int, default=2 ** 27)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    E.STATE.chunk_elems = a.chunk_elems
    model, blocks, minfo = build_model(a.base, a.adapter, a.seed)
    layers = E.patch_model(model, E.STATE)
    log(f"patched attention layers {layers}")
    tap = GradTap(blocks)
    log(f"LoRA params {tap.n_params / 1e6:.1f}M in {len(blocks)} blocks")
    rows = R.load_set(a.heldout, a.set, a.task or None, a.rows, a.row_offset)
    names = ALL_PATTERNS if a.patterns == "all" else a.patterns.split(",")
    modes = [m for m in a.modes.split(",") if m]
    my = [(i, r) for i, r in enumerate(rows) if i % a.nshards == a.shard]
    log(f"{len(rows)} rows in {a.set}/{a.task or '*'}; shard {a.shard}/{a.nshards}: {len(my)} rows; "
        f"{len(names)} patterns x {modes}")
    fj = open(os.path.join(a.out, f"rows_s{a.shard}.jsonl"), "a")
    done = set()
    for fn in os.listdir(a.out):
        if fn.startswith("row_") and fn.endswith(".pt"):
            done.add(int(fn[4:-3]))
    for n_done, (i, row) in enumerate(my):
        if i in done:
            continue
        t0 = time.time()
        if a.drop:
            lo, hi = map(float, a.drop.split(","))
            row = R.gold_drop(row, lo, hi, seed=1000 + row["row"])
        info = R.segments(row)
        info["seg_t"] = torch.tensor(info["seg"], device="cuda")
        ids = torch.tensor(row["ids"], device="cuda")
        lm = torch.tensor(row["lmask"], device="cuda")
        T = info["T"]
        stats = E.EdgeStats(seg=info["seg_t"], nseg=info["nseg"], per_head=info["nseg"] <= a.per_head_max_seg,
                            raw_hist=T <= a.raw_hist_max_T)
        # ---- reference: full forward, exact backward with Q1 accounting (detach mode, no pattern = full)
        E.STATE.mode, E.STATE.pattern, E.STATE.stats = "detach", None, stats
        tap.zero()
        loss, ntok = row_loss(model, ids, lm)
        loss.backward(retain_graph=True)
        E.STATE.stats = None
        ref = tap.flat()
        ref_sk = tap.sketch(ref)
        t_ref = time.time() - t0
        q1, Zsum, Et, blocks_np = summarize_q1(stats, info, layers, n_done < a.save_blocks_rows)
        del stats
        rec = dict(row=row["row"], set=a.set, task=row["task"], T=T, nd=info["nd"], n_gold=len(info["gold"]),
                   numbered_ok=info["numbered_ok"], ce_sum=float(loss), n_ans=ntok, adapter=a.adapter,
                   drop_r=row.get("drop_r"), orig_T=row.get("orig_T"), seg_len=info["seg_len"].tolist(),
                   cls=info["cls"].tolist(), t_ref=t_ref, q1=q1, q2=[])
        sk = {"ref": ref_sk.cpu()}
        if a.selftest:
            tap.zero()
            loss.backward(retain_graph=True)
            again = tap.flat()
            d, n1, n2 = compare(ref, again)
            exact = all(torch.equal(x, y) for x, y in zip(ref, again))
            log(f"selftest: second full backward over the same graph identical={exact} cos={sum(d) / math.sqrt(sum(n1) * sum(n2)):.8f}")
            rec["selftest_repeat_exact"] = exact
        pats = build_patterns(names, info, Zsum, Et, layers, seed=a.seed * 7919 + row["row"])
        fracs = {p.name: p.kept_fraction(T, layers) for p in pats}
        if "detach" in modes:
            E.STATE.mode = "detach"
            for j, p in enumerate(pats):
                E.STATE.pattern = p
                tap.zero()
                loss.backward(retain_graph=j < len(pats) - 1)
                cur = tap.flat()
                d, n1, n2 = compare(ref, cur)
                rec["q2"].append(dict(pattern=p.name, mode="detach", frac=float(np.mean(list(fracs[p.name].values()))),
                                      frac_layer=list(fracs[p.name].values()), dots=d, n1=n1, n2=n2))
                sk[f"detach/{p.name}"] = tap.sketch(cur).cpu()
                del cur
        del loss
        E.STATE.pattern = None
        if "mask" in modes:
            for p in pats:
                E.STATE.mode, E.STATE.pattern = "mask", p
                tap.zero()
                l2, _ = row_loss(model, ids, lm)
                l2.backward()
                cur = tap.flat()
                d, n1, n2 = compare(ref, cur)
                rec["q2"].append(dict(pattern=p.name, mode="mask", frac=float(np.mean(list(fracs[p.name].values()))),
                                      frac_layer=list(fracs[p.name].values()), dots=d, n1=n1, n2=n2, ce_sum=float(l2)))
                sk[f"mask/{p.name}"] = tap.sketch(cur).cpu()
                del cur, l2
        E.STATE.mode, E.STATE.pattern = "full", None
        if a.selftest:  # unpatched flash model loss vs the patched one
            saved = {}
            from olmo_core.nn.attention import Attention

            for key, block in model.blocks.items():
                for m in block.modules():
                    if isinstance(m, Attention) and "sdpa" in m.__dict__:
                        saved[m] = m.__dict__.pop("sdpa")
            with torch.no_grad():
                lf, _ = row_loss(model, ids, lm)
            for m, f in saved.items():
                m.sdpa = f
            log(f"selftest: CE flash {float(lf) / ntok:.5f} vs edge_attn {rec['ce_sum'] / ntok:.5f} per token")
            rec["selftest_flash_ce_sum"] = float(lf)
        torch.save({"sketch": sk, "blocks": blocks_np}, os.path.join(a.out, f"row_{i}.pt"))
        rec["t_row"] = time.time() - t0
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        best = sorted([(q["mode"], q["pattern"], q["frac"], sum(q["dots"]) / math.sqrt(max(1e-300, sum(q["n1"]) * sum(q["n2"]))))
                       for q in rec["q2"]], key=lambda x: x[2])
        log(f"row {i} (src {row['row']}) T={T} nd={info['nd']} gold={len(info['gold'])} CE/tok={rec['ce_sum'] / ntok:.4f} "
            f"ref {t_ref:.0f}s total {rec['t_row']:.0f}s | " + " ".join(f"{m[0]}:{n}@{f:.3f}={c:.3f}" for m, n, f, c in best[:60]))
        torch.cuda.empty_cache()
    log("DONE")


if __name__ == "__main__":
    main()
