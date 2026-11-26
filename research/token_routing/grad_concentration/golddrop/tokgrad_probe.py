"""Token-level gradient mass under the gold-drop + recipe training format (sanity check, 2026-11-25).

Same gradient quantity as ``../length_sweep/sweep_probe.py`` / ``../probe.py`` (reused by import): frozen merged
checkpoint, teacher-forced answer CE (no-cot), every nn.Linear in blocks 7-31; token t's term in a linear is rank 1,
energy e_t = |delta_t|^2 |x_t|^2 (the "gradient mass" below); blocks 0-6 run without autograd. Energy only (no
TensorSketch / cos pass). Per row, three formats:
  A  whole row, dense forward (the earlier dense analysis's view)
  B  the row as gold drop trains on it: every gold claim + a share r ~ U(0.05, 0.5) of the distractor documents kept
     (whole documents, markers included; olmo_core.nn.gold_doc_drop), the rest removed before block 0; no router
  C  B + the recipe router (olmo_core.nn.online_token_router, stages 7:0.2,16:0.07, doc_share 1, explore 0.1): tokens
     cut at blocks 7 and 16 by the real router code (mid_compact), with the router's fitted state from the checkpoint's
     trainer state when present, else refit here on separate held-out rows (--fit-rows).
B and C use the same drop draw. Energies are mapped back to the original row's token positions. FLOPs per row follow
the trainer's FLOP meter (dense LoRA: frozen base 2/3 + adapters; router: blocks at their own lengths, 2/3 + adapters).

    python tokgrad_probe.py --ckpt <merged dir> --label gdd256-s11496 --rows rows.npz --sets lc16k --draws 2 \
        --router-state <train_rank0.pt or ''> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
GC = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(GC, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, GC)
sys.path.insert(0, os.path.join(GC, "length_sweep"))
import probe as P  # noqa: E402
import sweep_probe as SP  # noqa: E402

SPLIT = 7
FRACS = (0.5, 0.9, 0.99)
ROUTER_FLAGS = dict(keep_frac=0.2, explore_frac=0.1, doc_share=1.0, dense_every=0, drop_layer=7,
                    stages="7:0.2,16:0.07", early_lora=64, early_lora_blocks=32, warmup_random=True)


def log(m):
    print(f"[tokgrad] {m}", flush=True)


def build_router(d_model, seed=0):
    from olmo_core.nn.online_token_router import OnlineRouterConfig, OnlineTokenRouter

    cfg = OnlineRouterConfig(seed=seed, **ROUTER_FLAGS)
    return OnlineTokenRouter(cfg, doc_start_id=P.DOC_START, doc_end_id=P.DOC_END, eos_id=P.EOS, d_model=d_model)


def load_router_state(router, path):
    """OnlineRouterCallback.state_dict ('chain') from a trainer state file; True if found."""
    if not path or not os.path.exists(path):
        return False
    s = torch.load(path, map_location="cpu", weights_only=False)
    cbs = s.get("callbacks", s)
    st = None
    for k, v in (cbs.items() if isinstance(cbs, dict) else []):
        if isinstance(v, dict) and v.get("chain"):
            st = v
            log(f"router state from callback {k!r}")
    if st is None:
        return False
    chain = router.chain()
    assert len(chain) == len(st["chain"]), (len(chain), len(st["chain"]))
    for r, d in zip(chain, st["chain"]):
        for k, v in d.items():
            setattr(r, k, v.cuda() if torch.is_tensor(v) else ({kk: vv.cuda() if torch.is_tensor(vv) else vv for kk, vv in v.items()} if isinstance(v, dict) else v))
        r._pending = []
    log(f"router stages {len(chain)}, step {chain[0].step}, w {None if chain[0].w is None else tuple(chain[0].w.shape)}")
    return True


def gold_drop(ids, gold, rng, lo=0.05, hi=0.5):
    """(kept original positions, r): every gold document + round(r * n_distractors) random distractors; dropped
    documents removed marker to marker (olmo_core.nn.gold_doc_drop, per-document mode, no whole rows)."""
    spans = P_spans(ids)
    nd = len(spans)
    g = set(int(x) for x in gold)
    ng = [d for d in range(nd) if d not in g]
    rng.shuffle(ng)
    r = rng.uniform(lo, hi)
    keep_docs = g | set(ng[:int(round(r * len(ng)))])
    rm = np.zeros(len(ids), bool)
    for d, (s, e) in enumerate(spans):
        if d not in keep_docs:
            rm[s:e] = True
    return np.flatnonzero(~rm), r


def P_spans(ids):
    st = np.flatnonzero(ids == P.DOC_START)
    en = np.flatnonzero(ids == P.DOC_END)
    assert len(st) == len(en)
    return list(zip(st.tolist(), (en + 1).tolist()))


def row_forward(model, ids, lmask, router=None):
    """sweep_probe.row_loss with optional router mid-stack cuts. Returns (loss, n_answer, frames) where frames[b] =
    positions (in the input row) of the tokens block b ran on (None = all)."""
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    d0 = model._gsl_dev0
    x = torch.from_numpy(ids).to(d0)
    lm = torch.from_numpy(lmask).to(d0)
    pred = torch.nonzero(lm[1:], as_tuple=True)[0]
    tgt = x[pred + 1]
    T = x.numel()
    _, _, abk, pbk, _ = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    stages = {}
    if router is not None:
        lab = torch.full((1, T), -100, dtype=torch.long, device=d0)
        lab[0, pred] = tgt
        cid = build_chunk_ids_from_tokens(x[None], P.DOC_START, P.DOC_END, P.EOS, mode="chunked")
        router.select(model, x[None], cid, lab, -100)
        stages = {r.cfg.drop_layer: r for r in router.chain()}
    pos = torch.arange(T, device=d0)  # current frame -> input-row positions
    frames = {}
    kw = dict(abk)
    with torch.no_grad():
        h = model.embeddings(x[None])
    with set_checkpoint_early_stop(False):
        for key, block in model.blocks.items():
            b = int(key)
            if b in stages:
                st = stages[b]
                h, kw = st.mid_compact(model, h, kw)
                order, valid, _ = st._mid_gather
                pos = pos[order[0][valid[0]]]
            frames[b] = pos.clone()
            k2 = {**kw, **pbk.get(b, {})}
            if b < SPLIT:
                with torch.no_grad():
                    h = block(h, **k2)
                if b == SPLIT - 1:
                    h = h.detach().requires_grad_(True)
            else:
                h = checkpoint(lambda hh, bb=block, k=k2: bb(hh, **k), h, use_reentrant=False)
    # answer positions in the final frame (non-body: always kept)
    inv = torch.full((T,), -1, dtype=torch.long, device=d0)
    inv[pos] = torch.arange(pos.numel(), device=d0)
    pc = inv[pred]
    ok = pc >= 0
    _, _, _, _, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pc[ok][None])
    logits = model.lm_head(h, **lmk)[0].float()
    loss = F.cross_entropy(logits, tgt[ok], reduction="sum")
    return loss, int(ok.sum()), int((~ok).sum()), frames


def flops_row(model, L, L7=None, L16=None):
    """Per-row FLOPs as the trainer's meter charges them. A/B (dense LoRA, --lora-rank 64, all blocks): dense count
    + per block (-1/3 f_b + adapters). C (router, early LoRA on all 32 blocks): blocks at their own lengths, each
    2/3 f_b + adapters, head at the final length."""
    blocks = list(model.blocks.values())
    ab = []
    for blk in blocks:
        ab.append(sum(6.0 * 64 * (m.in_features + m.out_features) for m in blk.modules() if isinstance(m, torch.nn.Linear)))
    head = lambda n: float(model.lm_head.num_flops_per_token(int(n)))  # noqa: E731
    f = lambda i, n: float(blocks[i].num_flops_per_token(int(n)))  # noqa: E731
    if L7 is None:
        return L * (sum(f(i, L) * (2 / 3) + ab[i] for i in range(len(blocks))) + head(L))
    tot = L * sum(f(i, L) * (2 / 3) + ab[i] for i in range(SPLIT))
    tot += L7 * sum(f(i, L7) * (2 / 3) + ab[i] for i in range(SPLIT, 16))
    tot += L16 * (sum(f(i, L16) * (2 / 3) + ab[i] for i in range(16, len(blocks))) + head(L16))
    return tot


def n_for(e, fracs=FRACS):
    """fraction of the entries of e (>= 0) needed for each share of its sum (largest first)."""
    s = np.sort(e.astype(np.float64))[::-1]
    if s.size == 0 or s.sum() <= 0:
        return [float("nan")] * len(fracs)
    c = np.cumsum(s) / s.sum()
    return [float((np.searchsorted(c, q) + 1) / s.size) for q in fracs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--sets", default="lc16k")
    ap.add_argument("--n", type=int, default=16, help="measured rows per set")
    ap.add_argument("--fit-rows", type=int, default=0, help="rows after the measured ones used to refit the router "
                    "when no router state is found (lc16k only)")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--formats", default="A,B,C")
    ap.add_argument("--router-state", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.rows)
    gold = json.loads(str(z["gold"]))
    offs, sets = z["offsets"], z["sets"]
    rows = {}
    for i in range(len(sets)):
        rows.setdefault(str(sets[i]), []).append(dict(ids=z["ids"][offs[i]:offs[i + 1]].astype(np.int64),
                                                      lmask=z["mask"][offs[i]:offs[i + 1]].astype(bool), gold=gold[i]))
    model = SP.build_model(P.find_ckpt(a.ckpt), [0])
    lins = [ln for ln in P.list_linears(model) if ln.block >= SPLIT]
    tap = SP.Tap(lins, 1024, 8192)
    tap.mode = "energy"
    fmts = a.formats.split(",")
    router = None
    if "C" in fmts:
        router = build_router(int(model.embeddings.weight.shape[1]))
        src = "checkpoint"
        if not load_router_state(router, a.router_state):
            src = "refit"
            fit = rows["lc16k"][a.n:a.n + a.fit_rows]
            assert fit, "no router state and no --fit-rows"
            log(f"no router state: refitting on {len(fit)} held-out rows (not the measured ones), 2 drop draws each")
            rng = random.Random(123)
            step = 0
            for rep in range(2):
                for row in fit:
                    step += 1
                    for r in router.chain():
                        r.begin_step(step, model=model)
                    kp, _ = gold_drop(row["ids"], row["gold"], rng)
                    loss, _, _, _ = row_forward(model, row["ids"][kp], row["lmask"][kp], router)
                    tap.E = {}
                    tap.backward(loss)
            for r in router.chain():
                r.begin_step(step + 1, model=model)
            log(f"router refit done: step {router.step}, w {tuple(router.w.shape) if router.w is not None else None}")
        router_src = src

    def run(ids, lmask, use_router):
        tap.fired, tap.E = 0, {}
        loss, nans, nlost, frames = row_forward(model, ids, lmask, router if use_router else None)
        tap.backward(loss)
        assert tap.fired == len(lins), (tap.fired, len(lins))
        T = len(ids)
        M = np.zeros((32 - SPLIT, T), np.float32)  # per block energy in the input-row frame
        for ln in lins:
            fr = frames[ln.block].cpu().numpy()
            e = tap.E[ln.idx].numpy()
            assert e.size == fr.size, (ln.name, e.size, fr.size)
            M[ln.block - SPLIT, fr] += e
        return M, float(loss) / max(nans, 1), nlost, {b: frames[b].cpu().numpy() for b in (0, 7, 16)}

    out_rows = []
    t0 = time.time()
    rng = random.Random(20261125)
    done = 0
    for sname in a.sets.split(","):
        for ri, row in enumerate(rows[sname][:a.n]):
            ids, lm = row["ids"], row["lmask"]
            T = len(ids)
            bm, cid = P.body_mask(torch.from_numpy(ids).cuda())
            body = bm.cpu().numpy()
            cid = cid.cpu().numpy()
            gold_tok = np.isin(cid, np.array(row["gold"])) & (cid >= 0)
            rec = dict(set=sname, row=ri, T=T, n_body=int(body.sum()), n_gold_tok=int(gold_tok.sum()), draws=[])
            store = {}
            MA, ce, _, _ = run(ids, lm, False)
            sA = MA.sum(0)
            rec.update(ceA=ce, flopsA=flops_row(model, T))
            store["A"] = MA.astype(np.float16)
            for d in range(a.draws):
                kp, r = gold_drop(ids, row["gold"], rng)
                dr = dict(r=r, L=len(kp))
                MB, ceB, _, _ = run(ids[kp], lm[kp], False)
                dr.update(ceB=ceB, flopsB=flops_row(model, len(kp)))
                MBo = np.zeros((MB.shape[0], T), np.float32)
                MBo[:, kp] = MB
                store[f"B{d}"] = MBo.astype(np.float16)
                if "C" in fmts:
                    MC, ceC, nlost, lens = run(ids[kp], lm[kp], True)
                    MCo = np.zeros((MC.shape[0], T), np.float32)
                    MCo[:, kp] = MC
                    store[f"C{d}"] = MCo.astype(np.float16)
                    # router keep masks (original frame) at blocks 7 and 16: tokens with any energy there
                    k7 = np.zeros(T, bool)
                    k16 = np.zeros(T, bool)
                    k7[kp[lens[7]]] = True  # tokens blocks 7-15 ran on
                    k16[kp[lens[16]]] = True
                    dr.update(ceC=ceC, ans_lost=nlost, L7=len(lens[7]), L16=len(lens[16]),
                              flopsC=flops_row(model, len(kp), len(lens[7]), len(lens[16])))
                    store[f"k7_{d}"] = k7
                    store[f"k16_{d}"] = k16
                store[f"kp{d}"] = kp
                rec["draws"].append(dr)
            np.savez_compressed(os.path.join(a.out, f"{sname}_r{ri:03d}.npz"), body=body, gold_tok=gold_tok, cid=cid,
                                **store)
            out_rows.append(rec)
            done += 1
            el = time.time() - t0
            if done in (1, 2, 5, 10) or done % 8 == 0:
                log(f"{sname} row {ri} T={T} ceA={ce:.3f} draws={[(round(x['r'], 2), x['L'], x.get('L7'), x.get('L16')) for x in rec['draws']]} "
                    f"{el:.0f}s; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    json.dump(dict(label=a.label, ckpt=a.ckpt, router_src=router_src if router else None, rows=out_rows,
                   seconds=time.time() - t0), open(os.path.join(a.out, "rows.json"), "w"), indent=1)
    log(f"TOKGRAD_DONE {a.out} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
