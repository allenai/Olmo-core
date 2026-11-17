"""Per-token x per-layer gradient maps on frozen checkpoints (Prasann, 2026-11-04; experiment tokdrop_lr).

For every row (the same unseen fresh-pool rows as the support / routed probes: rows >= 7,808 of <task>_fresh, md5-checked
against the training shards, 4 rows per task per 16-row batch) one dense forward + backward of the summed answer CE, and
per (block l, token t):
  E[l, t]  the token's weight-gradient energy in block l: sum over the block's linears of ||delta_t||^2 ||x_t||^2 (the
           rank-1 attribution energy the 0.95 oracle ranks by; down<L> = E[L:].sum(0))
  A[l, t]  the activation-gradient norm ||dL/dh_l[t]|| at block l's INPUT (the residual stream after l blocks)
With --router (the recipe checkpoint): the ridge emulation of the online router (routed_probe.py's ``rtr``: features at
the split from the dense pass, target row-centred log ||dL/dh||, 2-fold fit within the batch + decayed history, 10%
exploration, per-document allocation) picks the recipe's nested 7:0.2 -> 16:0.07 keep set (stage 2 scored on the
routed h16 in-pass), and the same row is run again ROUTED (dropped tokens gathered out, original RoPE positions); E / A
of that pass are stored with NaN for dropped cells.

Writes <out-dir>/<label>/r<idx>.npz (log10 E, log10 A, token types, doc ids, ...) and rows.json. gradmap_plot.py draws.

    python research/token_routing/grad_concentration/gradmap.py --ckpt <dir> --label f1952 --out-dir /data/.../gradmap
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe as P  # noqa: E402
import routed_probe as RP  # noqa: E402

NB = 32
RECIPE = ((7, 0.2), (16, 0.07))


def log(m):
    print(f"[gradmap] {m}", flush=True)


def lg(t):
    return torch.log10(t.float().clamp(min=1e-30)).cpu().numpy().astype(np.float32)


def run(model, tap, r, stages=None):
    """One forward + backward; returns CE, E (NB, T) and A (NB, T) over ORIGINAL positions (NaN where dropped),
    the captured dense inputs at blocks 7 / 16 (with grads) and the final column map."""
    T = r["T"]
    tap.mode = "energy"
    tap.E = torch.zeros(NB, T, device="cuda")
    tap.fired = 0
    loss, capt, cur = RP.run_row(model, tap, r, stages=stages, cap=tuple(range(NB)), ckpt=True)
    tap.backward(loss)
    if tap.fired != len(tap.lins):
        raise SystemExit(f"hooks fired {tap.fired} for {len(tap.lins)} linears")
    E = tap.E
    A = torch.full((NB, T), float("nan"), device="cuda")
    kept = torch.zeros(NB, T, dtype=torch.bool, device="cuda")
    for b in range(NB):
        c = tap.cur[b]
        A[b, c] = capt[b].grad[0].float().norm(dim=-1)
        kept[b, c] = True
    E = torch.where(kept, E, torch.full_like(E, float("nan")))
    tap.E = None
    keep_capt = {L: capt[L] for L in (7, 16)} if not stages else {}
    return float(loss), E, A, keep_capt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", default="contradiction,nq,outlier,oolong")
    ap.add_argument("--rows", type=int, default=128, help="rows per task")
    ap.add_argument("--row-offset", type=int, default=7808)
    ap.add_argument("--shards", default=P.SHARDS)
    ap.add_argument("--seen-shards", default="mt4f7808,mt4s1952")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--router", action="store_true", help="recipe: ridge router keep set + routed pass")
    ap.add_argument("--lora-scale", type=float, default=32.0)
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = os.path.join(a.out_dir, a.label)
    os.makedirs(out, exist_ok=True)
    model = P.build_model(P.find_ckpt(a.ckpt), a.attn_backend)
    lins = P.list_linears(model)
    RP.merge_lora(model, lins, P.find_ckpt(a.ckpt), a.lora_scale)
    tap = RP.Tap(lins, 8)
    tasks = a.tasks.split(",")
    seen = [os.path.join(a.shards, s) for s in a.seen_shards.split(",") if s]
    rbt, _ = P.load_rows(a.shards, tasks, a.rows, a.row_offset, 12288, seen)
    per = a.batch // len(tasks)
    nbatch = min(len(v) for v in rbt.values()) // per
    gen = torch.Generator(device="cpu").manual_seed(1234)
    ridge = {}
    meta = []
    t0 = time.time()
    n_tot = nbatch * a.batch
    idx = 0
    for J in range(nbatch):
        batch = [(t, rr) for t in tasks for rr in rbt[t][J * per:(J + 1) * per]]
        R = [RP.prep_row(rr, t) for t, rr in batch]
        rec = []
        for i, ((t, rr), r) in enumerate(zip(batch, R)):
            ce, E, A, capt = run(model, tap, r)
            item = dict(idx=idx + i, batch=J, task=t, k=int(rr["k"]), md5=rr["md5"], T=r["T"],
                        n_body=int(r["body"].sum()), n_answer=int(r["pred"].numel()), gold_docs=r["gold_docs"],
                        numbered_ok=r["numbered_ok"], ce=ce, ce_tok=ce / max(1, int(r["pred"].numel())))
            arrs = dict(E=lg(E), A=lg(A), ty=r["ty"].cpu().numpy().astype(np.int8),
                        cid=r["cid"].cpu().numpy().astype(np.int16), qm=r["qm"].cpu().numpy(),
                        body=r["body"].cpu().numpy(), ids=rr["ids"].astype(np.int32))
            if a.router:
                s = RP.scalar_feats(r)
                r["s"] = s
                for L in (7, 16):
                    hn = RP.rms(capt[L][0].detach())
                    g = capt[L].grad[0].float()
                    q = RP.query_vec(hn, r["qm"])
                    r[f"hn{L}"], r[f"q{L}"] = hn.half(), q
                    tg = torch.log(g.norm(dim=-1) + 1e-12)
                    bidx = torch.nonzero(r["body"], as_tuple=True)[0]
                    smp = bidx[torch.randperm(bidx.numel(), generator=gen)[:RP.SAMPLES].to(bidx.device)]
                    r[f"fit{L}"] = (RP.feats(hn[smp], q, s[smp]), tg[smp] - tg[smp].mean())
            del capt
            rec.append((item, arrs))
            n = idx + i + 1
            if n in (1, 2, 5, 10) or n % 32 == 0:
                el = time.time() - t0
                log(f"[dense] row {n}/{n_tot} {t} T={r['T']} CE/tok={item['ce_tok']:.3f} gold={r['gold_docs']} "
                    f"{el:.0f}s, ETA {el / n * (n_tot - n) * (2 if a.router else 1):.0f}s; "
                    f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        if a.router:
            nf = R[0]["s"].shape[1] + 1 + 2 * R[0]["hn7"].shape[1]
            for L in (7, 16):
                ridge.setdefault(L, RP.Ridge(nf))
                st = []
                for fo in (0, 1):
                    parts = [ridge[L].stats(*r[f"fit{L}"]) for i, r in enumerate(R) if i % 2 == fo]
                    st.append((sum(p_[0] for p_ in parts), sum(p_[1] for p_ in parts)))
                for fo in (0, 1):
                    w = ridge[L].solve([st[1 - fo]])
                    for i, r in enumerate(R):
                        if i % 2 == fo:
                            r[f"w{L}"] = w
                            r[f"rscore{L}"] = RP.feats(r[f"hn{L}"].float(), r[f"q{L}"], r["s"]) @ w
                ridge[L].push(st)
            (L1, k1), (L2, k2) = RECIPE
            f2 = k2 / k1
            for i, r in enumerate(R):
                item, arrs = rec[i]
                X, y = r[f"fit{L1}"]
                body = r["body"]
                m1 = RP.doc_topk(r[f"rscore{L1}"], body, r["cid"], k1, prio=RP.explore_mask(body, k1, gen))
                hold = {}

                def fn2(h, cur, r=r):
                    hn = RP.rms(h[0])
                    bc = r["body"][cur]
                    q = RP.query_vec(hn, r["qm"][cur])
                    sc = RP.feats(hn, q, r["s"][cur]) @ r[f"w{L2}"]
                    kk = RP.doc_topk(sc, bc, r["cid"][cur], f2, prio=RP.explore_mask(bc, f2, gen))
                    m2 = torch.zeros_like(r["body"])
                    m2[cur[kk]] = True
                    hold["m2"] = m2
                    return kk | ~bc
                stages = {L1: (lambda h, cur, m1=m1, body=body: (m1 | ~body)[cur]), L2: fn2}
                ce_r, E_r, A_r, _ = run(model, tap, r, stages=stages)
                for L in (7, 16):
                    Xf, yf = r[f"fit{L}"]
                    p = Xf @ r[f"w{L}"]
                    item[f"router_spearman{L}"] = float(torch.corrcoef(torch.stack(
                        [p.argsort().argsort().float(), yf.argsort().argsort().float()]))[0, 1])
                item.update(ce_routed=ce_r, ce_routed_tok=ce_r / max(1, item["n_answer"]),
                            keep7=float(m1.sum() / max(1, int(body.sum()))),
                            keep16=float(hold["m2"].sum() / max(1, int(body.sum()))))
                arrs.update(E_routed=lg(E_r), A_routed=lg(A_r), keep7=m1.cpu().numpy(), keep16=hold["m2"].cpu().numpy(),
                            rscore7=r["rscore7"].float().cpu().numpy())
                for L in (7, 16):
                    for k_ in ("hn", "q", "fit", "w", "rscore"):
                        r.pop(f"{k_}{L}", None)
                n = idx + i + 1
                if n in (1, 2, 5, 10) or n % 32 == 0:
                    log(f"[routed] row {n}/{n_tot} CE/tok dense {item['ce_tok']:.3f} routed {item['ce_routed_tok']:.3f} "
                        f"keep7 {item['keep7']:.3f} keep16 {item['keep16']:.3f} rho7 {item['router_spearman7']:.2f}")
        for item, arrs in rec:
            np.savez_compressed(os.path.join(out, f"r{item['idx']:04d}.npz"), **arrs)
            meta.append(item)
        idx += len(R)
        json.dump(dict(label=a.label, ckpt=a.ckpt, eval_size_rows=len(meta), rows=meta),
                  open(os.path.join(out, "rows.json"), "w"))
    log(f"done {len(meta)} rows in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
