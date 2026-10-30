"""Waterfall from routed_probe.py sketches: oracle attribution -> true routed gradient -> expected gradient -> bias
accumulation -> Adam -> router vs oracle, per selection, plus where the lost update lives (task / block / token type).

    python research/token_routing/grad_concentration/analyze_gap.py --gap-dir <dir> --labels f1952,s488 --out x.json

All quantities are inner products of TensorSketches, so they come from per-group Gram matrices of the saved vectors:
  base vectors per batch b (16 rows = one optimizer step): D_b, A_{s,b}, R_{s,b} (and Adam-preconditioned copies)
  E_{-b} = sum_{b' != b} D_b'  (leave-one-out expected gradient, (B-1) x 16 rows)
  debiased expected cosine  cos(E[X], E[Y]) from the cross-batch terms only: <E X, E Y> ~ sum_{b != b'} <X_b, Y_b'> / B(B-1)
  residuals: Res_true = D - R (what routing loses), Res_attr = D - A (what the attribution partial sum drops)
Errors: mean +- sd / sqrt(B) over batches (batches are independent row sets).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NG = 64


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deq(z, key):
    x = torch.from_numpy(z[key].astype(np.float32))
    sc = torch.from_numpy(z[key + "_sc"].astype(np.float32))
    return x * sc[..., None]


def ms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan"), float("nan")]
    return [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")]


def gram(vecs):
    """vecs (N, NG, m) -> per-group Gram (NG, N, N) float64."""
    v = vecs.to(dev()).transpose(0, 1).double()  # (NG, N, m)
    return (v @ v.transpose(1, 2)).cpu()


def cos_from(G, i, j, groups):
    g = G[groups].sum(0)
    return float(g[i, j] / math.sqrt(max(float(g[i, i] * g[j, j]), 1e-300)))


class Space:
    """Gram of base vectors + linear combinations. base index = (batch, name)."""

    def __init__(self, G, names, B):
        self.G, self.names, self.B = G, names, B  # G (NG, B*V, B*V)
        self.V = len(names)
        self.ix = {n: i for i, n in enumerate(names)}

    def vec(self, b, n):
        c = torch.zeros(self.B * self.V, dtype=torch.float64)
        c[b * self.V + self.ix[n]] = 1.0
        return c

    def ip(self, c1, c2, groups):
        g = self.G[groups].sum(0)
        return float(c1 @ g @ c2)

    def cos(self, c1, c2, groups):
        g = self.G[groups].sum(0)
        a, b, c = float(c1 @ g @ c2), float(c1 @ g @ c1), float(c2 @ g @ c2)
        return a / math.sqrt(max(b * c, 1e-300))


def blocks_for(sp, Gs, X, Y):
    """(B, B) matrix of <X_b, Y_c> in the summed-group Gram Gs."""
    ix = torch.tensor([b * sp.V + sp.ix[X] for b in range(sp.B)])
    iy = torch.tensor([b * sp.V + sp.ix[Y] for b in range(sp.B)])
    return Gs[ix][:, iy].numpy()


def metrics_from_blocks(M, B):
    """M[(X, Y)] = (B, B) inner products for X, Y in D, A, R. Everything below is algebra on these."""
    def blk(X, Y):
        return M[(X, Y)] if (X, Y) in M else M[(Y, X)].T

    # residuals as combinations: Rt = D - R, Ra = D - A
    comb = {"D": {"D": 1}, "A": {"A": 1}, "R": {"R": 1}, "Rt": {"D": 1, "R": -1}, "Ra": {"D": 1, "A": -1}}

    def C(X, Y):
        out = 0
        for x, a in comb[X].items():
            for y, b in comb[Y].items():
                out = out + a * b * blk(x, y)
        return out

    def dg(X, Y):
        return np.diag(C(X, Y))

    def cosd(X, Y):  # per-batch cos(X_b, Y_b)
        return dg(X, Y) / np.sqrt(np.maximum(dg(X, X) * dg(Y, Y), 1e-300))

    v = {}
    v["attr_cos"] = ms(cosd("A", "D"))
    v["routed_cos"] = ms(cosd("R", "D"))
    v["routed_vs_attr_cos"] = ms(cosd("R", "A"))
    v["attr_relnorm"] = ms(np.sqrt(dg("A", "A") / dg("D", "D")))
    v["routed_relnorm"] = ms(np.sqrt(dg("R", "R") / dg("D", "D")))
    if B > 2:
        DD = C("D", "D")
        EE = np.array([DD.sum() - 2 * DD[b].sum() + DD[b, b] for b in range(B)])  # ||E_-b||^2

        def withE(X):  # <X_b, E_-b>
            XD = C(X, "D")
            return XD.sum(1) - np.diag(XD)
        cE = {X: withE(X) / np.sqrt(dg(X, X) * EE) for X in ("D", "A", "R")}
        v["dense_cos_E"] = ms(cE["D"])
        v["attr_cos_E"] = ms(cE["A"])
        v["routed_cos_E"] = ms(cE["R"])
        v["attr_cos_E_ratio"] = ms(cE["A"] / cE["D"])
        v["routed_cos_E_ratio"] = ms(cE["R"] / cE["D"])
        v["attr_proj_E"] = ms(withE("A") / withE("D"))
        v["routed_proj_E"] = ms(withE("R") / withE("D"))
        off = ~np.eye(B, dtype=bool)

        def xcos(X, Y):
            XY = C(X, Y)
            return (XY / np.sqrt(np.outer(dg(X, X), dg(Y, Y))))[off]
        v["xbatch_dense"] = ms(xcos("D", "D"))
        v["xbatch_attr"] = ms(xcos("A", "D"))
        v["xbatch_routed"] = ms(xcos("R", "D"))

        def eip(X, Y):
            return float(C(X, Y)[off].mean())

        def ecos(X, Y):
            xy, xx, yy = eip(X, Y), eip(X, X), eip(Y, Y)
            return xy / math.sqrt(xx * yy) if xx > 0 and yy > 0 else float("nan")
        v["exp_cos_attr"] = ecos("A", "D")
        v["exp_cos_routed"] = ecos("R", "D")
        v["exp_norm_ratio_routed"] = math.sqrt(eip("R", "R") / eip("D", "D")) if eip("R", "R") > 0 else float("nan")
        v["exp_norm_ratio_attr"] = math.sqrt(eip("A", "A") / eip("D", "D")) if eip("A", "A") > 0 else float("nan")
        v["exp_proj_routed"] = eip("R", "D") / eip("D", "D")
        v["exp_proj_attr"] = eip("A", "D") / eip("D", "D")
        for nm_, X in (("dense", "D"), ("res_true", "Rt"), ("res_attr", "Ra"), ("routed", "R")):
            v[f"bias_share_{nm_}"] = eip(X, X) / float(dg(X, X).mean())
            v[f"pair_cos_{nm_}"] = ms(xcos(X, X))
        # routed-vs-routed consistency (does routing give a consistent direction of its own?)
        adj = [(b, b + 1) for b in range(B - 1)]
        for nm_, X in (("dense", "D"), ("routed", "R"), ("attr", "A")):
            XX = C(X, X)
            v[f"adj_cos_{nm_}"] = ms([XX[b, c] / math.sqrt(XX[b, b] * XX[c, c]) for b, c in adj])
        v["exp_cos_res_true_vs_D"] = ecos("Rt", "D")
        v["exp_cos_res_attr_vs_D"] = ecos("Ra", "D")
        # bias-only cosine bound: the routed update with its mean residual removed would be the dense one, so the
        # per-step cos ceiling set by the BIAS alone is cos(E[D], E[D] - E[Rt]) = exp_cos_routed (same thing); the
        # noise-free share of the residual is bias_share_res_true.
        rng = np.random.default_rng(0)
        for X, key in (("R", "cum_cos_routed"), ("A", "cum_cos_attr")):
            cum = np.zeros(B)
            XD, XX = C(X, "D"), C(X, X)
            for _ in range(16):
                p = rng.permutation(B)
                for n in range(1, B + 1):
                    q = p[:n]
                    cum[n - 1] += XD[np.ix_(q, q)].sum() / math.sqrt(XX[np.ix_(q, q)].sum() * DD[np.ix_(q, q)].sum()) / 16
            v[key] = cum.tolist()
    return v


def analyze_space(sp, sels, groups_of, B):
    out = {}
    for s in sels:
        o = {}
        for view, gr in groups_of(s).items():
            Gs = sp.G[gr].sum(0)
            M = {}
            for X, nX in (("D", "D"), ("A", f"A_{s}"), ("R", f"R_{s}")):
                for Y, nY in (("D", "D"), ("A", f"A_{s}"), ("R", f"R_{s}")):
                    M[(X, Y)] = blocks_for(sp, Gs, nX, nY)
            o[view] = metrics_from_blocks(M, B)
        out[s] = o
    return out


def load_label(gdir):
    files = sorted(glob.glob(os.path.join(gdir, "b*.npz")))
    metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(gdir, "meta_shard*.json")))]
    return files, metas


def split_of(sel):
    if sel.endswith("recipe"):
        return 7
    return int(sel.split("_L")[1].split("_")[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gap-dir", required=True)
    ap.add_argument("--labels", default="f1952")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-types", action="store_true")
    a = ap.parse_args()
    allout = {}
    for lab in a.labels.split(","):
        files, metas = load_label(os.path.join(a.gap_dir, lab))
        if not files:
            print(f"{lab}: no batches")
            continue
        meta = metas[0]
        sels, focus, tasks = meta["sels"], meta["focus"], meta["tasks"]
        rows = [r for m in metas for r in m["rows"]]
        B = len(files)
        print(f"== {lab}: {B} batches ({len(rows)} rows), {len(sels)} selections")
        zs = [np.load(f) for f in files]
        names = ["D"] + [f"A_{s}" for s in sels] + [f"R_{s}" for s in sels]
        res = {"label": lab, "n_batches": B, "rows": len(rows), "eval_size_rows": len(rows),
               "small_flag": (f"eval_size={len(rows)} rows < 500" if len(rows) < 500 else "")}

        def groups_of(s):
            L = split_of(s)
            return {"whole": list(range(NG)), "late": list(range(2 * L, NG)), "mixer": list(range(0, NG, 2)),
                    "mlp": list(range(1, NG, 2))}

        for space, pre in (("raw", "tot_"), ("adam", "adam_tot_")):
            if pre + "D" not in zs[0]:
                continue
            vecs = torch.stack([torch.stack([deq(z, pre + n) for n in names]) for z in zs])  # (B, V, NG, m)
            Bv, V = vecs.shape[:2]
            G = gram(vecs.reshape(Bv * V, NG, -1))
            sp = Space(G, names, B)
            res[space] = analyze_space(sp, sels, groups_of, B)
            # per-group share of the lost update (true and attribution residual), mean over batches
            pg = {}
            for s in sels:
                sh_t, sh_a, sh_D, rel = [], [], [], []
                for g in range(NG):
                    Gg = sp.G[g]
                    M = {(X, Y): blocks_for(sp, Gg, nX, nY) for X, nX in (("D", "D"), ("A", f"A_{s}"), ("R", f"R_{s}"))
                         for Y, nY in (("D", "D"), ("A", f"A_{s}"), ("R", f"R_{s}"))}
                    dd, rr_, aa = np.diag(M[("D", "D")]), np.diag(M[("R", "R")]), np.diag(M[("A", "A")])
                    dr, da = np.diag(M[("D", "R")]), np.diag(M[("D", "A")])
                    sh_t.append(dd - 2 * dr + rr_)
                    sh_a.append(dd - 2 * da + aa)
                    sh_D.append(dd)
                sh_t, sh_a, sh_D = np.array(sh_t), np.array(sh_a), np.array(sh_D)  # (NG, B)
                pg[s] = {"res_true": (sh_t / sh_t.sum(0)).mean(1).tolist(), "res_attr": (sh_a / sh_a.sum(0)).mean(1).tolist(),
                         "dense": (sh_D / sh_D.sum(0)).mean(1).tolist(),
                         "res_true_over_dense_norm2": (sh_t / np.maximum(sh_D, 1e-300)).mean(1).tolist()}
            res[space + "_per_group"] = pg
            del vecs, G, sp
        # ---------------------------------------------------------------- per task
        if "tsk_D" in zs[0]:
            pt = {}
            Dt = torch.stack([deq(z, "tsk_D") for z in zs])  # (B, nT, NG, mf)
            nT = Dt.shape[1]
            for s in sels:
                At = torch.stack([deq(z, f"tsk_A_{s}") for z in zs])
                Rt = torch.stack([deq(z, f"tsk_R_{s}") for z in zs])
                vecs = torch.stack([Dt, At, Rt], 2)  # (B, nT, 3, NG, mf)
                G = gram(vecs.reshape(-1, NG, vecs.shape[-1])).sum(0)  # whole model (N, N)

                def ix(b, t, k):
                    return (b * nT + t) * 3 + k
                o = {}
                for t in range(nT):
                    cr, ca, share, cRE, cDE, eprojR = [], [], [], [], [], []
                    for b in range(B):
                        d, aa, r = ix(b, t, 0), ix(b, t, 1), ix(b, t, 2)
                        cr.append(float(G[r, d] / math.sqrt(G[r, r] * G[d, d])))
                        ca.append(float(G[aa, d] / math.sqrt(G[aa, aa] * G[d, d])))
                        # share of the batch's lost update (D - R summed over tasks) that this task carries
                        tot = [ix(b, u, 0) for u in range(nT)], [ix(b, u, 2) for u in range(nT)]
                        c_all = torch.zeros(G.shape[0], dtype=torch.float64)
                        c_all[tot[0]] = 1
                        c_all[tot[1]] = -1
                        c_t = torch.zeros_like(c_all)
                        c_t[d], c_t[r] = 1, -1
                        share.append(float(c_t @ G @ c_all / (c_all @ G @ c_all)))
                        e = torch.zeros_like(c_all)
                        for c2 in range(B):
                            if c2 != b:
                                e[ix(c2, t, 0)] = 1
                        cd = torch.zeros_like(c_all)
                        cd[d] = 1
                        cr_ = torch.zeros_like(c_all)
                        cr_[r] = 1
                        cDE.append(float(cd @ G @ e / math.sqrt(float(cd @ G @ cd) * float(e @ G @ e))))
                        cRE.append(float(cr_ @ G @ e / math.sqrt(float(cr_ @ G @ cr_) * float(e @ G @ e))))
                        eprojR.append(float(cr_ @ G @ e / (cd @ G @ e)))
                    o[tasks[t]] = {"routed_cos": ms(cr), "attr_cos": ms(ca), "lost_share": ms(share),
                                   "dense_cos_Etask": ms(cDE), "routed_cos_Etask": ms(cRE),
                                   "routed_proj_Etask": ms(eprojR)}
                pt[s] = o
                del G, vecs
            res["per_task"] = pt
        # ---------------------------------------------------------------- per token type (focus selections)
        if not a.no_types and f"typ_A8_{focus[0]}" in zs[0]:
            pty = {}
            TY = meta["types"]
            for s in focus:
                A8 = torch.stack([deq(z, f"typ_A8_{s}") for z in zs])  # (B, nT, NG, 8, mf)
                R4 = torch.stack([deq(z, f"typ_R_{s}") for z in zs])  # (B, nT, NG, 4, mf)
                A8 = A8.sum(1).transpose(1, 2)  # (B, 8, NG, mf)  tasks summed
                R4 = R4.sum(1).transpose(1, 2)  # (B, 4, NG, mf)
                vecs = torch.cat([A8, R4], 1)  # (B, 12, NG, mf)
                Gg = gram(vecs.reshape(-1, NG, vecs.shape[-1]))  # (NG, B*12, B*12)
                G = Gg.sum(0)
                n = B * 12
                comps = {}
                rows_c = []
                for b in range(B):
                    o = b * 12
                    c = {}
                    for k, tn in enumerate(TY):
                        dr = torch.zeros(n, dtype=torch.float64)
                        dr[o + 4 + k] = 1  # dense terms of this type's DROPPED tokens
                        sh = torch.zeros(n, dtype=torch.float64)
                        sh[o + k] = 1
                        sh[o + 8 + k] = -1  # kept tokens of this type: dense term - routed term (forward removal)
                        c[f"dropped_{tn}"] = dr
                        c[f"shift_{tn}"] = sh
                    lost = sum(c.values())
                    Dv = torch.zeros(n, dtype=torch.float64)
                    Dv[o:o + 8] = 1
                    E = torch.zeros(n, dtype=torch.float64)
                    for b2 in range(B):
                        if b2 != b:
                            E[b2 * 12:b2 * 12 + 8] = 1
                    ll = float(lost @ G @ lost)
                    de = float(Dv @ G @ E)
                    row = {}
                    for k_, cv in c.items():
                        row[k_] = (float(cv @ G @ lost) / ll, float(cv @ G @ E) / de if B > 1 else float("nan"))
                    row["_lost_rel"] = math.sqrt(ll / float(Dv @ G @ Dv))
                    rows_c.append(row)
                for k_ in rows_c[0]:
                    if k_.startswith("_"):
                        comps[k_] = ms([r[k_] for r in rows_c])
                    else:
                        comps[k_] = {"share_of_lost_norm2": ms([r[k_][0] for r in rows_c]),
                                     "lost_progress_on_E": ms([r[k_][1] for r in rows_c])}
                pty[s] = comps
                del Gg, G, vecs
            res["per_type"] = pty
        # ---------------------------------------------------------------- row-level CE / keep / router stats
        ce = {}
        for s in sels:
            per = {}
            for t in tasks:
                rr = [r for r in rows if r["task"] == t and s in r["ce_routed"]]
                if rr:
                    per[t] = {"ce_dense": ms([r["ce_dense"] / r["n_answer"] for r in rr]),
                              "ce_routed": ms([r["ce_routed"][s] / r["n_answer"] for r in rr]),
                              "keep": {L: float(np.mean([r["keep_frac"][s][L] for r in rr]))
                                       for L in rr[0]["keep_frac"][s]}}
            ce[s] = per
        res["ce"] = ce
        res["gold_parse"] = {t: {"rows": len([r for r in rows if r["task"] == t]),
                                 "with_gold": len([r for r in rows if r["task"] == t and r["gold_docs"]]),
                                 "numbered_ok": len([r for r in rows if r["task"] == t and r["numbered_ok"]])}
                             for t in tasks}
        res["router_spearman"] = {L: ms([r["router_spearman"][L] for r in rows]) for L in ("7", "16")}
        res["energy_type"] = {L: np.mean([r["energy_type"][L] for r in rows], 0).tolist() for L in ("7", "16")}
        res["type_token_share"] = (np.sum([r["type_tokens"] for r in rows], 0) / sum(r["T"] for r in rows)).tolist()
        res["rtr2_overlap"] = ms([r["rtr2_overlap"] for r in rows if r.get("rtr2_overlap") is not None])
        ex = [e for m in metas for e in m["exact"]]
        res["exact_check"] = ex
        ad = {k: d for k, d in (meta.get("adam_diag") or {}).items() if not k.startswith("_")}
        res["adam_source"] = (meta.get("adam_diag") or {}).get("_source")
        res["adam_diag_summary"] = ({"r2_log_median": float(np.nanmedian([d["r2_log"] for d in ad.values()])),
                                     "cos_D_median": float(np.nanmedian([d["cos_D"] for d in ad.values()])),
                                     "cos_D_min": float(np.nanmin([d["cos_D"] for d in ad.values()]))}
                                    if ad and np.isfinite([d["cos_D"] for d in ad.values()]).any() else None)
        allout[lab] = res
        # ---------------------------------------------------------------- print the waterfall
        for space in ("raw", "adam"):
            if space not in res:
                continue
            print(f"  [{space}] sel: attr_cos | routed_cos | cosE ratio attr/routed | exp_cos attr/routed | "
                  f"bias_share res_true / dense | pair cos R-R / D-D | bias_share routed / dense")
            for s in sels:
                v = res[space][s]["whole"]

                def f(k):
                    x = v.get(k)
                    return "nan" if x is None else (f"{x[0]:.3f}" if isinstance(x, list) else f"{x:.3f}")
                print(f"    {s:18s} {f('attr_cos')} | {f('routed_cos')} | {f('attr_cos_E_ratio')}/{f('routed_cos_E_ratio')}"
                      f" | {f('exp_cos_attr')}/{f('exp_cos_routed')} | {f('bias_share_res_true')}/{f('bias_share_dense')}"
                      f" | {f('pair_cos_routed')}/{f('pair_cos_dense')} | {f('bias_share_routed')}/{f('bias_share_dense')}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(allout, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
