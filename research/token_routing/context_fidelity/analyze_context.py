"""Tables from context_probe.py outputs: per checkpoint x selection x variant, gradient fidelity (per-step cos to the
dense step, debiased expected cos and norm ratio, raw and Adam), answer CE vs dense, hidden-state error at blocks
15 / 23 / 31, analytic step cost and best-case per-step speedup, and the share of the plain-drop gap each variant closes.

    python research/token_routing/context_fidelity/analyze_context.py --dir /data/prasann/ctxfid/sketches \
        --labels f1952,recipe --out results/ctx.json

Expected quantities use cross-batch terms only (batches are independent row sets): <E X, E Y> ~ mean_{b != c} <X_b, Y_c>;
errors on them are leave-one-batch-out jackknife SEs. Per-step quantities: mean +- SE over batches.
CE: mean over rows of (summed answer CE / answer tokens), as analyze_gap.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch

TASKS = ("contradiction", "nq", "outlier", "oolong")


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


def gram(V, dev):
    """V (N, dim) -> (N, N) float64."""
    v = V.to(dev).double()
    return (v @ v.T).cpu().numpy()


def exp_stats(G, iD, iR):
    """iD, iR: index arrays over batches. Returns per-step cos, exp cos, exp norm ratio, proj, with jackknife SEs."""
    B = len(iD)

    def est(sel):
        d, r = iD[sel], iR[sel]
        off = ~np.eye(len(sel), dtype=bool)
        dd = G[np.ix_(d, d)][off].mean()
        rr = G[np.ix_(r, r)][off].mean()
        rd = G[np.ix_(r, d)][off].mean()
        c = rd / math.sqrt(dd * rr) if dd > 0 and rr > 0 else float("nan")
        nr = math.sqrt(rr / dd) if dd > 0 and rr > 0 else float("nan")
        return c, nr, rd / dd if dd > 0 else float("nan")

    full = est(np.arange(B))
    jk = np.array([est(np.delete(np.arange(B), b)) for b in range(B)])
    se = [float(math.sqrt((B - 1) / B * np.nansum((jk[:, k] - np.nanmean(jk[:, k])) ** 2))) for k in range(3)]
    step = np.array([G[iR[b], iD[b]] / math.sqrt(G[iR[b], iR[b]] * G[iD[b], iD[b]]) for b in range(B)])
    step_nr = np.array([math.sqrt(G[iR[b], iR[b]] / G[iD[b], iD[b]]) for b in range(B)])
    return dict(step_cos=ms(step), step_norm_ratio=ms(step_nr), exp_cos=[full[0], se[0]],
                exp_norm_ratio=[full[1], se[1]], exp_proj=[full[2], se[2]])


def load_label(d, lab):
    files = sorted(glob.glob(os.path.join(d, lab, "b*.npz")))
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, lab, "meta_shard*.json")))]
    rows = [r for m in metas for r in m["rows"]]
    # keep only batches that have both sketches and rows
    have = {int(os.path.basename(f)[1:4]) for f in files}
    rows = [r for r in rows if r["batch"] in have]
    done = {r["batch"] for r in rows}
    files = [f for f in files if int(os.path.basename(f)[1:4]) in done]
    return files, rows, metas[0]


def analyze(d, lab, dev):
    files, rows, meta = load_label(d, lab)
    sels, vars_ = meta["selectors"], meta["variants"]
    combos = [f"{s}_{v}" for s in sels for v in vars_]
    B = len(files)
    out = dict(label=lab, n_batches=B, rows=len(rows), rows_per_task={t: sum(r["task"] == t for r in rows) for t in TASKS},
               corr_diag=meta.get("corr_diag", {}), lora_merged=meta.get("lora_merged"), raw={}, adam={}, per_task={})
    zs = [np.load(f) for f in files]
    combos = [c for c in combos if f"tot_R_{c}" in zs[0].files]
    names = ["D"] + combos
    for space, pref in (("raw", "tot_"), ("adam", "adam_tot_")):
        if f"{pref}D" not in zs[0].files:
            continue
        V = torch.stack([deq(z, f"{pref}D" if n == "D" else f"{pref}R_{n}").reshape(-1) for z in zs for n in names])
        G = gram(V, dev)
        del V
        nN = len(names)
        iD = np.arange(B) * nN
        for j, c in enumerate(combos):
            out[space][c] = exp_stats(G, iD, iD + j + 1)
        dd = G[np.ix_(iD, iD)][~np.eye(B, dtype=bool)]
        out[space]["_dense"] = dict(pair_cos_DD=ms(dd / np.sqrt(np.outer(np.diag(G)[iD], np.diag(G)[iD])
                                                            [~np.eye(B, dtype=bool)])))
    # per task (folded sketches, raw)
    for ti, t in enumerate(TASKS):
        V = torch.stack([deq(z, "tsk_D" if n == "D" else f"tsk_R_{n}")[ti].reshape(-1) for z in zs for n in names])
        G = gram(V, dev)
        nN = len(names)
        iD = np.arange(B) * nN
        out["per_task"][t] = {c: exp_stats(G, iD, iD + j + 1) for j, c in enumerate(combos)}
    # CE, hidden errors, flops
    ce, err, fl, misc = {}, {}, {}, {}
    for c in combos:
        ce[c], err[c] = {}, {}
        for t in list(TASKS) + ["pooled"]:
            rr = [r for r in rows if (t == "pooled" or r["task"] == t) and c in r["v"]]
            ce[c][t] = dict(dense=ms([r["ce_dense"] / r["n_answer"] for r in rr]),
                            routed=ms([r["v"][c]["ce"] / r["n_answer"] for r in rr]))
            err[c][t] = {bk: dict(kept=ms([r["v"][c]["err"][bk][0] for r in rr]),
                                  answer=ms([r["v"][c]["err"][bk][1] for r in rr])) for bk in ("15", "23", "31")}
        rr = [r for r in rows if c in r["v"]]
        fl[c] = dict(frac=sum(r["v"][c]["flops"] for r in rr) / sum(r["flops_dense"] for r in rr))
        fl[c]["speedup"] = 1.0 / fl[c]["frac"]
        misc[c] = dict(keep_body=ms([r["v"][c]["keep_body"] for r in rr]), kept_tokens=ms([r["v"][c]["n_kept"] for r in rr]),
                       n_standin=ms([r["v"][c]["n_standin"] for r in rr]), T=ms([r["T"] for r in rr]))
    out.update(ce=ce, err=err, flops=fl, misc=misc,
               router_spearman=ms([r["router_spearman"] for r in rows]))
    # gap closed vs plain drop, same selector
    gap = {}
    for s in sels:
        base = f"{s}_drop"
        if base not in combos:
            continue
        for c in combos:
            if not c.startswith(s + "_"):
                continue
            g = {}
            for space in ("raw", "adam"):
                if c in out.get(space, {}):
                    x0, x = out[space][base]["exp_cos"][0], out[space][c]["exp_cos"][0]
                    g[f"exp_cos_{space}"] = (x - x0) / (1 - x0) if x0 < 1 else float("nan")
                    x0, x = out[space][base]["step_cos"][0], out[space][c]["step_cos"][0]
                    g[f"step_cos_{space}"] = (x - x0) / (1 - x0) if x0 < 1 else float("nan")
            for bk in ("15", "23", "31"):
                e0, e = err[base]["pooled"][bk]["kept"][0], err[c]["pooled"][bk]["kept"][0]
                g[f"err{bk}"] = 1 - e / e0 if e0 > 0 else float("nan")
            c0, cv, cd = ce[base]["pooled"]["routed"][0], ce[c]["pooled"]["routed"][0], ce[c]["pooled"]["dense"][0]
            g["ce"] = (c0 - cv) / (c0 - cd) if c0 != cd else float("nan")
            gap[c] = g
    out["gap_closed"] = gap
    return out


def fmt_table(res):
    lines = []
    for lab, o in res.items():
        lines.append(f"\n## {lab}: {o['rows']} rows ({o['rows_per_task']}), {o['n_batches']} batches of 16")
        lines.append("| sel_variant | step cos raw | step cos Adam | exp cos raw | exp cos Adam | exp norm ratio | "
                     "err15 | err23 | err31 | CE (dense) | cost | speedup | gap closed: exp cos raw / Adam / err31 / CE |")
        lines.append("|" + "---|" * 14)
        for c in o["flops"]:
            r, ad = o["raw"].get(c, {}), o["adam"].get(c, {})
            f = lambda v: f"{v[0]:.3f}±{v[1]:.3f}" if v else "-"  # noqa: E731
            e = o["err"][c]["pooled"]
            ce = o["ce"][c]["pooled"]
            g = o["gap_closed"].get(c, {})
            lines.append(f"| {c} | {f(r.get('step_cos'))} | {f(ad.get('step_cos'))} | {f(r.get('exp_cos'))} | "
                         f"{f(ad.get('exp_cos'))} | {f(r.get('exp_norm_ratio'))} | {e['15']['kept'][0]:.3f} | "
                         f"{e['23']['kept'][0]:.3f} | {e['31']['kept'][0]:.3f} | {ce['routed'][0]:.3f} ({ce['dense'][0]:.3f}) | "
                         f"{o['flops'][c]['frac']:.3f} | {o['flops'][c]['speedup']:.2f}x | "
                         f"{g.get('exp_cos_raw', float('nan')):.2f} / {g.get('exp_cos_adam', float('nan')):.2f} / "
                         f"{g.get('err31', float('nan')):.2f} / {g.get('ce', float('nan')):.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {lab: analyze(a.dir, lab, dev) for lab in a.labels.split(",")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    md = fmt_table(res)
    open(a.out.replace(".json", ".md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
