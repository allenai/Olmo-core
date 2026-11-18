"""Summarize sweep_probe.py outputs: per run (one checkpoint x one row length) and across lengths.

Per run, per selection rule (down7 = global oracle over the 16-row step, rowq7 = per-row quota, docq7 = per-document
quota, rand7 = random): k = share of body tokens needed for the kept-token update of blocks 7-31 to reach cos 0.95 /
0.9 with the step's full update -- analyze.py's definition (cos summed over the (block, mixer|mlp) groups of blocks
7-31; x axis = [0] + the realized kept-body share per bin, averaged over steps; linear interpolation; mean over
steps). Error bars: bootstrap over steps (2,000 resamples, 95% percentile interval). Tokens = k x body tokens per
step (and / 16 per row).
Row level ("rowself"): each row's own per-row-quota update against that row's own full update; k per row, mean and
per task, bootstrap over rows.
Across lengths: log-log fit of k(per-row quota, cos 0.95) against mean row length, slope with a bootstrap interval
(steps resampled within each length, refit). slope -1 = constant token count; 0 = constant fraction.

    python analyze_sweep.py --runs <label>=<dir>[,...] --out summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

TARGETS = (0.9, 0.95)
RANKS = ("down7_body", "rowq7_body", "docq7_body", "rand7_body")
NAMES = {"down7_body": "global oracle", "rowq7_body": "per-row quota", "docq7_body": "per-document quota",
         "rand7_body": "random"}
G0 = 14  # groups 2*7 .. 63 = blocks 7-31
B = 2000


def k_at(ks, ys, t):
    ks, ys = np.asarray(ks, float), np.asarray(ys, float)
    if ys[0] >= t:
        return float(ks[0])
    for i in range(1, len(ks)):
        if ys[i] >= t:
            f = (t - ys[i - 1]) / max(ys[i] - ys[i - 1], 1e-12)
            return float(ks[i - 1] + f * (ks[i] - ks[i - 1]))
    return float("nan")


def boot(x, rng, stat=np.nanmean):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, len(x), (B, len(x)))
    v = np.array([stat(x[i]) for i in idx])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def load_run(d):
    steps = sorted(glob.glob(os.path.join(d, "step_*.npz")))
    A, R = [], []
    for f in steps:
        j = int(os.path.basename(f)[5:9])
        A.append(dict(np.load(f)))
        R.append(json.load(open(os.path.join(d, f"rows_{j:04d}.json"))))
    return A, R


def step_k(A, r, t, kx=None):
    """per-step k for rule r at threshold t; kx = shared x axis (analyze.py: mean kfrac over steps)."""
    out = []
    for a in A:
        dot, sq, full = a[f"{r}|dot"][G0:].sum(0), a[f"{r}|sq"][G0:].sum(0), a[f"{r}|full"][G0:].sum()
        cos = dot / np.sqrt(np.maximum(sq * full, 1e-300))
        out.append(k_at(kx, cos, t))
    return np.array(out)


def summarize_run(label, d, rng):
    A, R = load_run(d)
    if not A:
        return None
    ks = R[0]["ks"]
    rows = [x for r in R for x in r["rows"]]
    nb_step = np.array([a["down7_body|nelig"][0] for a in A], float)
    T = np.array([x["T"] for x in rows], float)
    out = {"label": label, "dir": d, "steps": len(A), "eval_size_rows": len(rows), "T_mean": float(T.mean()),
           "T_min": int(T.min()), "T_max": int(T.max()), "yarn": R[0].get("yarn", 0.0), "ckpt": R[0]["ckpt"],
           "rung": R[0].get("rung"), "rows_from": R[0].get("rows_from"), "body_tokens_per_step": float(nb_step.mean()),
           "body_share": float(np.mean([x["n_body"] / x["T"] for x in rows])),
           "ce_mean": float(np.mean([x["ce"] for x in rows])),
           "peak_gib": R[-1].get("peak_gib"), "rules": {}, "tasks": {}}
    for r in RANKS:
        if r.startswith("rand"):
            kx = [0.0] + list(ks)
        else:
            kx = [0.0] + list(np.mean([a[f"{r}|kfrac"] for a in A], 0))
        e = {"name": NAMES[r]}
        for t in TARGETS:
            kk = step_k(A, r, t, kx)
            e[f"k_cos{t}"] = float(np.nanmean(kk))
            e[f"k_cos{t}_sd"] = float(np.nanstd(kk, ddof=1)) if len(kk) > 1 else 0.0
            e[f"k_cos{t}_ci"] = boot(kk, rng)
            e[f"k_cos{t}_nan_steps"] = int(np.isnan(kk).sum())
            tok = kk * nb_step
            e[f"tokens_step_cos{t}"] = float(np.nanmean(tok))
            e[f"tokens_step_cos{t}_ci"] = boot(tok, rng)
            e[f"tokens_row_cos{t}"] = float(np.nanmean(tok)) / 16
            e[f"_steps_cos{t}"] = kk.tolist()
        out["rules"][r] = e
    # row level: each row's own per-row-quota update vs its own full update
    if "rowself|dot" in A[0]:
        krow = {t: [] for t in TARGETS}
        for a in A:
            for q in range(a["rowself|dot"].shape[0]):
                cos = a["rowself|dot"][q] / np.sqrt(np.maximum(a["rowself|sq"][q] * a["rowself|full"][q], 1e-300))
                kx = [0.0] + list(a["rowself|kfrac"][q])
                for t in TARGETS:
                    krow[t].append(k_at(kx, cos, t))
        for t in TARGETS:
            kr = np.array(krow[t])
            nbr = np.array([x["n_body"] for x in rows], float)
            out[f"rowself_k_cos{t}"] = float(np.nanmean(kr))
            out[f"rowself_k_cos{t}_median"] = float(np.nanmedian(kr))
            out[f"rowself_k_cos{t}_ci"] = boot(kr, rng)
            out[f"rowself_tokens_cos{t}"] = float(np.nanmean(kr * nbr))
            out[f"rowself_tokens_cos{t}_ci"] = boot(kr * nbr, rng)
            out[f"_rows_rowself_cos{t}"] = kr.tolist()
        for task in sorted({x["task"] for x in rows}):
            sel = np.array([x["task"] == task for x in rows])
            tr = [x for x in rows if x["task"] == task]
            te = {"rows": int(sel.sum()), "T_mean": float(np.mean([x["T"] for x in tr])),
                  "n_docs_mean": float(np.mean([x["n_docs"] for x in tr])),
                  "n_docs_range": [int(min(x["n_docs"] for x in tr)), int(max(x["n_docs"] for x in tr))],
                  "body_tokens_mean": float(np.mean([x["n_body"] for x in tr])),
                  "ce_mean": float(np.mean([x["ce"] for x in tr]))}
            for t in TARGETS:
                kr = np.array(krow[t])[sel]
                nbr = np.array([x["n_body"] for x in tr], float)
                te[f"rowself_k_cos{t}"] = float(np.nanmean(kr))
                te[f"rowself_k_cos{t}_ci"] = boot(kr, rng)
                te[f"rowself_tokens_cos{t}"] = float(np.nanmean(kr * nbr))
                te[f"rowself_nan_rows_cos{t}"] = int(np.isnan(kr).sum())
            out["tasks"][task] = te
    return out


def fit(points, rng, key="rowq7_body", t=0.95, kind="step"):
    """log k = a + b log T over lengths; points: list of run summaries (one per length). Bootstrap: resample the
    per-step (or per-row) k within each length, refit."""
    xs, ys = [], []
    for p in points:
        v = np.array(p["rules"][key][f"_steps_cos{t}"] if kind == "step" else p[f"_rows_rowself_cos{t}"], float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        xs.append(np.log(p["T_mean"]))
        ys.append(v)
    if len(xs) < 2:
        return None
    xs = np.array(xs)
    b0 = np.polyfit(xs, np.log([y.mean() for y in ys]), 1)[0]
    bs = []
    for _ in range(B):
        m = [y[rng.integers(0, len(y), len(y))].mean() for y in ys]
        bs.append(np.polyfit(xs, np.log(m), 1)[0])
    return {"slope": float(b0), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "lengths": [float(np.exp(x)) for x in xs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="label=dir,... (one checkpoint series; label = length tag)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-steps-fit", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    runs = []
    for item in a.runs.split(","):
        lab, d = item.split("=", 1)
        s = summarize_run(lab, d, rng)
        if s is None:
            print(f"[skip] {lab}: no steps in {d}")
            continue
        runs.append(s)
        line = " | ".join(f"{NAMES[r]} {s['rules'][r]['k_cos0.95'] * 100:.2f}% ({s['rules'][r]['tokens_step_cos0.95']:.0f} tok/step)"
                          for r in RANKS)
        print(f"{lab}: T~{s['T_mean']:.0f}, {s['eval_size_rows']} rows / {s['steps']} steps; cos.95: {line}"
              + (f"; rowself {s['rowself_k_cos0.95'] * 100:.2f}%" if "rowself_k_cos0.95" in s else ""))
    runs.sort(key=lambda s: s["T_mean"])
    fits = {}
    for t in TARGETS:
        for r in RANKS:
            fits[f"{r}|cos{t}"] = fit([s for s in runs if s["steps"] >= a.min_steps_fit], rng, r, t)
        if all("_rows_rowself_cos0.95" in s for s in runs):
            fits[f"rowself|cos{t}"] = fit(runs, rng, t=t, kind="row")
    json.dump({"runs": runs, "fits": fits}, open(a.out, "w"), indent=1)
    for k, v in fits.items():
        if v:
            print(f"fit {k}: slope {v['slope']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
