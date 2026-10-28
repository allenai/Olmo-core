"""Merge grad_fidelity2.py shard JSONs (numpy only; login-node safe) into one summary.

    python grad_fidelity2_merge.py --parts <dir of part JSONs> --out merged.json [--md table.md]

Per (checkpoint tag, setting, rule): medians over held-out rows of the per-row metrics (single-draw cos, debiased
cos(gbar, g_full), relative bias, variance), means of the 16-row-batch and 8-batch-accumulated cosines (over draws and
batches / groups), the median dCE and the realised kept share; for each gradient view (trained / all).
"""
import argparse
import glob
import json
import os

import numpy as np

ROW_METRICS = ("cos1", "cos_bar", "cos_bar_raw", "relbias", "relbias_raw", "var", "norm_ratio")


def summarize(parts):
    rows = [r for p in parts for r in p["rows"]]
    keys = sorted({(r["tag"], r["setting"], r["rule"]) for r in rows})
    out = []
    for tag, s, rule in keys:
        rr = [r for r in rows if (r["tag"], r["setting"], r["rule"]) == (tag, s, rule)]
        bb = [b for p in parts for b in p["batches"] if (b["tag"], b["setting"], b["rule"]) == (tag, s, rule)]
        aa = [x for p in parts for x in p["accum"] if (x["tag"], x["setting"], x["rule"]) == (tag, s, rule)
              and x.get("n_batches", 8) == 8]
        rec = dict(tag=tag, setting=s, rule=rule, eval_size=len(rr), n_batch_draws=len(bb), n_accum_draws=len(aa),
                   dce=float(np.median([r["dce"] for r in rr])), kept_frac=float(np.mean([r["kept_frac"] for r in rr])),
                   tasks=sorted({r["task"] for r in rr}))
        for v in ("trained", "all"):
            for m in ROW_METRICS:
                k = f"{v}_{m}"
                vals = [r[k] for r in rr if k in r]
                if vals:
                    rec[k] = float(np.median(vals))
            for name, src in (("batch_cos", bb), ("accum_cos", aa)):
                vals = [x[f"{v}_cos"] for x in src if f"{v}_cos" in x]
                rec[f"{v}_{name}"] = float(np.mean(vals)) if vals else float("nan")
        out.append(rec)
    return out


def table(summ, view="trained"):
    hdr = (f"| ckpt | setting | rule | rows | single-draw cos | cos(gbar) | rel bias | var | 16-row batch cos | "
           f"8-batch cos | dCE |")
    lines = [hdr, "|" + "---|" * 11]
    for s in summ:
        g = lambda k: s.get(f"{view}_{k}", float("nan"))  # noqa: E731
        lines.append(f"| {s['tag']} | {s['setting']} | {s['rule']} | {s['eval_size']} | {g('cos1'):+.3f} | "
                     f"{g('cos_bar'):+.3f} | {g('relbias'):.3f} | {g('var'):.3f} | {g('batch_cos'):+.3f} | "
                     f"{g('accum_cos'):+.3f} | {s['dce']:+.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.parts, "**", "*.json"), recursive=True))
    parts = []
    for f in files:
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "rows" in d and "batches" in d:
            parts.append(d)
    summ = summarize(parts)
    json.dump(dict(n_parts=len(parts), summary=summ), open(a.out, "w"), indent=1)
    md = "\n\n".join(f"### view: {v}\n\n{table(summ, v)}" for v in ("trained", "all"))
    if a.md:
        open(a.md, "w").write(md + "\n")
    print(f"{len(parts)} parts, {len(summ)} (ckpt, setting, rule) cells")
    print(md)


if __name__ == "__main__":
    main()
