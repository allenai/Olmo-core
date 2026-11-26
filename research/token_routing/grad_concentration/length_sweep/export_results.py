"""analyze_sweep.py summaries -> JSONL rows for ``ops result add --json`` (experiment grad_composition).

One row per (checkpoint, length, selection rule, threshold): metric ``body_keep_frac_cos{t}_<rule>`` (value = share of
body tokens needed, step-level, analyze.py's definition), plus ``body_keep_tokens_cos{t}_<rule>`` (per 16-row step)
and the row-level self-cosine ``rowself_keep_frac_cos{t}``. task = "mt4mix" (contradiction, nq, outlier, oolong; 4
rows each per step); per-task row-level rows use the task name. eval_size = rows.

    python export_results.py --summary sweep_mt11.json --train-run m64-dense-f1-lr5e-06 --prefix gsl-m11f1 --out x.jsonl
"""
from __future__ import annotations

import argparse
import json

SHORT = {"down7_body": "global", "rowq7_body": "perrow", "docq7_body": "perdoc", "rand7_body": "random"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.load(open(a.summary))
    rows = []
    for s in S["runs"]:
        base = dict(run_id=f"{a.prefix}-{s['label']}", train_run=a.train_run, experiment="grad_composition",
                    rung=s["label"], eval_size=s["eval_size_rows"], cot_label="no-cot",
                    eval_backend="sweep_probe (attribution, TensorSketch 32768)", ckpt_path=s["ckpt"],
                    git_commit=a.commit)
        for r, e in s["rules"].items():
            for t in (0.9, 0.95):
                rows.append({**base, "task": "mt4mix", "metric": f"body_keep_frac_cos{t}_{SHORT[r]}",
                             "value": round(e[f"k_cos{t}"], 6)})
                rows.append({**base, "task": "mt4mix", "metric": f"body_keep_tokens_step_cos{t}_{SHORT[r]}",
                             "value": round(e[f"tokens_step_cos{t}"], 1)})
        for t in (0.9, 0.95):
            if f"rowself_k_cos{t}" in s:
                rows.append({**base, "task": "mt4mix", "metric": f"rowself_keep_frac_cos{t}",
                             "value": round(s[f"rowself_k_cos{t}"], 6)})
                for task, v in s["tasks"].items():
                    rows.append({**base, "task": task, "eval_size": v["rows"], "metric": f"rowself_keep_frac_cos{t}",
                                 "value": round(v[f"rowself_k_cos{t}"], 6)})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
