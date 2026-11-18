"""Rows for `ops result add --check` from summarize_kl.py's JSON (full waves) or a quick-read summary.

Per run (length L, arm, seed): dense held-out answer CE at the last probe (per task, eval_size 48 each, and the
3-task mean, eval_size 144) and the FLOP speedup vs dense full FT at dft's mid-epoch CE (mean and worst task).
train_run = the ops run id of the (last) Beaker / slurm job that ran it.

    python3 record_results.py --summary kl_summary.json --prefix kl --out rows.jsonl
    ops result add --json rows.jsonl --check
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ELP = os.path.dirname(HERE)


def job_ids():
    m = {}
    sb = json.load(open(f"{ELP}/state_beaker.json"))["jobs"]
    for ex, v in sb.items():  # insertion order = launch order: later launches overwrite (relaunches)
        for r in v["runs"]:
            m[r] = (f"elp-{v['tag']}-bk{ex[-8:]}", ex, v.get("commit"))
    st = json.load(open(f"{ELP}/state.json"))["jobs"]
    for jid, v in st.items():
        for r in v["runs"]:
            m.setdefault(r, (f"elp-{v['tag']}-job{jid}", jid, os.path.basename(v.get("code", "")).replace("code_", "") or None))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--prefix", default="kl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.load(open(a.summary))
    J = job_ids()
    rows = []
    for L, R in S.items():
        for arm, r in R["rows"].items():
            for e in r["seeds"]:
                name = f"elp-{a.prefix}{L}-{arm}-sd{e['seed']}"
                tr, jid, commit = J.get(name, (name, None, None))
                base = {"experiment": "tokdrop_lr", "run_id": f"{name}-probe", "train_run": tr, "rung": L,
                        "cot_label": "no-cot", "eval_backend": "elp-dense-probe(teacher-forced answer CE, dense forward)",
                        "eval_job_id": jid, "ckpt_path": "none (early-loss proxy: no checkpoint is saved)"}
                if commit:
                    base["git_commit"] = commit
                for t, v in e["task_final_ce"].items():
                    rows.append({**base, "task": t, "metric": "heldout_ce_dense_final", "value": round(v, 5),
                                 "eval_size": R["rows_per_task"][t]})
                rows.append({**base, "task": "mean3", "metric": "heldout_ce_dense_final", "value": round(e["final_ce"], 5),
                             "eval_size": R["eval_size"]})
                if arm != "dft":
                    for k, mt in (("x_mid", "flop_x_vs_dft_at_mid_ce"), ("worst_x_mid", "flop_x_vs_dft_at_mid_ce_worst_task")):
                        if e[k] is not None:
                            rows.append({**base, "task": "mean3" if k == "x_mid" else e["worst_task_mid"], "metric": mt,
                                         "value": round(e[k], 3), "eval_size": R["eval_size"] if k == "x_mid" else 48})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
