"""Rows for `ops result add --check` from analyze_ng.py's JSON.

    python3 record_ng.py --results ng_results.json --out rows.jsonl ; ops result add --json rows.jsonl --check
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ELP = os.path.dirname(HERE)
DESC = {"dft": "dense full FT", "k20": "recipe (LoRA r64 + router 20%/7%)",
        "rddA": "recipe + random doc drop keep U(.05,.5), 15% whole, labels rewritten",
        "rddB": "recipe + random doc drop keep U(.25,.75), 15% whole, labels rewritten",
        "rddAnr": "recipe + rddA's doc drop WITHOUT label rewrite (token-span removal)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.load(open(a.results))
    sb = json.load(open(f"{ELP}/state_beaker.json"))["jobs"]
    J = {r: (f"elp-{v['tag']}-bk{ex[-8:]}", ex, v.get("commit")) for ex, v in sb.items() for r in v["runs"]}
    rows = []
    for arm, seeds in S["arms"].items():
        for sd, r in seeds.items():
            name = f"elp-ng16k-{arm}-sd{sd}"
            tr, jid, commit = J.get(name, (name, None, None))
            base = {"experiment": "tokdrop_lr", "run_id": f"{name}-probe", "train_run": tr, "rung": "16k",
                    "cot_label": "no-cot", "eval_backend": "elp-dense-probe(teacher-forced answer CE, dense forward)",
                    "eval_job_id": jid, "ckpt_path": "none (early-loss proxy: no checkpoint is saved)",
                    "notes": f"random_doc_drop ng16k (reorder/oolong/grouping): {DESC.get(arm, arm)}"}
            if commit:
                base["git_commit"] = commit
            for t, v in r["final_by_task"].items():
                rows.append({**base, "task": t, "metric": "heldout_ce_dense_final", "value": round(v, 5),
                             "eval_size": S["rows_per_task"][t]})
            rows.append({**base, "task": "mean3", "metric": "heldout_ce_dense_final", "value": round(r["final_ce"], 5),
                         "eval_size": S["eval_size"]})
            for k, mt in (("x_dft_step", "step_cost_vs_dft"), ("x_dft_at_min", "flop_x_vs_dft_at_own_min_ce"),
                          ("x_k20_at_min", "flop_x_vs_k20_at_own_min_ce")):
                if r.get(k) is not None and not (arm == "dft" and k != "x_dft_step"):
                    rows.append({**base, "task": "mean3", "metric": mt, "value": round(r[k], 4), "eval_size": S["eval_size"]})
    with open(a.out, "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(len(rows), "rows ->", a.out)


if __name__ == "__main__":
    main()
