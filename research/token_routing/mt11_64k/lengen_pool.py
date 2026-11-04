"""Pool the mt11_64k length-generalization evals and compare arms at matched FLOPs.

Input: the Beaker log of a ``lengen_collect_beaker.sh`` job (LGM lines = one olmo-eval job's metrics.json), pooled
exactly as ``launch_ctc_suite.py --collect`` does (shards weighted by instance count; binomial SE).
Training PF: the mt11-64k collector's ``results/points.json`` (existing m64 runs) plus ``extra_pf.json`` beside this
file for the m16 (16k-window) runs.

Matched-FLOP delta per (task, rung): arm score minus the dense LR-5e-6 curve (f16..f1, seed 0) interpolated linearly in
log PF at the arm's PF, as in collect_m64.py.

    python3 research/token_routing/mt11_64k/lengen_pool.py --collect-job <beaker exp id> [--record]
"""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_m64 import interp_log  # noqa: E402

BEAKER = os.path.expanduser("~/.local/bin/beaker")
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
POINTS = os.environ.get(
    "LG_POINTS",
    "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/mt11-64k/research/token_routing/"
    "mt11_64k/results/points.json")
EXTRA_PF = os.path.join(HERE, "extra_pf.json")
OUT = os.path.join(HERE, "results_lengen")
TASK = {"ctc_nq": "nq", "ctc_hpqa": "hotpotqa", "ctc_qdmatch_nq": "qdmatch_nq", "ctc_outlier": "outlier",
        "ctc_oolong": "oolong", "ctc_contradiction": "contradiction"}
DENSE = ["m64-dense-f16-lr5e-06", "m64-dense-f8-lr5e-06", "m64-dense-f4-lr5e-06", "m64-dense-f2-lr5e-06",
         "m64-dense-f1-lr5e-06"]
RUNGS = ["r64k", "r128k", "r256k"]


#: launch-run suffixes of the same checkpoint (titan resubmission). A checkpoint with a suffixed launch is pooled
#: from that launch only, so a bin that also finished under the cancelled original is never double-counted.
SUFFIXES = ("-ti",)


def base_run(src):
    run = src.replace("m11lg-", "")
    for suf in SUFFIXES:
        if run.endswith(suf):
            return run[: -len(suf)]
    return run


def pool(log_text):
    acc, empty = {}, {}
    lines = [l.split("\t") for l in log_text.splitlines()]
    srcs = {p[1] for p in lines if len(p) >= 4 and p[0].endswith("LGM")}
    use = {}
    for src in sorted(srcs):  # prefer the suffixed (later) launch of a checkpoint
        b = base_run(src)
        if b not in use or src != f"m11lg-{b}":
            use[b] = src
    for parts in lines:
        if len(parts) >= 4 and parts[0].endswith("LGM"):
            run = base_run(parts[1])
            if use.get(run) != parts[1]:
                continue
            for t in json.loads(parts[3]):
                name, scorer = t["primary_metric"].split(":")
                v, n = t["metrics"][name][scorer], t["num_instances"]
                a = acc.setdefault((run, t["task"]), {"metric": t["primary_metric"], "sum": 0.0, "n": 0})
                a["sum"] += v * n
                a["n"] += n
        elif len(parts) >= 6 and parts[0].endswith("LGEMPTY"):
            k = base_run(parts[1])
            if use.get(k) != parts[1]:
                continue
            e = empty.setdefault(k, [0, 0])
            e[0] += int(parts[4])
            e[1] += int(parts[5])
    rows = []
    for (run, task), a in sorted(acc.items()):
        row, rung = task.split(":")
        v = a["sum"] / a["n"]
        se = math.sqrt(max(v * (1 - v), 0) / a["n"]) if 0 <= v <= 1 else float("nan")
        rows.append(dict(run=run, task=TASK.get(row, row), row=row, rung=rung, metric=a["metric"], value=v,
                         eval_size=a["n"], se=se))
    return rows, empty


def pf_table():
    pf = {t["run"]: t["pf"] for t in json.load(open(POINTS))["train"] if t.get("pf")}
    if os.path.exists(EXTRA_PF):
        pf.update(json.load(open(EXTRA_PF)))
    return pf


def matched(rows, pf):
    by = {(r["run"], r["task"], r["rung"]): r for r in rows}
    out = []
    for r in rows:
        if r["run"] in DENSE or r["run"] not in pf:
            continue
        curve = [(pf[d], by[(d, r["task"], r["rung"])]["value"]) for d in DENSE if (d, r["task"], r["rung"]) in by]
        ref, extrap = interp_log(curve, pf[r["run"]]) if len(curve) >= 2 else (None, True)
        out.append(dict(run=r["run"], task=r["task"], rung=r["rung"], pf=pf[r["run"]], score=r["value"],
                        dense_at_pf=ref, delta=None if ref is None else r["value"] - ref, extrapolated=extrap,
                        eval_size=r["eval_size"], n_dense=len(curve)))
    return out


FULL = {"r64k": 500, "r128k": 500, "r256k": 125}


def record(rows, collect_job):
    """One ops result row per COMPLETE pooled cell (eval_size = the full policy size), each recorded once
    (results_lengen/recorded.json); run_id = m11lg-<train run>, the pooled eval pass of that checkpoint."""
    done_path = os.path.join(OUT, "recorded.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    new = [r for r in rows if r["eval_size"] == FULL.get(r["rung"]) and f"{r['run']}|{r['task']}|{r['rung']}" not in done]
    if not new:
        print("record: nothing new")
        return
    path = os.path.join(OUT, "record.jsonl")
    with open(path, "w") as f:
        for r in new:
            f.write(json.dumps(dict(
                run_id=f"m11lg-{r['run']}", train_run=r["run"], experiment="mt11_64k", task=r["task"],
                rung=r["rung"].lstrip("r"), metric=r["metric"].split(":")[0], value=round(r["value"], 4),
                eval_size=r["eval_size"], cot_label="no-cot", eval_job_id=collect_job,
                ckpt_path=f"/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/{r['run']}",
                git_commit="olmo-eval 71aaf0be (prasann/ctc-suite-mt11lengen) + OLMo-core 2011b86d5",
                eval_backend="olmo-eval launch_ctc_suite full chat+doc-markers (r256k YaRN x2 chunked)")) + "\n")
    p = subprocess.run([OPS, "result", "add", "--json", path, "--check"], capture_output=True, text=True)
    print((p.stdout + p.stderr).strip()[-3000:])
    if p.returncode == 0:
        done |= {f"{r['run']}|{r['task']}|{r['rung']}" for r in new}
        json.dump(sorted(done), open(done_path, "w"), indent=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-job", required=True)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    log = subprocess.run([BEAKER, "experiment", "logs", a.collect_job], capture_output=True, text=True,
                         timeout=600).stdout
    if "LGDONE" not in log:
        raise SystemExit("collect job log incomplete (no LGDONE)")
    rows, empty = pool(log)
    pf = pf_table()
    m = matched(rows, pf)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(rows=rows, matched=m, empty=empty, pf=pf), open(os.path.join(OUT, "lengen.json"), "w"), indent=1)
    runs = sorted({r["run"] for r in rows}, key=lambda x: pf.get(x, 0))
    print("score table (eval_size<500 flagged)")
    for run in runs:
        cells = [r for r in rows if r["run"] == run]
        print(f"\n{run}  PF {pf.get(run, float('nan')):.0f}")
        for r in sorted(cells, key=lambda r: (r["task"], RUNGS.index(r["rung"]) if r["rung"] in RUNGS else 9)):
            fl = f"  ⚠ eval_size={r['eval_size']} (±{r['se']:.3f})" if r["eval_size"] < 500 else ""
            print(f"  {r['task']:14} {r['rung']:6} {r['value']:.4f}{fl}")
    print("\nmatched-FLOP deltas vs dense LR5e-6 curve")
    for x in sorted(m, key=lambda x: (x["run"], x["rung"], x["task"])):
        d = "  n/a" if x["delta"] is None else f"{x['delta']:+.3f}"
        print(f"  {x['run']:34} {x['rung']:6} {x['task']:14} {x['score']:.3f} vs {x['dense_at_pf'] or float('nan'):.3f}"
              f" {d}{' EXTRAP' if x['extrapolated'] else ''} (dense pts {x['n_dense']})")
    if empty:
        print("\nempty generations:", {k: f"{v[1]}/{v[0]}" for k, v in empty.items()})
    if a.record:
        record(rows, a.collect_job)


if __name__ == "__main__":
    main()
