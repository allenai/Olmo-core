"""Record the model-scale ablation's LOCAL (jsteinhardt) evals into ops results (scale_run_local.sbatch writes one
eval.log per cell under <root>/out/<run>/<task>_<rung>/). Same row format as record_beaker_evals.py; a (train run,
task, rung, metric) already recorded is skipped. Reads the node's /data through /net (inspection-only, small files).

    python research/token_routing/ds64_fast8k/record_local_scale_evals.py --out /path/rows.jsonl
"""

import argparse
import glob
import json
import math
import os
import subprocess

from record_beaker_evals import KEY_TASK, LINE, OPS, recorded

ROOT = "/net/sneetches/data/prasann/scale_abl"
#: commit of the code staged to the node (code_<sha>) that every local scale run trains and evaluates with
CODE_COMMIT = "6173899e6dc086dc9456a6249b59a03eaa5044fe"


def slurm_jobs():
    """{run_id: slurm job id} from the run registry (the evals run inside the training job)."""
    out = subprocess.run([OPS, "runs", "--experiment", "tokdrop_lr", "--cluster", "slurm", "--json"], capture_output=True,
                         text=True, timeout=120).stdout
    return {r["run_id"]: r.get("job_id") for r in (json.loads(out) if out.strip().startswith("[") else [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--experiment", default="tokdrop_lr")
    a = ap.parse_args()
    have, _ = recorded()
    jobs = slurm_jobs()
    rows = []
    for log in sorted(glob.glob(f"{a.root}/out/f8k-mt4-*-tds*/*_*/eval.log")):
        cell_dir = os.path.dirname(log)
        if not os.path.exists(os.path.join(cell_dir, ".done")):
            continue
        run = os.path.basename(os.path.dirname(cell_dir))
        for m in LINE.finditer(open(log, errors="replace").read()):
            t, rung, metric, v, n = m.groups()
            t = KEY_TASK.get(t, t)
            if (run, t, rung, metric) in have:
                continue
            v, n = float(v), int(n)
            se = math.sqrt(max(v * (1 - v), 1e-9) / n) if 0 <= v <= 1 else None
            rows.append(dict(run_id=f"{run}-ev{t}-{rung}", train_run=run, experiment=a.experiment, task=t, rung=rung,
                             metric=metric, value=v, eval_size=n, se=se and round(se, 6), cot_label="no-cot",
                             eval_backend="olmo-core-native-docchunk-ladder (local sneetches)",
                             ckpt_path=f"sneetches:/data/prasann/scale_abl/runs/{run}",
                             git_commit=CODE_COMMIT, eval_job_id=f"slurm:{jobs.get(run)}"))
            have.add((run, t, rung, metric))
    with open(a.out, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} new rows -> {a.out}")
    if rows:
        r = subprocess.run([OPS, "result", "add", "--json", a.out, "--check"], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())


if __name__ == "__main__":
    main()
