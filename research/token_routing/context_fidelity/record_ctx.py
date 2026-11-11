"""Record analyze_context.py headline numbers with ONE `ops result add --json ... --check` call per label.

    python3 research/token_routing/context_fidelity/record_ctx.py results/ctx_f1952.json f1952 <eval-run-id> \
        <train-run> <commit> <ckpt> <job-ids>
Run once per analysis (rows are appended, not deduplicated). eval_size = probe rows (pooled) or rows per task.
"""
import json
import subprocess
import sys
import tempfile

OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
path, lab, run_id, train_run, commit, ckpt, job = sys.argv[1:8]
o = json.load(open(path))[lab]
base = dict(run_id=run_id, train_run=train_run, experiment="tokdrop_lr", rung="8k", cot_label="no-cot",
            eval_backend="context_probe+analyze_context", git_commit=commit, ckpt_path=ckpt, eval_job_id=job)
rows = []


def add(task, metric, value, n):
    if value is None or value != value:
        return
    rows.append(dict(base, task=task, metric=metric, value=float(f"{value:.6g}"), eval_size=int(n)))


N = o["rows"]
for c in o["flops"]:
    for space in ("raw", "adam"):
        v = o[space].get(c)
        if v:
            add("mt4", f"ctxfid_{space}_{c}_step_cos", v["step_cos"][0], N)
            add("mt4", f"ctxfid_{space}_{c}_exp_cos", v["exp_cos"][0], N)
            add("mt4", f"ctxfid_{space}_{c}_exp_norm_ratio", v["exp_norm_ratio"][0], N)
    add("mt4", f"ctxfid_{c}_step_cost_frac", o["flops"][c]["frac"], N)
    for bk in ("15", "23", "31"):
        add("mt4", f"ctxfid_{c}_hidden_relerr_b{bk}", o["err"][c]["pooled"][bk]["kept"][0], N)
    g = o["gap_closed"].get(c, {})
    for k in ("exp_cos_adam", "ce"):
        add("mt4", f"ctxfid_{c}_gap_closed_{k}", g.get(k), N)
    for t, ce in o["ce"][c].items():
        n = N if t == "pooled" else o["rows_per_task"][t]
        add("mt4" if t == "pooled" else t, f"ctxfid_{c}_answer_ce", ce["routed"][0], n)
        add("mt4" if t == "pooled" else t, "ctxfid_dense_answer_ce", ce["dense"][0], n)
seen, uniq = set(), []
for r in rows:
    k = (r["task"], r["metric"])
    if k not in seen:
        seen.add(k)
        uniq.append(r)
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(uniq, f)
out = subprocess.run([OPS, "result", "add", "--json", f.name, "--check"], capture_output=True, text=True)
lines = (out.stdout + out.stderr).strip().splitlines()
print(f"{len(uniq)} rows; rc={out.returncode}")
print("\n".join(lines[-25:]))
