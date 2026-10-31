"""Record the headline gradient-gap numbers of an analyze_gap.py output with `ops result add --check`.

    python3 research/token_routing/grad_concentration/record_gap.py results/gap_f1952.json f1952 <run-id> <train-run> <commit> <ckpt> <job-id>
Run it ONCE per analysis (rows are appended, not deduplicated).
"""
import json
import subprocess
import sys

OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
path, lab, run_id, train_run, commit, ckpt, job = sys.argv[1:8]
d = json.load(open(path))[lab]
rows = []
for s in ("og_recipe", "od_recipe", "rnd_recipe", "rtr_recipe", "og_L16_k0.07", "rnd_L16_k0.07", "rtr_L16_k0.07"):
    for space in ("raw", "adam"):
        v = d[space][s]["whole"]
        for key in ("attr_cos", "routed_cos", "exp_cos_attr", "exp_cos_routed", "exp_proj_routed", "pair_cos_routed",
                    "bias_share_routed"):
            x = v[key]
            val = x[0] if isinstance(x, list) else x
            rows.append((f"gradgap_{space}_{s}_{key}", val))
    rows.append((f"gradgap_raw_dense_pair_cos", d["raw"][s]["whole"]["pair_cos_dense"][0]))
for s in ("og_recipe", "rnd_recipe", "rtr_recipe"):
    for t, v in d["ce"][s].items():
        rows.append((f"gradgap_routed_ce_{s}_{t}", v["ce_routed"][0]))
        rows.append((f"gradgap_dense_ce_{t}", v["ce_dense"][0]))
seen = set()
for metric, val in rows:
    if metric in seen:
        continue
    seen.add(metric)
    r = subprocess.run([OPS, "result", "add", "--run-id", run_id, "--train-run", train_run, "--experiment", "tokdrop_lr",
                        "--task", "mt4", "--rung", "8k", "--metric", metric, "--value", f"{val:.6g}", "--eval-size",
                        str(d["rows"]), "--cot-label", "no-cot", "--eval-backend", "routed_probe+analyze_gap",
                        "--git-commit", commit, "--ckpt-path", ckpt, "--eval-job-id", job, "--check"], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    print(metric, f"{val:.4f}", "|", out[-1][:160] if out else "", flush=True)
