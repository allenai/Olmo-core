#!/bin/bash
# Print the selective-backprop counters (train/sbp/*) and flops.json of finished cross-row runs from their weka save
# folders (the login node mounts no weka, and the Beaker runs' wandb project is not readable with the local key).
# CPU-only Beaker job; read its log with `beaker experiment logs <id>`.
#
#   bash research/token_routing/cross_row/read_summaries_beaker.sh
set -uo pipefail
CKPTS=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts
RUNS="${RUNS:-$(python3 -c "import json; print(' '.join(n for n, r in json.load(open('research/token_routing/cross_row/state.json'))['runs'].items() if not r.get('smoke')))")}"
gantry run --name "xrow-read-summaries-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' --cluster ai2/phobos --cluster ai2/hammond \
  --gpus 0 --cpus 1 --memory 4GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env CKPTS="$CKPTS" --env RUNS="$RUNS" \
  --allow-dirty --timeout 0 --yes -- bash -c '
for R in $RUNS; do
  /opt/conda/bin/python - "$CKPTS/$R" <<PY
import glob, json, sys
d = sys.argv[1]
out = {"run": d.rsplit("/", 1)[-1]}
for f in sorted(glob.glob(d + "/wandb/wandb/run-*/files/wandb-summary.json"))[-1:]:
    s = json.load(open(f))
    out.update({k: s[k] for k in s if "sbp" in k or k in ("flop_meter/actual_pflops", "train/CE loss")})
try:
    fl = json.load(open(d + "/flops.json"))
    out.update(actual_pflops=fl.get("actual_pflops"), actual_over_dense=fl.get("actual_over_dense"))
except Exception as e:
    out["flops_err"] = str(e)
print("XROW_SUMMARY " + json.dumps(out), flush=True)
PY
done
echo XROW_SUMMARIES_DONE' 2>&1 | grep -E "beaker.org/ex|rror|xperiment" | head -5
