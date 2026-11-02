#!/usr/bin/env bash
# Per-row + floor arms (launch_perrow_floor.py) on a Berkeley jsteinhardt node instead of Beaker: same trainer
# config as the recipe's Beaker runs (2 GPUs, 16 rows/step, micro 2, seq 12288, LR 1e-5, fp32 marker-repaired base,
# fast8k mt4 fresh shard), via recipes/train/research_trainer/run_ctc_local.sbatch. Inputs are staged on the node's
# /data first (code by git archive, base + shard from S3 _transfer/; see the session's staging command):
#   /data/prasann/perrow_floor/{code_<sha>, bases/q35-4b-base-markerfix, shards/mt4<budget>}
#
#   ARM=ds0f2 SEED=0 MAX_STEPS=4 NODE=cubbins CODE_SHA=<sha> research/token_routing/online_router/run_perrow_floor_local.sh
set -euo pipefail
ARM="${ARM:?ds0|ds0f1|ds0f2|recipe}"
SEED="${SEED:-0}"
BUDGET="${BUDGET:-f1952}"
NODE="${NODE:-cubbins}"
CODE_SHA="${CODE_SHA:?staged code dir suffix}"
MAX_STEPS="${MAX_STEPS:-0}"
D=/data/prasann/perrow_floor
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_NFS="$(git -C "$HERE" rev-parse --show-toplevel)"
read -r NAME FLAGS < <(HERE="$HERE" python3 - "$ARM" "$BUDGET" "$SEED" "$MAX_STEPS" <<'PY'
import importlib.util, os, sys
arm, budget, seed, ms = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
spec = importlib.util.spec_from_file_location("lpf", os.path.join(os.environ["HERE"], "launch_perrow_floor.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(m.run_name(arm, budget, seed, ms > 0) + "-local", m.ARMS[arm][1])
PY
)
export REPO="$D/code_$CODE_SHA" NGPU=2 TASK=mt4 DATA_SRC="$D/shards/mt4$BUDGET" VARIANT=softtoken SCALE=4b \
  MODEL_FAMILY=qwen3_5 SEQ_LEN=12288 EPOCHS=1 LR=1e-05 GLOBAL_BATCH=16 MICRO_BATCH=2 SEED="$SEED" \
  BASE_SRC="$D/bases/q35-4b-base-markerfix" WANDB_GROUP=f8k-q35-4b MAX_STEPS="$MAX_STEPS" NO_FINAL_CKPT=1 \
  RUN="$NAME" EXTRA="$FLAGS"
OUT=$(cd "$REPO_NFS" && sbatch --partition=jsteinhardt --account=site --qos=preemptive_high -w "$NODE" \
  --gres=gpu:H200:2 --cpus-per-task=16 --mem=160G --time="${TIME:-08:00:00}" --job-name="$NAME" \
  --export=ALL recipes/train/research_trainer/run_ctc_local.sbatch)
echo "$OUT"
JOB=$(awk '{print $NF}' <<<"$OUT")
/accounts/projects/berkeleynlp/prasann/bin/ops register --run-id "$NAME" --experiment tokdrop_lr --cluster slurm \
  --kind train --job-id "$JOB" --expected-minutes "$([ "$MAX_STEPS" -gt 0 ] && echo 30 || echo 240)" \
  --log-path "/data/prasann/joblogs/ctc_suite_${NAME}_${JOB}.log" --ckpt-path "/data/prasann/ctc_suite/ckpts/$NAME" \
  --wandb-url https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/f8k-q35-4b --git "$REPO_NFS"
echo "$NAME $JOB"
