#!/bin/bash
# Oracle-selection training (olmo_core.nn.oracle_token_select) on jsteinhardt: smoke tests / short reads.
# Copies the code from SRC (a checkout on NFS, read once) to node /data; base + shard must already be on the node's /data
# (mooney: /data/prasann/ctc_suite/bases/q35-4b-base-markerfix, /data/prasann/gradconc/shards/mt4s1952).
#   sbatch -w mooney -J oracle-<arm> --export=ALL,SRC=<checkout>,RUN=<run name>,ARM_ARGS="--oracle-scope batch ..." \
#     research/token_routing/oracle_train/run_oracle_local.sbatch      (MAX_STEPS=N, NOWANDB=1, DATA=<shard dir>)
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/data/prasann/joblogs/%x-%j.out
set -eo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
SRC=${SRC:?set SRC}; RUN=${RUN:?set RUN}
ROOT=/data/prasann/oracle
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
PY=$ENV/bin/python
BASE=${BASE:-/data/prasann/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim}
DATA=${DATA:-/data/prasann/gradconc/shards/mt4s1952}
CODE=$ROOT/code_${SLURM_JOB_ID}
mkdir -p "$CODE" "$ROOT/tmp" "$ROOT/runs" /data/prasann/joblogs
(cd "$SRC" && tar -c --exclude=.git src recipes research/token_routing/oracle_train) | tar -x -C "$CODE"
[ -f "$BASE/.metadata" ] || { echo "FATAL: base $BASE missing .metadata"; exit 3; }
export PATH="$ENV/bin:$PATH" PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache HF_HOME=$ROOT/hf_cache
if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${NOWANDB:-}" ] && [ -r "$HOME/.netrc" ]; then
  WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc"); export WANDB_API_KEY
fi
WANDBARG=""; [ -n "${NOWANDB:-}" ] && WANDBARG="--no-wandb"
MAXARG=""; [ "${MAX_STEPS:-0}" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
SAVE=$ROOT/runs/$RUN
mkdir -p "$SAVE"
cd "$CODE"
# the recipe lora64all32r20to7 (launch_fast8k.py), selection by the oracle instead of the router
RECIPE="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 \
--st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 \
--st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32 --attn-backend flash_2 --st-otr-score oracle"
OPS_RUN_ID="$RUN" OPS_EXPERIMENT=tokdrop_lr OPS_KIND=train OPS_EXPECTED_MINUTES=${MINUTES:-60} OPS_FIRST_OUTPUT=1200 \
  OPS_STALL=1800 bash "$REPO/recipes/env/ops_job.sh" \
  "$PY" -m torch.distributed.run --nproc_per_node=2 --master_port=$((29000 + RANDOM % 1000)) \
  recipes/train/research_trainer/train_ctc_suite.py \
  --task mt4 --variant softtoken --model-family qwen3_5 --model-scale 4b \
  --run-name "$RUN" --wandb-group "${WANDB_GROUP:-f8k-oracle}" --base-checkpoint "$BASE" \
  --data "$DATA" --work-dir "$ROOT/work_${SLURM_JOB_ID}" --save-folder "$SAVE" \
  --seq-len ${SEQ_LEN:-12288} --epochs 1 --lr ${LR:-1e-05} --global-batch 16 --micro-batch-instances 2 --seed ${SEED:-0} \
  --save-checkpoint --no-compile $WANDBARG $MAXARG ${NOFINAL:+--no-final-checkpoint} $RECIPE ${ARM_ARGS:-}
echo "ORACLE_DONE $RUN"
