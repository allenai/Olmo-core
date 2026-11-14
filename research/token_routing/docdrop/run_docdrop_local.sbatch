#!/bin/bash
# Doc-drop training on jsteinhardt (records/docdrop-nongold-baseline.md). Code staged on the node first (stage_code.sh);
# base + shards must be on the node's /data. Launched by launch_docdrop.py (env: RUN, VARIANT, DATA, ARM_ARGS, CODE,
# MAX_STEPS, SEED, LR, MINUTES, NGPU, SAVECK, NOWANDB). Requeued jobs resume from --save-folder.
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --requeue
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/data/prasann/joblogs/%x-%j.out
set -eo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
RUN=${RUN:?set RUN}; CODE=${CODE:?set CODE}; DATA=${DATA:?set DATA}
ROOT=/data/prasann/docdrop
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
PY=$ENV/bin/python
BASE=${BASE:-/data/prasann/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim}
mkdir -p "$ROOT/tmp" "$ROOT/runs" /data/prasann/joblogs
[ -f "$BASE/.metadata" ] || { echo "FATAL: base $BASE missing .metadata"; exit 3; }
[ -d "$CODE/src" ] || { echo "FATAL: staged code $CODE missing"; exit 3; }
[ -f "$DATA/metadata.json" ] || { echo "FATAL: shard $DATA missing"; exit 3; }
export PATH="$ENV/bin:$PATH" PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache HF_HOME=$ROOT/hf_cache
export HOME=/data/prasann/home_wts CUDA_HOME=/usr/local/cuda-12.8
if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${NOWANDB:-}" ] && [ -r "/accounts/projects/berkeleynlp/prasann/.netrc" ]; then
  WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' /accounts/projects/berkeleynlp/prasann/.netrc); export WANDB_API_KEY
fi
WANDBARG=""; [ -n "${NOWANDB:-}" ] && WANDBARG="--no-wandb"
MAXARG=""; [ "${MAX_STEPS:-0}" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
SAVE=$ROOT/runs/$RUN
mkdir -p "$SAVE"
cd "$CODE"
OPS_RUN_ID="$RUN" OPS_EXPERIMENT=tokdrop_lr OPS_KIND=train OPS_EXPECTED_MINUTES=${MINUTES:-60} OPS_FIRST_OUTPUT=1200 \
  OPS_STALL=1800 bash "$REPO/recipes/env/ops_job.sh" \
  "$PY" -m torch.distributed.run --nproc_per_node=${NGPU:-2} --master_port=$((29000 + RANDOM % 1000)) \
  recipes/train/research_trainer/train_ctc_suite.py \
  --task mt4 --variant "${VARIANT:-full}" --model-family qwen3_5 --model-scale 4b \
  --run-name "$RUN" --wandb-group "${WANDB_GROUP:-f8k-docdrop}" --base-checkpoint "$BASE" \
  --data "$DATA" --work-dir "$ROOT/work_${SLURM_JOB_ID}" --save-folder "$SAVE" \
  --seq-len ${SEQ_LEN:-12288} --epochs 1 --lr ${LR:-1e-05} --global-batch 16 --micro-batch-instances 2 --seed ${SEED:-0} \
  ${SAVECK:+--save-checkpoint} --no-compile $WANDBARG $MAXARG --no-final-checkpoint ${ARM_ARGS:-}
echo "DOCDROP_DONE $RUN"
