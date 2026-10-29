#!/bin/bash
# CTC-suite LAMBDA launcher: Berkeley SLURM partition `lambda` (A100 SXM4-80GB, no weka/Beaker,
# NO INTERNET on head or compute nodes -- see the working precedent in
# corpus-reasoning/jobs/*lambda*.sh, esp. stage_lambda_niah.sh + lambda_train_niah_cmp.sh). Mirrors
# run_ctc_local.sbatch's knobs but targets the LROOT NFS layout those scripts already use --
# lambda nodes DO have node-local NVMe (/data, /tmp), but the established convention here is NFS
# under LROOT, so this launcher does NOT invent a node-local staging path.
#
# Env activation follows the lambda scripts exactly: a conda-pack'd relocatable env at
# $ENV, activated via `source $ENV/bin/activate` + `conda-unpack` (NOT `conda activate`).
# WANDB_MODE=offline and HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1 are load-bearing, not optional --
# api.wandb.ai, huggingface.co, and github.com are all unreachable from lambda (verified: curl
# --max-time 8 -> RC 28 / http 000 on all three, from both the headnode and a compute node).
#
# Knobs (env): TASK DATA_SRC(dir under LROOT/data) VARIANT SCALE RUN EPOCHS SEQ_LEN LR
#              GLOBAL_BATCH MICRO_BATCH BASE_SRC ATTN_BACKEND WANDB_GROUP MAX_STEPS NGPU SEED
# Placement (env; applied ONLY when this file is executed directly -- it then self-submits):
#              TIME (default 01:00:00) NODE (optional -w)
#
#   # pilot (this validation run): contradiction n20 shard, full attn, 1 epoch, seq 2048
#   TASK=contradiction DATA_SRC=contradiction_n20_docdense_nocot_qwen35 VARIANT=full \
#     SCALE=0.8b EPOCHS=1 SEQ_LEN=2048 RUN=ctc-contra-full-08b-lambda-pilot \
#     ./run_ctc_lambda.sbatch
#   # or plain sbatch (placement via sbatch flags, which override the header):
#   sbatch --time=00:30:00 -w lambda-hyperplane01 \
#     --export=ALL,TASK=...,DATA_SRC=...,VARIANT=chunked,RUN=... run_ctc_lambda.sbatch
#SBATCH --job-name=ctc-suite-lambda
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/accounts/projects/sewonm/prasann/ctc_suite/logs/ctc_suite_%x_%j.log

# ---- self-submit when executed outside slurm: env knobs -> sbatch placement flags ----
if [ -z "${SLURM_JOB_ID:-}" ]; then
  export CTC_LAUNCH_TS="$(date -Is)"   # provenance: true submit time, not job-start time
  exec sbatch \
    --time="${TIME:-01:00:00}" \
    ${QOS:+--qos="$QOS"} \
    ${NODE:+-w "$NODE"} \
    ${RUN:+--job-name="$RUN"} \
    --export=ALL "$0"
fi

# NOTE: no `-u` (unlike run_ctc_local.sbatch) -- the conda-pack'd env's `activate` script
# references unset vars like CONDA_PREFIX internally and dies under `set -u` (verified: job
# 109679 failed in <1s on `CONDA_PREFIX: unbound variable`). Matches every working lambda script.
set -o pipefail

LROOT=/accounts/projects/sewonm/prasann/ctc_suite
REPO="${REPO:-$LROOT/OLMo-core}"   # override to run a separately staged checkout (default: the shared one)
NGPU="${NGPU:-8}"
TASK="${TASK:?set TASK=<suite task name> (e.g. contradiction)}"
# DATA_DIR (absolute, e.g. node-local /tmp/prasann/...) overrides DATA_SRC -- Lambda /accounts is often at quota.
[ -n "${DATA_DIR:-}" ] || DATA_SRC="${DATA_SRC:?set DATA_SRC=<shard dir NAME under $LROOT/data> or DATA_DIR=<abs shard dir>}"
VARIANT="${VARIANT:?set VARIANT=full|chunked|chunked-mix|softtoken|...}"
EXTRA="${EXTRA:-}"   # extra train_ctc_suite flags, expanded unquoted (e.g. the --st-* token-drop flags)
SCALE="${SCALE:-0.8b}"
SEQ_LEN="${SEQ_LEN:-2048}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-5}"
GLOBAL_BATCH="${GLOBAL_BATCH:-8}"
MICRO_BATCH="${MICRO_BATCH:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
# SAVE_CKPT_FLAG="" skips the model-only save (memory probes); unset keeps --save-checkpoint
SEED="${SEED:-0}"
# 40960-seq at 0.8B OOMs on 80GB A100 without these (validated: job 109705 OOM'd at step 1,
# 77.7GB allocated -- 0.8B's default shard_degree=1/no-AC only fits 141GB H200). Force full AC +
# FSDP param sharding on lambda. Empty -> trainer default (auto).
ACT_CKPT="${ACT_CKPT:-full}"              # full | none | auto
SHARD_DEGREE="${SHARD_DEGREE:-$NGPU}"     # FSDP shard_degree = the GPUs this job holds
ATTN_BACKEND="${ATTN_BACKEND:-flash_2}"   # A100 = sm80, flash_2 verified importable + supported
RUN="${RUN:-ctc-${TASK}-${VARIANT}-${SCALE}-lambda}"
WANDB_GROUP="${WANDB_GROUP:-ctc-suite-${TASK}}"
BASE_SRC="${BASE_SRC:-$LROOT/bases/q35-08b-base-modelonly}"
export CTC_LAUNCH_TS="${CTC_LAUNCH_TS:-$(date -Is)}"
echo "=== HOST=$(hostname) TASK=$TASK VARIANT=$VARIANT SCALE=$SCALE SEQ_LEN=$SEQ_LEN RUN=$RUN \
EPOCHS=$EPOCHS GLOBAL_BATCH=$GLOBAL_BATCH ATTN_BACKEND=$ATTN_BACKEND START=$(date '+%F %T') ==="

# ---- env: conda-pack'd relocatable env, activated the way every working lambda script does ----
ENV=/accounts/projects/sewonm/prasann/olmo_test/env
source "$ENV/bin/activate"
conda-unpack 2>/dev/null || true
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
# HF_HOME must point at a cache that actually HAS the tokenizer: lambda is air-gapped, so with
# HF_HUB_OFFLINE=1 a miss is a hard failure, not a download. $LROOT/hf-cache never existed -- training
# survived only because the shards are pre-tokenized and never load a tokenizer. EVAL does load one,
# so it would have died on every run. The populated cache (Qwen3.5-0.8B-Base tokenizer, which the
# whole qwen3_5 family shares) lives in the sewonm home cache below.
export HF_HOME="${HF_HOME_OVERRIDE:-/accounts/projects/sewonm/prasann/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline WANDB_API_KEY=offline_dummy_air_gapped
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
# Triton cache: node-local and STABLE across jobs. Keying it on $SLURM_JOB_ID (the old behaviour)
# meant every job JIT-recompiled the Gated-DeltaNet kernels from scratch -- several minutes of ptxas
# before step 1, paid again on every run of a fan-out, and easy to misread as a hung job. /tmp is
# node-local (safe to share, unlike an NFS cache -- see the flashinfer NFS wedge) and Triton's cache
# writes are atomic renames, so concurrent jobs on a node can share it.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"

# ---- base: read the shared distcp directly from LROOT (NFS); guard the .metadata marker ----
BASE_CKPT="$BASE_SRC/model_and_optim"
if [ ! -f "$BASE_CKPT/.metadata" ]; then
  echo "FATAL: base checkpoint $BASE_CKPT missing .metadata -> would silently train from scratch"; exit 3
fi

# ---- data + workdir + save folder ----
# Default to LROOT (NFS), mirroring the working lambda scripts. But NFS is a per-USER 1.30T quota
# that a fan-out of 12G distcp ckpts blows through (see records/overnight-stage345-plan.md quota
# watch). SAVE_ROOT/WORK_ROOT let a launch redirect ckpt+cache to compute-node-local /data (14TB
# NVMe, NOT quota'd) -- the job runs ON a compute node so /data exists there. Harvest = node-pin the
# eval to the training host (log banner prints HOST). Data is READ-only from NFS, doesn't grow quota.
SAVE_ROOT="${SAVE_ROOT:-$LROOT/ckpts}"
WORK_ROOT="${WORK_ROOT:-$LROOT}"
DATA_LOCAL="${DATA_DIR:-$LROOT/data/$DATA_SRC}"
[ -f "$DATA_LOCAL/metadata.json" ] || { echo "FATAL: $DATA_LOCAL/metadata.json missing -- stage the shard first"; exit 3; }
WORK_LOCAL=$WORK_ROOT/cache-$SLURM_JOB_ID
mkdir -p "$WORK_LOCAL"
SAVE_LOCAL=$SAVE_ROOT/$RUN
mkdir -p "$SAVE_LOCAL"
echo "BASE_CKPT=$BASE_CKPT DATA_LOCAL=$DATA_LOCAL SAVE=$SAVE_LOCAL WORK=$WORK_LOCAL"

cd "$REPO"
MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
MASTER_PORT=$((29000 + RANDOM % 1000))

"$ENV/bin/python" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  recipes/train/research_trainer/train_ctc_suite.py \
  --task "$TASK" --variant "$VARIANT" --model-scale "$SCALE" \
  ${MODEL_FAMILY:+--model-family "$MODEL_FAMILY"} \
  --no-final-checkpoint \
  --run-name "$RUN" --wandb-group "$WANDB_GROUP" --base-checkpoint "$BASE_CKPT" \
  --data "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_LOCAL" \
  --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --lr "$LR" --attn-backend "$ATTN_BACKEND" \
  --global-batch "$GLOBAL_BATCH" --micro-batch-instances "$MICRO_BATCH" --seed "$SEED" \
  ${ACT_CKPT:+--activation-checkpointing "$ACT_CKPT"} \
  ${SHARD_DEGREE:+--shard-degree "$SHARD_DEGREE"} \
  ${SAVE_CKPT_FLAG---save-checkpoint} --no-compile \
  $MAXARG $EXTRA
echo "=== DONE $(date '+%F %T')  rc=$? ==="
