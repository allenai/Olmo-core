#!/bin/bash
# Oracle-selection training on the Lambda A100-80GB cluster (no shared FS / no internet: code, base and shard are
# rsync'd / streamed in first, see the README). Base + shard are copied to node /tmp; checkpoints stay on node /tmp.
#   ssh lambda "cd $L/oracle/code && sbatch -w <node> -J <run> --export=ALL,RUN=<run>,ARM_ARGS='...' \
#       research/token_routing/oracle_train/run_oracle_lambda.sbatch"
#SBATCH --partition=lambda
#SBATCH --qos=preemptive_high
#SBATCH --account=site
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=14:00:00
#SBATCH --output=/accounts/projects/sewonm/prasann/ctc_suite/oracle/logs/%x-%j.out
set -eo pipefail
L=/accounts/projects/sewonm/prasann/ctc_suite
RUN=${RUN:?set RUN}
CODE=$L/oracle/code
ENV=/accounts/projects/sewonm/prasann/olmo_test/env
source "$ENV/bin/activate"
T=/tmp/prasann/oracle
mkdir -p $T/bases $T/shards $T/runs $T/triton
# node-local copies (shared by the jobs on this node; flock so concurrent jobs copy once)
(
  flock 9
  [ -f $T/bases/q35-4b-base-markerfix/model_and_optim/.metadata ] || \
    [ -f /tmp/prasann/tdlr/base/q35-4b-base-markerfix/model_and_optim/.metadata ] || \
    cp -r $L/oracle/stage/ctc_suite/bases/q35-4b-base-markerfix $T/bases/
  [ -f $T/shards/mt4f1952/metadata.json ] || cp -r $L/oracle/stage/oracle/shards/mt4f1952 $T/shards/
) 9>$T/stage.lock
BASE=$T/bases/q35-4b-base-markerfix/model_and_optim
# the tdlr campaign's node-local copy of the same base (identical .metadata md5) when present
[ -f /tmp/prasann/tdlr/base/q35-4b-base-markerfix/model_and_optim/.metadata ] && \
  BASE=/tmp/prasann/tdlr/base/q35-4b-base-markerfix/model_and_optim
[ -f "$BASE/.metadata" ] || { echo "FATAL: base missing .metadata"; exit 3; }
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/accounts/projects/sewonm/prasann/.cache/huggingface
export WANDB_MODE=offline WANDB_API_KEY=offline_dummy_air_gapped WANDB_DIR=$T/runs/$RUN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=$T/triton TMPDIR=$T
SAVE=$T/runs/$RUN
mkdir -p "$SAVE"
cd "$CODE"
RECIPE="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 \
--st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-early-lora 64 \
--st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32 --attn-backend flash_2 --st-otr-score oracle"
echo "=== HOST=$(hostname) RUN=$RUN START=$(date '+%F %T') ARM_ARGS=${ARM_ARGS:-}"
"$ENV/bin/python" -m torch.distributed.run --nproc_per_node=2 --master_port=$((29000 + RANDOM % 1000)) \
  recipes/train/research_trainer/train_ctc_suite.py \
  --task mt4 --variant softtoken --model-family qwen3_5 --model-scale 4b \
  --run-name "$RUN" --wandb-group f8k-oracle --base-checkpoint "$BASE" \
  --data $T/shards/mt4f1952 --work-dir "$T/work_${SLURM_JOB_ID}" --save-folder "$SAVE" \
  --seq-len 12288 --epochs 1 --lr 1e-05 --global-batch 16 --micro-batch-instances 2 --seed 0 \
  --save-checkpoint --no-compile --no-final-checkpoint ${MAXARG:-} $RECIPE ${ARM_ARGS:-}
echo "=== ORACLE_DONE $RUN $(date '+%F %T')"
