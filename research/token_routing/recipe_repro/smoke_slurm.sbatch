#!/bin/bash
# OPTIONAL, Berkeley slurm only: GPU smoke of the recipe and dense arms of this branch's trainer, the same
# train_ctc_suite.py flags repro.py resolves (new4 geometry: 16 rows/step, seq 16,384, micro 2, 2 GPUs), 8 steps each,
# with the model-only export at the end (what the eval reads). Code staged on the node's /data first:
#   git archive HEAD | srun -p jsteinhardt --qos=preemptive -w mooney ... tar -x -C /data/prasann/recipe_repro/code_<sha>
#   CODE=/data/prasann/recipe_repro/code_<sha> sbatch -w mooney research/token_routing/recipe_repro/smoke_slurm.sbatch
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --job-name=rr-smoke
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=60
#SBATCH --output=/data/prasann/joblogs/%x-%j.out
set -o pipefail
CODE=${CODE:?}; ARMS=${ARMS:-recipe dense}; STEPS=${STEPS:-8}
ROOT=/data/prasann/recipe_repro
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo; PY=$ENV/bin/python
BASE=/data/prasann/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim
DATA=${DATA:-/data/prasann/early_loss_proxy/shards/mtb16k}
mkdir -p $ROOT/tmp $ROOT/runs /data/prasann/joblogs
export PATH="$ENV/bin:$PATH" PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache HF_HOME=$ROOT/hf_cache
export HOME=/data/prasann/home_wts CUDA_HOME=/usr/local/cuda-12.8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"
RECIPE=$("$PY" -c "import sys; sys.argv=['x']; sys.path.insert(0,'research/token_routing/recipe_repro'); import repro; print(repro.RECIPE)")
for ARM in $ARMS; do
  case $ARM in
    recipe) VAR=softtoken; LR=1e-5; FLAGS="$RECIPE --attn-backend flash_2";;
    dense)  VAR=full; LR=5e-6; FLAGS="";;
  esac
  NAME=rr-smoke-$ARM-$SLURM_JOB_ID; SAVE=$ROOT/runs/$NAME
  echo "=== $ARM start $(date +%T) flags: $FLAGS"; T0=$(date +%s)
  # shellcheck disable=SC2086
  "$PY" -m torch.distributed.run --nproc_per_node=2 --master_port=$((29000 + RANDOM % 1000)) \
    recipes/train/research_trainer/train_ctc_suite.py --task mt4 --data "$DATA" --variant $VAR --model-scale 4b \
    --seq-len 16384 --epochs 1 --lr $LR --global-batch 16 --micro-batch-instances 2 --base-checkpoint "$BASE" \
    --work-dir $ROOT/work_$SLURM_JOB_ID --save-folder "$SAVE" --run-name $NAME --model-family qwen3_5 \
    --save-checkpoint --no-final-checkpoint --max-steps $STEPS --no-compile --no-wandb $FLAGS > $ROOT/runs/$NAME.log 2>&1
  RC=$?; T1=$(date +%s)
  grep -aE "flop_meter/actual_pflops|train/CE loss|\[online-router\] step|max_memory|peak|Error|error" $ROOT/runs/$NAME.log | tail -n 40
  [ $RC -ne 0 ] && tail -n 80 $ROOT/runs/$NAME.log
  echo "export dir:"; ls "$SAVE" 2>&1 | head
  echo "=== $ARM end rc=$RC wall $((T1 - T0))s"
done
echo RR_SMOKE_DONE
