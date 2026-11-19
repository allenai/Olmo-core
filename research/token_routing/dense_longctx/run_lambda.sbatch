#!/bin/bash
# Generic Lambda payload for the dense long-context work. Submit with an explicit gres, e.g.
#   sbatch --partition=lambda --account=site --qos=preemptive_high --gres=gpu:A100:1 --nodes=1 --time=01:00:00 \
#     --job-name=d1m-x --output=$L/logs/%x_%j.log --export=ALL,NGPU=1,CMD='research/token_routing/dense_longctx/test_int32_overflow.py --ops conv' \
#     $L/research/token_routing/dense_longctx/run_lambda.sbatch
# NGPU=1 runs `python CMD`; NGPU>1 runs `python -m torch.distributed.run --nproc_per_node NGPU CMD`.
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
set -o pipefail   # no -u: the conda-pack activate trips on unbound vars
L=${LAMBDA_REPO:-/accounts/projects/sewonm/prasann/ctc_suite/OLMo-core-dense1m}
cd "$L"
ENV=/accounts/projects/sewonm/prasann/olmo_test/env
source "$ENV/bin/activate"
conda-unpack 2>/dev/null || true
export PYTHONPATH="$L/src:$L/research/common"
export HF_HOME=/accounts/projects/sewonm/prasann/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF:-expandable_segments:True}
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache; mkdir -p "$TRITON_CACHE_DIR"
CMD="${CMD//@/,}"
echo "=== $(hostname) job=$SLURM_JOB_ID NGPU=${NGPU:-1} start=$(date '+%F %T') CMD=$CMD"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
CMD="${CMD//@/,}"   # sbatch --export splits on commas: write them as @ in CMD
NGPU=${NGPU:-1}
if [[ "${CMD%% *}" == *.sh ]]; then
  bash $CMD
elif [ "$NGPU" -gt 1 ]; then
  "$ENV/bin/python" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=$((29500 + SLURM_JOB_ID % 1000)) $CMD
else
  "$ENV/bin/python" $CMD
fi
rc=$?
echo "=== done rc=$rc end=$(date '+%F %T')"
exit $rc
