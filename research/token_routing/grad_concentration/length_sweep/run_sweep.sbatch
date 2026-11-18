#!/bin/bash
# sweep_probe.py on a node with the inputs on node-local /data (checkpoints under /data/prasann/gradconc/ckpts,
# rows under /data/prasann/gradlen/tok). Code staged at $CODE (git archive | srun tar -x). Everything via the
# environment (sbatch --export=ALL; never put comma lists in --export):
#   CODE LABEL CKNAME ARGS [MINUTES]   e.g. ARGS="--rows-from tok --tok-root /data/prasann/gradlen/tok/both --rung 32k
#   --rows 128 --steps 0:8 --devices 0"
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/%x-%j.out
set -eo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CODE=${CODE:?}; LABEL=${LABEL:?}; CKNAME=${CKNAME:?}
ROOT=/data/prasann/gradlen
PY=${PY:-/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python}
export TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=$ROOT/tmp/triton HF_HOME=$ROOT/hf_cache PYTHONPATH="$CODE/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$ROOT/results/$LABEL"
cd "$CODE"
CK=/data/prasann/gradconc/ckpts/$CKNAME/model_and_optim
ls "$CK" | head -2
OPS_RUN_ID="gsl-${LABEL}-${SLURM_JOB_ID}" OPS_EXPERIMENT=grad_composition OPS_KIND=eval OPS_PARENT_RUN="$CKNAME" \
  OPS_CKPT_PATH="$CK" OPS_EXPECTED_MINUTES=${MINUTES:-120} OPS_FIRST_OUTPUT=1800 OPS_STALL=3600 \
  bash "$REPO/recipes/env/ops_job.sh" "$PY" -u research/token_routing/grad_concentration/length_sweep/sweep_probe.py \
    --ckpt "$CK" --out "$ROOT/results/$LABEL" $ARGS
echo "GSL_JOB_DONE $LABEL"
