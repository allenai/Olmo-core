#!/bin/bash
# gradmap.py (per-token x per-layer gradient maps) on jsteinhardt, node-local everything. Runs one or more checkpoints
# in sequence on 1 GPU. Code is copied from SRC (read once); checkpoints / shards are mooney's staged copies.
#   sbatch -w mooney -J gradmap-<tag> --export=ALL,SRC=<checkout>,JOBS="label=ckpt[:router] ..." run_gradmap.sbatch
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/data/prasann/%x-%j.out
set -eo pipefail
SRC=${SRC:?set SRC}; JOBS=${JOBS:?set JOBS}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
ROOT=/data/prasann/gradmap
CODE=$ROOT/code_${SLURM_JOB_ID}
mkdir -p "$CODE" "$ROOT/tmp"
(cd "$SRC" && tar -c --exclude=.git --exclude='results*' src research/token_routing/grad_concentration) | tar -x -C "$CODE"
export TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=$ROOT/tmp/triton HF_HOME=$ROOT/hf_cache PYTHONPATH="$CODE/src"
cd "$CODE"
for J in $JOBS; do
  LAB=${J%%=*}; REST=${J#*=}; CK=${REST%%:*}; EXTRA=""
  [ "$REST" != "$CK" ] && EXTRA="--router"
  OPS_RUN_ID="gradmap-${LAB}-${SLURM_JOB_ID}" OPS_EXPERIMENT=tokdrop_lr OPS_KIND=eval OPS_PARENT_RUN="$LAB" \
    OPS_EXPECTED_MINUTES=${MINUTES:-40} OPS_FIRST_OUTPUT=900 OPS_STALL=1200 \
    bash /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/recipes/env/ops_job.sh "$PY" -u \
      research/token_routing/grad_concentration/gradmap.py --ckpt "$CK" --label "$LAB" \
      --shards /data/prasann/gradconc/shards --out-dir "$ROOT/out" ${ARGS:-} $EXTRA
done
echo "GRADMAP_DONE $JOBS"
