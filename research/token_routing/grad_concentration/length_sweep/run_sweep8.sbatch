#!/bin/bash
# sweep_probe.py over 8 H200s on any free jsteinhardt H200 node (blocks 7-31 spread over the GPUs), for rows that do
# not fit one GPU (> ~350k tokens). Stages code (git archive of $SHA), the checkpoint and the rung's rows from S3.
# Env: SHA LABEL RUNG ROWS YARN [STEPS] [TASKS]
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclude=balrog,saruman,shadowfax,smaug
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/%x-%j.out
set -eo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
SHA=${SHA:?}; CODE=/data/prasann/gradlen/code_${SHA:0:9}; PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
CK=${CKNAME:-m64-dense-f1-lr5e-06}; Q=${QPOS:-both}; TASKS=${TASKS:-contradiction,nq,outlier,oolong}
export TMPDIR=/data/prasann/gradlen/tmp TRITON_CACHE_DIR=/data/prasann/gradlen/tmp/triton PYTHONPATH="$CODE/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TMPDIR /data/prasann/gradlen/results/$LABEL
[ -d "$CODE" ] || { mkdir -p "$CODE" && git -C "$REPO" archive "$SHA" | tar -x -C "$CODE"; }
cd "$CODE"
s3down() { "$PY" -u -c "import sys; sys.path.insert(0, 'research/token_routing/olmo3_cptdrop'); import s3_sync as s
s.ROOT = '$3'; sys.argv = ['s3_sync', 'down', '$1', '$2']; s.main()" | tail -1; }
s3down /data/prasann/gradconc/ckpts/$CK $CK checkpoints/prasanns/_transfer/gradconc
# TASK_RUNGS="contradiction=448k outlier=448k" overrides the rung per task
for t in ${TASKS//,/ }; do
  R=$RUNG; for kv in ${TASK_RUNGS:-}; do [ "${kv%%=*}" = "$t" ] && R=${kv#*=}; done
  s3down /data/prasann/gradlen/tok/$Q/${t}_$R tok/$Q/${t}_$R checkpoints/prasanns/_transfer/gradlen
done
OPS_RUN_ID="gsl-${LABEL}-${SLURM_JOB_ID}" OPS_EXPERIMENT=grad_composition OPS_KIND=eval OPS_PARENT_RUN="$CK" \
  OPS_EXPECTED_MINUTES=${MINUTES:-400} OPS_FIRST_OUTPUT=2400 OPS_STALL=3600 bash "$REPO/recipes/env/ops_job.sh" "$PY" -u \
  research/token_routing/grad_concentration/length_sweep/sweep_probe.py --ckpt /data/prasann/gradconc/ckpts/$CK/model_and_optim \
  --rows-from tok --tok-root /data/prasann/gradlen/tok/$Q --tasks $TASKS --rung $RUNG --rows $ROWS --devices 0,1,2,3,4,5,6,7 \
  --yarn ${YARN:-0} ${STEPS:+--steps $STEPS} ${TASK_RUNGS:+--task-rungs ${TASK_RUNGS// /,}} --out /data/prasann/gradlen/results/$LABEL
echo "GSL8_DONE $LABEL"
