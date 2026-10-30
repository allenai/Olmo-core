#!/bin/bash
# probe.py on one checkpoint, 1 GPU on jsteinhardt (inputs staged by stage_s3_beaker.sh). Pulls the model-only
# checkpoint and the probe shards from S3 to node-local /data (boto3; shards once per node under flock), then runs.
# Code staged at $CODE first (login node):
#   git archive HEAD | srun -p jsteinhardt --qos=preemptive_high -w sneetches -c1 --mem=2G -t 5 \
#     bash --noprofile --norc -c 'mkdir -p $CODE && tar -x -C $CODE'
#   sbatch -w sneetches -J gc-<label> --export=ALL,CODE=...,LABEL=<label>,CKNAME=<s3 dir name>[,EXTRA=...] run_local.sbatch
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/%x-%j.out
set -eo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ROOT=${GCROOT:-/data/prasann/gradconc}  # balrog: /tmp/prasann/gradconc (no /data/prasann there)
CODE=${CODE:?set CODE}; LABEL=${LABEL:?set LABEL}; CKNAME=${CKNAME:?set CKNAME}
PY=${PY:-/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python}
export TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=$ROOT/tmp/triton HF_HOME=$ROOT/hf_cache PYTHONPATH="$CODE/src"
mkdir -p "$TMPDIR" "$ROOT/ckpts" "$ROOT/results/$LABEL" "$ROOT/logs"
cd "$CODE"
s3down() {  # s3down <local dir> <rel under _transfer/gradconc>
  "$PY" -u -c "import sys; sys.path.insert(0, 'research/token_routing/olmo3_cptdrop'); import s3_sync as s
s.ROOT = 'checkpoints/prasanns/_transfer/gradconc'; sys.argv = ['s3_sync', 'down', '$1', '$2']; s.main()" | tail -2
}
(
  flock 9
  for S in contradiction_fresh nq_fresh outlier_fresh oolong_fresh mt4f7808 mt4s1952; do
    [ -s "$ROOT/shards/$S/metadata.json" ] || s3down "$ROOT/shards/$S" "shards/$S"
  done
) 9>"$ROOT/shards.lock"
[ -s "$ROOT/ckpts/$CKNAME/model_and_optim/.metadata" ] || s3down "$ROOT/ckpts/$CKNAME" "$CKNAME"
ls "$ROOT/ckpts/$CKNAME/model_and_optim" | head -3
OPS_RUN_ID="gradconc-${LABEL}-${SLURM_JOB_ID}" OPS_EXPERIMENT=tokdrop_lr OPS_KIND=eval OPS_PARENT_RUN="$CKNAME" \
  OPS_CKPT_PATH="$ROOT/ckpts/$CKNAME" OPS_EXPECTED_MINUTES=${MINUTES:-150} OPS_FIRST_OUTPUT=1200 OPS_STALL=1800 \
  bash "$REPO/recipes/env/ops_job.sh" "$PY" -u research/token_routing/grad_concentration/probe.py \
    --ckpt "$ROOT/ckpts/$CKNAME/model_and_optim" --shards "$ROOT/shards" --label "$LABEL" \
    --out "$ROOT/results/$LABEL/probe.json" ${EXTRA:-}
echo "PROBE_DONE $LABEL"
