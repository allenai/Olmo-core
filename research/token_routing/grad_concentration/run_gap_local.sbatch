#!/bin/bash
# routed_probe.py on jsteinhardt (fallback when Beaker is backed up). Copies the code from SRC (a checkout on NFS, read
# once) to node /data, pulls the model-only checkpoint + probe shards from S3 (stage_s3_beaker.sh put them there), runs.
#   sbatch -w <node> -J gradgap-<label>-s<i> --export=ALL,SRC=<checkout>,LABEL=<label>,CKNAME=<s3 ckpt dir>,ARGS="..." \
#     run_gap_local.sbatch          (STAGE_ONLY=1: copy + download, no GPU work)
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/%x-%j.out
set -eo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ROOT=/data/prasann/gradconc
OUTROOT=/data/prasann/gradgap
SRC=${SRC:?set SRC}; LABEL=${LABEL:?set LABEL}; CKNAME=${CKNAME:?set CKNAME}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
CODE=$OUTROOT/code_${SLURM_JOB_ID}
mkdir -p "$CODE" "$OUTROOT/logs" "$ROOT/ckpts" "$OUTROOT/tmp" "$OUTROOT/results"
(cd "$SRC" && tar -c --exclude=.git --exclude='results*' src research/token_routing/grad_concentration \
  research/token_routing/olmo3_cptdrop/s3_sync.py) | tar -x -C "$CODE"
export TMPDIR=$OUTROOT/tmp TRITON_CACHE_DIR=$OUTROOT/tmp/triton HF_HOME=$OUTROOT/hf_cache PYTHONPATH="$CODE/src"
cd "$CODE"
s3down() {
  "$PY" -u -c "import sys; sys.path.insert(0, 'research/token_routing/olmo3_cptdrop'); import s3_sync as s
s.ROOT = 'checkpoints/prasanns/_transfer/gradconc'; sys.argv = ['s3_sync', 'down', '$1', '$2']; s.main()" | tail -1
}
(
  flock 9
  for S in contradiction_fresh nq_fresh outlier_fresh oolong_fresh mt4f7808 mt4s1952; do s3down "$ROOT/shards/$S" "shards/$S"; done
  s3down "$ROOT/ckpts/$CKNAME" "$CKNAME"
  for S in ${TRAIN_SHARDS:-}; do s3down "$ROOT/shards/$S" "shards/$S"; done
  # ADAM_S3=1: Adam factor files staged by stage_s3_beaker.sh (.../gradconc/adam/adam_<label>.pt)
  [ "${ADAM_S3:-0}" = 1 ] && s3down "$ROOT/adam" "adam"
) 9>"$ROOT/stage.lock"
if [ "${ADAM_S3:-0}" = 1 ]; then ARGS="${ARGS:-} --adam-factors $ROOT/adam/adam_${LABEL}.pt"; fi
# ADAM_FROM=<shared path>: node-local copy of the Adam factors, passed as --adam-factors
if [ -n "${ADAM_FROM:-}" ]; then cp "$ADAM_FROM" "$OUTROOT/adam_from_${SLURM_JOB_ID}.pt"; ARGS="${ARGS:-} --adam-factors $OUTROOT/adam_from_${SLURM_JOB_ID}.pt"; fi
[ "${STAGE_ONLY:-0}" = 1 ] && { du -sh $ROOT/*; echo STAGE_DONE; exit 0; }
OPS_RUN_ID="gradgap-${LABEL}-${SLURM_JOB_ID}" OPS_EXPERIMENT=tokdrop_lr OPS_KIND=eval OPS_PARENT_RUN="$CKNAME" \
  OPS_EXPECTED_MINUTES=${MINUTES:-150} OPS_FIRST_OUTPUT=1200 OPS_STALL=1800 \
  bash "$REPO/recipes/env/ops_job.sh" "$PY" -u research/token_routing/grad_concentration/routed_probe.py \
    --ckpt "$ROOT/ckpts/$CKNAME/model_and_optim" --shards "$ROOT/shards" --label "$LABEL" \
    --save-dir "$OUTROOT/sketches" --out "$OUTROOT/results/${LABEL}_${SLURM_JOB_ID}.json" ${ARGS:-}
# ADAM_PUBLISH=<path>: copy the factor file (small) to a shared path for jobs on other nodes
[ -n "${ADAM_PUBLISH:-}" ] && cp "$OUTROOT/adam_${LABEL}.pt" "$ADAM_PUBLISH" && echo "published $ADAM_PUBLISH"
echo "PROBE_DONE $LABEL"
