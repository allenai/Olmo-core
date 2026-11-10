#!/bin/bash
# cpt_gradmap.py (CPT per-token x per-block gradient maps, OLMo 3 7B) on one GPU of a jsteinhardt node that holds the
# checkpoints on /data (cubbins: runs/ + base/; mooney: base/ only). Code is tarred once from SRC (a checkout) into a
# fresh node-local dir. One sbatch = one checkpoint x one dev set:
#   sbatch -w cubbins -J cptgm-<label> --export=ALL,SRC=<checkout>,CK=runs/o3cd-Dlr20-480,YF=4,DEV=dev32k_1024,ROWS=1+7,LABEL=dlr480_32k run_cpt_gradmap.sbatch
# ROWS uses '+' between row indices (sbatch --export splits on commas). CK / DEV are relative to /data/prasann/olmo3_cptdrop (CK=base/stage2-step47684 for the base). ARGS = extra flags.
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/data/prasann/grad_composition/logs/%x-%j.out
set -eo pipefail
SRC=${SRC:?set SRC}; CK=${CK:?set CK}; YF=${YF:?set YF}; DEV=${DEV:?set DEV}; ROWS=${ROWS:?set ROWS}; LABEL=${LABEL:?set LABEL}
ROWS=${ROWS//+/,}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
ROOT=/data/prasann/olmo3_cptdrop
WORK=/data/prasann/grad_composition
OUT=${OUT:-$WORK/cpt_gradmap}
CODE=$WORK/code_gm_${SLURM_JOB_ID}
mkdir -p "$CODE" "$WORK/tmp" "$WORK/logs" "$OUT"
(cd "$SRC" && tar -c --exclude='__pycache__' src research/token_routing/olmo3_cptdrop \
   research/token_routing/grad_composition/cpt_probe.py research/token_routing/grad_concentration/cpt_gradmap.py) | tar -x -C "$CODE"
export TMPDIR=$WORK/tmp TRITON_CACHE_DIR=$WORK/tmp/triton HF_HOME=$ROOT/hf_cache PYTHONPATH="$CODE/src"
cd "$CODE"
nvidia-smi --query-gpu=index,memory.used --format=csv
OPS_RUN_ID="cptgm-${LABEL}-${SLURM_JOB_ID}" OPS_EXPERIMENT=grad_composition OPS_KIND=eval OPS_PARENT_RUN="$(basename "$CK")" \
  OPS_CKPT_PATH="$ROOT/$CK" OPS_EXPECTED_MINUTES=${MINUTES:-30} OPS_FIRST_OUTPUT=900 OPS_STALL=1500 \
  bash /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/recipes/env/ops_job.sh "$PY" -u \
    research/token_routing/grad_concentration/cpt_gradmap.py --ckpt "$ROOT/$CK" --yarn-factor "$YF" \
    --dev "$ROOT/data/$DEV" --rows "$ROWS" --label "$LABEL" --out-dir "$OUT" ${ARGS:-}
echo "CPTGM_DONE $LABEL"
