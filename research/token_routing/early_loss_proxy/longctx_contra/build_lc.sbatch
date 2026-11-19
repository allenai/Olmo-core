#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00
#SBATCH -J lc-build
#SBATCH -o /data/prasann/joblogs/lc-build-%j.out
# CPU job: gen -> tokenize -> build of longctx_contra/build_lc_pools.py from a staged code dir ($CODE).
set -eo pipefail
CODE=${CODE:?}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export TMPDIR=/data/prasann/tmp HF_HOME=/data/prasann/early_loss_proxy/v2build/hf PYTHONUNBUFFERED=1
cd "$CODE"
for st in ${STAGES:-gen tokenize build}; do
  echo "=== $st $(date +%T)"
  $PY research/token_routing/early_loss_proxy/longctx_contra/build_lc_pools.py $st ${LENS:+--lens $LENS} ${SIZES:+--sizes $SIZES} ${SUFFIX:+--suffix $SUFFIX}
done
echo "=== LC_BUILD_DONE $(date +%T)"
