#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00
#SBATCH -J ng-build
#SBATCH -o /data/prasann/joblogs/ng-build-%j.out
# CPU job: stages of random_doc_drop/build_ng_pools.py from a staged code dir ($CODE). STAGES="gen base" or
# VARIANTS="rddA_s0:0.05:0.5:0.15:0 ..." (name:lo:hi:whole:seed).
set -eo pipefail
CODE=${CODE:?}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export TMPDIR=/data/prasann/tmp HF_HOME=/data/prasann/early_loss_proxy/v2build/hf PYTHONUNBUFFERED=1
cd "$CODE"
B=research/token_routing/early_loss_proxy/random_doc_drop/build_ng_pools.py
for st in ${STAGES:-}; do
  echo "=== $st $(date +%T)"; $PY $B $st ${STAGE_ARGS:-}
done
for v in ${VARIANTS:-}; do
  IFS=: read -r N LO HI WH SD <<< "$v"
  echo "=== variant $v $(date +%T)"; $PY $B variants --name $N --lo $LO --hi $HI --whole $WH --seed $SD
done
[ -n "${ORDER:-}" ] && $PY $B order --pools "$ORDER"
echo "=== NG_BUILD_DONE $(date +%T)"
