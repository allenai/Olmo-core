#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00
#SBATCH -J kl-build
#SBATCH -o /data/prasann/joblogs/kl-build-%j.out
# CPU job: gen -> heldout -> tokenize -> build -> order of keepfrac_len/build_kl_pools.py from a staged code dir ($CODE).
set -eo pipefail
CODE=${CODE:?}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export TMPDIR=/data/prasann/tmp HF_HOME=/data/prasann/early_loss_proxy/v2build/hf PYTHONUNBUFFERED=1
cd "$CODE"
for st in ${STAGES:-gen heldout tokenize build order}; do
  echo "=== $st $(date +%T)"
  $PY research/token_routing/early_loss_proxy/keepfrac_len/build_kl_pools.py $st
done
echo "=== KL_BUILD_DONE $(date +%T)"
