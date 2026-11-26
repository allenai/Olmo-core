#!/bin/bash
# CPU job: gen_rows.py gen + tok on a node-local work dir. Code staged at $CODE (git archive | srun tar -x).
#   sbatch -w mooney --export=ALL,CODE=/data/prasann/gradlen/code_<sha>,LENS=2k,... gen_rows.sbatch
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH -J gsl-gen
#SBATCH -o /data/prasann/joblogs/gsl-gen-%j.out
set -eo pipefail
CODE=${CODE:?}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export TMPDIR=/data/prasann/tmp HF_HOME=/data/prasann/early_loss_proxy/v2build/hf PYTHONUNBUFFERED=1
mkdir -p /data/prasann/gradlen
cd "$CODE"
for st in ${STAGES:-gen tok}; do
  echo "=== $st $(date +%T)"
  $PY research/token_routing/grad_concentration/length_sweep/gen_rows.py $st --lens "${LENS:-2k,4k,8k,16k,32k,64k,128k,256k}" --jobs ${JOBS:-16}
done
echo "=== GSL_GEN_DONE $(date +%T)"
