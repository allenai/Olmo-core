#!/bin/bash
# Launch the remaining eval runs (4 jobs each) in waves, keeping at most 8 of this validation's eval jobs active.
# Bounded (exits after all are launched or after MAX_S). Run in the background from the repo root:
#   bash research/token_routing/early_loss_proxy/validation/eval_waves.sh run1,run2,...
set -u
RUNS=${1:?comma list of runs}; MAX_S=${MAX_S:-21600}; T0=$(date +%s)
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
L=research/token_routing/early_loss_proxy/validation/launch_val.py
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=src
for run in ${RUNS//,/ }; do
  while :; do
    active=$($PY $L status 2>/dev/null | awk '$1=="eval" && ($3=="S" || $3=="R" || $3=="P")' | wc -l)
    [ "$active" -le 4 ] && break
    [ $(( $(date +%s) - T0 )) -gt "$MAX_S" ] && { echo "timeout waiting to launch $run"; exit 1; }
    sleep 300
  done
  $PY $L evals --runs "$run" --min-runtime 2h 2>&1 | grep "^eval\|skip\|wait"
done
echo WAVES_DONE
