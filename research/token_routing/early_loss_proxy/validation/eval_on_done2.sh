#!/bin/bash
# Phase 2 (loop/validate2): launch each run's generation evals as soon as its training job finishes (exit 0).
# Bounded (MAX_S); run in the background from the repo root:
#   bash research/token_routing/early_loss_proxy/validation/eval_on_done2.sh
# Tasks: contradiction, outlier, qdmatch_nq (+ reorder on the p50 run only, to re-check the grader).
set -u
MAX_S=${MAX_S:-21600}; T0=$(date +%s)
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
L=research/token_routing/early_loss_proxy/validation/launch_val.py
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=src ELPV_PHASE=2
export BEAKER_HOSTNAMES=${BEAKER_HOSTNAMES:-@research/token_routing/early_loss_proxy/validation/jupiter_hosts_ok.txt}
declare -A launched
while :; do
  st=$($PY $L status 2>/dev/null)
  pending=0
  while read -r kind name state ex; do
    [ "$kind" = run ] || continue
    if [ "$state" = DONE ] && [ -z "${launched[$name]:-}" ]; then
      tasks=contradiction,outlier,qdmatch_nq
      [[ $name == elpv2-p50* ]] && tasks=$tasks,reorder
      $PY $L evals --runs "$name" --tasks $tasks --min-runtime 2h 2>&1 | grep "^eval\|skip\|wait"
      launched[$name]=1
    elif [ "$state" = FAILED ]; then
      echo "TRAIN FAILED $name $ex"; launched[$name]=failed
    elif [ -z "${launched[$name]:-}" ]; then
      pending=1
    fi
  done <<< "$st"
  [ $pending = 0 ] && { echo ALL_EVALS_LAUNCHED; exit 0; }
  [ $(( $(date +%s) - T0 )) -gt "$MAX_S" ] && { echo "timeout"; exit 1; }
  sleep 300
done
