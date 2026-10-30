#!/bin/bash
# Wait for each decay-fix training run to finish, then launch its per-task ladder (4 x 2-GPU) and its answer-CE job
# (1 GPU), and register the CE job with ops. Bounded: gives up after MAX_MIN minutes. Login-node safe (beaker CLI only).
#   bash research/token_routing/decay_fix/orchestrate.sh run1 run2 ...      (run names in state.json)
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
D=$REPO/research/token_routing/decay_fix
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PATH=$HOME/.local/bin:$PATH
MAX_MIN=${MAX_MIN:-420}
declare -A done_
start=$(date +%s)
while :; do
  left=0
  for run in "$@"; do
    [ -n "${done_[$run]:-}" ] && continue
    ex=$(python3 -c "import json; print(json.load(open('$D/state.json'))['runs']['$run']['ex'])")
    st=$(beaker experiment get "$ex" --format json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin); e=d[0] if isinstance(d,list) else d
j=(e.get('jobs') or [{}])[-1].get('status',{})
print('done' if j.get('finalized') and j.get('exitCode',1)==0 else ('failed' if j.get('finalized') else 'running'))" 2>/dev/null)
    if [ "$st" = done ]; then
      echo "$(date '+%F %T') $run finished -> ladder + CE"
      (cd "$REPO" && $PY "$D/launch_decay.py" launch-ladder --runs "$run" 2>&1 | grep -- "->")
      (cd "$REPO" && PYTHONPATH=$REPO/src $PY "$D/launch_decay.py" ce --runs "$run" 2>&1 | grep -- "->")
      done_[$run]=1
    elif [ "$st" = failed ]; then
      echo "$(date '+%F %T') $run FAILED (beaker $ex)"; done_[$run]=1
    else
      left=$((left + 1))
    fi
  done
  [ $left -eq 0 ] && { echo "all finished"; exit 0; }
  [ $(( ($(date +%s) - start) / 60 )) -ge $MAX_MIN ] && { echo "gave up after $MAX_MIN min ($left left)"; exit 1; }
  sleep 300
done
