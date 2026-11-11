#!/bin/bash
# Block until a Beaker experiment finishes (bounded): prints its final status. Usage: wait_beaker.sh <experiment id> [max minutes]
EX=$1; MAX=${2:-240}
for i in $(seq 1 $MAX); do
  S=$(timeout 60 ~/.local/bin/beaker experiment get "$EX" --format json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)[0]; j=d['jobs'][-1]; st=j.get('status',{})
print('done' if st.get('finalized') else 'running', st.get('exitCode'))" 2>/dev/null)
  case "$S" in done*) echo "$EX $S"; exit 0;; esac
  sleep 60
done
echo "$EX TIMEOUT"
