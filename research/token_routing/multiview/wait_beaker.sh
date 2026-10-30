#!/bin/bash
# wait_beaker.sh <experiment id> <regex> <max minutes>: poll the Beaker log every 60 s until <regex> appears or the job
# finalizes; print the matching / last lines. For run_in_background use only.
EX=$1; PAT=$2; MAX=${3:-30}
B=~/.local/bin/beaker
for _ in $(seq 1 "$MAX"); do
  LOG=$($B experiment logs "$EX" 2>/dev/null)
  if echo "$LOG" | grep -aqE "$PAT"; then echo "$LOG" | grep -aE "$PAT" | tail -5 | cut -c1-400; exit 0; fi
  ST=$($B experiment get "$EX" --format json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print('final' if d['jobs'][-1]['status'].get('finalized') else 'live')" 2>/dev/null)
  if [ "$ST" = final ]; then echo "FINALIZED"; echo "$LOG" | grep -aE "Error|rror:" | tail -5 | cut -c1-400; exit 1; fi
  sleep 60
done
echo "TIMEOUT"
