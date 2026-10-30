#!/bin/bash
# wait_all_final.sh <max minutes> <exp id>...: poll every 2 min; exit when ALL listed Beaker experiments finalize,
# printing "<id> <name> exit=<code>" for each. For run_in_background use only.
MAX=$1; shift
for _ in $(seq 1 $((MAX / 2))); do
  OUT=""; N=0
  for EX in "$@"; do
    R=$(~/.local/bin/beaker experiment get "$EX" --format json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; s=d['jobs'][-1]['status']
print(f\"{d['name']} exit={s.get('exitCode')}\" if s.get('finalized') else '')" 2>/dev/null)
    if [ -n "$R" ]; then N=$((N + 1)); OUT="$OUT$EX $R"$'\n'; fi
  done
  if [ "$N" -eq "$#" ]; then printf "%s" "$OUT"; echo ALL_FINAL; exit 0; fi
  sleep 120
done
echo TIMEOUT
