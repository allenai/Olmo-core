#!/bin/bash
# Fetch the answer-CE JSONs of every finished CE job in state.json into results/ce/<job>.json (Beaker result datasets).
#   bash research/token_routing/decay_fix/fetch_ce.sh
set -u
D=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$D/results/ce"
python3 -c "
import json; d=json.load(open('$D/state.json'))
for k, v in d.get('ce', {}).items(): print(k, v['ex'])
" | while read job ex; do
  [ -s "$D/results/ce/$job.json" ] && continue
  tmp=$(mktemp -d)
  if beaker experiment results "$ex" --output "$tmp" >/dev/null 2>&1; then
    f=$(find "$tmp" -name 'score-answer-ce.json' | head -1)
    if [ -n "$f" ]; then cp "$f" "$D/results/ce/$job.json"; echo "fetched $job"; else echo "no result yet: $job"; fi
  else
    echo "not finished: $job"
  fi
done
