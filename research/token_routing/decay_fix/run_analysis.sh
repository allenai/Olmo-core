#!/bin/bash
# FTM table for every decay-fix run that has answer-CE results (f1 from ops results).
#   bash research/token_routing/decay_fix/run_analysis.sh
D=$(cd "$(dirname "$0")" && pwd)
cd "$D"
EXTRA=$(python3 -c "
import json, glob
runs = []
for f in sorted(glob.glob('results/ce/*.json')):
    for k in json.load(open(f)):
        if '-tdf' in k: runs.append(k)
print(','.join(runs))")
CE=$(for f in results/ce/*.json; do printf -- "--ce %s " "$f"; done)
python3 analyze_decay.py --f1-ops $CE --pf-json results/pf.json --extra-runs "$EXTRA" --json-out results/ftm_all.json
