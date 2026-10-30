#!/bin/bash
# Fetch the KV-only probe results (jupiter fleet in gfkv_jobs.txt; ceres twins were cancelled) and summarize.
#   bash research/token_routing/online_router/fetch_gfkv.sh <out dir>
set -uo pipefail
cd "$(dirname "$0")"
export PATH=$HOME/.local/bin:$PATH
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
OUT=${1:?out dir}; mkdir -p "$OUT"
grep -v CANCELLED gfkv_jobs.txt | while read -r name id; do
  [ -s "$OUT/$name/grad-fidelity2.json" ] && continue
  ds=$(beaker experiment get "$id" --format json 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['jobs'][-1]['result']['beaker'])" 2>/dev/null) || continue
  beaker dataset fetch "$ds" -o "$OUT/$name" >/dev/null 2>&1 && echo "fetched $name"
done
$PY grad_fidelity2_merge.py --parts "$OUT" --out "$OUT/merged.json" --md "$OUT/table.md" | head -1
for v in trained all; do $PY kv_probe_table.py --merged "$OUT/merged.json" --view $v --md "$OUT/kv_table_$v.md" >/dev/null; done
cat "$OUT/kv_table_trained.md"
