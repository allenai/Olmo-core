#!/bin/bash
# CPU gantry job: inspect candidate source shards for the mt11 64k mix (inspect_sources.py) + list what exists.
# Runs the PUSHED commit of this worktree's branch.   bash research/token_routing/mt11_64k/inspect_beaker.sh
set -euo pipefail
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
gantry run --name "mt11-64k-inspect-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster 'ai2/jupiter*' \
  --gpus 0 --cpus 8 --memory 64GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --no-python \
  --weka oe-training-default:/weka/oe-training-default \
  --env W="$W" --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
S=$W/ctc_sft_sets
echo "== ctc_sft_sets:"; ls $S
for d in $S/setA_max20_evaliid $S/setA_seed43 $S/setA_seed44 $S/setA_seed43/setA_max20_evaliid $S/setA_seed44/setA_max20_evaliid $S/setA_pareto; do echo "== $d"; ls $d 2>&1 | head -30; done
echo "== suite rows staged:"; ls -R $W/_eval_bundle/ctc_suite_rows 2>&1 | head -80
python3 -c "import json;m=json.load(open(\"$S/setA_max20_evaliid/manifest.json\"));print(json.dumps({k:v for k,v in m.items() if k not in (\"cells\",)} ,indent=0)[:3000]);c=m.get(\"cells\") or m.get(\"builds\") or [];print(len(c));[print(x.get(\"task\"),x.get(\"bucket\"),x.get(\"want\"),x.get(\"rows\"),x.get(\"status\")) for x in c]" 2>&1 | head -150
python3 research/token_routing/mt11_64k/inspect_sources.py --roots $S/setA_max20_evaliid/shards_qwen3_5_gold $S/setA_max20_evaliid/shards_qwen35_256k $S/setA_seed43/setA_max20_evaliid/shards_qwen35_256k $S/setA_seed44/setA_max20_evaliid/shards_qwen35_256k --out $W/mt11_64k/reports/inspect_sources.json
echo INSPECT_DONE'
