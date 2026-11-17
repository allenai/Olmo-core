#!/bin/bash
# CPU gantry job: build the mt11 64k training shard from setA (seed 42) gold shards, then verify it.
#   source: $W/ctc_sft_sets/setA_max20_evaliid/shards_qwen3_5_gold/<task>  (all 11 tasks, full gold sidecars,
#           query position both, Qwen3.5 markers; rows 2k..256k)
#   mix:    per task, rows <= 65,536 tokens, ds64's short-heavy token shares over 6 length bins
#           (<=3k 30% / <=6k 20% / <=12k 15% / <=24k 13% / <=48k 12% / <=64k 10%) of TOKENS_PER_SRC
#   out:    $W/mt11_64k/shards/$NAME  (+ reports/verify_$NAME.json)
# Runs the PUSHED commit.   NAME=mt11L64 TOKENS_PER_SRC=64000000 bash research/token_routing/mt11_64k/build_beaker.sh
set -euo pipefail
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
NAME="${NAME:-mt11L64}"; TOKENS_PER_SRC="${TOKENS_PER_SRC:-64000000}"; SEED="${SEED:-20261023}"
MIX="${MIX:-3072:0.30,6144:0.20,12288:0.15,24576:0.13,49152:0.12,65536:0.10}"
#: row cap (training window). mt11L16: MAXLEN=16384 MIX=3072:1,6144:1,12288:1,16384:1 TOKENS_PER_SRC=1000000000000 = every row <= 16k
MAXLEN="${MAXLEN:-65536}"
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
LOG=research/token_routing/mt11_64k/launch_logs/build_${NAME}_$(date +%m%d%H%M).log
mkdir -p "$(dirname $LOG)"
"$GANTRY" run --name "mt11-64k-build-$NAME-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster 'ai2/jupiter*' \
  --gpus 0 --cpus 8 --memory 96GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --no-python \
  --weka oe-training-default:/weka/oe-training-default \
  --env W="$W" --env NAME="$NAME" --env TOK="$TOKENS_PER_SRC" --env SEED="$SEED" --env MIX="$MIX" --env MAXLEN="$MAXLEN" \
  --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
PYB=/opt/conda/bin/python
SRC=$W/ctc_sft_sets/setA_max20_evaliid/shards_qwen3_5_gold
OUT=$W/mt11_64k/shards/$NAME; mkdir -p $W/mt11_64k/reports
if [ -s $OUT/metadata.json ]; then echo "[skip] $OUT built"; else
  $PYB research/token_routing/ds64_fast8k/merge_tasks.py --out $OUT --name mt11 --max-row-len $MAXLEN \
    --length-mix "$MIX" --tokens-per-src $TOK --sample-seed $SEED \
    --src $(for t in nq hotpotqa qdmatch_nq outlier oolong contradiction xabsence reorder rerank strmatch grouping; do echo $SRC/$t; done) \
    || { echo "!!! merge failed"; exit 1; }
fi
$PYB -c "import json;m=json.load(open(\"$OUT/metadata.json\"));print({k:m.get(k) for k in (\"num_instances\",\"num_tokens\",\"num_loss_tokens\",\"max_example_len\",\"min_example_len\",\"query_position\",\"marker_set\")});[print(x[\"src\"].split(\"/\")[-1],x[\"num_instances\"],x.get(\"num_tokens\"),[(b[\"hi\"],b[\"rows\"],b[\"tokens\"],b[\"available_rows\"]) for b in x.get(\"length_mix\",[])]) for x in m[\"merged_from\"]]"
PYTHONPATH=src $PYB research/token_routing/mt11_64k/verify_shard.py --shard $OUT --seq-len $MAXLEN --out $W/mt11_64k/reports/verify_$NAME.json
echo "verify rc=$?"; du -sh $OUT; echo BUILD_DONE' 2>&1 | tee "$LOG" | grep -oE "ex/[A-Z0-9]{26}" | head -1
