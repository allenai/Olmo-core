#!/bin/bash
# FRESH, larger 8k pools for the 4-task mix, so cheap router arms can be compared to dense at matched FLOPs by
# training on MORE NEW rows (Prasann 2026-10-24: "don't repeat epochs, just build more data").
#
# The existing mt4s<rows> shards are first-N prefixes of the ds64 8k pools (~2,700 rows/task, 1,953 used), so they
# stop at 1,952 rows/task. This builds a NEW pool per task with the same generator call ds64 used
# (research/token_routing/ds64/build_ds64_data_beaker.sh: `ctc-data build --rungs 8k --pool auto`), a different
# seed, and N rows; tokenizes it exactly like _build_fast8k_inner.sh (oolong with --gold-sidecar-empty-for-missing,
# as build_mt4_beaker.sh); checks train/eval overlap against the 8k and 2k eval rungs; then (MERGE=1) merges
# first-N prefixes into $F/shards/mt4f<rows> for each ROWS.
#
#   TASK=nq bash research/token_routing/ds64_fast8k/build_mt4_fresh_beaker.sh            # one job per task
#   MERGE=1 ROWS="1952 3904 5856 7808" bash research/token_routing/ds64_fast8k/build_mt4_fresh_beaker.sh
#   MERGE=1 SKIP_STATS=1 ROWS="488 976" bash ...   # merge only (overlap already checked on these pools; 2026-11-10)
set -uo pipefail
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
N="${N:-8200}"            # pool rows per task (8k rung); merged sizes use first-N prefixes
SEED="${SEED:-5310}"      # ds64 used 4200+i; the fresh pool must differ
TASK="${TASK:-}"; MERGE="${MERGE:-0}"; SKIP_STATS="${SKIP_STATS:-0}"; ROWS="${ROWS:-1952 3904 5856 7808}"
NAME="fast8k-fresh-${TASK:-merge}-$(date +%m%d%H%M)"
gantry run --name "$NAME" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 8 --memory 64GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env WEKA="$WEKA" --env N="$N" --env SEED="$SEED" --env TASK="$TASK" --env MERGE="$MERGE" --env SKIP_STATS="$SKIP_STATS" --env ROWS="$ROWS" \
  --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
PYB=/opt/conda/bin/python
F=$WEKA/ds64/fast8k
TOK=$WEKA/hf_tokenizers/Qwen3.5-0.8B-Base
$PYB -m pip install -q -e . -e ./ctc 2>&1 | tail -2
$PYB -c "import numpy, transformers, torch, olmo_core, ctc.data.cli" || { echo "!!! deps missing"; exit 1; }
if [ "$MERGE" = 1 ]; then
  [ "$SKIP_STATS" = 1 ] || for T in contradiction nq outlier oolong; do  # train/eval overlap of each fresh pool (must be ~0) before merging
    for RUNG in 8192 2048; do
      $PYB research/token_routing/ds64_fast2k/fast2k_stats.py --arm-jsonl $F/pool_fresh_$T/$T/train.jsonl \
        --shard-dir $F/shards/${T}_fresh --eval-jsonl $WEKA/outlier_lengthmix/eval_rungs/$T/rung_$RUNG.jsonl \
        --out $F/shards/${T}_fresh/fast8k_stats_$RUNG.json > /tmp/st.json || { echo "!!! stats failed $T $RUNG"; exit 1; }
      $PYB -c "import json,sys;d=json.load(open(\"/tmp/st.json\"));o=d.get(\"overlap\",d);print(\"[overlap] $T $RUNG\",{k:v for k,v in o.items() if \"overlap\" in k})"
    done
  done
  for R in $ROWS; do
    OUT=$F/shards/mt4f$R
    if [ -s $OUT/metadata.json ]; then echo "[skip] $OUT"; continue; fi
    $PYB research/token_routing/ds64_fast8k/merge_tasks.py --out $OUT --rows-per-src $R --name mt4 \
      --src $F/shards/contradiction_fresh $F/shards/nq_fresh $F/shards/outlier_fresh $F/shards/oolong_fresh \
      || { echo "!!! merge failed $R"; exit 1; }
    echo "[merge] mt4f$R"
  done
  echo "=== MERGE DONE ==="; exit 0
fi
case "$TASK" in
  nq) CONV=retrieval; CHUNK=document; EXTRA="" ;;
  oolong) CONV=oolong; CHUNK=line; EXTRA="--gold-sidecar-empty-for-missing" ;;
  *) CONV=$TASK; CHUNK=document; EXTRA="" ;;
esac
POOL=$F/pool_fresh_$TASK
if [ ! -s $POOL/$TASK/train.jsonl ] || [ $(wc -l < $POOL/$TASK/train.jsonl) -lt $N ]; then
  echo "--- pool $TASK 8k: $N rows, seed $SEED $(date +%T) ---"
  $PYB -m ctc.data.cli build --task $TASK --out $POOL --split train --rungs 8k --train $N --seed $SEED --pool auto --force \
    || { echo "!!! pool FAILED"; exit 1; }
fi
echo "pool rows: $(wc -l < $POOL/$TASK/train.jsonl)"
OUT=$F/shards/${TASK}_fresh
if [ -s $OUT/metadata.json ]; then echo "[skip] $OUT tokenized"; else
  mkdir -p $OUT
  PYTHONPATH=src $PYB recipes/data/convert_unified_to_document_landmark.py \
    --input-jsonl $POOL/$TASK/train.jsonl --task $CONV --out-dir $OUT --emit dense --marker-set qwen3_5 \
    --tokenizer $TOK --seq-len 24576 --query-position after --cot-mode none \
    --chunk-by $CHUNK --emit-gold-sidecar $EXTRA --num-proc 8 || { echo "!!! tokenize FAILED"; exit 1; }
fi
for RUNG in 8192 2048; do
  $PYB research/token_routing/ds64_fast2k/fast2k_stats.py --arm-jsonl $POOL/$TASK/train.jsonl --shard-dir $OUT \
    --eval-jsonl $WEKA/outlier_lengthmix/eval_rungs/$TASK/rung_$RUNG.jsonl --out $OUT/fast8k_stats_$RUNG.json || exit 1
done
echo "=== DONE $TASK ==="'
