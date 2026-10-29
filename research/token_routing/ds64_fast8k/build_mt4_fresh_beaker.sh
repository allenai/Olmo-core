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
#
# FAMILY=qwen3 (2026-10-29, Qwen3-4B recipe study): re-tokenizes the SAME fresh pools (same rows, so a mt4f<N>q3
# shard holds the rows of mt4f<N>) with the Qwen3 tokenizer + --marker-set qwen3, into ${T}_fresh_q3 and
# mt4f<rows>q3. The tokenizer is a NOTHINK copy of Qwen/Qwen3-4B (generation prompt always ends with the empty
# think block, as Qwen3.5's does; the stock template puts <think></think> in the answer loss span -- memory
# qwen3-stock-template-think-in-answer-mask), built on weka by the job if missing and checked before use.
#   FAMILY=qwen3 TASK=all bash research/token_routing/ds64_fast8k/build_mt4_fresh_beaker.sh   # 4 tasks + merge, one job
set -uo pipefail
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
N="${N:-8200}"            # pool rows per task (8k rung); merged sizes use first-N prefixes
SEED="${SEED:-5310}"      # ds64 used 4200+i; the fresh pool must differ
TASK="${TASK:-}"; MERGE="${MERGE:-0}"; ROWS="${ROWS:-1952 3904 5856 7808}"; FAMILY="${FAMILY:-qwen3_5}"
NAME="fast8k-fresh-${FAMILY}-${TASK:-merge}-$(date +%m%d%H%M)"
CPUS=8; [ "$TASK" = all ] && CPUS=32
gantry run --name "$NAME" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus $CPUS --memory 128GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env WEKA="$WEKA" --env N="$N" --env SEED="$SEED" --env TASK="$TASK" --env MERGE="$MERGE" --env ROWS="$ROWS" --env FAMILY="$FAMILY" \
  --allow-dirty --timeout 0 --yes -- bash research/token_routing/ds64_fast8k/_build_mt4_fresh_inner.sh
