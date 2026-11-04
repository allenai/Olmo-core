#!/bin/bash
# CPU gantry job: the two pre-training checks, with the repo + ctc pip-installed (olmo_core imports need its deps):
#   1. verify_shard.py on the built mt11 64k shard (rows, length bins, gold-sidecar lookup as the trainer does it)
#   2. validate_suite_ladder_tasks.py on ALL 500 rows of every staged long suite cell (r16k/r32k/r64k; prefix
#      parity at query position both, grading, decode budget, prefill-length tail)
# Runs the PUSHED commit.   NAME=mt11L64 bash research/token_routing/mt11_64k/check_beaker.sh
set -euo pipefail
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
NAME="${NAME:-mt11L64}"; DO="${DO:-shard,long}"; SEQ="${SEQ:-65536}"
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
LOG=research/token_routing/mt11_64k/launch_logs/check_${NAME}_$(date +%m%d%H%M).log
mkdir -p "$(dirname $LOG)"
"$GANTRY" run --name "mt11-64k-check-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster 'ai2/jupiter*' \
  --gpus 0 --cpus 16 --memory 96GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --no-python \
  --weka oe-training-default:/weka/oe-training-default \
  --env W="$W" --env NAME="$NAME" --env DO="$DO" --env SEQ="$SEQ" --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
PYB=/opt/conda/bin/python
$PYB -m pip install -q -e . -e ctc/ 2>&1 | tail -2
$PYB -c "import olmo_core.config, ctc" || { echo "!!! deps missing"; exit 1; }
mkdir -p $W/mt11_64k/reports
export PYTHONPATH=src:recipes/eval/legacy_native
case ",$DO," in *,shard,*)
  $PYB research/token_routing/mt11_64k/verify_shard.py --shard $W/mt11_64k/shards/$NAME --seq-len $SEQ --out $W/mt11_64k/reports/verify_$NAME.json
  echo "verify rc=$?" ;; esac
case ",$DO," in *,long,*)
  DST=$W/_eval_bundle/ctc_suite_rows
  V="research/token_routing/ds64_fast8k/validate_suite_ladder_tasks.py --rows-root $DST --tokenizer $W/hf_tokenizers/Qwen3.5-0.8B-Base --train-tokenizer Qwen/Qwen3.5-4B --query-position both --train-query-position both --rows 500"
  $PYB $V --tasks hotpotqa qdmatch_nq --base-tasks nq outlier oolong contradiction --rungs 16384 32768 65536 --out $W/mt11_64k/reports/validate_long_a.json; echo "validate a rc=$?"
  $PYB $V --tasks xabsence strmatch grouping --rungs 16384 32768 --out $W/mt11_64k/reports/validate_long_b.json; echo "validate b rc=$?"
  $PYB $V --tasks reorder --rungs 16384 --out $W/mt11_64k/reports/validate_long_c.json; echo "validate c rc=$?" ;; esac
echo CHECK_DONE' 2>&1 | tee "$LOG" | grep -oE "ex/[A-Z0-9]{26}" | head -1
