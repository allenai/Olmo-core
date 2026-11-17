#!/bin/bash
# CPU gantry job: stage the CTC-suite LONG rungs (r16k / r32k / r64k, 500 rows each, olmo-eval's pinned HF revision)
# beside the r2k/r8k rows already at $W/_eval_bundle/ctc_suite_rows (ADD-only: existing files are never
# replaced; an existing file must hash-match the fresh fetch), then run the ladder validator on ALL 500 rows of
# every long cell: eval-prefill vs training-row prefix parity at query position both (the mt11 shards' layout),
# gold scores max / garbage 0, decode budget, and the prefill-length tail that sizes the eval MAX_LENGTH.
# Runs the PUSHED commit.   bash research/token_routing/mt11_64k/stage_long_rows_beaker.sh
set -euo pipefail
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
LOG=research/token_routing/mt11_64k/launch_logs/stage_long_$(date +%m%d%H%M).log
mkdir -p "$(dirname $LOG)"
"$GANTRY" run --name "mt11-64k-suite-long-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster 'ai2/jupiter*' \
  --gpus 0 --cpus 16 --memory 96GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --no-python \
  --weka oe-training-default:/weka/oe-training-default \
  --env W="$W" --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
PYB=/opt/conda/bin/python
$PYB -m pip install -q -e ctc/ 2>&1 | tail -1
$PYB -c "import pyarrow, huggingface_hub, ctc" || pip install -q pyarrow huggingface_hub
F=research/token_routing/ds64_fast8k/fetch_suite_eval_rows.py
TMP=/tmp/suite_long; rm -rf $TMP
$PYB $F --out $TMP --rungs r16k r32k r64k --skip-missing || { echo "!!! fetch failed"; exit 1; }
cat $TMP/suite/MANIFEST.json
DST=$W/_eval_bundle/ctc_suite_rows
cd $TMP/suite
for f in */rung_*.jsonl; do
  if [ -s $DST/$f ]; then
    cmp -s $f $DST/$f && echo "[same] $f" || { echo "!!! $DST/$f exists and DIFFERS"; exit 1; }
  else mkdir -p $DST/$(dirname $f) && cp $f $DST/$f.tmp && mv $DST/$f.tmp $DST/$f && echo "[added] $f"; fi
done
cp MANIFEST.json $DST/MANIFEST_long.json
cd - >/dev/null
$PYB $F --verify $DST --manifest $DST/MANIFEST_long.json || { echo "!!! weka copy failed verification"; exit 1; }
mkdir -p $W/mt11_64k/reports
V="research/token_routing/ds64_fast8k/validate_suite_ladder_tasks.py --rows-root $DST --tokenizer $W/hf_tokenizers/Qwen3.5-0.8B-Base --train-tokenizer Qwen/Qwen3.5-4B --query-position both --train-query-position both --rows 500"
export PYTHONPATH=src:recipes/eval/legacy_native
$PYB $V --tasks hotpotqa qdmatch_nq --base-tasks nq outlier oolong contradiction --rungs 16384 32768 65536 --out $W/mt11_64k/reports/validate_long_a.json; echo "validate a rc=$?"
$PYB $V --tasks xabsence strmatch grouping --rungs 16384 32768 --out $W/mt11_64k/reports/validate_long_b.json; echo "validate b rc=$?"
$PYB $V --tasks reorder --rungs 16384 --out $W/mt11_64k/reports/validate_long_c.json; echo "validate c rc=$?"
echo STAGE_LONG_DONE' 2>&1 | tee "$LOG" | grep -oE "ex/[A-Z0-9]{26}" | head -1
