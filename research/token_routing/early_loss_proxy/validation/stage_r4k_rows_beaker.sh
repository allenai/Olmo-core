#!/bin/bash
# CPU gantry job: stage the CTC-suite r4k rows (500 each, olmo-eval's pinned HF revision 04ab8600) of the four mtb16k
# tasks beside the r2k/r8k/r16k rows at $W/_eval_bundle/ctc_suite_rows, ADD-only (an existing file must hash-match the
# fresh fetch; r8k/r16k are re-fetched and compared), then run the ladder validator on ALL 500 rows of every
# 4k/8k/16k cell at query position AFTER against the mtb16k converter (train tokenizer Qwen3.5-0.8B-Base, query after):
# prefix parity, gold scores max / garbage 0, decode budget, prefill-length tail (sizes MAX_LENGTH).
# Runs the PUSHED commit.   bash research/token_routing/early_loss_proxy/validation/stage_r4k_rows_beaker.sh
set -euo pipefail
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
"$GANTRY" run --name "elpv-suite-r4k-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster 'ai2/jupiter*' \
  --gpus 0 --cpus 16 --memory 96GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --no-python \
  --weka oe-training-default:/weka/oe-training-default \
  --env W="$W" --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
PYB=/opt/conda/bin/python
$PYB -m pip install -q -e ctc/ 2>&1 | tail -1
$PYB -m pip install -q -e . 2>&1 | tail -1
$PYB -c "import pyarrow, huggingface_hub, ctc" || pip install -q pyarrow huggingface_hub
F=research/token_routing/ds64_fast8k/fetch_suite_eval_rows.py
TMP=/tmp/suite_elpv; rm -rf $TMP
$PYB $F --out $TMP --subsets contradiction_iid outlier reorder qdmatch_nq --rungs r4k r8k r16k || { echo "!!! fetch failed"; exit 1; }
cat $TMP/suite/MANIFEST.json
DST=$W/_eval_bundle/ctc_suite_rows
cd $TMP/suite
for f in */rung_*.jsonl; do
  if [ -s $DST/$f ]; then
    cmp -s $f $DST/$f && echo "[same] $f" || { echo "!!! $DST/$f exists and DIFFERS"; exit 1; }
  else mkdir -p $DST/$(dirname $f) && cp $f $DST/$f.tmp && mv $DST/$f.tmp $DST/$f && echo "[added] $f"; fi
done
cp MANIFEST.json $DST/MANIFEST_elpv.json
cd - >/dev/null
$PYB $F --verify $DST --manifest $DST/MANIFEST_elpv.json || { echo "!!! weka copy failed verification"; exit 1; }
mkdir -p $W/early_loss_proxy/validation
export PYTHONPATH=src:recipes/eval/legacy_native
V="research/token_routing/ds64_fast8k/validate_suite_ladder_tasks.py --rows-root $DST --tokenizer $W/hf_tokenizers/Qwen3.5-0.8B-Base --train-tokenizer $W/hf_tokenizers/Qwen3.5-0.8B-Base --query-position after --train-query-position after --rows 500 --rungs 4096 8192 16384"
$PYB $V --tasks reorder qdmatch_nq --base-tasks outlier contradiction --out $W/early_loss_proxy/validation/validate_suite_qafter.json; echo "validate rc=$?"
cat $W/early_loss_proxy/validation/validate_suite_qafter.json
echo STAGE_R4K_DONE' 2>&1 | tee "${LOG:-/dev/null}" | grep -oE "ex/[A-Z0-9]{26}" | head -1
