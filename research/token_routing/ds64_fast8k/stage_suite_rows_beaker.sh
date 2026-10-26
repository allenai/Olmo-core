#!/bin/bash
# Stage the CTC-suite eval rows (11 mt11 subsets incl. contradiction_iid and rerank, r2k + r8k, 500 rows each, at
# olmo-eval's pinned HF revision) onto weka for the Beaker mt11 evals:
#   $WEKA/_eval_bundle/ctc_suite_rows/<subset>/rung_<tokens>.jsonl  (+ MANIFEST.json)
# A CPU-only gantry job: fetch_suite_eval_rows.py into node /tmp, verify against the checked-in
# suite_rows_MANIFEST.json (an independent fetch: same rows, same sha256), copy to weka, verify the
# weka copy again (add-only: existing files, incl. the 64k study's long rungs, are never replaced). Gantry runs the PUSHED commit.
#
#   bash research/token_routing/ds64_fast8k/stage_suite_rows_beaker.sh
set -uo pipefail
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
"$GANTRY" run --name "suite-rows-stage-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' --cluster ai2/phobos --cluster ai2/hammond \
  --gpus 0 --cpus 4 --memory 32GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env WEKA="$WEKA" --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
PYB=/opt/conda/bin/python
F=research/token_routing/ds64_fast8k/fetch_suite_eval_rows.py
$PYB -c "import pyarrow, huggingface_hub" 2>/dev/null || pip install -q pyarrow huggingface_hub || { echo "!!! deps"; exit 1; }
TMP=/tmp/suite_fetch; rm -rf $TMP
$PYB $F --out $TMP || { echo "!!! fetch failed"; exit 1; }
$PYB $F --verify $TMP/suite || { echo "!!! local fetch differs from suite_rows_MANIFEST.json"; exit 1; }
DST=$WEKA/_eval_bundle/ctc_suite_rows
# ADD-ONLY merge into $DST. Never swap or replace the directory: it also holds the r16k/r32k/r64k long-rung files
# staged by research/token_routing/mt11_64k/stage_long_rows_beaker.sh (a whole-dir swap on 2026-10-26 deleted them).
mkdir -p $DST || exit 1
for f in $(cd $TMP/suite && find . -type f); do
  if [ -e "$DST/$f" ]; then
    cmp -s "$TMP/suite/$f" "$DST/$f" || { echo "!!! $f differs from the staged copy; not overwriting"; exit 1; }
  else
    mkdir -p "$(dirname "$DST/$f")" && cp "$TMP/suite/$f" "$DST/$f" && echo "[added] $f"
  fi
done
$PYB $F --verify $DST || exit 1
ls -la $DST; echo STAGE_DONE'
