#!/bin/bash
# CPU-only Beaker job (a few minutes): prints OK / MISSING (+ mode, owner) for every shared input the recipe
# reproduction reads, as seen from YOUR Beaker account. Needs gantry (`pip install beaker-gantry`) and a pushed commit.
#   BEAKER_WORKSPACE=ai2/<ws> BEAKER_BUDGET=ai2/<budget> bash research/token_routing/recipe_repro/check_access_beaker.sh
set -euo pipefail
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
gantry run --name "recipe-repro-access-$(date +%m%d%H%M)" \
  -w "${BEAKER_WORKSPACE:-ai2/flex2}" -b "${BEAKER_BUDGET:-ai2/oe-other}" \
  --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' --cluster ai2/neptune --cluster ai2/saturn \
  --gpus 0 --cpus 2 --memory 8GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --no-python \
  --weka oe-training-default:/weka/oe-training-default \
  --env W="$W" --allow-dirty --timeout 0 --yes -- bash -c '
id
chk() { if [ -r "$1" ]; then echo "OK      $(stat -c "%A %U" "$1") $1"; else echo "MISSING $1"; fi; }
chk $W/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim/.metadata
chk $W/hf_tokenizers/Qwen3.5-0.8B-Base/tokenizer.json
for r in 488 976 1952 3904 5856 7808; do chk $W/ds64/fast8k/shards/mt4f$r; ls $W/ds64/fast8k/shards/mt4f$r | head -3; done
chk $W/early_loss_proxy/shards/mtb16k; ls $W/early_loss_proxy/shards/mtb16k | head -5
chk $W/mt11_64k/shards/mt11L64; ls $W/mt11_64k/shards/mt11L64 | head -3
for f in outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl \
         outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl \
         _eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl \
         _eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl \
         _eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl \
         _eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl; do chk $W/$f; done
for s in contradiction_iid outlier reorder qdmatch_nq; do for t in 4096 8192 16384; do chk $W/_eval_bundle/ctc_suite_rows/$s/rung_$t.jsonl; done; done
for s in nq hotpotqa qdmatch_nq outlier oolong contradiction_iid xabsence reorder strmatch grouping; do ls $W/_eval_bundle/ctc_suite_rows/$s/ 2>&1 | tr "\n" " "; echo " <- $s"; done
chk $W/_eval_bundle; ls $W/_eval_bundle | head -20
echo ACCESS_CHECK_DONE' 2>&1 | grep -oE "beaker.org/ex/[A-Z0-9]{26}" | head -1
