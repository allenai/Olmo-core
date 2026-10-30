#!/usr/bin/env bash
# Stage the Qwen3-4B recipe study's inputs (marker-repaired base + the mt4f<N>q3 shards + the nothink tokenizer) from
# weka to S3 so Berkeley slurm (jsteinhardt) can train when Beaker does not schedule. Then, on the target node:
#   AWS_PROFILE=S3 aws s3 sync s3://ai2-llm/checkpoints/prasanns/_transfer/q3recipe/ /data/prasann/q3recipe/
# (node-local /data, never /scratch or /net).
#   ITEMS="base mt4f1952q3" bash research/token_routing/ds64_fast8k/stage_q3_s3_beaker.sh
set -uo pipefail
P=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
ITEMS="${ITEMS:-base tok mt4f1952q3}"
read -r -d '' WORK <<'EOW'
set -uo pipefail
mkdir -p ~/.aws && printf '%s' "$AWS_CREDS" > ~/.aws/credentials && printf '%s' "$AWS_CFG" > ~/.aws/config
pip install -q awscli 2>&1 | tail -1
DST=s3://ai2-llm/checkpoints/prasanns/_transfer/q3recipe
for IT in $ITEMS; do
  case $IT in
    base) SRC=$P/ctc_suite/bases/qwen3-4b-base-trainedmark ;;
    tok) SRC=$P/hf_tokenizers/Qwen3-4B-nothink ;;
    evalfiles)  # the mt4 ladder rows (launch_fast8k.RUNGS), kept at their weka-relative paths under eval/
      for f in outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl \
               outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl \
               _eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl \
               _eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl \
               _eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl \
               _eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl; do
        AWS_PROFILE=S3 aws s3 cp $P/$f $DST/eval/$f --only-show-errors || { echo "!!! FAILED $f"; exit 1; }
      done; echo "SYNCED evalfiles"; continue ;;
    ckpt:*) SRC=$P/ctc_suite/ckpts/${IT#ckpt:}; IT=ckpts/${IT#ckpt:} ;;  # a Beaker-trained run, for a local eval
    *) SRC=$P/ds64/fast8k/shards/$IT ;;
  esac
  [ -d $SRC ] || { echo "!!! missing $SRC"; exit 1; }
  echo "--- $IT: $SRC ($(du -sh $SRC | cut -f1)) $(date +%T)"; ls $SRC | head
  AWS_PROFILE=S3 aws s3 sync $SRC $DST/$IT/ --only-show-errors --exclude "*/step*/*" --exclude "wandb/*" && echo "SYNCED $IT" || { echo "!!! FAILED $IT"; exit 1; }
done
echo TRANSFER_DONE
EOW
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "stage-q3recipe-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 8 --memory 32GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env ITEMS="$ITEMS" --env P="$P" \
  --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -E "beaker.org/ex|rror" | head -3
