#!/bin/bash
# Copy the FRESH-pool mt4 shards (build_mt4_fresh_beaker.sh: mt4f1952 / f3904 / f5856 / f7808) from weka to S3 so
# they can be pulled onto a Berkeley node (model-scale ablation, 2026-10-29: jupiter was not scheduling 2-GPU jobs).
#   POOLS="1952 3904 5856 7808" bash research/token_routing/ds64_fast8k/sync_fresh_shards_s3_beaker.sh
set -uo pipefail
POOLS="${POOLS:-1952 3904 5856 7808}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
DST="${DST:-s3://ai2-llm/checkpoints/prasanns/_transfer/tokdrop_lr}"
gantry run --name "fast8k-mt4f-s3-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 4 --memory 32GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env WEKA="$WEKA" --env POOLS="$POOLS" --env DST="$DST" \
  --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
mkdir -p ~/.aws && printf "%s\n" "$AWS_CREDS" > ~/.aws/credentials && printf "%s\n" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
pip install -q awscli >/dev/null 2>&1 || { echo "!!! awscli install failed"; exit 1; }
for R in $POOLS; do
  SRC=$WEKA/ds64/fast8k/shards/mt4f$R
  [ -s $SRC/metadata.json ] || { echo "!!! missing $SRC"; exit 1; }
  aws s3 sync --quiet $SRC $DST/shards/mt4f$R || { echo "!!! sync failed $R"; exit 1; }
  echo "[sync] mt4f$R $(du -sh $SRC | cut -f1)"
done
aws s3 ls --recursive $DST/shards | grep mt4f | awk "{s+=\$3} END {print \"S3 mt4f objects:\", NR, \"bytes:\", s}"
echo S3_DONE'
