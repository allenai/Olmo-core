#!/bin/bash
# Ship finished fast8k training exports (config.json + model-only model_and_optim) from weka to S3, so they can be
# scored on Berkeley nodes when the Beaker eval queue is backed up (2026-10-28). One long-lived CPU job: it waits for
# each run in RUNS (in order) to finish its post-fit export, syncs it, and writes <run>/_SYNCED; exits when all are
# synced or after HOURS.
#
#   RUNS="run1 run2 ..." bash research/token_routing/layer_drop/ckpts_to_s3_beaker.sh
set -uo pipefail
RUNS="${RUNS:?set RUNS}"
HOURS="${HOURS:-12}"
CK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts
DST="${DST:-s3://ai2-llm/checkpoints/prasanns/_transfer/layerskip/ckpts}"
gantry run --name "layerskip-ckpts-s3-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 8 --memory 32GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env CK="$CK" --env RUNS="$RUNS" --env DST="$DST" --env HOURS="$HOURS" \
  --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
mkdir -p ~/.aws && printf "%s\n" "$AWS_CREDS" > ~/.aws/credentials && printf "%s\n" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
pip install -q awscli >/dev/null 2>&1 || { echo "!!! awscli install failed"; exit 1; }
end=$(( $(date +%s) + HOURS * 3600 )); done_list=""
while [ "$(date +%s)" -lt "$end" ]; do
  left=0
  for r in $RUNS; do
    case " $done_list " in *" $r "*) continue ;; esac
    if [ -f $CK/$r/model_and_optim/.metadata ] && [ -f $CK/$r/config.json ]; then
      aws s3 sync --quiet $CK/$r/model_and_optim $DST/$r/model_and_optim \
        && aws s3 cp --quiet $CK/$r/config.json $DST/$r/config.json \
        && { [ -f $CK/$r/flops.json ] && aws s3 cp --quiet $CK/$r/flops.json $DST/$r/flops.json; true; } \
        && echo synced | aws s3 cp --quiet - $DST/$r/_SYNCED \
        && { done_list="$done_list $r"; echo "[synced] $r $(date +%T)"; } || echo "!!! sync failed $r"
    else
      left=$((left + 1))
    fi
  done
  [ "$left" -eq 0 ] && { echo ALL_SYNCED; exit 0; }
  sleep 120
done
echo TIMEOUT'
