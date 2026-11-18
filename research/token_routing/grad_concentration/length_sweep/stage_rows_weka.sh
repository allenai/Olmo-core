#!/usr/bin/env bash
# Copy the sweep's tokenized rows S3 -> weka for Beaker runs (rows uploaded from the build node with s3_sync.py up to
# s3://ai2-llm/checkpoints/prasanns/_transfer/gradlen/tok/{after,both}/). CPU-only gantry job.
set -uo pipefail
read -r -d '' WORK <<'EOW'
set -uo pipefail
mkdir -p ~/.aws && printf '%s' "$AWS_CREDS" > ~/.aws/credentials && printf '%s' "$AWS_CFG" > ~/.aws/config
pip install -q awscli 2>&1 | tail -1
for Q in after both; do
  AWS_PROFILE=S3 aws s3 sync s3://ai2-llm/checkpoints/prasanns/_transfer/gradlen/tok/$Q /weka/oe-training-default/ai2-llm/checkpoints/prasanns/gradlen/tok/$Q --only-show-errors && echo "SYNCED $Q $(du -sh /weka/oe-training-default/ai2-llm/checkpoints/prasanns/gradlen/tok/$Q | cut -f1)"
done
ls /weka/oe-training-default/ai2-llm/checkpoints/prasanns/gradlen/tok/both | head -50
echo STAGE_DONE
EOW
export PATH=$HOME/.local/bin:/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "gsl-stage-rows-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 8 --memory 32GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -E "beaker.org/ex|rror|xperiment" | head -5
