#!/usr/bin/env bash
# Stage fresh-pool shards (weka ds64/fast8k/shards/<name>, gold sidecar included) onto a Berkeley node's /data, for local
# copies of Beaker runs that sit pending on jupiter.
#   step up:   bash stage_shards_s3.sh up mt4f488 mt4f976          # gantry CPU job: weka -> s3 _transfer/oracle/shards/
#   step down: bash stage_shards_s3.sh down <node> mt4f488 mt4f976  # sbatch on <node>: s3 -> /data/prasann/oracle/shards/
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
MODE=$1; shift
REL=checkpoints/prasanns/_transfer/oracle
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
if [ "$MODE" = up ]; then
  NAMES="$*"
  export PATH=$HOME/.local/bin:/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
  gantry run --name "oracle-shards-s3up-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
    --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
    --gpus 0 --cpus 4 --memory 16GiB --priority urgent --min-runtime 1h \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
    --weka oe-training-default:/weka/oe-training-default \
    --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
    --env NAMES="$NAMES" --env W="$W" --env REL="$REL" --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
mkdir -p ~/.aws && printf "%s" "$AWS_CREDS" > ~/.aws/credentials && printf "%s" "$AWS_CFG" > ~/.aws/config
pip install -q awscli 2>&1 | tail -1
for N in $NAMES; do
  AWS_PROFILE=S3 aws s3 sync $W/ds64/fast8k/shards/$N/ s3://ai2-llm/$REL/shards/$N/ --only-show-errors \
    && echo "UP $N $(ls $W/ds64/fast8k/shards/$N | tr "\n" " ")" || echo "!!! FAILED $N"
done
echo S3UP_DONE' 2>&1 | grep -E "beaker.org/ex|rror" | head -3
  exit
fi
[ "$MODE" = down ] || { echo "mode up|down"; exit 2; }
NODE=$1; shift
for N in "$@"; do
  sbatch -p jsteinhardt --qos=preemptive_high -w "$NODE" -c 4 --mem 16G -t 60 -J "s3down-$N" \
    -o /data/prasann/joblogs/s3down-$N-%j.out --wrap "cd $REPO/research/token_routing/olmo3_cptdrop && \
    /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python -c \"import sys, s3_sync as s; s.ROOT='$REL'; \
    sys.argv=['s3_sync','down','/data/prasann/oracle/shards/$N','shards/$N']; s.main()\" | tail -3 && \
    ls /data/prasann/oracle/shards/$N && echo S3DOWN_DONE"
done
