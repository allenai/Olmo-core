#!/bin/bash
# Stage node-local early-loss-proxy inputs to weka (the beaker.md two-step): (1) <node>:/data -> S3 from a small
# jsteinhardt srun ON that node (never the login node, never /net as I/O), (2) S3 -> weka in a CPU Beaker job.
#   bash stage_to_weka.sh <node> <rel> [<rel> ...]      rel = path under /data/prasann/early_loss_proxy (file or dir)
#   e.g. bash stage_to_weka.sh mooney shards/mtb16k heldout_v2.npz
# -> /weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy/<rel>. The CPU job runs s3sync.py from
# the PUSHED commit of this branch (push first). Prints the Beaker experiment id; register it with ops.
set -eo pipefail
NODE=${1:?node}; shift
[ $# -ge 1 ] || { echo "usage: $0 <node> <rel>..."; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
L=/data/prasann/early_loss_proxy
S3=s3://ai2-llm/checkpoints/prasanns/early_loss_proxy
WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy
NPY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python  # node-local env, has boto3
for REL in "$@"; do
  srun --partition=jsteinhardt --qos=preemptive_high --nodelist="$NODE" --cpus-per-task=4 --mem=8G --time=01:00:00 \
    bash -c "P=$L/$REL; if [ -f \$P ]; then D=$S3/\$(dirname $REL); D=\${D%/.}; else D=$S3/$REL; fi; \
             AWS_PROFILE=S3 $NPY - up \$P \$D" < "$HERE/s3sync.py"
done
BR=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
REF=$(git -C "$REPO" rev-parse HEAD)
git -C "$REPO" branch -r --contains "$REF" | grep -q . || { echo "push $BR first (the CPU job runs s3sync.py from it)"; exit 2; }
WORK='mkdir -p ~/.aws; printf "%s" "$AWS_CREDS" >~/.aws/credentials; printf "%s" "$AWS_CFG" >~/.aws/config; set -e;'
for REL in "$@"; do
  WORK+=" AWS_PROFILE=S3 /opt/conda/bin/python research/token_routing/early_loss_proxy/s3sync.py down $S3/$REL $WK/$REL;"
  WORK+=" ls -la $WK/$REL | head -20;"
done
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
gantry run --name "elp-stage-weka" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster ai2/jupiter --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --ref "$REF" --branch "$BR" --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -o "beaker.org/ex/[A-Z0-9]*" | head -1
