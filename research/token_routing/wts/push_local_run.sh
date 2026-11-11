#!/usr/bin/env bash
# Move a wts run trained on a Berkeley node (/data/prasann/wts/runs/<run>) to weka,
# evals (launch_wts.py eval) read it from $CKPTS/<run> exactly as for the Beaker-trained arms (copied from oracle_train/push_local_run.sh).
#   step up:   bash push_local_run.sh up   <run> <node>   # sbatch on <node>: boto3 upload to S3 (s3_sync.py)
#   step down: bash push_local_run.sh down <run> [<run> ...]  # gantry CPU job: S3 -> weka ctc_suite/ckpts/<run>/
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
MODE=$1; shift
REL=checkpoints/prasanns/_transfer/wts
if [ "$MODE" = up ]; then
  RUN=$1; NODE=$2
  sbatch -p jsteinhardt --qos=preemptive_high -w "$NODE" -c 4 --mem 16G -t 180 -J "s3up-$RUN" \
    -o /data/prasann/joblogs/s3up-$RUN-%j.out --wrap "rm -rf /data/prasann/wts/runs/$RUN/step[0-9]* && cd $REPO/research/token_routing/olmo3_cptdrop && \
    /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python -c \"import sys, s3_sync as s; s.ROOT='$REL'; \
    sys.argv=['s3_sync','up','/data/prasann/wts/runs/$RUN','runs/$RUN']; s.main()\" | tail -3 && echo S3UP_DONE"
  exit
fi
[ "$MODE" = down ] || { echo "mode up|down"; exit 2; }
NAMES="$*"
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
export PATH=$HOME/.local/bin:/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "wts-s3down-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 8 --memory 32GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env NAMES="$NAMES" --env W="$W" --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
mkdir -p ~/.aws && printf "%s" "$AWS_CREDS" > ~/.aws/credentials && printf "%s" "$AWS_CFG" > ~/.aws/config
pip install -q awscli 2>&1 | tail -1
for N in $NAMES; do
  AWS_PROFILE=S3 aws s3 sync s3://ai2-llm/checkpoints/prasanns/_transfer/wts/runs/$N/ $W/ctc_suite/ckpts/$N/ \
    --only-show-errors --exclude ".s3_etags.json" --exclude "step[0-9]*" && echo "DOWN $N $(du -sh $W/ctc_suite/ckpts/$N | cut -f1)" \
    || echo "!!! FAILED $N"
  rm -rf $W/ctc_suite/ckpts/$N/step[0-9]*   # resume points: evals read the run root (final model_and_optim)
  AWS_PROFILE=S3 aws s3 rm --recursive --only-show-errors s3://ai2-llm/checkpoints/prasanns/_transfer/wts/runs/$N/ \
    --exclude "*" --include "step[0-9]*"
  ls $W/ctc_suite/ckpts/$N
done
echo S3DOWN_DONE' 2>&1 | grep -E "beaker.org/ex|rror|xperiment" | head -5
