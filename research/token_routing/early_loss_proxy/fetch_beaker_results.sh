#!/bin/bash
# Pull a Beaker early-loss-proxy tag's result JSONs + logs (small files; each job mirrors them to S3 as every run
# ends) to the Berkeley login node:  bash fetch_beaker_results.sh <tag> [dest_root]
#   -> /tmp/claude-3018/elp_beaker_results/<tag>/   (then: score.py --results <that dir> ...)
set -eo pipefail
TAG=${1:?tag}
DEST=${2:-/tmp/claude-3018/elp_beaker_results}/$TAG
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python  # has boto3; ~/.aws has the S3 profile
mkdir -p "$DEST"
AWS_PROFILE=${AWS_PROFILE:-S3} "$PY" -I "$(dirname "$0")/s3sync.py" down \
  "s3://ai2-llm/checkpoints/prasanns/early_loss_proxy/results/$TAG" "$DEST" --force
ls "$DEST" | grep -c '\.json$' | xargs -I{} echo "{} result JSONs in $DEST"
grep -l '"done": true' "$DEST"/elp-*.json 2>/dev/null | sed 's|.*/||; s|^|done: |' || true
