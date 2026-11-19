#!/bin/bash
# Stage this worktree's code to Lambda (no shared filesystem, no git there). Run from the Berkeley login node.
#   bash research/token_routing/dense_longctx/stage_lambda.sh
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
L=${LAMBDA_REPO:-/accounts/projects/sewonm/prasann/ctc_suite/OLMo-core-dense1m}
ssh lambda "mkdir -p $L/research/token_routing $L/logs"
for d in src recipes research/common research/token_routing/dense_longctx; do
  ssh lambda "mkdir -p $L/$d"
  rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='results*' \
    "$REPO/$d/" "lambda:$L/$d/"
done
echo "staged $REPO -> lambda:$L ($(cd "$REPO" && git rev-parse --short HEAD), dirty files: $(cd "$REPO" && git status --porcelain | wc -l))"
