#!/bin/bash
# Stage this checkout's COMMITTED HEAD onto Lambda (NFS, under the ctc_suite LROOT). Prints the remote code dir.
#   bash stage_code_lambda.sh   -> /accounts/projects/sewonm/prasann/ctc_suite/elp/code/code_<sha>
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
C=$(git -C "$REPO" rev-parse --short HEAD); D=/accounts/projects/sewonm/prasann/ctc_suite/elp/code/code_$C
git -C "$REPO" archive HEAD src recipes research/token_routing/early_loss_proxy research/common \
  | ssh lambda "mkdir -p $D && tar -x -C $D" >&2
echo "$D"
