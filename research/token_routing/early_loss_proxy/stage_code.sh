#!/bin/bash
# Stage this checkout's HEAD onto a jsteinhardt node's /data (git archive on the LOGIN node, streamed via srun stdin).
#   bash stage_code.sh <node>   -> prints the staged dir /data/prasann/early_loss_proxy/code_<sha>
set -eo pipefail
NODE=${1:?node}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
C=$(git -C "$REPO" rev-parse --short HEAD); D=/data/prasann/early_loss_proxy/code_$C
git -C "$REPO" archive HEAD src recipes research/token_routing/early_loss_proxy research/common \
  | srun --partition=jsteinhardt --qos=preemptive_high --nodelist="$NODE" --cpus-per-task=1 --mem=2G --time=00:05:00 \
    bash -c "mkdir -p $D && tar -x -C $D && mkdir -p /data/prasann/early_loss_proxy/results /data/prasann/joblogs" >&2
echo "$D"
