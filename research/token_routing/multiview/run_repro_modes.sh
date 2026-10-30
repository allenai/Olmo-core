#!/bin/bash
# Stage a git-archive tar (stdin) on the node and run fsdp_cpu_repro.py in several modes.
TAG=$1; shift
D=/data/prasann/mv_$TAG
mkdir -p "$D" && tar -x -C "$D" || exit 1
cd "$D" || exit 1
export PYTHONDONTWRITEBYTECODE=1 HOME=/data/prasann/home_mv TMPDIR=/data/prasann/tmp_mv
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
port=29540
for mode in "$@"; do
  port=$((port + 1))
  echo "=== $mode"
  MASTER_PORT=$port PYTHONPATH="$D/src:$D" timeout 300 "$PY" research/token_routing/multiview/fsdp_cpu_repro.py $mode 2>&1 \
    | grep -E "${PAT:-ZZZ}|SIMPLE_OK|REPRO_OK|Error|requires_grad|early adapters|block0" | grep -v Warning | tail -${TAIL:-4}
done
