#!/bin/bash
# Run pytest from a staged code dir on a compute node (node-local env, caches on /data). Usage: run_tests_node.sh <code dir> <pytest args...>
set -euo pipefail
D=$1; shift
export PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HOME=/data/prasann/home_wts TMPDIR=/data/prasann/tmp_wts TRITON_CACHE_DIR=/data/prasann/triton_wts PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
mkdir -p $HOME $TMPDIR $TRITON_CACHE_DIR
cd $D
PYTHONPATH=$D/src:$D/recipes/train:$D $PY -m pytest -p no:cacheprovider -q "$@"
