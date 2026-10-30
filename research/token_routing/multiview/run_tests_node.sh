#!/bin/bash
# Stage a `git archive` tar (stdin) into node-local /data and run the multi-view CPU tests there.
#   git archive -o /tmp/mv.tar <commit>
#   srun --partition=berkeleynlp --qos=normal --nodelist=horton --cpus-per-task=4 --mem=16G --time=00:20:00 \
#       bash research/token_routing/multiview/run_tests_node.sh <tag> [pytest paths...] < /tmp/mv.tar
TAG=$1; shift
D=/data/prasann/mv_$TAG
mkdir -p "$D" && tar -x -C "$D" || exit 1
cd "$D" || exit 1
export PYTHONDONTWRITEBYTECODE=1 HOME=/data/prasann/home_mv TMPDIR=/data/prasann/tmp_mv
mkdir -p "$HOME" "$TMPDIR"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
TESTS=("$@")
[ ${#TESTS[@]} -eq 0 ] && TESTS=(src/test/nn/multiview_test.py src/test/nn/online_token_router_test.py recipes/train/variants_options_test.py)
PYTHONPATH="$D/src:$D" timeout 900 "$PY" -m pytest -x -q -p no:cacheprovider "${TESTS[@]}" 2>&1 | tail -60
