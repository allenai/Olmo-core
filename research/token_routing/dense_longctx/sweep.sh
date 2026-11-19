#!/bin/bash
# Run a list of dense_step_bench.py configs back to back inside one allocation (Lambda payload, via run_lambda.sbatch:
# CMD="research/token_routing/dense_longctx/sweep.sh <listfile>"; the run_lambda env must be active, so call it with
# NGPU=1 -- each line picks its own nproc). Line format:  <nproc> <out-name> <bench args...>   (# = comment)
L=${LAMBDA_REPO:-/accounts/projects/sewonm/prasann/ctc_suite/OLMo-core-dense1m}
R=${RESULTS:-$L/results}; mkdir -p "$R"
PY=/accounts/projects/sewonm/prasann/olmo_test/env/bin/python
grep -v '^\s*#' "$1" | while read -r np name args; do
  [ -z "$np" ] && continue
  echo "##### $name (nproc=$np) $args  $(date '+%T')"
  timeout ${CASE_TIMEOUT:-1500} $PY -m torch.distributed.run --nproc_per_node "$np" --master_port $((29600 + RANDOM % 300)) \
    "$L/research/token_routing/dense_longctx/dense_step_bench.py" $args --out "$R/$name.json" 2>&1 \
    | grep -E --line-buffered '^\{|SUMMARY|OOM|Error|error:|Traceback' | grep --line-buffered -v Warning
  echo "##### $name rc=${PIPESTATUS[0]}"
done
