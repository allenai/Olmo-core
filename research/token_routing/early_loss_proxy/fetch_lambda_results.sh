#!/bin/bash
# Pull the small result JSONs + logs of a Lambda tag to /tmp/claude-3018/elp_lambda_results/<tag>/ (outside git).
#   bash fetch_lambda_results.sh <tag> [jobid]    # jobid also pulls the slurm log
set -eo pipefail
TAG=${1:?tag}; OUT=/tmp/claude-3018/elp_lambda_results/$TAG
R=/accounts/projects/sewonm/prasann/ctc_suite/elp
mkdir -p "$OUT"
rsync -a --include='*.json' --include='*.log' --include='*.tsv' --exclude='*' "lambda:$R/results/$TAG/" "$OUT/"
rsync -a "lambda:$R/logs/elp-$TAG-${2:-*}.out" "$OUT/" 2>/dev/null || true
rsync -a "lambda:$R/step0_cache.json" "$OUT/../step0_cache_lambda.json" 2>/dev/null || true
ls "$OUT"
