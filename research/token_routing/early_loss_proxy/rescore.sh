#!/bin/bash
# Fetch remote results, score every new-pool tag against the references (results/v2 on sneetches), rebuild loop_state.json.
cd "$(dirname "$0")"
for t in $(ls /tmp/claude-3018/elp_beaker_results 2>/dev/null) hp keep; do bash fetch_beaker_results.sh $t >/dev/null 2>&1; done
for t in $(ssh -o ConnectTimeout=10 lambda 'ls /accounts/projects/sewonm/prasann/ctc_suite/elp/results' 2>/dev/null | grep -v -E "smoke|^t$"); do bash fetch_lambda_results.sh $t >/dev/null 2>&1; done
DIRS=""
for d in /net/sneetches/data/prasann/early_loss_proxy/results/* /net/mooney/data/prasann/early_loss_proxy/results/* /tmp/claude-3018/elp_lambda_results/* /tmp/claude-3018/elp_beaker_results/*; do
  case "$(basename $d)" in v1|s50|smoke*|*8k|*v1pool|t|lsmoke*|bksmoke*|bkenv*) continue;; esac
  [ -d "$d" ] && DIRS="$DIRS $d"
done
python3 score.py --results $DIRS --ref-tag v2 --out hub_tab/board_v2.json "$@"
