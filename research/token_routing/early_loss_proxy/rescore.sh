#!/bin/bash
# Fetch remote results, score every new-pool tag against the references (results/v2 on sneetches), rebuild loop_state.json.
cd "$(dirname "$0")"
[ -n "$NOFETCH" ] || for t in $(ls /tmp/claude-3018/elp_beaker_results 2>/dev/null) hp keep short detach full ${BK_TAGS:-}; do bash fetch_beaker_results.sh $t >/dev/null 2>&1; done
[ -n "$NOFETCH" ] || for t in $(ssh -o ConnectTimeout=10 lambda 'ls /accounts/projects/sewonm/prasann/ctc_suite/elp/results' 2>/dev/null | grep -v -E "smoke|^t$"); do bash fetch_lambda_results.sh $t >/dev/null 2>&1; done
DIRS=""
for d in /net/sneetches/data/prasann/early_loss_proxy/results/* /net/mooney/data/prasann/early_loss_proxy/results/* /tmp/claude-3018/elp_lambda_results/* /tmp/claude-3018/elp_beaker_results/*; do
  # heldout_v2 runs only: v3 / longpos use heldout_v3 (scored by v3.py); kl* / q32k / q64k belong to the keep-vs-length
  # session (other held-out sets); memscale* are 1-3 step memory probes on 4 rows. Mixing them in gives meaningless ratios.
  case "$(basename $d)" in v1|s50|smoke*|*8k|*v1pool|t|lsmoke*|bksmoke*|bkenv*|v3|longpos|kl*|q32k|q64k|memscale*) continue;; esac
  [ -d "$d" ] && DIRS="$DIRS $d"
done
mkdir -p /tmp/claude-3018/elp_refs/v2; cp /tmp/claude-3018/elp_beaker_results/hp/elp-hp-denselora-lr2e5-sd0.json /tmp/claude-3018/elp_refs/v2/elp-v2-denselora2e5-sd0.json 2>/dev/null
python3 score.py --results $DIRS /tmp/claude-3018/elp_refs/v2 --ref dense,denselora,denselora2e5 --ref-tag v2 --out hub_tab/board_v2.json "$@"
