#!/bin/bash
# The three long-context contradiction arms (Prasann, 2026-11-19). One 8-GPU Beaker job each via launch_lc_smoke.py
# (payload run_lc_smoke.sh, dense_probe wrapper: --probe-max-pf stops at the FLOP budget). Checkpoints on weka
# (runs_lc/<arm>/stepN) for checkpoint-then-score (score_ckpts.sh): every arm is scored through the same 1-GPU dense
# forward. Usage: bash launch_arms.sh gdd256 | dense256 | gdd1m   [DRY=1]
set -e
ARM=${1:?arm}
cd "$(dirname "$0")/../../../.."
L=research/token_routing/early_loss_proxy/longctx_contra/launch_lc_smoke.py
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy
SR=$W/runs_lc
BUDGET=15206   # PF: 32 GPU-h of dense LoRA at the measured 256k rate (101 GPU-s per 252k row, 13.3 PF actual)
# recipe router (keepfrac_len k20), long-row flags measured in phase 1
R="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32 --attn-backend flash_2 --activation-checkpointing full --skip-dry-run --st-otr-warmup-random"
LORA="--lora-rank 64 --lora-blocks 0 --lora-scale 32 --attn-backend flash_2 --activation-checkpointing full"
COMMON=(--gpus 8 --cluster ai2/jupiter-cirrascale-2 --exclude-host jupiter-cs-aus-152 --save-root $SR --keep-ckpt ${DRY:+--dry-run})
case $ARM in
  gdd256)  # 18,000 distinct + 1,980 reused-gold 256k rows (tail) + 71,993 short rows; 8 rows/step, bucketed steps
    python3 $L --tag lcA-gdd256 --minutes 720 "${COMMON[@]}" --presync "shards/lc256kmix_gdd heldout_lc.npz" \
      --run "gdd256|probe|8|softtoken|1e-5|lc256kmix_gdd|262144|8|999999|lc16k:16|$R --st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.05 --prefix-chunk 4096 --no-data-shuffle --ephemeral-save-interval 100 --save-interval 1450 --probe-max-pf $BUDGET --probe-pf-points 3800,7600,11400"
    ;;
  dense256)  # 1,091 256k rows + 4,800 short rows packed into 262,144-token windows; 2 windows/step (CP4 x DP2)
    python3 $L --tag lcA-dense256 --minutes 300 "${COMMON[@]}" --presync "shards/lc256kmix_dense heldout_lc.npz" \
      --run "dense256|probe|8|full|1e-5|lc256kmix_dense|262144|2|999999|lc16k:16|$LORA --cp-degree 4 --pack --ephemeral-save-interval 20 --save-interval 75 --probe-max-pf $BUDGET"
    ;;
  gdd1m)  # 1M rows (keep 0.05-0.4, no whole rows, int32 guard) + short rows; 8 rows/step, bucketed steps
    python3 $L --tag lcA-gdd1m --minutes 720 "${COMMON[@]}" --presync "shards/lc1mmix_gdd heldout_lc.npz" \
      --run "gdd1m|probe|8|softtoken|1e-5|lc1mmix_gdd|1048576|8|999999|lc16k:16|$R --st-gdd-keep-range 0.05,0.4 --st-gdd-whole-frac 0 --prefix-chunk 4096 --gdn-int32-safe --no-data-shuffle --ephemeral-save-interval 100 --save-interval 1200 --probe-max-pf $BUDGET --probe-pf-points 3800,7600,11400"
    ;;
esac
