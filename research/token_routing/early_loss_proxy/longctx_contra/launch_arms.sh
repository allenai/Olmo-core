#!/bin/bash
# The three long-context contradiction arms (Prasann, 2026-11-19). One 8-GPU Beaker job each via launch_lc_smoke.py
# (payload run_lc_smoke.sh, dense_probe wrapper: --probe-max-pf stops at the FLOP budget). Checkpoints on weka
# (runs_lc/<arm>/stepN) for checkpoint-then-score (score_ckpts.sh): every arm is scored through the same 1-GPU dense
# forward. Usage: bash launch_arms.sh gdd256 | dense256 | gdd1m | gddag (G= SI= NAME= FF=)   [DRY=1]
#        bash launch_arms.sh score <arm> <step>[,<step>...]   (checkpoint-then-score: 1-GPU dense forward path, rows split
#        over 8 data-parallel GPUs; lc256k 160 + lc16k 160 rows; the 1M arm also gets lc1m 64 rows under CP4)
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
  gddag)  # gdd256 + group-partitioned attention (attention layers only; 2026-11-20): gdd256's exact config and data
    # order plus --st-attn-groups G (default 4), decoys = number of gold docs, full_frac 0.1; to ~5,600 PF (FLOP-matched:
    # it consumes more rows than gdd256 at the same PF). SI = save interval in steps (gdd256: 1450 = ~1,821 PF), chosen
    # from the screen's per-step PF ratio so checkpoints land near 1.8k / 3.7k / 5.6k PF. NAME must be new (no leftover ckpts).
    G=${G:-4}; SI=${SI:-1450}; NAME=${NAME:-gddag$G}
    python3 $L --tag lcA-$NAME --minutes ${MIN:-330} "${COMMON[@]}" --presync "shards/lc256kmix_gdd heldout_lc.npz" \
      --run "$NAME|probe|8|softtoken|1e-5|lc256kmix_gdd|262144|8|999999|lc16k:16|$R --st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.05 --prefix-chunk 4096 --no-data-shuffle --st-attn-groups $G --st-attn-group-decoys -1 --st-attn-group-full-frac ${FF:-0.1} --ephemeral-save-interval 100 --save-interval $SI --probe-max-pf ${MAXPF:-5650} --probe-pf-points 1821,3765,5583"
    ;;
  dense256)  # 1,091 256k rows + 4,800 short rows packed into 262,144-token windows; 2 windows/step (CP4 x DP2)
    python3 $L --tag lcA-dense256 --minutes 300 "${COMMON[@]}" --presync "shards/lc256kmix_dense heldout_lc.npz" \
      --run "dense256b|probe|8|full|1e-5|lc256kmix_dense|262144|2|999999|lc16k:16|$LORA --cp-degree 4 --pack --ephemeral-save-interval 20 --save-interval 75 --probe-max-pf $BUDGET"
    ;;
  score)
    SA=${2:?arm}; STEPS=${3:?steps}
    # every arm is scored in the SAME model class (dense LoRA): the router arms' early-LoRA adapters load under the
    # same parameter names (verified 2026-11-19: a gold-drop checkpoint scores 1.1678 both ways, lcsmO)
    V=full; F="$LORA"
    RUNS=()
    for st in ${STEPS//,/ }; do
      CK=$SR/$SA/step$st/model_and_optim
      [ "${ONLY1M:-}" ] || RUNS+=(--run "score-$SA-s$st|probe|8|$V|1e-5|kl2k|4096|8|1|lc256k:160,lc16k:160|$F $( [ $st = 0 ] || echo --base-checkpoint $CK )")
      # 1M held-out: CP8 (one data-parallel group, no duplicated rows; int32 bound 2,097,152), one job per checkpoint
      case $SA in *1m|*1m_mix) [ "${SKIP1M:-}" ] || python3 $L --env DENSE_PROBE_SCORE_ONLY=1 --tag lcS-$SA --minutes 60 --gpus 8 \
        --cluster ai2/jupiter-cirrascale-2 --exclude-host jupiter-cs-aus-152 ${DRY:+--dry-run} \
        --run "score1m-$SA-s$st|probe|8|full|1e-5|lc1mdense84|1048576|1|1|lc1m:64|$LORA --cp-degree 8 --shard-degree 1 --gdn-int32-safe --ffn-chunk-tokens 32768 --skip-dry-run $( [ $st = 0 ] || echo --base-checkpoint $CK )" ;; esac
    done
    [ ${#RUNS[@]} -gt 0 ] && python3 $L --env DENSE_PROBE_SCORE_ONLY=1 --tag lcS-$SA --minutes $((40 * $(echo $STEPS | tr ',' ' ' | wc -w))) --gpus 8 --cluster ai2/jupiter-cirrascale-2 \
      --exclude-host jupiter-cs-aus-152 ${DRY:+--dry-run} "${RUNS[@]}"
    ;;
  dense1m)  # 84 x 1M rows = the first 84 long rows of the gold-drop 1M pool in its order (no short rows: --pack under CP8
    # OOMs, the guard is skipped on cu_doc_lens calls); 1 row/step, CP8; 168.5 s/row on 8 H100 -> 84 rows ~ 31.5 GPU-h
    python3 $L --tag lcA-dense1m --minutes 270 "${COMMON[@]}" --presync "shards/lc1mdense84 heldout_lc.npz" \
      --run "dense1m|probe|8|full|1e-5|lc1mdense84|1048576|1|999999|lc16k:16|$LORA --cp-degree 8 --gdn-int32-safe --ffn-chunk-tokens 32768 --shard-degree 1 --skip-dry-run --no-data-shuffle --ephemeral-save-interval 5 --save-interval 21"
    ;;
  dense1m_mix)  # Prasann 11-19: dense LoRA 1M on gdd1m's mix = the first 1,515 rows of lc1mmix_gdd in its order (83 x 1M +
    # 1,432 short; short:long 17.3 vs gdd1m's 14.5), packed into 1,048,576-token windows, CP8, 1 window/step. Windows whose
    # first document is a 1M row run unpacked (--pack-unpack-long; co-packed docs masked); short windows stay packed with
    # GDN calls grouped at 131,072 tokens (exact). Smoke lcsmQ5: 1M step 167 s / 51 GiB, short window 118 s / 55 GiB.
    python3 $L --tag lcA-dense1mmix --minutes 270 "${COMMON[@]}" --env OLMO_GDN_PACKED_GROUP=131072 \
      --presync "shards/lc1mmixdense heldout_lc.npz" \
      --run "dense1m_mix|probe|8|full|1e-5|lc1mmixdense|1048576|1|999999|lc16k:16|$LORA --cp-degree 8 --pack --pack-unpack-long 262144 --gdn-int32-safe --gdn-segment-tokens 131072 --ffn-chunk-tokens 16384 --shard-degree 1 --skip-dry-run --no-data-shuffle --ephemeral-save-interval 5 --save-interval 24"
    ;;
  dense1m_chain|chainsmoke)  # fallback for dense1m_mix (Prasann 11-19): chained trainer runs in ONE job, each branching
    # (--branch-from @PREV@ --branch-optim-only: LoRA + optimizer carried, fresh step counter) from the previous run's last
    # checkpoint: 1M segments as dense1m (CP8, unpacked) and short segments as dense256b's short windows (262k, CP4, packed).
    # LR: wsd, warmup only in the first segment, constant 1e-5, linear decay to 0 over the final 1M segment.
    LF="$LORA --cp-degree 8 --gdn-int32-safe --ffn-chunk-tokens 32768 --shard-degree 1 --skip-dry-run --no-data-shuffle --ephemeral-save-interval 5"
    SF="$LORA --cp-degree 4 --pack --no-data-shuffle --skip-dry-run"
    CH="--branch-from @PREV@ --branch-optim-only"
    C0="--scheduler wsd --wsd-decay-fraction 0 --warmup-fraction 0"
    if [ "$ARM" = chainsmoke ]; then
      python3 $L --tag lcsmChain --minutes 60 "${COMMON[@]}" --presync "shards/lc1mc_smkL shards/lc1mc_smkS heldout_lc.npz" \
        --run "chs_L1|probe|8|full|1e-5|lc1mc_smkL|1048576|1|999999|lc16k:4|$LF --scheduler wsd --wsd-decay-fraction 0 --warmup-fraction 0.03" \
        --run "chs_S1|probe|8|full|1e-5|lc1mc_smkS|262144|2|999999|lc16k:4|$SF $CH $C0" \
        --run "chs_L2|probe|8|full|1e-5|lc1mc_smkL|1048576|1|999999|lc16k:4|$LF $CH --scheduler linear --warmup-fraction 0"
    else
      python3 $L --tag lcA-dense1mchain --minutes 300 "${COMMON[@]}" \
        --presync "shards/lc1mc_L1 shards/lc1mc_L2 shards/lc1mc_L3 shards/lc1mc_L4 shards/lc1mc_S1 shards/lc1mc_S2 shards/lc1mc_S3 shards/lc1mc_S4 heldout_lc.npz" \
        --run "d1mc_L1|probe|8|full|1e-5|lc1mc_L1|1048576|1|999999|lc16k:16|$LF --scheduler wsd --wsd-decay-fraction 0 --warmup-fraction 0.03" \
        --run "d1mc_S1|probe|8|full|1e-5|lc1mc_S1|262144|2|999999|lc16k:16|$SF $CH $C0" \
        --run "d1mc_L2|probe|8|full|1e-5|lc1mc_L2|1048576|1|999999|lc16k:16|$LF $CH $C0" \
        --run "d1mc_S2|probe|8|full|1e-5|lc1mc_S2|262144|2|999999|lc16k:16|$SF $CH $C0" \
        --run "d1mc_L3|probe|8|full|1e-5|lc1mc_L3|1048576|1|999999|lc16k:16|$LF $CH $C0" \
        --run "d1mc_S3|probe|8|full|1e-5|lc1mc_S3|262144|2|999999|lc16k:16|$SF $CH $C0" \
        --run "d1mc_S4|probe|8|full|1e-5|lc1mc_S4|262144|2|999999|lc16k:16|$SF $CH $C0" \
        --run "d1mc_L4|probe|8|full|1e-5|lc1mc_L4|1048576|1|999999|lc16k:16|$LF $CH --scheduler linear --warmup-fraction 0"
    fi
    ;;
  gdd1m)  # 1M rows (keep 0.05-0.4, no whole rows, int32 guard) + short rows; 8 rows/step, bucketed steps
    python3 $L --tag lcA-gdd1m --minutes 720 "${COMMON[@]}" --presync "shards/lc1mmix_gdd heldout_lc.npz" \
      --run "gdd1m|probe|8|softtoken|1e-5|lc1mmix_gdd|1048576|8|999999|lc16k:16|$R --st-gdd-keep-range 0.05,0.4 --st-gdd-whole-frac 0 --prefix-chunk 4096 --gdn-int32-safe --no-data-shuffle --ephemeral-save-interval 100 --save-interval 600 --probe-max-pf 17000"  # generous: read at the dense-1M arm's PF
    ;;
esac
