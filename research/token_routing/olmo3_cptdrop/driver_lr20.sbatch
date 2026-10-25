#!/bin/bash
# olmo3_cptdrop at peak LR 2e-5 (the LR probe: 2e-5 beats the OLMo recipe's 2.07e-4 by -0.133 +-0.002 nats at
# 120 steps and keeps improving with compute; 2.07e-4 plateaus and forgets). One 8-GPU node, in order, stopping at
# the first failure; runs whose export exists are skipped (rerunnable).
#   DC  dense curve: Dlr20-{12,24,48} fresh WSD runs (+ Dlr20-120/240 already from LRPROBE), Dlr10-120 (LR 1e-5)
#   IR  IRANDlr20-2600 (saves every 195) + decay branches 650 / 1300
#   HO  HOlr20-{120,240,480}: IRAND for 390 / 585 / 1170 steps, then dense to the budget (trainer state kept)
#   SB  dense small-batch control DB8lr20-{12,24,48}: 8 rows/step at the same token budget = 2x the optimizer
#       steps (do drop arms win by dropping, or only by taking more optimizer steps per FLOP?)
#   SMK  20-step SD20 smoke (slots in positional-block mode were never trained before), SMOKE=1 = no checkpoints
#   LOW  the steep end, where dense still moves (12 steps already give 0.150 of 0.165 nats): dense 4 / 8 steps
#        (4: decay 1 step), I-rand 22 / 43 (= 4 / 8 dense steps), SD20 260 (+65, +130 branches), LR checks
#        Dlr10-12 and Dlr05-120 (1e-5 beat 2e-5 at 120 steps)
#   OTRS mid-stack online router (CTC suite's otr10M7; records/online-token-router.md): 20-step smoke, then fresh
#        runs at ~12 / 24 / 48 / 120 dense-step FLOPs (est. 0.39x dense/step with a dense step every 8: 31 / 62 / 124
#        / 310 steps) + the no-full-context-step variant (est. 0.30x: 40 / 80 / 160)
#   SEED Dlr20-120-s1 (seed noise of the dense reference at LR 2e-5; was 0.009 nats at 2.07e-4)
#   IRS  IRANDlr20-260 (saves every 58) + branches 65 / 130: I-rand at dense-equivalent 12 / 24 / 48 steps
#   IFS  IFLlr20-215 (saves every 48) + branches 54 / 108: I-fl 25% at dense-equivalent 12 / 24 / 48 steps
#   D480 Dlr20-480 fresh (the long end of the dense curve)
#   EV  8-GPU row-chunked dev CE of everything above
#   CTX the same plus ctx_gain (--ctx-window 8192: last 4k targets with full context vs the last 8k only) for
#       every finished run + stage2 / stage3, into devce_ctx/ (dev CE saturates by ~12 dense steps = domain
#       adaptation; ctx_gain tracks long-context learning, which OLMo's 50B-token stage 3 is about)
#   STAGES="DC IR HO EV" CODE_SHA=<sha> sbatch -w cubbins research/token_routing/olmo3_cptdrop/driver_lr20.sbatch
#SBATCH --job-name=o3cd-lr20
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=16:00:00
#SBATCH --output=/data/prasann/olmo3_cptdrop/logs/%x-%j.out
set -uo pipefail
ROOT=/data/prasann/olmo3_cptdrop
CODE_SHA="${CODE_SHA:?}"
CODE="$ROOT/code_$CODE_SHA"
ARM_SH="$CODE/research/token_routing/olmo3_cptdrop/run_arm.sbatch"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export CODE_SHA LR=2e-5
log() { echo "[driver $(date '+%F %T')] $*"; }
run() {
  local name=$1; shift
  if [ -f "$ROOT/runs/$name/model_and_optim/.metadata" ]; then log "SKIP $name (export exists)"; return 0; fi
  log "START $name"; env "$@" RUN="$name" bash "$ARM_SH" > "$ROOT/logs/$name.log" 2>&1; local rc=$?
  log "END $name rc=$rc"; return $rc
}
for st in ${STAGES:-DC SB IRS IFS EV SEED IR HO D480 EV CTX}; do
  case $st in
    DC)
      for b in 12 24 48; do run "o3cd-Dlr20-$b" ARM=D MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 8)) || exit 10; done
      run o3cd-Dlr10-120 ARM=D LR=1e-5 MAX_STEPS_TOTAL=120 EXPECTED_MIN=30 || exit 11 ;;
    SB)
      for b in 12 24 48; do run "o3cd-DB8lr20-$b" ARM=D GB_OVERRIDE=8 MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 3 + 8)) || exit 13; done ;;
    SMK)
      run o3cd-SD20smoke ARM=SD20 MAX_STEPS_TOTAL=20 SMOKE=1 NOWANDB=1 EXPECTED_MIN=12 || exit 40 ;;
    LOW)
      run o3cd-Dlr20-4 ARM=D MAX_STEPS_TOTAL=4 DECAY_FRAC=0.25 EXPECTED_MIN=8 || exit 41
      run o3cd-Dlr20-8 ARM=D MAX_STEPS_TOTAL=8 EXPECTED_MIN=9 || exit 42
      run o3cd-IRANDlr20-22 ARM=IRAND MAX_STEPS_TOTAL=22 EXPECTED_MIN=9 || exit 43
      run o3cd-IRANDlr20-43 ARM=IRAND MAX_STEPS_TOTAL=43 EXPECTED_MIN=10 || exit 44
      run o3cd-SD20lr20-260 ARM=SD20 MAX_STEPS_TOTAL=260 SAVE_INTERVAL=58 EXPECTED_MIN=30 || exit 45
      run o3cd-SD20lr20-65 ARM=SD20 MAX_STEPS_TOTAL=65 BRANCH_FROM="$ROOT/runs/o3cd-SD20lr20-260/step58" EXPECTED_MIN=8 || exit 46
      run o3cd-SD20lr20-130 ARM=SD20 MAX_STEPS_TOTAL=130 BRANCH_FROM="$ROOT/runs/o3cd-SD20lr20-260/step116" EXPECTED_MIN=8 || exit 47
      run o3cd-Dlr10-12 ARM=D LR=1e-5 MAX_STEPS_TOTAL=12 EXPECTED_MIN=10 || exit 48
      run o3cd-Dlr05-120 ARM=D LR=5e-6 MAX_STEPS_TOTAL=120 EXPECTED_MIN=30 || exit 49 ;;
    OTRS)
      run o3cd-OTRsmoke ARM=OTR MAX_STEPS_TOTAL=20 SMOKE=1 NOWANDB=1 EXPECTED_MIN=15 || exit 50
      for b in 31 62 124 310; do run "o3cd-OTR10M7lr20-$b" ARM=OTR MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 6 + 10)) || exit 51; done
      for b in 40 80 160; do run "o3cd-OTR10M7ndlr20-$b" ARM=OTR OTR_DENSE=0 MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 8 + 10)) || exit 52; done ;;
    SEED)
      run o3cd-Dlr20-120-s1 ARM=D MAX_STEPS_TOTAL=120 SEED=1 EXPECTED_MIN=30 || exit 14 ;;
    IRS)
      run o3cd-IRANDlr20-260 ARM=IRAND MAX_STEPS_TOTAL=260 SAVE_INTERVAL=58 EXPECTED_MIN=30 || exit 23
      run o3cd-IRANDlr20-65 ARM=IRAND MAX_STEPS_TOTAL=65 BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-260/step58" EXPECTED_MIN=8 || exit 24
      run o3cd-IRANDlr20-130 ARM=IRAND MAX_STEPS_TOTAL=130 BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-260/step116" EXPECTED_MIN=8 || exit 25 ;;
    IFS)
      run o3cd-IFLlr20-215 ARM=IFL MAX_STEPS_TOTAL=215 SAVE_INTERVAL=48 EXPECTED_MIN=30 || exit 26
      run o3cd-IFLlr20-54 ARM=IFL MAX_STEPS_TOTAL=54 BRANCH_FROM="$ROOT/runs/o3cd-IFLlr20-215/step48" EXPECTED_MIN=8 || exit 27
      run o3cd-IFLlr20-108 ARM=IFL MAX_STEPS_TOTAL=108 BRANCH_FROM="$ROOT/runs/o3cd-IFLlr20-215/step96" EXPECTED_MIN=8 || exit 28 ;;
    D480)
      run o3cd-Dlr20-480 ARM=D MAX_STEPS_TOTAL=480 EXPECTED_MIN=110 || exit 12 ;;
    IR)
      run o3cd-IRANDlr20-2600 ARM=IRAND MAX_STEPS_TOTAL=2600 SAVE_INTERVAL=195 EXPECTED_MIN=190 || exit 20
      run o3cd-IRANDlr20-650 ARM=IRAND MAX_STEPS_TOTAL=650 BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-2600/step585" EXPECTED_MIN=15 || exit 21
      run o3cd-IRANDlr20-1300 ARM=IRAND MAX_STEPS_TOTAL=1300 BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-2600/step1170" EXPECTED_MIN=20 || exit 22 ;;
    HO)
      for spec in "390:120" "585:240" "1170:480"; do
        S0=${spec%%:*}; B=${spec##*:}
        DS=$(python3 -c "print(round($B - $S0 * 0.184))")
        run "o3cd-HOlr20-$B" ARM=D MAX_STEPS_TOTAL=$((S0 + DS)) BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-2600/step$S0" EXPECTED_MIN=$((DS / 4 + 10)) || exit 30
      done ;;
    EV)
      SPECS=""
      for r in OTR10M7lr20-31 OTR10M7lr20-62 OTR10M7lr20-124 OTR10M7lr20-310 OTR10M7ndlr20-40 OTR10M7ndlr20-80 OTR10M7ndlr20-160 Dlr20-4 Dlr20-8 IRANDlr20-22 IRANDlr20-43 SD20lr20-65 SD20lr20-130 SD20lr20-260 Dlr10-12 Dlr05-120 Dlr20-12 Dlr20-24 Dlr20-48 DB8lr20-12 DB8lr20-24 DB8lr20-48 Dlr20-120-s1 IRANDlr20-65 IRANDlr20-130 IRANDlr20-260 IFLlr20-54 IFLlr20-108 IFLlr20-215 Dlr10-120 Dlr20-480 IRANDlr20-650 IRANDlr20-1300 IRANDlr20-2600 HOlr20-120 HOlr20-240 HOlr20-480; do
        [ -f "$ROOT/runs/o3cd-$r/model_and_optim/.metadata" ] && [ ! -f "$ROOT/devce/${r}_dev32k_1024.json" ] && SPECS="$SPECS $r:runs/o3cd-$r:4"
      done
      log "EV specs:$SPECS"
      SPECS="$SPECS" SHARDS="dev32k_1024:128 short4k_v2:256" CODE="$CODE" PY="$PY" ROOT="$ROOT" \
        bash "$CODE/research/token_routing/olmo3_cptdrop/devce_multi.sh" > "$ROOT/logs/devce-lr20-$SLURM_JOB_ID.log" 2>&1 \
        || log "WARN: eval had failures"
      grep -h "merged" "$ROOT/logs/devce-lr20-$SLURM_JOB_ID.log" ;;
    CTX)
      SPECS=""; OD="$ROOT/devce_ctx"
      [ -f "$OD/stage2-yarn4_dev32k_1024.json" ] || SPECS="$SPECS stage2-yarn4:base/stage2-step47684:4"
      [ -f "$OD/stage3-yarn8_dev32k_1024.json" ] || SPECS="$SPECS stage3-yarn8:base/stage3-step11921:8"
      for d in "$ROOT"/runs/o3cd-*; do
        r=${d##*/o3cd-}
        case "$r" in *-p1|*lr65*|D-*|DLOW*) continue ;; esac
        [ -f "$d/model_and_optim/.metadata" ] && [ ! -f "$OD/${r}_dev32k_1024.json" ] && SPECS="$SPECS $r:runs/o3cd-$r:4"
      done
      log "CTX specs:$SPECS"
      SPECS="$SPECS" SHARDS="dev32k_1024:128" OUT_DIR="$OD" EVAL_EXTRA="--ctx-window 8192" CODE="$CODE" PY="$PY" ROOT="$ROOT" \
        bash "$CODE/research/token_routing/olmo3_cptdrop/devce_multi.sh" > "$ROOT/logs/devce-ctx-$SLURM_JOB_ID.log" 2>&1 \
        || log "WARN: ctx eval had failures"
      grep -h "merged" "$ROOT/logs/devce-ctx-$SLURM_JOB_ID.log" ;;
  esac
done
log "LR20 DONE"
