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
#   OTRA aggressive router cuts (needs the router branch's flags, claude/online-router-1bc01a >= 717c4d0ff), at ~12 /
#        24 / 48 dense-step FLOPs from estimated ratios: early stack forward-only on routed steps (--st-otr-early-nograd,
#        est. 0.26x: 46 / 92 / 185), pyramid 3:0.5,7:0.1,16:0.03 (est. 0.30x: 40 / 80 / 160), pyramid + forward-only
#        early stack (est. 0.20x: 60 / 120 / 240), merge-dropped (each block's dropped tokens -> one unlabelled mean
#        token at the split; router branch >= 34aef2dd3; est. 0.41x: 31 / 62 / 124). m always from the measured flops.json
#   OTRX router + early-exit LM loss on the dropped tokens at the split (--st-otr-exit-loss-weight W: final norm +
#        lm_head on their block-7 h, same loss_div, head FLOPs charged): W 1.0 and 0.3 at the OTR10M7 step counts
#        31 / 62 / 124 / 310 (measured PF decides the budget); and with NO full-context steps (OTR_DENSE=0, the
#        peak-memory bet: de0 collapsed at 1.80 with 10% labels) at 40 / 80 / 160. Needs the router branch with the flag.
#   OTRY early-nograd (best router variant) with a bounded peak: periodic steps at 50% keep (--st-otr-dense-keep-frac
#        0.5) at 46 / 92 / 185 -- the periodic F-steps still count as dense steps, so blocks [0, 7) get full forward +
#        backward there (router 15bf04ff3 test pins it) -- and no periodic steps (de0, early stack frozen after step 1)
#        at 60 / 120 / 240
#   FRZ  dense CPT + warmup-then-freeze of embeddings + blocks [0, K) (no token drop: the router at keep 1.0 is
#        only the carrier for --st-otr-freeze-after; frozen blocks run forward-only and lose their optimizer state):
#        K 8 / 16 x freeze after 25% / 50% of the steps, at 12 / 24 / 48 / 120 steps; 20-step smoke first
#   MOD  mixture-of-depths (router branch >= 56c96377d): at block 8 the router picks c of each 512-block's tokens;
#        blocks 8, 10, ..., 30 run only on them (the rest pass the residual through); every other block and the LM head
#        see the full row, so every label goes through the full depth. c 0.5 / 0.25, no full-context steps, at 12 / 24 /
#        48 / 120 steps (measured PF decides the budget); 20-step smoke first
#   PFZ  dense CPT + progressive freeze (router branch >= bf671531e) at keep 1.0, drop layer 12 so every frozen block
#        is forward-only: "4:0.25,7:0.5,12:0.75" at 12 / 24 / 48 / 120 steps
#   LORA dense CPT with a low-rank early stack from step 0 (router branch >= 076464355): embeddings + blocks [0, K)
#        frozen, every linear in them gets a rank-16 adapter (scale 32); keep 1.0, drop layer K; K 8 / 16 at 12 / 24 /
#        48 / 120 steps; smoke first. eval_devce re-installs the adapters from config.json["early_lora"]
#   MODB mixture-of-depths at c 0.35 (grad_composition: importance is flat in CPT, routing needs >= 30% kept)
#   LOC  ALL-LOCAL training (--olmo3-train-window 4096: the 8 full-attention layers windowed too; evals full): dense
#        at 12 / 24 / 48 / 120 steps, and + freeze K8 after 25% (keep 1.0 router carrier) at the same budgets
#   LORA64 LoRA early stack at rank 64 (router session, SFT: r64 matches de0 at 12 GiB less peak; r16 lags), scale 32,
#        K 16 / 8 at 12 / 24 / 48 / 120 steps
#   LORALOC LoRA early stack (r64, scale 32) + all-local training (--olmo3-train-window 4096), K 16 / 8 at 12 / 24 /
#        48 / 120 steps (the only peak-memory lever so far x the attention-FLOP lever)
#   REP  rescore the dense reference curve + stage2 / stage3 with the current eval (near / far repeat CE + ctx_gain)
#        into devce_rep/ (the original dense evals predate the repeat bins)
#   LORA64X LoRA r64 K16 at 240 / 480 steps on fresh rows (extends the best CPT arm across the measured dense curve)
#   LORA128 LoRA r128 K16 at 48 / 120 (does rank 128 close the remaining far-only gap?)
#   LORA64K24 LoRA r64 on embeddings + blocks [0, 24) at 48 / 120 (deeper freeze: more peak saving; does far CE hold?)
#   LORA64S1 seed-1 rerun of LoRA r64 K16 at 120 steps
#   LORAFZ LoRA r64 K16, then adapters + blocks [0,16) frozen at 75% of the steps (router branch >= 5e1202e4b; best
#        SFT compute + memory point), at 48 / 120 steps
#   LORAS1 seed-1 (different data order) rerun of LoRA r16 K16 at 48 / 120 steps: is the arm itself stable?
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
  # a run without its export but with stepN dirs was cut by a preemption/requeue (run_arm refuses a silent resume):
  # clear it and start over, unless it saves branch points (SAVE_INTERVAL), which must be resumed by hand
  if ls -d "$ROOT/runs/$name"/step* >/dev/null 2>&1 && ! printf '%s\n' "$@" | grep -q '^SAVE_INTERVAL='; then
    log "CLEAR partial $name (no export; restarting after a requeue)"; rm -rf -- "$ROOT/runs/$name" "$ROOT/work/$name"
  fi
  log "START $name"; env "$@" RUN="$name" bash "$ARM_SH" > "$ROOT/logs/$name.log" 2>&1; local rc=$?
  log "END $name rc=$rc"
  # a run without SAVE_INTERVAL is never branched from: drop its final trainer checkpoint (model + optimizer +
  # trainer state, ~110 GB) once the export exists (2026-10-23: 109 of them filled cubbins /data and hung a save)
  if [ $rc -eq 0 ] && [ -f "$ROOT/runs/$name/model_and_optim/.metadata" ] && ! printf '%s\n' "$@" | grep -q '^SAVE_INTERVAL='; then
    for d in "$ROOT/runs/$name"/step[0-9]*; do [ -d "$d" ] && rm -rf -- "$d"; done
  fi
  return $rc
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
    OTRA)
      for b in 46 92 185; do run "o3cd-OTR10M7nglr20-$b" ARM=OTR OTR_EXTRA="--st-otr-early-nograd" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 8 + 10)) || { log "FAIL $b in OTRA variant: continuing"; break; }; done
      for b in 40 80 160; do run "o3cd-OTRpyrlr20-$b" ARM=OTR OTR_EXTRA="--st-otr-stages 3:0.5,7:0.1,16:0.03" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 8 + 10)) || { log "FAIL $b in OTRA variant: continuing"; break; }; done
      for b in 31 62 124; do run "o3cd-OTR10M7mglr20-$b" ARM=OTR OTR_EXTRA="--st-otr-merge-dropped" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 6 + 10)) || { log "FAIL $b in OTRA variant: continuing"; break; }; done
      for b in 60 120 240; do run "o3cd-OTRpyrnglr20-$b" ARM=OTR OTR_EXTRA="--st-otr-stages 3:0.5,7:0.1,16:0.03 --st-otr-early-nograd" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 8 + 10)) || { log "FAIL $b in OTRA variant: continuing"; break; }; done ;;
    OTRX)
      for w in 1.0 0.3; do
        for b in 31 62 124 310; do
          run "o3cd-OTR10M7x${w/./}lr20-$b" ARM=OTR OTR_EXTRA="--st-otr-exit-loss-weight $w" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 6 + 10)) \
            || { log "FAIL exit-loss W=$w at $b: continuing"; break; }
        done
        for b in 40 80 160; do
          run "o3cd-OTR10M7ndx${w/./}lr20-$b" ARM=OTR OTR_DENSE=0 OTR_EXTRA="--st-otr-exit-loss-weight $w" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 8 + 10)) \
            || { log "FAIL exit-loss de0 W=$w at $b: continuing"; break; }
        done
      done
      # wide-keep periodic steps (peak-memory fallback): every 8th step routes at 50% instead of full context
      for b in 31 62 124; do
        run "o3cd-OTR10M7f50x10lr20-$b" ARM=OTR OTR_EXTRA="--st-otr-exit-loss-weight 1.0 --st-otr-dense-keep-frac 0.5" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 6 + 10)) \
          || { log "FAIL f50 exit-loss at $b: continuing"; break; }
      done ;;
    OTRY)
      for b in 46 92 185; do
        run "o3cd-OTR10M7ngf50lr20-$b" ARM=OTR OTR_EXTRA="--st-otr-early-nograd --st-otr-dense-keep-frac 0.5" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 8 + 10)) \
          || { log "FAIL ngf50 at $b: continuing"; break; }
      done
      for b in 60 120 240; do
        run "o3cd-OTR10M7ndnglr20-$b" ARM=OTR OTR_DENSE=0 OTR_EXTRA="--st-otr-early-nograd" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 10 + 10)) \
          || { log "FAIL ndng at $b: continuing"; break; }
      done ;;
    FRZ)
      run o3cd-FRZsmoke ARM=OTR OTR_KEEP=1.0 OTR_L=16 OTR_DENSE=0 OTR_EXTRA="--st-otr-freeze-after 5" MAX_STEPS_TOTAL=20 SMOKE=1 NOWANDB=1 EXPECTED_MIN=15 || exit 60
      for K in 16 8; do for f in 25 50; do for b in 12 24 48 120; do
        n=$(( (b * f + 50) / 100 )); [ $n -lt 1 ] && n=1
        run "o3cd-DFRZk${K}f${f}lr20-$b" ARM=OTR OTR_KEEP=1.0 OTR_L=$K OTR_DENSE=0 OTR_EXTRA="--st-otr-freeze-after $n" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DFRZ K=$K f=$f at $b: continuing"; break; }
      done; done; done ;;
    MOD)
      run o3cd-MODsmoke ARM=OTR OTR_KEEP=0.5 OTR_L=8 OTR_DENSE=0 OTR_EXTRA="--st-otr-mod-layers 8-31:2" MAX_STEPS_TOTAL=20 SMOKE=1 NOWANDB=1 EXPECTED_MIN=15 || exit 70
      for c in 50 25; do for b in 12 24 48 120; do
        run "o3cd-MODc${c}lr20-$b" ARM=OTR OTR_KEEP=0.$c OTR_L=8 OTR_DENSE=0 OTR_EXTRA="--st-otr-mod-layers 8-31:2" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL MOD c=$c at $b: continuing"; break; }
      done; done ;;
    PFZ)
      for b in 12 24 48 120; do
        run "o3cd-DPFZlr20-$b" ARM=OTR OTR_KEEP=1.0 OTR_L=12 OTR_DENSE=0 OTR_EXTRA="--st-otr-freeze-schedule 4:0.25,7:0.5,12:0.75" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DPFZ at $b: continuing"; break; }
      done ;;
    LORA)
      run o3cd-LORAsmoke ARM=OTR OTR_KEEP=1.0 OTR_L=16 OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 16 --st-otr-early-lora-scale 32" MAX_STEPS_TOTAL=20 SMOKE=1 NOWANDB=1 EXPECTED_MIN=15 || exit 80
      for K in 16 8; do for b in 12 24 48 120; do
        name="o3cd-DLORA16k${K}lr20-$b"
        run "$name" ARM=OTR OTR_KEEP=1.0 OTR_L=$K OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 16 --st-otr-early-lora-scale 32" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DLORA K=$K at $b: continuing"; break; }
        # router branch <= 1e5a14c85 never writes config.json["early_lora"] (SoftToken.export_config defined twice);
        # eval_devce rebuilds the block from the checkpoint's lora keys + provenance.json and refuses partial loads
        grep -q '"st_otr_early_lora": *16' "$ROOT/runs/$name/provenance.json" || { log "FAIL $name: provenance.json lacks st_otr_early_lora"; break; }
      done; done ;;
    MODB)
      for b in 12 24 48 120; do
        run "o3cd-MODc35lr20-$b" ARM=OTR OTR_KEEP=0.35 OTR_L=8 OTR_DENSE=0 OTR_EXTRA="--st-otr-mod-layers 8-31:2" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL MOD c=35 at $b: continuing"; break; }
      done ;;
    LOC)
      for b in 12 24 48 120; do
        run "o3cd-DLOClr20-$b" ARM=D EXTRA="--olmo3-train-window 4096" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 8)) || { log "FAIL DLOC at $b: continuing"; break; }
      done
      for b in 12 24 48 120; do
        n=$(( (b * 25 + 50) / 100 )); [ $n -lt 1 ] && n=1
        run "o3cd-DLOCfrzk8f25lr20-$b" ARM=OTR OTR_KEEP=1.0 OTR_L=8 OTR_DENSE=0 OTR_EXTRA="--st-otr-freeze-after $n" EXTRA="--olmo3-train-window 4096" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DLOC+frz at $b: continuing"; break; }
      done ;;
    LORA64)
      for K in 16 8; do for b in 12 24 48 120; do
        run "o3cd-DLORA64k${K}lr20-$b" ARM=OTR OTR_KEEP=1.0 OTR_L=$K OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 64 --st-otr-early-lora-scale 32" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DLORA64 K=$K at $b: continuing"; break; }
      done; done ;;
    LORALOC)
      for K in 16 8; do for b in 12 24 48 120; do
        name="o3cd-DLORA64k${K}loclr20-$b"
        run "$name" ARM=OTR OTR_KEEP=1.0 OTR_L=$K OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 64 --st-otr-early-lora-scale 32" EXTRA="--olmo3-train-window 4096" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DLORA+LOC K=$K at $b: continuing"; break; }
      done; done ;;
    REP)
      SPECS=""; OD="$ROOT/devce_rep"
      for r in stage2-yarn4:base/stage2-step47684:4 stage3-yarn8:base/stage3-step11921:8; do
        [ -f "$OD/${r%%:*}_dev32k_1024.json" ] || SPECS="$SPECS $r"; done
      for r in Dlr20-4 Dlr20-8 Dlr20-12 Dlr20-24 Dlr20-48 Dlr20-120-s1 Dlr20-240 Dlr20-480 DB8lr20-12 DB8lr20-24 DB8lr20-48 Dlr10-120; do
        [ -f "$ROOT/runs/o3cd-$r/model_and_optim/.metadata" ] && [ ! -f "$OD/${r}_dev32k_1024.json" ] && SPECS="$SPECS $r:runs/o3cd-$r:4"
      done
      log "REP specs:$SPECS"
      SPECS="$SPECS" SHARDS="dev32k_1024:128" OUT_DIR="$OD" EVAL_EXTRA="--ctx-window 8192" CODE="$CODE" PY="$PY" ROOT="$ROOT" \
        bash "$CODE/research/token_routing/olmo3_cptdrop/devce_multi.sh" > "$ROOT/logs/devce-rep-$SLURM_JOB_ID.log" 2>&1 \
        || log "WARN: rescore had failures"
      grep -h "merged" "$ROOT/logs/devce-rep-$SLURM_JOB_ID.log" ;;
    LORA64X)
      for b in 240 480; do
        run "o3cd-DLORA64k16lr20-$b" ARM=OTR OTR_KEEP=1.0 OTR_L=16 OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 64 --st-otr-early-lora-scale 32" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DLORA64 K16 at $b: continuing"; break; }
      done ;;
    LORA128)
      for b in 48 120; do
        run "o3cd-DLORA128k16lr20-$b" ARM=OTR OTR_KEEP=1.0 OTR_L=16 OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 128 --st-otr-early-lora-scale 32" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DLORA128 K16 at $b: continuing"; break; }
      done ;;
    LORA64K24)
      for b in 48 120; do
        run "o3cd-DLORA64k24lr20-$b" ARM=OTR OTR_KEEP=1.0 OTR_L=24 OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 64 --st-otr-early-lora-scale 32" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DLORA64 K24 at $b: continuing"; break; }
      done ;;
    LORA64S1)
      run "o3cd-DLORA64k16lr20-120-s1" ARM=OTR OTR_KEEP=1.0 OTR_L=16 OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 64 --st-otr-early-lora-scale 32" SEED=1 MAX_STEPS_TOTAL=120 EXPECTED_MIN=40 \
        || log "FAIL DLORA64 K16 s1" ;;
    LORAFZ)
      for b in 48 120; do
        n=$(( b * 3 / 4 ))
        run "o3cd-DLORA64k16fa75lr20-$b" ARM=OTR OTR_KEEP=1.0 OTR_L=16 OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-freeze-after $n" MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DLORA64 fa75 at $b: continuing"; break; }
      done ;;
    LORAS1)
      for b in 48 120; do
        run "o3cd-DLORA16k16lr20-$b-s1" ARM=OTR OTR_KEEP=1.0 OTR_L=16 OTR_DENSE=0 OTR_EXTRA="--st-otr-early-lora 16 --st-otr-early-lora-scale 32" SEED=1 MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 10)) \
          || { log "FAIL DLORA16 s1 at $b: continuing"; break; }
      done ;;
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
      for r in $(for K in 16 8; do for b in 12 24 48 120; do echo DLORA64k${K}loclr20-$b DLORA64k${K}lr20-$b; done; done) DLORA64k16lr20-240 DLORA64k16lr20-480 DLORA128k16lr20-48 DLORA128k16lr20-120 DLORA64k16lr20-120-s1 DLORA64k24lr20-48 DLORA64k24lr20-120 $(for b in 12 24 48 120; do echo MODc35lr20-$b DLOClr20-$b DLOCfrzk8f25lr20-$b; done) $(for K in 16 8; do for b in 12 24 48 120; do echo DLORA16k${K}lr20-$b; done; done) DPFZlr20-12 DPFZlr20-24 DPFZlr20-48 DPFZlr20-120 $(for c in 50 25; do for b in 12 24 48 120; do echo MODc${c}lr20-$b; done; done) $(for K in 16 8; do for f in 25 50; do for b in 12 24 48 120; do echo DFRZk${K}f${f}lr20-$b; done; done; done) OTR10M7ngf50lr20-46 OTR10M7ngf50lr20-92 OTR10M7ngf50lr20-185 OTR10M7ndnglr20-60 OTR10M7ndnglr20-120 OTR10M7ndnglr20-240 OTR10M7f50x10lr20-31 OTR10M7f50x10lr20-62 OTR10M7f50x10lr20-124 OTR10M7ndx10lr20-40 OTR10M7ndx10lr20-80 OTR10M7ndx10lr20-160 OTR10M7ndx03lr20-40 OTR10M7ndx03lr20-80 OTR10M7ndx03lr20-160 OTR10M7x10lr20-31 OTR10M7x10lr20-62 OTR10M7x10lr20-124 OTR10M7x10lr20-310 OTR10M7x03lr20-31 OTR10M7x03lr20-62 OTR10M7x03lr20-124 OTR10M7x03lr20-310 OTR10M7mglr20-31 OTR10M7mglr20-62 OTR10M7mglr20-124 OTR10M7nglr20-46 OTR10M7nglr20-92 OTR10M7nglr20-185 OTRpyrlr20-40 OTRpyrlr20-80 OTRpyrlr20-160 OTRpyrnglr20-60 OTRpyrnglr20-120 OTRpyrnglr20-240 OTR10M7lr20-31 OTR10M7lr20-62 OTR10M7lr20-124 OTR10M7lr20-310 OTR10M7ndlr20-40 OTR10M7ndlr20-80 OTR10M7ndlr20-160 Dlr20-4 Dlr20-8 IRANDlr20-22 IRANDlr20-43 SD20lr20-65 SD20lr20-130 SD20lr20-260 Dlr10-12 Dlr05-120 Dlr20-12 Dlr20-24 Dlr20-48 DB8lr20-12 DB8lr20-24 DB8lr20-48 Dlr20-120-s1 IRANDlr20-65 IRANDlr20-130 IRANDlr20-260 IFLlr20-54 IFLlr20-108 IFLlr20-215 Dlr10-120 Dlr20-480 IRANDlr20-650 IRANDlr20-1300 IRANDlr20-2600 HOlr20-120 HOlr20-240 HOlr20-480; do
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
