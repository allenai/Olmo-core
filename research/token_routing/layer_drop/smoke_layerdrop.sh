#!/bin/bash
# Layer-drop wall-clock smoke (2026-10-22): Qwen3.5-4B repaired base, fast8k mt4s488 shard, 2 GPUs,
# the tokdrop_lr geometry (seq 10240, gb 16, micro 2, unpacked, no compile). Arms run back to back
# in ONE allocation so they share the node:
#   dense | dense + LayerDrop p=0.25 | tier67a50m32 | tier67a50m32 + LayerDrop p=0.25
# (ARMS=... also: dense_ldlin / tier67_ldlin = linear p=0.4 excl. first 4 / last 2; dense_fix3)
# Run on the node with the code staged under $ROOT (git archive | srun tar -x), e.g.
#   srun --partition=jsteinhardt --qos=preemptive -w sneetches --gres=gpu:2 ... bash smoke_layerdrop.sh
# Per arm: logs/<arm>.log (trainer step times + flop_meter metrics). MAX_STEPS (default 12).
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
D=${D:-/data/prasann/layerskip}
PY=${PY:-/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python}
MAX_STEPS=${MAX_STEPS:-12}
ARMS=${ARMS:-"dense dense_ld25 tier67 tier67_ld25"}
export PYTHONPATH="$ROOT/src" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export HOME=/data/prasann/home HF_HUB_CACHE=/data/prasann/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export TRITON_CACHE_DIR=/data/prasann/triton_cache TMPDIR=/data/prasann/tmp
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
mkdir -p "$D/logs" "$TMPDIR" "$TRITON_CACHE_DIR"
cd "$ROOT"
RULE="--st-keep-mode gold_plus_random --st-keep-frac 0.6667 --st-keep-token-rule first_last --st-keep-token-k 0.05 --st-kept-doc-k 0.5 --st-keep-token-min 32 --st-header-extra-tokens 8 --st-drop-slots --attn-backend flash_2"
LD="--layer-drop-p 0.25"
GS_STATIC=0.0845,0.0236,0.0304,0.0283,0.0256,0.0325,0.0426,0.0372,0.0249,0.0267,0.0512,0.0508,0.0314,0.0255,0.0491,0.0459,0.0204,0.0188,0.0316,0.0236,0.0083,0.0073,0.0136,0.0118,0.0040,0.0034,0.0056,0.0102,0.0073,0.0095,0.0218,0.0876
GS_HI=$(seq -s, 0 31)
for arm in $ARMS; do
  PROF=""; case $arm in *_prof) PROF="--torch-profile" ;; esac
  case $arm in
    dense) V=full; X="" ;;
    dense_ld25) V=full; X="$LD" ;;
    tier67) V=softtoken; X="$RULE" ;;
    tier67_ld25) V=softtoken; X="$RULE $LD" ;;
    dense_ldlin) V=full; X="--layer-drop-p 0.4 --layer-drop-schedule linear --layer-drop-exclude-first 4 --layer-drop-exclude-last 2" ;;
    dense_fix3) V=full; X="--train-skip-layers 20,24,28" ;;  # three GatedDeltaNet blocks
    # grad layer select A/B (2026-10-28): static top-8 = 31,0,10,11,14,15,6,7 (cut at block 0 every step) vs the top
    # 8 blocks only (cut at 24); *_prof = the same with --torch-profile (top CUDA ops over steps 3-4)
    dense_prof) V=full; X="" ;;
    gsu25st|gsu25st_prof) V=full; X="--grad-select-frac 0.25 --grad-select-mode update --grad-select-rank static --grad-select-static-energy $GS_STATIC" ;;
    gsu25hi) V=full; X="--grad-select-frac 0.25 --grad-select-mode update --grad-select-rank static --grad-select-static-energy $GS_HI" ;;
    tier67_ldlin) V=softtoken; X="$RULE --layer-drop-p 0.4 --layer-drop-schedule linear --layer-drop-exclude-first 4 --layer-drop-exclude-last 2" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  run=layerdrop-smoke-$arm
  rm -rf "$D/save/$run" "$D/work/$run"
  echo "=== $arm: --variant $V $X" | tee "$D/logs/$arm.log"
  /usr/bin/time -f "[wall] %e s" "$PY" -m torch.distributed.run --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
    recipes/train/research_trainer/train_ctc_suite.py \
    --task mt4 --variant $V --model-scale 4b --model-family qwen3_5 --no-wandb \
    --run-name "$run" --base-checkpoint "$D/base/q35-4b-base-markerfix/model_and_optim" \
    --data "$D/shards/mt4s488" --work-dir "$D/work/$run" --save-folder "$D/save/$run" \
    --seq-len 10240 --global-batch 16 --micro-batch-instances 2 --epochs 1 --lr 5e-6 --seed 0 \
    --max-steps "$MAX_STEPS" --no-final-checkpoint --no-compile $X $PROF >> "$D/logs/$arm.log" 2>&1
  echo "=== $arm exit $?" | tee -a "$D/logs/$arm.log"
  [ -f "$D/save/$run/flops.json" ] && cat "$D/save/$run/flops.json" >> "$D/logs/$arm.log"
done
