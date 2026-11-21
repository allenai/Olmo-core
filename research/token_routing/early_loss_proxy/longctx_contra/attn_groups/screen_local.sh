#!/bin/bash
# Group-attention screen (2026-11-20): per-step actual PF / peak GiB / s on real lc256kmix_gdd 256k rows, gdd256's
# settings, 1 GPU per config, for attention off and G = 1, 2, 4, 8, on gold-dropped rows (keep 0.05-0.5, whole 0.05)
# and on WHOLE rows (whole 1.0). full_frac 0 here (every row grouped) so the per-config cost is clean.
# Runs from a staged tree on a node's /data: D=<staged repo> POOL=<mini pool dir> HO=<held-out npz> OUT=<dir>.
# Usage (inside an allocation with NG GPUs): bash screen_local.sh "<cfg> <cfg> ..."  cfg = drop_off | drop_g4 | whole_g2 ...
set -u
D=${D:?}; POOL=${POOL:?}; HO=${HO:?}; OUT=${OUT:?}
PY=${PY:-/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python}
BASE=${BASE:-/data/prasann/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim}
STEPS=${STEPS:-12}
mkdir -p "$OUT"
export PYTHONPATH="$D/src" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32 --attn-backend flash_2 --activation-checkpointing full --skip-dry-run --st-otr-warmup-random --prefix-chunk 4096 --no-data-shuffle"
gpu=0
for cfg in $1; do
  mode=${cfg%_*}; g=${cfg#*_}
  if [ "$mode" = drop ]; then GD="--st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.05"; else GD="--st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 1.0"; fi
  if [ "$g" = off ]; then AG=""; else AG="--st-attn-groups ${g#g} --st-attn-group-full-frac 0"; fi
  L=/data/prasann/attn_groups/run_${cfg}_$$
  mkdir -p "$L/tmp"
  ( cd "$D" && CUDA_VISIBLE_DEVICES=$gpu TMPDIR=$L/tmp TRITON_CACHE_DIR=$L/triton HF_HOME=$L/hf HOME=$L \
    "$PY" -m torch.distributed.run --nproc_per_node=1 --master_port=$((29100 + gpu)) \
      research/token_routing/early_loss_proxy/dense_probe.py --probe-steps $STEPS --probe-sets lc16k:1 \
      --probe-heldout "$HO" --probe-out "$OUT/$cfg.json" -- \
      --task mt4 --variant softtoken --model-family qwen3_5 --model-scale 4b --run-name "scr_$cfg" \
      --base-checkpoint "$BASE" --data "$POOL" --work-dir "$L/work" --save-folder "$L/save" --seq-len 262144 \
      --epochs 1 --lr 1e-5 --global-batch 1 --micro-batch-instances 1 --seed 0 --no-compile --no-wandb \
      --no-final-checkpoint --max-steps $STEPS $R $GD $AG > "$OUT/$cfg.log" 2>&1; echo "rc=$?" >> "$OUT/$cfg.log" ) &
  gpu=$((gpu + 1))
done
wait
echo SCREEN_DONE
