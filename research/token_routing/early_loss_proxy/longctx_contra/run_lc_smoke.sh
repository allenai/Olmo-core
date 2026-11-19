#!/bin/bash
# Beaker payload for the longctx_contra feasibility smokes: runs back to back, each either through dense_probe.py
# (MODE=probe: per-step train_s / peak GiB / FLOPs in the result JSON, plus a dense held-out forward at step 0) or the
# bare trainer (MODE=direct: needed for context-parallel runs, which dense_probe's unsharded forward does not support).
# Env: TAG SPEC_B64 (TSV: name<TAB>mode<TAB>ngpu<TAB>variant<TAB>lr<TAB>data<TAB>seqlen<TAB>gb<TAB>steps<TAB>probe_sets<TAB>flags)
# Results + logs: weka $ROOT/results/$TAG/ and s3://ai2-llm/checkpoints/prasanns/early_loss_proxy/results/$TAG/.
set -o pipefail
TAG=${TAG:?}
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
ROOT=$W/early_loss_proxy
S3=s3://ai2-llm/checkpoints/prasanns/early_loss_proxy
BASE=${BASE:-$W/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim}
HELDOUT=${HELDOUT:-$ROOT/heldout_lc.npz}
OUT=$ROOT/results/$TAG
JID=${BEAKER_JOB_ID:-local$$}
LOC=/tmp/lc_$JID
PY=${PY:-python}
mkdir -p "$OUT" "$LOC/tmp" "$LOC/runs"
export PYTHONPATH="$PWD/src" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TMPDIR=$LOC/tmp TRITON_CACHE_DIR=$LOC/triton_cache HF_HOME=$LOC/hf_cache
export AWS_PROFILE=${AWS_PROFILE:-S3} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "[lc-smoke] job $JID host $(hostname) commit $(git rev-parse --short HEAD 2>/dev/null) tag $TAG start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sync_out() { "$PY" research/token_routing/early_loss_proxy/s3sync.py up --force "$@" 2>&1 | tail -n 1; }
echo "$SPEC_B64" | base64 -d > "$OUT/spec_${JID}.tsv"
while IFS=$'\t' read -r NAME MODE NGPU VARIANT LR DATA SEQLEN GB STEPS PSETS FLAGS; do
  [ -z "$NAME" ] && continue
  SAVE=$LOC/runs/$NAME; rm -rf "$SAVE"; mkdir -p "$SAVE"
  echo "=== $NAME mode=$MODE ngpu=$NGPU variant=$VARIANT data=$DATA seq=$SEQLEN gb=$GB steps=$STEPS flags=$FLAGS start $(date +%T)"
  T0=$(date +%s)
  TRAIN=(--task mt4 --variant "$VARIANT" --model-family qwen3_5 --model-scale 4b --run-name "$NAME"
         --base-checkpoint "$BASE" --data "$ROOT/shards/$DATA" --work-dir "$LOC/work_$NAME" --save-folder "$SAVE"
         --seq-len "$SEQLEN" --epochs 1 --lr "$LR" --global-batch "$GB" --micro-batch-instances 1 --seed 0
         --no-compile --no-wandb --no-final-checkpoint --max-steps "$STEPS")
  if [ "$MODE" = probe ]; then
    # shellcheck disable=SC2086
    "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=$((29000 + RANDOM % 1000)) \
      research/token_routing/early_loss_proxy/dense_probe.py --probe-steps "$STEPS" --probe-sets "$PSETS" \
      --probe-heldout "$HELDOUT" --probe-out "$OUT/$NAME.json" --probe-memtrace 2 -- "${TRAIN[@]}" $FLAGS \
      > "$LOC/$NAME.log" 2>&1
  else
    # shellcheck disable=SC2086
    "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=$((29000 + RANDOM % 1000)) \
      recipes/train/research_trainer/train_ctc_suite.py "${TRAIN[@]}" $FLAGS > "$LOC/$NAME.log" 2>&1
  fi
  RC=$?
  T1=$(date +%s)
  cp "$LOC/$NAME.log" "$OUT/$NAME.log"
  [ -f "$SAVE/flops.json" ] && cp "$SAVE/flops.json" "$OUT/$NAME.flops.json"
  grep -a "dense-probe\|flop-meter\] wrote\|step=\|train/CE\|TFLOPs_per_GPU\|total petaflops\|OutOfMemory\|out of memory\|Error\|nan" "$LOC/$NAME.log" | cut -c1-300 | tail -n 40
  [ $RC -ne 0 ] && tail -n 40 "$LOC/$NAME.log"
  echo "{\"name\": \"$NAME\", \"rc\": $RC, \"wall_s\": $((T1 - T0)), \"host\": \"$(hostname)\"}" > "$OUT/$NAME.rc.json"
  for f in "$OUT/$NAME".*; do sync_out "$f" "$S3/results/$TAG"; done
  rm -rf "$SAVE" "$LOC/work_$NAME"
  echo "=== $NAME end $(date +%T) rc=$RC wall $((T1 - T0))s"
done < "$OUT/spec_${JID}.tsv"
rm -rf "$LOC"
echo "LC_SMOKE_DONE"
