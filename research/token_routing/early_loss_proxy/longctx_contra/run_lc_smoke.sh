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
for REL in ${PRESYNC:-}; do  # rel paths under early_loss_proxy/ to pull S3 -> weka first (new inputs)
  "$PY" research/token_routing/early_loss_proxy/s3sync.py down "$S3/$REL" "$ROOT/$REL" 2>&1 | tail -n 1
done
echo "$SPEC_B64" | base64 -d > "$OUT/spec_${JID}.tsv"
while IFS=$'\t' read -r NAME MODE NGPU VARIANT LR DATA SEQLEN GB STEPS PSETS FLAGS; do
  [ -z "$NAME" ] && continue
  if [ -n "${SAVE_ROOT:-}" ]; then
    # real arms: save folder on weka, never wiped, so a preempted / restarted job auto-resumes from the newest
    # (ephemeral) checkpoint and the permanent stepN checkpoints stay for checkpoint-then-score
    SAVE=$SAVE_ROOT/$NAME; mkdir -p "$SAVE"
  else
    SAVE=$LOC/runs/$NAME; rm -rf "$SAVE"; mkdir -p "$SAVE"
  fi
  echo "=== $NAME mode=$MODE ngpu=$NGPU variant=$VARIANT data=$DATA seq=$SEQLEN gb=$GB steps=$STEPS flags=$FLAGS start $(date +%T)"
  T0=$(date +%s)
  if [ -n "${KEEPCKPT:-}" ]; then CKPTFL="--max-checkpoints 0"; else CKPTFL="--no-final-checkpoint"; fi
  TRAIN=(--task mt4 --variant "$VARIANT" --model-family qwen3_5 --model-scale 4b --run-name "$NAME"
         --base-checkpoint "$BASE" --data "$ROOT/shards/$DATA" --work-dir "$LOC/work_$NAME" --save-folder "$SAVE"
         --seq-len "$SEQLEN" --epochs 1 --lr "$LR" --global-batch "$GB" --micro-batch-instances 1 --seed 0
         --no-compile --no-wandb $CKPTFL --max-steps "$STEPS")
  # live progress for long runs: every SYNC_S seconds copy the log tail and the result JSON to weka + S3
  ( while sleep "${SYNC_S:-600}"; do
      tail -c 2000000 "$LOC/$NAME.log" > "$OUT/$NAME.live.log" 2>/dev/null
      sync_out "$OUT/$NAME.live.log" "$S3/results/$TAG" >/dev/null
      [ -f "$OUT/$NAME.json" ] && sync_out "$OUT/$NAME.json" "$S3/results/$TAG" >/dev/null
    done ) &
  SYNCPID=$!
  # @PREV@ in FLAGS = the newest stepN checkpoint of the PREVIOUS run in this spec (chained segments: --branch-from @PREV@)
  if [[ "$FLAGS" == *@PREV@* ]]; then
    PREVCK=$(ls -d "${PREVSAVE:?no previous run}"/step* 2>/dev/null | sort -V | tail -1)
    [ -n "$PREVCK" ] || { echo "FATAL: no checkpoint in $PREVSAVE"; exit 4; }
    FLAGS=${FLAGS//@PREV@/$PREVCK}
    echo "[chain] $NAME branches from $PREVCK"
  fi
  # --probe-* words in FLAGS go to dense_probe (before the --), the rest to the trainer
  PFL=(); TFL=(); set -f; W=($FLAGS); set +f; i=0
  while [ $i -lt ${#W[@]} ]; do
    if [[ ${W[$i]} == --probe-* ]]; then PFL+=("${W[$i]}" "${W[$((i+1))]}"); i=$((i+2)); else TFL+=("${W[$i]}"); i=$((i+1)); fi
  done
  FLAGS="${TFL[*]}"
  if [ "$MODE" = resume ]; then
    # resume check: stop at STEPS/2 (an ephemeral checkpoint must exist by then), rerun the same run name / save
    # folder to STEPS; the second log must show the trainer loading the checkpoint and continuing past it
    H=$((STEPS / 2))
    TRAIN1=("${TRAIN[@]}"); TRAIN1[$((${#TRAIN1[@]} - 1))]=$H
    # shellcheck disable=SC2086
    "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=$((29000 + RANDOM % 1000)) \
      recipes/train/research_trainer/train_ctc_suite.py "${TRAIN1[@]}" $FLAGS > "$LOC/$NAME.part1.log" 2>&1
    echo "[resume] part 1 rc=$? (max-steps $H); save folder: $(ls "$SAVE" | tr '\n' ' ')"
    cp "$LOC/$NAME.part1.log" "$OUT/$NAME.part1.log"
    # shellcheck disable=SC2086
    "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=$((29000 + RANDOM % 1000)) \
      recipes/train/research_trainer/train_ctc_suite.py "${TRAIN[@]}" $FLAGS > "$LOC/$NAME.log" 2>&1
    grep -a -i "checkpoint\|resum\|step=" "$LOC/$NAME.log" | cut -c1-250 | head -20
  elif [ "$MODE" = probe ]; then
    # shellcheck disable=SC2086
    "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=$((29000 + RANDOM % 1000)) \
      research/token_routing/early_loss_proxy/dense_probe.py --probe-steps "$STEPS" --probe-sets "$PSETS" \
      --probe-heldout "$HELDOUT" --probe-out "$OUT/$NAME.json" --probe-memtrace 2 "${PFL[@]}" -- "${TRAIN[@]}" $FLAGS \
      > "$LOC/$NAME.log" 2>&1
  else
    # shellcheck disable=SC2086
    "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=$((29000 + RANDOM % 1000)) \
      recipes/train/research_trainer/train_ctc_suite.py "${TRAIN[@]}" $FLAGS > "$LOC/$NAME.log" 2>&1
  fi
  RC=$?
  T1=$(date +%s)
  kill "$SYNCPID" 2>/dev/null
  cp "$LOC/$NAME.log" "$OUT/$NAME.log"
  [ -f "$SAVE/flops.json" ] && cp "$SAVE/flops.json" "$OUT/$NAME.flops.json"
  grep -a "dense-probe\|flop-meter\] wrote\|step=\|train/CE\|TFLOPs_per_GPU\|total petaflops\|OutOfMemory\|out of memory\|Error\|nan" "$LOC/$NAME.log" | cut -c1-300 | tail -n 40
  [ $RC -ne 0 ] && tail -n 40 "$LOC/$NAME.log"
  echo "{\"name\": \"$NAME\", \"rc\": $RC, \"wall_s\": $((T1 - T0)), \"host\": \"$(hostname)\"}" > "$OUT/$NAME.rc.json"
  for f in "$OUT/$NAME".*; do sync_out "$f" "$S3/results/$TAG"; done
  [ -z "${SAVE_ROOT:-}" ] && rm -rf "$SAVE"
  rm -rf "$LOC/work_$NAME"
  PREVSAVE=$SAVE
  echo "=== $NAME end $(date +%T) rc=$RC wall $((T1 - T0))s"
done < "$OUT/spec_${JID}.tsv"
rm -rf "$LOC"
echo "LC_SMOKE_DONE"
