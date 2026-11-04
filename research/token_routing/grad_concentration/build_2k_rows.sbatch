#!/bin/bash
# Short (2k-rung) rows for gradmap.py: the same generator + tokenizer call as the 8k fresh pools
# (research/token_routing/ds64_fast8k/build_mt4_fresh_beaker.sh) at --rungs 2k, a new seed, N rows per task, written as
# <OUT>/<task>_fresh so probe.load_rows reads them unchanged. CPU only (pool auto = seed pool from the Hub).
#   sbatch -w mooney -J gradmap-build2k --export=ALL,SRC=<checkout> build_2k_rows.sbatch
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/data/prasann/%x-%j.out
set -uo pipefail
SRC=${SRC:?set SRC}; N=${N:-160}; SEED=${SEED:-9917}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
ROOT=/data/prasann/gradmap
CODE=$ROOT/code_${SLURM_JOB_ID}
OUT=$ROOT/shards2k
mkdir -p "$CODE" "$ROOT/tmp" "$OUT"
(cd "$SRC" && tar -c --exclude=.git --exclude='results*' src ctc recipes/data) | tar -x -C "$CODE"
export TMPDIR=$ROOT/tmp HF_HOME=$ROOT/hf_cache PYTHONPATH="$CODE/src:$CODE/ctc/src" PYTHONWARNINGS=ignore \
  TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$CODE"
for TASK in contradiction nq outlier oolong; do
  case "$TASK" in
    nq) CONV=retrieval; CHUNK=document; EXTRA="" ;;
    oolong) CONV=oolong; CHUNK=line; EXTRA="--gold-sidecar-empty-for-missing" ;;
    *) CONV=$TASK; CHUNK=document; EXTRA="" ;;
  esac
  POOL=$ROOT/pool2k_$TASK
  echo "--- $TASK 2k: $N rows seed $SEED $(date +%T)"
  "$PY" -m ctc.data.cli build --task $TASK --out $POOL --split train --rungs 2k --train $N --seed $SEED --pool auto --force \
    || { echo "!!! pool FAILED $TASK"; exit 1; }
  echo "pool rows: $(wc -l < $POOL/$TASK/train.jsonl)"
  rm -rf "$OUT/${TASK}_fresh"; mkdir -p "$OUT/${TASK}_fresh"
  "$PY" recipes/data/convert_unified_to_document_landmark.py \
    --input-jsonl $POOL/$TASK/train.jsonl --task $CONV --out-dir $OUT/${TASK}_fresh --emit dense --marker-set qwen3_5 \
    --tokenizer Qwen/Qwen3.5-0.8B-Base --seq-len 24576 --query-position after --cot-mode none \
    --chunk-by $CHUNK --emit-gold-sidecar $EXTRA --num-proc 8 || { echo "!!! tokenize FAILED $TASK"; exit 1; }
done
ls -la $OUT/*_fresh | head -40
echo "BUILD2K_DONE"
