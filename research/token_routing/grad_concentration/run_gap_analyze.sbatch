#!/bin/bash
# analyze_gap.py on the node that holds the sketches; copies the (small) summary JSON + log back into SRC/results.
#   sbatch -w cubbins -J gradgap-analyze-<labels> --export=ALL,SRC=<checkout>,LABELS=f1952 run_gap_analyze.sbatch
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/%x-%j.out
set -eo pipefail
SRC=${SRC:?}; LABELS=${LABELS:?}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
OUTROOT=/data/prasann/gradgap
CODE=$OUTROOT/code_${SLURM_JOB_ID}
mkdir -p "$CODE" "$OUTROOT/tmp"
(cd "$SRC" && tar -c research/token_routing/grad_concentration/analyze_gap.py) | tar -x -C "$CODE"
export TMPDIR=$OUTROOT/tmp
TAG=$(echo "$LABELS" | tr ',' '-')
"$PY" -u "$CODE/research/token_routing/grad_concentration/analyze_gap.py" --gap-dir "$OUTROOT/sketches" \
  --labels "$LABELS" --out "$OUTROOT/gap_${TAG}.json" ${ARGS:-} 2>&1 | tee "$OUTROOT/gap_${TAG}.log"
mkdir -p "$SRC/research/token_routing/grad_concentration/results"
cp "$OUTROOT/gap_${TAG}.json" "$OUTROOT/gap_${TAG}.log" "$SRC/research/token_routing/grad_concentration/results/"
echo ANALYZE_DONE
