#!/bin/bash
# cpt_gradmap_plot.py on cubbins (CPU; reads the node-local cpt_gradmap npz). matplotlib is pip-installed into a node-local
# target dir (the node envs have none). PNGs + summary.json land in $WORK/cpt_gradmap_figs; copy them into the repo after.
#   sbatch -w cubbins --export=ALL,SRC=<checkout> run_cpt_gradmap_plot.sbatch
#SBATCH --job-name=cptgm-plot
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/data/prasann/grad_composition/logs/%x-%j.out
set -eo pipefail
SRC=${SRC:?set SRC}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
WORK=/data/prasann/grad_composition
LIB=$WORK/pylib
CODE=$WORK/code_plot_${SLURM_JOB_ID}
OUTF=${OUTF:-$WORK/cpt_gradmap_figs}
mkdir -p "$CODE" "$WORK/tmp" "$OUTF"
(cd "$SRC" && tar -c research/token_routing/grad_concentration/cpt_gradmap_plot.py) | tar -x -C "$CODE"
export TMPDIR=$WORK/tmp MPLCONFIGDIR=$WORK/tmp/mpl
if [ ! -d "$LIB/matplotlib" ]; then
  "$PY" -m pip install -q --target "$LIB" --no-deps matplotlib contourpy cycler fonttools kiwisolver pyparsing
fi
export PYTHONPATH="$LIB"
"$PY" -u "$CODE/research/token_routing/grad_concentration/cpt_gradmap_plot.py" --in-dir "$WORK/cpt_gradmap" \
  --sets "${SETS:-32k:base_32k:cpt_32k:stage3_32k,64k:base_64k:cpt_64k,128k:base_128k:cpt_128k}" \
  --names "${NAMES:-base_32k=stage2 base (YaRN x4),cpt_32k=dense CPT 32k 480 steps (Dlr20-480),stage3_32k=OLMo 3 stage 3 (YaRN x8; reference),base_64k=stage2 base (YaRN x8),cpt_64k=dense CPT 64k 480 steps (D64lr20-480),base_128k=stage2 base (YaRN x16),cpt_128k=dense CPT 128k 120 steps (D128lr20-120)}" \
  --vocab "$WORK/dolma2_vocab.json" --out-dir "$OUTF" ${ARGS:-}
ls "$OUTF" | wc -l; du -sh "$OUTF"
echo "PLOT_DONE"
