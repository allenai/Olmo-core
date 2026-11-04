#!/bin/bash
# gradmap_plot.py on mooney (CPU; reads the node-local gradmap npz). matplotlib is pip-installed into a node-local
# target dir (the node envs have none). PNGs land in $ROOT/figs/<SET>; copy them into the repo afterwards.
#   sbatch -w mooney -J gradmap-plot --export=ALL,SRC=<checkout>,IN=/data/prasann/gradmap/out,SET=8k run_gradmap_plot.sbatch
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/data/prasann/%x-%j.out
set -eo pipefail
SRC=${SRC:?set SRC}; IN=${IN:?set IN}; SET=${SET:?set SET}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
ROOT=/data/prasann/gradmap
LIB=$ROOT/pylib
CODE=$ROOT/code_${SLURM_JOB_ID}
mkdir -p "$CODE" "$ROOT/tmp"
(cd "$SRC" && tar -c research/token_routing/grad_concentration/gradmap_plot.py) | tar -x -C "$CODE"
export TMPDIR=$ROOT/tmp HF_HOME=$ROOT/hf_cache MPLCONFIGDIR=$ROOT/tmp/mpl
if [ ! -d "$LIB/matplotlib" ]; then
  "$PY" -m pip install -q --target "$LIB" --no-deps matplotlib contourpy cycler fonttools kiwisolver pyparsing
fi
export PYTHONPATH="$LIB"
"$PY" -c "import matplotlib, numpy; print('matplotlib', matplotlib.__version__, 'numpy', numpy.__version__)"
"$PY" -u "$CODE/research/token_routing/grad_concentration/gradmap_plot.py" --in-dir "$IN" --out-dir "$ROOT/figs/$SET" \
  --labels "${LABELS:-base,f1952,f7808,recipe}" ${ARGS:-}
ls "$ROOT/figs/$SET" | head -80
echo "PLOT_DONE $SET"
