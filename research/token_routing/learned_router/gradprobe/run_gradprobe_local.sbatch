#!/usr/bin/env bash
# Gradient token-importance probe (grad_importance.py) on one GPU, from a code tree staged on node-local /data
# (git ls-files | tar on the login node | srun tar -x into $CODE). Checkpoint + rung rows must already be on
# this node's /data/prasann/devloss_grid (devloss_grid/stage_to_node.sh, or one rsync of the ckpt dir).
#   CODE=/data/prasann/gradprobe/code_<ts> TASK=nq RUNG=8k ROWS=2 OUT=/data/prasann/gradprobe/out/nq_8k_r2.json \
#     [SCORE_CKPT=<ckpt dir>: gradients from it, drop + CE with CKPT, via cross_score.py] sbatch --nodelist=lorax research/token_routing/learned_router/gradprobe/run_gradprobe_local.sbatch
#SBATCH --job-name=gradprobe
#SBATCH --partition=berkeleynlp
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/gradprobe_%j.log

set -uo pipefail
: "${CODE:?set CODE=<staged code tree on /data>}" "${OUT:?set OUT=<json on /data>}"
TASK=${TASK:-nq}; RUNG=${RUNG:-8k}; ROWS=${ROWS:-2}
case "$CODE$OUT" in /data/*/data/*) ;; *) echo "!!! CODE and OUT must be on node-local /data"; exit 1;; esac
cd "$CODE"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$CODE/src:$CODE/research/common" PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HOME=/data/prasann HF_HOME=/data/prasann/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore DEVLOSS_SKIP_CHECKS=1
export TMPDIR=/data/prasann/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache XDG_CACHE_HOME=/data/prasann/xdg_cache TORCHINDUCTOR_CACHE_DIR=/data/prasann/inductor_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$(dirname "$OUT")"
[ -d /usr/local/cuda-12.8 ] && { export CUDA_HOME=/usr/local/cuda-12.8; export PATH=$CUDA_HOME/bin:$PATH; }
TOK=/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
ROOT=/data/prasann/devloss_grid
CKPT=${CKPT:-$ROOT/ckpts/ctc-4b-$TASK-full}
if [ -f "$CKPT/.text_export_done" ]; then FMT=hf; elif [ -f "$CKPT/model_and_optim/.metadata" ]; then FMT=distcp
elif [ -f "$CKPT/.resave_done" ]; then FMT=olmo_st
else echo "!!! no staged checkpoint (HF export, distcp or olmo_st) at $CKPT"; exit 1; fi
fmt_of() { if [ -f "$1/.text_export_done" ]; then echo hf; elif [ -f "$1/model_and_optim/.metadata" ]; then echo distcp
  elif [ -f "$1/.resave_done" ]; then echo olmo_st; else echo "!!! no staged checkpoint at $1" >&2; exit 1; fi; }
echo "== gradprobe $TASK@$RUNG rows=$ROWS on $(hostname -s) job ${SLURM_JOB_ID:-?} ($(date +%T)) code=$CODE"
if [ -n "${SCORE_CKPT:-}" ]; then  # cross-model: gradients from SCORE_CKPT, drop + CE with CKPT (cross_score.py)
  "$PY" -u research/token_routing/learned_router/gradprobe/cross_score.py --task "$TASK" --rung "$RUNG" --rows "$ROWS" \
    --ckpt "$CKPT" --ckpt-format "$FMT" --score-ckpt "$SCORE_CKPT" --score-ckpt-format "$(fmt_of "$SCORE_CKPT")" \
    --data-root "$ROOT/data" --tokenizer "$TOK" --out "$OUT" ${EXTRA:-}
else
"$PY" -u research/token_routing/learned_router/gradprobe/grad_importance.py --task "$TASK" --rung "$RUNG" --rows "$ROWS" \
  --ckpt "$CKPT" --ckpt-format "$FMT" --data-root "$ROOT/data" --tokenizer "$TOK" --out "$OUT" ${EXTRA:-}
fi
echo "== exit $? ($(date +%T))"
