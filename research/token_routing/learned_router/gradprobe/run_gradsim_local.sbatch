#!/usr/bin/env bash
# Full-context vs token-dropped parameter-gradient similarity (grad_similarity.py, or SCRIPT=grad_tokens.py) on one GPU, from a code tree staged on
# node-local /data (same staging as run_gradprobe_local.sbatch). One model, several tasks per job.
#   CODE=/data/prasann/gradprobe/<tree> CKPT=/data/prasann/devloss_grid/ckpts/setA32k TASKS=nq,outlier,contradiction ROWS=32 \
#     OUT_PREFIX=/data/prasann/gradprobe/out/gsim_setA32k \
#     sbatch --partition=jsteinhardt --qos=preemptive_high --nodelist=mooney --gres=gpu:H200:1 .../run_gradsim_local.sbatch
#SBATCH --job-name=gradsim
#SBATCH --partition=berkeleynlp
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/data/prasann/joblogs/gradsim_%j.log

set -uo pipefail
: "${CODE:?set CODE}" "${CKPT:?set CKPT}" "${OUT_PREFIX:?set OUT_PREFIX}"
TASKS=${TASKS:-nq,outlier,contradiction}; RUNG=${RUNG:-8k}; ROWS=${ROWS:-2}
case "$CODE$OUT_PREFIX" in /data/*/data/*) ;; *) echo "!!! CODE and OUT_PREFIX must be on node-local /data"; exit 1;; esac
cd "$CODE"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$CODE/src:$CODE/research/common" PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HOME=/data/prasann HF_HOME=/data/prasann/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore DEVLOSS_SKIP_CHECKS=1
export TMPDIR=/data/prasann/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache XDG_CACHE_HOME=/data/prasann/xdg_cache TORCHINDUCTOR_CACHE_DIR=/data/prasann/inductor_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$(dirname "$OUT_PREFIX")"
[ -d /usr/local/cuda-12.8 ] && { export CUDA_HOME=/usr/local/cuda-12.8; export PATH=$CUDA_HOME/bin:$PATH; }
export PROBE_FAMILY=${PROBE_FAMILY:-qwen3_5}  # qwen3_5 (GDN hybrid) | qwen3 (pure attention)
# qwen3: the "nothink" copy (generation prompt ends with the empty think block, so the answer loss mask matches the
# Qwen3.5 layout; the stock template puts <think>\n\n</think>\n\n inside the answer span)
if [ -n "${TOK:-}" ]; then :; elif [ "$PROBE_FAMILY" = qwen3 ]; then TOK=/data/prasann/hf_tokenizers/Qwen3-4B-nothink
else TOK=/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68; fi
SCRIPT=${SCRIPT:-grad_similarity.py}  # or grad_tokens.py (per-token / per-layer decomposition; same arguments)
if [ -f "$CKPT/.text_export_done" ]; then FMT=hf; elif [ -f "$CKPT/model_and_optim/.metadata" ]; then FMT=distcp
elif [ -f "$CKPT/.resave_done" ]; then FMT=olmo_st; else echo "!!! no staged checkpoint at $CKPT"; exit 1; fi
echo "== gradsim[$SCRIPT, $PROBE_FAMILY] $TASKS@$RUNG rows=$ROWS ckpt=$CKPT ($FMT) on $(hostname -s) job ${SLURM_JOB_ID:-?} ($(date +%T)) code=$CODE"
"$PY" -u research/token_routing/learned_router/gradprobe/$SCRIPT --tasks "$TASKS" --rung "$RUNG" --rows "$ROWS" \
  --ckpt "$CKPT" --ckpt-format "$FMT" --data-root /data/prasann/devloss_grid/data --tokenizer "$TOK" --out-prefix "$OUT_PREFIX" ${EXTRA:-}
echo "== exit $? ($(date +%T))"
