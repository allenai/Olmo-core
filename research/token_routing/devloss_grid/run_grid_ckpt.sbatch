#!/usr/bin/env bash
# Dev-loss grid for ONE checkpoint over a task list (e.g. a multi-task SFT model such as setA), any family, from a
# code tree staged on node-local /data (git ls-files | tar on the login node, srun tar -x). One driver process per
# task (all rungs in it, model loaded once per task) -> $RES_DIR/<task>_{rung}.json; resumable (existing JSONs skip).
#
#   CODE=/data/prasann/gradprobe/<tree> CKPT=/data/prasann/devloss_grid/ckpts/q3setA PROBE_FAMILY=qwen3 \
#     TOK=/data/prasann/devloss_grid/ckpts/q3setA_tokenizer RES_DIR=/data/prasann/devloss_grid/results_q3setA \
#     sbatch --partition=jsteinhardt --qos=preemptive_high --nodelist=mooney --gres=gpu:H200:1 run_grid_ckpt.sbatch
#
# Env: TASKS (roster keys, default the 11 setA tasks), RUNGS (default 2k,8k,32k), ROWS (aligned, default 48,48,32),
# SCHEMES (default: the artifact's top heuristic candidates + references), PROBE_FAMILY (qwen3_5 | qwen3), TOK.
#SBATCH --job-name=grid_ckpt
#SBATCH --partition=berkeleynlp
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/grid_ckpt_%j.log

set -uo pipefail
: "${CODE:?set CODE}" "${CKPT:?set CKPT}" "${RES_DIR:?set RES_DIR}"
case "$CODE$RES_DIR" in /data/*/data/*) ;; *) echo "!!! CODE and RES_DIR must be on node-local /data"; exit 1;; esac
TASKS=${TASKS:-ctc_nq,ctc_hpqa,ctc_qdmatch_nq,ctc_outlier,ctc_oolong,ctc_contradiction,ctc_xabsence,ctc_reorder,ctc_rerank,ctc_strmatch,ctc_grouping}
RUNGS=${RUNGS:-2k,8k,32k}; ROWS=${ROWS:-48,48,32}
SCHEMES=${SCHEMES:-full,gold_fl20p8,gold_fl20p8_noslot,gold_rand20p8_noslot,gold_first20,gold_fl64,gold_only_noslot,first128,fl20p8_noslot,rand20,k0}
export PROBE_FAMILY=${PROBE_FAMILY:-qwen3_5}
TOK=${TOK:-/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68}
cd "$CODE"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$CODE/src:$CODE/research/common" PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HOME=/data/prasann HF_HOME=/data/prasann/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore DEVLOSS_SKIP_CHECKS=1
export TMPDIR=/data/prasann/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache XDG_CACHE_HOME=/data/prasann/xdg_cache TORCHINDUCTOR_CACHE_DIR=/data/prasann/inductor_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$RES_DIR"
[ -d /usr/local/cuda-12.8 ] && { export CUDA_HOME=/usr/local/cuda-12.8; export PATH=$CUDA_HOME/bin:$PATH; }
if [ -f "$CKPT/.text_export_done" ]; then FMT=hf; elif [ -f "$CKPT/model_and_optim/.metadata" ]; then FMT=distcp
elif [ -f "$CKPT/.resave_done" ]; then FMT=olmo_st; else echo "!!! no staged checkpoint at $CKPT"; exit 1; fi
echo "== grid $PROBE_FAMILY ckpt=$CKPT ($FMT) tok=$TOK on $(hostname -s) job ${SLURM_JOB_ID:-?} ($(date +%T)) tasks=$TASKS rungs=$RUNGS rows=$ROWS"
IFS=, read -ra TL <<< "$TASKS"
i=0
for T in "${TL[@]}"; do
  i=$((i + 1))
  # rungs whose JSON already exists are skipped (resume); contradiction's 2k rung is rung_2560 (driver maps it)
  todo_r=(); todo_n=(); IFS=, read -ra RL <<< "$RUNGS"; IFS=, read -ra NL <<< "$ROWS"
  for k in "${!RL[@]}"; do [ -f "$RES_DIR/${T}_${RL[$k]}.json" ] || { todo_r+=("${RL[$k]}"); todo_n+=("${NL[$k]}"); }; done
  [ ${#todo_r[@]} -eq 0 ] && { echo ".. [$i/${#TL[@]}] $T: done already"; continue; }
  R=$(IFS=,; echo "${todo_r[*]}"); N=$(IFS=,; echo "${todo_n[*]}")
  echo ".. [$i/${#TL[@]}] $T rungs $R rows $N ($(date +%T))"
  "$PY" -u research/token_routing/devloss_grid/ctc_devloss_grid.py --task "$T" --rung "$R" --rows "$N" --ckpt "$CKPT" \
    --ckpt-format "$FMT" --data-root /data/prasann/devloss_grid/data --tokenizer "$TOK" --schemes "$SCHEMES" \
    --out "$RES_DIR/${T}_{rung}.json" ${EXTRA:-} 2>&1 | grep -v Warning
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || echo "!!! $T exited $rc"
done
echo "== exit 0 ($(date +%T))"
