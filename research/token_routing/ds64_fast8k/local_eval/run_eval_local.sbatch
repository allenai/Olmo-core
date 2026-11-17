#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/prasann/joblogs/%x-%j.out
# fast8k 4-task eval of ONE checkpoint on Berkeley slurm: the eval the Beaker `ev-*` jobs of
# `launch_fast8k.py eval` run (contradiction / nq / outlier / oolong x 8k + 2k, 500 rows per cell, query position
# after, no-cot, docchunk ladder, bs=1 greedy decode, same rung files, prompts and graders), through the SAME
# on-node runner (recipes/eval/legacy_native/run_beaker_multirung_eval.sh), one runner call per (task, rung) cell
# (8k first). A preempted job is requeued by slurm and skips the cells it already finished. Submit with
# submit_eval_local.py (resolves the commit, checks the node, registers the job); see README.md.
#
# Env in (submit_eval_local.py sets them):
#   RUN       training run name (results are filed under train_run=$RUN)
#   CKPT      node-local dir holding config.json + model_and_optim/ (a train_ctc_suite export), or
#   CKPT_S3   key prefix under s3://ai2-llm/ holding model_and_optim/ (staged to $ROOT/ckpts/$RUN, ETag-checked)
#   SRC, CODE_REF  checkout + commit whose eval code runs (git archive'd once per commit to $ROOT/code/<sha>)
#   TOOLS     this folder (cells.py, run_cells.sh; copied to the node at start, independent of CODE_REF)
#   DENSE_CONFIG=qwen3_5_4B  write a plain Qwen3.5-4B config.json when the checkpoint has none (model-only
#             resaves, e.g. grad_concentration's LoRA-merged copies); the loader still refuses a dir with LoRA keys
#   TASKS (default contradiction,nq,outlier,oolong), RUNGS (8k,2k), NGPU (default: the job's GPUs)
#   OPS_EXPERIMENT (tokdrop_lr), OPS_RUN_ID (default $RUN-evslurm-$SLURM_JOB_ID), RECORD=0 skips the ops rows
#   LADDER_GEN_BATCH  default 1 = Beaker's bs=1 decode; >1 is the batched decode run_eval_lambda.sbatch uses
# Out: $ROOT/runs/$RUN/eval/<key>_multirung.json + per-rung responses (the Beaker layout), cells/<task>_<rung>/,
# and ops result rows (record_beaker_evals.py's fields) under $OPS_EXPERIMENT.
set -o pipefail
ROOT=/data/prasann/f8k_eval
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
OPS=/accounts/projects/berkeleynlp/prasann/bin/ops
RUN="${RUN:?set RUN}"; SRC="${SRC:?set SRC}"; CODE_REF="${CODE_REF:?set CODE_REF}"; TOOLS_SRC="${TOOLS:?set TOOLS}"
export TASKS="${TASKS:-contradiction,nq,outlier,oolong}" RUNGS="${RUNGS:-8k,2k}"
export NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-2}}"
export OPS_EXPERIMENT="${OPS_EXPERIMENT:-tokdrop_lr}"
export RUN_ID="${OPS_RUN_ID:-$RUN-evslurm-${SLURM_JOB_ID:-$$}}"
export WORK=$ROOT/runs/$RUN ROOT RUN PY
echo "=== f8k local eval host=$(hostname -s) job=${SLURM_JOB_ID:-} restarts=${SLURM_RESTART_COUNT:-0} RUN=$RUN" \
     "ref=$CODE_REF NGPU=$NGPU qos=${SLURM_JOB_QOS:-} $(date '+%F %T') ==="
[ -x "$PY" ] || { echo "FATAL: node-local env $PY missing on $(hostname -s)"; exit 4; }
mkdir -p "$ROOT/code" "$ROOT/tmp/triton" "$ROOT/ckpts" "$ROOT/eval" "$ROOT/results" "$WORK/cells" "$WORK/tools"
cd "$ROOT"   # job I/O on /data, never the submit cwd (an NFS checkout)
export TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=$ROOT/tmp/triton HF_HOME=/data/prasann/hf_cache
export PATH="$(dirname "$PY"):$PATH"   # the runner's torchrun / python resolve to the node-local env
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_MODE=offline
cp "$TOOLS_SRC/cells.py" "$TOOLS_SRC/run_cells.sh" "$TOOLS_SRC/../../olmo3_cptdrop/s3_sync.py" "$WORK/tools/" \
  || { echo "FATAL: cannot copy tools from $TOOLS_SRC"; exit 2; }
export TOOLS=$WORK/tools
H() { "$PY" "$TOOLS/cells.py" "$@"; }

# ---- code: the caller's commit, read once from the NFS checkout's git objects, unpacked on /data ----
SHA=$(git -C "$SRC" rev-parse --verify "$CODE_REF^{commit}") || { echo "FATAL: $CODE_REF not in $SRC"; exit 2; }
export CODE=$ROOT/code/$SHA
(
  flock 9
  if [ ! -f "$CODE/.complete" ]; then
    rm -rf "$CODE.tmp" && mkdir -p "$CODE.tmp" && git -C "$SRC" archive "$SHA" | tar -x -C "$CODE.tmp" \
      && touch "$CODE.tmp/.complete" && rm -rf "$CODE" && mv "$CODE.tmp" "$CODE" || exit 2
  fi
) 9>"$ROOT/code.lock" || { echo "FATAL: code staging failed"; exit 2; }
# finished cells are reused only when they were scored by the same commit; FRESH=1 (first start only) rescoring all
if [ "${FRESH:-0}" = 1 ] && [ "${SLURM_RESTART_COUNT:-0}" = 0 ]; then rm -rf "$WORK/cells" "$WORK/eval"; mkdir -p "$WORK/cells"; fi
if [ -n "$(ls -A "$WORK/cells" 2>/dev/null)" ] && [ "$(cat "$WORK/code_sha" 2>/dev/null)" != "$SHA" ]; then
  echo "FATAL: $WORK/cells were scored by commit $(cat "$WORK/code_sha" 2>/dev/null || echo '?'), not $SHA;" \
       "resubmit with --fresh to rescore every cell"; exit 2
fi
echo "$SHA" > "$WORK/code_sha"

# ---- eval rows (S3 hub, md5-checked) and the checkpoint ----
( flock 9; H data --root "$ROOT/eval" ) 9>"$ROOT/eval.lock" || { echo "FATAL: eval data staging failed"; exit 3; }
if [ -n "${CKPT_S3:-}" ]; then
  CKPT=$ROOT/ckpts/$RUN
  P="${CKPT_S3%/}"
  ( flock 9; [ -f "$CKPT/model_and_optim/.metadata" ] || "$PY" -u -c "
import sys; sys.path.insert(0, '$TOOLS'); import s3_sync as s
s.ROOT = '${P%/*}'; sys.argv = ['s3_sync', 'down', '$CKPT', '${P##*/}']; s.main()" ) 9>"$ROOT/ckpts/$RUN.lock"
fi
export CKPT="${CKPT:?set CKPT (node-local dir) or CKPT_S3}"; CKPT="${CKPT%/}"
[ -f "$CKPT/model_and_optim/.metadata" ] || { echo "FATAL: no model_and_optim/.metadata under $CKPT"; ls -la "$CKPT"; exit 3; }
if [ ! -f "$CKPT/config.json" ]; then
  [ "${DENSE_CONFIG:-}" = qwen3_5_4B ] || {
    echo "FATAL: $CKPT has no config.json (a model-only resave? pass DENSE_CONFIG=qwen3_5_4B)"; exit 3; }
  PYTHONPATH="$CODE/src" "$PY" - "$CKPT/config.json" <<'PY' || { echo "FATAL: could not write config.json"; exit 3; }
import json, sys
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
# the model block every fast8k Qwen3.5-4B export records (train_ctc_suite: vocab 248320, flash_2 pinned)
cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320, attn_backend=AttentionBackendName("flash_2"))
json.dump({"model": cfg.as_config_dict(), "model_family": "qwen3_5",
           "_written_by": "run_eval_local.sbatch DENSE_CONFIG=qwen3_5_4B (model-only resave had no config.json)"},
          open(sys.argv[1], "w"))
print("wrote", sys.argv[1])
PY
fi

# ---- env check: fail fast rather than score a broken import as 0.000 ----
PYTHONPATH="$CODE/recipes/eval/legacy_native:$CODE/src" "$PY" - <<'PY' || { echo "FATAL: env check failed"; exit 4; }
import importlib
import olmo_core, torch
for m in ("flash_attn", "fla", "scipy", "sklearn", "transformers"):
    importlib.import_module(m)
print("olmo_core from", olmo_core.__file__, "| torch", torch.__version__, "| gpus", torch.cuda.device_count())
PY
mkdir -p "$ROOT/weka_stub/checkpoints/prasanns/_eval_bundle"   # the runner cd's to $WEKA_LLM/.../_eval_bundle
echo "cells left: $(H plan --work "$WORK" --tasks "$TASKS" --rungs "$RUNGS" | wc -l) | ckpt $CKPT | code $CODE"

OPS_RUN_ID="$RUN_ID" OPS_KIND=eval OPS_PARENT_RUN="$RUN" OPS_TASK=mt4 OPS_EVAL_VARIANT=f8k4-docchunk \
  OPS_CKPT_PATH="$CKPT" OPS_EXPECTED_MINUTES="${EXPECTED_MINUTES:-90}" OPS_FIRST_OUTPUT=1200 OPS_STALL=2400 \
  OPS_NOTES="fast8k 4-task local eval (run_eval_local.sbatch), code $SHA" \
  bash "$SRC/recipes/env/ops_job.sh" bash "$TOOLS/run_cells.sh"
rc=$?
LEFT=$(H plan --work "$WORK" --tasks "$TASKS" --rungs "$RUNGS" | wc -l)
H merge --work "$WORK" --tasks "$TASKS" --rungs "$RUNGS"
for f in "$WORK"/eval/*_multirung.json; do [ -f "$f" ] && cp "$f" "$ROOT/results/${RUN}_$(basename "$f")"; done
if [ "${RECORD:-1}" = 1 ]; then
  H rows --work "$WORK" --tasks "$TASKS" --rungs "$RUNGS" --run "$RUN" --run-id "$RUN_ID" \
    --job-id "${SLURM_JOB_ID:-local}" --experiment "$OPS_EXPERIMENT" --ckpt "$CKPT" --node "$(hostname -s)" \
    --git-commit "$SHA" --out "$WORK/ops_rows.jsonl"
  # file only rows this pass has not filed yet (a requeued job adds just its new cells)
  timeout 120 python3 "$OPS" result show --json > "$WORK/ops_have.json" 2>/dev/null
  python3 - "$WORK/ops_rows.jsonl" "$WORK/ops_have.json" "$WORK/ops_new.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
try:
    have = {(r.get("run_id"), r.get("task"), str(r.get("rung")), r.get("metric")) for r in json.load(open(sys.argv[2]))}
except Exception:
    have = set()
new = [r for r in rows if (r["run_id"], r["task"], r["rung"], r["metric"]) not in have]
open(sys.argv[3], "w").write("".join(json.dumps(r) + "\n" for r in new))
print(f"[ops] {len(new)} new result rows ({len(rows) - len(new)} already recorded)")
PY
  [ -s "$WORK/ops_new.jsonl" ] && timeout 120 python3 "$OPS" result add --json "$WORK/ops_new.jsonl" --check 2>&1 | tail -12
fi
echo "=== DONE rc=$rc cells_left=$LEFT results=$WORK/eval $(date '+%F %T') ==="
[ "$LEFT" -eq 0 ] && exit 0
exit $(( rc ? rc : 6 ))
