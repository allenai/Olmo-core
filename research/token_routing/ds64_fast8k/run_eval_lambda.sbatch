#!/bin/bash
#SBATCH --job-name=f8k-eval-lambda
#SBATCH --partition=lambda
#SBATCH --account=site
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/accounts/projects/sewonm/prasann/ctc_suite/logs/f8k_eval_%x_%j.log
# fast8k ladder eval ON LAMBDA (tokdrop_lr): the same on-node runner the Beaker evals use
# (recipes/eval/legacy_native/run_beaker_multirung_eval.sh, docchunk ladder, fast8k's arguments), pointed at
# node-local /tmp checkpoints and eval files. Submit with explicit gres and node, e.g.
#   sbatch --gres=gpu:A100:2 --qos=preemptive_high -w lambda-hyperplane02 --job-name=<run>-ev<task> \
#     --export=ALL,RUN=<run>,TASK=<contradiction|nq|outlier|oolong>,ROOT=/tmp/prasann/tdlr run_eval_lambda.sbatch
# (mt11 runs: EVAL_ROWS=suite QUERY_POSITION=both, automatic for RUN=*mt11*; see below)
# CTC-suite tasks (TASK=hotpotqa|qdmatch_nq|xabsence|reorder|strmatch|grouping, 2026-10-21) read the
# suite's own r2k/r8k rows from $ROOT/eval/suite/<subset>/rung_{2048,8192}.jsonl (fetched with
# fetch_suite_eval_rows.py, staged to node /tmp) and are graded by the vendored ctc spec exactly as
# olmo-eval grades the suite row (eval_lc_native_docchunk_ladder.py SUITE_TASKS).
# EVAL_ROWS=suite (2026-10-22; the default for RUN names containing "mt11"): the mt11 mix trains on setA
# rows, which were built IID with the CTC-suite rows, so contradiction/nq/outlier/oolong read the suite's
# rows too ($ROOT/eval/suite/{contradiction_iid,nq,outlier,oolong}/; contradiction's 2k rung is
# rung_2560) and are graded by the ctc spec olmo-eval uses (ladder --ctc-grade). EVAL_ROWS=old = the
# ds64 files below, which the mt4 runs (ds64 fast8k pools) are IID with: the default for every other RUN.
# QUERY_POSITION must match the TRAINING SHARDS: mt4 = after, mt11/setA = both (metadata.json
# "query_position"); it defaults to both under EVAL_ROWS=suite, else after. With after, an mt11 model's
# prompts are out of format for nq/oolong/contradiction/hotpotqa/xabsence/reorder/strmatch
# (validate_suite_ladder_tasks.py --train-query-position both: prefix parity 0/N).
# Results: $ROOT/ckpts/<run>/eval/<key>_multirung.json (+ predictions), copied to $LROOT/eval_results/.
set -o pipefail   # no -u: it kills the conda-pack activate
LROOT=/accounts/projects/sewonm/prasann/ctc_suite
REPO=${REPO_OVERRIDE:-$LROOT/OLMo-core}  # REPO_OVERRIDE: evaluate with another checkout (e.g. newer eval hooks)
ROOT="${ROOT:-/tmp/prasann/tdlr}"
RUN="${RUN:?set RUN}"; TASK="${TASK:?set TASK}"
NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-2}}"
case "$RUN" in *mt11*) EVAL_ROWS="${EVAL_ROWS:-suite}" ;; *) EVAL_ROWS="${EVAL_ROWS:-old}" ;; esac
[ "$EVAL_ROWS" = suite ] && QUERY_POSITION="${QUERY_POSITION:-both}" || QUERY_POSITION="${QUERY_POSITION:-after}"
case "$EVAL_ROWS:$TASK" in
  suite:contradiction) KEY=contra; R8=suite/contradiction_iid/rung_8192.jsonl; R2=suite/contradiction_iid/rung_2560.jsonl ;;
  suite:nq|suite:outlier|suite:oolong) KEY=$TASK; R8=suite/$TASK/rung_8192.jsonl; R2=suite/$TASK/rung_2048.jsonl ;;
  old:*|suite:*) KEY= ;;
  *) echo "unknown EVAL_ROWS=$EVAL_ROWS (old|suite)"; exit 2 ;;
esac
if [ -n "$KEY" ]; then export LADDER_CTC_GRADE="$TASK"; else case "$TASK" in
  contradiction) KEY=contra; R8=_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl
                 R2=_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl ;;
  nq)      KEY=nq;      R8=outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl; R2=outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl ;;
  outlier) KEY=outlier; R8=outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl; R2=outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl ;;
  oolong)  KEY=oolong;  R8=_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl
           R2=_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl ;;
  hotpotqa|qdmatch_nq|xabsence|reorder|strmatch|grouping)
           KEY=$TASK;   R8=suite/$TASK/rung_8192.jsonl; R2=suite/$TASK/rung_2048.jsonl ;;
  *) echo "unknown TASK=$TASK"; exit 2 ;;
esac; fi
RUN_DIR=$ROOT/ckpts/$RUN
CKPT=$(for d in $(find "$RUN_DIR" -name .metadata -path '*model_and_optim*' 2>/dev/null); do dirname "$(dirname "$d")"; done | sort -V | tail -1)
[ -n "$CKPT" ] || { echo "FATAL: no complete model_and_optim under $RUN_DIR"; exit 3; }
ENV=/accounts/projects/sewonm/prasann/olmo_test/env
source "$ENV/bin/activate"; conda-unpack 2>/dev/null || true
export PATH="$ENV/bin:$PATH"   # torchrun from the node-visible env, never a system one
export HF_HOME="${HF_HOME_OVERRIDE:-/accounts/projects/sewonm/prasann/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache; mkdir -p "$TRITON_CACHE_DIR"
# the runner cd's into $WEKA_LLM/checkpoints/prasanns/_eval_bundle (rung files below are absolute)
mkdir -p "$ROOT/weka_stub/checkpoints/prasanns/_eval_bundle"
# scipy/scikit-learn (outlier/rerank metrics) were installed offline from wheels on 2026-10-20;
# lambda has no pypi, so the runner's own pip fallback cannot fetch them
"$ENV/bin/python" -c "import scipy, sklearn" || { echo "FATAL: scipy/sklearn missing from $ENV"; exit 4; }
# Batched greedy decode (eval_lc_native_docchunk_ladder.py --gen-batch, 2026-10-22): validated against
# the bs=1 loop on contradiction@8k (f1 0.641 vs 0.639 over 500 rows, 22/24 identical generations,
# 2.6x faster at 98k tokens/batch). LADDER_GEN_BATCH=1 restores the bs=1 loop.
export LADDER_GEN_BATCH="${LADDER_GEN_BATCH:-32}" LADDER_GEN_BATCH_TOKENS="${LADDER_GEN_BATCH_TOKENS:-196608}"
FILES="{\"$TASK\": {\"8k\": \"$ROOT/eval/$R8\", \"2k\": \"$ROOT/eval/$R2\"}}"
echo "=== LAMBDA fast8k eval host=$(hostname) RUN=$RUN TASK=$TASK CKPT=$CKPT NGPU=$NGPU $(date '+%F %T') ==="
echo "    EVAL_ROWS=$EVAL_ROWS QUERY_POSITION=$QUERY_POSITION LADDER_CTC_GRADE=${LADDER_CTC_GRADE:-} files: $FILES"
cd "$REPO"
RUN="$RUN" TASK="$KEY" VARIANT=docchunk STEP='' CKPT="$CKPT" EVAL_OUT_DIR="$RUN_DIR/eval" \
  PROMPT_FORMAT=chat QUERY_POSITION="$QUERY_POSITION" MAX_TEST=500 MAX_LENGTH=20000 BATCH_SIZE=8 NGPU="$NGPU" \
  LADDER_XLONG=0 COT_MODE=none LADDER_VERSION=v2 TOKENIZER=Qwen/Qwen3.5-0.8B-Base \
  DC_RUNG_FILES="$FILES" DC_RUNGS=8k,2k WEKA_LLM="$ROOT/weka_stub" RESULTS="$ROOT/eval_results" REPO="$REPO" \
  bash recipes/eval/legacy_native/run_beaker_multirung_eval.sh
rc=$?
# a ladder whose rung files were not found writes an EMPTY result with rc=0 (2026-10-20: the
# override was keyed "contra" while the ladder looks up "contradiction"); fail loudly instead
"$ENV/bin/python" - "$RUN_DIR/eval/${KEY}_multirung.json" <<'PY' || rc=5
import json, sys
d = json.load(open(sys.argv[1]))
n = d.get("_meta", {}).get("eval_size") or {}
print("eval_size per rung:", n)
sys.exit(0 if n and all(v for v in n.values()) else 1)
PY
mkdir -p "$LROOT/eval_results/$RUN" && cp "$RUN_DIR"/eval/*.json "$RUN_DIR"/eval/*_responses.jsonl "$LROOT/eval_results/$RUN/" 2>/dev/null
echo "=== DONE rc=$rc $(date '+%F %T') ==="
exit $rc
