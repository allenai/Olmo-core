#!/bin/bash
# The cell loop of run_eval_local.sbatch (run under recipes/env/ops_job.sh). One call of the Beaker on-node runner
# per missing (task, rung) cell, 8k first; each cell writes its own result JSON + responses + log under
# $WORK/cells/<task>_<rung>/. Env: PY TOOLS WORK CODE CKPT RUN RUN_ID NGPU ROOT TASKS RUNGS [LADDER_GEN_BATCH].
set -o pipefail
failed=0
while read -r TASK RUNG KEY FILE; do
  [ -n "$TASK" ] || continue
  CELL=$WORK/cells/${TASK}_${RUNG}; rm -rf "$CELL"; mkdir -p "$CELL"
  echo "--- cell $TASK@$RUNG ($(date '+%T')) $FILE"
  # Built HERE, not inline below: in the runner's env prefix TASK is reassigned to the runner key ("contra"),
  # and bash expands later assignment words with the NEW value, so an inline {"$TASK": ...} keyed the override
  # "contra" while the ladder looks up "contradiction" -> empty result (2026-11-10; Lambda hit the same 10-20).
  DCF="{\"$TASK\": {\"$RUNG\": \"$FILE\"}}"
  # Everything below is what run_q4b_beaker_multirung_eval.py hands the runner for a launch_fast8k mt4 eval
  # (eval_cmd: --variant docchunk --query-position after --cot-mode none --max-test 500 --max-length 20000
  # --batch-size 8, --dc-rung-files/--dc-rungs, ladder v2 default, prompt format chat), one rung at a time.
  ( cd "$CODE" && RUN="$RUN-ev$TASK" TASK="$KEY" VARIANT=docchunk STEP='' CKPT="$CKPT" EVAL_OUT_DIR="$CELL" \
    PROMPT_FORMAT=chat QUERY_POSITION=after MAX_TEST=500 MAX_LENGTH=20000 BATCH_SIZE=8 NGPU="$NGPU" \
    LADDER_XLONG=0 COT_MODE=none LADDER_VERSION=v2 TOKENIZER=Qwen/Qwen3.5-0.8B-Base REQUIRE_EVAL_SIZE=1 \
    LADDER_GEN_BATCH="${LADDER_GEN_BATCH:-1}" \
    DC_RUNG_FILES="$DCF" DC_RUNGS="$RUNG" \
    WEKA_LLM="$ROOT/weka_stub" RESULTS="$CELL" REPO="$CODE" OPS_EVAL_RUN_ID="$RUN_ID" \
    bash recipes/eval/legacy_native/run_beaker_multirung_eval.sh ) > "$CELL/cell.log" 2>&1
  rc=$?
  echo "$RUN_ID ${SLURM_JOB_ID:-local}" > "$CELL/scored_by"   # rows are filed under the pass that scored the cell
  grep -E "^\[ladder:|\[ids\]|early-lora|built from|ERROR|Traceback|FATAL" "$CELL/cell.log" | tail -8
  echo "--- cell $TASK@$RUNG rc=$rc ($(date '+%T'))"
  [ $rc -eq 0 ] || failed=$((failed + 1))
done < <("$PY" "$TOOLS/cells.py" plan --work "$WORK" --root "$ROOT/eval" --tasks "$TASKS" --rungs "$RUNGS")
exit $failed
