#!/bin/bash
# Qwen3-4B recipe study (2026-10-29) on Berkeley slurm (jsteinhardt), for when Beaker jupiter does not schedule:
# the SAME train_ctc_suite.py command launch_fast8k.py builds for Beaker (arm flags, geometry, seed, no final
# full checkpoint, expandable segments), with weka paths swapped for the node-local staged copies
# (stage_q3_s3_beaker.sh -> S3 -> /data/prasann/q3recipe). Then, unless NO_EVAL=1, the fast8k ladder eval of every
# mt4 task on the trained checkpoint (8k + 2k rungs, 500 rows, no-cot, query position after), as run_eval_lambda.sbatch.
#
#   sbatch -p jsteinhardt --qos=preemptive_high --account=site -w sneetches --gres=gpu:H200:2 \
#     --job-name=<run> --export=ALL,ARM=dense,BUDGET=f1952,LR=5e-6,CODE=/data/prasann/q3recipe/code_<sha> \
#     research/token_routing/ds64_fast8k/run_q3_local.sbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/q3recipe_%x_%j.log
set -uo pipefail
ROOT=/data/prasann/q3recipe
CODE="${CODE:?set CODE=/data/prasann/q3recipe/code_<sha> (staged with git archive | srun tar -x)}"
ARM="${ARM:?}"; BUDGET="${BUDGET:?}"; LR="${LR:?}"; SEED="${SEED:-0}"
# Q3_SIZE (size ablation): 4b = the original study (base staged at $ROOT/base, run names -q3-); other sizes read
# $ROOT/bases/qwen3-<size>-base-trainedmark and tag run names -q3<size without dot>- (e.g. -q317b-)
Q3_SIZE="${Q3_SIZE:-4b}"
if [ "$Q3_SIZE" = 4b ]; then BASE_DIR=$ROOT/base; SFX=-q3; else BASE_DIR=$ROOT/bases/qwen3-$Q3_SIZE-base-trainedmark; SFX=-q3${Q3_SIZE//./}; fi
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$CODE/src:$CODE/ctc/src" PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export HOME=$ROOT/home TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=$ROOT/triton_cache HF_HOME=$ROOT/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HOME $TMPDIR $TRITON_CACHE_DIR $HF_HOME $ROOT/ckpts
if [ -z "${WANDB_API_KEY:-}" ] && [ -r /accounts/projects/berkeleynlp/prasann/.netrc ]; then
  export WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' /accounts/projects/berkeleynlp/prasann/.netrc)
fi
NGPU=${SLURM_GPUS_ON_NODE:-2}
cd "$CODE"
# launch_fast8k's own arm -> (variant, data, args, extra) mapping, paths rewritten to the node-local copies
mapfile -t ARGS < <(F8K_FAMILY=qwen3 F8K_SUFFIX=$SFX F2K_SCALE=$Q3_SIZE F8K_NGPU=$NGPU BASE_DIR=$BASE_DIR python - "$ARM" "$BUDGET" "$LR" "$SEED" <<'PY'
import os, shlex, sys
arm, budget, lr, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, "research/token_routing/ds64_fast8k")
import launch_fast8k as L
L.LR_TAG = "" if (lr == L.DEFAULT_LR and not L.SUFFIX) else f"-lr{lr:g}"  # as launch_fast8k.main
variant, data, largs, extra = L.arm_args("mt4", arm, budget)
label = f"{arm}sd{seed}" if seed else arm
name = L.run_name("mt4", label, budget) + "-sn"  # -sn: Berkeley-slurm copy, never collides with the Beaker run
root = "/data/prasann/q3recipe"
data = f"{root}/{os.path.basename(data)}"
largs = [f"{os.environ['BASE_DIR']}/model_and_optim" if a == L.BASE else a for a in largs]
cmd = ["--task", "mt4", "--variant", variant, "--model-family", "qwen3", "--model-scale", L.SCALE, "--data", data,
       "--run-name", name, "--wandb-group", L.WANDB_GROUP, "--save-folder", f"{root}/ckpts/{name}",
       "--work-dir", f"{root}/work-{name}", "--epochs", str(L.budget_epochs(budget)), "--lr", str(lr),
       "--save-checkpoint", "--no-final-checkpoint", "--no-compile", "--seed", str(seed)] + largs + shlex.split(extra)
print(name)
for a in cmd:
    print(a)
PY
)
RUN=${ARGS[0]}; ARGS=("${ARGS[@]:1}")
[ -n "$RUN" ] || { echo "FATAL: arg build failed"; exit 2; }
echo "=== q3recipe host=$(hostname) RUN=$RUN NGPU=$NGPU CODE=$CODE $(date '+%F %T') ==="
echo "    train_ctc_suite.py ${ARGS[*]}"
[ -f $BASE_DIR/model_and_optim/.metadata ] || { echo "FATAL: base not staged at $BASE_DIR"; exit 3; }
# a Beaker twin of this run started first: touch $ROOT/skip/<run> and the chain moves on
[ -f $ROOT/skip/$RUN ] && { echo "=== SKIP $RUN (Beaker twin) ==="; exit 0; }
# trained = the end-of-fit flops.json + a complete model export (config.json alone is written at the START of fit, so
# a preempted/requeued run must train again; the trainer auto-resumes from its own save folder)
if [ ! -f $ROOT/ckpts/$RUN/flops.json ] || ! find $ROOT/ckpts/$RUN -name .metadata -path '*model_and_optim*' | grep -q . \
   || [ -n "${FORCE_TRAIN:-}" ]; then
  python -m torch.distributed.run --nproc_per_node=$NGPU --master_port=$((29000 + RANDOM % 1000)) \
    recipes/train/research_trainer/train_ctc_suite.py "${ARGS[@]}"
  rc=$?; echo "=== TRAIN rc=$rc $(date '+%F %T') ==="; [ $rc -eq 0 ] || exit $rc
fi
mkdir -p "$ROOT/weka_stub/checkpoints/prasanns/_eval_bundle" $ROOT/pending_evals
export LADDER_GEN_BATCH="${LADDER_GEN_BATCH:-32}" LADDER_GEN_BATCH_TOKENS="${LADDER_GEN_BATCH_TOKENS:-196608}"
E=$ROOT/eval
final=0
scored() {  # a result counts only with a nonzero eval_size on every rung (a failed ladder writes a stub JSON)
  python -c "import json,sys;n=json.load(open(sys.argv[1])).get('_meta',{}).get('eval_size') or {};sys.exit(0 if n and all(n.values()) else 1)" "$1" 2>/dev/null
}
eval_run() {  # eval_run <run> [tasks]: the fast8k ladder (8k + 2k, 500 rows, no-cot, query after) of each mt4 task
  local R=$1 RUN_DIR=$ROOT/ckpts/$1 CKPT TASK KEY R8 R2 FILES OUT_DIR rc
  CKPT=$(for d in $(find "$RUN_DIR" -name .metadata -path '*model_and_optim*' 2>/dev/null); do dirname "$(dirname "$d")"; done | sort -V | tail -1)
  [ -n "$CKPT" ] || { echo "FATAL: no complete model_and_optim under $RUN_DIR"; final=3; return; }
  for TASK in ${2:-contradiction nq outlier oolong}; do
    case $TASK in
      contradiction) KEY=contra; R8=_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl
                     R2=_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl ;;
      nq|outlier) KEY=$TASK; R8=outlier_lengthmix/eval_rungs/$TASK/rung_8192.jsonl; R2=outlier_lengthmix/eval_rungs/$TASK/rung_2048.jsonl ;;
      oolong) KEY=oolong; R8=_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl
              R2=_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl ;;
    esac
    OUT_DIR=$RUN_DIR/eval
    scored $OUT_DIR/${KEY}_multirung.json && { echo "[skip] $R $TASK already scored"; continue; }
    # built BEFORE the env prefix below, which rebinds TASK to the runner's key (contra): the ladder looks rung files
    # up by its own task name (contradiction)
    FILES="{\"$TASK\": {\"8k\": \"$E/$R8\", \"2k\": \"$E/$R2\"}}"
    echo "=== EVAL $R $TASK $(date '+%F %T') ==="
    RUN="$R" TASK="$KEY" VARIANT=docchunk STEP='' CKPT="$CKPT" EVAL_OUT_DIR="$OUT_DIR" \
      PROMPT_FORMAT=chat QUERY_POSITION=after MAX_TEST=500 MAX_LENGTH=20000 BATCH_SIZE=8 NGPU=$NGPU \
      LADDER_XLONG=0 COT_MODE=none LADDER_VERSION=v2 TOKENIZER=$ROOT/tok REQUIRE_EVAL_SIZE=1 \
      DC_RUNG_FILES="$FILES" DC_RUNGS=8k,2k \
      WEKA_LLM="$ROOT/weka_stub" RESULTS="$ROOT/eval_results" REPO="$CODE" \
      bash recipes/eval/legacy_native/run_beaker_multirung_eval.sh
    rc=$?; echo "=== EVAL $TASK rc=$rc ==="; [ $rc -eq 0 ] || final=$rc
  done
  mkdir -p /scratch/users/prasann/q3recipe_results/$R && cp $RUN_DIR/eval/*.json /scratch/users/prasann/q3recipe_results/$R/ 2>/dev/null
}
[ -n "${NO_EVAL:-}" ] || eval_run $RUN "${EVAL_TASKS:-}"
# pending_evals/<run>: re-score another finished run here (e.g. one whose eval failed); claimed with mv so two
# chains never take the same one
for f in $ROOT/pending_evals/*; do
  [ -e "$f" ] || continue
  mv "$f" "$f.claimed.$$" 2>/dev/null || continue
  echo "=== PENDING EVAL $(basename $f) ==="; eval_run "$(basename $f)"; rm -f "$f.claimed.$$"
done
echo "=== DONE rc=$final $(date '+%F %T') ==="
exit $final
