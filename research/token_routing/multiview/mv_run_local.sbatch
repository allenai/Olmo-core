#!/bin/bash
# Multi-view run, LOCAL (jsteinhardt) path: one job = one fast8k run of the random-choice recipe with --st-otr-views K
# on 2 GPUs of one H200 node, then its 8 eval cells (contradiction / nq / outlier / oolong x 8k / 2k, 500 rows, no-cot,
# query after) two at a time, through the Beaker eval runner. Adapted from ds64_fast8k/scale_run_local.sbatch (branch
# claude/recipe-scale-ablation-71d007): same trainer geometry as the Beaker fresh-pool runs (unpacked, seq 12288, 16
# rows/step, micro 2, full AC, LR 1e-5). The 4B repaired base is pulled once from S3 (_transfer/q35-4b-base-markerfix).
# MAX_STEPS > 0 = smoke (stop after N steps, no evals).
#   sbatch -p jsteinhardt --account=site --qos=preemptive_high -w sneetches --gres=gpu:2 -c 16 --mem=200G -t 16:00:00 \
#     --job-name=<run> --export=ALL,CODE=/data/prasann/scale_abl/code_<sha>,VIEWS=2,ROWS=1952 \
#     research/token_routing/multiview/mv_run_local.sbatch
#SBATCH --output=/data/prasann/scale_abl/logs/%x_%j.log
set -uo pipefail
VIEWS="${VIEWS:?}"; ROWS="${ROWS:?}"; LR="${LR:-1e-5}"; CODE="${CODE:?}"; MAX_STEPS="${MAX_STEPS:-0}"
R=/data/prasann/scale_abl; A="${ASSETS:-$R/assets}"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' /accounts/projects/berkeleynlp/prasann/.netrc)
[ -n "$WANDB_API_KEY" ] || { echo "!!! no WANDB_API_KEY"; exit 4; }
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONPATH=$CODE/src
export AWS_SHARED_CREDENTIALS_FILE=/accounts/projects/berkeleynlp/prasann/.aws/credentials AWS_CONFIG_FILE=/accounts/projects/berkeleynlp/prasann/.aws/config
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
mkdir -p $TMPDIR $TRITON_CACHE_DIR $R/shards $R/runs $R/work $R/out $R/bases $R/logs
cd $CODE

s3pull() {  # s3pull <prefix under checkpoints/prasanns/_transfer/> <dst dir>; one downloader per node (mkdir lock)
  local pre=$1 dst=$2
  [ -f $dst/.complete ] && return 0
  if mkdir $dst.lock 2>/dev/null; then
    $PY - $pre $dst <<'EOF' || { rmdir $dst.lock; return 1; }
import os, sys, boto3
pre, dst = "checkpoints/prasanns/_transfer/" + sys.argv[1], sys.argv[2]
c = boto3.client("s3"); n = 0
for page in c.get_paginator("list_objects_v2").paginate(Bucket="ai2-llm", Prefix=pre):
    for o in page.get("Contents", []):
        p = os.path.join(dst, o["Key"][len(pre):]); os.makedirs(os.path.dirname(p), exist_ok=True)
        c.download_file("ai2-llm", o["Key"], p); n += 1
assert n, f"nothing at s3://ai2-llm/{pre}"
open(os.path.join(dst, ".complete"), "w").write(str(n)); print(f"[s3] {n} files -> {dst}", flush=True)
EOF
    rmdir $dst.lock
  else
    while [ ! -f $dst/.complete ]; do sleep 20; done
  fi
}

BASED=$R/bases/q35-4b-base-markerfix
s3pull q35-4b-base-markerfix/ $BASED || { echo "!!! base download failed"; exit 1; }
BASE=$BASED/model_and_optim
D=$R/shards/mt4f$ROWS
s3pull tokdrop_lr/shards/mt4f$ROWS/ $D || { echo "!!! shard download failed"; exit 1; }

X="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32 --st-otr-score random --attn-backend flash_2 --st-otr-views $VIEWS"
LRN=$($PY -c "print(f'{float(\"$LR\"):g}')")
RUN=f8k-mt4-lora64all32rnd20to7mv$VIEWS-f$ROWS-tdmvl-lr$LRN
STEPS=""
if [ "$MAX_STEPS" -gt 0 ]; then RUN=mvsmoke-$VIEWS-$SLURM_JOB_ID; STEPS="--max-steps $MAX_STEPS"; fi
echo "[$(date +%T)] run=$RUN node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"

SAVE=$R/runs/$RUN
if [ ! -f $SAVE/config.json ]; then
  echo "[$(date +%T)] train start"
  $PY -m torch.distributed.run --standalone --nproc_per_node=2 recipes/train/research_trainer/train_ctc_suite.py \
    --task mt4 --data $D --variant softtoken --model-family qwen3_5 --model-scale 4b \
    --seq-len 12288 --epochs 1 --lr $LR --global-batch 16 --micro-batch-instances 2 --base-checkpoint $BASE \
    --work-dir $R/work/$RUN --save-folder $SAVE --run-name $RUN --wandb-group f8k-q35-4b-local --no-compile \
    --activation-checkpointing full --save-checkpoint --no-final-checkpoint $STEPS $X > $R/logs/train_$RUN.log 2>&1
  rc=$?
  grep -aE "multiview\]|flop-meter\] wrote|Traceback|Error" $R/logs/train_$RUN.log | grep -av UserWarning | tail -8 | cut -c1-600
  grep -aE "online-router\] step" $R/logs/train_$RUN.log | tail -2 | cut -c1-300
  echo "[$(date +%T)] train rc=$rc"; [ $rc = 0 ] && [ -f $SAVE/config.json ] || { tail -30 $R/logs/train_$RUN.log; exit 1; }
fi
[ "$MAX_STEPS" -gt 0 ] && { echo SMOKE_DONE; exit 0; }

declare -A F=(
  [contradiction:8k]=_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl
  [contradiction:2k]=_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl
  [nq:8k]=outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl [nq:2k]=outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl
  [outlier:8k]=outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl [outlier:2k]=outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl
  [oolong:8k]=_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl
  [oolong:2k]=_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl)
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); i=0
for cell in contradiction:8k nq:8k outlier:8k oolong:8k contradiction:2k nq:2k outlier:2k oolong:2k; do
  t=${cell%:*}; g=${cell#*:}; key=$([ $t = contradiction ] && echo contra || echo $t)
  OUT=$R/out/$RUN/${t}_$g; mkdir -p $OUT
  [ -f $OUT/.done ] && continue
  slot=$((i % 2)); i=$((i+1))
  ( env CUDA_VISIBLE_DEVICES=${GPUS[$slot]} RUN=$RUN TASK=$key VARIANT=docchunk STEP= CKPT=$SAVE EVAL_OUT_DIR=$OUT PROMPT_FORMAT=chat \
      QUERY_POSITION=after MAX_TEST=500 MAX_LENGTH=20000 BATCH_SIZE=8 NGPU=1 LADDER_XLONG=0 COT_MODE=none LADDER_VERSION=v2 \
      TOKENIZER=Qwen/Qwen3.5-0.8B-Base DC_RUNG_FILES="{\"$t\": {\"$g\": \"$A/eval/${F[$cell]}\"}}" DC_RUNGS=$g \
      WEKA_LLM=$A/weka_stub RESULTS=$R/eval_results REPO=$CODE LADDER_GEN_BATCH=32 LADDER_GEN_BATCH_TOKENS=196608 \
      PORT=$((29600 + 7 * slot + (SLURM_JOB_ID % 50) * 20)) bash recipes/eval/legacy_native/run_beaker_multirung_eval.sh \
      > $OUT/eval.log 2>&1
    grep -aE "\[ladder:" $OUT/eval.log | tail -1 && touch $OUT/.done ) &
  [ $slot = 1 ] && wait
done
wait
echo "[$(date +%T)] evals: $(ls $R/out/$RUN/*/.done 2>/dev/null | wc -l)/8 done"
grep -ahE "\[ladder:" $R/out/$RUN/*/eval.log
echo RUN_DONE
