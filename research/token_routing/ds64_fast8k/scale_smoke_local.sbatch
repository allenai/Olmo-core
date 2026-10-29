#!/bin/bash
# Model-scale ablation LOCAL smoke (tokdrop_lr, 2026-10-29): Beaker jupiter was not scheduling, so the smoke runs on
# jsteinhardt. One job on 2 GPUs of mooney: (1) repair the 0.8B (or SCALE) base's marker rows with
# fix_marker_embeddings_qwen35.py, (2) dense and the recipe for MAX_STEPS steps each on the mt4s1952 shard,
# (3) one 1-GPU eval cell (nq @2k, MAX_TEST rows) of each export through the Beaker eval runner.
# Code is staged to node /data first (git archive | srun tar -x); CODE points at it.
#   sbatch -p jsteinhardt --account=site --qos=preemptive_high -w mooney --gres=gpu:2 -c 16 --mem=200G -t 2:00:00 \
#     --export=ALL,CODE=/data/prasann/scale_abl/code_<sha>,SCALE=0.8b research/token_routing/ds64_fast8k/scale_smoke_local.sbatch
#SBATCH --job-name=sca-smoke
#SBATCH --output=/data/prasann/scale_abl/logs/%x_%j.log
set -uo pipefail
SCALE="${SCALE:-0.8b}"; T=$(echo $SCALE | tr -d .); MAX_STEPS="${MAX_STEPS:-4}"; MAX_TEST="${MAX_TEST:-100}"
CODE="${CODE:?}"; R=/data/prasann/scale_abl; A="${ASSETS:-/data/prasann/scale_abl/assets}"; PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
# olmo-core needs WANDB_API_KEY in the env (only rank 0 raises; the other rank then spins in NCCL): backfill from ~/.netrc
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' /accounts/projects/berkeleynlp/prasann/.netrc)
[ -n "$WANDB_API_KEY" ] || { echo "!!! no WANDB_API_KEY"; exit 4; }
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONWARNINGS=ignore PYTHONPATH=$CODE/src
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
mkdir -p $TMPDIR $TRITON_CACHE_DIR $R/bases $R/runs $R/work $R/out
cd $CODE
SRC=$A/ctc_suite/bases/q35-${T}-base-modelonly/model_and_optim
BASE=$R/bases/q35-${T}-base-markerfix
if [ ! -f $BASE/model_and_optim/.metadata ]; then
  echo "=== marker fix $SCALE: $SRC -> $BASE"
  $PY recipes/data/fix_marker_embeddings_qwen35.py --base $SRC --out $BASE --model-scale $SCALE || { echo "!!! FIX FAILED"; exit 1; }
fi
$PY recipes/data/fix_marker_embeddings.py --audit-only --base $BASE/model_and_optim --family qwen3_5 \
  --marker-set doc_start,doc_end,landmark,pad 2>&1 | tail -8
NL=$($PY -c "print({'0.8b':24,'2b':24,'4b':32,'9b':32}['$SCALE'])")
S2=$([ $NL = 24 ] && echo 12 || echo 16)
RECIPE="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,$S2:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-early-lora-blocks $NL --attn-backend flash_2"
for ARM in dense recipe; do
  RUN=sca-smoke-$T-$ARM-$SLURM_JOB_ID
  if [ $ARM = dense ]; then V=full; X=""; else V=softtoken; X="$RECIPE"; fi
  echo "=== train $RUN"
  $PY -m torch.distributed.run --standalone --nproc_per_node=2 recipes/train/research_trainer/train_ctc_suite.py \
    --task mt4 --data $A/tdlr/shards/mt4s1952 --variant $V --model-family qwen3_5 --model-scale $SCALE \
    --seq-len 12288 --epochs 1 --lr 1e-5 --global-batch 16 --micro-batch-instances 2 --base-checkpoint $BASE/model_and_optim \
    --work-dir $R/work/$RUN --save-folder $R/runs/$RUN --run-name $RUN --wandb-group sca-smoke --no-compile \
    --max-steps $MAX_STEPS --activation-checkpointing full --save-checkpoint --no-final-checkpoint $X > $R/logs/train_$RUN.log 2>&1
  echo "=== train rc=$?"; cat $R/logs/train_$RUN.log \
    | grep -aE "step=|Traceback|Error|error:|audit|AUDIT|marker|flop_meter|reserved mem|online-router\] step|n_layers|OOM|nan" | grep -av "UserWarning" | tail -60
  ls $R/runs/$RUN
  OUT=$R/out/$RUN; mkdir -p $OUT
  echo "=== eval $RUN nq@2k MAX_TEST=$MAX_TEST"
  env CUDA_VISIBLE_DEVICES=0 RUN=$RUN TASK=nq VARIANT=docchunk STEP= CKPT=$R/runs/$RUN EVAL_OUT_DIR=$OUT PROMPT_FORMAT=chat \
    QUERY_POSITION=after MAX_TEST=$MAX_TEST MAX_LENGTH=20000 BATCH_SIZE=8 NGPU=1 LADDER_XLONG=0 COT_MODE=none LADDER_VERSION=v2 \
    TOKENIZER=Qwen/Qwen3.5-0.8B-Base DC_RUNG_FILES="{\"nq\": {\"2k\": \"$A/eval/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl\"}}" \
    DC_RUNGS=2k WEKA_LLM=$A/weka_stub RESULTS=$R/eval_results REPO=$CODE LADDER_GEN_BATCH=32 \
    LADDER_GEN_BATCH_TOKENS=196608 PORT=29651 bash recipes/eval/legacy_native/run_beaker_multirung_eval.sh 2>&1 \
    | grep -aE "\[ladder|parse|Traceback|Error|error:|empty" | tail -15
done
echo SMOKE_DONE
