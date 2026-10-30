#!/bin/bash
# Multi-view (--st-otr-views) LOCAL smoke on horton, 1 GPU, berkeleynlp --qos=normal (debug only): the random-choice
# recipe (lora64all32rnd20to7) at k = 1 and at k = VIEWS for MAX_STEPS steps each on the mt4s488 shard, global batch 4,
# micro 2 (two micro-batches per step, FSDP on one rank, full AC). Prints CE, the FLOP meter's per-step ratio and peak
# memory. Code is staged to node /data first (run_tests_node.sh stages a git archive into /data/prasann/mv_<tag>).
#   sbatch -p berkeleynlp --qos=normal -w horton --gres=gpu:1 -c 8 --mem=120G -t 1:00:00 \
#     --export=ALL,CODE=/data/prasann/mv_<sha> research/token_routing/multiview/mv_smoke_local.sbatch
#SBATCH --job-name=mv-smoke
#SBATCH --output=/data/prasann/mv_logs/%x_%j.log
set -uo pipefail
CODE="${CODE:?}"; VIEWS="${VIEWS:-2}"; MAX_STEPS="${MAX_STEPS:-4}"; R=/data/prasann/mv_smoke
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled WANDB_API_KEY=disabled PYTHONWARNINGS=ignore PYTHONPATH=$CODE/src
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
mkdir -p $TMPDIR $TRITON_CACHE_DIR $R/runs $R/work $R/logs
cd $CODE
BASE=/data/prasann/dropstruct/base/q35-4b-base-markerfix/model_and_optim
D=/data/prasann/tdlr/shards/mt4s488
RND="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32 --st-otr-score random --attn-backend flash_2"
for K in ${KS:-1 $VIEWS}; do
  RUN=mv-smoke-k$K-$SLURM_JOB_ID
  echo "=== [$(date +%T)] train $RUN (views $K)"
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 recipes/train/research_trainer/train_ctc_suite.py \
    --task mt4 --data $D --variant softtoken --model-family qwen3_5 --model-scale 4b \
    --seq-len 12288 --epochs 1 --lr 1e-5 --global-batch 4 --micro-batch-instances 2 --base-checkpoint $BASE \
    --work-dir $R/work/$RUN --save-folder $R/runs/$RUN --run-name $RUN --wandb-group mv-smoke --no-wandb --no-compile \
    --max-steps $MAX_STEPS --activation-checkpointing full --no-final-checkpoint $RND --st-otr-views $K \
    > $R/logs/train_$RUN.log 2>&1
  echo "=== [$(date +%T)] train rc=$?"
  grep -aE "multiview|Traceback|Error|error:|step_actual_over_dense|actual_pflops|CE loss|peak_mem|OOM|nan" $R/logs/train_$RUN.log \
    | grep -av "UserWarning" | tail -60
done
echo SMOKE_DONE
