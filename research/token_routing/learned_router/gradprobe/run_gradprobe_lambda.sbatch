#!/usr/bin/env bash
# grad_importance.py on ONE Lambda A100, from inputs streamed into the node's /tmp (the /accounts quota is the usual
# blocker on Lambda; see lambda_cluster.md). Expected layout on the node (G=/tmp/prasann/gp):
#   $G/gradprobe/code_1013_1550/{src,research}  $G/devloss_grid/{data,ckpts/<CK>}  $G/hf_cache/.../snapshots/<rev>
# Results + logs go to the small NFS dir $LROOT/gradprobe/ (read from the head node). Pin the node with --nodelist.
#   ssh lambda 'sbatch --nodelist=lambda-hyperplane05 --export=ALL,CK=ctc-4b-nq-full,TASK=nq,RUNG=16k,ROWS=16,TAG=...,EXTRA="..." \
#       /accounts/projects/sewonm/prasann/ctc_suite/gradprobe/run_gradprobe_lambda.sbatch'
#SBATCH --job-name=gradprobe
#SBATCH --partition=lambda
#SBATCH --account=site
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/accounts/projects/sewonm/prasann/ctc_suite/gradprobe/logs/gradprobe_%j.log

set -o pipefail  # no -u: it breaks conda-pack envs (lambda_cluster.md)
: "${CK:?}" "${TASK:?}" "${RUNG:?}" "${ROWS:?}" "${TAG:?}"
G=/tmp/prasann/gp
LROOT=/accounts/projects/sewonm/prasann/ctc_suite
PY=/accounts/projects/sewonm/prasann/olmo_test/env/bin/python
CODE=$G/gradprobe/code_1013_1550
[ -d "$CODE/src" ] || { echo "!!! no staged code on $(hostname) ($CODE)"; exit 1; }
cd "$CODE"
mkdir -p /tmp/prasann/tmp /tmp/prasann/triton /tmp/prasann/xdg "$LROOT/gradprobe/out"
export PYTHONPATH="$CODE/src:$CODE/research/common" PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DEVLOSS_SKIP_CHECKS=1 PYTHONWARNINGS=ignore WANDB_MODE=offline
export HOME=/tmp/prasann TMPDIR=/tmp/prasann/tmp TRITON_CACHE_DIR=/tmp/prasann/triton XDG_CACHE_HOME=/tmp/prasann/xdg
TOK=$G/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
CKPT=$G/devloss_grid/ckpts/$CK
if [ -f "$CKPT/.text_export_done" ]; then FMT=hf; elif [ -f "$CKPT/model_and_optim/.metadata" ]; then FMT=distcp
elif [ -f "$CKPT/.resave_done" ]; then FMT=olmo_st
else echo "!!! no staged checkpoint at $CKPT on $(hostname)"; exit 1; fi
echo "== gradprobe $TAG ($TASK@$RUNG rows=$ROWS ckpt=$CK/$FMT) on $(hostname) job ${SLURM_JOB_ID:-?} ($(date +%T))"
"$PY" -u research/token_routing/learned_router/gradprobe/grad_importance.py --task "$TASK" --rung "$RUNG" --rows "$ROWS" \
  --ckpt "$CKPT" --ckpt-format "$FMT" --data-root "$G/devloss_grid/data" --tokenizer "$TOK" \
  --attn-backend flash_2 --out "$LROOT/gradprobe/out/$TAG.json" ${EXTRA:-}
echo "== exit $? ($(date +%T))"
