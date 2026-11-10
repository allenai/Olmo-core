#!/bin/bash
#SBATCH --job-name=gradcomp
#SBATCH --partition=lambda
#SBATCH --account=site
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/accounts/projects/sewonm/prasann/ctc_suite/logs/gradcomp_%x_%j.log
# probe.py on Lambda A100 from node-local /tmp checkpoints + shards (the tdlr staging on lambda-hyperplane02).
# Code: rsync src/ + research/token_routing/{grad_composition,online_router} to $LROOT/OLMo-core-gradcomp.
#   sbatch --gres=gpu:A100:1 --qos=preemptive_high -w lambda-hyperplane02 --job-name=gradcomp-<label> \
#     --export=ALL,LABEL=<label>,CKPT=/tmp/prasann/tdlr/ckpts/<run> run_probe_lambda.sbatch
# Rows default to mt11s720 rows 480-511 per task (unseen by the base, every mt4 run and mt11 runs up to s480).
# Outputs: /tmp/prasann/gradcomp/out/<label>/ (copied to $LROOT/gradcomp_out/<label>/ without the dumps).
set -o pipefail   # no -u: it kills the conda-pack activate
LROOT=/accounts/projects/sewonm/prasann/ctc_suite
REPO=$LROOT/OLMo-core-gradcomp
LABEL="${LABEL:?set LABEL}"; CKPT="${CKPT:?set CKPT}"
SHARD="${SHARD:-/tmp/prasann/tdlr/shards/mt11s720}"
ROW_OFFSET="${ROW_OFFSET:-480}"; ROWS="${ROWS:-32}"
ENV=/accounts/projects/sewonm/prasann/olmo_test/env
source "$ENV/bin/activate"; conda-unpack 2>/dev/null || true
export PATH="$ENV/bin:$PATH"
export HF_HOME=/accounts/projects/sewonm/prasann/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache; mkdir -p "$TRITON_CACHE_DIR"
OUT=/tmp/prasann/gradcomp/out/$LABEL; mkdir -p "$OUT"
cd "$REPO"
PYTHONPATH=$REPO/src "$ENV/bin/python" -u research/token_routing/grad_composition/probe.py \
  --ckpt "$CKPT" --label "$LABEL" --shard "$SHARD" --row-offset "$ROW_OFFSET" --rows "$ROWS" \
  --dump-dir /tmp/prasann/gradcomp/dumps --tokenizer Qwen/Qwen3.5-0.8B-Base --out "$OUT/probe.json" $EXTRA
rc=$?
mkdir -p "$LROOT/gradcomp_out/$LABEL" && cp "$OUT"/probe* "$LROOT/gradcomp_out/$LABEL/" 2>/dev/null
echo "probe exit $rc"
exit $rc
