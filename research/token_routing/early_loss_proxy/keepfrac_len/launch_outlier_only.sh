#!/bin/bash
# Outlier-only 16k runs on kl16k_outlier (75 steps, 2 GPUs) via launch_proxy.py: launch_outlier_only.sh CODE ARM "flags" [ARM "flags" ...]
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/elp-lambda-kl || exit 1
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python; L=research/token_routing/early_loss_proxy/launch_proxy.py
R="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32 --attn-backend flash_2"
CODE=$1; shift
ARMS=(); CUST=()
while [ $# -gt 0 ]; do ARMS+=("$1"); CUST+=(--custom "$1=softtoken,1e-05,$R $2"); shift 2; done
A=$(IFS=,; echo "${ARMS[*]}")
$PY $L --tag kl16kol --seeds 0 --steps 0,10,25,50,75 --sets kl16k:48 --heldout /data/prasann/early_loss_proxy/heldout_kl16k_outlier.npz \
  --data /data/prasann/early_loss_proxy/shards/kl16k_outlier --seq-len 16384 --gpus 2 --code "$CODE" --node sneetches \
  --qos preemptive_high --minutes $((15 * ${#ARMS[@]})) --arms "$A" "${CUST[@]}"
