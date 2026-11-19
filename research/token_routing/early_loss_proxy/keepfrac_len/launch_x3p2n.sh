#!/bin/bash
# Three-task 16k runs on kl16k_x3p2_n3440 (645 steps, 4 GPUs) via launch_proxy.py. Env: NODE, DATA, SEED, STEPS.
#   NODE=cubbins SEED=1 bash launch_x3p2n.sh tpA_w5_r05to50_x3p2n "--st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.05 --st-gdd-groups <pool>/gdd_groups.json"
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/elp-lambda-kl || exit 1
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python; L=research/token_routing/early_loss_proxy/launch_proxy.py
R="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32 --attn-backend flash_2"
$PY $L --tag kl16kx --seeds ${SEED:-0} --steps ${STEPS:-0,25,50,100,150,225,300,375,450,525,600,645} --sets kl16k:48 \
  --heldout /data/prasann/early_loss_proxy/heldout_kl.npz --data ${DATA:-/data/prasann/early_loss_proxy/shards/kl16k_x3p2_n3440} \
  --seq-len 16384 --gpus 4 --code /data/prasann/early_loss_proxy/code_13bf548f4 --node ${NODE:-sneetches} --qos preemptive_high \
  --minutes 70 --arms "$1" --custom "$1=softtoken,1e-05,$R $2"
