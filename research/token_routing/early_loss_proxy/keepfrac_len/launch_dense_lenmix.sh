#!/bin/bash
# Dense arms on the arm-E short-row length mix (kl16k_lenmixE_172_1138_1200_930), one epoch = 645 steps, via launch_proxy.py.
#   NODE=mooney GPUS=4 SEED=0 bash launch_dense_lenmix.sh F_dft    # dense full FT (kl16k dft settings: lr 5e-6)
#   NODE=mooney GPUS=2 SEED=0 bash launch_dense_lenmix.sh G_ref    # dense LoRA r64 (kl16k ref settings: lr 1e-5)
# Probes bracket ~333 PF (dense 2.19 PF/step -> step ~152; LoRA 1.65 -> ~201).
# Budget-matched (full LR schedule ends at ~333 PF): prefix pools from slice_prefix_pool.py
#   DATA=<pool>_n816  STEPS=0,25,50,75,100,125,153     bash launch_dense_lenmix.sh Fb_dft   (153 steps x 2.19 PF)
#   DATA=<pool>_n1072 STEPS=0,25,50,75,100,125,150,175,201 bash launch_dense_lenmix.sh Gb_ref   (201 steps x 1.65 PF)
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/elp-lambda-kl || exit 1
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python; L=research/token_routing/early_loss_proxy/launch_proxy.py
case "$1" in
  Fb_dft) SPEC="Fb_dft=full,5e-06,--attn-backend flash_2" ;;
  Gb_ref) SPEC="Gb_ref=full,1e-05,--lora-rank 64 --lora-blocks 0 --lora-scale 32 --attn-backend flash_2" ;;
  F_dft) SPEC="F_dft=full,5e-06,--attn-backend flash_2" ;;
  G_ref) SPEC="G_ref=full,1e-05,--lora-rank 64 --lora-blocks 0 --lora-scale 32 --attn-backend flash_2" ;;
  *) echo "arm F_dft or G_ref"; exit 1 ;;
esac
$PY $L --tag kl16kx --seeds ${SEED:-0} --steps ${STEPS:-0,25,50,75,100,125,150,175,200,225,300,375,450,525,600,645} --sets kl16k:48 \
  --heldout /data/prasann/early_loss_proxy/heldout_kl.npz --data ${DATA:-/data/prasann/early_loss_proxy/shards/kl16k_lenmixE_172_1138_1200_930} \
  --seq-len 16384 --gpus ${GPUS:-4} --code /data/prasann/early_loss_proxy/code_13bf548f4 --node ${NODE:-mooney} --qos ${QOS:-preemptive_high} \
  --minutes ${MINUTES:-120} --arms "$1" --custom "$SPEC"
