#!/bin/bash
# Beaker memory probes (2 x H100, FSDP-2, full AC): NAME VARIANT T EXTRA
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/online-router-1bc01a
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
OTR="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-drop-layer 7"
sub() {  # NAME VARIANT T EXTRA
  E="--attn-backend flash_2 --long-rows-target $3 $4"; [ "$2" = softtoken ] && E="$OTR $E"
  timeout 300 $PY recipes/train/research_trainer/beaker_ctc_suite.py --task mt4 --variant $2 --model-family qwen3_5 \
    --model-scale 4b --data-root $W/ds64/fast8k/shards/mt4s1464 --run-name $1 --exact-run-name --num-nodes 1 \
    --num-gpus 2 --epochs 1 --lr 5e-06 --cluster ai2/jupiter-cirrascale-2 --wandb-group f8k-otr --priority urgent \
    --no-compile --seq-len $3 --global-batch 2 --micro-batch-instances 1 --max-steps 3 --activation-checkpointing full \
    --base-checkpoint $W/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim --experiment learned_router \
    --expected-minutes 20 --extra-args "$E" launch 2>&1 | grep -oE "01[A-Z0-9]{24}" | head -1 | sed "s/^/$1\t/"
}
P="--mem-probe-steps 3"; S="--skip-dry-run --st-otr-warmup-random $P"; X="--ffn-chunk-tokens 8192 --gdn-segment-tokens 16384"
A1="--st-otr-dense-every 0 --st-otr-stages 7:0.2,16:0.07"; A2="--st-otr-keep-frac 0.2 --st-otr-dense-every 0"
for T in 32768 65536; do
  sub f8k-bkprobe-dense-T$T full $T "$P"
  sub f8k-bkprobe-otr10M7de0-T$T softtoken $T "--st-otr-keep-frac 0.1 --st-otr-dense-every 0 $P"
done
for T in 32768 65536 131072; do
  sub f8k-bkprobe-r20to7M7de0-T$T softtoken $T "$A1 $S"
  sub f8k-bkprobe-otr20M7de0-T$T softtoken $T "$A2 $S"
done
for T in 131072 196608 262144; do
  sub f8k-bkprobex-r20to7M7de0-T$T softtoken $T "$A1 $S $X"
  sub f8k-bkprobex-otr20M7de0-T$T softtoken $T "$A2 $S $X"
done
