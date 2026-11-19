#!/bin/bash
# The phase-1 feasibility smokes as launched (job B = lcsmB, job A final = lcsmA5). Flags:
R="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32 --attn-backend flash_2 --activation-checkpointing full --skip-dry-run --st-otr-warmup-random"
LORA="--lora-rank 64 --lora-blocks 0 --lora-scale 32 --attn-backend flash_2"
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/longctx-contra-golddrop
L=research/token_routing/early_loss_proxy/longctx_contra/launch_lc_smoke.py
G="--st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.15"
PC="--prefix-chunk 4096"
python3 $L --tag lcsmA5 --cluster ai2/jupiter-cirrascale-2 --gpus 1 --minutes 75 ${DRY:+--dry-run} \
  --run "gdd256|probe|1|softtoken|1e-5|lc256k|262144|1|8|lc128k:1|$R $G" \
  --run "rcw256|probe|1|softtoken|1e-5|lc256k|262144|1|2|lc128k:1|$R --st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 1.0" \
  --run "rcw256pc|probe|1|softtoken|1e-5|lc256k|262144|1|2|lc128k:1|$R --st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 1.0 $PC" \
  --run "gdd1m50pc|probe|1|softtoken|1e-5|lc1m|1048576|1|2|lc128k:1|$R --st-gdd-keep-range 0.5,0.5 --st-gdd-whole-frac 0.0 $PC" \
  --run "gdd1m25|probe|1|softtoken|1e-5|lc1m|1048576|1|2|lc128k:1|$R --st-gdd-keep-range 0.25,0.25 --st-gdd-whole-frac 0.0" \
  --run "gdd1mpc|probe|1|softtoken|1e-5|lc1m|1048576|1|4|lc128k:1|$R --st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.0 $PC" \
  --run "fwd1m|probe|1|softtoken|1e-5|lc1m|1048576|1|1|lc1m:1|$R --st-gdd-keep-range 0.05,0.05 --st-gdd-whole-frac 0.0"
python3 $L --tag lcsmB --gpus 8 --minutes 45 ${DRY:+--dry-run} \
  --run "dl256cp4|direct|8|full|1e-5|lc256k|262144|2|3|-|$LORA --activation-checkpointing full --cp-degree 4" \
  --run "dl256cp2|direct|8|full|1e-5|lc256k|262144|4|3|-|$LORA --activation-checkpointing full --cp-degree 2" \
  --run "dl1mcp4|direct|8|full|1e-5|lc1m|1048576|2|2|-|$LORA --activation-checkpointing full --cp-degree 4"
