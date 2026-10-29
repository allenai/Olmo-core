#!/bin/bash
# KV-only context probe (2026-10-29): grad_fidelity2.py with --modes plain,kv,kvdet,kva,kvadet on the recipe and base
# checkpoints, splits after blocks 1 / 3 / 7 (one- and two-stage), the gradfid2 rows (mt4f1952, 2 x 32 rows per
# task; router fit on 32 mt4s1464 rows per task). One 1-GPU Beaker job per (ckpt, setting, row group); ids appended
# to gfkv_jobs.txt; every job registered with ops. Merge: grad_fidelity2_merge.py --parts <dir of fetched results>.
#   bash research/token_routing/online_router/launch_gfkv.sh [settings...]
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH=$HOME/.local/bin:$PATH
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
OPS=/accounts/projects/berkeleynlp/prasann/bin/ops
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
declare -A CK=([recipe]=$W/ctc_suite/ckpts/f8k-mt4-lora64K32r20to7-s1464-tdb-lr1e-05
               [base]=$W/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim)
SETS=("$@"); [ ${#SETS[@]} -gt 0 ] || SETS=(1st_1_10 1st_3_10 1st_7_10 2st_1_16 2st_3_16 2st_7_16)
for tag in recipe base; do
  for s in "${SETS[@]}"; do
    for g in 0 1; do
      name="gfkv-$tag-${s//_/-}-g$g"
      out=$(timeout 300 $PY research/ffn_compute/flop_scaling/beaker_bench_launch.py --cluster ai2/jupiter \
        --script research/token_routing/online_router/grad_fidelity2.py --name "$name" --extra \
        "--ckpt ${CK[$tag]} --tag $tag --shard $W/ds64/fast8k/shards/mt4f1952 --fit-shard $W/ds64/fast8k/shards/mt4s1464 \
         --row-offset $((32 * g)) --rows-per-task 32 --fit-offset 0 --fit-rows 32 --settings $s \
         --modes plain,kv,kvdet,kva,kvadet --rules oracle,router,random --k 4 --kb 4 --selftest" 2>&1 | grep SUBMITTED)
      id=$(sed -n 's/.*id=\([^ ]*\).*/\1/p' <<<"$out")
      echo "$name $id" | tee -a research/token_routing/online_router/gfkv_jobs.txt
      $OPS register --run-id "$name" --experiment tokdrop_lr --cluster beaker --job-id "$id" \
        --expected-minutes 60 --kind eval --git . >/dev/null
    done
  done
done
