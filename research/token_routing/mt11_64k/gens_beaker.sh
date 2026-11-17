#!/bin/bash
# CPU gantry job: show_gens.py over eval generation files on weka.  FILES="<run>-ev<task>/eval/<t>_multirung.generations.jsonl ..." RUNGS="8k 16k"
set -euo pipefail
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
"$GANTRY" run --name "mt11-64k-gens-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster 'ai2/jupiter*' \
  --gpus 0 --cpus 2 --memory 16GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --no-python \
  --weka oe-training-default:/weka/oe-training-default \
  --env W="$W" --env FILES="$FILES" --env RUNGS="${RUNGS:-}" --env N="${N:-3}" --allow-dirty --timeout 0 --yes -- bash -c '
cd $W/ctc_suite/ckpts 2>/dev/null; ls -d $W/m64-*ev* 2>/dev/null | head -3
F=""; for f in $FILES; do for root in $W $W/ctc_suite/ckpts $W/ctc_suite/evals; do [ -f $root/$f ] && F="$F $root/$f" && break; done; done
[ -z "$F" ] && { echo "!!! none found; searching"; find $W -maxdepth 4 -path "*m64-dense-f8-lr2e-06-evreorder*" 2>/dev/null | head; exit 1; }
python3 /gantry-runtime/research/token_routing/mt11_64k/show_gens.py --files $F ${RUNGS:+--rungs $RUNGS} --n $N; echo GENS_DONE' 2>&1 | grep -oE "ex/[A-Z0-9]{26}" | head -1
