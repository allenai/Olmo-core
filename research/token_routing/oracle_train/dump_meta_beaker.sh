#!/usr/bin/env bash
# Print the training-time summary of Beaker-trained oracle / random runs into a Beaker job log (weka is not readable
# from Berkeley): per run, `summarize.py --oneline` (METAJSON line: windowed means of keep, gold/distractor keep, attr
# cos, routed-grad cos, CE dense/routed, FLOPs) and flops.json (FLOPJSON line, the FLOP meter). Read the log with
# `beaker experiment logs <id> | grep -E '^(METAJSON|FLOPJSON)'`.
#   bash research/token_routing/oracle_train/dump_meta_beaker.sh <run> [<run> ...]
set -uo pipefail
NAMES="$*"
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
export PATH=$HOME/.local/bin:/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "oracle-metadump-$(date +%m%d%H%M%S)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 2 --memory 8GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env NAMES="$NAMES" --env W="$W" --allow-dirty --timeout 0 --yes -- bash -c '
for N in $NAMES; do
  D=$W/ctc_suite/ckpts/$N
  if [ -s $D/oracle_meta/steps.jsonl ]; then
    /opt/conda/bin/python research/token_routing/oracle_train/summarize.py $D --every 100000 --oneline $N | grep "^METAJSON"
  else echo "NOMETA $N"; fi
  if [ -s $D/flops.json ]; then echo "FLOPJSON $N $(tr -d "\n" < $D/flops.json)"; else echo "NOFLOPS $N"; fi
  ls $D | tr "\n" " "; echo
done
echo METADUMP_DONE' 2>&1 | grep -E "beaker.org/ex|rror" | head -3
