#!/bin/bash
# Beaker CPU job: the 32k CPT shard for recipe_cpt (research/token_routing/recipe_cpt/README.md).
# Same source, same token stream, same 512-token marker-wrapped pseudo-doc geometry as softdetach's cpt_u1B
# (softdetach/build_cpt_shard.py), cut at 63 pseudo-docs per row instead of 127: a row is 63 x 512 + EOS = 32,257 tokens.
#   cpt32k_u1B   31,008 rows = 1,938 steps x 16 rows = 1,000,225,056 row tokens (source parts 0-27, as cpt_u1B)
#   dev          softdetach's held-out cpt_dev (LAST source part) is reused as is; this job checks that the last
#                part is outside 0-27.
#   PATH=$HOME/.local/bin:$PATH bash research/token_routing/recipe_cpt/build_cpt32k_shard_beaker.sh
set -uo pipefail
SRC=/weka/oe-training-default/ai2-llm/checkpoints/amandab/dolma3_longmino_mix_sample15B_qwen3_5
OUT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/recipe_cpt/shards
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
ROWS=31008; ROWLEN=32257
read -r -d '' WORK <<EOF
set -uo pipefail
export PYTHONWARNINGS=ignore
PYB=/opt/conda/bin/python; \$PYB -m pip install -q -e . 2>&1 | tail -1
N=\$(ls $SRC/part-*.npy | wc -l); echo "source parts: \$N (train reads 0-27, dev = part \$((N-1)))"
[ \$N -ge 29 ] || { echo "!!! fewer than 29 source parts: dev part would overlap train"; exit 1; }
mkdir -p $OUT
[ -s $OUT/cpt32k_u1B/metadata.json ] || PYTHONPATH=src \$PYB research/token_routing/softdetach/build_cpt_shard.py --src $SRC --out $OUT/cpt32k_u1B --tokens $((ROWS*ROWLEN)) --parts 0-27 --blocks 63 || exit 1
\$PYB -c "
import json; m=json.load(open('$OUT/cpt32k_u1B/metadata.json'))
print({k: m[k] for k in ('num_instances','num_tokens','num_loss_tokens','max_example_len','min_example_len','source_parts')})
assert m['num_instances'] == $ROWS and m['max_example_len'] == $ROWLEN, 'shard size mismatch'
print('SHARD OK')" || exit 1
ls -la $OUT/cpt32k_u1B /weka/oe-training-default/ai2-llm/checkpoints/prasanns/softdetach_cpt/shards/cpt_dev
cat /weka/oe-training-default/ai2-llm/checkpoints/prasanns/softdetach_cpt/shards/cpt_dev/metadata.json
echo "=== softdetach dense CPT checkpoints (control candidates) ==="
ls -d /weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/sdcpt-q35-4b-dense* 2>&1
for d in /weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/sdcpt-q35-4b-dense*; do echo "\$d:"; ls \$d | head; done
echo "=== DONE ==="
EOF
gantry run --name "recipe-cpt-data-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' --cluster 'ai2/saturn*' --gpus 0 --cpus 8 --memory 160GiB --priority urgent \
  --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false --branch "$BRANCH" --allow-dirty \
  --weka oe-training-default:/weka/oe-training-default \
  --timeout 0 --yes --no-logs -- bash -c "$WORK"
