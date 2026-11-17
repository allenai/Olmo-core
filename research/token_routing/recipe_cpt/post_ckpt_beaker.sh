#!/bin/bash
# One 1-GPU Beaker job per CPT checkpoint: wait for it, export it as a plain base (LoRA merged for the recipe arm,
# model-only re-save for dense), and score full-context dev CE (cpt_dev, 32 held-out 64k rows) on the export.
#   RUN=rcpt-q35-4b-recipe-u1B STEP=242 ARM=recipe bash post_ckpt_beaker.sh      # a mid-run checkpoint
#   RUN=rcpt-q35-4b-recipe-u1B STEP=final ARM=recipe bash post_ckpt_beaker.sh    # the post-fit export
#   CK=<base>/model_and_optim NAME=base ARM=base bash post_ckpt_beaker.sh          # score a base, no export
#   Launch it once the checkpoint exists: it holds a GPU while it waits (WAITMAX seconds, default 0 = fail fast).
#   SMOKE=1 also scores the UNMERGED checkpoint (adapters re-installed) to check merged == unmerged.
# Exports land in $WEKA/recipe_cpt/bases/<RUN>-step<STEP>/model_and_optim (SFT: --base-checkpoint <that>).
set -uo pipefail
ARM="${ARM:?recipe|dense|base}"; ROWS="${ROWS:-32}"; SMOKE="${SMOKE:-0}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE=$WEKA/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim
DEV=$WEKA/softdetach_cpt/shards/cpt_dev
if [ "$ARM" = base ]; then NAME="${NAME:?}"; CKP="${CK:?}"; else
  RUN="${RUN:?}"; STEP="${STEP:?}"; NAME="$RUN-step$STEP"
  if [ "$STEP" = final ]; then CKP=$WEKA/ctc_suite/ckpts/$RUN/model_and_optim; else CKP=$WEKA/ctc_suite/ckpts/$RUN/step$STEP/model_and_optim; fi
fi
EXP=$WEKA/recipe_cpt/bases/$NAME
OUT=$WEKA/recipe_cpt/devloss
read -r -d '' WORK <<EOF
set -uo pipefail
export PYTHONWARNINGS=ignore PATH=/opt/conda/bin:\$PATH
PYB=/opt/conda/bin/python; \$PYB -m pip install -q --no-deps -e . 2>&1 | tail -1
\$PYB -c "import torch, fla, olmo_core; print(torch.__version__)" || { echo "!!! deps missing"; exit 1; }
W=0; until [ -f $CKP/.metadata ]; do
  [ \$W -ge ${WAITMAX:-0} ] && { echo "!!! no checkpoint $CKP (WAITMAX=${WAITMAX:-0}s)"; exit 2; }
  sleep 120; W=\$((W+120)); done; sleep 60; echo "checkpoint: $CKP (waited \${W}s)"
mkdir -p $OUT
if [ "$ARM" = base ]; then
  PYTHONPATH=src \$PYB research/token_routing/recipe_cpt/eval_cpt_devloss.py --ckpt $CKP --dev $DEV --rows $ROWS --out $OUT/$NAME.json || exit 1
  exit 0
fi
if [ "$SMOKE" = 1 ] && [ "$ARM" = recipe ]; then
  PYTHONPATH=src \$PYB research/token_routing/recipe_cpt/eval_cpt_devloss.py --ckpt $CKP --dev $DEV --rows $ROWS --lora-rank 64 --lora-scale 32 --out $OUT/$NAME.unmerged.json || exit 1
fi
if [ ! -f $EXP/model_and_optim/.metadata ]; then
  if [ "$ARM" = recipe ]; then FL="--base $BASE"; else FL="--no-lora"; fi
  PYTHONPATH=src \$PYB research/token_routing/recipe_cpt/merge_lora_export.py --ckpt $CKP --out $EXP \$FL || exit 1
fi
PYTHONPATH=src \$PYB research/token_routing/recipe_cpt/eval_cpt_devloss.py --ckpt $EXP/model_and_optim --dev $DEV --rows $ROWS --out $OUT/$NAME.json || exit 1
echo "=== DONE $NAME ==="
EOF
gantry run --name "rcpt-post-$NAME-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --gpus 1 --cpus 16 --memory 240GiB --priority urgent --min-runtime "${MINRT:-1h}" \
  --beaker-image akshitab/olmo-core-tch2130cu129-sm80-2026-09-11 --install false --branch "$BRANCH" --allow-dirty \
  --weka oe-training-default:/weka/oe-training-default \
  --timeout 0 --yes -- bash -c "$WORK"
