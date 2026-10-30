#!/usr/bin/env bash
# Qwen3 size ablation (2026-10-29): convert Qwen/Qwen3-<S>-Base -> olmo-core, repair the marker embeddings (cosine AND
# norm: fix_marker_embeddings.py seeds each marker from a trained donor row and asserts both gates), and stage the
# repaired base to S3 for Berkeley slurm. Writes $P/ctc_suite/bases/qwen3-<s>-base-trainedmark/model_and_optim (the
# path beaker_ctc_suite.py defaults to for --model-family qwen3) and s3://.../_transfer/q3recipe/bases/<same name>/.
# One CPU gantry job, sizes in sequence; a size whose trainedmark base already exists is only audited + staged.
#   SIZES="1.7B 8B 0.6B" bash research/token_routing/ds64_fast8k/build_q3_bases_beaker.sh
set -uo pipefail
P=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
SIZES="${SIZES:-1.7B 8B 0.6B}"
read -r -d '' WORK <<'EOW'
set -uo pipefail
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
PYB=/opt/conda/bin/python
$PYB -m pip install -q -e . 2>&1 | tail -1
pip install -q awscli 2>&1 | tail -1
# the image's torchvision does not match its torch, and transformers imports it for Qwen3 modeling -> uninstall it
$PYB -m pip uninstall -y -q torchvision 2>&1 | tail -1
mkdir -p ~/.aws && printf '%s' "$AWS_CREDS" > ~/.aws/credentials && printf '%s' "$AWS_CFG" > ~/.aws/config
for S in $SIZES; do
  s=$(echo $S | tr 'B' 'b')
  RAW=$P/ctc_suite/bases/qwen3-$s-base-hfconv
  FIX=$P/ctc_suite/bases/qwen3-$s-base-trainedmark
  echo "=== $S $(date +%T) ==="
  if [ ! -f $FIX/model_and_optim/.metadata ]; then
    if [ ! -f $RAW/model_and_optim/.metadata ]; then
      PYTHONPATH=src $PYB recipes/train/research_trainer/base_conversion/convert_hf_to_olmo.py \
        --base-model Qwen/Qwen3-$S-Base --out $RAW || { echo "!!! convert FAILED $S"; exit 1; }
    fi
    PYTHONPATH=src $PYB recipes/data/fix_marker_embeddings.py --base $RAW/model_and_optim --out $FIX \
      --family qwen3 --model-size $S --audit-json $FIX/marker_audit.json || { echo "!!! repair FAILED $S"; exit 1; }
    cp $RAW/config.json $FIX/config.json 2>/dev/null
  fi
  PYTHONPATH=src $PYB recipes/data/fix_marker_embeddings.py --audit-only --base $FIX/model_and_optim --family qwen3 \
    --marker-set doc_start,doc_end --audit-json /tmp/audit_$s.json || { echo "!!! audit FAILED $S"; exit 1; }
  $PYB -c "import json;d=json.load(open('/tmp/audit_$s.json'));print('[audit] $S', {k: d.get(k) for k in ('pairwise_cosine','marker_norm_ratios','pass')})"
  AWS_PROFILE=S3 aws s3 sync $FIX s3://ai2-llm/checkpoints/prasanns/_transfer/q3recipe/bases/qwen3-$s-base-trainedmark/ \
    --only-show-errors --no-follow-symlinks && echo "SYNCED $S" || { echo "!!! s3 FAILED $S"; exit 1; }
done
echo BASES_DONE
EOW
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
gantry run --name "q3-bases-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 16 --memory 192GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env SIZES="$SIZES" --env P="$P" \
  --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -E "beaker.org/ex|rror" | head -3
