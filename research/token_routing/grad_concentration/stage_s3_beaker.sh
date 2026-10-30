#!/usr/bin/env bash
# Stage the grad_concentration inputs from weka to S3 for local (Berkeley) runs, when Beaker GPUs are backed up:
#   * dense mt4 checkpoints as MODEL-ONLY bf16 distcp (recipe: pooled_kv/transfer_dense_ckpts_gantry.sh,
#     memory weka-s3-checkpoint-transfer) -> s3://ai2-llm/checkpoints/prasanns/_transfer/gradconc/<name>/
#   * the probe shards ({task}_fresh, mt4f7808, mt4s1952; npy + metadata only) -> .../_transfer/gradconc/shards/
# Then on the target node:  AWS_PROFILE=S3 aws s3 sync s3://ai2-llm/checkpoints/prasanns/_transfer/gradconc/ /data/prasann/gradconc/
set -uo pipefail
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
NAMES="${NAMES:-f8k-mt4-dense-f1952-tde-lr5e-06 f8k-mt4-dense-f3904-tde-lr5e-06 f8k-mt4-dense-f5856-tde-lr5e-06 f8k-mt4-dense-f7808-tde-lr5e-06 f8k-mt4-dense-s488st31-tdc-lr5e-06 f8k-mt4-dense-s488st61-tdc-lr5e-06 f8k-mt4-dense-s488-tdc-lr5e-06 f8k-mt4-dense-s976-tdc-lr5e-06 f8k-mt4-dense-s1464-tdc-lr5e-06 f8k-mt4-dense-s1952-tdc-lr5e-06}"
read -r -d '' WORK <<'EOW'
set -uo pipefail
mkdir -p ~/.aws && printf '%s' "$AWS_CREDS" > ~/.aws/credentials && printf '%s' "$AWS_CFG" > ~/.aws/config
pip install -q awscli 2>&1 | tail -1
DST=s3://ai2-llm/checkpoints/prasanns/_transfer/gradconc
SH=$W/ds64/fast8k/shards
[ "${SKIP_SHARDS:-0}" = 1 ] || for S in contradiction_fresh nq_fresh outlier_fresh oolong_fresh mt4f7808 mt4s1952; do
  AWS_PROFILE=S3 aws s3 sync $SH/$S $DST/shards/$S/ --only-show-errors --exclude "*" --include "*.npy" \
    --include "metadata.json" && echo "SYNCED shard $S $(du -sh $SH/$S | cut -f1)" || echo "!!! FAILED shard $S"
done
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
for NAME in $NAMES; do
  case $NAME in /*) CK=$NAME; NAME=$(basename $(dirname $NAME));; *) CK=$W/ctc_suite/ckpts/$NAME/model_and_optim;; esac
  [ -d "$CK" ] || CK=$(ls -d $W/ctc_suite/ckpts/$NAME/step*/model_and_optim 2>/dev/null | tail -1)
  [ -n "$CK" ] && [ -d "$CK" ] || { echo "!!! no model_and_optim for $NAME"; continue; }
  OUT=/tmp/transfer/$NAME; mkdir -p $OUT
  echo "--- $NAME: $CK -> $OUT $(date +%T)"
  python - "$CK" "$OUT" <<'PY'
import os, sys, time, torch, torch.distributed as dist
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
ck, out = sys.argv[1], sys.argv[2]
dist.init_process_group("gloo", rank=0, world_size=1)
t = time.time()
m = TransformerConfig.qwen3_5_4B(vocab_size=248320, attn_backend=AttentionBackendName("flash_2")).build(init_device="cpu")
load_model_and_optim_state(ck, m)
m = m.to(torch.bfloat16)
save_model_and_optim_state(out + "/model_and_optim", m, save_overwrite=True)
print("resaved", out, f"{sum(p.numel() for p in m.parameters()) / 1e9:.2f}B params, {time.time() - t:.0f}s", flush=True)
PY
  if [ $? -ne 0 ]; then echo "!!! FAILED $NAME (python)"; rm -rf $OUT; continue; fi
  AWS_PROFILE=S3 aws s3 sync $OUT $DST/$NAME/ --only-show-errors && echo "SYNCED $NAME $(du -sh $OUT | cut -f1)" \
    || echo "!!! FAILED $NAME (s3 sync)"
  rm -rf $OUT
done
echo TRANSFER_DONE
EOW
export PATH=$HOME/.local/bin:/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "gradconc-stage-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 16 --memory 160GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env NAMES="$NAMES" --env W="$W" --env SKIP_SHARDS="${SKIP_SHARDS:-0}" \
  --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -E "beaker.org/ex|rror|xperiment" | head -5
