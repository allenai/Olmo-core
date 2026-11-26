#!/usr/bin/env bash
# Stage the trainable tensors (LoRA adapters + LM head + any key missing from the base) of the long-context
# contradiction checkpoints from weka to S3, so the edge-gradient analysis can run on Berkeley slurm.
#   ARMS="gdd256 dense256b" (default); for each arm: the earliest kept stepN and the last one.
# -> s3://ai2-llm/checkpoints/prasanns/_transfer/attn_edge_grad/<arm>_step<N>.pt  ({key: tensor}, plus "__meta__")
# Then on the node: AWS_PROFILE=S3 aws s3 sync s3://ai2-llm/checkpoints/prasanns/_transfer/attn_edge_grad/ /data/prasann/attn_edge_grad/ckpt/
set -uo pipefail
ARMS="${ARMS:-gdd256 dense256b}"
read -r -d '' WORK <<'EOW'
set -uo pipefail
mkdir -p ~/.aws && printf '%s' "$AWS_CREDS" > ~/.aws/credentials && printf '%s' "$AWS_CFG" > ~/.aws/config
pip install -q awscli 2>&1 | tail -1
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
DST=s3://ai2-llm/checkpoints/prasanns/_transfer/attn_edge_grad
BASE=$W/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim
mkdir -p /tmp/xfer
for ARM in $ARMS; do
  ls -d $W/early_loss_proxy/runs_lc/$ARM/step* 2>/dev/null | sed 's/.*step//' | sort -n > /tmp/steps_$ARM
  echo "--- $ARM steps: $(tr '\n' ' ' < /tmp/steps_$ARM)"
  for ST in $(head -1 /tmp/steps_$ARM) $(tail -1 /tmp/steps_$ARM); do
    CK=$W/early_loss_proxy/runs_lc/$ARM/step$ST/model_and_optim
    OUT=/tmp/xfer/${ARM}_step$ST.pt
    [ -f $OUT ] && continue
    python - "$CK" "$BASE" "$OUT" <<'PY'
import sys, torch
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
ck, base, out = sys.argv[1:]
mk = set(get_checkpoint_metadata(ck).state_dict_metadata)
bk = set(get_checkpoint_metadata(base).state_dict_metadata)
want = sorted(k for k in mk if k.startswith("model.") and ("lora_" in k or "lm_head" in k or k not in bk))
vals = load_keys(ck, want)
sd = {k: v.detach().cpu() for k, v in zip(want, vals)}
sd["__meta__"] = {"ckpt": ck, "n_keys": len(want), "n_model_keys": sum(k.startswith("model.") for k in mk),
                  "not_in_base": [k for k in want if k not in bk and "lora_" not in k]}
torch.save(sd, out)
print("saved", out, len(want), "keys; not in base (non-lora):", sd["__meta__"]["not_in_base"][:20], flush=True)
PY
    AWS_PROFILE=S3 aws s3 cp $OUT $DST/$(basename $OUT) --only-show-errors && echo "SYNCED $OUT $(du -sh $OUT | cut -f1)"
  done
done
echo TRANSFER_DONE
EOW
export PATH=$HOME/.local/bin:/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "attnedge-stage-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 16 --memory 120GiB --priority urgent --min-runtime 30m \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env ARMS="$ARMS" --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -E "beaker.org/ex|rror|xperiment" | head -5
