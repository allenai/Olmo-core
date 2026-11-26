#!/bin/bash
# Beaker payload (launch_stage.py): weka checkpoints of the 256k contradiction arms -> merged model-only bf16 copies on S3
# (s3://ai2-llm/checkpoints/prasanns/_transfer/tokgrad/<arm>-s<step>/), plus each checkpoint's trainer state
# (train/rank0.pt: the online router's fitted state lives in the OnlineRouterCallback entry) and config.json.
# Env: CKPTS = "arm:step arm:step ..."; then on the Berkeley node: python ../../early_loss_proxy/s3sync.py down <s3> <dir>.
set -o pipefail
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
SR=$W/early_loss_proxy/runs_lc
DST=s3://ai2-llm/checkpoints/prasanns/_transfer/tokgrad
PY=${PY:-python}
HERE=research/token_routing/grad_concentration/golddrop
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 AWS_PROFILE=${AWS_PROFILE:-S3}
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
echo "[stage] host $(hostname) commit $(git rev-parse --short HEAD 2>/dev/null) start $(date -u +%FT%TZ) CKPTS=$CKPTS"
for arm in gdd256 dense256b; do echo "--- $SR/$arm:"; ls "$SR/$arm" | tr '\n' ' '; echo; done
for spec in $CKPTS; do
  arm=${spec%%:*}; st=${spec##*:}; CK=$SR/$arm/step$st; NAME=$arm-s$st; OUT=/tmp/tokgrad/$NAME
  echo "=== $NAME: $CK $(date +%T)"; ls -la "$CK" "$CK/train" 2>&1 | head -30
  [ -d "$CK/model_and_optim" ] || { echo "!!! missing $CK/model_and_optim"; continue; }
  mkdir -p "$OUT"
  "$PY" $HERE/merge_lora_ckpt.py "$CK" "$OUT" || { echo "!!! merge failed $NAME"; rm -rf "$OUT"; continue; }
  [ -f "$CK/train/rank0.pt" ] && cp "$CK/train/rank0.pt" "$OUT/train_rank0.pt"
  [ -f "$CK/config.json" ] && cp "$CK/config.json" "$OUT/config.json"
  "$PY" - "$OUT/train_rank0.pt" <<'PY' 2>&1 | tail -20
import sys, torch
try:
    s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except Exception as e:
    print("no trainer state:", e); sys.exit(0)
print("trainer state keys:", list(s)[:20])
cb = s.get("callbacks", {})
print("callbacks:", list(cb))
for k, v in cb.items():
    if "router" in k.lower() or (isinstance(v, dict) and "chain" in v):
        ch = v.get("chain", [])
        print("router state", k, "stages", len(ch), [{kk: (tuple(vv.shape) if hasattr(vv, "shape") else vv) for kk, vv in c.items() if kk in ("w", "step", "n_seen")} for c in ch])
PY
  "$PY" research/token_routing/early_loss_proxy/s3sync.py up "$OUT" "$DST/$NAME" 2>&1 | tail -n 2 && echo "SYNCED $NAME $(du -sh $OUT | cut -f1)"
  rm -rf "$OUT"
done
echo STAGE_DONE
