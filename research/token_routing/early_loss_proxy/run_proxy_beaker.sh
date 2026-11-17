#!/bin/bash
# Beaker payload of the early dense-loss proxy: the run_proxy.sbatch loop, flag for flag, on a 2-GPU Beaker job.
# Launched by launch_proxy_beaker.py (gantry clones the PUSHED commit; cwd = repo root; python = image system python).
# Env: STEPS SETS TAG SPEC_B64 (TSV: name<TAB>variant<TAB>lr<TAB>seed<TAB>trainer flags), optional DATA HELDOUT BASE
# SEQLEN NGPU GB MICRO STEP0 STOPCE MAXTRAIN FORCE. Results JSON + log per run on weka in $ROOT/results/$TAG/, each
# mirrored to s3://ai2-llm/checkpoints/prasanns/early_loss_proxy/results/$TAG/ as soon as the run ends
# (fetch_beaker_results.sh pulls them to Berkeley). Checkpoints and work dirs stay on the node's local disk.
set -o pipefail
TAG=${TAG:?}; STEPS=${STEPS:?}; SETS=${SETS:?}
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
ROOT=$W/early_loss_proxy
S3=s3://ai2-llm/checkpoints/prasanns/early_loss_proxy
BASE=${BASE:-$W/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim}
DATA=${DATA:-$ROOT/shards/mtb16k}
HELDOUT=${HELDOUT:-$ROOT/heldout_v2.npz}
SEQLEN=${SEQLEN:-16384}; NGPU=${NGPU:-2}; GB=${GB:-16}; MICRO=${MICRO:-2}
STEP0=${STEP0:-$ROOT/step0_cache.json}
OUT=$ROOT/results/$TAG
JID=${BEAKER_JOB_ID:-local$$}
LOC=/tmp/elp_$JID  # node-local scratch (save folders, work dir)
PY=${PY:-python}
CODE=$PWD
mkdir -p "$OUT" "$LOC/tmp" "$LOC/runs"
[ -f "$BASE/.metadata" ] || { echo "FATAL: base $BASE missing .metadata"; exit 3; }
[ -f "$DATA/metadata.json" ] || { echo "FATAL: pool $DATA missing"; exit 3; }
[ -f "$HELDOUT" ] || { echo "FATAL: held-out $HELDOUT missing"; exit 3; }
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TMPDIR=$LOC/tmp TRITON_CACHE_DIR=$LOC/triton_cache HF_HOME=$LOC/hf_cache
export AWS_PROFILE=${AWS_PROFILE:-S3}
echo "[elp-beaker] job $JID host $(hostname) commit $(git rev-parse --short HEAD 2>/dev/null) tag $TAG start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sync_out() {  # best effort: a failed upload must not fail the run (the weka copy is the primary)
  "$PY" research/token_routing/early_loss_proxy/s3sync.py up --force "$@" 2>&1 | tail -n 2
}
echo "$SPEC_B64" | base64 -d > "$OUT/spec_${JID}.tsv"
sync_out "$OUT/spec_${JID}.tsv" "$S3/results/$TAG"
while IFS=$'\t' read -r NAME VARIANT LR SEED FLAGS; do
  [ -z "$NAME" ] && continue
  if [ -f "$OUT/$NAME.json" ] && [ -z "${FORCE:-}" ] && grep -q '"done": true' "$OUT/$NAME.json"; then
    echo "skip $NAME (done)"; continue
  fi
  SAVE=$LOC/runs/$NAME; rm -rf "$SAVE"; mkdir -p "$SAVE"
  echo "=== $NAME variant=$VARIANT lr=$LR seed=$SEED flags=$FLAGS start $(date +%T)"
  T0=$(date +%s)
  # shellcheck disable=SC2086
  "$PY" -m torch.distributed.run --nproc_per_node=$NGPU --master_port=$((29000 + RANDOM % 1000)) \
    research/token_routing/early_loss_proxy/dense_probe.py \
    --probe-steps "$STEPS" --probe-sets "$SETS" --probe-heldout "$HELDOUT" --probe-out "$OUT/$NAME.json" --probe-step0-cache "$STEP0" ${STOPCE:+--probe-stop-ce $STOPCE} ${MAXTRAIN:+--probe-max-train-s $MAXTRAIN} -- \
    --task mt4 --variant "$VARIANT" --model-family qwen3_5 --model-scale 4b \
    --run-name "$NAME" --base-checkpoint "$BASE" --data "$DATA" --work-dir "$LOC/work" \
    --save-folder "$SAVE" --seq-len $SEQLEN --epochs 1 --lr "$LR" --global-batch $GB --micro-batch-instances $MICRO \
    --seed "$SEED" --no-compile --no-wandb --no-final-checkpoint $FLAGS > "$LOC/$NAME.log" 2>&1
  RC=$?
  T1=$(date +%s)
  cp "$LOC/$NAME.log" "$OUT/$NAME.log"
  grep -a "dense-probe\|flop_meter\|OutOfMemory\|CUDA out of memory" "$LOC/$NAME.log" | cut -c1-400 | head -n 40
  [ $RC -ne 0 ] && tail -n 60 "$LOC/$NAME.log"
  DATA=$DATA SEQLEN=$SEQLEN NGPU=$NGPU GB=$GB MICRO=$MICRO RC=$RC "$PY" - "$OUT/$NAME.json" "$((T1 - T0))" <<'EOF'
import json, os, socket, sys
p, wall = sys.argv[1], float(sys.argv[2])
try:
    d = json.load(open(p))
except FileNotFoundError:
    print("NO RESULT", p); sys.exit(0)
d["done"] = bool(d.get("finished")) or len(d["probes"]) == len(d["probe_steps"])
d["process_wall_s"] = wall
env = {k: os.environ.get(k) for k in ("DATA", "SEQLEN", "NGPU", "GB", "MICRO", "RC")}
d["beaker"] = dict(env, job=os.environ.get("BEAKER_JOB_ID"), experiment=os.environ.get("BEAKER_EXPERIMENT_ID"),
                   host=socket.gethostname(), commit=os.popen("git rev-parse HEAD").read().strip())
# score.py reads the node name from d["slurm"]["SLURMD_NODENAME"]; keep that key so its table works unchanged
d["slurm"] = dict(env, SLURM_JOB_ID=os.environ.get("BEAKER_JOB_ID"), SLURMD_NODENAME="beaker:" + socket.gethostname(),
                  CODE=d["beaker"]["commit"][:9])
json.dump(d, open(p, "w"))
print("result", p, "done", d["done"], "process wall", wall)
EOF
  sync_out "$OUT/$NAME.json" "$S3/results/$TAG"
  sync_out "$OUT/$NAME.log" "$S3/results/$TAG"
  rm -rf "$SAVE"
  echo "=== $NAME end $(date +%T) rc=$RC wall $((T1 - T0))s"
done < "$OUT/spec_${JID}.tsv"
rm -rf "$LOC"
echo "ELP_DONE"
