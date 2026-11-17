#!/bin/bash
# Lambda twin of run_proxy.sbatch: same trainer flags / probe flags / arm loop, 2 A100-80GB. Do NOT execute directly;
# launch_proxy_lambda.py submits it with sbatch (env: CODE STEPS SETS HELDOUT TAG DATA SEQLEN NGPU GB MICRO
# STOPCE MAXTRAIN SPEC (path to the TSV spec on Lambda)). Differences from Berkeley: conda-pack env, NFS paths under
# LROOT/elp (Lambda has no usable /data), runs/work dirs on node-local /tmp, own step-0 cache.
#SBATCH --partition=lambda
#SBATCH --account=site
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
set -o pipefail   # no -u: the conda-pack activate script dies under it
CODE=${CODE:?}; TAG=${TAG:?}
LROOT=/accounts/projects/sewonm/prasann/ctc_suite
ROOT=$LROOT/elp
ENV=/accounts/projects/sewonm/prasann/olmo_test/env
PY=$ENV/bin/python
BASE=$ROOT/bases/q35-4b-base-markerfix/model_and_optim
DATA=${DATA:-$ROOT/shards/mtb16k}
HELDOUT=${HELDOUT:-$ROOT/heldout_v2.npz}
SEQLEN=${SEQLEN:-16384}; NGPU=${NGPU:-2}; GB=${GB:-16}; MICRO=${MICRO:-2}
STEP0=${STEP0:-$ROOT/step0_cache.json}
OUT=$ROOT/results/$TAG
LOCAL=/tmp/$USER/elp                      # node-local, not quota'd
mkdir -p "$OUT" "$LOCAL/tmp" "$LOCAL/runs"
[ -f "$BASE/.metadata" ] || { echo "FATAL: base $BASE missing .metadata"; exit 3; }
[ -f "$DATA/metadata.json" ] || { echo "FATAL: pool $DATA missing"; exit 3; }
[ -f "$HELDOUT" ] || { echo "FATAL: held-out $HELDOUT missing"; exit 3; }
source "$ENV/bin/activate"; conda-unpack 2>/dev/null || true
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline WANDB_API_KEY=offline_dummy_air_gapped
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TMPDIR=$LOCAL/tmp OMP_NUM_THREADS=8
export HF_HOME=/accounts/projects/sewonm/prasann/.cache/huggingface
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache; mkdir -p "$TRITON_CACHE_DIR"
cd "$CODE"
cp "$SPEC" "$OUT/spec_${SLURM_JOB_ID}.tsv"
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID code=$CODE data=$DATA heldout=$HELDOUT"
while IFS=$'\t' read -r NAME VARIANT LR SEED FLAGS; do
  [ -z "$NAME" ] && continue
  if [ -f "$OUT/$NAME.json" ] && [ -z "${FORCE:-}" ] && grep -q '"done": true' "$OUT/$NAME.json"; then
    echo "skip $NAME (done)"; continue
  fi
  SAVE=$LOCAL/runs/$NAME-$SLURM_JOB_ID; rm -rf "$SAVE"; mkdir -p "$SAVE"
  echo "=== $NAME variant=$VARIANT lr=$LR seed=$SEED flags=$FLAGS start $(date +%T)"
  T0=$(date +%s)
  # shellcheck disable=SC2086
  "$PY" -m torch.distributed.run --nproc_per_node=$NGPU --master_port=$((29000 + RANDOM % 1000)) \
    research/token_routing/early_loss_proxy/dense_probe.py \
    --probe-steps "$STEPS" --probe-sets "$SETS" --probe-heldout "$HELDOUT" --probe-out "$OUT/$NAME.json" --probe-step0-cache "$STEP0" ${STOPCE:+--probe-stop-ce $STOPCE} ${MAXTRAIN:+--probe-max-train-s $MAXTRAIN} -- \
    --task mt4 --variant "$VARIANT" --model-family qwen3_5 --model-scale 4b \
    --run-name "$NAME" --base-checkpoint "$BASE" --data "$DATA" --work-dir "$LOCAL/work_${SLURM_JOB_ID}" \
    --save-folder "$SAVE" --seq-len $SEQLEN --epochs 1 --lr "$LR" --global-batch $GB --micro-batch-instances $MICRO \
    --seed "$SEED" --no-compile --no-wandb --no-final-checkpoint $FLAGS > "$OUT/$NAME.log" 2>&1
  RC=$?
  grep -a "dense-probe" "$OUT/$NAME.log"
  [ $RC -ne 0 ] && tail -n 60 "$OUT/$NAME.log"
  T1=$(date +%s)
  DATA=$DATA SEQLEN=$SEQLEN NGPU=$NGPU GB=$GB MICRO=$MICRO "$PY" - "$OUT/$NAME.json" "$((T1 - T0))" "$RC" <<'PYEOF'
import json, os, sys
p, wall = sys.argv[1], float(sys.argv[2])
try:
    d = json.load(open(p))
except FileNotFoundError:
    print("NO RESULT", p); sys.exit(0)
d["done"] = bool(d.get("finished")) or len(d["probes"]) == len(d["probe_steps"])
d["process_wall_s"] = wall
d["rc"] = int(sys.argv[3])
d["slurm"] = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURMD_NODENAME", "DATA", "SEQLEN", "NGPU", "GB", "MICRO", "CODE")}
d["cluster"] = "lambda"
json.dump(d, open(p, "w"))
print("result", p, "done", d["done"], "process wall", wall)
PYEOF
  rm -rf "$SAVE"
  echo "=== $NAME end $(date +%T) rc=$RC wall $((T1 - T0))s"
done < "$OUT/spec_${SLURM_JOB_ID}.tsv"
rm -rf "$LOCAL/work_${SLURM_JOB_ID}"
echo "ELP_DONE"
