#!/bin/bash
# Early dense-loss proxy (records/early-dense-loss-proxy.md): a list of arm runs back to back on one jsteinhardt node,
# 2 GPUs. Code staged on the node first (research/token_routing/wts/stage_code.sh); base, shards, held-out rows on its
# /data. Launched by launch_proxy.py (env: CODE STEPS SETS HELDOUT TAG SPEC_B64 = base64 of TSV lines
# name<TAB>variant<TAB>lr<TAB>seed<TAB>trainer flags). No checkpoints; results JSON per run in results/$TAG/.
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/data/prasann/joblogs/%x-%j.out
set -o pipefail
CODE=${CODE:?}; TAG=${TAG:?}
ROOT=/data/prasann/early_loss_proxy
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
PY=$ENV/bin/python
BASE=/data/prasann/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim
DATA=${DATA:-/data/prasann/early_loss_proxy/shards/mtb16k}
SEQLEN=${SEQLEN:-16384}; NGPU=${NGPU:-2}; GB=${GB:-16}; MICRO=${MICRO:-2}
STEP0=${STEP0:-$ROOT/step0_cache.json}
OUT=$ROOT/results/$TAG
mkdir -p "$ROOT/tmp" "$ROOT/runs" "$OUT" /data/prasann/joblogs
[ -f "$BASE/.metadata" ] || { echo "FATAL: base $BASE missing .metadata"; exit 3; }
[ -f "$DATA/metadata.json" ] || { echo "FATAL: pool $DATA missing"; exit 3; }
[ -f "$HELDOUT" ] || { echo "FATAL: held-out $HELDOUT missing"; exit 3; }
export PATH="$ENV/bin:$PATH" PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache HF_HOME=$ROOT/hf_cache
export HOME=/data/prasann/home_wts CUDA_HOME=/usr/local/cuda-12.8
cd "$CODE"
echo "$SPEC_B64" | base64 -d > "$OUT/spec_${SLURM_JOB_ID}.tsv"
while IFS=$'\t' read -r NAME VARIANT LR SEED FLAGS; do
  [ -z "$NAME" ] && continue
  if [ -f "$OUT/$NAME.json" ] && [ -z "${FORCE:-}" ] && grep -q '"done": true' "$OUT/$NAME.json"; then
    echo "skip $NAME (done)"; continue
  fi
  SAVE=$ROOT/runs/$NAME-$SLURM_JOB_ID; rm -rf "$SAVE"; mkdir -p "$SAVE"
  echo "=== $NAME variant=$VARIANT lr=$LR seed=$SEED flags=$FLAGS start $(date +%T)"
  T0=$(date +%s)
  # shellcheck disable=SC2086
  "$PY" -m torch.distributed.run --nproc_per_node=$NGPU --master_port=$((29000 + RANDOM % 1000)) \
    research/token_routing/early_loss_proxy/dense_probe.py \
    --probe-steps "$STEPS" --probe-sets "$SETS" --probe-heldout "$HELDOUT" --probe-out "$OUT/$NAME.json" --probe-step0-cache "$STEP0" -- \
    --task mt4 --variant "$VARIANT" --model-family qwen3_5 --model-scale 4b \
    --run-name "$NAME" --base-checkpoint "$BASE" --data "$DATA" --work-dir "$ROOT/work_${SLURM_JOB_ID}" \
    --save-folder "$SAVE" --seq-len $SEQLEN --epochs 1 --lr "$LR" --global-batch $GB --micro-batch-instances $MICRO \
    --seed "$SEED" --no-compile --no-wandb --no-final-checkpoint $FLAGS > "$OUT/$NAME.log" 2>&1
  RC=$?
  grep -a "dense-probe" "$OUT/$NAME.log"
  [ $RC -ne 0 ] && tail -n 60 "$OUT/$NAME.log"
  T1=$(date +%s)
  DATA=$DATA SEQLEN=$SEQLEN NGPU=$NGPU GB=$GB MICRO=$MICRO "$PY" - "$OUT/$NAME.json" "$((T1 - T0))" <<'EOF'
import json, sys
p, wall = sys.argv[1], float(sys.argv[2])
try:
    d = json.load(open(p))
except FileNotFoundError:
    print("NO RESULT", p); sys.exit(0)
d["done"] = len(d["probes"]) == len(d["probe_steps"])
d["process_wall_s"] = wall
d["slurm"] = {k: __import__("os").environ.get(k) for k in ("SLURM_JOB_ID", "SLURMD_NODENAME", "DATA", "SEQLEN", "NGPU", "GB", "MICRO", "CODE")}
json.dump(d, open(p, "w"))
print("result", p, "done", d["done"], "process wall", wall)
EOF
  rm -rf "$SAVE"
  echo "=== $NAME end $(date +%T) rc=$RC wall $((T1 - T0))s"
done < "$OUT/spec_${SLURM_JOB_ID}.tsv"
rm -rf "$ROOT/work_${SLURM_JOB_ID}"
echo "ELP_DONE"
