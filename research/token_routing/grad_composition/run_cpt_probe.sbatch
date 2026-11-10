#!/bin/bash
# cpt_probe.py (OLMo 3 7B CPT gradient composition) on cubbins: a 1-row smoke on GPU 0, then one checkpoint per GPU in
# parallel (SPECS = label:ckpt_dir:yarn, ckpt relative to $ROOT). Code staged at $CODE first (login node):
#   git archive HEAD | srun -p jsteinhardt --qos=preemptive_high -w cubbins -c1 --mem=2G -t 5 \
#     bash --noprofile --norc -c 'mkdir -p $CODE && tar -x -C $CODE'  # a FRESH $CODE dir
#   sbatch -w cubbins --export=ALL,CODE=/data/prasann/grad_composition/code_<sha> run_cpt_probe.sbatch
#SBATCH --job-name=gradcomp-cpt
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=01:30:00
#SBATCH --output=/data/prasann/grad_composition/logs/%x-%j.out
set -eo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ROOT=/data/prasann/olmo3_cptdrop
CODE=${CODE:?set CODE}
OUT=${OUT:-/data/prasann/grad_composition/cpt}
ROWS=${ROWS:-48}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
SPECS=${SPECS:-"base:base/stage2-step47684:4 dlr20-12:runs/o3cd-Dlr20-12:4 dlr20-48:runs/o3cd-Dlr20-48:4 dlr20-120:runs/o3cd-Dlr20-120:4 dlr20-240:runs/o3cd-Dlr20-240:4 dlr20-480:runs/o3cd-Dlr20-480:4 otr10m7-310:runs/o3cd-OTR10M7lr20-310:4 stage3:base/stage3-step11921:8"}
export HF_HOME="$ROOT/hf_cache" TMPDIR=/data/prasann/grad_composition/tmp TRITON_CACHE_DIR=/data/prasann/grad_composition/tmp/triton
export PYTHONPATH="$CODE/src:$CODE/research/token_routing/olmo3_cptdrop"
mkdir -p "$TMPDIR" "$OUT" /data/prasann/grad_composition/logs
cd "$CODE"
nvidia-smi --query-gpu=index,memory.used --format=csv
# smoke: 1 row, base, GPU 0
CUDA_VISIBLE_DEVICES=0 "$PY" -u research/token_routing/grad_composition/cpt_probe.py --ckpt "$ROOT/base/stage2-step47684" \
  --yarn-factor 4 --dev "$ROOT/data/dev32k_1024" --rows 1 --dump-rows 0 --label smoke --out "$OUT/smoke/cpt.json"
i=0
for SPEC in $SPECS; do
  IFS=: read -r LABEL CK YF <<<"$SPEC"
  mkdir -p "$OUT/$LABEL"; CUDA_VISIBLE_DEVICES=$i OPS_RUN_ID="gradcomp-cpt-${LABEL}-${SLURM_JOB_ID}" OPS_EXPERIMENT=grad_composition OPS_KIND=eval \
    OPS_PARENT_RUN="$(basename "$CK")" OPS_CKPT_PATH="$ROOT/$CK" OPS_LOG_PATH="$OUT/$LABEL.log" OPS_EXPECTED_MINUTES=40 OPS_FIRST_OUTPUT=900 OPS_STALL=1200 \
    bash "$REPO/recipes/env/ops_job.sh" "$PY" -u research/token_routing/grad_composition/cpt_probe.py \
      --ckpt "$ROOT/$CK" --yarn-factor "$YF" --dev "$ROOT/data/dev32k_1024" --rows "$ROWS" --label "$LABEL" \
      --out "$OUT/$LABEL/cpt.json" > "$OUT/$LABEL.log" 2>&1 &
  i=$((i + 1))
  sleep 20  # stagger the 7B checkpoint loads
done
wait
echo "all done"
