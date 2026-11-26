#!/bin/bash
# 8-GPU max-length probe: blocks 7-31 spread over 8 H200s (sweep_probe.py --devices 0..7 --memprobe T) for an
# ascending list of row lengths; stages the checkpoint and the rows from S3 if the node lacks them. One JSON per T in
# /data/prasann/gradlen/results/memprobe8/. Env: CODE, TS (space-separated lengths), YARN.
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclude=balrog,saruman,shadowfax,smaug
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/%x-%j.out
set -o pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CODE=${CODE:?}; PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
CK=m64-dense-f1-lr5e-06
export TMPDIR=/data/prasann/gradlen/tmp TRITON_CACHE_DIR=/data/prasann/gradlen/tmp/triton PYTHONPATH="$CODE/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TMPDIR /data/prasann/gradlen/results/memprobe8
# stage the code on whatever node this lands on (one small read of the repo over NFS)
[ -d "$CODE" ] || { mkdir -p "$CODE" && git -C "$REPO" archive "${SHA:?set SHA}" | tar -x -C "$CODE"; }
cd "$CODE"
s3down() { "$PY" -u -c "import sys; sys.path.insert(0, 'research/token_routing/olmo3_cptdrop'); import s3_sync as s
s.ROOT = '$3'; sys.argv = ['s3_sync', 'down', '$1', '$2']; s.main()" | tail -1; }
s3down /data/prasann/gradconc/ckpts/$CK $CK checkpoints/prasanns/_transfer/gradconc
s3down /data/prasann/gradlen/tok/both/nq_8k tok/both/nq_8k checkpoints/prasanns/_transfer/gradlen
for T in ${TS:-393216 524288 655360 786432 1048576}; do
  echo "=== T=$T $(date +%T)"
  OPS_RUN_ID="gsl-memprobe8-$T-${SLURM_JOB_ID}" OPS_EXPERIMENT=grad_composition OPS_KIND=other OPS_EXPECTED_MINUTES=60 \
  OPS_FIRST_OUTPUT=1800 OPS_STALL=3600 bash "$REPO/recipes/env/ops_job.sh" "$PY" -u \
    research/token_routing/grad_concentration/length_sweep/sweep_probe.py --ckpt /data/prasann/gradconc/ckpts/$CK/model_and_optim \
    --rows-from tok --tok-root /data/prasann/gradlen/tok/both --tasks nq --rung 8k --rows 4 --devices 0,1,2,3,4,5,6,7 \
    --yarn ${YARN:-4} --memprobe $T --out /data/prasann/gradlen/results/memprobe8 || true
  grep -q '"ok": true' /data/prasann/gradlen/results/memprobe8/memprobe_$T.json 2>/dev/null || { echo "stop at T=$T"; break; }
done
echo MEMPROBE8_DONE
