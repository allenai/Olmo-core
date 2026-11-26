#!/bin/bash
#SBATCH --job-name=tokgrad-gdd
#SBATCH --partition=berkeleynlp
#SBATCH --qos=normal
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=60
#SBATCH --output=/data/prasann/tokgrad/logs/%j.log
# One job, one GPU, <= 1 h (sanity check): gold-drop gradient probe on 3 checkpoints, 16k rows first, 64k if time.
# Code staged at $CODE (git archive), merged checkpoints + rows at /data/prasann/tokgrad/{ckpts,rows.npz}.
set -o pipefail
D=/data/prasann/tokgrad
CODE=${CODE:?}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH=$CODE/src PYTHONUNBUFFERED=1 TMPDIR=$D/tmp TRITON_CACHE_DIR=$D/triton HF_HOME=$D/hf HOME=$D/home
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TMPDIR $TRITON_CACHE_DIR $HF_HOME $HOME
P=$CODE/research/token_routing/grad_concentration/golddrop/tokgrad_probe.py
run() {  # label ckpt sets n formats
  $PY $P --ckpt $D/ckpts/$1 --label $1 --rows $D/rows.npz --sets $2 --n $3 --formats $4 --draws 2 \
    --router-state $D/ckpts/$1/train_rank0.pt --out $D/results/$1-$2 2>&1 | grep -v Warning
}
for L in ${LABELS:-gdd256-s11496 gdd256-s5800 dense256b-s615}; do
  case $L in dense*) F=A,B;; *) F=A,B,C;; esac
  run $L lc16k ${N16:-16} $F
done
for L in ${LABELS:-gdd256-s11496 gdd256-s5800 dense256b-s615}; do
  case $L in dense*) F=A,B;; *) F=A,B,C;; esac
  run $L kl64k ${N64:-8} $F
done
echo ALL_DONE
