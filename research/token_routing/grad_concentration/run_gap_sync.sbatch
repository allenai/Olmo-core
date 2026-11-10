#!/bin/bash
# Move a label's sketches between nodes through S3 (shards that ran on different nodes -> one node for analyze_gap.py).
#   sbatch -w <node> -J gradgap-sync --export=ALL,SRC=<checkout>,LABEL=recipe,DIRECTION=up|down run_gap_sync.sbatch
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/prasann/%x-%j.out
set -eo pipefail
SRC=${SRC:?}; LABEL=${LABEL:?}; DIRECTION=${DIRECTION:?}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
CODE=/data/prasann/gradgap/code_${SLURM_JOB_ID}
mkdir -p "$CODE" "/data/prasann/gradgap/sketches/$LABEL"
(cd "$SRC" && tar -c research/token_routing/olmo3_cptdrop/s3_sync.py) | tar -x -C "$CODE"
cd "$CODE"
"$PY" -u -c "import sys; sys.path.insert(0, 'research/token_routing/olmo3_cptdrop'); import s3_sync as s
s.ROOT = 'checkpoints/prasanns/_transfer/gradconc'; sys.argv = ['s3_sync', '$DIRECTION', '/data/prasann/gradgap/sketches/$LABEL', 'sketches/$LABEL']; s.main()" | tail -2
ls /data/prasann/gradgap/sketches/$LABEL | wc -l
echo SYNC_DONE
