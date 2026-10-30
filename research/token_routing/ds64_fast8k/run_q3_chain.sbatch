#!/bin/bash
# Several run_q3_local.sbatch runs back to back in ONE allocation (2 GPUs), so a slot shared with other sessions'
# queued jobs is not lost between runs. SPECS = space-separated arm:budget:lr; a run that already trained is only
# evaluated (run_q3_local skips training when flops.json + a model export exist). Each run logs to its own
# /data/prasann/joblogs/q3recipe_<arm>-<budget>-<lr>_chain<jobid>.log (what q3_flops_to_match.py reads).
#
#   sbatch -p jsteinhardt --qos=preemptive_high --account=site -w sneetches --gres=gpu:H200:2 --job-name=q3chainA \
#     --export=ALL,CODE=/data/prasann/q3recipe/code_<sha>,SPECS="dense:f7808:5e-6 dense:f3904:5e-6" \
#     research/token_routing/ds64_fast8k/run_q3_chain.sbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/prasann/joblogs/q3recipe_%x_%j.log
set -uo pipefail
CODE="${CODE:?}"; SPECS="${SPECS:?}"
for spec in $SPECS; do
  # arm:budget:lr, or size:arm:budget:lr (Q3_SIZE per run; size ablation)
  if [ $(tr -cd ':' <<< "$spec" | wc -c) -eq 3 ]; then IFS=: read -r size arm budget lr <<< "$spec"; else size=${Q3_SIZE:-4b}; IFS=: read -r arm budget lr <<< "$spec"; fi
  log=/data/prasann/joblogs/q3recipe_${size}-${arm}-${budget}-${lr}_chain${SLURM_JOB_ID}.log
  echo "=== chain $(date '+%F %T') $spec -> $log ==="
  Q3_SIZE=$size ARM=$arm BUDGET=$budget LR=$lr CODE=$CODE bash $CODE/research/token_routing/ds64_fast8k/run_q3_local.sbatch > $log 2>&1
  echo "=== chain $spec rc=$? $(date '+%F %T') ==="
done
