#!/bin/bash --noprofile
# Stream gradprobe inputs from THIS Berkeley node's /data straight into a Lambda compute node's /tmp/prasann/gp
# (Lambda's /accounts quota is usually full; node-local /tmp is not quota'd). Run inside a CPU job on the source node
# with a clean environment, otherwise inherited shell hooks can spawn `conda` and wedge on NFS:
#   srun --export=NONE --nodelist=horton ... /bin/bash --noprofile --norc stage_to_lambda.sh lambda-hyperplane05 ckpts/ctc-4b-nq-full ...
# Args: <lambda node> <devloss_grid-relative checkpoint dirs...>. Always ships code, tokenizer snapshot and rung rows.
set -eo pipefail
NODE=$1; shift
cd /data/prasann
items=(gradprobe/code_1013_1550/src gradprobe/code_1013_1550/research
       hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
       devloss_grid/data/nq devloss_grid/data/outlier devloss_grid/data/contradiction_iid)
for c in "$@"; do items+=("devloss_grid/$c"); done
echo "== $(hostname) -> $NODE: ${items[*]} ($(date +%T))"
# -h: dereference the HF snapshot symlinks (blobs live elsewhere in the cache)
/usr/bin/tar -ch "${items[@]}" | /usr/bin/ssh -o BatchMode=yes lambda \
  "srun --partition=lambda --account=site --qos=normal --nodelist=$NODE --cpus-per-task=4 --mem=16G --time=02:00:00 \
   --job-name=gp_stage /bin/bash --noprofile --norc -c 'mkdir -p /tmp/prasann/gp && tar -x -C /tmp/prasann/gp && du -sh /tmp/prasann/gp/devloss_grid/ckpts/* && hostname'"
echo "== done $(date +%T)"
