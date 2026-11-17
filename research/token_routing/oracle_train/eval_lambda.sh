#!/bin/bash
# Evaluate an oracle run trained on Lambda (checkpoint on node /tmp/prasann/oracle/runs/<run>) with the same on-node
# runner the tokdrop Lambda evals use (research/token_routing/ds64_fast8k/run_eval_lambda.sbatch: the Beaker
# multirung runner, fast8k args, 4-task 8k + 2k rungs x 500, no-cot, query after). Run ON the Lambda head node:
#   bash $L/oracle/code/research/token_routing/oracle_train/eval_lambda.sh <run> <node>
# (ROOT=/tmp/prasann/oracle: ckpts -> runs and eval -> the tdlr eval files, both as PARENT symlinks, since the runner's
# `find $ROOT/ckpts/<run>` does not descend into a symlinked start directory)
# Results: /tmp/prasann/oracle/runs/<run>/eval/*_multirung.json, copied to $L/eval_results/<run>/.
set -euo pipefail
RUN=$1; NODE=$2
L=/accounts/projects/sewonm/prasann/ctc_suite
CODE=$L/oracle/code
for TASK in contradiction nq outlier oolong; do
  sbatch --partition=lambda --account=site --qos=preemptive_high -w "$NODE" --gres=gpu:A100:1 -c 8 --mem 96G \
    -t 03:00:00 -J "$RUN-ev$TASK" -o "$L/oracle/logs/%x-%j.out" --export=ALL,RUN="$RUN",TASK="$TASK",NGPU=1,ROOT=/tmp/prasann/oracle,REPO_OVERRIDE="$CODE" \
    --wrap "ls -d /tmp/prasann/oracle/runs/$RUN/model_and_optim && ln -sfn runs /tmp/prasann/oracle/ckpts && \
            ln -sfn /tmp/prasann/tdlr/eval /tmp/prasann/oracle/eval && \
            bash $CODE/research/token_routing/ds64_fast8k/run_eval_lambda.sbatch"
done
