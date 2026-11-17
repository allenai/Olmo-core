set -e
D=/accounts/projects/sewonm/prasann/ctc_suite/OLMo-core-otr16
mkdir -p $D/logs
cd $D/recipes/train/research_trainer
COMMON="--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-drop-layer 7"
sub() {  # RUN EXTRA
  REPO=$D TASK=mt4 VARIANT=softtoken SCALE=4b MODEL_FAMILY=qwen3_5 DATA_DIR=/tmp/prasann/tdlr/shards/${DATA:-mt4s1464}   RUN=$1 EPOCHS=1 SEQ_LEN=${SEQ:-10240} LR=${LRX:-2e-06} GLOBAL_BATCH=2 MICRO_BATCH=1 ACT_CKPT=full SAVE_CKPT_FLAG= MAX_STEPS=3 NGPU=2 SEED=$3   BASE_SRC=/tmp/prasann/tdlr/base/q35-4b-base-markerfix SAVE_ROOT=/tmp/prasann/tdlr/ckpts WORK_ROOT=/tmp/prasann/tdlr/work   WANDB_GROUP=f8k-otr EXTRA="$COMMON $2"   sbatch -w lambda-hyperplane02 --partition=lambda --account=site --qos=preemptive --gres=gpu:A100:2     --cpus-per-task=16 --mem=200G --time=01:30:00 --job-name=$1 --output=$D/logs/%x_%j.log --export=ALL run_ctc_lambda.sbatch
}
sub_dense() {  # RUN EXTRA
  REPO=$D TASK=mt4 VARIANT=full SCALE=4b MODEL_FAMILY=qwen3_5 DATA_DIR=/tmp/prasann/tdlr/shards/${DATA:-mt4s1464}   RUN=$1 EPOCHS=1 SEQ_LEN=${SEQ:-10240} LR=${LRX:-2e-06} GLOBAL_BATCH=2 MICRO_BATCH=1 ACT_CKPT=full SAVE_CKPT_FLAG= MAX_STEPS=3 NGPU=2 SEED=$3   BASE_SRC=/tmp/prasann/tdlr/base/q35-4b-base-markerfix SAVE_ROOT=/tmp/prasann/tdlr/ckpts WORK_ROOT=/tmp/prasann/tdlr/work   WANDB_GROUP=f8k-otr EXTRA="$2"   sbatch -w lambda-hyperplane02 --partition=lambda --account=site --qos=preemptive --gres=gpu:A100:2     --cpus-per-task=16 --mem=200G --time=01:30:00 --job-name=$1 --output=$D/logs/%x_%j.log --export=ALL run_ctc_lambda.sbatch
}
SEQ=131072 DATA=mt4L131072 LRX=5e-06 sub_dense f8k-memprobe5-dense-T131072-lam "--skip-dry-run --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
SEQ=196608 DATA=mt4L196608 LRX=5e-06 sub_dense f8k-memprobe5-dense-T196608-lam "--skip-dry-run --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
SEQ=196608 DATA=mt4L196608 LRX=5e-06 sub f8k-memprobe5-otr10M7de0-T196608-lam "--st-otr-keep-frac 0.1 --st-otr-dense-every 0 --skip-dry-run --st-otr-warmup-random --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
SEQ=262144 DATA=mt4L262144 LRX=5e-06 sub f8k-memprobe5-otr10M7de0-T262144-lam "--st-otr-keep-frac 0.1 --st-otr-dense-every 0 --skip-dry-run --st-otr-warmup-random --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
SEQ=131072 DATA=mt4L131072 LRX=5e-06 sub f8k-memprobe5-recipe-T131072-lam "--st-otr-dense-every 0 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --skip-dry-run --st-otr-warmup-random --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
SEQ=131072 DATA=mt4L131072 LRX=5e-06 sub f8k-memprobe5-loraall-T131072-lam "--st-otr-dense-every 0 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --skip-dry-run --st-otr-warmup-random --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
SEQ=196608 DATA=mt4L196608 LRX=5e-06 sub f8k-memprobe5-recipe-T196608-lam "--st-otr-dense-every 0 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --skip-dry-run --st-otr-warmup-random --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
SEQ=196608 DATA=mt4L196608 LRX=5e-06 sub f8k-memprobe5-loraall-T196608-lam "--st-otr-dense-every 0 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --skip-dry-run --st-otr-warmup-random --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
SEQ=262144 DATA=mt4L262144 LRX=5e-06 sub f8k-memprobe5-recipe-T262144-lam "--st-otr-dense-every 0 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --skip-dry-run --st-otr-warmup-random --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
SEQ=262144 DATA=mt4L262144 LRX=5e-06 sub f8k-memprobe5-loraall-T262144-lam "--st-otr-dense-every 0 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --skip-dry-run --st-otr-warmup-random --ffn-chunk-tokens 8192 --mem-probe-steps 3" 0
