#!/bin/bash
# Export a multimodal SFT checkpoint: the whole-model HF directory (vision evals) with the promoted
# tokenizer, then the vision-stripped language model (text evals), then a text-generation smoke.
# Runs in the training image on one GPU.
#   olmo35_mm_sft_export.sh STEP_DIR OUT_DIR [--skip-validation]
set -uo pipefail
STEP=$1; OUT=$2; SKIP=${3:-}
H=/weka/oe-training-default/jasonr/home
OC=$H/OLMo-core-wt-mmsft-train
SL=$H/scaling-ladders-wt-mm
TOK=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/a7/cache-neptune/numpy_sft/bf82a8fab8-6068a350-rowaligned/tokenizer
echo MMSFT-EXPORT-START $(date -u +%FT%TZ) $STEP
pip install -q -e "$OC[fla,kernel-fun]" 2>&1 | tail -1
pip install -q -e "$SL/ladders/olmoe3/transformers_plugin" 2>&1 | tail -1
export PYTHONPATH=$OC/src:$SL
(cd $SL && git log -1 --format='scaling-ladders %H'); (cd $OC && git log -1 --format='olmo-core %H')
mkdir -p $OUT
python $SL/ladders/olmoe3/workloads/convert_olmo_core_checkpoint_to_hf.py convert $STEP --output $OUT/hf_mm \
  --tokenizer $TOK $SKIP; RC=$?
echo MMSFT-CONVERT-EXIT $RC
if [ $RC != 0 ] && [ -z "$SKIP" ]; then
  echo MMSFT-CONVERT-RETRY-SKIP-VALIDATION
  python $SL/ladders/olmoe3/workloads/convert_olmo_core_checkpoint_to_hf.py convert $STEP --output $OUT/hf_mm \
    --tokenizer $TOK --skip-validation; RC=$?
  echo MMSFT-CONVERT-EXIT $RC
fi
[ $RC = 0 ] || exit $RC
python $OC/src/scripts/olmo35_mm_strip_vision.py $OUT/hf_mm $OUT/hf_text || exit 3
python - "$OUT/hf_text" <<'EOF'
import sys, torch, transformers
d = sys.argv[1]
tok = transformers.AutoTokenizer.from_pretrained(d)
assert tok.encode("<think>", add_special_tokens=False) == [100256], tok.encode("<think>", add_special_tokens=False)
assert tok.encode("</think>", add_special_tokens=False) == [100266]
m = transformers.AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
for q in ("What is 17 * 23?", "Write one sentence about the ocean."):
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt").cuda()
    out = m.generate(ids, max_new_tokens=48, do_sample=False)
    logits = m(ids).logits
    print("SMOKE", repr(q), "finite", bool(torch.isfinite(logits).all()), repr(tok.decode(out[0, ids.shape[1]:])))
EOF
echo MMSFT-EXPORT-EXIT $? $(date -u +%FT%TZ)
