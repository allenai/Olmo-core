#!/bin/bash
# Runs INSIDE the gantry container (build_mt4_fresh_beaker.sh submits it; env: WEKA N SEED TASK MERGE ROWS FAMILY).
# Its own file so TASK=all can re-invoke it per task ("$0").
set -uo pipefail
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
PYB=/opt/conda/bin/python
F=$WEKA/ds64/fast8k
TOK=$WEKA/hf_tokenizers/Qwen3.5-0.8B-Base; MS=qwen3_5; SFX=""
CHILD="${CHILD:-0}"  # TASK=all children: the parent already installed and checked the tokenizer
[ "$CHILD" = 1 ] || $PYB -m pip install -q -e . -e ./ctc 2>&1 | tail -2
$PYB -c "import numpy, transformers, torch, olmo_core, ctc.data.cli" || { echo "!!! deps missing"; exit 1; }
if [ "$FAMILY" = qwen3 ]; then
  TOK=$WEKA/hf_tokenizers/Qwen3-4B-nothink; MS=qwen3; SFX=_q3
  [ "$CHILD" = 1 ] || $PYB - "$TOK" <<"PYEOF" || { echo "!!! nothink tokenizer check failed"; exit 1; }
import os, sys
from transformers import AutoTokenizer
out = sys.argv[1]
if not os.path.exists(f"{out}/tokenizer_config.json"):
    t = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    old = "enable_thinking is defined and enable_thinking is false"
    assert old in t.chat_template, "Qwen3 template changed; patch by hand"
    t.chat_template = t.chat_template.replace(old, "true")
    t.save_pretrained(out)
    open(f"{out}/README", "w").write("Qwen/Qwen3-4B tokenizer; generation prompt always emits the empty think block "
                                     "(answer loss span = answer only, as Qwen3.5). Built by build_mt4_fresh_beaker.sh FAMILY=qwen3.\n")
t = AutoTokenizer.from_pretrained(out)
m = [{"role": "user", "content": "Q?"}]
p = t.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
f = t.apply_chat_template(m + [{"role": "assistant", "content": "[7]"}], tokenize=False)
assert p.endswith("<think>\n\n</think>\n\n"), repr(p)
assert f.startswith(p), (p, f)
print("[nothink] ok; answer region =", repr(f[len(p):]), "box ids", t.convert_tokens_to_ids(["<|box_start|>", "<|box_end|>"]))
PYEOF
fi
if [ "$MERGE" = 1 ]; then
  for T in contradiction nq outlier oolong; do  # train/eval overlap of each fresh pool (must be ~0) before merging
    for RUNG in 8192 2048; do
      $PYB research/token_routing/ds64_fast2k/fast2k_stats.py --arm-jsonl $F/pool_fresh_$T/$T/train.jsonl \
        --shard-dir $F/shards/${T}_fresh$SFX --eval-jsonl $WEKA/outlier_lengthmix/eval_rungs/$T/rung_$RUNG.jsonl \
        --out $F/shards/${T}_fresh$SFX/fast8k_stats_$RUNG.json > /tmp/st.json || { echo "!!! stats failed $T $RUNG"; exit 1; }
      $PYB -c "import json,sys;d=json.load(open(\"/tmp/st.json\"));o=d.get(\"overlap\",d);print(\"[overlap] $T $RUNG\",{k:v for k,v in o.items() if \"overlap\" in k})"
    done
  done
  for R in $ROWS; do
    OUT=$F/shards/mt4f$R${SFX#_}
    if [ -s $OUT/metadata.json ]; then echo "[skip] $OUT"; continue; fi
    $PYB research/token_routing/ds64_fast8k/merge_tasks.py --out $OUT --rows-per-src $R --name mt4 \
      --src $F/shards/contradiction_fresh$SFX $F/shards/nq_fresh$SFX $F/shards/outlier_fresh$SFX $F/shards/oolong_fresh$SFX \
      || { echo "!!! merge failed $R"; exit 1; }
    echo "[merge] mt4f$R${SFX#_} $($PYB -c "import json;d=json.load(open(\"$OUT/metadata.json\"));print({k:d.get(k) for k in (\"num_instances\",\"max_example_len\",\"marker_set\")})")"
  done
  echo "=== MERGE DONE ==="; exit 0
fi
if [ "$TASK" = all ]; then  # the 4 task builds in parallel (8 procs each), then the merge, in one job
  for T in contradiction nq outlier oolong; do CHILD=1 TASK=$T bash "$0" > /tmp/build_$T.log 2>&1 & done
  wait; for T in contradiction nq outlier oolong; do echo "=== $T ==="; tail -5 /tmp/build_$T.log; grep -q "=== DONE $T" /tmp/build_$T.log || exit 1; done
  CHILD=1 MERGE=1 bash "$0"; exit $?
fi
case "$TASK" in
  nq) CONV=retrieval; CHUNK=document; EXTRA="" ;;
  oolong) CONV=oolong; CHUNK=line; EXTRA="--gold-sidecar-empty-for-missing" ;;
  *) CONV=$TASK; CHUNK=document; EXTRA="" ;;
esac
POOL=$F/pool_fresh_$TASK
if [ ! -s $POOL/$TASK/train.jsonl ] || [ $(wc -l < $POOL/$TASK/train.jsonl) -lt $N ]; then
  echo "--- pool $TASK 8k: $N rows, seed $SEED $(date +%T) ---"
  $PYB -m ctc.data.cli build --task $TASK --out $POOL --split train --rungs 8k --train $N --seed $SEED --pool auto --force \
    || { echo "!!! pool FAILED"; exit 1; }
fi
echo "pool rows: $(wc -l < $POOL/$TASK/train.jsonl)"
OUT=$F/shards/${TASK}_fresh$SFX
if [ -s $OUT/metadata.json ]; then echo "[skip] $OUT tokenized"; else
  mkdir -p $OUT
  PYTHONPATH=src $PYB recipes/data/convert_unified_to_document_landmark.py \
    --input-jsonl $POOL/$TASK/train.jsonl --task $CONV --out-dir $OUT --emit dense --marker-set $MS \
    --tokenizer $TOK --seq-len 24576 --query-position after --cot-mode none \
    --chunk-by $CHUNK --emit-gold-sidecar $EXTRA --num-proc 8 || { echo "!!! tokenize FAILED"; exit 1; }
fi
for RUNG in 8192 2048; do
  $PYB research/token_routing/ds64_fast2k/fast2k_stats.py --arm-jsonl $POOL/$TASK/train.jsonl --shard-dir $OUT \
    --eval-jsonl $WEKA/outlier_lengthmix/eval_rungs/$TASK/rung_$RUNG.jsonl --out $OUT/fast8k_stats_$RUNG.json || exit 1
done
echo "=== DONE $TASK ==="
