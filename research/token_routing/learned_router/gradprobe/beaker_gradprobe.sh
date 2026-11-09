#!/usr/bin/env bash
# One grad_importance.py run inside a gantry job (H100). The code is NOT the gantry clone: launch_beaker.py's inline
# bootstrap writes the AWS creds, installs awscli, downloads a tarball of the working tree from S3 and runs this
# script from inside it, so unpublished code never needs a public push (gantry only clones an already-pushed
# commit to satisfy its pushed-HEAD check).
# Env: CKPT (weka model_and_optim dir), TASK (nq|outlier|contradiction), RUNG (8k|16k|32k), ROWS, TAG,
#      OUT_S3 (prefix for <TAG>.json / <TAG>.log), EXTRA (grad_importance.py flags); AWS_PROFILE=S3 set by the bootstrap.
set -uo pipefail
: "${CKPT:?}" "${TASK:?}" "${RUNG:?}" "${ROWS:?}" "${TAG:?}" "${OUT_S3:?}"
mkdir -p /tmp/gp_out
export PYTHONPATH="$PWD/src:$PWD/research/common" PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEVLOSS_SKIP_CHECKS=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
echo "== gradprobe $TAG on $(hostname) ($(date +%T)) ckpt=$CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# rows: first $ROWS of the public CTC-suite eval rung (the same parquet the local grid rows were cut from)
python - "$TASK" "$RUNG" "$ROWS" <<'PY' || { echo "!!! row fetch failed"; exit 95; }
import json, os, sys
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
task, rung, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
subset = {"contradiction": "contradiction_iid"}.get(task, task)
tokens = {"2k": 2048, "8k": 8192, "16k": 16384, "32k": 32768}[rung]
p = hf_hub_download("PrasannSinghal/ctc-suite-eval", f"data/{subset}/r{rung}.parquet", repo_type="dataset")
rows = pq.read_table(p).slice(0, max(n, 16)).to_pylist()
os.makedirs(f"/tmp/gp_data/{subset}", exist_ok=True)
with open(f"/tmp/gp_data/{subset}/rung_{tokens}.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, default=str) + "\n")
print(f"fetched {len(rows)} rows of {subset}/r{rung}")
PY
BACKEND=flash_2
python -c "import flash_attn" 2>/dev/null || { echo ".. flash_attn not importable: --attn-backend torch"; BACKEND=torch; }
# flash-attn 2 ships no Blackwell (sm_100) kernels: use torch SDPA on B200 (ai2/titan)
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "B200" && { echo ".. B200: --attn-backend torch"; BACKEND=torch; }
python -u research/token_routing/learned_router/gradprobe/grad_importance.py --task "$TASK" --rung "$RUNG" --rows "$ROWS" \
  --ckpt "$CKPT" --ckpt-format distcp --data-root /tmp/gp_data --attn-backend "$BACKEND" --out "/tmp/gp_out/$TAG.json" ${EXTRA:-} \
  2>&1 | tee "/tmp/gp_out/$TAG.log"
rc=${PIPESTATUS[0]}
aws s3 cp --quiet "/tmp/gp_out/$TAG.log" "$OUT_S3/$TAG.log"
[ -f "/tmp/gp_out/$TAG.json" ] && aws s3 cp --quiet "/tmp/gp_out/$TAG.json" "$OUT_S3/$TAG.json"
echo "== exit $rc ($(date +%T))"
exit $rc
