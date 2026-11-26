"""Fresh single-length eval rows for the gradient-sparsity-vs-length sweep (tasks of the original concentration
table: contradiction, nq, outlier, oolong).

Same generator and converter as the setA / fast8k builds: ``recipes/data/ctc_sft/ctc_calibrated.py build --split
train`` with documents per row from ``recipes/data/ctc_sft/eval_calibration.json`` (ctc's own extrapolation past
256k), oolong with ``-C min_tokens=<budget>``; then ``convert_unified_to_document_landmark.py --emit dense
--marker-set qwen3_5 --cot-mode none`` (oolong ``--chunk-by line``). The seed is NOT setA's 42 (ctc keys streams by
(seed, split, rung), so seed 42 rows would be the mt11 checkpoint's own training rows): ``--seed 20261117``.
Each cell is tokenized twice, with the query after the documents (the fast8k / mt4 checkpoints' layout) and with
the query on both sides (setA / mt11's layout).

    $PY gen_rows.py gen  --lens 2k,4k,...   (CPU; parallel over cells)
    $PY gen_rows.py tok  --lens ...
Output: $WORK/gen/<task>_<rung>/<task>/train.jsonl, $WORK/tok/<qpos>/<task>_<rung>/ (converter shard dir).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
WORK = os.environ.get("GSL_WORK", "/data/prasann/gradlen")
SEEDS = "/data/prasann/ctc_seed_pools"
TOKENIZER = "/data/prasann/hf_models/Qwen3.5-0.8B-Base"
TASKS = ["contradiction", "nq", "outlier", "oolong"]
SPEC = {"contradiction": "contradiction", "nq": "retrieval", "outlier": "outlier", "oolong": "oolong"}
CHUNK_BY = {"oolong": "line"}
POOL = {"contradiction": "auto", "nq": f"{SEEDS}/nq.seed.jsonl.gz", "outlier": f"{SEEDS}/outlier.seed.jsonl.gz",
        "oolong": f"{SEEDS}/oolong.seed.jsonl.gz"}
RUNG_TOKENS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072,
               "256k": 262144, "320k": 327680, "384k": 393216, "448k": 458752, "512k": 524288, "1m": 1048576}
#: rows generated per (task, rung): >= the probe's rows per task plus slack for rows the converter drops
ROWS = {"2k": 160, "4k": 160, "8k": 160, "16k": 160, "32k": 160, "64k": 160, "128k": 48, "256k": 20, "320k": 4, "384k": 4, "448k": 8, "512k": 8,
        "1m": 4}
SEED = 20261117


def env():
    return dict(os.environ, PYTHONPATH=f"{REPO}/src:{REPO}/ctc/src", TOKENIZERS_PARALLELISM="false",
                HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONWARNINGS="ignore")


def cal_cell(task, rung):
    cal = json.load(open(f"{REPO}/recipes/data/ctc_sft/eval_calibration.json"))["tasks"][task]
    return cal.get(rung)


def _gen(job):
    t, r = job
    out = f"{WORK}/gen/{t}_{r}"
    tgt = f"{out}/{t}/train.jsonl"
    if os.path.exists(tgt) and os.path.getsize(tgt):
        return t, r, "cached", 0.0, None
    t0 = time.time()
    c = cal_cell(t, r)
    cmd = [sys.executable, f"{REPO}/recipes/data/ctc_sft/ctc_calibrated.py", "build", "--task", t, "--split", "train",
           "--rungs", r, "--train", str(ROWS[r]), "--seed", str(SEED), "--out", out, "--pool", POOL[t]]
    e = env()
    if c:
        e["CTC_DOCS_OVERRIDE"] = json.dumps({f"{t}:{r}": int(c["n_docs"])})
        for kv in c.get("set", []):
            cmd += ["-C", kv]
    elif t == "oolong":
        cmd += ["-C", f"min_tokens={int(RUNG_TOKENS[r] * 0.89)}"]  # the 256k cell's x0.89-0.93 cap, extrapolated
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    open(f"{out}.log", "w").write(" ".join(cmd) + "\n" + p.stdout[-20000:] + p.stderr[-20000:])
    ok = p.returncode == 0 and os.path.exists(tgt)
    return t, r, "built" if ok else "FAILED " + (p.stdout + p.stderr)[-500:], time.time() - t0, c and c["n_docs"]


def gen(a):
    os.makedirs(f"{WORK}/gen", exist_ok=True)
    jobs = sorted([(t, r) for t in a.tasks.split(",") for r in a.lens.split(",")], key=lambda j: -RUNG_TOKENS[j[1]])
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for t, r, st, sec, nd in ex.map(_gen, jobs):
            print(f"[gen] {t}@{r} n_docs={nd or 'ctc-extrapolated'} {st} ({sec:.0f}s)", flush=True)


def _tok(job):
    t, r, q = job
    src = f"{WORK}/gen/{t}_{r}/{t}/train.jsonl"
    out = f"{WORK}/tok/{q}/{t}_{r}"
    if os.path.exists(f"{out}/metadata.json"):
        return t, r, q, "cached"
    if not os.path.exists(src):
        return t, r, q, "NO SOURCE"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, f"{REPO}/recipes/data/convert_unified_to_document_landmark.py", "--input-jsonl", src,
           "--task", SPEC[t], "--out-dir", out, "--emit", "dense", "--marker-set", "qwen3_5", "--tokenizer",
           TOKENIZER, "--seq-len", "1200000", "--query-position", q, "--cot-mode", "none", "--chunk-by",
           CHUNK_BY.get(t, "document"), "--num-proc", "2"]
    p = subprocess.run(cmd, cwd=REPO, env=env(), capture_output=True, text=True)
    open(f"{out}.log", "w").write(" ".join(cmd) + "\n" + p.stdout[-20000:] + p.stderr[-20000:])
    if p.returncode != 0:
        return t, r, q, "FAILED " + (p.stdout + p.stderr)[-500:]
    m = json.load(open(f"{out}/metadata.json"))
    return t, r, q, (f"{m['num_instances']} rows, len {m['min_example_len']}-{m['max_example_len']}, "
                     f"dropped {m.get('num_dropped')}")


def tok(a):
    jobs = [(t, r, q) for t in a.tasks.split(",") for r in a.lens.split(",") for q in a.qpos.split(",")]
    jobs.sort(key=lambda j: -RUNG_TOKENS[j[1]])
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(_tok, jobs):
            print(f"[tok] {res[0]}@{res[1]} q={res[2]}: {res[3]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "tok"])
    ap.add_argument("--lens", default="2k,4k,8k,16k,32k,64k,128k,256k")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--qpos", default="after,both")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    {"gen": gen, "tok": tok}[a.stage](a)
