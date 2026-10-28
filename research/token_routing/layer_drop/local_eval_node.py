"""Score fast8k mt4 checkpoints on a Berkeley node while the Beaker eval queue is backed up (2026-10-28).

Runs INSIDE a slurm allocation of NGPU GPUs. Checkpoints come from S3 (ckpts_to_s3_beaker.sh writes <run>/_SYNCED once
a run's export is there); each (run, task, rung) cell is one 1-GPU eval -- the same runner and arguments as the Beaker
evals (run_beaker_multirung_eval.sh: docchunk, query position after, no-cot, 500 rows, max length 20000), with the
batched greedy decode Lambda used (validated equal to bs=1). Several nodes share one claim directory on NFS (an atomic
mkdir per cell), so no cell is scored twice. A cell's log lands in <claims>/<cell>.log; the login-side recorder parses
its `[ladder:...]` lines into ops results.

    python local_eval_node.py --root /data/prasann/lsk_eval --code <code dir> --claims <NFS dir> --runs <file> --ngpu 4
"""

import argparse
import json
import os
import subprocess
import sys
import time

BUCKET = "ai2-llm"
CKPT_PREFIX = "checkpoints/prasanns/_transfer/layerskip/ckpts"
EVAL_PREFIX = "checkpoints/prasanns/_transfer/tokdrop_lr/eval"
TASKS = {  # task -> (ladder key, {rung: eval file under EVAL_PREFIX})
    "contradiction": ("contra", {
        "8k": "_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
        "2k": "_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"}),
    "nq": ("nq", {"8k": "outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
                  "2k": "outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl"}),
    "outlier": ("outlier", {"8k": "outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                            "2k": "outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl"}),
    "oolong": ("oolong", {"8k": "_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
                          "2k": "_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl"}),
}
CELLS = [(t, r) for r in ("8k", "2k") for t in ("contradiction", "nq", "outlier", "oolong")]


def s3():
    import boto3

    return boto3.Session(profile_name="S3").client("s3")


def download_prefix(c, prefix, dst):
    pag = c.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=prefix + "/"):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix) + 1:]
            out = os.path.join(dst, rel)
            if os.path.exists(out) and os.path.getsize(out) == o["Size"]:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            c.download_file(BUCKET, o["Key"], out + ".part")
            os.replace(out + ".part", out)


def synced(c, run):
    try:
        c.head_object(Bucket=BUCKET, Key=f"{CKPT_PREFIX}/{run}/_SYNCED")
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--runs", required=True, help="file: run names in priority order (re-read every loop)")
    ap.add_argument("--ngpu", type=int, required=True)
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--env-bin", default=os.path.dirname(sys.executable))
    a = ap.parse_args()
    c = s3()
    os.makedirs(a.claims, exist_ok=True)
    ev = os.path.join(a.root, "eval")
    for t, (_, rungs) in TASKS.items():
        for f in rungs.values():
            out = os.path.join(ev, f)
            if not os.path.exists(out):
                os.makedirs(os.path.dirname(out), exist_ok=True)
                c.download_file(BUCKET, f"{EVAL_PREFIX}/{f}", out)
    # torchrun's shebang names the env's ORIGINAL prefix; a wrapper keeps a relocated env (balrog /tmp) working
    wrap = os.path.join(a.root, "bin")
    os.makedirs(wrap, exist_ok=True)
    with open(os.path.join(wrap, "torchrun"), "w") as f:
        f.write(f"#!/bin/bash\nexec {a.env_bin}/python -m torch.distributed.run \"$@\"\n")
    os.chmod(os.path.join(wrap, "torchrun"), 0o755)
    os.makedirs(os.path.join(a.root, "weka_stub/checkpoints/prasanns/_eval_bundle"), exist_ok=True)
    slots = {g: None for g in range(a.ngpu)}  # gpu -> (Popen, cell)
    host = os.uname().nodename
    end = time.time() + a.hours * 3600
    have_ckpt = set()
    print(f"[local-eval] {host}: {a.ngpu} GPUs, root {a.root}", flush=True)
    while time.time() < end:
        for g, cur in list(slots.items()):
            if cur is not None and cur[0].poll() is not None:
                p, cell = cur
                done = os.path.join(a.claims, cell + (".done" if p.returncode == 0 else ".failed"))
                open(done, "w").write(f"{host} rc={p.returncode} {time.strftime('%F %T')}\n")
                print(f"[local-eval] {cell} rc={p.returncode}", flush=True)
                slots[g] = None
        runs = [r.strip() for r in open(a.runs).read().split() if r.strip()]
        pending = 0
        for run in runs:
            todo = [(t, r) for t, r in CELLS
                    if not os.path.exists(os.path.join(a.claims, f"{run}__{t}__{r}"))]
            if not todo:
                continue
            pending += len(todo)
            free_now = sum(1 for cur in slots.values() if cur is None)
            # download only when a GPU is free, or to prefetch ONE run ahead (an 18G export takes ~2.5 min, and the
            # loop does not refill finished slots while it downloads)
            staged = [x for x in have_ckpt if any(not os.path.exists(os.path.join(a.claims, f"{x}__{t}__{r}"))
                                                  for t, r in CELLS)]
            if run not in have_ckpt and (free_now > 0 or not staged):
                if not synced(c, run):
                    continue
                print(f"[local-eval] downloading {run}", flush=True)
                download_prefix(c, f"{CKPT_PREFIX}/{run}", os.path.join(a.root, "ckpts", run))
                have_ckpt.add(run)
            if run not in have_ckpt:
                continue
            for t, r in todo:
                free = [g for g, cur in slots.items() if cur is None]
                if not free:
                    break
                cell = f"{run}__{t}__{r}"
                try:
                    os.mkdir(os.path.join(a.claims, cell))  # atomic claim across nodes
                except FileExistsError:
                    continue
                g = free[0]
                key, rungs = TASKS[t]
                out = os.path.join(a.root, "out", run, f"{t}_{r}")
                os.makedirs(out, exist_ok=True)
                env = dict(os.environ)
                env.update({
                    "PATH": f"{wrap}:{a.env_bin}:{env.get('PATH', '')}", "CUDA_VISIBLE_DEVICES": str(g),
                    "PYTHONPATH": os.path.join(a.code, "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "WANDB_MODE": "disabled", "RUN": run, "TASK": key, "VARIANT": "docchunk", "STEP": "",
                    "CKPT": os.path.join(a.root, "ckpts", run), "EVAL_OUT_DIR": out, "PROMPT_FORMAT": "chat",
                    "QUERY_POSITION": "after", "MAX_TEST": "500", "MAX_LENGTH": "20000", "BATCH_SIZE": "8",
                    "NGPU": "1", "LADDER_XLONG": "0", "COT_MODE": "none", "LADDER_VERSION": "v2",
                    "TOKENIZER": "Qwen/Qwen3.5-0.8B-Base",
                    "DC_RUNG_FILES": json.dumps({t: {r: os.path.join(ev, rungs[r])}}), "DC_RUNGS": r,
                    "WEKA_LLM": os.path.join(a.root, "weka_stub"), "RESULTS": os.path.join(a.root, "eval_results"),
                    "REPO": a.code, "LADDER_GEN_BATCH": "32", "LADDER_GEN_BATCH_TOKENS": "196608",
                    "PORT": str(29600 + 7 * g),
                })
                log = open(os.path.join(a.claims, cell + ".log"), "w")
                p = subprocess.Popen(["bash", "recipes/eval/legacy_native/run_beaker_multirung_eval.sh"],
                                     cwd=a.code, env=env, stdout=log, stderr=subprocess.STDOUT)
                slots[g] = (p, cell)
                print(f"[local-eval] start {cell} on gpu {g}", flush=True)
        if pending == 0 and all(v is None for v in slots.values()):
            print("[local-eval] every listed cell is claimed and nothing is running: done", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
