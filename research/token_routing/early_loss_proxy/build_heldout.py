"""Build the fixed held-out probe rows for the early dense-loss proxy (records/early-dense-loss-proxy.md).

Two kinds of rows, every one tokenized exactly like the training shards (``convert_unified_to_document_landmark.py
--emit dense``, qwen3_5 markers, query AFTER, no CoT, documents wrapped; oolong chunked by line):

- ``suite8k`` / ``suite2k``: the FIRST rows of the 4-task eval rung files the final score is measured on (the same
  files and order as ``launch_fast8k.RUNGS``, node-local copies under /data/prasann/f8k_eval/eval), with the gold
  answer as the loss span. These are a subset of the rows the 8k / 2k eval scores.
- ``train``: training-distribution rows the runs never see: rows 1952.. of each task of ``mt4f3904`` (whose first
  1952 rows per task ARE ``mt4f1952``, byte for byte; checked here).

Overlap check: every held-out row's prompt (tokens before the answer) and its question span (tokens after the last
document marker) are hashed and looked up among all 7,808 training rows of ``mt4f1952``.

    $PY research/token_routing/early_loss_proxy/build_heldout.py --out /data/prasann/early_loss_proxy/heldout_v1.npz
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EV = "/data/prasann/f8k_eval/eval"
RUNGS = {
    "outlier": {"8k": f"{EV}/outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                "2k": f"{EV}/outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl"},
    "nq": {"8k": f"{EV}/outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
           "2k": f"{EV}/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl"},
    "contradiction": {"8k": f"{EV}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
                      "2k": f"{EV}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"},
    "oolong": {"8k": f"{EV}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
               "2k": f"{EV}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl"},
}
#: converter --task / --chunk-by, as the training source shards (gradconc/shards/<task>_fresh/metadata.json)
CONV = {"contradiction": ("contradiction", "document"), "nq": ("retrieval", "document"),
        "outlier": ("outlier", "document"), "oolong": ("oolong", "line")}
TASKS = ["contradiction", "nq", "outlier", "oolong"]  # = the t0..t3 order of the mt4 shards (metadata merged_from)
TOKENIZER = "/data/prasann/hf_models/Qwen3.5-0.8B-Base"
TRAIN = "/data/prasann/oracle/shards/mt4f1952"
TRAIN_X = "/data/prasann/oracle/shards/mt4f3904"
EOS, DOC_START, DOC_END = 248044, 248049, 248050


def split_rows(ids: np.ndarray, mask: np.ndarray):
    """Rows of a raw shard (each ends in EOS); rows without loss tokens are skipped, as the trainer does."""
    ends = np.nonzero(ids == EOS)[0]
    out, s = [], 0
    for e in ends:
        r, m = ids[s:e + 1], mask[s:e + 1]
        s = e + 1
        if m.any():
            out.append((np.asarray(r, dtype=np.int64), np.asarray(m, dtype=bool)))
    return out


def load_shard_rows(d: str):
    meta = json.load(open(f"{d}/metadata.json"))
    rows = []
    for f in sorted(glob.glob(f"{d}/token_ids_part_*.npy")):
        mf = f.replace("token_ids_part_", "labels_mask_part_")
        mf = mf if os.path.exists(mf) else f.replace("token_ids_part_", "labels_mask_")
        ids = np.memmap(f, dtype=np.dtype(meta["dtype"]), mode="r")
        mask = np.memmap(mf, dtype=np.dtype(meta.get("mask_dtype", "bool")), mode="r")
        rows += split_rows(ids, mask)
    return rows


def keys(row: np.ndarray, mask: np.ndarray):
    a0 = int(np.nonzero(mask)[0][0])
    prompt = row[:a0]
    last_end = np.nonzero(prompt == DOC_END)[0]
    q = prompt[int(last_end[-1]) + 1:] if len(last_end) else prompt[-64:]
    h = lambda x: hashlib.sha1(np.ascontiguousarray(x, dtype=np.int64).tobytes()).hexdigest()  # noqa: E731
    return h(prompt), h(q)


def convert(task: str, jsonl: str, n: int, out_dir: str):
    conv, chunk = CONV[task]
    head = f"{out_dir}/head.jsonl"
    with open(jsonl) as f, open(head, "w") as g:
        for i, line in enumerate(f):
            if i >= n:
                break
            g.write(line)
    cmd = [sys.executable, "recipes/data/convert_unified_to_document_landmark.py", "--input-jsonl", head,
           "--task", conv, "--out-dir", out_dir, "--emit", "dense", "--marker-set", "qwen3_5", "--tokenizer", TOKENIZER,
           "--seq-len", "65536", "--query-position", "after", "--cot-mode", "none", "--chunk-by", chunk,
           "--num-proc", "4"]
    subprocess.run(cmd, check=True, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src",
                                                         TOKENIZERS_PARALLELISM="false"))
    return load_shard_rows(out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=128, help="rows per (task, set) stored (the probe uses a prefix)")
    a = ap.parse_args()

    # mt4f1952 must be the per-task prefix of mt4f3904 (so rows 1952.. of f3904 were never trained on)
    m = json.load(open(f"{TRAIN}/metadata.json"))
    for t in range(4):
        f1, f2 = f"{TRAIN}/token_ids_part_t{t}_000.npy", f"{TRAIN_X}/token_ids_part_t{t}_000.npy"
        n1 = os.path.getsize(f1)
        with open(f1, "rb") as x, open(f2, "rb") as y:
            assert x.read() == y.read(n1), f"{f2} does not start with {f1}"
    train_keys = {"prompt": set(), "q": set()}
    n_train = 0
    for t in range(4):
        ids = np.memmap(f"{TRAIN}/token_ids_part_t{t}_000.npy", dtype=np.dtype(m["dtype"]), mode="r")
        mk = np.memmap(f"{TRAIN}/labels_mask_t{t}_000.npy", dtype=np.dtype(m.get("mask_dtype", "bool")), mode="r")
        for r, mm in split_rows(ids, mk):
            p, q = keys(r, mm)
            train_keys["prompt"].add(p)
            train_keys["q"].add(q)
            n_train += 1
    print(f"[heldout] {n_train} training rows hashed", flush=True)

    ids_all, mask_all, offs, sets, tasks, src_idx = [], [], [0], [], [], []
    stats = {"train_rows_hashed": n_train, "sets": {}}

    def add(set_name, task, rows, idx0=0):
        hits_p = hits_q = 0
        for i, (r, mm) in enumerate(rows[:a.rows]):
            p, q = keys(r, mm)
            hits_p += p in train_keys["prompt"]
            hits_q += q in train_keys["q"]
            ids_all.append(r.astype(np.uint32))
            mask_all.append(mm)
            offs.append(offs[-1] + len(r))
            sets.append(set_name)
            tasks.append(task)
            src_idx.append(idx0 + i)
        n = min(len(rows), a.rows)
        lens = [len(r) for r, _ in rows[:a.rows]]
        stats["sets"][f"{set_name}/{task}"] = {
            "rows": n, "len_mean": float(np.mean(lens)), "len_max": int(max(lens)),
            "ans_tokens_mean": float(np.mean([mm.sum() for _, mm in rows[:a.rows]])),
            "prompt_in_train": hits_p, "question_in_train": hits_q}
        print(f"[heldout] {set_name}/{task}: {n} rows, len mean {np.mean(lens):.0f} max {max(lens)}, "
              f"prompt in train {hits_p}, question in train {hits_q}", flush=True)

    with tempfile.TemporaryDirectory(dir="/data/prasann/tmp") as tmp:
        for rung in ("8k", "2k"):
            for task in TASKS:
                d = f"{tmp}/{task}_{rung}"
                os.makedirs(d)
                add(f"suite{rung}", task, convert(task, RUNGS[task][rung], a.rows, d))
    mx = json.load(open(f"{TRAIN_X}/metadata.json"))
    for t, task in enumerate(TASKS):
        ids = np.memmap(f"{TRAIN_X}/token_ids_part_t{t}_000.npy", dtype=np.dtype(mx["dtype"]), mode="r")
        mk = np.memmap(f"{TRAIN_X}/labels_mask_t{t}_000.npy", dtype=np.dtype(mx.get("mask_dtype", "bool")), mode="r")
        rows = split_rows(ids, mk)
        assert len(rows) >= 1952 + a.rows, (task, len(rows))
        add("train", task, rows[1952:], idx0=1952)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, ids=np.concatenate(ids_all), mask=np.concatenate(mask_all), offsets=np.asarray(offs),
             sets=np.asarray(sets), tasks=np.asarray(tasks), src_idx=np.asarray(src_idx))
    json.dump(stats, open(a.out.replace(".npz", "_stats.json"), "w"), indent=1)
    print(f"[heldout] wrote {a.out} ({len(sets)} rows)", flush=True)


if __name__ == "__main__":
    main()
