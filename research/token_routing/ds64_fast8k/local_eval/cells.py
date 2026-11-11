#!/usr/bin/env python3
"""Cell bookkeeping for the local (Berkeley slurm) fast8k 4-task eval (run_eval_local.sbatch). Stdlib only.

A *cell* is one (task, rung) of the eval the Beaker ``ev-*`` jobs of ``launch_fast8k.py eval`` run: contradiction,
nq, outlier and oolong at the 8k and 2k rungs, 500 rows each, query position ``after``, no-cot, docchunk ladder,
dense emitter, bs=1 greedy decode. The job runs ``recipes/eval/legacy_native/run_beaker_multirung_eval.sh`` (the
same on-node runner Beaker and Lambda use) once per cell, so a requeued job skips the cells it already finished.

    cells.py plan     --work W [--tasks ...]       # missing cells, 8k first: "contradiction 8k" per line
    cells.py merge    --work W                     # cells/<task>_<rung>/ -> eval/<key>_multirung.json (Beaker layout)
    cells.py rows     --work W --run R --run-id ID --job-id J [--experiment E] [--out rows.jsonl]
    cells.py data     --root D                     # eval rows from S3 (md5-checked against MANIFEST.json)
    cells.py upload   --root D                     # one-off: push the eval rows + MANIFEST.json to S3

``rows`` builds ops result rows from each cell's log with the exact regex and fields
``record_beaker_evals.py`` uses for the Beaker ``ev-*`` logs (train_run / task / rung / metric / eval_size / se /
cot_label), so ``collect_pareto_points._ops_scores``, ``make_pertask_section.py`` and ``ops result add --check``
read local and Beaker evals alike.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys

#: launch_fast8k.RUNGS, relative to the eval root (the weka ai2-llm/checkpoints/prasanns/ tree on Beaker; the
#: S3 hub and the node copy keep the same relative paths). KEEP IN SYNC with launch_fast8k.py.
RUNG_FILES = {
    "contradiction": {"8k": "_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
                      "2k": "_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"},
    "nq": {"8k": "outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
           "2k": "outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl"},
    "outlier": {"8k": "outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                "2k": "outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl"},
    "oolong": {"8k": "_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
               "2k": "_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl"},
}
TASKS = list(RUNG_FILES)
RUNGS = ["8k", "2k"]  # 8k first: the primary metric lands first if the job is preempted
#: launch_fast8k.TASK_KEY: the runner's TASK (and the result file stem) per ladder task
TASK_KEY = {"nq": "nq", "outlier": "outlier", "oolong": "oolong", "contradiction": "contra"}
EVAL_SIZE = 500
BUCKET = "ai2-llm"
S3_EVAL = "checkpoints/prasanns/_transfer/f8k_eval"
#: record_beaker_evals.LINE / KEY_TASK (copied, not imported: the job runs this from the staged code snapshot of
#: whichever branch the caller evaluates, which may predate that file)
LINE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ([a-z0-9_]+)=(-?[0-9.]+) \(n=(\d+)")
KEY_TASK = {"contra": "contradiction"}


def cell_dir(work, task, rung):
    return os.path.join(work, "cells", f"{task}_{rung}")


def cell_json(work, task, rung):
    return os.path.join(cell_dir(work, task, rung), f"{TASK_KEY[task]}_multirung.json")


def cell_result(work, task, rung):
    """(score, eval_size) of a finished cell, else None (missing, unreadable, empty or short)."""
    try:
        d = json.load(open(cell_json(work, task, rung)))
    except (OSError, ValueError):
        return None
    key = f"{task}_{rung}"
    n = ((d.get("_meta") or {}).get("eval_size") or {}).get(key)
    v = d.get(key)
    if v is None or not n or n < EVAL_SIZE:
        return None
    return float(v), int(n)


def plan(work, tasks=TASKS, rungs=RUNGS):
    """Missing (task, rung) cells, rung-major (every 8k cell before any 2k cell)."""
    return [(t, r) for r in rungs for t in tasks if cell_result(work, t, r) is None]


def merge(work, tasks=TASKS, rungs=RUNGS):
    """Write eval/<key>_multirung.json per task from its finished cells, in the layout a two-rung Beaker run writes
    (``{task}_{rung}`` scores, ``_meta.eval_size`` / ``parse_rate`` per key), plus the per-rung responses under
    the Beaker names. Returns {task: [rungs merged]}."""
    out_dir = os.path.join(work, "eval")
    os.makedirs(out_dir, exist_ok=True)
    done = {}
    for t in tasks:
        merged, have = None, []
        for r in rungs:
            if cell_result(work, t, r) is None:
                continue
            d = json.load(open(cell_json(work, t, r)))
            key = f"{t}_{r}"
            if merged is None:
                merged = {k: v for k, v in d.items() if k != "_meta" and not re.match(r".+_\d+k$", k)}
                merged["_meta"] = {k: v for k, v in (d.get("_meta") or {}).items()
                                   if k not in ("eval_size", "parse_rate")}
                merged["_meta"]["eval_size"] = {}
            merged[key] = d[key]
            meta = d.get("_meta") or {}
            merged["_meta"]["eval_size"][key] = meta["eval_size"][key]
            if key in (meta.get("parse_rate") or {}):
                merged["_meta"].setdefault("parse_rate", {})[key] = meta["parse_rate"][key]
            stem = f"{TASK_KEY[t]}_multirung_{t}_{r}_responses.jsonl"
            src = os.path.join(cell_dir(work, t, r), stem)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(out_dir, stem))
            have.append(r)
        if merged is not None:
            with open(os.path.join(out_dir, f"{TASK_KEY[t]}_multirung.json"), "w") as f:
                json.dump(merged, f, indent=2)
            done[t] = have
    return done


def rows(work, run, run_id, job_id, experiment="tokdrop_lr", ckpt=None, node=None, tasks=TASKS, rungs=RUNGS,
         git_commit=None):
    """ops result rows from each finished cell's log -- the fields record_beaker_evals.py writes."""
    out = []
    for t in tasks:
        for r in rungs:
            if cell_result(work, t, r) is None:
                continue
            try:
                log = open(os.path.join(cell_dir(work, t, r), "cell.log"), errors="replace").read()
            except OSError:
                continue
            # file the cell under the job that scored it (a resubmitted job must not re-file earlier jobs' cells)
            try:
                cell_run_id, cell_job = open(os.path.join(cell_dir(work, t, r), "scored_by")).read().split()
            except (OSError, ValueError):
                cell_run_id, cell_job = run_id, job_id
            seen = set()
            for m in LINE.finditer(log):
                lt, rung, metric, v, n = m.groups()
                lt = KEY_TASK.get(lt, lt)
                if (lt, rung) != (t, r) or (lt, rung, metric) in seen:
                    continue
                seen.add((lt, rung, metric))
                v, n = float(v), int(n)
                se = math.sqrt(max(v * (1 - v), 1e-9) / n) if 0 <= v <= 1 else None
                out.append(dict(run_id=cell_run_id, train_run=run, experiment=experiment, task=lt, rung=rung,
                                metric=metric, value=v, eval_size=n, se=se and round(se, 6), cot_label="no-cot",
                                eval_backend="olmo-core-native-docchunk-ladder (slurm)",
                                ckpt_path=f"{node}:{ckpt}" if node and ckpt else ckpt,
                                eval_job_id=f"slurm:{cell_job}", git_commit=git_commit))
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _s3():
    import boto3  # the node env has it; the login node's system python3 does not

    return boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "S3")).client("s3")


def stage_data(root):
    """Copy the eight rung files from the S3 hub to ``root`` unless present with the manifest's md5. Raises on a
    checksum mismatch after download (a corrupt or changed file must not be scored)."""
    s3 = _s3()
    man = json.loads(s3.get_object(Bucket=BUCKET, Key=f"{S3_EVAL}/MANIFEST.json")["Body"].read())
    for rel, info in man["files"].items():
        p = os.path.join(root, rel)
        if os.path.exists(p) and os.path.getsize(p) == info["bytes"] and md5(p) == info["md5"]:
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        s3.download_file(BUCKET, f"{S3_EVAL}/eval/{rel}", p + ".part")
        got = md5(p + ".part")
        if got != info["md5"]:
            raise SystemExit(f"md5 mismatch for {rel}: {got} != {info['md5']}")
        os.replace(p + ".part", p)
        print(f"staged {rel} ({info['bytes'] / 1e6:.1f} MB)", flush=True)
    missing = [rel for t in RUNG_FILES.values() for rel in t.values() if rel not in man["files"]]
    if missing:
        raise SystemExit(f"manifest lacks {missing}")
    return man


def upload(root, source_note):
    s3 = _s3()
    files = {}
    for t in RUNG_FILES.values():
        for rel in t.values():
            p = os.path.join(root, rel)
            files[rel] = {"md5": md5(p), "bytes": os.path.getsize(p), "rows": sum(1 for _ in open(p))}
            s3.upload_file(p, BUCKET, f"{S3_EVAL}/eval/{rel}")
            print(f"up {rel} {files[rel]}", flush=True)
    man = {"files": files, "source": source_note,
           "note": "launch_fast8k.RUNGS (fast8k mt4 eval): weka ai2-llm/checkpoints/prasanns/<rel>"}
    s3.put_object(Bucket=BUCKET, Key=f"{S3_EVAL}/MANIFEST.json", Body=json.dumps(man, indent=1).encode())
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "merge", "rows", "data", "upload"])
    ap.add_argument("--work")
    ap.add_argument("--root")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--rungs", default=",".join(RUNGS))
    ap.add_argument("--run")
    ap.add_argument("--run-id")
    ap.add_argument("--job-id")
    ap.add_argument("--experiment", default="tokdrop_lr")
    ap.add_argument("--ckpt")
    ap.add_argument("--node")
    ap.add_argument("--git-commit")
    ap.add_argument("--out")
    ap.add_argument("--source-note", default="")
    a = ap.parse_args(argv)
    tasks = [t for t in a.tasks.split(",") if t]
    rungs = [r for r in a.rungs.split(",") if r]
    bad = [t for t in tasks if t not in RUNG_FILES] + [r for r in rungs if r not in RUNGS]
    if bad:
        ap.error(f"unknown task/rung {bad} (tasks {TASKS}, rungs {RUNGS})")
    if a.cmd == "plan":  # "task rung runner_key rung_file" per missing cell
        for t, r in plan(a.work, tasks, rungs):
            print(t, r, TASK_KEY[t], os.path.join(a.root or "", RUNG_FILES[t][r]))
    elif a.cmd == "merge":
        print(json.dumps(merge(a.work, tasks, rungs)))
    elif a.cmd == "rows":
        rs = rows(a.work, a.run, a.run_id, a.job_id, a.experiment, a.ckpt, a.node, tasks, rungs, a.git_commit)
        text = "".join(json.dumps(r) + "\n" for r in rs)
        if a.out:
            open(a.out, "w").write(text)
        else:
            sys.stdout.write(text)
        print(f"{len(rs)} rows", file=sys.stderr)
    elif a.cmd == "data":
        stage_data(a.root)
    elif a.cmd == "upload":
        upload(a.root, a.source_note)


if __name__ == "__main__":
    main()
