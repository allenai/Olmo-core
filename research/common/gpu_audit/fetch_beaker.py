#!/usr/bin/env python3
"""Fetch per-job Beaker timing for every experiment we know of (login-node safe, network only).

IDs come from the ops run registry (Beaker launches since --since) plus the ops Beaker caches,
which also hold jobs that `ops status` saw but nobody registered. Output: one JSON line per Beaker
*job* (an experiment has one job per attempt, so preemptions / retries show up as extra jobs).

  python3 fetch_beaker.py --since 2026-10-13 --out raw/beaker_jobs.jsonl
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

STATE = Path("/accounts/projects/berkeleynlp/prasann/ops_state")
BEAKER = str(Path.home() / ".local/bin/beaker")


def known_ids(since):
    ids = set()
    for line in open(STATE / "runs.jsonl"):
        d = json.loads(line)
        if d.get("cluster") == "beaker" and d.get("job_id") and (d.get("launched_at") or d.get("ts", "")) >= since:
            ids.add(d["job_id"])
    cache = STATE / "cache"
    for k in json.load(open(cache / "beaker_names.json")):
        ids.add(k)
    for v in json.load(open(cache / "beaker_seen.json")).values():
        if isinstance(v, dict) and v.get("exp_id"):
            ids.add(v["exp_id"])
    for v in json.load(open(cache / "beaker_final.json")).values():
        if isinstance(v, dict) and v.get("exp_id"):
            ids.add(v["exp_id"])
    return sorted(ids)


def fetch(batch):
    r = subprocess.run([BEAKER, "experiment", "get", *batch, "--format", "json"], capture_output=True, text=True, timeout=180)
    if r.returncode == 0:
        return json.loads(r.stdout)
    if len(batch) == 1:  # a job id or a deleted experiment: skip
        return []
    mid = len(batch) // 2
    return fetch(batch[:mid]) + fetch(batch[mid:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-13")
    ap.add_argument("--out", default="raw/beaker_jobs.jsonl")
    ap.add_argument("--batch", type=int, default=40)
    a = ap.parse_args()
    ids = known_ids(a.since)
    print(f"{len(ids)} candidate ids", file=sys.stderr)
    seen = set()
    with open(a.out, "w") as f:
        for i in range(0, len(ids), a.batch):
            for e in fetch(ids[i:i + a.batch]):
                if e["id"] in seen:
                    continue
                seen.add(e["id"])
                for attempt, j in enumerate(e.get("jobs", [])):
                    spec = (j.get("execution") or {}).get("spec") or {}
                    ctx = spec.get("context") or {}
                    st = j.get("status") or {}
                    f.write(json.dumps({
                        "exp_id": e["id"], "exp_name": e.get("name"), "exp_created": e.get("created"),
                        "job_id": j.get("id"), "attempt": attempt, "n_attempts": len(e["jobs"]),
                        "cluster": (spec.get("constraints") or {}).get("cluster"),
                        "node": j.get("node"), "gpus": (spec.get("resources") or {}).get("gpuCount") or 0,
                        "priority": ctx.get("priority"), "min_runtime_s": (ctx.get("minRuntime") or 0) / 1e9,
                        "preemptible": ctx.get("preemptible"), "auto_resume": ctx.get("autoResume"),
                        "created": st.get("created"), "scheduled": st.get("scheduled"), "started": st.get("started"),
                        "exited": st.get("exited"), "finalized": st.get("finalized"), "canceled": st.get("canceled"),
                        "canceled_for": st.get("canceledFor"), "canceled_code": st.get("canceledCode"),
                        "exit_code": st.get("exitCode"), "failed": st.get("failed"), "message": st.get("message"),
                    }) + "\n")
            if (i // a.batch) % 10 == 0:
                print(f"{i + a.batch}/{len(ids)} ids, {len(seen)} experiments", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
