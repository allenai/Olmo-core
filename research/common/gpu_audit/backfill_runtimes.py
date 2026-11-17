#!/usr/bin/env python3
"""Seed the registry's runtime history (``runtime_minutes``) from the audit's per-attempt job table.

The launchers' minRuntime default (ops/opslib/runtime.py) takes the median ``runtime_minutes`` of
similar finished runs. That field only starts being written by ``ops status --record`` from
2026-11-10 on, so the history begins empty and every run falls back to 1 h. This writes one update
event per registered run whose last attempt in ``data/jobs.csv`` COMPLETED (Beaker, slurm and Lambda),
with that attempt's run time. Dry run unless ``--write``; ``--write`` appends to the LIVE shared
registry (``$OPS_STATE_DIR`` overrides), source ``backfill:gpu_audit``, skipping runs that already
have ``runtime_minutes``.

    python3 research/common/gpu_audit/backfill_runtimes.py            # counts only
    python3 research/common/gpu_audit/backfill_runtimes.py --write
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "ops"))
from opslib import registry, safe  # noqa: E402


def candidates(jobs_csv: Path, runs: dict) -> dict:
    last = {}
    for j in csv.DictReader(open(jobs_csv)):
        rid = j.get("run_id")
        if not rid or j.get("registered") != "True":
            continue
        key = (rid, j["job_id"])
        if key not in last or int(j["attempt"] or 0) >= int(last[key]["attempt"] or 0):
            last[key] = j
    out = {}
    for (rid, _), j in last.items():
        r = runs.get(rid)
        if not r or r.get("runtime_minutes") is not None or j["state"] != "COMPLETED":
            continue
        mins = round(float(j["run_full_h"]) * 60, 1)
        if mins > 0:
            out[rid] = mins
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-csv", default=str(HERE / "data" / "jobs.csv"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    path = safe.state_dir() / "runs.jsonl"
    runs = registry.load_runs(path)
    todo = candidates(Path(a.jobs_csv), runs)
    kinds = {}
    for rid in todo:
        k = runs[rid].get("kind")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{len(todo)} runs to seed into {path} (by kind: {kinds})")
    if not a.write:
        print("dry run; pass --write to append the update events")
        return
    rows = [registry.make_update(rid, runtime_minutes=m, state_source="gpu_audit jobs.csv (last attempt COMPLETED)",
                                 source="backfill:gpu_audit") for rid, m in todo.items()]
    n = registry.append_rows(path, rows)
    print(f"appended {n} update events")


if __name__ == "__main__":
    main()
