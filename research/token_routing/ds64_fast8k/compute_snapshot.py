"""Snapshot of the FLOP-drop jobs running or queued right now, for the hub's experiment log.

Reads `ops status --json` (one cached call), keeps the FLOP-drop experiments, adds GPUs per job (Beaker job spec /
slurm AllocTRES), and tags each job with the experiment-log entry whose `match` substrings appear in its name.

    python research/token_routing/ds64_fast8k/compute_snapshot.py --out research/token_routing/ds64_fast8k/compute_now.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
EXPS = {"tokdrop_lr", "ds64_fast8k", "learned_router", "mt11_64k", "olmo3_cptdrop", "grad_composition"}
HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, timeout=120):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def beaker_gpus(job_id: str):
    out = sh(["beaker", "job", "get", job_id, "--format", "json"], timeout=60)
    try:
        d = json.loads(out)
        d = d[0] if isinstance(d, list) else d
        return int(d.get("requests", {}).get("gpuCount") or d["execution"]["spec"]["resources"]["gpuCount"])
    except Exception:
        return None


def slurm_gpus(job_id: str):
    out = sh(["scontrol", "show", "job", str(job_id)], timeout=30)
    m = re.search(r"gres/gpu(?::[^=]*)?=(\d+)", out) or re.search(r"gpu:(?:[A-Za-z0-9]+:)?(\d+)", out)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "compute_now.json"))
    ap.add_argument("--log", default=os.path.join(HERE, "experiment_log.json"))
    a = ap.parse_args()
    os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")
    st = json.loads(sh([OPS, "status", "--json"], timeout=180) or "{}")
    log = json.load(open(a.log))
    jobs = []
    for r in st.get("rows", []):
        if r.get("experiment") not in EXPS or r.get("state") not in ("RUNNING", "PENDING"):
            continue
        name = r.get("name") or r.get("run_id") or ""
        cl = r.get("cluster")
        g = beaker_gpus(r["job_id"]) if cl == "beaker" else slurm_gpus(r["job_id"]) if cl == "slurm" else None
        entry = next((e["title"] for e in log if any(m in name for m in e.get("match", []))), None)
        jobs.append({"name": name, "cluster": "Beaker (jupiter)" if cl == "beaker" else
                     ("jsteinhardt / berkeleynlp slurm" if cl == "slurm" else cl), "node": r.get("node"),
                     "state": r["state"], "gpus": g, "elapsed_min": round((r.get("elapsed_s") or r.get("pending_s") or 0) / 60),
                     "entry": entry})
    snap = {"generated_at": st.get("generated_at"), "jobs": jobs}
    json.dump(snap, open(a.out, "w"), indent=1)
    run = [j for j in jobs if j["state"] == "RUNNING"]
    print(f"{len(run)} running ({sum(j['gpus'] or 0 for j in run)} GPUs), {len(jobs) - len(run)} queued -> {a.out}")


if __name__ == "__main__":
    main()
