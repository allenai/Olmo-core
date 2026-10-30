"""Reconcile the Qwen3 study's Beaker/sneetches twins (each run is launched on both; the first to start wins).

* Beaker twin started (or done) and the local ``-sn`` copy has not started training -> skip file for the local copy
  (``/data/prasann/q3recipe/skip/<run>-sn`` on sneetches; ``run_q3_local.sbatch`` exits on it).
* Local ``-sn`` copy started training and the Beaker twin is still pending -> stop the Beaker twin (+ ops update).

Prints one line per action. Run from the repo root on the login node:

    python3 research/token_routing/ds64_fast8k/q3_twins.py
"""

import glob
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = f"{HERE}/state_q3.json"
LOGS = "/net/sneetches/data/prasann/joblogs"
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
HDR = re.compile(r"=== q3recipe host=\S+ RUN=(\S+) ")
SEEN = f"{HERE}/results_q3/twins_skipped.txt"  # skip files already written (gitignored results dir)


def ops_id(run):
    """The ops registry id of a Beaker twin: 4B runs plain, 1.7B relaunch -bk2, other sizes -bk."""
    return run + ("-bk2" if "-q317b-" in run else "-bk" if re.search(r"-q3\d", run) else "")


def beaker_state(ex):
    try:
        j = json.loads(subprocess.run([BEAKER, "experiment", "get", ex, "--format", "json"], capture_output=True,
                                      text=True, timeout=60).stdout)
    except Exception:
        return "unknown"
    j = j[0] if isinstance(j, list) else j
    st = (j.get("jobs") or [{}])[-1].get("status", {})
    if st.get("canceled"):
        return "canceled"
    if st.get("exited"):
        return "exited"
    return "started" if st.get("started") else "pending"


def local_started():
    """{-sn run: 'training' | 'skipped'} from the run logs on sneetches (read-only over /net)."""
    out = {}
    for path in glob.glob(f"{LOGS}/q3recipe_*.log"):
        try:
            txt = open(path, errors="replace").read(200000)
        except OSError:
            continue
        m = HDR.search(txt)
        if m:
            out[m.group(1)] = "skipped" if "=== SKIP" in txt else ("training" if "[step=" in txt or "TRAIN rc" in txt
                                                                 or "already scored" in txt else "started")
    return out


def main():
    st = json.load(open(STATE))
    loc = local_started()
    seen = set(open(SEEN).read().split()) if os.path.exists(SEEN) else set()
    skips = []
    for run, r in st["runs"].items():
        if "st4" in run or not r.get("ex") or r.get("state") in ("FAILED",):
            continue
        sn = f"{run}-sn"
        b = beaker_state(r["ex"])
        if b in ("started", "exited") and loc.get(sn) not in ("training", "skipped") and sn not in seen:
            skips.append(sn)
        elif b == "pending" and loc.get(sn) == "training":
            subprocess.run([BEAKER, "experiment", "stop", r["ex"]], capture_output=True, timeout=60)
            subprocess.run([OPS, "update", ops_id(run), "final_state=cancelled",
                            "notes=local -sn twin started first"], capture_output=True, timeout=60)
            print(f"TWIN cancelled Beaker {run} ({r['ex']}): local twin training")
    if skips:
        cmd = "cd /data/prasann/q3recipe/skip && touch " + " ".join(skips)
        rc = subprocess.run(["srun", "--partition=jsteinhardt", "--qos=preemptive_high", "--account=site",
                             "--nodelist=sneetches", "--cpus-per-task=1", "--mem=1G", "--time=00:02:00",
                             "bash", "-c", cmd], capture_output=True, text=True, timeout=600).returncode
        for s in skips:
            print(f"TWIN skip {s} (Beaker twin started first){'' if rc == 0 else ' -- srun FAILED'}")
        if rc == 0:
            os.makedirs(os.path.dirname(SEEN), exist_ok=True)
            open(SEEN, "a").write("".join(f"{s}\n" for s in skips))


if __name__ == "__main__":
    main()
