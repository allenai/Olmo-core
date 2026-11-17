"""Launch probe.py on Beaker (jupiter, 1 GPU, urgent, minRuntime 1h), one job per checkpoint, and register each.

    python research/token_routing/grad_concentration/launch.py --labels base,f1952 [--extra "--rows 8"] [--prefix smoke-]
    python research/token_routing/grad_concentration/launch.py --fetch     # pull finished results into results/
Gantry runs the PUSHED commit: commit and push first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
CK = f"{W}/ctc_suite/ckpts"
PY = "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python"
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
STATE = os.path.join(HERE, "state.json")
RESULTS = os.path.join(HERE, "results")

#: label -> (checkpoint, optimizer steps, rows/task seen). Dense mt4, LR 5e-6, 16 rows/step (4 tasks).
#: s* = original ds64 pool (st31/st61 = the s488 run cut mid-schedule); f* = fresh pools, each its own full schedule.
CKPTS = {
    "base": (f"{W}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim", 0, 0),
    "s-st31": (f"{CK}/f8k-mt4-dense-s488st31-tdc-lr5e-06", 31, 124),
    "s-st61": (f"{CK}/f8k-mt4-dense-s488st61-tdc-lr5e-06", 61, 244),
    "s488": (f"{CK}/f8k-mt4-dense-s488-tdc-lr5e-06", 122, 488),
    "s976": (f"{CK}/f8k-mt4-dense-s976-tdc-lr5e-06", 244, 976),
    "s1464": (f"{CK}/f8k-mt4-dense-s1464-tdc-lr5e-06", 366, 1464),
    "s1952": (f"{CK}/f8k-mt4-dense-s1952-tdc-lr5e-06", 488, 1952),
    "f1952": (f"{CK}/f8k-mt4-dense-f1952-tde-lr5e-06", 488, 1952),
    "f3904": (f"{CK}/f8k-mt4-dense-f3904-tde-lr5e-06", 976, 3904),
    "f5856": (f"{CK}/f8k-mt4-dense-f5856-tde-lr5e-06", 1464, 5856),
    "f7808": (f"{CK}/f8k-mt4-dense-f7808-tde-lr5e-06", 1952, 7808),
}


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(st):
    json.dump(st, open(STATE, "w"), indent=1)


def launch(a):
    st = load()
    commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    for lab in a.labels.split(","):
        key = a.prefix + lab
        if key in st and not a.force:
            print(f"[skip] {key} {st[key]['ex']}")
            continue
        ck, steps, rows = CKPTS[lab]
        extra = f"--ckpt {ck} --label {key} {a.extra}".strip()
        cmd = [PY, "-u", f"{REPO}/research/ffn_compute/flop_scaling/beaker_bench_launch.py", "--cluster", a.cluster,
               "--gpus", "1", "--script", "research/token_routing/grad_concentration/probe.py", "--extra", extra]
        out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{REPO}/src"},
                             cwd=REPO, timeout=900)
        m = re.search(r"SUBMITTED id=(\S+)", out.stdout)
        if not m:
            print(f"{lab}: launch FAILED\n{(out.stdout + out.stderr)[-1500:]}")
            continue
        ex = m.group(1)
        st[key] = {"ex": ex, "ckpt": ck, "steps": steps, "rows_seen": rows, "commit": commit, "extra": a.extra}
        save(st)
        r = subprocess.run([OPS, "register", "--run-id", f"gradconc-{key}", "--experiment", "tokdrop_lr",
                            "--cluster", "beaker", "--kind", "train", "--job-id", ex, "--expected-minutes",
                            str(a.minutes), "--git", REPO], capture_output=True, text=True)
        print(f"{key} -> {ex} (ops register rc={r.returncode} {r.stderr.strip()[-200:]})")


def fetch(a):
    st = load()
    os.makedirs(RESULTS, exist_ok=True)
    for lab, d in st.items():
        if a.labels_given and lab not in a.labels.split(","):
            continue
        dst = os.path.join(RESULTS, lab)
        if os.path.exists(os.path.join(dst, "probe.json")) and not a.force:
            continue
        r = subprocess.run([BEAKER, "experiment", "results", d["ex"], "--output", dst], capture_output=True, text=True,
                           timeout=900)
        for root, _, files in os.walk(dst):  # results land under <dst>/<task name>/...; flatten
            for f in files:
                if root != dst:
                    os.replace(os.path.join(root, f), os.path.join(dst, f))
        ok = os.path.exists(os.path.join(dst, "probe.json"))
        print(f"{lab}: {'ok' if ok else 'not ready / failed: ' + (r.stderr or r.stdout)[-300:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=",".join(CKPTS))
    ap.add_argument("--extra", default="")
    ap.add_argument("--prefix", default="", help="state / run-id prefix (e.g. smoke-)")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    a.labels_given = "--labels" in sys.argv
    fetch(a) if a.fetch else launch(a)


if __name__ == "__main__":
    sys.exit(main())
