"""Launch probe.py on Beaker (jupiter, 1 H100, urgent, minRuntime 1h), one job per checkpoint, and register each.

    python research/token_routing/grad_composition/launch.py --labels base,mt4-st31 [--extra "--rows 32"]
    python research/token_routing/grad_composition/launch.py --fetch     # pull finished results into results/
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
STATE = os.path.join(HERE, os.environ.get("GC_STATE", "state.json"))
RESULTS = os.path.join(HERE, "results")

#: label -> (checkpoint, optimizer steps). mt4 dense = the tokdrop_lr matched-FLOP dense curve (LR 5e-6, 16 rows/step);
#: mt11 dense at LR 2e-6; routers = the online mid-stack router (keep 10%, drop before block 8).
CKPTS = {
    "base": (f"{W}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim", 0),
    "mt4-st31": (f"{CK}/f8k-mt4-dense-s488st31-tdc-lr5e-06", 31),
    "mt4-st61": (f"{CK}/f8k-mt4-dense-s488st61-tdc-lr5e-06", 61),
    "mt4-st122": (f"{CK}/f8k-mt4-dense-s488-tdc-lr5e-06", 122),
    "mt4-st244": (f"{CK}/f8k-mt4-dense-s976-tdc-lr5e-06", 244),
    "mt4-st366": (f"{CK}/f8k-mt4-dense-s1464-tdc-lr5e-06", 366),
    "mt11-st165": (f"{CK}/f8k-mt11-dense-s240-tdc-lr2e-06", 165),
    "mt11-st330": (f"{CK}/f8k-mt11-dense-s480-tdc-lr2e-06", 330),
    "mt11-st495": (f"{CK}/f8k-mt11-dense-s720-tdc-lr2e-06", 495),
    "mt4-otr10M8-st366": (f"{CK}/f8k-mt4-otr10M8-s1464-tdb-lr2e-06", 366),
    "mt11-otr10M8-st495": (f"{CK}/f8k-mt11-otr10M8-s720-tdb-lr2e-06", 495),
    "mt4-st366-lr2e-6": (f"{CK}/f8k-mt4-dense-s1464-tdc-lr2e-06", 366),
    "mt11-otr10M8-st330": (f"{CK}/f8k-mt11-otr10M8-s480-tdb-lr2e-06", 330),
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
        ck, steps = CKPTS[lab]
        extra = f"--ckpt {ck} --label {key} {a.extra}".strip()
        cmd = [PY, "-u", f"{REPO}/research/ffn_compute/flop_scaling/beaker_bench_launch.py", "--cluster",
               "ai2/jupiter-cirrascale-2", "--gpus", "1", "--script", "research/token_routing/grad_composition/probe.py",
               "--extra", extra]
        out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{REPO}/src"},
                             cwd=REPO, timeout=900).stdout
        m = re.search(r"SUBMITTED id=(\S+)", out)
        if not m:
            print(f"{lab}: launch FAILED\n{out[-1500:]}")
            continue
        ex = m.group(1)
        st[key] = {"ex": ex, "ckpt": ck, "steps": steps, "commit": commit, "extra": a.extra}
        save(st)
        subprocess.run([OPS, "register", "--run-id", f"gradcomp-{key}", "--experiment", "grad_composition",
                        "--cluster", "beaker", "--kind", "train", "--job-id", ex, "--expected-minutes",
                        str(a.minutes), "--git", REPO], capture_output=True, text=True)
        print(f"{key} -> {ex}")


def fetch(a):
    st = load()
    os.makedirs(RESULTS, exist_ok=True)
    for lab, d in st.items():
        dst = os.path.join(RESULTS, lab)
        if os.path.exists(os.path.join(dst, "probe.json")) and not a.force:
            continue
        r = subprocess.run([BEAKER, "experiment", "results", d["ex"], "--output", dst], capture_output=True, text=True,
                           timeout=600)
        # results land under <dst>/<task name>/...; flatten
        for root, _, files in os.walk(dst):
            for f in files:
                if root != dst:
                    os.replace(os.path.join(root, f), os.path.join(dst, f))
        ok = os.path.exists(os.path.join(dst, "probe.json"))
        print(f"{lab}: {'ok' if ok else 'not ready / failed: ' + (r.stderr or r.stdout)[-300:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=",".join(CKPTS))
    ap.add_argument("--extra", default="")
    ap.add_argument("--prefix", default="", help="label prefix (e.g. bk-mt4rows-)")
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    fetch(a) if a.fetch else launch(a)


if __name__ == "__main__":
    sys.exit(main())
