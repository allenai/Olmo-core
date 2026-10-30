"""Launch routed_probe.py on Beaker (jupiter, 1 GPU each, urgent, minRuntime 1h, ai2/flex2 / ai2/oe-other), one job per
(checkpoint, shard), and register each with ops. Sketches go to weka (GAP_DIR/<label>/b<j>.npz); analyze_gap.py reads them.

    python research/token_routing/grad_concentration/launch_gap.py --labels f1952 --num-shards 8 [--extra "--max-batches 1"]
    python research/token_routing/grad_concentration/launch_gap.py --analyze --labels f1952,s488,f7808   # Beaker CPU-ish job
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
sys.path.insert(0, HERE)
from launch import CKPTS, OPS, PY  # noqa: E402

CKPTS = {**CKPTS, "recipe": (f"/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/"
                             f"f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05", 488, 1952)}

W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
GAP_DIR = f"{W}/gradgap"
STATE = os.path.join(HERE, "state_gap.json")


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(st):
    json.dump(st, open(STATE, "w"), indent=1)


def submit(script, extra, cluster, minutes, run_id, st_key, st, info):
    commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    cmd = [PY, "-u", f"{REPO}/research/ffn_compute/flop_scaling/beaker_bench_launch.py", "--cluster", cluster,
           "--gpus", "1", "--script", script, "--extra", extra, "--min-runtime", "1h", "--priority", "urgent"]
    out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{REPO}/src"},
                         cwd=REPO, timeout=900)
    m = re.search(r"SUBMITTED id=(\S+)", out.stdout)
    if not m:
        print(f"{st_key}: launch FAILED\n{(out.stdout + out.stderr)[-1500:]}")
        return None
    ex = m.group(1)
    st[st_key] = {"ex": ex, "commit": commit, "extra": extra, **info}
    save(st)
    r = subprocess.run([OPS, "register", "--run-id", run_id, "--experiment", "tokdrop_lr", "--cluster", "beaker",
                        "--kind", "eval", "--job-id", ex, "--expected-minutes", str(minutes), "--git", REPO],
                       capture_output=True, text=True)
    print(f"{st_key} -> {ex} (ops register rc={r.returncode} {r.stderr.strip()[-200:]})")
    return ex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="f1952")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--shards-only", default="", help="comma list of shard indices to (re)launch")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--extra", default="")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--no-adam", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--adam-only", action="store_true", help="one job per label: Adam factors -> GAP_DIR/adam_<label>.pt")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    st = load()
    if a.analyze:
        key = f"{a.prefix}analyze"
        extra = f"--gap-dir {GAP_DIR} --labels {a.labels} {a.extra}".strip()
        submit("research/token_routing/grad_concentration/analyze_gap.py", extra, a.cluster, 60,
               f"gradgap-{key}-{a.labels.replace(',', '-')}", key + "-" + a.labels, st, {})
        return
    if a.adam_only:
        for lab in a.labels.split(","):
            ck = CKPTS[lab][0]
            key = f"{a.prefix}adam-{lab}"
            extra = (f"--ckpt {ck} --label {lab} --rows {a.rows} --adam-ckpt {ck} "
                     f"--only-adam-factors {GAP_DIR}/adam_{lab}.pt {a.extra}").strip()
            submit("research/token_routing/grad_concentration/routed_probe.py", extra, a.cluster, 40,
                   f"gradgap-{key}", key, st, {"label": lab})
        return
    for lab in a.labels.split(","):
        ck = CKPTS[lab][0]
        only = [int(x) for x in a.shards_only.split(",")] if a.shards_only else range(a.num_shards)
        for sh in only:
            key = f"{a.prefix}{lab}-s{sh}of{a.num_shards}"
            if key in st and not a.force:
                print(f"[skip] {key} {st[key]['ex']}")
                continue
            adam = "" if a.no_adam else f"--adam-factors {GAP_DIR}/adam_{lab}.pt"
            extra = (f"--ckpt {ck} --label {a.prefix}{lab} --rows {a.rows} --num-shards {a.num_shards} --shard {sh} "
                     f"--save-dir {GAP_DIR} {adam} {a.extra}").strip()
            submit("research/token_routing/grad_concentration/routed_probe.py", extra, a.cluster, a.minutes,
                   f"gradgap-{key}", key, st, {"label": lab, "shard": sh, "num_shards": a.num_shards})


if __name__ == "__main__":
    sys.exit(main())
