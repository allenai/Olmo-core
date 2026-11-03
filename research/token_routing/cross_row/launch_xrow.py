"""Launch / evaluate the CROSS-ROW allocation arms (tokdrop_lr, 2026-11-02) on Beaker jupiter.

Every arm is the token-drop recipe (``f8k-mt4-lora64all32r20to7-f<rows>-tde-lr1e-05``: whole-model LoRA r64 + the
two-stage online router 7:0.2,16:0.07, unpacked, seq 12288, 16 rows/step, LR 1e-5, base q35-4b-base-markerfix) with
ONE change:

- ``xstep``   : ``--st-otr-keep-mode step`` -- one router-score threshold per optimizer step shared by all 16 rows (both
                ranks), an integral controller holding the run-average keep at 20% / 7% (so FLOPs match the recipe),
                per-row cap 3x budget, per-document floor 2 tokens. 2 GPUs, micro-batch 2 (the recipe geometry).
- ``sbptop30``: selective backprop, ``top`` mode, q 0.3: every row runs the routed forward; only rows whose answer loss
                is in the top 30% of a running window back-propagate. 1 GPU, micro-batch 1 (= one decision per row).
- ``sbpsmp30``: the same at q 0.3 in ``sample`` mode (loss-proportional, 1/p weights: unbiased).

Evals: the four mt4 tasks, 8k + 2k rungs x 500 rows, no-cot, query after (``launch_fast8k.eval_cmd``), ``<run>-ev<task>``.

    python research/token_routing/cross_row/launch_xrow.py --arms xstep,sbptop30 --rows 1952 --seeds 1,2 train
    python research/token_routing/cross_row/launch_xrow.py --arms xstep --rows 1952 --seeds 1 --max-steps 6 --smoke train
    python research/token_routing/cross_row/launch_xrow.py eval      # every run marked DONE in state.json, not yet scored
    python research/token_routing/cross_row/launch_xrow.py status
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "ds64_fast8k"))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "ds64_fast2k"))
from launch_fast2k import BASE, CKPTS, W, parse_id, sh  # noqa: E402
from launch_fast8k import MT4_TASKS, eval_cmd  # noqa: E402

LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
SHARDS = f"{W}/ds64/fast8k/shards"
STATE = f"{HERE}/state.json"
SUFFIX = os.environ.get("XROW_SUFFIX", "-tdx")
CLUSTER = os.environ.get("XROW_CLUSTER", "ai2/jupiter-cirrascale-2")
LR = "1e-05"
RECIPE = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
          "--st-otr-early-lora-blocks 32 --attn-backend flash_2")
#: arm -> (extra flags on top of the recipe, GPUs, micro-batch rows, est. minutes per 1,952 rows/task)
ARMS = {
    "xstep": ("--st-otr-keep-mode step --st-otr-row-cap 3.0 --st-otr-min-per-doc 2", 2, 2, 210),
    "sbptop30": ("--st-otr-sbp-mode top --st-otr-sbp-q 0.3", 1, 1, 260),
    "sbptop50": ("--st-otr-sbp-mode top --st-otr-sbp-q 0.5", 1, 1, 300),
    "sbpsmp30": ("--st-otr-sbp-mode sample --st-otr-sbp-q 0.3", 1, 1, 260),
}


def run_name(arm: str, seed: int, rows: int, smoke: bool = False) -> str:
    return f"f8k-mt4-lora64all32{arm}sd{seed}-f{rows}{SUFFIX}{'smk' if smoke else ''}-lr{LR}"


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def do_train(a):
    st = load()
    for rows in [int(x) for x in a.rows.split(",")]:
        for arm in a.arms.split(","):
            extra, gpus, micro, mins = ARMS[arm]
            for seed in [int(x) for x in a.seeds.split(",")]:
                name = run_name(arm, seed, rows, a.smoke)
                if name in st["runs"] and not a.force:
                    print(f"[skip] {name} ({st['runs'][name]['ex']})")
                    continue
                steps = a.max_steps or rows * 4 // 16
                exp_min = max(30, int(mins * rows / 1952 * steps / (rows * 4 // 16) * 1.2) + 20)
                cmd = [sys.executable, "-u", LAUNCHER, "--task", "mt4", "--variant", "softtoken",
                       "--model-family", "qwen3_5", "--model-scale", "4b", "--data-root", f"{SHARDS}/mt4f{rows}",
                       "--run-name", name, "--exact-run-name", "--num-nodes", "1", "--num-gpus", str(gpus),
                       "--epochs", "1", "--lr", LR, "--cluster", CLUSTER, "--wandb-group", "f8k-q35-4b",
                       "--priority", "urgent", "--no-follow", "--no-compile", "--seed", str(seed),
                       # a 1-GPU selective-backprop run needs ~100 min; with minRuntime 1h jupiter preempted 3 of 4 runs
                       # right after the hour, twice (2026-11-02), restarting them from step 0
                       "--min-runtime", os.environ.get("XROW_MIN_RUNTIME", "1h"),
                       "--experiment", "tokdrop_lr", "--expected-minutes", str(exp_min),
                       "--seq-len", "12288", "--global-batch", "16", "--micro-batch-instances", str(micro),
                       "--base-checkpoint", BASE, "--extra-args", f"{RECIPE} {extra}"]
                if a.max_steps:
                    cmd += ["--max-steps", str(a.max_steps)]
                cmd += ["launch"]
                print(shlex.join(cmd), flush=True)
                if a.dry_run:
                    continue
                rc, out = sh(cmd)
                ex = parse_id(out)
                os.makedirs(f"{HERE}/launch_logs", exist_ok=True)
                open(f"{HERE}/launch_logs/{name}.train.log", "w").write(out)
                print(f"  -> rc={rc} ex={ex}: {out.strip().splitlines()[-1][:200] if out.strip() else ''}", flush=True)
                st["runs"][name] = {"ex": ex, "arm": arm, "seed": seed, "rows": rows, "gpus": gpus, "micro": micro,
                                    "steps_expected": steps, "smoke": a.smoke,
                                    "state": "S" if ex else "LAUNCH-FAILED"}
                save(st)


def do_eval(a):
    st = load()
    names = a.runs.split(",") if a.runs else list(st["runs"])
    for name in names:
        r = st["runs"].get(name)
        if r is None:
            print(f"[skip] {name} not in state")
            continue
        if r.get("state") != "DONE" and not a.force:
            print(f"[wait] {name} train={r.get('state')} (mark DONE in state.json once the export exists)")
            continue
        for task in MT4_TASKS:
            ename = f"{name}-ev{task}"
            if st["evals"].get(ename, {}).get("ex") and not a.force:
                print(f"[skip] {ename} ({st['evals'][ename]['ex']})")
                continue
            cmd = eval_cmd(ename, task, f"{CKPTS}/{name}", False)
            if a.dry_run:
                print(shlex.join(cmd))
                continue
            rc, out = sh(cmd)
            ex = parse_id(out)
            os.makedirs(f"{HERE}/launch_logs", exist_ok=True)
            open(f"{HERE}/launch_logs/{ename}.eval.log", "w").write(out)
            st["evals"][ename] = {"ex": ex, "train_run": name, "task": task, "state": "S" if ex else "LAUNCH-FAILED"}
            save(st)
            print(f"eval {ename} -> {ex or 'FAILED: ' + out[-300:]}", flush=True)


def do_status(a):
    st = load()
    for n, r in st["runs"].items():
        print(f"{n}  ex={r['ex']}  state={r['state']}")
    for n, r in st["evals"].items():
        print(f"  {n}  ex={r['ex']}  state={r['state']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="xstep")
    ap.add_argument("--rows", default="1952")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="name the run ...smk (a short smoke, never evaluated)")
    ap.add_argument("--runs", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("mode", choices=["train", "eval", "status"])
    a = ap.parse_args()
    {"train": do_train, "eval": do_eval, "status": do_status}[a.mode](a)


if __name__ == "__main__":
    main()
