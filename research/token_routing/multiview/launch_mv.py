"""Launch / evaluate MULTI-VIEW runs (``--st-otr-views k``, olmo_core.nn.multiview) on Beaker jupiter.

Same geometry and flags as the recipe's fresh-pool runs (``f8k-mt4-lora64all32r20to7-f<rows>-tde-lr1e-05``: 2 GPUs,
unpacked, seq 12288, 16 rows/step, micro 2, LR 1e-5, base q35-4b-base-markerfix, whole-model LoRA r64 + two-stage
online router 7:0.2,16:0.07, no full-context steps), except RANDOM token choice (``--st-otr-score random``, which ties
the router: 0.914 vs 0.915) and k views per batch. Evals: the four mt4 tasks, 8k + 2k rungs x 500 rows, no-cot, query
after -- the fast8k eval command (``launch_fast8k.eval_cmd``), one job per task, ``<run>-ev<task>``.

    python research/token_routing/multiview/launch_mv.py --views 2,3 --rows 1952,3904 train
    python research/token_routing/multiview/launch_mv.py eval        # every finished run not yet scored
    python research/token_routing/multiview/launch_mv.py status
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
SUFFIX = os.environ.get("MV_SUFFIX", "-tdmv")
CLUSTER = os.environ.get("MV_CLUSTER", "ai2/jupiter-cirrascale-2")
LR = "1e-05"
RECIPE_RND = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
              "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
              "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
              "--st-otr-early-lora-blocks 32 --st-otr-score random --attn-backend flash_2")
#: recipe f1952 ran ~200 min on 2 H100s; per-row cost 0.10 + 0.12k vs the recipe's 0.225
MIN_PER_ROW = 200.0 / 1952 / 0.225


def run_name(k: int, rows: int) -> str:
    return f"f8k-mt4-lora64all32rnd20to7mv{k}-f{rows}{SUFFIX}-lr{LR}"


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def do_train(a):
    st = load()
    for rows in [int(x) for x in a.rows.split(",")]:
        for k in [int(x) for x in a.views.split(",")]:
            name = run_name(k, rows)
            if name in st["runs"] and not a.force:
                print(f"[skip] {name} ({st['runs'][name]['ex']})")
                continue
            exp_min = int(MIN_PER_ROW * rows * (0.10 + 0.12 * k) * 1.2)
            cmd = [sys.executable, "-u", LAUNCHER, "--task", "mt4", "--variant", "softtoken", "--model-family", "qwen3_5",
                   "--model-scale", "4b", "--data-root", f"{SHARDS}/mt4f{rows}", "--run-name", name, "--exact-run-name",
                   "--num-nodes", "1", "--num-gpus", "2", "--epochs", "1", "--lr", LR, "--cluster", CLUSTER,
                   "--wandb-group", "f8k-q35-4b", "--priority", "urgent", "--no-follow", "--no-compile",
                   "--experiment", "tokdrop_lr", "--expected-minutes", str(exp_min),
                   "--seq-len", "12288", "--global-batch", "16", "--micro-batch-instances", "2",
                   "--base-checkpoint", BASE,
                   "--extra-args", f"{RECIPE_RND} --st-otr-views {k}", "launch"]
            print(shlex.join(cmd), flush=True)
            if a.dry_run:
                continue
            rc, out = sh(cmd)
            ex = parse_id(out)
            os.makedirs(f"{HERE}/launch_logs", exist_ok=True)
            open(f"{HERE}/launch_logs/{name}.train.log", "w").write(out)
            print(f"  -> rc={rc} ex={ex}: {out.strip().splitlines()[-1][:200] if out.strip() else ''}", flush=True)
            st["runs"][name] = {"ex": ex, "views": k, "rows": rows, "steps_expected": rows * 4 // 16,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "status"])
    ap.add_argument("--views", default="2,3")
    ap.add_argument("--rows", default="1952,3904")
    ap.add_argument("--runs", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"train": do_train, "eval": do_eval, "status": do_status}[a.mode](a)


if __name__ == "__main__":
    main()
