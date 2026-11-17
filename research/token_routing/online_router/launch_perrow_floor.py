"""Per-row budget + per-document floor on the token-drop recipe (tokdrop_lr, 2026-11-02).

The recipe (``lora64all32r20to7``: whole-model LoRA r64 + online router, all tokens through blocks 0-6, keep 20% of
the body after block 7 and 7% after block 16) gives every DOCUMENT its own share of the budget (``--st-otr-doc-share
1.0``). These arms give the ROW the budget instead (``doc_share 0``: spread over documents by router score), with a
floor of K tokens per document (``--st-otr-min-per-doc K``, counted against the budget, applied at both stages).

Reproduces the recipe's Beaker command exactly (from its launch log: fast8k mt4 fresh pool, seq 12288, 16 rows/step,
micro 2, 2 GPUs, LR 1e-5, repaired Qwen3.5-4B base, wandb group f8k-q35-4b). Run it FROM THE CHECKOUT whose pushed
HEAD should run (gantry ships the cwd's HEAD). Evals: ``ds64_fast8k/launch_fast8k.py`` eval mode reads the
fast8k state.json; this script keeps its own state at ``perrow_floor_state.json`` and prints the eval commands.

    python research/token_routing/online_router/launch_perrow_floor.py --arms ds0f2 --budget f1952 --max-steps 4 launch
    python research/token_routing/online_router/launch_perrow_floor.py --arms ds0,ds0f1,ds0f2 --seed 1 launch
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
LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
SHARDS = f"{W}/ds64/fast8k/shards"
BASE = f"{W}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
STATE = f"{HERE}/perrow_floor_state.json"
SUFFIX = "-tde"
LR = 1e-5

RECIPE = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
          "--st-otr-explore-frac 0.1 --st-otr-doc-share {share} --st-otr-dense-every 0 --st-otr-drop-layer 7 "
          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
          "--st-otr-early-lora-blocks 32{extra} --attn-backend flash_2")

#: short arm -> (fast8k arm name, trainer flags). The recipe itself is re-run only as a seed control.
ARMS = {
    "recipe": ("lora64all32r20to7", RECIPE.format(share="1.0", extra="")),
    "ds0": ("lora64all32r20to7ds0", RECIPE.format(share="0.0", extra="")),
    "ds0f1": ("lora64all32r20to7ds0f1", RECIPE.format(share="0.0", extra=" --st-otr-min-per-doc 1")),
    "ds0f2": ("lora64all32r20to7ds0f2", RECIPE.format(share="0.0", extra=" --st-otr-min-per-doc 2")),
}
EXPECTED_MIN = {"f1952": 200, "f3904": 400}


def run_name(arm, budget, seed, smoke):
    a = ARMS[arm][0] + (f"sd{seed}" if seed else "")
    return f"f8k-mt4-{a}-{budget}{SUFFIX}-lr{LR:g}" + ("-smoke" if smoke else "")


def do_eval(a, st):
    """Score each finished run per task on the 8k + 2k rungs x 500 rows, no-cot, query after: the same command
    ``ds64_fast8k/launch_fast8k.py`` eval mode builds for the recipe runs (``eval_cmd``, mt4 branch)."""
    sys.path.insert(0, f"{REPO}/research/token_routing/ds64_fast8k")
    sys.path.insert(0, f"{REPO}/research/token_routing/ds64_fast2k")
    from launch_fast2k import CKPTS, TOKENIZER, parse_id, sh  # noqa: E402
    from launch_fast8k import EVAL_LAUNCHER, MT4_TASKS, RUNGS, TASK_KEY  # noqa: E402

    names = a.runs.split(",") if a.runs else [n for n, r in st["runs"].items() if not r.get("max_steps")]
    for name in names:
        for task in MT4_TASKS:
            ename = f"{name}-ev{task}"
            if st["evals"].get(ename, {}).get("ex"):
                print(f"[skip] {ename} ({st['evals'][ename]['ex']})")
                continue
            cmd = [sys.executable, "-u", EVAL_LAUNCHER, ename, "ai2/jupiter-cirrascale-2",
                   "--task", TASK_KEY[task], "--variant", "docchunk", "--ckpt", f"{CKPTS}/{name}",
                   "--query-position", "after", "--cot-mode", "none", "--tokenizer", TOKENIZER, "--ngpu", "2",
                   "--max-test", "500", "--priority", "urgent", "--max-length", "20000", "--batch-size", "8",
                   "--dc-rung-files", json.dumps({task: RUNGS[task]}), "--dc-rungs", ",".join(RUNGS[task])]
            rc, out = sh(cmd)
            ex = parse_id(out)
            os.makedirs(f"{HERE}/launch_logs", exist_ok=True)
            open(f"{HERE}/launch_logs/{ename}.eval.log", "w").write(out)
            print(f"eval {ename} -> {ex or 'FAILED: ' + out[-400:]}", flush=True)
            if ex:
                st["evals"][ename] = {"ex": ex, "train_run": name, "task": task}
                json.dump(st, open(STATE + ".tmp", "w"), indent=1)
                os.replace(STATE + ".tmp", STATE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="ds0f2")
    ap.add_argument("--budget", default="f1952", choices=sorted(EXPECTED_MIN))
    ap.add_argument("--seed", type=int, default=0, help="0 = the trainer default (as the recipe's seed-0 runs)")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps (name gets -smoke)")
    ap.add_argument("--runs", default="", help="eval: comma run names (default: every non-smoke run in the state)")
    ap.add_argument("mode", choices=["launch", "dry_run", "eval"])
    a = ap.parse_args()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}}
    st.setdefault("evals", {})
    if a.mode == "eval":
        return do_eval(a, st)
    for arm in a.arms.split(","):
        name = run_name(arm, a.budget, a.seed, a.max_steps > 0)
        if name in st["runs"] and a.mode == "launch":
            print(f"[skip] {name} ({st['runs'][name]['ex']})")
            continue
        cmd = [sys.executable, "-u", LAUNCHER, "--task", "mt4", "--variant", "softtoken", "--model-family", "qwen3_5",
               "--model-scale", "4b", "--data-root", f"{SHARDS}/mt4{a.budget}", "--run-name", name,
               "--exact-run-name", "--num-nodes", "1", "--num-gpus", "2", "--epochs", "1", "--lr", str(LR),
               "--cluster", "ai2/jupiter-cirrascale-2", "--wandb-group", "f8k-q35-4b", "--priority", "urgent",
               "--min-runtime", "1h", "--no-follow", "--no-compile", "--seq-len", "12288", "--global-batch", "16",
               "--micro-batch-instances", "2", "--base-checkpoint", BASE, "--experiment", "tokdrop_lr",
               "--arm", ARMS[arm][0], "--expected-minutes", str(30 if a.max_steps else EXPECTED_MIN[a.budget])]
        if a.seed:
            cmd += ["--seed", str(a.seed)]
        if a.max_steps:
            cmd += ["--max-steps", str(a.max_steps)]
        cmd += ["--extra-args", ARMS[arm][1], a.mode]
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        out = p.stdout + p.stderr
        os.makedirs(f"{HERE}/launch_logs", exist_ok=True)
        open(f"{HERE}/launch_logs/{name}.{a.mode}.log", "w").write(out)
        m = re.search(r"SUBMITTED id=(\S+)", out)
        ex = m.group(1) if m else None
        print(f"{name} -> rc={p.returncode} {ex or out[-600:]}", flush=True)
        if a.mode == "launch" and ex:
            st["runs"][name] = {"ex": ex, "arm": ARMS[arm][0], "budget": a.budget, "seed": a.seed,
                                "max_steps": a.max_steps}
            json.dump(st, open(STATE + ".tmp", "w"), indent=1)
            os.replace(STATE + ".tmp", STATE)


if __name__ == "__main__":
    main()
