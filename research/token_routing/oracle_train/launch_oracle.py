"""Launch / evaluate ORACLE-selection training runs (gradient upper bounds for token dropping) on Beaker jupiter.

Every step: dense forward+backward -> exact per-token LoRA-gradient attribution -> the smallest top-k set per stage
with attribution cos >= 0.95 (allocation per document / row / whole batch) -> routed forward+backward on that set,
which is the gradient applied (``olmo_core.nn.oracle_token_select``). Same geometry as the recipe
``lora64all32r20to7`` (``research/token_routing/ds64_fast8k/launch_fast8k.py``): repaired Qwen3.5-4B base,
whole-model LoRA r64 (scale 32), LR 1e-5, 2 GPUs, micro 2, 16 rows/step, the fresh 4-task pool, drop structure
7:0.2 / 16:0.07 nested -- with the router's choice replaced by the oracle's. Evals: the 4-task 8k + 2k rungs x 500
rows, no-cot, query after (the legacy native multirung eval the tdb/tde runs use).

    python research/token_routing/oracle_train/launch_oracle.py --arms orcbatch --budgets f1952 launch
    python research/token_routing/oracle_train/launch_oracle.py --runs <run> eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "ds64_fast8k"))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "ds64_fast2k"))
from launch_fast2k import BASE, CKPTS, W, parse_id, sh  # noqa: E402
from launch_fast8k import EVAL_LAUNCHER, MT4_TASKS, RUNGS, TASK_KEY, TOKENIZER  # noqa: E402

LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
SHARDS = f"{W}/ds64/fast8k/shards"
STATE = f"{HERE}/state.json"
SUFFIX = "-orc"
WANDB_GROUP = os.environ.get("ORC_WANDB_GROUP", "f8k-oracle")

#: the recipe lora64all32r20to7, flag for flag, with the oracle choosing instead of the router
RECIPE = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
          "--st-otr-early-lora-blocks 32 --attn-backend flash_2 --st-otr-score oracle")
_ALL = "--oracle-curve-scopes doc,row,batch"  # every run logs the exact cos curves of all three scopes

ARMS = {
    "orcdoc": f"{RECIPE} --oracle-scope doc {_ALL}",
    "orcrow": f"{RECIPE} --oracle-scope row {_ALL}",
    "orcbatch": f"{RECIPE} --oracle-scope batch {_ALL}",
    # + uniformly random extra body tokens = 2x the oracle set (per row, per stage): are "negatives" needed?
    "orcbatchr2": f"{RECIPE} --oracle-scope batch --oracle-random-mult 2 {_ALL}",
    "orcdocr2": f"{RECIPE} --oracle-scope doc --oracle-random-mult 2 {_ALL}",
}


def run_name(arm, budget, lr, seed=0):
    return f"f8k-mt4-{arm}{f'sd{seed}' if seed else ''}-{budget.replace('@', 'st')}{SUFFIX}-lr{lr:g}"


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def do_launch(a, mode):
    st = load()
    for arm in a.arms.split(","):
        for budget in a.budgets.split(","):
            name = run_name(arm, budget, a.lr, a.seed)
            if name in st["runs"] and mode == "launch" and not a.force:
                print(f"[skip] {name} ({st['runs'][name]['ex']})")
                continue
            rows, _, steps = budget[1:].partition("@")
            data = f"{SHARDS}/mt4{budget[0]}{rows}"
            cmd = [sys.executable, "-u", LAUNCHER, "--task", "mt4", "--variant", "softtoken",
                   "--model-family", "qwen3_5", "--model-scale", "4b", "--data-root", data,
                   "--run-name", name, "--exact-run-name", "--num-nodes", "1", "--num-gpus", "2",
                   "--epochs", "1", "--lr", str(a.lr), "--cluster", a.cluster,
                   "--wandb-group", WANDB_GROUP, "--priority", "urgent", "--min-runtime", "1h", "--no-follow",
                   "--no-compile", "--seq-len", str(a.seq_len), "--global-batch", "16",
                   "--micro-batch-instances", "2", "--seed", str(a.seed), "--base-checkpoint", BASE,
                   "--experiment", "tokdrop_lr"]
            if steps:
                cmd += ["--max-steps", steps]
            cmd += ["--extra-args", ARMS[arm]]
            rc, out = sh(cmd + [mode])
            ex = parse_id(out)
            print(f"{name} -> rc={rc} {ex or out[-600:]}")
            if mode == "launch" and ex:
                st["runs"][name] = {"ex": ex, "arm": arm, "budget": budget, "lr": a.lr, "seed": a.seed,
                                    "meta": f"{CKPTS}/{name}/oracle_meta"}
                save(st)


def do_eval(a):
    st = load()
    for name in a.runs.split(","):
        for task in MT4_TASKS:
            ename = f"{name}-ev{task}"
            if ename in st["evals"] and not a.force:
                print(f"[skip] {ename}")
                continue
            cmd = [sys.executable, "-u", EVAL_LAUNCHER, ename, a.cluster,
                   "--task", TASK_KEY[task], "--variant", "docchunk", "--ckpt", f"{CKPTS}/{name}",
                   "--query-position", "after", "--cot-mode", "none", "--tokenizer", TOKENIZER,
                   "--ngpu", "2", "--max-test", "500", "--priority", "urgent", "--max-length", "20000",
                   "--batch-size", "8", "--dc-rung-files", json.dumps({task: RUNGS[task]}),
                   "--dc-rungs", ",".join(RUNGS[task])]
            rc, out = sh(cmd)
            ex = parse_id(out)
            print(f"eval {ename} -> {ex or 'FAILED: ' + out[-400:]}")
            if ex:
                st["evals"][ename] = {"ex": ex, "train_run": name, "task": task}
                save(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="orcbatch")
    ap.add_argument("--budgets", default="f1952", help="f<rows/task>[@<max steps>] (fresh pool shard mt4f<rows>)")
    ap.add_argument("--lr", type=float, default=1e-5, help="the recipe's LR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=12288)
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--runs", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("mode", choices=["launch", "dry_run", "eval"])
    a = ap.parse_args()
    do_eval(a) if a.mode == "eval" else do_launch(a, a.mode)


if __name__ == "__main__":
    main()
