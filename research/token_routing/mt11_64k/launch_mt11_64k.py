"""
Long-context, large-data 11-task CTC mix (``mt11L64``): dense vs the mid-stack ONLINE token router at matched
FLOPs, rows up to 65,536 tokens (Prasann, 2026-10-22: "for the 11-task mix try to go up to like 64k and have a
larger amount of data ... since it seems like these methods maybe scale better with more flops and longer
contexts?" / "64k context length").

DATA (``build_beaker.sh``): setA seed 42 gold shards (all 11 tasks, full gold sidecars, query position both),
rows <= 65,536 tokens, ds64's short-heavy token shares (30/20/15/13/12/10 over <=3k/6k/12k/24k/48k/64k) of
64M tokens per task: 156,587 rows, 662M tokens, 12.3k-15.8k rows per task (outlier/oolong/rerank/strmatch have no
rows in the top bin -- their 64k-bucket rows are all > 65,536 tokens).

GEOMETRY (every arm): Beaker jupiter, 8 GPUs, UNPACKED rows at seq 65,536, 128 rows/step, micro-batch 1, flash_2,
repaired Qwen3.5-4B base, per-example FLOP meter (``flop_meter/actual_pflops``). ``dense`` is the PAD-FREE dense
path (``densepf``: the soft-token path with every document kept whole -- same tokens and maths as dense; the
padded ``full`` variant would push 128 x 65,536 tokens/step through the model for ~570k real ones). On the mt4
fast8k screen densepf scored 0.907 / 0.920 / 0.228 vs dense 0.916 / 0.918 / 0.252 (contradiction / nq / outlier
@8k, eval_size 500 each), i.e. dense within noise.

BUDGETS: ``f1`` = one epoch (1,223 steps); ``f2`` / ``f4`` / ``f8`` = 1/2, 1/4, 1/8 of it via --max-tokens (counted
as rows x seq_len per step), each with its own COMPLETE LR schedule (--max-steps would only hard-stop mid-schedule).
A smaller budget sees the first k steps of the same shuffled epoch, i.e. a uniform random subset of the rows.

ARMS: ``dense`` (LR 2e-6 and 5e-6; the reference is the per-cell better of the two), ``otr10M7`` / ``otr10M8``
(launch_fast8k.py's flags verbatim + 512 target samples per row, as launch_ds64_otr.py). The router's charged
full-context step every 8 is one full row per micro-batch at micro 1 -- the same memory as the dense arm -- so it
needs no separate memory plan beyond micro 1 (and every multi-hour run saves a resume point every 300 steps).

    python research/token_routing/mt11_64k/launch_mt11_64k.py --arms dense --budgets f8 --lrs 5e-6 dry_run
    python research/token_routing/mt11_64k/launch_mt11_64k.py --arms dense --budgets f8,f4,f2,f1 --lrs 2e-6,5e-6 launch
    python research/token_routing/mt11_64k/launch_mt11_64k.py status
    python research/token_routing/mt11_64k/launch_mt11_64k.py --runs <run> eval      # 10 tasks x 2k..64k x 500
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "ds64_fast2k"))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "ds64_fast8k"))
from launch_fast2k import BASE, CKPTS, TOKENIZER, W, beaker_status, parse_id, sh  # noqa: E402
from launch_fast8k import ARM_EXTRA, SUITE_EVAL_ENV, SUITE_EVAL_TASKS, TASK_KEY  # noqa: E402

LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
EVAL_LAUNCHER = f"{REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py"
STATE = os.path.join(HERE, "state.json")
SHARD = os.environ.get("M64_SHARD", f"{W}/mt11_64k/shards/mt11L64")
#: realised rows of the shard (build log 01M5KQ85DJ4H4GD8SYCQ8HBD7C / reports/verify_mt11L64.json)
ROWS = int(os.environ.get("M64_ROWS", "156587"))
SEQ_LEN = 65536
GB = 128
#: 8 by default; M64_NGPU=4 when whole nodes queue for hours (2026-10-23: 99 urgent whole-node jobs ahead). Rows/step
#: and micro-batch are unchanged, so scores and FLOPs are unaffected; only wall-clock changes (compare like with like).
GPUS = int(os.environ.get("M64_NGPU", "8"))
MICRO = 1
STEPS_FULL = ROWS // GB
FRACTIONS = {"f1": 1, "f2": 2, "f4": 4, "f8": 8, "f16": 16}
#: any-of placement (comma list): whole 8-GPU nodes queue for days on jupiter alone (2026-10-23: 67 urgent ahead)
CLUSTER = os.environ.get("M64_CLUSTER", "ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale")
WANDB_GROUP = os.environ.get("M64_WANDB_GROUP", "m64-q35-4b")
EXPERIMENT = "mt11_64k"
#: resume points: urgent jobs are preemptible after minRuntime 1h, and the first router f4 attempt lost ~1h to a
#: preemption with no checkpoint yet (2026-10-23); keep 1 (~55 GB at 4B)
SAVE_INTERVAL = 50
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "online_router"))
from launch_otr import ARMS as OTR_ARMS  # noqa: E402

ARMS = {
    "dense": ARM_EXTRA["densepf"],
    **{a: ARM_EXTRA[a] + " --st-otr-samples-per-row 512" for a in ("otr10M7", "otr10M8", "otr30M7", "otr30M8")},
    # the adopted universal recipe (LR-sweep session, 2026-10-25): LoRA r64 on the early stack + two-stage router
    # (keep 20% after block 7, 7% after block 16), per-document floor 16, no full-context steps, no freeze --
    # launch_otr.py's lora64r20to7f16 verbatim. Its checkpoints carry adapter keys: score them from a commit
    # >= 984825862 (eval guard rebuilds the adapters; older evals silently score the plain base).
    "lora64r20to7f16": OTR_ARMS["lora64r20to7f16"] + " --st-otr-samples-per-row 512",
}
#: eval: the CTC-suite rows at every rung the subset has (r64k for 6 tasks), 500 rows, query position both,
#: ctc-spec grading (run_q4b_beaker_multirung_eval.py --suite-rows --suite-rungs); rerank has no ladder grader
EVAL_RUNGS = os.environ.get("M64_EVAL_RUNGS", "2k,8k,16k,32k,64k")
EVAL_MAX_LENGTH = os.environ.get("M64_EVAL_MAX_LENGTH", "73728")
EVAL_MIN_RUNTIME = {"grouping": "4h"}


#: ``smk``: the launch smoke -- the f1 schedule hard-stopped after 10 steps (step 1 keep-all, steps 2-7 routed, step 8
#: the router's full-context step; ~1 row > 49k tokens per step), exported and then evaluated on one task at 64k
SMOKE_STEPS = 10


def steps_for(budget):
    return SMOKE_STEPS if budget == "smk" else math.ceil(STEPS_FULL / FRACTIONS[budget])


def expected_minutes(arm, steps):
    return int(30 + 0.5 * steps * (1.0 if arm == "dense" else 0.6))


def min_runtime(arm, steps):
    """Beaker minRuntime covering the whole run (x1.3, at least 1h): urgent jobs are preemptible once minRuntime
    has passed, and the 2-3 h runs were preempted at ~1 h and restarted from scratch twice (2026-10-23)."""
    # Beaker refuses minRuntime > 8h; longer runs rely on the resume point every SAVE_INTERVAL steps past that
    return f"{min(8, max(1, math.ceil(expected_minutes(arm, steps) * 1.3 / 60)))}h"


def run_name(arm, budget, lr, seed=0):
    return f"m64-{arm}-{budget}-lr{lr:g}" + (f"-sd{seed}" if seed else "")


def train_cmd(arm, budget, lr, mode, seed=0):
    steps = steps_for(budget)
    extra = f"{ARMS[arm]} --attn-backend flash_2"
    if steps > SAVE_INTERVAL and budget != "smk":
        extra += f" --save-interval {SAVE_INTERVAL} --max-checkpoints 1"
    if budget == "smk":
        extra += f" --max-steps {SMOKE_STEPS}"
    elif budget != "f1":
        extra += f" --max-tokens {steps * GB * SEQ_LEN}"  # complete LR schedule over `steps` steps
    cmd = [sys.executable, "-u", LAUNCHER, "--task", "mt11", "--variant", "softtoken", "--model-family", "qwen3_5",
           "--model-scale", "4b", "--data-root", SHARD, "--run-name", run_name(arm, budget, lr, seed), "--exact-run-name",
           "--num-nodes", "1", "--num-gpus", str(GPUS), "--epochs", "1", "--lr", str(lr), "--cluster", CLUSTER,
           "--wandb-group", WANDB_GROUP, "--priority", "urgent", "--min-runtime", min_runtime(arm, steps),
           "--no-follow", "--no-compile",
           "--experiment", EXPERIMENT, "--arm", arm, "--seq-len", str(SEQ_LEN), "--global-batch", str(GB),
           "--micro-batch-instances", str(MICRO), "--base-checkpoint", BASE,
           "--expected-minutes", str(expected_minutes(arm, steps))]
    if seed:  # a different data order AND init of the new params; same shard, same budget
        cmd += ["--seed", str(seed)]
    return cmd + ["--extra-args", extra, mode]


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def do_launch(a, mode):
    st = load()
    for arm in a.arms.split(","):
        for budget in a.budgets.split(","):
            for lr in (float(x) for x in a.lrs.split(",")):
                name = run_name(arm, budget, lr, a.seed)
                if name in st["runs"] and mode == "launch" and not a.force:
                    print(f"[skip] {name} ({st['runs'][name].get('ex')})")
                    continue
                rc, out = sh(train_cmd(arm, budget, lr, mode, a.seed))
                os.makedirs(f"{HERE}/launch_logs", exist_ok=True)
                open(f"{HERE}/launch_logs/{name}.{mode}.log", "w").write(out)
                ex = parse_id(out)
                for ln in out.splitlines():
                    if ln.startswith("[beaker_ctc_suite] cmd=") or "Traceback" in ln or "Error" in ln:
                        print("  " + ln[:1200])
                print(f"{name} [{steps_for(budget)} steps] -> rc={rc} {ex or out[-300:]}", flush=True)
                if mode == "launch":
                    st["runs"][name] = {"ex": ex, "arm": arm, "budget": budget, "lr": lr, "seed": a.seed, "gpus": GPUS,
                                        "steps_expected": steps_for(budget), "state": "S" if ex else "LAUNCH-FAILED"}
                    save(st)


def eval_cmd(ename, task, ckpt, dry):
    cmd = [sys.executable, "-u", EVAL_LAUNCHER, ename, CLUSTER, "--task", TASK_KEY.get(task, task),
           "--variant", "docchunk", "--ckpt", ckpt, "--query-position", "both", "--cot-mode", "none",
           "--tokenizer", TOKENIZER, "--ngpu", "2", "--max-test", "500", "--priority", "urgent",
           "--max-length", EVAL_MAX_LENGTH, "--batch-size", "8", "--suite-rows", "--suite-rungs", EVAL_RUNGS,
           "--experiment", EXPERIMENT]
    cmd += [x for kv in SUITE_EVAL_ENV for x in ("--env", kv)]
    return cmd + (["--dry-run"] if dry else [])


def do_eval(a):
    st = load()
    for name in a.runs.split(","):
        r = st["runs"].get(name)
        if r is None:
            print(f"[skip] {name} not in state.json")
            continue
        if r.get("state") != "DONE" and not a.force and not a.dry:
            print(f"[wait] {name} train={r.get('state')}")
            continue
        for task in (a.tasks.split(",") if a.tasks else SUITE_EVAL_TASKS):
            ename = f"{name}-ev{task}"
            if st["evals"].get(ename, {}).get("ex") and not a.force and not a.dry:
                print(f"[skip] {ename} ({st['evals'][ename]['ex']})")
                continue
            # minRuntime per task (run_q4b_beaker_multirung_eval.py reads BEAKER_MIN_RUNTIME): a full 2k..64k pass takes
            # 40-80 min, grouping 2.5-3 h (its 32k rung decodes up to 4,096 tokens/row); a 1h eval was preempted at 2.5h
            os.environ["BEAKER_MIN_RUNTIME"] = EVAL_MIN_RUNTIME.get(task, "2h")
            rc, out = sh(eval_cmd(ename, task, f"{CKPTS}/{name}", a.dry))
            ex = None if a.dry else parse_id(out)
            if a.dry:
                print(f"{ename} [dry rc={rc}] " + " ".join(l.strip() for l in out.splitlines() if "cmd" in l.lower())[:1500])
                continue
            open(f"{HERE}/launch_logs/{ename}.eval.log", "w").write(out)
            st["evals"][ename] = {"ex": ex, "train_run": name, "task": task, "state": "S" if ex else "LAUNCH-FAILED"}
            save(st)
            print(f"eval {ename} -> {ex or 'FAILED: ' + out[-300:]}", flush=True)


def do_status(a):
    st = load()
    for kind in ("runs", "evals"):
        for name, r in st[kind].items():
            if not r.get("ex") or r.get("state") in ("DONE", "FAILED", "CANCELLED"):
                continue
            s, rc = beaker_status(r["ex"])
            if s == "F":
                r["state"], r["rc"] = ("DONE" if rc == 0 else "FAILED"), rc
            elif s != "?":
                r["state"] = s
    save(st)
    for name, r in sorted(st["runs"].items()):
        ev = {k: v.get("state") for k, v in st["evals"].items() if v["train_run"] == name}
        done = sum(v == "DONE" for v in ev.values())
        print(f"{name:28} {r.get('ex')} train={r.get('state')}  evals {done}/{len(ev)} done "
              + " ".join(f"{k.split('-ev')[-1]}={v}" for k, v in sorted(ev.items()) if v != "DONE"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="dense")
    ap.add_argument("--budgets", default="f1")
    ap.add_argument("--lrs", default="5e-6")
    ap.add_argument("--seed", type=int, default=0, help="training seed (0 = default; else run name gets -sd<seed>)")
    ap.add_argument("--runs", default="")
    ap.add_argument("--tasks", default="", help="eval: comma subset of the 10 suite tasks")
    ap.add_argument("--dry", action="store_true", help="eval: launcher --dry-run, nothing submitted")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("mode", choices=["launch", "dry_run", "eval", "status"])
    a = ap.parse_args()
    {"eval": do_eval, "status": do_status}.get(a.mode, lambda x: do_launch(x, a.mode))(a)


if __name__ == "__main__":
    main()
