"""FLOP-matched VALIDATION of the early dense-loss proxy (experiment ``tokdrop_lr``, 2026-11-17).

Full training runs on the proxy's pool ``mtb16k`` (contradiction / outlier / reorder / qdmatch_nq at 4k / 8k / ~14.5k
tokens, 1,952 rows per task = 488 steps of 16 rows), then generation-graded evals on the CTC-suite rows of the same
four tasks at the 4k / 8k / 16k rungs (500 rows per cell), to check whether the 50-200-step held-out answer-CE proxy
ranks and prices the recipe correctly.

Trainer flags are the proxy's (``run_proxy_beaker.sh`` / ``launch_proxy.py`` ARMS, minus the probe): 16 rows/step,
micro-batch 2, sequence 16,384, unpacked, ``--no-compile``, base ``q35-4b-base-markerfix``. Launcher:
``recipes/train/research_trainer/beaker_ctc_suite.py`` (train -> model-only export to weka ckpts), as
``ds64_fast8k/launch_fast8k.py`` does. Budgets: f1 = one epoch (488 steps); f2 / f4 = ``--max-tokens`` of half /
a quarter of the tokens (244 / 122 steps), each a complete LR schedule (warmup 3% + linear decay over its own
length). GPUs: 2 (the proxy's 2 x micro 2, so the soft-token FLOP meter matches the proxy) except the dense full
fine-tune, which runs on 4 GPUs (same 16 rows/step, micro 2) because its 2-GPU footprint is ~78 GiB reserved on H200;
dense FLOPs do not depend on the GPU split.

Eval: ``recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py --suite-rows --suite-rungs 4k,8k,16k`` (native
docchunk ladder, ctc-spec grading, query position ``after`` = the mtb16k converter flag, no-cot), one 2-GPU job per
(run, task).

    PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
    PYTHONPATH=src $PY research/token_routing/early_loss_proxy/validation/launch_val.py train [--runs a,b] [--dry]
    PYTHONPATH=src $PY research/token_routing/early_loss_proxy/validation/launch_val.py status
    PYTHONPATH=src $PY research/token_routing/early_loss_proxy/validation/launch_val.py evals [--runs a,b] [--dry]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from launch_proxy import LORA, RECIPE  # noqa: E402  (the proxy's flag strings, verbatim)

W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
BASE = f"{W}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
DATA = f"{W}/early_loss_proxy/shards/mtb16k"
CKPTS = f"{W}/ctc_suite/ckpts"
TOKENIZER = f"{W}/hf_tokenizers/Qwen3.5-0.8B-Base"
TRAIN_LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
EVAL_LAUNCHER = f"{REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py"
STATE = f"{HERE}/state.json"
EXPERIMENT = "tokdrop_lr"
CLUSTER = "ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale"
WANDB_GROUP = "elpv-q35-4b"
SEQ, GB, MICRO = 16384, 16, 2
STEPS_F1 = 488  # 7,808 rows / 16
#: budget -> (--max-tokens or 0, optimizer steps)
BUDGETS = {"f1": (0, STEPS_F1), "f2": (STEPS_F1 // 2 * GB * SEQ, STEPS_F1 // 2),
           "f4": (STEPS_F1 // 4 * GB * SEQ, STEPS_F1 // 4)}
#: arm -> (trainer variant, lr, flags, gpus, est. s/step on 2-4 H100)
ARMS = {"denselora": ("full", 1e-5, LORA, 2, 17.0),
        "recipe": ("softtoken", 1e-5, RECIPE, 2, 7.0),
        "dense": ("full", 5e-6, "", 4, 11.0)}
#: the runs of this validation: name -> (arm, budget, seed)
RUNS = {
    "elpv-denselora-f1": ("denselora", "f1", 0),
    "elpv-denselora-f2": ("denselora", "f2", 0),
    "elpv-denselora-f4": ("denselora", "f4", 0),
    "elpv-recipe-f1": ("recipe", "f1", 0),
    "elpv-recipe-f1-sd1": ("recipe", "f1", 1),
    "elpv-recipe-f2": ("recipe", "f2", 0),
    "elpv-dense-f1": ("dense", "f1", 0),
}
#: eval: launcher task key per suite task; rungs; prefill cap (>= the longest 16k-rung prompt + answer)
EVAL_TASKS = {"contradiction": "contra", "outlier": "outlier", "reorder": "reorder", "qdmatch_nq": "qdmatch_nq"}
EVAL_RUNGS = os.environ.get("ELPV_EVAL_RUNGS", "4k,8k,16k")
EVAL_MAX_LENGTH = os.environ.get("ELPV_EVAL_MAX_LENGTH", "24576")
EVAL_ENV = ["LADDER_GEN_BATCH=32", "LADDER_GEN_BATCH_TOKENS=196608", "KEEP_RESPONSES=1"]


def sh(cmd, timeout=1800):
    r = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def parse_id(out):
    for ln in out.splitlines():
        if "SUBMITTED id=" in ln:
            return ln.split("id=")[1].split()[0]
    return None


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def expected_minutes(arm, budget):
    return math.ceil(BUDGETS[budget][1] * ARMS[arm][4] / 60 + 12)


def train_cmd(name, mode):
    arm, budget, seed = RUNS[name]
    variant, lr, flags, gpus, _ = ARMS[arm]
    max_tokens, _ = BUDGETS[budget]
    exp_min = expected_minutes(arm, budget)
    extra = f"{flags} --max-tokens {max_tokens}".strip() if max_tokens else flags
    cmd = [sys.executable, "-u", TRAIN_LAUNCHER, "--task", "mt4", "--variant", variant, "--model-family", "qwen3_5",
           "--model-scale", "4b", "--data-root", DATA, "--base-checkpoint", BASE, "--run-name", name,
           "--exact-run-name", "--num-nodes", "1", "--num-gpus", str(gpus), "--epochs", "1", "--lr", f"{lr:g}",
           "--seq-len", str(SEQ), "--global-batch", str(GB), "--micro-batch-instances", str(MICRO),
           "--seed", str(seed), "--cluster", CLUSTER, "--wandb-group", WANDB_GROUP, "--priority", "urgent",
           "--min-runtime", f"{min(240, math.ceil(1.3 * exp_min))}m", "--no-follow", "--no-compile",
           "--experiment", EXPERIMENT, "--arm", f"{arm}-{budget}", "--expected-minutes", str(exp_min)]
    if extra:
        cmd += ["--extra-args", extra]
    return cmd + [mode]


def do_train(a):
    st = load()
    for name in (a.runs.split(",") if a.runs else RUNS):
        if name in st["runs"] and not a.force and not a.dry:
            print(f"[skip] {name} already launched ({st['runs'][name]['ex']})")
            continue
        cmd = train_cmd(name, "dry_run" if a.dry else "launch")
        rc, out = sh(cmd)
        ex = parse_id(out)
        os.makedirs(f"{HERE}/launch_logs", exist_ok=True)
        open(f"{HERE}/launch_logs/{name}.train{'.dry' if a.dry else ''}.log", "w").write(" ".join(cmd) + "\n" + out)
        if a.dry:
            print(f"{name} [dry rc={rc}] " + " ".join(ln for ln in out.splitlines() if "cmd=" in ln)[:1500])
            continue
        arm, budget, seed = RUNS[name]
        st["runs"][name] = {"ex": ex, "arm": arm, "budget": budget, "seed": seed,
                            "steps_expected": BUDGETS[budget][1], "state": "S" if ex else "LAUNCH-FAILED"}
        save(st)
        print(f"train {name} -> {ex or 'FAILED: ' + out[-600:]}", flush=True)


def eval_cmd(ename, task, ckpt, dry):
    cmd = [sys.executable, "-u", EVAL_LAUNCHER, ename, "ai2/jupiter-cirrascale-2", "--task", EVAL_TASKS[task],
           "--variant", "docchunk", "--ckpt", ckpt, "--query-position", "after", "--cot-mode", "none",
           "--tokenizer", TOKENIZER, "--ngpu", "2", "--max-test", "500", "--priority", "urgent",
           "--max-length", EVAL_MAX_LENGTH, "--batch-size", "8", "--suite-rows", "--suite-rungs", EVAL_RUNGS,
           "--experiment", EXPERIMENT]
    cmd += [x for kv in EVAL_ENV for x in ("--env", kv)]
    return cmd + (["--dry-run"] if dry else [])


def do_evals(a):
    st = load()
    names = a.runs.split(",") if a.runs else list(st["runs"])
    for name in names:
        r = st["runs"].get(name, {})
        if r.get("state") != "DONE" and not a.force and not a.dry:
            print(f"[wait] {name} train={r.get('state')}")
            continue
        ckpt = a.ckpt or f"{CKPTS}/{name}"
        for task in (a.tasks.split(",") if a.tasks else EVAL_TASKS):
            ename = f"{name}-ev{task}"
            if st["evals"].get(ename, {}).get("ex") and not a.force and not a.dry:
                print(f"[skip] {ename} ({st['evals'][ename]['ex']})")
                continue
            os.environ["BEAKER_MIN_RUNTIME"] = a.min_runtime
            rc, out = sh(eval_cmd(ename, task, ckpt, a.dry))
            if a.dry:
                print(f"{ename} [dry rc={rc}] " + out[-2500:])
                continue
            ex = parse_id(out)
            os.makedirs(f"{HERE}/launch_logs", exist_ok=True)
            open(f"{HERE}/launch_logs/{ename}.eval.log", "w").write(out)
            st["evals"][ename] = {"ex": ex, "train_run": name, "task": task, "ckpt": ckpt,
                                  "state": "S" if ex else "LAUNCH-FAILED"}
            save(st)
            print(f"eval {ename} -> {ex or 'FAILED: ' + out[-400:]}", flush=True)


def beaker_status(ex):
    try:
        out = subprocess.run([os.path.expanduser("~/.local/bin/beaker"), "experiment", "get", ex, "--format",
                              "json"], capture_output=True, text=True, timeout=120).stdout
        s = json.loads(out)[0]["jobs"][-1]["status"]
    except Exception:  # noqa: BLE001
        return "?", None
    if s.get("finalized"):
        return "F", s.get("exitCode")
    return ("R" if s.get("started") else "P"), None


def do_status(a):
    st = load()
    for kind in ("runs", "evals"):
        for name, r in st[kind].items():
            if r.get("ex") and r.get("state") not in ("DONE", "FAILED", "CANCELLED"):
                s, rc = beaker_status(r["ex"])
                if s == "F":
                    r["state"], r["rc"] = ("DONE" if rc == 0 else "FAILED"), rc
                elif s != "?":
                    r["state"] = s
            print(f"{kind[:-1]:4s} {name:42s} {r.get('state')} {r.get('ex')}")
    save(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "evals", "status"])
    ap.add_argument("--runs", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--ckpt", default="", help="evals: score this checkpoint dir instead of the run's save folder")
    ap.add_argument("--min-runtime", default="2h", help="evals: Beaker minRuntime")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"train": do_train, "evals": do_evals, "status": do_status}[a.mode](a)


if __name__ == "__main__":
    main()
