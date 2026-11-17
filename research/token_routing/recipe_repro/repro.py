"""Reproduce the FLOP-drop recipe's Pareto results (dense vs recipe at several data budgets) on Beaker.

One script, three steps, three mixes. See README.md in this folder.

    python research/token_routing/recipe_repro/repro.py train   --mix fast8k --arm recipe --budget f1952 [--dry]
    python research/token_routing/recipe_repro/repro.py eval    --mix fast8k --run rr-fast8k-recipe-f1952 [--dry]
    python research/token_routing/recipe_repro/repro.py status
    python research/token_routing/recipe_repro/repro.py collect --mix fast8k        # PF + scores -> curve + FTM
    python research/token_routing/recipe_repro/repro.py reference --mix fast8k      # Prasann's numbers, same table

Mixes (exact weka paths in MIXES below):
  fast8k  4 tasks contradiction / nq / outlier / oolong, 8k rows (<= 12,288 tokens); where the established recipe
          numbers come from (8k FLOPs-to-match 4.6x at 1,952 rows/task).
  new4    4 tasks contradiction / outlier / reorder / qdmatch_nq, rows at 4k / 8k / ~14.5k tokens (pool mtb16k, built
          2026-11-17 for the early-loss-proxy work). Reorder's grader is broken (every arm ~0): aggregates exclude it.
  mt11    11 tasks, rows up to 64k tokens (pool mt11L64, experiment mt11_64k). 8 or 4 GPUs per run; expensive.

Per-user settings come from env vars (nothing to edit): BEAKER_WORKSPACE, BEAKER_BUDGET, WANDB_ENTITY, WANDB_PROJECT,
CTC_USER_DIR (your folder under /weka/.../checkpoints/ for checkpoints and eval outputs), RR_PREFIX (run-name prefix).
State (run name -> Beaker experiment id) is kept in runs_state.json beside this file (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TRAIN_LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
EVAL_LAUNCHER = f"{REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py"
STATE = os.environ.get("RR_STATE", os.path.join(HERE, "runs_state.json"))

#: shared, read-only inputs (Prasann's weka folder; readable from any AI2 Beaker job that mounts oe-training-default)
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
BASE = f"{W}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"  # marker-embedding-repaired Qwen3.5-4B
TOKENIZER = f"{W}/hf_tokenizers/Qwen3.5-0.8B-Base"  # same vocab as the 4B; what every eval used
#: per-user outputs
USER_DIR = os.environ.get("CTC_USER_DIR", "prasanns")
CKPTS = f"/weka/oe-training-default/ai2-llm/checkpoints/{USER_DIR}/ctc_suite/ckpts"
PREFIX = os.environ.get("RR_PREFIX", "rr")
CLUSTER = os.environ.get("RR_CLUSTER", "ai2/jupiter-cirrascale-2")

# ---------------------------------------------------------------------------------------------------------------
# Trainer flags, verbatim from the launchers that produced the reference numbers (check_flags() asserts this).
# ---------------------------------------------------------------------------------------------------------------
#: THE recipe (arm name lora64all32r20to7): whole-model LoRA r64 / scale 32 on all 32 blocks + online token router.
#: All tokens run blocks 0-6; after block 7 each document keeps its top 20% body tokens (router score), after block 16
#: the top 7%; prompt / markers / query / answer always kept; original position ids. No full-context steps.
RECIPE = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
          "--st-otr-early-lora-blocks 32")
#: mt11 64k used the earlier variant lora64r20to7f16: LoRA on blocks 0-6 only, + at least 16 kept tokens per document,
#: + 512 router target samples per row (whole-model LoRA was never run at 64k).
RECIPE_MT11 = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --attn-backend flash_2 "
               "--st-otr-keep-frac 0.1 --st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 "
               "--st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-min-per-doc 16 "
               "--st-otr-samples-per-row 512")
#: dense full fine-tune, pad-free: the soft-token path with every document kept whole (mt11 only; same maths as dense)
DENSE_PF = "--st-gold-blind --st-keep-prob 1.0 --st-drop-slots"
#: dense LoRA (new4's reference curve): rank 64 / scale 32 on every block, frozen base
DENSE_LORA = "--lora-rank 64 --lora-blocks 0 --lora-scale 32 --attn-backend flash_2"

#: arm -> (trainer variant, LR, extra trainer flags)
#: budget -> (data dir, optimizer steps, --max-tokens or 0). A budget below one epoch uses --max-tokens so the LR
#: schedule (3% warmup, linear decay) completes; --max-steps would stop mid-schedule.
EVAL_ENV_SUITE = ["LADDER_GEN_BATCH=32", "LADDER_GEN_BATCH_TOKENS=196608"]
MIXES = {
    "fast8k": dict(
        desc="contradiction / nq / outlier / oolong, 8k rows; fresh pools mt4f<rows> (generator seed 5310)",
        task="mt4", seq=12288, gb=16, micro=2, gpus={"dense": 2, "recipe": 2},
        arms={"dense": ("full", 5e-6, ""), "recipe": ("softtoken", 1e-5, RECIPE + " --attn-backend flash_2")},
        budgets={f"f{r}": (f"{W}/ds64/fast8k/shards/mt4f{r}", r * 4 // 16, 0) for r in (488, 976, 1952, 3904, 5856, 7808)},
        sec_per_step={"dense": 11.3, "recipe": 5.3},  # measured: dense f1952 1.53 h / 488 steps; recipe 0.72 h
        eval_tasks=["contradiction", "nq", "outlier", "oolong"], eval_rungs=["8k", "2k"], primary_rung="8k",
        no_signal=[], query="after", max_length=20000,
        rung_files={
            "outlier": {"8k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                        "2k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl"},
            "nq": {"8k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
                   "2k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl"},
            "contradiction": {
                "8k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
                "2k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"},
            "oolong": {
                "8k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
                "2k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl"}},
        dense_curve_arm="dense"),
    "new4": dict(
        desc="contradiction / outlier / reorder / qdmatch_nq at 4k / 8k / ~14.5k tokens; pool mtb16k (1,952 rows/task)",
        task="mt4", seq=16384, gb=16, micro=2, gpus={"dense": 4, "denselora": 2, "recipe": 2},
        arms={"dense": ("full", 5e-6, ""), "denselora": ("full", 1e-5, DENSE_LORA),
              "recipe": ("softtoken", 1e-5, RECIPE + " --attn-backend flash_2")},
        budgets={f"f{k}": (f"{W}/early_loss_proxy/shards/mtb16k", 488 // k, 0 if k == 1 else 488 // k * 16 * 16384)
                 for k in (1, 2, 4)},
        sec_per_step={"dense": 7.3, "denselora": 15.0, "recipe": 6.0},  # measured on the 11-17 validation runs
        eval_tasks=["contradiction", "outlier", "reorder", "qdmatch_nq"], eval_rungs=["4k", "8k", "16k"],
        primary_rung="8k", no_signal=["reorder"], query="after", max_length=40960, suite_rows=True,
        eval_env=EVAL_ENV_SUITE + ["KEEP_RESPONSES=1"],
        dense_curve_arm="denselora"),  # only dense LoRA has a 3-point curve; dense full FT has one point (f1)
    "mt11": dict(
        desc="11 CTC tasks (setA generator, seed 42), rows <= 65,536 tokens; pool mt11L64 (156,587 rows)",
        task="mt11", seq=65536, gb=128, micro=1, gpus={"dense": 8, "recipe": 4},
        arms={"dense": ("softtoken", 5e-6, DENSE_PF + " --attn-backend flash_2"),
              "recipe": ("softtoken", 5e-6, RECIPE_MT11)},
        budgets={f"f{k}": (f"{W}/mt11_64k/shards/mt11L64", math.ceil(1223 / k), 0 if k == 1 else math.ceil(1223 / k) * 128 * 65536)
                 for k in (1, 2, 4, 8, 16)},
        sec_per_step={"dense": 23.0, "recipe": 36.0},  # dense f4 1.94 h / 306 steps on 8 GPUs; recipe f4 3.05 h on 4
        eval_tasks=["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction", "xabsence", "reorder",
                    "strmatch", "grouping"], eval_rungs=["2k", "8k", "16k", "32k", "64k"], primary_rung="8k",
        no_signal=["reorder@8k", "reorder@16k", "grouping@32k"], query="both", max_length=73728, suite_rows=True,
        eval_env=EVAL_ENV_SUITE, dense_curve_arm="dense"),
}
EVAL_KEY = {"contradiction": "contra"}  # eval launcher task keys
PRIMARY_METRIC = {"oolong": "score", "grouping": "pairwise_f1", "reorder": "kendall_tau"}  # else f1


def sh(cmd, timeout=1800):
    env = dict(os.environ, PYTHONPATH=f"{REPO}/src:{os.environ.get('PYTHONPATH', '')}")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def parse_id(out):
    m = re.search(r"SUBMITTED id=(\S+)", out) or re.search(r"beaker\.org/ex/([A-Z0-9]{26})", out)
    return m.group(1) if m else None


def run_name(mix, arm, budget, seed, tag):
    return f"{PREFIX}-{mix}-{arm}-{budget}" + (f"-sd{seed}" if seed else "") + (f"-{tag}" if tag else "")


def expected_minutes(m, arm, steps):
    return math.ceil(steps * m["sec_per_step"][arm] / 60 + 15)  # + model load / export / eval-free teardown


def arm_spec(a, m):
    """(variant, LR, flags): a named arm of the mix, or a spin-off = a new --arm name + --flags (+ --lr)."""
    if a.arm in m["arms"]:
        variant, lr, flags = m["arms"][a.arm]
        if a.flags:
            flags = f"{flags} {a.flags}"  # extra flags on top of a named arm
    else:
        variant, lr, flags = "softtoken", 1e-5, a.flags + " --attn-backend flash_2"
    return variant, (a.lr or lr), flags


def train_cmd(a, m, name):
    variant, lr, flags = arm_spec(a, m)
    data, steps, max_tokens = m["budgets"][a.budget]
    if a.max_steps:
        steps = a.max_steps
    exp = expected_minutes(m, a.arm, steps)
    min_rt = f"{max(60, min(240, math.ceil(1.3 * exp)))}m"  # min(4h, 1.3 x expected); never 0
    extra = flags
    if a.max_steps:
        extra += f" --max-steps {a.max_steps}"
    elif max_tokens:
        extra += f" --max-tokens {max_tokens}"
    if m.get("save_interval") and steps > m["save_interval"]:
        extra += f" --save-interval {m['save_interval']} --max-checkpoints 1"
    gpus = a.gpus or m["gpus"][a.arm]
    cmd = [sys.executable, "-u", TRAIN_LAUNCHER, "--task", m["task"], "--variant", variant,
           "--model-family", "qwen3_5", "--model-scale", "4b", "--data-root", data, "--base-checkpoint", BASE,
           "--run-name", name, "--exact-run-name", "--num-nodes", "1", "--num-gpus", str(gpus), "--epochs", "1",
           "--lr", f"{lr:g}", "--seq-len", str(m["seq"]), "--global-batch", str(m["gb"]),
           "--micro-batch-instances", str(m["micro"]), "--cluster", a.cluster, "--wandb-group", f"{PREFIX}-{a.mix}",
           "--priority", "urgent", "--min-runtime", min_rt, "--expected-minutes", str(exp), "--no-follow",
           "--no-compile", "--experiment", os.environ.get("RR_EXPERIMENT", "tokdrop_lr"), "--arm", a.arm]
    if a.seed:
        cmd += ["--seed", str(a.seed)]
    # over Beaker's 4h minRuntime cap the job can be preempted mid-run: a rolling resume point every ~30 min
    # (train_ctc_suite.py --resume-interval: keeps one checkpoint, the relaunched job resumes from it)
    cmd += ["--resume-interval", str(max(25, round(steps * 30 / exp)) if exp > 240 else 0)]
    return cmd + ["--extra-args", extra.strip(), "dry_run" if a.dry else "launch"], steps, exp, min_rt


def do_train(a):
    m, st = MIXES[a.mix], load()
    name = run_name(a.mix, a.arm, a.budget, a.seed, a.tag)
    if name in st["runs"] and not a.dry and not a.force:
        raise SystemExit(f"{name} already launched ({st['runs'][name]['ex']}); pick a new --tag (a reused name "
                         "silently auto-resumes from the old save folder)")
    cmd, steps, exp, min_rt = train_cmd(a, m, name)
    print(" ".join(cmd))
    rc, out = sh(cmd)
    print("\n".join(ln for ln in out.splitlines() if "[beaker_ctc_suite]" in ln or "SUBMITTED" in ln
                    or "Error" in ln or "error" in ln)[-4000:] if not a.verbose else out)
    if rc:
        raise SystemExit(f"launcher rc={rc}:\n{out[-3000:]}")
    if a.dry:
        return
    ex = parse_id(out)
    st["runs"][name] = {"ex": ex, "mix": a.mix, "arm": a.arm, "budget": a.budget, "seed": a.seed, "steps": steps,
                        "expected_minutes": exp, "min_runtime": min_rt, "ckpt": f"{CKPTS}/{name}"}
    save(st)
    print(f"{name} -> https://beaker.org/ex/{ex}")


def eval_cmd(m, ename, task, ckpt, results_dir, dry):
    cmd = [sys.executable, "-u", EVAL_LAUNCHER, ename, CLUSTER, "--task", EVAL_KEY.get(task, task),
           "--variant", "docchunk", "--ckpt", ckpt, "--results-dir", results_dir,
           "--query-position", m["query"], "--cot-mode", "none", "--tokenizer", TOKENIZER, "--ngpu", "2",
           "--max-test", "500", "--priority", "urgent", "--max-length", str(m["max_length"]), "--batch-size", "8"]
    if m.get("suite_rows"):
        cmd += ["--suite-rows", "--suite-rungs", ",".join(m["eval_rungs"])]
        cmd += [x for kv in m.get("eval_env", []) for x in ("--env", kv)]
    else:
        cmd += ["--dc-rung-files", json.dumps({EVAL_KEY.get(task, task): m["rung_files"][task]}),
                "--dc-rungs", ",".join(m["eval_rungs"])]
    return cmd + (["--dry-run"] if dry else [])


def do_eval(a):
    m, st = MIXES[a.mix], load()
    runs = a.run.split(",") if a.run else [n for n, r in st["runs"].items() if r["mix"] == a.mix]
    os.environ.setdefault("BEAKER_MIN_RUNTIME", a.min_runtime)
    for name in runs:
        ckpt = a.ckpt or st["runs"].get(name, {}).get("ckpt") or f"{CKPTS}/{name}"
        for task in (a.tasks.split(",") if a.tasks else m["eval_tasks"]):
            ename = f"{name}-ev{task}"
            if ename in st["evals"] and not a.force and not a.dry:
                print(f"[skip] {ename} already launched ({st['evals'][ename]['ex']})")
                continue
            rc, out = sh(eval_cmd(m, ename, task, ckpt, f"{ckpt}/eval", a.dry))
            if a.dry:
                print(f"--- {ename} (dry-run rc={rc})\n" + "\n".join(out.splitlines()[-12:]))
                continue
            ex = parse_id(out)
            st["evals"][ename] = {"ex": ex, "train_run": name, "task": task}
            save(st)
            print(f"eval {ename} -> {ex or 'FAILED: ' + out[-600:]}")


# ---------------------------------------------------------------------------------------------------------------
# collect: Beaker logs -> PF + per-cell scores -> curve table + FLOPs-to-match
# ---------------------------------------------------------------------------------------------------------------
BEAKER = shutil.which("beaker") or os.path.expanduser("~/.local/bin/beaker")
PF_RE = re.compile(r"flop_meter/actual_pflops[\"'=: ]+([0-9.]+(?:e[+-]?\d+)?)")
SCORE_RE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ([a-z0-9_]+)=(-?[0-9.]+|None) \(n=(\d+)")


def beaker_log(ex):
    cache = os.path.join(HERE, "logs")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{ex}.log")
    if os.path.exists(path):
        return open(path, errors="replace").read()
    js = subprocess.run([BEAKER, "experiment", "get", ex, "--format", "json"], capture_output=True, text=True).stdout
    try:
        done = json.loads(js)[0]["jobs"][-1]["status"].get("finalized")
    except Exception:  # noqa: BLE001
        done = False
    txt = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=900).stdout
    if done:
        open(path, "w").write(txt)
    return txt


def se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def ftm(curve, pf, score):
    """FLOPs-to-match: dense PF needed to reach ``score`` / ``pf``. ``curve`` = [(dense PF, score)], interpolated
    linearly in log PF on its running-max envelope. Below the cheapest dense point -> an upper bound ("<="); above the
    most expensive one -> a lower bound (">=")."""
    curve = sorted(curve)
    env, best = [], -1e9
    for p, s in curve:
        best = max(best, s)
        env.append((p, best))
    if score <= env[0][1]:
        return env[0][0] / pf, "<="
    if score > env[-1][1]:
        return env[-1][0] / pf, ">="
    for (p0, s0), (p1, s1) in zip(env, env[1:]):
        if s0 < score <= s1:
            t = (score - s0) / (s1 - s0)
            return math.exp(math.log(p0) + t * (math.log(p1) - math.log(p0))) / pf, ""
    return float("nan"), "?"


def aggregates(m, cells):
    """cells {(task, rung): (value, eval_size)} -> {name: (mean, se, n_cells)}; no-signal cells excluded."""
    def ok(t, g):
        return t not in m["no_signal"] and f"{t}@{g}" not in m["no_signal"] and (t, g) in cells

    def agg(keys):
        got = [cells[k] for k in keys]
        if not got:
            return None
        return (sum(v for v, _ in got) / len(got), math.sqrt(sum(se(v, n) ** 2 for v, n in got)) / len(got), len(got))
    tasks = m["eval_tasks"]
    out = {f"{g} mean": agg([(t, g) for t in tasks if ok(t, g)]) for g in m["eval_rungs"]}
    out["all-rung mean"] = agg([(t, g) for t in tasks for g in m["eval_rungs"] if ok(t, g)])
    return out


def table(mix, points, header):
    """points: [{run, arm, budget, seed, pf, cells}] -> printed curve + FTM vs the mix's dense curve."""
    m = MIXES[mix]
    for p in points:
        p["agg"] = aggregates(m, p["cells"])
    keys = [f"{m['primary_rung']} mean"] + [k for k in points[0]["agg"] if k != f"{m['primary_rung']} mean"] if points else []
    dense = [p for p in points if p["arm"] == m["dense_curve_arm"] and p["pf"]]
    print(header)
    print(f"{'run':44s} {'PF':>8s} " + " ".join(f"{k:>18s}" for k in keys) + f"  FTM({keys[0] if keys else ''})")
    for p in sorted(points, key=lambda p: (p["arm"], p["pf"] or 0)):
        row = []
        for k in keys:
            a = p["agg"].get(k)
            row.append(f"{a[0]:.3f} ±{a[1]:.3f} ({a[2]})" if a else "-")
        f = ""
        if p["arm"] != m["dense_curve_arm"] and p["pf"] and len(dense) >= 2 and p["agg"].get(keys[0]):
            curve = [(d["pf"], d["agg"][keys[0]][0]) for d in dense if d["agg"].get(keys[0])]
            x, q = ftm(curve, p["pf"], p["agg"][keys[0]][0])
            f = f"{q}{x:.2f}x"
        pf = f"{p['pf']:.0f}" if p["pf"] else "?"
        print(f"{p['run']:44s} {pf:>8s} " + " ".join(f"{r:>18s}" for r in row) + f"  {f}")
    print("(cells per mean in parentheses; ±SE = binomial over eval rows only, seed spread not included; "
          f"excluded as no-signal: {m['no_signal'] or 'none'})")


def do_collect(a):
    st, m = load(), MIXES[a.mix]
    points = []
    for name, r in st["runs"].items():
        if r["mix"] != a.mix or not r.get("ex"):
            continue
        pfs = PF_RE.findall(beaker_log(r["ex"]))
        cells = {}
        for ename, e in st["evals"].items():
            if e["train_run"] != name or not e.get("ex"):
                continue
            for t, g, metric, v, n in SCORE_RE.findall(beaker_log(e["ex"])):
                t = "contradiction" if t == "contra" else t
                if t == e["task"] and metric == PRIMARY_METRIC.get(t, "f1") and v != "None":
                    cells[(t, g)] = (float(v), int(n))
        small = [f"{t}@{g} eval_size={n}" for (t, g), (_, n) in cells.items() if n < 500]
        if small:
            print(f"WARNING {name}: eval_size < 500 in {small}")
        points.append(dict(run=name, arm=r["arm"], budget=r["budget"], seed=r.get("seed", 0),
                           pf=float(pfs[-1]) if pfs else None, cells=cells))
    if not points:
        raise SystemExit(f"no runs for mix {a.mix} in {STATE}")
    table(a.mix, points, f"== {a.mix}: your runs (PF = last flop_meter/actual_pflops in the training log)")
    json.dump([{**p, "cells": {f"{t}@{g}": v for (t, g), v in p["cells"].items()}} for p in points],
              open(os.path.join(HERE, f"results_{a.mix}.json"), "w"), indent=1)


def do_reference(a):
    ref = json.load(open(os.path.join(HERE, "reference.json")))[a.mix]
    pts = [{**p, "cells": {tuple(k.split("@")): tuple(v) for k, v in p["cells"].items()}} for p in ref["points"]]
    table(a.mix, pts, f"== {a.mix}: reference runs ({ref['source']})")
    for note in ref.get("notes", []):
        print("note:", note)


def do_status(a):
    st = load()
    for kind in ("runs", "evals"):
        for name, r in st[kind].items():
            js = subprocess.run([BEAKER, "experiment", "get", r.get("ex") or "-", "--format", "json"],
                                capture_output=True, text=True).stdout
            try:
                s = json.loads(js)[0]["jobs"][-1]["status"]
                state = ("exit=" + str(s.get("exitCode"))) if s.get("finalized") else ("running" if s.get("started") else "pending")
            except Exception:  # noqa: BLE001
                state = "?"
            print(f"{kind[:-1]:5s} {name:52s} {state:10s} https://beaker.org/ex/{r.get('ex')}")


def check_flags():
    """The flag strings above must equal the launchers that produced the reference numbers."""
    sys.path[:0] = [f"{REPO}/research/token_routing/{d}" for d in ("ds64_fast2k", "ds64_fast8k", "online_router")]
    from launch_fast8k import ARM_EXTRA  # noqa: PLC0415
    from launch_otr import ARMS as OTR  # noqa: PLC0415
    assert ARM_EXTRA["lora64all32r20to7"] == RECIPE, "fast8k recipe flags drifted"
    assert ARM_EXTRA["densepf"] == DENSE_PF, "densepf flags drifted"
    assert OTR["lora64r20to7f16"] + " --st-otr-samples-per-row 512" == RECIPE_MT11, "mt11 recipe flags drifted"
    print("flags OK: recipe / mt11 recipe / densepf match launch_fast8k.py and launch_otr.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "eval", "status", "collect", "reference", "check-flags"])
    ap.add_argument("--mix", choices=list(MIXES), default="fast8k")
    ap.add_argument("--arm", default="recipe", help="dense | recipe (new4 also: denselora)")
    ap.add_argument("--budget", default="", help="fast8k: f488..f7808 (rows/task); new4: f4/f2/f1; mt11: f16..f1 "
                    "(fraction of one epoch)")
    ap.add_argument("--flags", default="", help="spin-off: trainer flags for a NEW --arm name (soft-token path, "
                    "LR 1e-5 unless --lr), or extra flags appended to a named arm")
    ap.add_argument("--lr", type=float, default=0.0, help="override the arm's LR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the run name (use for smokes and reruns)")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke: hard-stop after N steps")
    ap.add_argument("--gpus", type=int, default=0, help="override GPUs per run (FLOPs and scores do not depend on it)")
    ap.add_argument("--cluster", default=CLUSTER, help="Beaker cluster(s), comma = any-of")
    ap.add_argument("--run", default="", help="eval: comma list of run names (default: every run of --mix)")
    ap.add_argument("--tasks", default="", help="eval: subset of the mix's tasks")
    ap.add_argument("--ckpt", default="", help="eval: score this checkpoint dir instead of the run's save folder")
    ap.add_argument("--min-runtime", default="2h", help="eval: Beaker minRuntime")
    ap.add_argument("--dry", action="store_true", help="print + validate the Beaker config, submit nothing")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.mode == "train":
        m = MIXES[a.mix]
        if (a.arm not in m["arms"] and not a.flags) or a.budget not in m["budgets"]:
            raise SystemExit(f"{a.mix}: arms {list(m['arms'])} (or a new --arm name with --flags), "
                             f"budgets {list(m['budgets'])}")
        m["sec_per_step"].setdefault(a.arm, m["sec_per_step"]["recipe"])
    {"train": do_train, "eval": do_eval, "status": do_status, "collect": do_collect, "reference": do_reference,
     "check-flags": lambda _: check_flags()}[a.mode](a)


if __name__ == "__main__":
    main()
