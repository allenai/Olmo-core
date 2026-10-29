"""
FAST 2k screening loop for soft-token (pooled-doc KV) training recipes -- the <1 h sibling of
``research/token_routing/ds64/launch_ds64.py``'s 4-20 GPU-hour ladder.

WHY 2k IS ENOUGH TO SCREEN A RECIPE. In the linear-cost regime compaction saves the same FRACTION
of FLOPs at any context length, so *accuracy vs FLOPs at a matched budget* is already well posed at
2k: a recipe that cannot beat dense at matched FLOPs on 2k rows will not start doing so at 32k.
What 2k canNOT tell you is **length generalisation** -- ds64's headline outlier finding is that
keep-1/3 reaches 0.890 at 2k and 0.018 at 32k -- so a fast2k win is a licence to spend a full
ladder, never a substitute for one. Read every fast2k number as "does this recipe learn the task at
all, and at what FLOP cost", and nothing more.

WHAT IS HELD FIXED vs ds64. Same task, same generator, same 2k rung pool, same repaired Qwen3.5-4B
base, same ``--st-*`` flags, same eval file (the 2k rung of the ds64 ladder, 500 rows). What differs
is deliberate and is the whole speedup:

  * 2k-ONLY data (``ds64/fast2k/shards/<task>_f<B>``), not the short-heavy 2k-56k mix;
  * seq-len 4096 instead of 65536 -- so a padded unpacked row wastes ~1.7x, not ~14x;
  * **every arm trains UNPACKED**, dense included. ds64 packs dense and pads the soft arms, which
    makes their rows/step and step counts differ; here every arm sees the SAME rows in the SAME
    order for the same number of steps, so the only thing that varies is the compaction. The FLOP
    meter charges non-pad tokens (``FlopMeterCallback(pad_id=...)``), so padding costs wall-clock,
    never FLOPs, and the matched-FLOP comparison stays honest;
  * 32 rows/step instead of 128, and budgets of 1024 / 2048 / 4096 examples.

    python research/token_routing/ds64_fast2k/launch_fast2k.py --arms dense --budgets 2M,4M,8M dry_run
    python research/token_routing/ds64_fast2k/launch_fast2k.py --arms dense,kvgb50,xhdr17,xhdr50 launch
    python research/token_routing/ds64_fast2k/launch_fast2k.py --arms xhdr17-warm --budgets 4M launch   # warm start
    python research/token_routing/ds64_fast2k/launch_fast2k.py eval          # 2k-rung-only eval of finished runs
    python research/token_routing/ds64_fast2k/launch_fast2k.py status
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
EVAL_LAUNCHER = f"{REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py"
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
SCALE = os.environ.get("F2K_SCALE", "4b")
# the SAME repaired base the ds64 arms use -- never a fresh Qwen3.5 (marker rows are untrained,
# see CLAUDE.md "REPAIR THE BASE CHECKPOINT FIRST")
BASE = {"0.8b": f"{W}/ctc_suite/bases/q35-08b-base-markerfix/model_and_optim",  # fix_marker_embeddings_qwen35.py, 09-02
        "2b": f"{W}/ctc_suite/bases/q35-2b-base-markerfix/model_and_optim",
        "4b": f"{W}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim",
        "9b": f"{W}/ctc_suite/bases/q35-9b-base-markerfix/model_and_optim",
        "27b": f"{W}/ctc_suite/bases/q35-27b-base-markerfix/model_and_optim"}[SCALE]
SHARDS = f"{W}/ds64/fast2k/shards"
CKPTS = f"{W}/ctc_suite/ckpts"
LEDGER = f"{REPO}/research/token_routing/ds64_fast2k/LAUNCH_LEDGER.tsv"
STATE = f"{REPO}/research/token_routing/ds64_fast2k/state.json"
TOKENIZER = f"{W}/hf_tokenizers/Qwen3.5-0.8B-Base"

TASK = os.environ.get("F2K_TASK", "outlier")
BUDGETS = ["2M", "4M", "8M"]            # nominal tokens @2048/example -> 1024 / 2048 / 4096 rows
EXAMPLES = {"2M": 977, "4M": 1953, "8M": 3906}   # = budget_tokens / 2048, measured by the build job
GLOBAL_BATCH = int(os.environ.get("F2K_GB", "32"))      # rows/optimizer step, every arm
# Two GPUs, not one: a 4B model's AdamW state alone is ~32 GB fp32 on top of ~8 GB bf16 params and
# ~8 GB grads, which leaves an 80 GB H100 with almost nothing for activations at a single rank.
# FSDP over 2 ranks halves params+optim and still backfills a packed cluster quickly. Keep
# GLOBAL_BATCH % (micro * ngpu) == 0.
GPUS = int(os.environ.get("F2K_NGPU", "2"))
NODES = 1
# Train-time padded row length. Must be >= the shard's max_example_len (the trainer refuses
# otherwise -- PadToLength would SKIP long rows). 4096 covers the 2k rung's ~2.4k-token examples;
# raise it if the build job's metadata says max_example_len > 4096.
SEQ_LEN = int(os.environ.get("F2K_SEQ_LEN", "4096"))
SOFT_BACKEND = os.environ.get("F2K_SOFT_BACKEND", "flash_2")
# THREE clusters on purpose. The point of fast2k is turnaround, and the ds64 campaign keeps a wave
# of 4-GPU urgent jobs queued on jupiter alone -- a 2-GPU fast2k job then sits behind our OWN work
# (measured 2026-09-14: 38 min queued on jupiter+saturn with zero starts). ceres is a separate H100
# pool ds64 never targets; saturn is A100-80GB, slower per step but fine for a 4B model at 2k.
# Cross-cluster is safe here because fast2k quotes FLOPs and accuracy, never wall-clock.
CLUSTER = os.environ.get("F2K_CLUSTER", "ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale,ai2/saturn-cirrascale")
EVAL_CLUSTER = os.environ.get("F2K_EVAL_CLUSTER", "ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale,ai2/saturn-cirrascale")
WANDB_GROUP = os.environ.get("F2K_WANDB_GROUP", f"f2k-q35-{SCALE}")

# ---------------------------------------------------------------------------------------------
# Arm vocabulary -- deliberately the SAME names and the SAME flag strings as research/token_routing/ds64/launch_ds64.py
# so a fast2k screen and a ds64 ladder run of the same name are the same recipe at two lengths.
# ``--st-header-stop-id 5491`` is outlier-specific: Qwen3.5 fuses `']:'` into ONE token, so id 25
# (a bare ':') NEVER occurs in an outlier header and mark_doc_headers_free would silently fall back
# to its 32-token cap. See records/ds64-handoff.md section 8.
# ---------------------------------------------------------------------------------------------
_XHDR = "--st-header-stop-id 5491 --st-header-stop-count 1"
_XSLOT = f"--st-slot-tokenizer {TOKENIZER}"
ARM_EXTRA = {
    "kv33": "--st-keep-frac 0.3333 --st-keep-mode gold_plus_random",
    "kv17": "--st-keep-frac 0.1667 --st-keep-mode gold_plus_random",
    "kvgb": "--st-gold-blind --st-keep-prob 0.3333",
    "kvgb50": "--st-gold-blind --st-keep-prob 0.5",
    "xhdr00": f"--st-gold-blind --st-keep-prob 0.0 {_XHDR}",
    "xhdr17": f"--st-gold-blind --st-keep-prob 0.1667 {_XHDR}",
    "xhdr33": f"--st-gold-blind --st-keep-prob 0.3333 {_XHDR}",
    "xhdr50": f"--st-gold-blind --st-keep-prob 0.5 {_XHDR}",
    # Content-only slot arms (records/outlier-richer-slot-probe.md). The xhdr* collapse is a
    # READOUT failure, not a keep-policy one: a pooled doc's plain mean input embedding is ~94%
    # shared common-word mass, so every document sits at cosine 0.93-0.94 from the corpus
    # centroid and the slot is unreadable. --st-slot-mode rebuilds it from CONTENT tokens
    # (cmean) and additionally centres + rescales it (cent_cmean), which lifts the eval-side
    # ORACLE readout from 0.390 to 0.731 at 2k (k/n floor 0.225) and 0.075 to 0.239 at 8k, at
    # zero extra tokens and zero extra FLOPs. It has to happen in the slot construction because
    # detach_soft_kv gives the projector no LM gradient on pooled slots. Same header-real,
    # gold-blind construction as xhdr*, so cc17/cm17 are read against xhdr17 (0.234 = the 3/14
    # guess floor) and cc50 against xhdr50 (0.895).
    # --st-slot-tokenizer is pinned to the weka copy so the punctuation half of the stop set is
    # deterministic and needs no HF-hub call at job start (without it the trainer falls back to a
    # frequency-only stop set and says so in the log).
    "cm17": f"--st-gold-blind --st-keep-prob 0.1667 {_XHDR} --st-slot-mode cmean {_XSLOT}",
    "cc17": f"--st-gold-blind --st-keep-prob 0.1667 {_XHDR} --st-slot-mode cent_cmean {_XSLOT}",
    "cc00": f"--st-gold-blind --st-keep-prob 0.0 {_XHDR} --st-slot-mode cent_cmean {_XSLOT}",
    "cc50": f"--st-gold-blind --st-keep-prob 0.5 {_XHDR} --st-slot-mode cent_cmean {_XSLOT}",
}
ARM_MICRO = {"dense": 4}
DEFAULT_MICRO = 2

# Warm-start arms: `<arm>-warm` trains `<arm>`'s recipe from ANOTHER fast2k run's exported weights
# instead of the base. This is the two-phase option: the trainer loads a --base-checkpoint with
# load_optim_state=False / load_trainer_state=False, so phase 2 gets a fresh LR schedule and a
# fresh budget, and phase 1's flags do not carry over. Value = the run name to warm from.
WARM_FROM = {"xhdr17-warm": ("xhdr17", f"{TASK}-kvgb50-4M"),
             "xhdr50-warm": ("xhdr50", f"{TASK}-kvgb50-4M"),
             "xhdr00-warm": ("xhdr00", f"{TASK}-kvgb50-4M")}


def run_name(task, arm, budget):
    return f"f2k-{task}-{arm}-{budget}"


def resolve(arm):
    """(base recipe arm, base checkpoint) for ``arm`` -- resolves the ``-warm`` suffix."""
    if arm in WARM_FROM:
        recipe, src = WARM_FROM[arm]
        return recipe, f"{CKPTS}/f2k-{src}/model_and_optim"
    return arm, BASE


def arm_args(task, arm, budget):
    recipe, base = resolve(arm)
    data = f"{SHARDS}/{task}_f{budget}"
    micro = ARM_MICRO.get(recipe, DEFAULT_MICRO)
    if GLOBAL_BATCH % (micro * GPUS * NODES):
        raise SystemExit(f"global batch {GLOBAL_BATCH} not divisible by micro {micro} x {GPUS * NODES} GPUs")
    common = ["--seq-len", str(SEQ_LEN), "--global-batch", str(GLOBAL_BATCH),
              "--micro-batch-instances", str(micro), "--base-checkpoint", base]
    if recipe == "dense":
        # NOT --pack: every arm trains unpacked so rows/step and step counts match exactly.
        return "full", data, common, ""
    if recipe in ARM_EXTRA:
        return "softtoken", data, common, f"{ARM_EXTRA[recipe]} --attn-backend {SOFT_BACKEND}"
    raise SystemExit(f"unknown arm {arm} (known: dense, {', '.join(ARM_EXTRA)}, "
                     f"plus the -warm variants {', '.join(WARM_FROM)})")


def sh(cmd, timeout=1800):
    r = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"),
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save_state(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def parse_id(out):
    for ln in out.splitlines():
        if "SUBMITTED id=" in ln:
            return ln.split("id=")[1].split()[0]
    return None


def do_launch(args, mode):
    st = load_state()
    rows = []
    for arm in args.arms.split(","):
        for budget in args.budgets.split(","):
            variant, data, largs, extra = arm_args(args.task, arm, budget)
            name = run_name(args.task, arm, budget)
            if name in st["runs"] and mode == "launch" and not args.force:
                print(f"[skip] {name} already launched ({st['runs'][name].get('ex')})")
                continue
            cmd = [sys.executable, "-u", LAUNCHER, "--task", args.task, "--variant", variant,
                   "--model-family", "qwen3_5", "--model-scale", SCALE, "--data-root", data,
                   "--run-name", name, "--exact-run-name", "--num-nodes", str(NODES),
                   "--num-gpus", str(GPUS), "--epochs", "1", "--lr", str(args.lr),
                   "--cluster", CLUSTER, "--wandb-group", args.wandb_group, "--priority", "urgent",
                   "--no-follow", "--no-compile"] + largs + (["--extra-args", extra] if extra else []) + [mode]
            print(" ".join(cmd), flush=True)
            rc, out = sh(cmd)
            os.makedirs(f"{REPO}/research/token_routing/ds64_fast2k/launch_logs", exist_ok=True)
            open(f"{REPO}/research/token_routing/ds64_fast2k/launch_logs/{name}.{mode}.log", "w").write(out)
            ex = parse_id(out)
            tail = "\n".join(out.strip().splitlines()[-2:]).replace("\t", " ")[:200]
            print(f"  -> rc={rc} ex={ex}: {tail[:200]}", flush=True)
            if mode == "launch":
                st["runs"][name] = {"ex": ex, "task": args.task, "arm": arm, "budget": budget,
                                    "steps_expected": -(-EXAMPLES.get(budget, 0) // GLOBAL_BATCH),
                                    "gpus": GPUS * NODES, "state": "S" if ex else "LAUNCH-FAILED"}
                save_state(st)
                rows.append((args.task, arm, budget, name, rc, (ex or "") + " " + tail))
    if rows:
        with open(LEDGER, "a") as f:
            w = csv.writer(f, delimiter="\t")
            for task, arm, budget, name, rc, tail in rows:
                w.writerow(["fast2k", SCALE, task, arm, budget, name, CLUSTER,
                            dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "LAUNCHED" if rc == 0 else "LAUNCH-FAILED", tail])
    print(f"\nwandb: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/{args.wandb_group}")


# 2k-rung-ONLY eval: the same 500-row file and the same evaluator the ds64 orchestrator uses, with
# the rung dict cut down to one entry. run_q4b_beaker_multirung_eval.py already takes the rung set
# as data (--dc-rung-files / --dc-rungs), so no code change is needed to filter rungs.
RUNG_2K = {"outlier": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl",
           "nq": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl",
           "contradiction": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl",
           "oolong": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl"}
TASK_KEY = {"contradiction": "contra", "nq": "nq", "outlier": "outlier", "oolong": "oolong"}


def do_eval(args):
    st = load_state()
    names = args.runs.split(",") if args.runs else list(st["runs"])
    for name in names:
        r = st["runs"].get(name)
        if r is None:
            print(f"[skip] {name} not in state.json")
            continue
        if st["evals"].get(name, {}).get("ex") and not args.force:
            print(f"[skip] eval {name} already launched ({st['evals'][name]['ex']})")
            continue
        task = r["task"]
        files = {task: {"2k": RUNG_2K[task]}}
        cmd = [sys.executable, "-u", EVAL_LAUNCHER, name, EVAL_CLUSTER,
               "--task", TASK_KEY[task], "--variant", "docchunk", "--ckpt", f"{CKPTS}/{name}",
               "--query-position", "after", "--cot-mode", "none", "--tokenizer", TOKENIZER,
               "--ngpu", "2", "--max-test", "500", "--max-length", "8192", "--priority", "urgent",
               # the default batch-size 2 is sized for 40960-ctx generation; at 2k it just wastes
               # the eval's wall-clock, and the fast2k point is the turnaround time.
               "--batch-size", "16",
               "--dc-rung-files", json.dumps(files), "--dc-rungs", "2k"]
        rc, out = sh(cmd)
        ex = parse_id(out)
        open(f"{REPO}/research/token_routing/ds64_fast2k/launch_logs/{name}.eval.log", "w").write(out)
        st["evals"][name] = {"ex": ex, "state": "S" if ex else "LAUNCH-FAILED"}
        save_state(st)
        print(f"eval {name} -> {ex or 'FAILED: ' + out[-300:]}", flush=True)


def beaker_status(ex):
    try:
        r = subprocess.run(["beaker", "experiment", "get", ex, "--format", "json"],
                           env=dict(os.environ, PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:" + os.environ.get("PATH", "")),
                           capture_output=True, text=True, timeout=120)
        j = json.loads(r.stdout)
        j = j[0] if isinstance(j, list) else j
        s = j["jobs"][-1]["status"]
        return ("F", s.get("exitCode")) if s.get("finalized") else (("R" if s.get("started") else "S"), None)
    except Exception:  # noqa: BLE001
        return "?", None


def do_status(args):
    st = load_state()
    changed = False
    for kind in ("runs", "evals"):
        for name, r in st[kind].items():
            if not r.get("ex") or r.get("state") in ("DONE", "FAILED"):
                continue
            s, rc = beaker_status(r["ex"])
            if s == "F":
                r["state"] = "DONE" if rc == 0 else "FAILED"
                r["rc"] = rc
            elif s != "?":
                r["state"] = s
            changed = True
    if changed:
        save_state(st)
    for name, r in sorted(st["runs"].items()):
        e = st["evals"].get(name, {})
        print(f"{name:34} train={r.get('state'):12} {r.get('ex','')}  eval={e.get('state','-'):12} {e.get('ex','')}")
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--arms", default="dense")
    ap.add_argument("--budgets", default=",".join(BUDGETS))
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--wandb-group", default=WANDB_GROUP)
    ap.add_argument("--runs", default="", help="eval: comma run names (default = all in state.json)")
    ap.add_argument("--force", action="store_true", help="relaunch even if state.json has the run")
    ap.add_argument("mode", choices=["launch", "dry_run", "eval", "status"])
    args = ap.parse_args()
    if args.mode in ("launch", "dry_run"):
        do_launch(args, args.mode)
    elif args.mode == "eval":
        do_eval(args)
    else:
        do_status(args)


if __name__ == "__main__":
    main()
