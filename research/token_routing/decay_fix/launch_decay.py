"""
Large-data decay fixes for the token-drop recipe (tokdrop_lr, 2026-10-30): launch / score / eval.

RECIPE = ``lora64all32r20to7`` of research/token_routing/ds64_fast8k/launch_fast8k.py (whole-model rank-64 LoRA over a
frozen base, two-stage online router keep 20% after block 7 and 7% after block 16, no full-context steps, LR 1e-5),
on the FRESH mt4 pools (``mt4f<rows>``: rows/task never repeated), 2 GPUs, gb 16, micro 2, seq 12288 -- verbatim the
flags of the -tde recipe runs (f8k-mt4-lora64all32r20to7-f<rows>-tde-lr1e-05). Arms add ONE fix each:

  rl25   ReLoRA merge-and-restart every 25% of the steps (merge B.A into the base, re-init, reset Adam, 20-step LR
         re-warm). Same per-step FLOPs.
  lw35   late keep-widening: 35% / 15% instead of 20% / 7% in the last third (charged by the FLOP meter).

    python research/token_routing/decay_fix/launch_decay.py launch --arms rl25,lw35 --budgets f3904,f7808
    python research/token_routing/decay_fix/launch_decay.py launch-ladder --runs <run>,...   # per-task 8k+2k f1
    python research/token_routing/decay_fix/launch_decay.py ce --runs <run>,...      # answer-CE scoring (1 GPU)

State (job ids) in research/token_routing/decay_fix/state.json. Every launch is registered with ops (beaker_ctc_suite /
the multirung launcher register themselves; the CE jobs are registered here).
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
D = os.path.dirname(os.path.abspath(__file__))
STATE = f"{D}/state.json"
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
CKPTS = f"{W}/ctc_suite/ckpts"
SHARDS = f"{W}/ds64/fast8k/shards"
BASE = f"{W}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
CLUSTER = os.environ.get("DF_CLUSTER", "ai2/jupiter-cirrascale-2")
SUFFIX = "-tdf"
LR = "1e-05"
RECIPE = ("--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 "
          "--st-otr-explore-frac 0.1 --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 "
          "--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-scale 32 "
          "--st-otr-early-lora-blocks 32")
ARMS = {
    "lora64all32r20to7": RECIPE,  # the recipe itself (control re-run under this commit, if wanted)
    "lora64all32r20to7rl25": RECIPE + " --st-otr-relora-frac 0.25 --st-otr-relora-rewarm 20",
    "lora64all32r20to7lw35": RECIPE + " --st-otr-late-keep 0.667:0.35,0.15",
    "lora64all32r20to7rl25lw35": RECIPE + " --st-otr-relora-frac 0.25 --st-otr-relora-rewarm 20 "
                                          "--st-otr-late-keep 0.667:0.35,0.15",
}
SHORT = {"rl25": "lora64all32r20to7rl25", "lw35": "lora64all32r20to7lw35", "rl25lw35": "lora64all32r20to7rl25lw35",
         "recipe": "lora64all32r20to7"}
MT4 = ["contradiction", "nq", "outlier", "oolong"]
#: launch_fast8k.py RUNGS / TASK_KEY / eval_cmd (the per-task ladder every fast8k mt4 run was scored with)
RUNGS = {"outlier": {"8k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                     "2k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl"},
         "nq": {"8k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
                "2k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl"},
         "contradiction": {
             "8k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
             "2k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl"},
         "oolong": {
             "8k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
             "2k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl"}}
TASK_KEY = {"nq": "nq", "outlier": "outlier", "oolong": "oolong", "contradiction": "contra"}
PY = sys.executable


def sh(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})}, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


def parse_id(out):
    m = re.search(r"\b(01[0-9A-Z]{24})\b", out)
    return m.group(1) if m else None


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}, "ce": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def run_name(arm, budget, seed=0):
    return f"f8k-mt4-{arm}-{budget}{SUFFIX}{f'-sd{seed}' if seed else ''}-lr{LR}"


def do_launch(a):
    st = load()
    for short in a.arms.split(","):
        arm = SHORT.get(short, short)
        for budget in a.budgets.split(","):
            name = run_name(arm, budget, a.seed) + (f"-smoke{a.max_steps}" if a.max_steps else "")
            if name in st["runs"] and not a.force:
                print(f"[skip] {name}: {st['runs'][name]}")
                continue
            extra = ARMS[arm] + " --attn-backend flash_2" + (f" --seed {a.seed}" if a.seed else "")
            cmd = [PY, "-u", f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py", "--task", "mt4",
                   "--variant", "softtoken", "--model-family", "qwen3_5", "--model-scale", "4b",
                   "--data-root", f"{SHARDS}/mt4{budget}", "--run-name", name, "--exact-run-name",
                   "--num-nodes", "1", "--num-gpus", "2", "--epochs", "1", "--lr", LR, "--cluster", CLUSTER,
                   "--wandb-group", "f8k-q35-4b", "--priority", "urgent", "--no-follow", "--no-compile",
                   "--seq-len", "12288", "--global-batch", "16", "--micro-batch-instances", "2",
                   "--base-checkpoint", BASE] + (["--max-steps", str(a.max_steps)] if a.max_steps else []) + [
                   "--extra-args", extra, "dry_run" if a.dry else "launch"]
            print(" ".join(cmd), flush=True)
            rc, out = sh(cmd, env={"OPS_EXPERIMENT": "tokdrop_lr", "PYTHONPATH": f"{REPO}/src"})
            ex = parse_id(out)
            print(f"  -> rc={rc} ex={ex}\n" + "\n".join(out.strip().splitlines()[-4:]), flush=True)
            if not a.dry:
                st["runs"][name] = {"ex": ex, "arm": arm, "budget": budget, "seed": a.seed,
                                    "steps_expected": a.max_steps or int(budget[1:]) * 4 // 16}
                save(st)


def do_ladder(a):
    """Per-task generative ladder (8k + 2k, 500 rows, no-cot, query after) = launch_fast8k.eval_cmd verbatim."""
    st = load()
    for name in a.runs.split(","):
        for task in MT4:
            ename = f"{name}-ev{task}"
            if ename in st["evals"] and not a.force:
                print(f"[skip] {ename}: {st['evals'][ename]}")
                continue
            cmd = [PY, "-u", f"{REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py", ename, CLUSTER,
                   "--task", TASK_KEY[task], "--variant", "docchunk", "--ckpt", f"{CKPTS}/{name}",
                   "--query-position", "after", "--cot-mode", "none", "--tokenizer", "Qwen/Qwen3.5-0.8B",
                   "--ngpu", "2", "--max-test", "500", "--priority", "urgent", "--max-length", "20000",
                   "--batch-size", "8", "--dc-rung-files", json.dumps({task: RUNGS[task]}), "--dc-rungs", "8k,2k"]
            if a.dry:
                cmd.append("--dry-run")
            rc, out = sh(cmd, env={"OPS_EXPERIMENT": "tokdrop_lr", "PYTHONPATH": f"{REPO}/src"})
            ex = parse_id(out)
            print(f"{ename} -> rc={rc} ex={ex}", flush=True)
            if not a.dry:
                st["evals"][ename] = {"ex": ex, "train_run": name, "task": task}
                save(st)


def do_ce(a):
    """Answer-CE scoring, 1 GPU per call (several checkpoints per job with --group)."""
    st = load()
    names = a.runs.split(",")
    groups = [names[i:i + a.group] for i in range(0, len(names), a.group)]
    for g in groups:
        job = f"dfce-{g[0][8:60]}" if len(g) == 1 else f"dfce-{a.tag}-{groups.index(g)}"
        extra = " ".join(f"--ckpt {CKPTS}/{n}" for n in g) + f" --weka-out {W}/ctc_suite/decay_fix/{job}.json"
        cmd = [PY, f"{REPO}/research/ffn_compute/flop_scaling/beaker_bench_launch.py", "--cluster", CLUSTER,
               "--script", "research/token_routing/decay_fix/score_answer_ce.py", "--name", job,
               "--min-runtime", "1h", "--priority", "urgent", "--extra", extra]
        if a.dry:
            print(" ".join(cmd))
            continue
        rc, out = sh(cmd, env={"PYTHONPATH": f"{REPO}/src"})
        ex = parse_id(out)
        print(f"{job} ({g}) -> rc={rc} ex={ex}", flush=True)
        st["ce"][job] = {"ex": ex, "ckpts": g}
        save(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["launch", "launch-ladder", "ce"])
    ap.add_argument("--arms", default="rl25,lw35")
    ap.add_argument("--budgets", default="f3904,f7808")
    ap.add_argument("--runs", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke: cut at this many steps (name gets -smoke<N>)")
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--tag", default="grp")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"launch": do_launch, "launch-ladder": do_ladder, "ce": do_ce}[a.mode](a)


if __name__ == "__main__":
    main()
