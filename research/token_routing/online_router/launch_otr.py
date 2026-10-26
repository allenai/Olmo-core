"""Launch / evaluate ONLINE-router training arms on the fast8k multi-task mix (Beaker jupiter, 2 GPUs).

Same geometry as the tokdrop_lr screen (``research/token_routing/ds64_fast8k/launch_fast8k.py``, which this
imports read-only): repaired Qwen3.5-4B base, unpacked rows, 16 rows/step, micro 2, seq 10240, per-example FLOP
meter, eval per task on the 8k + 2k rungs x 500 rows with the legacy native multirung eval (no-cot, query after).
Arms are ``--st-keep-token-rule online`` flag strings (``olmo_core.nn.online_token_router``); see
``records/online-token-router.md`` for the design and the run matrix.

    python research/token_routing/online_router/launch_otr.py --arms otr30L4 --budgets s488 launch
    python research/token_routing/online_router/launch_otr.py --runs f8k-mt4-otr30L4-s488-otr-lr2e-06 eval
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
SUFFIX = os.environ.get("OTR_SUFFIX", "-otr")
WANDB_GROUP = os.environ.get("OTR_WANDB_GROUP", "f8k-otr")

_OTR = "--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --attn-backend flash_2"


def otr(keep, *, layer=4, drop=0, share=1.0, explore=0.1, dense_every=8, extra=""):
    """``dense_every``: every N-th step trains on the FULL context. Those steps give the router unbiased
    full-pass gradient targets for every token and the online fidelity estimate
    (online_router/fid_mass_*); they are ordinary dense steps, so the FLOP meter charges them in full."""
    feat = f"--st-otr-drop-layer {drop}" if drop else f"--st-otr-feature-layer {layer}"
    return (f"{_OTR} --st-otr-keep-frac {keep} --st-otr-explore-frac {explore} --st-otr-doc-share {share} "
            f"--st-otr-dense-every {dense_every} {feat} {extra}").strip()


#: arm -> trainer flags. Names: otr<keep%><L<feature layer> | M<mid-stack drop layer>>[-variant].
ARMS = {
    "otr30L4": otr(0.3),                                       # smoke: input drop, layer-4 prefix features
    "otr30M8": otr(0.3, drop=8),                               # mid-stack: dropped tokens skip blocks 8..31
    "otr30L4-ds05": otr(0.3, share=0.5),                       # half the budget row-level (leak test)
    "otr30L4-gold": otr(0.3, share=0.5, extra="--st-otr-use-gold"),
    "otr15L4": otr(0.15),
    "otr50L4": otr(0.5),
    "otr30L4-de4": otr(0.3, dense_every=4),                    # every 4th step full context
    "otr30L4-de0": otr(0.3, dense_every=1000),                 # no full-context steps after warmup
    "rnd30": otr(0.3, explore=1.0),                            # control: same budget, uniformly random tokens
    # "where vs which" (parent session, 2026-10-22): the mid-stack drop point with a router-free selection
    "rnd30M8": otr(0.3, drop=8, extra="--st-otr-score random"),
    "fl30M8": otr(0.3, drop=8, extra="--st-otr-score first_last"),
    "otr10M8": otr(0.1, drop=8),
    # cheaper mid-stack settings (parent session, 2026-10-22): shallower full-context prefix, fewer dense steps
    "otr10M4": otr(0.1, drop=4),
    "otr05M8": otr(0.05, drop=8),   # frontier push (parent, 2026-10-22): keep 5%
    # router + whole-block skip: every token also skips GDN blocks 20 / 24 / 28 (full attention at 3, 7, 11, ...)
    "otr10M8sk3": otr(0.1, drop=8, extra="--train-skip-layers 20,24,28"),
    "otr10M7": otr(0.1, drop=7),    # right before the block-7 full attention (direct test of the full-attention hypothesis)
    "rnd10M7": otr(0.1, drop=7, extra="--st-otr-score random"),   # strmatch test (parent, 2026-10-23): does random fail mt11 strmatch too?
    # allocation ablation (Prasann, 2026-10-23: "do you ever ablate the budget not being kept per document? Also not
    # having headers?"): global budget (doc share 0 / 0.5), and each document's first 8 body tokens always kept
    # aggressive cuts (Prasann, 2026-10-23: "much more aggressive cuts to computation"): the floors dominate
    # (blocks [0, 7) on every token ~22% of dense, the full-context step every 8 ~12%)
    "otr10M7ng": otr(0.1, drop=7, extra="--st-otr-early-nograd"),          # early blocks forward-only on routed steps
    "otr10M7tail": otr(0.1, drop=7, dense_every=0, extra="--st-otr-dense-tail-frac 0.1"),  # routed, then a dense tail
    "otr10M7de0": otr(0.1, drop=7, dense_every=0),                         # no full-context steps
    "otr10M7ngtail": otr(0.1, drop=7, dense_every=0, extra="--st-otr-dense-tail-frac 0.1 --st-otr-early-nograd"),
    # pyramid drop: keep 0.5 from block 3, 0.1 from 7, 0.03 from 16 (each stage its own router at its own layer)
    "mrg05M7": otr(0.05, drop=7, extra="--st-otr-merge-dropped"),   # merge instead of drop, keep 5%
    "mrg10M7": otr(0.1, drop=7, extra="--st-otr-merge-dropped"),
    # memory + FLOPs (Prasann, 2026-10-23: "ideas which can save both on FLOPs and memory"): NO full-context steps
    # per-document token floor (grad-composition: strmatch's ~25-token documents keep ~3 tokens at 10%, mostly header)
    "otr10M7f8": otr(0.1, drop=7, extra="--st-otr-min-per-doc 8"),
    "otr10M7f16": otr(0.1, drop=7, extra="--st-otr-min-per-doc 16"),
    # combined: warmup-then-freeze at half the steps + per-document floor 16, no full-context steps
    "fa247f16M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-freeze-after 247 --st-otr-min-per-doc 16"),  # mt11s720: 495 steps
    # 2026-10-24 levers on the 11-task mix (no full-context steps): progressive freeze, per-task routers, + floor 16
    "prog4712f16M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-freeze-schedule 4:0.25,7:0.5,12:0.75 --st-otr-min-per-doc 16"),
    "otr10M7f16pt": otr(0.1, drop=7, dense_every=0, extra="--st-otr-min-per-doc 16 --st-otr-per-task-key-tokens 11"),
    "lora64fa420r20to7f16": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-freeze-after 420 --st-otr-min-per-doc 16"),  # 85% of 495
    "lora64r30to10f16": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.3,16:0.10 --st-otr-early-lora 64 --st-otr-min-per-doc 16"),
    "lora64r20to7f32": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-min-per-doc 32"),
    "lora64r20to7f16pt": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-min-per-doc 16 --st-otr-per-task-key-tokens 11"),
    "lora64r20to10f16": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.10 --st-otr-early-lora 64 --st-otr-min-per-doc 16"),
    "lora64r20to7f16": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-min-per-doc 16"),
    # LoRA on every block (adapter scope K=32; matched the recipe on mt4s1464 at LR 1e-5, 2026-10-25)
    "lora64K32r20to7f16": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-min-per-doc 16"),
    # partial MLP on the LoRA-all recipe (mt4, no floor): v2 topk vs random (2026-10-25)
    "lora64K32r20to7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32"),
    # overnight hill-climb on the LoRA-all recipe (2026-10-26)
    "lora64K32r20to7pt": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-per-task-key-tokens 11"),
    "lora64K32th": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-keep-mode threshold --st-otr-row-cap 2"),
    "lora64K32ks": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-keep-sched 1.33:0.67"),
    "lora64K32cx": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-ctx-feats"),
    "lora64K32to": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-task-onehot 11"),
    "lora64K32r20at5": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 5:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32"),
    "lora64K32cxr20at5": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 5:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-ctx-feats"),
    "lora64K32m50c50": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-later-mass-target 0.5 --st-otr-later-max-keep-frac 0.5"),
    "lora64K32m70c50": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-later-mass-target 0.7 --st-otr-later-max-keep-frac 0.5"),
    "lora64K32mlp": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-mlp-hidden 64"),
    "lora64K32mlpr20at5": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 5:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-mlp-hidden 64"),
    "lora64K32rg": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-ridge-groups"),
    "lora64K32dg": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --st-otr-diag-every 20"),
    "lora64K32r20to12": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,12:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32"),
    "lora64K32r20to7tk50": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --ffn-width-frac 0.5 --ffn-width-mode topk"),
    "lora64K32r20to7tk75": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --ffn-width-frac 0.75 --ffn-width-mode topk"),
    "lora64K32r20to7fw75a": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --ffn-width-frac 0.75"),
    "lora64K32r20to7fw90a": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-early-lora-blocks 32 --ffn-width-frac 0.9"),
    "lora64fa371r20to10f16": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.10 --st-otr-early-lora 64 --st-otr-freeze-after 371 --st-otr-min-per-doc 16"),
    "lora64fa371r20to7f16": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-early-lora 64 --st-otr-freeze-after 371 --st-otr-min-per-doc 16"),
    "otr10M7f16sk3": otr(0.1, drop=7, dense_every=0, extra="--st-otr-min-per-doc 16 --train-skip-layers 20,24,28"),
    "r20to7f16": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 7:0.2,16:0.07 --st-otr-min-per-doc 16"),
    "lora64fa371f16M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-early-lora 64 --st-otr-freeze-after 371 --st-otr-min-per-doc 16"),  # mt11s720: 75% of 495
    "lora64f16M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-early-lora 64 --st-otr-min-per-doc 16"),
    "lora16f16M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-early-lora 16 --st-otr-min-per-doc 16"),
    "prog4712f16M7pt": otr(0.1, drop=7, dense_every=0, extra="--st-otr-freeze-schedule 4:0.25,7:0.5,12:0.75 --st-otr-min-per-doc 16 --st-otr-per-task-key-tokens 11"),
    # freeze-point sweep at mt4s1464 (366 steps): 25% / 50% / 75%
    "fa92M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-freeze-after 92"),
    "fa183M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-freeze-after 183"),
    "fa275M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-freeze-after 275"),
    "fa30M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-freeze-after 30"),   # warmup-then-freeze (s488: 30 of 122)
    "ng0M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-early-nograd"),       # early blocks forward-only always
    "frz0M7": otr(0.1, drop=7, dense_every=0, extra="--st-otr-freeze-early"),      # + frozen (no grads, no Adam state)
    "pyr3de0": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 3:0.5,7:0.1,16:0.03"),
    "pyr3frz0": otr(0.1, drop=7, dense_every=0, extra="--st-otr-stages 3:0.5,7:0.1,16:0.03 --st-otr-freeze-early"),
    "mrg05M7de0": otr(0.05, drop=7, dense_every=0, extra="--st-otr-merge-dropped"),
    "pyr3": otr(0.1, drop=7, extra="--st-otr-stages 3:0.5,7:0.1,16:0.03"),
    "pyr3ng": otr(0.1, drop=7, extra="--st-otr-stages 3:0.5,7:0.1,16:0.03 --st-otr-early-nograd"),
    "pyr2": otr(0.1, drop=7, extra="--st-otr-stages 7:0.1,16:0.03"),
    "otr10M7ds0": otr(0.1, drop=7, share=0.0),
    "otr10M7ds50": otr(0.1, drop=7, share=0.5),
    "otr10M7hdr8": otr(0.1, drop=7, extra="--st-otr-keep-first 8"),
    "otr10M5": otr(0.1, drop=5),
    "otr10M6": otr(0.1, drop=6),    # split-point bracket around 8 (parent, 2026-10-22): before the block-7 full attention
    "otr10M12": otr(0.1, drop=12),  # later split: floor 12/32 of the dense cost
    "otr10M1": otr(0.1, drop=1),   # Prasann: route after block 1 (one GDN layer of full-context mixing)
    # exploration-share ablation on otr10M8 (Prasann, 2026-10-22); x10 = otr10M8 itself, x100 = rnd10M8
    "otr10M8x00": otr(0.1, drop=8, explore=0.0),
    "otr10M8x30": otr(0.1, drop=8, explore=0.3),
    "otr10M8x50": otr(0.1, drop=8, explore=0.5),
    "rnd10M8": otr(0.1, drop=8, extra="--st-otr-score random"),
    # router-target ablation on otr10M8 (Prasann, 2026-10-22): gnorm = otr10M8 itself; gxh |<g,h>|; dh = residual change
    # ||h^8 - h^0|| (activation strength, does not vanish once a token is learned); mix = rank average of gnorm and dh
    "otr10M8gxh": otr(0.1, drop=8, extra="--st-otr-target gxh"),
    "otr10M8dh": otr(0.1, drop=8, extra="--st-otr-target dh"),
    "otr10M8mix": otr(0.1, drop=8, extra="--st-otr-target mix"),
    # fit-free rule (parent, 2026-10-22): each document's top 10% by ||h^8 - h^0|| at block 8; no exploration.
    # dh10M8 keeps otr10M8's charged full-context step every 8 (matched structure); dh10M8de0 is the pure rule
    "dh10M8": otr(0.1, drop=8, extra="--st-otr-score dh"),
    "dh10M8de0": otr(0.1, drop=8, dense_every=0, extra="--st-otr-score dh"),
    "otr10M8de16": otr(0.1, drop=8, dense_every=16),
    "otr10M4de16": otr(0.1, drop=4, dense_every=16),
}


#: per-mix training window: >= the shard's max_example_len (the trainer refuses a shorter one). mt11 rows reach
#: 10,231 tokens at s240 and the larger first-N shards can be longer.
SEQ_LEN = {"mt4": 10240, "mt11": 12288}
#: single-task fast8k pools (``--task``): the parent session's per-task -tdp campaign uses seq 12288
SINGLE_TASK_SEQ = 12288


def run_name(arm, budget, lr, mix="mt4", seed=0):
    """``f8k-<mix>-<arm>[sd<seed>]-<budget><suffix>-lr<lr>``; the seed rides on the arm so the Pareto collector's
    ``f8k-<mix>-<arm>-s<rows>-tdb-lr<lr>`` pattern still parses (the arm is ``otr30M8sd1``)."""
    return f"f8k-{mix}-{arm}{f'sd{seed}' if seed else ''}-{budget}{SUFFIX}-lr{lr:g}"


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def do_launch(a, mode):
    st = load()
    for arm in a.arms.split(","):
        for budget in a.budgets.split(","):
            if a.task:
                # single-task pool <task>_<budget> (budget g16M / g16M_eg); name f8k-<task>-<arm>-16M[_eg]...
                name = f"f8k-{a.task}-{arm}{f'sd{a.seed}' if a.seed else ''}-{budget.lstrip('g')}{SUFFIX}-lr{a.lr:g}"
            else:
                name = run_name(arm, budget, a.lr, a.mix, a.seed)
            if name in st["runs"] and mode == "launch" and not a.force:
                print(f"[skip] {name} ({st['runs'][name]['ex']})")
                continue
            rows, _, steps = budget[1:].partition("@")
            data = f"{SHARDS}/{a.task}_{budget}" if a.task else f"{SHARDS}/{a.mix}s{rows}"
            if a.task:
                steps = ""
            cmd = [sys.executable, "-u", LAUNCHER, "--task", a.task or a.mix, "--variant", "softtoken",
                   "--model-family", "qwen3_5", "--model-scale", "4b", "--data-root", data,
                   "--run-name", name, "--exact-run-name", "--num-nodes", "1", "--num-gpus", "2",
                   "--epochs", str(a.epochs), "--lr", str(a.lr), "--cluster", "ai2/jupiter-cirrascale-2",
                   "--wandb-group", WANDB_GROUP, "--priority", "urgent", "--no-follow", "--no-compile",
                   "--seq-len", str(SINGLE_TASK_SEQ if a.task else SEQ_LEN[a.mix]), "--global-batch", "16",
                   "--micro-batch-instances", "2",
                   "--seed", str(a.seed),
                   "--base-checkpoint", BASE, "--experiment", "learned_router", "--extra-args", ARMS[arm]]
            if steps:
                cmd[cmd.index("--extra-args"):cmd.index("--extra-args")] = ["--max-steps", steps]
            rc, out = sh(cmd + [mode])
            ex = parse_id(out)
            print(f"{name} -> rc={rc} {ex or out[-400:]}")
            if mode == "launch" and ex:
                st["runs"][name] = {"ex": ex, "arm": arm, "budget": budget, "lr": a.lr, "mix": a.task or a.mix,
                                    "seed": a.seed, **({"task": a.task} if a.task else {})}
                save(st)


def do_eval(a):
    st = load()
    for name in a.runs.split(","):
        single = st["runs"].get(name, {}).get("task")
        for task in ([single] if single else MT4_TASKS):
            ename = f"{name}-ev{task}"
            if ename in st["evals"] and not a.force:
                print(f"[skip] {ename}")
                continue
            files = {task: RUNGS[task]}
            cmd = [sys.executable, "-u", EVAL_LAUNCHER, ename, "ai2/jupiter-cirrascale-2",
                   "--task", TASK_KEY[task], "--variant", "docchunk", "--ckpt", f"{CKPTS}/{name}",
                   "--query-position", "after", "--cot-mode", "none", "--tokenizer", TOKENIZER,
                   "--ngpu", "2", "--max-test", "500", "--priority", "urgent", "--max-length", "20000",
                   "--batch-size", "8", "--dc-rung-files", json.dumps(files), "--dc-rungs", ",".join(RUNGS[task])]
            rc, out = sh(cmd)
            ex = parse_id(out)
            print(f"eval {ename} -> {ex or 'FAILED: ' + out[-400:]}")
            if ex:
                st["evals"][ename] = {"ex": ex, "train_run": name, "task": task}
                save(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="otr30L4")
    ap.add_argument("--budgets", default="s488", help="s<rows/task>[@<max steps>] (mt4s488 / mt4s976 / mt4s1464)")
    ap.add_argument("--lr", type=float, default=2e-6, help="the rule arms' LR in the tokdrop_lr screen")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--mix", default="mt4", choices=sorted(SEQ_LEN), help="task mix: shard <mix>s<rows>")
    ap.add_argument("--task", default="", help="single-task fast8k pool <task>_<budget> (budget g16M / g16M_eg) "
                    "instead of a mix; evals on that task only")
    ap.add_argument("--seed", type=int, default=0, help="run seed (data order, router exploration); tags the arm")
    ap.add_argument("--runs", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("mode", choices=["launch", "dry_run", "eval"])
    a = ap.parse_args()
    do_eval(a) if a.mode == "eval" else do_launch(a, a.mode)


if __name__ == "__main__":
    main()
