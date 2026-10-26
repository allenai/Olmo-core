"""
Submit a multi-rung NATIVE long-context eval as a **Beaker** job for one trained checkpoint --
fully on Beaker, NO local sync. This is the on-Beaker counterpart of the LOCAL driver
``run_q4b_stl_multirung_eval.sbatch``.

The eval reads everything from weka (eval code + eval data + the distcp checkpoint), so the job needs
no data copied to the node:

  * eval CODE + DATA + the goal-rung ladder files live in the weka eval bundle
    (``checkpoints/prasanns/_eval_bundle`` + ``_eval_bundle_eval500``); upload/refresh it with
    ``upload_lc_eval_bundle.sh`` before launching.
  * the checkpoint is the just-trained distcp step dir under
    ``checkpoints/prasanns/<run_name>/step*`` -- the on-node runner auto-globs the latest complete step.

The actual eval logic is the on-node runner ``run_beaker_multirung_eval.sh`` (uploaded to the bundle):
it runs ``torchrun --nproc_per_node=8 scripts/eval/eval_lc_native.py`` (8-way DP, native olmo_core
generate -- NO HF/vLLM, required for landmark/compressive). gantry's own torchrun wrapping is disabled
(``torchrun=False``) so the runner can issue its own (and multiple, for rerank) torchrun calls.

Per-task rungs (matching the local driver):
  contra 2k,8k,16k,32k | nq 3k,8k,16k,32k | outlier 3k,8k,16k,32k | oolong 8k,16k,32k |
  rerank CE files k20/k50/k100 (NDCG@10 + Kendall-tau). ``--max-test-samples 600``.

Usage::

    # one (run, task) -> one Beaker job; variant inferred from the run name
    PYTHONPATH=src python recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py \\
        q4b-dense-contra-ladder32k-10k ai2/neptune --task contra

    # all 5 tasks for a run (5 jobs)
    PYTHONPATH=src python recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py \\
        q4b-landmark-nq-ladder32k-10k ai2/neptune --task all

    # validate without submitting
    PYTHONPATH=src python recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py \\
        q4b-dense-contra-ladder32k-10k ai2/neptune --task contra --dry-run

    # CTC-suite rows (--suite-rows, 2026-10-22): score any of the 10 mt11 / setA tasks on the suite's
    # own r8k + r2k rows staged on weka (research/token_routing/ds64_fast8k/stage_suite_rows_beaker.sh),
    # graded by the vendored ctc spec exactly as olmo-eval grades the suite row. Docchunk ladder only.
    PYTHONPATH=src python recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py \\
        f8k-mt11-dense-s480-evhotpotqa ai2/jupiter-cirrascale-2 --task hotpotqa --variant docchunk \\
        --ckpt <weka model dir> --suite-rows --query-position both --dry-run
"""

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.utils import prepare_cli_environment

ALL_TASKS = [
    "contra",
    "nq",
    "rerank",
    "outlier",
    "oolong",
    "fiqa",
    "scifact",
    "outlier_review",
    "contra_fever",
]
VARIANTS = ["dense", "landmark", "compressive", "docchunk"]
#: CTC-suite tasks the docchunk ladder grades with the vendored ctc spec (eval_lc_native_docchunk_ladder.py
#: SUITE_TASKS). Not in ALL_TASKS, so ``--task all`` is unchanged; they need --suite-rows or --dc-rung-files.
SUITE_TASKS = ["hotpotqa", "qdmatch_nq", "xabsence", "reorder", "strmatch", "grouping"]
#: --suite-rows: launcher task -> (ladder task, suite subset, {rung label: file tokens}, ctc-grade the
#: base task). Contradiction reads ``contradiction_iid`` (its r2k rung is rung_2560, olmo-eval's alias);
#: nq / outlier / oolong / contradiction / rerank are graded by the ctc spec (ladder --ctc-grade) as olmo-eval does.
SUITE_ROWS = {
    **{t: (t, t, {"8k": 8192, "2k": 2048}, False) for t in SUITE_TASKS},
    "nq": ("nq", "nq", {"8k": 8192, "2k": 2048}, True),
    "outlier": ("outlier", "outlier", {"8k": 8192, "2k": 2048}, True),
    "oolong": ("oolong", "oolong", {"8k": 8192, "2k": 2048}, True),
    "contra": ("contradiction", "contradiction_iid", {"8k": 8192, "2k": 2560}, True),
    # 2026-10-25: rerank's suite rows (HF subset "rerank"), ctc-graded (ce_pos_recall). They must be
    # staged first: stage_suite_rows_beaker.sh (fetch_suite_eval_rows.py now includes rerank).
    "rerank": ("rerank", "rerank", {"8k": 8192, "2k": 2048}, True),
}
#: rows staged by stage_suite_rows_beaker.sh (verified against ds64_fast8k/suite_rows_MANIFEST.json)
SUITE_ROWS_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle/ctc_suite_rows"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _ops():
    """``ops/opslib/safe.py`` (stdlib-only; its calls never raise), or ``None``."""
    try:
        if str(REPO_ROOT / "ops") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "ops"))
        from opslib import safe

        return safe
    except Exception as e:  # noqa: BLE001
        print(f"[ops] WARNING: cannot import ops ({type(e).__name__}: {e}); eval not registered", flush=True)
        return None


def eval_pass_id(lc):
    """This eval pass's ops run id: the Beaker job name (``ev-<task>-<run>``), suffixed with a
    timestamp if that name is already registered (every pass is its own run)."""
    ops = _ops()
    return ops.unique_run_id(lc.name) if ops is not None else lc.name


def register_eval(args, lc, task, job_id, run_id=None):
    """Append this eval pass to the ops run registry (``ops/state/runs.jsonl``).

    ``run_id`` = :func:`eval_pass_id` (also exported to the job as ``OPS_EVAL_RUN_ID``, so the result
    JSON's ``_meta.eval_run_id`` matches), ``parent_run`` = the trained run. Never raises: a registry
    problem prints one warning and the launch carries on.
    """
    try:
        ops = _ops()
        if ops is None:
            return None
        git = getattr(lc, "git", None)
        # experiment: --experiment, else the training run's (registry), else $OPS_EXPERIMENT / name prefix
        return ops.register_eval_pass(
            run_id or lc.name,
            args.run_name,
            experiment=args.experiment,
            task=task,
            cluster="beaker",
            job_id=job_id,
            node=None,
            ckpt_path=args.ckpt or None,
            git_commit=getattr(git, "ref", None),
            git_branch=getattr(git, "branch", None),
            command=shlex.join(sys.argv),
            expected_minutes=args.expected_minutes,
        )
    except Exception as e:  # noqa: BLE001 - never fail a launch over the registry
        print(f"[ops] WARNING: registry write failed ({type(e).__name__}: {e}); continuing", flush=True)
        return None


def variant_from_run_name(run_name: str) -> str:
    # docchunk run names use the explicit "docchunk_dense" token; check docchunk first.
    if "docchunk" in run_name:
        return "docchunk"
    found = [v for v in ("dense", "landmark", "compressive") if v in run_name]
    if len(found) != 1:
        raise SystemExit(
            f"could not infer variant from run name {run_name!r} (found {found}); pass --variant."
        )
    return found[0]


def build_eval_launch_config(
    *,
    run_name,
    task,
    variant,
    cluster,
    step,
    ckpt,
    results_dir,
    prompt_format,
    query_position,
    ngpu,
    max_test,
    max_length,
    batch_size,
    priority,
    ladder_version,
    xlong,
    xlong_rungs,
    cot_mode,
    tokenizer="",
 dc_rung_files: str = "", dc_rungs: str = "", extra_env=()):
    # `cluster` may be a comma list for any-of placement (build_launch_config splits it the same
    # way). The weka root is identical across the weka-tagged clusters, so resolve it from the first
    # -- get_root_dir() takes ONE cluster name and 404s on a comma list.
    root_dir = get_root_dir(cluster.split(",")[0])  # e.g. /weka/oe-training-default/ai2-llm
    # Eval CODE now ships IN the cloned repo (recipes/eval/legacy_native/ctc_eval); the runner runs from the repo root
    # (gantry cwd). DATA still comes from weka (the runner derives BUNDLE/EVAL500 from WEKA_LLM=root_dir).
    runner = "recipes/eval/legacy_native/run_beaker_multirung_eval.sh"

    # The on-node runner reads its inputs from env; gantry torchrun wrapping is disabled so the runner
    # can drive its own 8-way `torchrun`. cmd[0]="bash" => not auto-prefixed with `python`.
    inner = (
        "".join(f"{kv} " for kv in extra_env)
        + f"RUN={run_name} TASK={task} VARIANT={variant} STEP='{step}' CKPT='{ckpt}' "
        f"EVAL_OUT_DIR='{results_dir}' PROMPT_FORMAT='{prompt_format}' "
        f"QUERY_POSITION='{query_position}' "
        f"MAX_TEST={max_test} MAX_LENGTH={max_length} BATCH_SIZE={batch_size} NGPU={ngpu} "
        f"LADDER_XLONG={int(xlong)} XLONG_RUNGS='{xlong_rungs}' COT_MODE='{cot_mode}' "
        f"LADDER_VERSION={ladder_version} "
        + (f"TOKENIZER={tokenizer} " if tokenizer else "")
        + (f"DC_RUNG_FILES='{dc_rung_files}' " if dc_rung_files else "")
        + (f"DC_RUNGS='{dc_rungs}' " if dc_rungs else "")
        + f"WEKA_LLM={root_dir} bash {runner}"
    )
    cmd = ["bash", "-lc", inner]

    # Ladder in the job name for anything but v2, so a v2 and a v3 submission for the same
    # (task, run) are distinguishable in the Beaker UI instead of two identically-named jobs.
    name = f"ev-{task}-{run_name}" + (f"-{ladder_version}" if ladder_version != "v2" else "")
    launch_config = build_launch_config(
        name=name,
        cmd=cmd,
        cluster=cluster,
        root_dir=root_dir,
        task_name="eval",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=ngpu,
    )
    launch_config.torchrun = False  # the runner issues its own torchrun(s)
    launch_config.allow_dirty = True  # ship the (uncommitted) launcher via an ephemeral ref
    launch_config.priority = priority
    # Beaker minRuntime (2026-10-19): on jupiter, urgent jobs with minRuntime=0 effectively never
    # schedule (0 of ~2,250 urgent 1-7-GPU jobs scheduled in 3 days had it). Override with
    # BEAKER_MIN_RUNTIME (empty or "0" = always-preemptible).
    _mr = os.environ.get("BEAKER_MIN_RUNTIME", "1h")
    launch_config.min_runtime = None if _mr in ("", "0", "0s") else _mr
    launch_config.step_soft_timeout = (
        None  # we submit with follow=False (don't block on many evals);
    )
    launch_config.step_timeout = None  # the 10-min default soft timeout forbids follow=False
    return launch_config


def suite_rows_args(task, root):
    """(DC_RUNG_FILES json, DC_RUNGS, extra env) scoring ``task`` on the staged CTC-suite rows."""
    import json

    ltask, subset, rungs, ctc_grade = SUITE_ROWS[task]
    files = {ltask: {lab: f"{root}/{subset}/rung_{tok}.jsonl" for lab, tok in rungs.items()}}
    # REQUIRE_EVAL_SIZE: the runner fails the job if any rung scored 0 rows (a missing/unread rung
    # file otherwise writes an empty result with rc=0)
    env = ["REQUIRE_EVAL_SIZE=1"] + ([f"LADDER_CTC_GRADE={ltask}"] if ctc_grade else [])
    return json.dumps(files, separators=(",", ":")), ",".join(rungs), env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_name", help="trained run name (checkpoints under checkpoints/prasanns/<run_name>)"
    )
    ap.add_argument("cluster", help="Beaker cluster, e.g. ai2/neptune (weka-backed)")
    ap.add_argument(
        "--task",
        default="all",
        help=f"comma list from {ALL_TASKS + SUITE_TASKS}, or 'all' (= {ALL_TASKS}; one Beaker job "
        "per task). The CTC-suite tasks need --suite-rows (or --dc-rung-files) and --variant docchunk.",
    )
    ap.add_argument(
        "--variant",
        default=None,
        choices=VARIANTS,
        help="dense|landmark|compressive|docchunk (default: infer from run name).",
    )
    ap.add_argument(
        "--step", default="", help="pin a step dir (e.g. step580); default = latest complete."
    )
    ap.add_argument(
        "--ckpt",
        default="",
        help="ABSOLUTE weka step dir to eval ANY checkpoint, e.g. "
        "/weka/oe-training-default/ai2-llm/checkpoints/<you>/<run>/step1234 . "
        "Overrides run_name globbing (run_name is then just a results label).",
    )
    ap.add_argument(
        "--results-dir",
        default="",
        help="ABSOLUTE weka dir for the per-task result JSONs "
        "(default: checkpoints/prasanns/<run_name>/eval).",
    )
    ap.add_argument(
        "--query-position",
        choices=["both", "after", "before"],
        default="both",
        help="Prompt layout; MUST match the SFT shards the model was trained on. "
        "xlong5_2k256k_qwen35 -> both (default, and what every result before "
        "2026-08-11 used); xlong5_2k256k_qwen35_qafter -> after. It lands in the "
        "inner eval_command, so the launch ledger records it automatically.",
    )
    ap.add_argument(
        "--prompt-format",
        choices=["chat", "raw", "alpaca"],
        default="chat",
        help="chat=SFT (apply_chat_template, matches training); raw=BASE/CPT models; alpaca=legacy.",
    )
    ap.add_argument("--max-test", type=int, default=600)
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument(
        "--batch-size", type=int, default=2
    )  # 40960-ctx generation on ~48GB neptune GPUs; 8 OOMs
    ap.add_argument(
        "--ngpu",
        type=int,
        default=2,
        help="GPUs per eval job (data-parallel over examples). 4B model fits on 1-2 GPUs; "
        "2 lets ~4x more evals run concurrently than 8 and fits fragmented free slots.",
    )
    ap.add_argument(
        "--tokenizer",
        default="",
        help="HF tokenizer id, forwarded to the on-node runner as TOKENIZER. "
        "MUST match the model family: the runner defaults to Qwen/Qwen3-4B "
        "(vocab 151936), so a Qwen3.5 checkpoint (vocab 248320) evaluated "
        "without this scores ~0 on EVERY task while the job reports success -- "
        "wrong token ids, not a broken model. Qwen3.5 -> Qwen/Qwen3.5-4B-Base.",
    )
    ap.add_argument("--priority", default="urgent")  # never below urgent (user directive)
    ap.add_argument(
        "--ladder-version",
        choices=["v2", "v3"],
        default="v2",
        help="v2 is the ONLY supported ladder: every rung of a task shares the SAME "
        "500 questions/answers and only distractors vary (reads the "
        "_eval_bundle_eval500_v2 weka bundle). v1 is DISABLED -- its per-rung "
        "question resampling put eval-set noise into every rung-to-rung delta, "
        "and both the runner and eval_lc_native.py now reject it.",
    )
    ap.add_argument(
        "--cot-mode",
        choices=["none", "plan"],
        default="none",
        help="docchunk OOLONG only: 'plan' builds the CoT prefill (match a CoT-trained "
        "checkpoint); default 'none' keeps the no-CoT eval byte-identical.",
    )
    ap.add_argument(
        "--xlong",
        action="store_true",
        help="OPT-IN: also run the ultra-long 64k/128k/256k/512k/1M/2M rungs (contra|nq|outlier). "
        "Forces bs=1 + raises MAX_LENGTH on-node. Use an 80GB GPU (ai2/jupiter); "
        "256k needs bs=1 single-GPU. Files must be built by build_xlong_rungs.py "
        "and uploaded to the v2 eval bundle.",
    )
    ap.add_argument(
        "--xlong-rungs",
        default="64k,128k",
        help="which xlong sizes to add when --xlong: 64k,128k,256k,512k,1M,2M. Anything above "
        "256k needs a YaRN serving copy (past Qwen3.5's native 262,144) and more "
        "than one 80GB GPU (KV ~32KB/token: 2M alone is ~69GB).",
    )
    ap.add_argument(
        "--dc-rung-files",
        default="",
        help="docchunk variant only: JSON {task: {label: weka path}} of explicit rung files "
        "(overrides the bundle's defaults; used to score marker-trained models on the exact "
        "rung files a dense campaign used)",
    )
    ap.add_argument("--dc-rungs", default="", help="docchunk: comma rung labels to score (pairs with --dc-rung-files)")
    ap.add_argument(
        "--suite-rows",
        action="store_true",
        help="docchunk only: score on the CTC-suite eval rows (r8k + r2k, 500 each) staged on weka "
        f"({sorted(SUITE_ROWS)}; contra = contradiction_iid), with the ctc-spec grading olmo-eval uses "
        "(LADDER_CTC_GRADE for nq/outlier/oolong/contra/rerank). Sets --dc-rung-files/--dc-rungs per task. "
        "Use --query-position both for setA / mt11-trained checkpoints.",
    )
    ap.add_argument("--suite-rows-root", default=SUITE_ROWS_ROOT, help="<subset>/rung_<tokens>.jsonl tree")
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="extra env var for the on-node runner (repeatable), e.g. "
        "OLMO_LANDMARK_SPARSE_DECODE=1 to route landmark decode through the sparse path.",
    )
    ap.add_argument(
        "--experiment",
        default=None,
        help="ops registry experiment key (default: the training run's, else $OPS_EXPERIMENT, else "
        "guessed from the run name's prefix, else 'unassigned').",
    )
    ap.add_argument(
        "--expected-minutes", type=float, default=None, help="ops registry: `ops status` flags OVERRUN past 2x"
    )
    ap.add_argument("--dry-run", action="store_true", help="build + print the job, do NOT submit.")
    args = ap.parse_args()

    for kv in args.env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=[^\s'\"]*", kv):
            raise SystemExit(f"--env expects KEY=VAL with no spaces/quotes, got {kv!r}")
    prepare_cli_environment()

    variant = args.variant or variant_from_run_name(args.run_name)
    tasks = (
        ALL_TASKS if args.task == "all" else [t.strip() for t in args.task.split(",") if t.strip()]
    )
    bad = [t for t in tasks if t not in ALL_TASKS + SUITE_TASKS]
    if bad:
        raise SystemExit(f"unknown task(s) {bad}; choose from {ALL_TASKS + SUITE_TASKS}.")
    if args.suite_rows:
        if variant != "docchunk":
            raise SystemExit("--suite-rows is graded by the docchunk ladder only (--variant docchunk)")
        if args.dc_rung_files or args.dc_rungs:
            raise SystemExit("--suite-rows sets the rung files itself; drop --dc-rung-files/--dc-rungs")
        bad = [t for t in tasks if t not in SUITE_ROWS]
        if bad:
            raise SystemExit(f"--suite-rows: no suite rows for {bad}; choose from {sorted(SUITE_ROWS)}")
    else:
        bad = [t for t in tasks if t in SUITE_TASKS and not args.dc_rung_files]
        if bad:
            raise SystemExit(f"{bad} have no default rung files: pass --suite-rows (or --dc-rung-files)")

    print(
        f"=== Beaker multirung eval | run={args.run_name} variant={variant} "
        f"tasks={tasks} cluster={args.cluster} dry_run={args.dry_run} ==="
    )
    for task in tasks:
        dc_rung_files, dc_rungs, extra_env = args.dc_rung_files, args.dc_rungs, list(args.env)
        if args.suite_rows:
            dc_rung_files, dc_rungs, suite_env = suite_rows_args(task, args.suite_rows_root)
            extra_env = suite_env + extra_env
        # docchunk now evaluates the FULL ladder (all 9 tasks incl. OOD) via
        # eval_lc_native_docchunk_ladder.py (box-marker chunked prefill + bs=1 KV-cached decode).
        lc = build_eval_launch_config(
            run_name=args.run_name,
            task=task,
            variant=variant,
            cluster=args.cluster,
            step=args.step,
            ckpt=args.ckpt,
            results_dir=args.results_dir,
            prompt_format=args.prompt_format,
            query_position=args.query_position,
            ngpu=args.ngpu,
            max_test=args.max_test,
            max_length=args.max_length,
            batch_size=args.batch_size,
            priority=args.priority,
            ladder_version=args.ladder_version,
            xlong=args.xlong,
            xlong_rungs=args.xlong_rungs,
            cot_mode=args.cot_mode,
            tokenizer=args.tokenizer,
            dc_rung_files=dc_rung_files,
            dc_rungs=dc_rungs,
            extra_env=extra_env,
        )
        # Each eval pass is its own ops run; the driver writes this id into the result's _meta.
        run_id = eval_pass_id(lc)
        lc.cmd[-1] = f"OPS_EVAL_RUN_ID={run_id} " + lc.cmd[-1]
        print(f"\n--- [{task}] {lc.name} (ops run_id {run_id}) ---")
        print(f"    cmd: {lc.cmd[-1]}")
        if args.dry_run:
            print("    [dry-run] not submitting.")
            continue
        workload = lc.launch(follow=False)
        wl_id = getattr(workload, "id", None) or getattr(getattr(workload, "experiment", None), "id", None)
        print(f"    submitted: {wl_id}")
        print(f"SUBMITTED id={wl_id}")
        register_eval(args, lc, task, wl_id, run_id=run_id)
    print("\n=== done ===")


if __name__ == "__main__":
    main()
