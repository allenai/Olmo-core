"""
Beaker launcher for the CTC-suite trainer (``train_ctc_suite.py``) at 4B/9B scale
(``records/ctc-suite-scaling-plan.md`` §4/§7: "all 4B and 9B training runs" go on
``ai2/jupiter``, urgent priority).

``train_ctc_suite.py`` is a **plain torchrun script** (task-agnostic local trainer), not built on
the ``ExperimentConfig``/``CliContext`` machinery the ``sft_docchunk`` family uses -- so this
launcher does NOT build an ``ExperimentConfig``. Instead it adapts the Beaker half of
``_docchunk_5task_32k_nocpt_common.py`` directly: :func:`olmo_core.internal.common.build_launch_config`
(-> ``OLMoCoreBeakerImage.stable``, urgent priority, weka mount, wandb/beaker-token secrets) wraps
a ``cmd`` that is just ``train_ctc_suite.py``'s own CLI. ``BeakerLaunchConfig`` auto-wraps any
``cmd[0]`` ending in ``.py`` with ``python`` and with ``torchrun`` (since ``num_gpus > 1``), so no
manual torchrun invocation is needed here (unlike the local ``run_ctc_local.sbatch`` launcher).

Data / base staging (weka is NOT reachable from Berkeley -- two-step S3 relay, see ``beaker.md``):
shards and the per-scale converted base are staged under
``/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/{shards,bases}/...`` -- see
``debug/ctc_suite_beaker_smoke/upload_to_s3.sh`` (cubbins /data -> S3) and the gantry S3->weka
sync job (``beaker.md`` template) run once from this host.

Usage::

    python -u recipes/train/research_trainer/beaker_ctc_suite.py \\
        --task outlier --variant chunked-mix --model-scale 4b \\
        --run-name ctc-outlier-cmix-4b --num-nodes 2 --epochs 1 \\
        launch

    # dry run (prints the resolved BeakerLaunchConfig, does not submit):
    python -u recipes/train/research_trainer/beaker_ctc_suite.py \\
        --task outlier --variant chunked-mix --model-scale 4b --run-name ctc-outlier-smoke \\
        --num-nodes 1 --max-steps 100 dry_run
"""

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path

from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.launch.beaker import BeakerEnvVar, OLMoCoreBeakerImage

REPO_ROOT = Path(__file__).resolve().parents[3]
WANDB_ENTITY = "prasanns-allen-institute-for-ai"
WANDB_PROJECT = "memory-networks"

# GPUs/node default for this family (H100 80GB jupiter nodes). Overridable per launch: sub-node
# slots (2/4 GPUs) backfill into freed slices far faster than whole nodes when the cluster is
# packed -- the 2026-08-17 wave sat queued for hours behind 975/975 slots. Wall-clock scales
# ~linearly with fewer GPUs (global batch is held at 8 instances via grad accum), so 2-GPU is only
# viable for short-seq tasks; seq-40960 runs need >=4 GPUs to finish inside a day.
NUM_GPUS = 8


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", required=True, help="suite task name, e.g. outlier / grouping")
    ap.add_argument(
        "--variant",
        required=True,
        choices=["full", "chunked", "chunked-mix", "sparselandmark", "pooledkv", "ffnmoe", "softtoken", "kvroute", "flexcompute"],
    )
    ap.add_argument(
        "--model-scale",
        required=True,
        choices=["0.6b", "0.8b", "1.7b", "2b", "4b", "9b", "27b", "7b", "7b-hybrid", "7b-noswa", "3b", "8b"],
        help="4b/9b for Qwen, 7b for OLMo-3 (7b-hybrid = allenai/Olmo-Hybrid-7B, the same-size "
        "same-data linear-attention twin used as the hybrid-vs-not control; 7b-noswa is the "
        "superseded no-sliding-window variant). 0.8b/2b were originally local-only (hence the old "
        "'0.8B trains locally' note), but the model-scale study needs them on Beaker too: on "
        "lambda they get 3 serialized lanes and two of those lanes are preemptible (runs were "
        "lost at 1h45m). Note --base-checkpoint must be passed explicitly for 0.8b -- the default "
        "path template yields 'q35-0.8b-base-modelonly' while the real directory is "
        "'q35-08b-base-modelonly'.",
    )
    ap.add_argument(
        "--model-family",
        default="qwen3_5",
        choices=["qwen3_5", "qwen3", "olmo3", "llama"],
        help="qwen3_5 (GDN hybrid, default), qwen3 (dense), olmo3 (OLMo 3 7B dense), or llama "
        "(Llama-3.x dense/GQA; 3b=Llama-3.2-3B, 8b=Llama-3.1-8B). Selects the default base "
        "checkpoint and is passed through to the trainer (which also auto-detects it from the "
        "shard). ``olmo3`` and ``llama`` have no default base -- pass --base-checkpoint "
        "explicitly. Note a llama run also needs llama-tokenized SHARDS: the marker ids come from "
        "RESERVED_IDS['llama'] and a qwen3_5 shard fed to a llama model is silently wrong.",
    )
    ap.add_argument(
        "--run-name", required=True, help="fresh name per config -- silent auto-resume trap"
    )
    # Long-context arms (256k): context parallelism + YaRN + packing, all passed to the trainer.
    ap.add_argument(
        "--cp-degree", type=int, default=0, help="Ulysses CP degree (dense<=8, hybrid<=4); 0=off"
    )
    ap.add_argument(
        "--rope-yarn-factor",
        type=float,
        default=0.0,
        help="YaRN factor (e.g. 8 for 32k->256k); dense only",
    )
    ap.add_argument("--rope-old-context", type=int, default=32768)
    ap.add_argument(
        "--rope-theta",
        type=float,
        default=0.0,
        help="override base rope_theta (NTK ext; ~8e6 for Qwen3-4B@256k)",
    )
    ap.add_argument(
        "--pack",
        action="store_true",
        help="bin-pack whole examples (needed for a mixed 8k..256k shard)",
    )
    ap.add_argument(
        "--activation-checkpointing", default=None, choices=["auto", "full", "budget", "none"]
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="training seed passthrough. Needed to test whether a result reproduces across seeds -- "
        "a single run per configuration cannot distinguish a real effect from seed variance.",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile (the trainer compiles by DEFAULT, so every Beaker run has so "
        "far been compiled). Needed to reproduce the frugal local recipe -- full activation "
        "checkpointing + no compile -- which the local launcher uses to avoid the "
        "full-AC + CP + compile recompute-metadata mismatch. NOTE: --activation-checkpointing "
        "budget REQUIRES compile, so pass 'full' alongside this.",
    )
    ap.add_argument(
        "--ac-budget", type=float, default=None, help="budget for --activation-checkpointing budget"
    )
    ap.add_argument(
        "--shard-degree", type=int, default=0, help="FSDP shard_degree override (0=auto)"
    )
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument(
        "--num-gpus",
        type=int,
        default=NUM_GPUS,
        help="GPUs per node (default 8). 2/4-GPU slots backfill a packed cluster much faster; "
        "pass a matching --shard-degree and keep --global-batch fixed for optimizer parity.",
    )
    ap.add_argument("--num-nodes", type=int, default=2, help="2 nodes for 4B, 4 for 9B per plan §4")
    ap.add_argument("--epochs", type=int, default=1, help="plan directive: 1 epoch at 20k examples")
    ap.add_argument("--seq-len", type=int, default=40960)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument(
        "--global-batch",
        type=int,
        default=None,
        help="instances/optimizer-step across all ranks; default = num_nodes*8 (1 instance/GPU, grad-accum 1)",
    )
    ap.add_argument("--micro-batch-instances", type=int, default=1)
    ap.add_argument(
        "--max-steps", type=int, default=0, help="0 = full (epoch-bounded); set for smoke tests"
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="shard dir override; default {weka_root}/checkpoints/prasanns/ctc_suite/shards/{task}_train",
    )
    ap.add_argument(
        "--base-checkpoint",
        default=None,
        help="model_and_optim distcp subdir override; default {weka_root}/checkpoints/prasanns/"
        "ctc_suite/bases/q35-{scale}-base-modelonly/model_and_optim",
    )
    ap.add_argument("--wandb-group", default=None, help="default: ctc-suite-<task>")
    ap.add_argument("--priority", default="urgent", help="ALWAYS urgent per project directive")
    ap.add_argument(
        "--min-runtime",
        default="1h",
        help="Beaker minRuntime (guaranteed runtime before preemption). Default 1h: on jupiter, urgent "
        "jobs with minRuntime=0 effectively never schedule (2026-10-19: 0 of ~2,250 urgent 1-7-GPU "
        "jobs scheduled in 3 days had minRuntime=0; every pending 0s job was days old). Pass '0' only "
        "to opt back into always-preemptible.",
    )
    ap.add_argument(
        "--no-follow",
        dest="follow",
        action="store_false",
        default=True,
        help="submit and return immediately instead of streaming the job's logs. The default "
        "(follow) blocks for the WHOLE training run, so a script launching several runs in a "
        "loop would submit the first and then sit on it for hours -- pass this when batch-"
        "launching. The experiment URL is printed either way.",
    )
    ap.add_argument(
        "--no-allow-dirty",
        dest="allow_dirty",
        action="store_false",
        default=True,
        help="require a clean tree (default: allow_dirty=True, matching _docchunk_5task_32k_nocpt_common.py -- "
        "this checkout is a shared working tree actively modified by concurrent jobs and is rarely clean; "
        "gantry always clones the PUSHED commit regardless of local dirtiness)",
    )
    ap.add_argument(
        "--keep-final-checkpoint",
        action="store_true",
        default=False,
        help="keep the full model+optimizer+train-state checkpoint CheckpointerCallback writes at "
        "the end of the run. Off by default: eval and rebasing both load the model-only save from "
        "the --save-checkpoint block after fit(), so that final step*/ dir is read by nothing and "
        "costs ~83G per 7B run (~49G at 4B). Leaving it on regrew weka from 7.2 TB to 23 TB between "
        "2026-07-28 and 2026-08-29. Pass this only for a long run on a preemptible queue where a "
        "resume point is worth the space; --save-interval N gives mid-run resume points separately.",
    )
    ap.add_argument(
        "--extra-args",
        default="",
        help="verbatim extra CLI for train_ctc_suite.py (shlex-split), e.g. the --ffn-moe-* / "
        "--st-* knobs of the ffnmoe / softtoken arms",
    )
    ap.add_argument(
        "--exact-run-name",
        action="store_true",
        help="use --run-name verbatim (no timestamp suffix) so the save folder is predictable, "
        "e.g. for a stage-2 run that warm-starts from stage 1's export. Reusing a name RESUMES.",
    )
    # Run registry (ops/state/runs.jsonl). Registration never fails or blocks a launch.
    ap.add_argument(
        "--experiment",
        default=None,
        help="ops registry experiment key (= ops/experiments/<name>/). Default: $OPS_EXPERIMENT, "
        "else a known run-name prefix, else the slugged wandb group (ctc-suite-<task> -> ctc_suite_<task>).",
    )
    ap.add_argument(
        "--arm", default=None, help="ops registry arm label (default: --variant); e.g. a campaign's arm name"
    )
    ap.add_argument(
        "--expected-minutes",
        type=float,
        default=None,
        help="expected wall-clock; `ops status` flags the run OVERRUN past 2x this",
    )
    ap.add_argument("mode", choices=["launch", "dry_run"])
    return ap.parse_args()


def _ops_safe():
    """``ops/opslib/safe.py`` (stdlib-only, never raises), or ``None`` if ops cannot be imported."""
    try:
        if str(REPO_ROOT / "ops") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "ops"))
        from opslib import safe

        return safe
    except Exception as e:  # noqa: BLE001 - registry problems never fail a launch
        print(f"[ops] WARNING: cannot import ops ({type(e).__name__}: {e}); run not registered", flush=True)
        return None


def register_run(opts, *, run_name, save_folder, wandb_group, git_commit=None, git_branch=None,
                 dirty=None, job_id=None):
    """Append this launch to the ops run registry. Never raises; returns the row or ``None``."""
    try:
        ops = _ops_safe()
        if ops is None:
            return None
        if dirty:
            print(
                f"[ops] note: working tree is dirty; Beaker runs the PUSHED commit {git_commit}, "
                "not your local edits",
                flush=True,
            )
        return ops.register_launch(
            git_dir=REPO_ROOT,
            run_id=run_name,
            experiment=opts.experiment
            or ops.derive_experiment(run_name, wandb_group, default=f"ctc_suite_{opts.task}"),
            kind="train",
            cluster="beaker",
            job_id=job_id,
            task=opts.task,
            arm=opts.arm or opts.variant,
            ckpt_path=save_folder,
            wandb_url=f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/groups/{wandb_group}",
            git_commit=git_commit,
            git_branch=git_branch,
            dirty=dirty,
            command=shlex.join(sys.argv),
            expected_minutes=opts.expected_minutes,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ops] WARNING: registry write failed ({type(e).__name__}: {e}); continuing", flush=True)
        return None


def main() -> None:
    opts = parse_args()

    # --cluster may be a comma list (any-of placement); the weka root is the same on all of them
    root_dir = get_root_dir(opts.cluster.split(",")[0])  # -> /weka/oe-training-default/ai2-llm on jupiter
    work_dir = get_work_dir(root_dir)

    data_root = (
        opts.data_root or f"{root_dir}/checkpoints/prasanns/ctc_suite/shards/{opts.task}_train"
    )
    # Per-family default base: qwen3_5 = the converted hybrid base already on weka; qwen3 = the
    # trained-marker-repaired dense base (staged for the qwen3-vs-qwen3.5 contradiction comparison).
    if opts.base_checkpoint:
        base_checkpoint = opts.base_checkpoint
    elif opts.model_family == "qwen3_5":
        base_checkpoint = f"{root_dir}/checkpoints/prasanns/ctc_suite/bases/q35-{opts.model_scale}-base-modelonly/model_and_optim"
    else:
        base_checkpoint = f"{root_dir}/checkpoints/prasanns/ctc_suite/bases/qwen3-{opts.model_scale}-base-trainedmark/model_and_optim"
    run_name = (
        opts.run_name
        if opts.exact_run_name
        else f"{opts.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    save_folder = f"{root_dir}/checkpoints/prasanns/ctc_suite/ckpts/{run_name}"
    wandb_group = opts.wandb_group or f"ctc-suite-{opts.task}"
    world_size = opts.num_nodes * opts.num_gpus
    global_batch = opts.global_batch or world_size  # 1 instance/GPU/step, grad-accum 1

    cmd = [
        "recipes/train/research_trainer/train_ctc_suite.py",
        "--task",
        opts.task,
        "--data",
        data_root,
        "--variant",
        opts.variant,
        "--model-scale",
        opts.model_scale,
        "--seq-len",
        str(opts.seq_len),
        "--epochs",
        str(opts.epochs),
        "--lr",
        str(opts.lr),
        "--global-batch",
        str(global_batch),
        "--micro-batch-instances",
        str(opts.micro_batch_instances),
        "--base-checkpoint",
        base_checkpoint,
        "--work-dir",
        f"{work_dir}-ctc-suite",
        "--save-folder",
        save_folder,
        "--run-name",
        run_name,
        "--wandb-group",
        wandb_group,
        "--wandb-entity",
        WANDB_ENTITY,
        "--model-family",
        opts.model_family,
        "--save-checkpoint",
    ]
    if not opts.keep_final_checkpoint:
        cmd += ["--no-final-checkpoint"]
    if opts.max_steps:
        cmd += ["--max-steps", str(opts.max_steps)]
    if opts.cp_degree:
        cmd += ["--cp-degree", str(opts.cp_degree)]
    if opts.rope_yarn_factor:
        cmd += [
            "--rope-yarn-factor",
            str(opts.rope_yarn_factor),
            "--rope-old-context",
            str(opts.rope_old_context),
        ]
    if opts.rope_theta:
        cmd += ["--rope-theta", str(opts.rope_theta)]
    if opts.pack:
        cmd += ["--pack"]
    if opts.no_compile:
        cmd += ["--no-compile"]
    if opts.seed is not None:
        cmd += ["--seed", str(opts.seed)]
    if opts.activation_checkpointing:
        cmd += ["--activation-checkpointing", opts.activation_checkpointing]
    if opts.ac_budget is not None:
        cmd += ["--ac-budget", str(opts.ac_budget)]
    if opts.shard_degree:
        cmd += ["--shard-degree", str(opts.shard_degree)]
    if opts.extra_args:
        cmd += shlex.split(opts.extra_args)

    launch_config = build_launch_config(
        name=run_name,
        cmd=cmd,
        cluster=opts.cluster,
        root_dir=root_dir,
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=opts.num_nodes,
        num_gpus=opts.num_gpus,
    )
    launch_config.priority = opts.priority
    launch_config.min_runtime = None if opts.min_runtime in ("", "0", "0s") else opts.min_runtime
    launch_config.allow_dirty = opts.allow_dirty
    # Doc-chunked attention's FlexAttention/eager mask path needs headroom beyond what the default
    # caching allocator leaves reachable at 40960 tokens (same fragmentation fix as the proven
    # sft_docchunk 40k Beaker runs).
    launch_config.env_vars.append(
        BeakerEnvVar(name="PYTORCH_CUDA_ALLOC_CONF", value="expandable_segments:True")
    )

    print(
        f"[beaker_ctc_suite] task={opts.task} variant={opts.variant} scale={opts.model_scale} "
        f"run_name={run_name} nodes={opts.num_nodes} world_size={world_size} "
        f"global_batch={global_batch} epochs={opts.epochs} max_steps={opts.max_steps or 'full'}"
    )
    print(f"[beaker_ctc_suite] data={data_root}")
    print(f"[beaker_ctc_suite] base_checkpoint={base_checkpoint}")
    print(f"[beaker_ctc_suite] save_folder={save_folder}")
    print(
        f"[beaker_ctc_suite] wandb: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/groups/{wandb_group}"
    )
    print(f"[beaker_ctc_suite] cmd={' '.join(cmd)}")

    if opts.mode == "dry_run":
        print("[beaker_ctc_suite] DRY RUN -- not submitting. Resolved BeakerLaunchConfig:")
        print(launch_config)
        return

    if not opts.follow:
        # build_launch_config defaults step_soft_timeout to 10 min, and BeakerLaunchConfig rejects
        # any step timeout (or Slack notification) when follow=False -- both are implemented by
        # watching the streamed log. Clearing them is what makes --no-follow usable; without this
        # the launch raises "Step soft timeout requires 'follow=True'" and submits NOTHING.
        launch_config.step_timeout = None
        launch_config.step_soft_timeout = None

    # The commit gantry ships is the launch config's git ref (the PUSHED HEAD), not the work tree.
    git = getattr(launch_config, "git", None)
    reg = dict(
        run_name=run_name,
        save_folder=save_folder,
        wandb_group=wandb_group,
        git_commit=getattr(git, "ref", None),
        git_branch=getattr(git, "branch", None),
    )
    try:
        reg["dirty"] = bool(git.is_dirty) if git is not None else None
    except Exception:  # noqa: BLE001 - is_dirty shells out to git
        reg["dirty"] = None
    # follow=True blocks until the job ENDS, so register before submitting (job id filled in after).
    followed = opts.follow and register_run(opts, **reg) is not None
    try:
        workload = launch_config.launch(follow=opts.follow)
    except Exception:
        if followed:
            _ops_safe().update_run(run_name, final_state="failed", state_source="beaker_ctc_suite: launch raised")
        raise
    # Print the id/name on ONE greppable line. `workload` renders as a multi-hundred-line protobuf
    # dump, so a batch launcher that keeps only the tail of stdout loses the id entirely and cannot
    # record what it just submitted.
    wl_id = getattr(workload, "id", None) or getattr(
        getattr(workload, "experiment", None), "id", None
    )
    wl_name = getattr(workload, "name", None) or run_name
    print(f"[beaker_ctc_suite] SUBMITTED id={wl_id} name={wl_name}")
    if followed:
        _ops_safe().update_run(run_name, job_id=wl_id)
    else:
        register_run(opts, job_id=wl_id, **reg)
    if wl_id:
        print(f"[beaker_ctc_suite] url=https://beaker.org/ex/{wl_id}")
    print(f"[beaker_ctc_suite] launched: {workload}")


if __name__ == "__main__":
    sys.exit(main())
