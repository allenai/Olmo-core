"""Owned launch-time checks and resumable node entrypoint for three MT branches."""

import hashlib
import os
import re
import subprocess
import sys
import time

import olmoe3_adaptive_mt_plan as p
from olmoe3_lr_sweep_watch import atomic_json


def register(r):
    """Register only this run's namespace with the existing verified-upload service."""
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    store = StateStore(p.base.CONTROL, p.base.STATE)
    store.register(
        Registration(
            run_id=r.run_id,
            lineage_id=r.run_id,
            checkpoint_root=str(r.root),
            bucket_id=p.base.BUCKET,
            remote_prefix=r.prefix,
            deletion_mode="apply",
            min_local_checkpoints=2,
            delete_grace_seconds=3600,
        )
    )


def node(schedule):
    """Restore, save twice and strictly resume before the remaining authorized MT."""
    from beaker import Beaker
    from olmoe3_hero_decay_runtime import verify_runtime
    from olmoe3_profile_node import resolve_ready_leader
    from olmoe3_profile_topology import validate_topology

    verify_runtime()
    r = p.run(schedule)
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    assert int(os.environ["BEAKER_REPLICA_COUNT"]) == r.nodes
    assert int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]) == p.GPUS_PER_NODE
    exp, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    r.root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        subprocess.run([sys.executable, p.SCRIPT, "config", schedule], check=True)
        register(r)
        from olmoe3_adaptive_decay_checks import check

        atomic_json(p.AUTO / "gpu_checks" / f"{exp}.json", check("cuda"))
        subprocess.run([sys.executable, p.SCRIPT, "data", schedule], check=True)
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, timeout=30)
    atomic_json(
        p.AUTO / "topology" / exp / f"{job}.json", validate_topology(topology, p.GPUS_PER_NODE)
    )
    ready = p.AUTO / "rendezvous" / exp
    atomic_json(ready / f"{job}.json", dict(job=job, rank=rank))
    with Beaker.from_env(check_for_upgrades=False) as b:
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(b, b.workload.get(exp), ready, r.nodes)
            if leader:
                break
            time.sleep(10)
        else:
            raise TimeoutError("Current replica rendezvous")
    _, host = leader
    checkpoints = sorted(
        int(x.name[4:])
        for x in r.root.glob("step*")
        if re.fullmatch(r"step\d+", x.name) and (x / ".metadata.json").exists()
    )
    start = checkpoints[-1] if checkpoints else p.START
    source = r.root / f"step{start}" if checkpoints else r.source
    p.base.validate_checkpoint(source, start if checkpoints else p.SOURCE_STEP, r.batch,
                               r.gpus if checkpoints else r.as_dict()["source_world_size"])
    port = 28000 + int(hashlib.sha256(exp.encode()).hexdigest()[:8], 16) % 1000
    for i, stop in enumerate((p.START + 2, p.START + 4, p.END)):
        if start >= stop:
            continue
        env = dict(
            os.environ,
            QKGAIN_RUN=r.run_id,
            QKGAIN_START=str(start),
            QKGAIN_STOP=str(stop),
            QKGAIN_LOAD=str(source),
        )
        print("ADAPTIVE_TRAIN_SEGMENT", r.run_id, start, stop, flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nnodes={r.nodes}",
                f"--nproc-per-node={p.GPUS_PER_NODE}",
                f"--node-rank={rank}",
                "--rdzv-backend=static",
                f"--rdzv-endpoint={host}:{port+i}",
                f"--rdzv-id={exp}-{i}",
                "--rdzv-conf=read_timeout=1200",
                "--max-restarts=0",
                p.SCRIPT,
                "train",
                r.run_id,
                "ai2/holmes",
            ],
            env=env,
            check=True,
        )
        source, start = r.root / f"step{stop}", stop
        p.base.validate_checkpoint(source, stop, r.batch, r.gpus)
    if rank == 0:
        atomic_json(
            r.root / "audit/success.json",
            dict(
                passed=True,
                run=r.as_dict(),
                step=p.END,
                gpus=r.gpus,
                checkpoint=str(source),
                experiment=exp,
                checkpoint_metadata_sha256=hashlib.sha256(
                    (source / ".metadata.json").read_bytes()
                ).hexdigest(),
            ),
        )
        atomic_json(
            p.AUTO / "completed" / f"{r.schedule}.json",
            dict(passed=True, checkpoint=str(source), experiment=exp, step=p.END),
        )


def main():
    """Dispatch only explicit CPU checks, node execution or training."""
    mode = sys.argv[1]
    if mode == "validate":
        from olmoe3_adaptive_decay_checks import check

        atomic_json(p.AUTO / "cpu_checks.json", check())
        for schedule in p.SCHEDULES:
            subprocess.run([sys.executable, p.SCRIPT, "config", schedule], check=True)
        atomic_json(
            p.AUTO / "config_validation.json",
            dict(passed=True, commit=os.environ["GIT_REF"]),
        )
    elif mode == "config":
        from olmoe3_adaptive_mt_train import validate

        validate(sys.argv[2])
    elif mode == "data":
        from olmoe3_adaptive_mt_data import components, verify_inputs
        from olmo_core.data import TokenizerConfig
        from types import SimpleNamespace
        data = components(SimpleNamespace(tokenizer=TokenizerConfig.dolma2(), work_dir=str(p.ROOT / "data-work/prepare")))
        data.dataset.source_mixture_config.build(npdtype=data.dataset.get_dtype(), sequence_length=8192)
        atomic_json(p.AUTO / "data" / f"inputs-{sys.argv[2]}.json", dict(passed=True, manifest_sha256=verify_inputs()))
    elif mode == "node":
        node(sys.argv[2])
    elif mode == "train":
        from olmoe3_adaptive_mt_train import train

        train()
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
