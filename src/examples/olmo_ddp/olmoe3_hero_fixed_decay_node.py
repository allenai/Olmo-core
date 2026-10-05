"""One distributed agent per node for olmoe3_hero_fixed_decay_example.py train.

Usage (Beaker task arguments, every replica):
    python src/examples/olmo_ddp/olmoe3_hero_fixed_decay_node.py PLAN CLUSTER [TRAIN_ARGS...]

Topology evidence and rendezvous markers live beside PLAN, never in the child save
folder: the trainer refuses to adopt a non-empty save folder without its plan copy.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from olmoe3_profile_node import resolve_ready_leader
from olmoe3_small_hero_runtime import verify_runtime


def main():
    """Resolve the current replica generation's leader, then run the decay trainer."""
    from beaker import Beaker

    verify_runtime()
    plan_path, cluster, *train_args = sys.argv[1:]
    plan = json.loads(Path(plan_path).read_text())
    nodes = plan["gpus"] // 8
    experiment, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    assert (
        int(os.environ["BEAKER_REPLICA_COUNT"]),
        int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]),
    ) == (nodes, 8), "Allocate plan['gpus'] // 8 replicas with 8 GPUs each"
    automation = Path(plan_path).resolve().parent
    subprocess.run(
        [
            sys.executable,
            "src/examples/olmo_ddp/olmoe3_profile_topology.py",
            str(automation / "topology" / experiment / job),
        ],
        check=True,
    )
    ready = automation / "rendezvous" / experiment
    ready.mkdir(parents=True, exist_ok=True)
    temporary = ready / f"{job}.tmp"
    temporary.write_text(json.dumps({"job": job, "rank": rank}))
    temporary.replace(ready / f"{job}.json")
    with Beaker.from_env(check_for_upgrades=False) as beaker:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(beaker, beaker.workload.get(experiment), ready, nodes)
            if leader:
                break
            print(f"Node {rank}: waiting for current replica assignments", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("Decay rendezvous not ready within 15 minutes")
    leader_job, host = leader
    port = 29000 + int(hashlib.sha256(experiment.encode()).hexdigest()[:8], 16) % 1000
    print(
        f"DECAY_AGENT node={rank} run={plan['run_name']} leader={leader_job} "
        f"endpoint={host}:{port} train_args={train_args}",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={nodes}",
            "--nproc-per-node=8",
            f"--node-rank={rank}",
            "--rdzv-backend=static",
            f"--rdzv-endpoint={host}:{port}",
            f"--rdzv-id={experiment}",
            "--rdzv-conf=read_timeout=900",
            "--max-restarts=0",
            "src/examples/olmo_ddp/olmoe3_hero_fixed_decay_example.py",
            "train",
            "--plan",
            plan_path,
            "--cluster",
            cluster,
            *train_args,
        ],
        check=True,
    )
    print(f"DECAY_NODE_COMPLETE node={rank}", flush=True)


if __name__ == "__main__":
    main()
