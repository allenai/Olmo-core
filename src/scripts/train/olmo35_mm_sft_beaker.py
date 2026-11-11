"""Launch an Olmo 3.5 multimodal SFT job (2 x 8 GPUs) on Beaker from a source tree on Weka.

The job runs the recipe from this checkout (no git clone, no secrets): every replica installs
the tree in the image and starts ``torchrun`` over both nodes. Spec as open-instruct's GPU text
SFT launcher builds it (``scripts/c12/h161_gpu_launch.sh``).

    python src/scripts/train/olmo35_mm_sft_beaker.py RUN_NAME [--dry] [--pre CMD] [--post CMD] \\
        -- --recipe.base=64k --recipe.stop_at_step=30 ...
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGE = "akshitab/olmo-core-tch2130cu130-fa4-rma-2026-09-11"
WORKSPACE = "ai2/olmo-instruct"
BUCKETS = (
    "oe-training-default",
    "olmo-3p5-checkpoints",
    "olmo-3p5-mid-checkpoints",
    "olmo-3p5-post-checkpoints",
    "oe-adapt-default",
    "dolma-3p5",
    "oe-mm-default",
)
HF = "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--pre", default="", help="Shell run on every replica before training.")
    parser.add_argument("--post", default="", help="Shell run on replica 0 after training.")
    parser.add_argument("--timeout", default="3h")
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Let Beaker restart the job after a node failure; it resumes from the run's latest checkpoint.",
    )
    parser.add_argument("--min-runtime", default=None, help="Beaker minRuntime, e.g. 1h.")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    overrides = argv[split + 1 :]

    tree = Path(__file__).resolve().parents[3]
    weka_tree = str(tree).replace("/root/", "/weka/oe-training-default/jasonr/home/", 1)
    script = f"{weka_tree}/src/scripts/train/Olmo35-Multimodal-SFT.py"
    torchrun = (
        f"torchrun --nnodes={args.nodes} --node_rank=$BEAKER_REPLICA_RANK "
        "--master_addr=$BEAKER_LEADER_REPLICA_HOSTNAME --master_port=29400 --nproc_per_node=8 "
        f"{script} train {args.run_name} ai2/holmes "
        + " ".join(shlex.quote(o) for o in overrides + [f"--recipe.num_nodes={args.nodes}"])
    )
    cmd = f"""set -uo pipefail
echo MMSFT-JOB-START $(date -u +%FT%TZ) job $BEAKER_JOB_ID replica $BEAKER_REPLICA_RANK
cd {weka_tree} && git log -1 --format='code %H' ; git status --short | head -20
pip install -q -e "{weka_tree}[fla,kernel-fun]" 'datasets>=4,<6' pypdfium2 h5py 2>&1 | tail -2
export PYTHONPATH={weka_tree}/src
python -c "import olmo_core,torch;print(olmo_core.__file__, torch.__version__, torch.cuda.get_device_name())"
{args.pre}
{torchrun}; RC=$?
echo MMSFT-TRAIN-EXIT $RC $(date -u +%FT%TZ)
if [ "$BEAKER_REPLICA_RANK" = 0 ]; then
{args.post or ':'}
fi
exit $RC"""
    env = [
        {"name": "OLMO_SHARED_FS", "value": "1"},
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        {"name": "OLMO_SYMM_VDEV2D_AUTO_BUILD", "value": "1"},
        {"name": "HF_HOME", "value": f"{HF}/huggingface"},
        {"name": "HF_DATASETS_CACHE", "value": "/weka/oe-training-default/jasonr/hf-home/datasets"},
        {"name": "HF_HUB_CACHE", "value": f"{HF}/hub"},
        {"name": "GRPC_ENABLE_FORK_SUPPORT", "value": "false"},
        {"name": "NCCL_DEBUG", "value": "ERROR"},
        {"name": "NCCL_SOCKET_IFNAME", "value": "ib"},
        {"name": "NCCL_IB_HCA", "value": "^=mlx5_bond_0"},
        {"name": "S3_PROFILE", "value": "default"},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
    ]
    task = {
        "name": args.run_name,
        "image": {"beaker": IMAGE},
        "command": ["/bin/bash", "-c"],
        "arguments": [cmd],
        "envVars": env,
        "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in BUCKETS],
        "result": {"path": "/output"},
        "resources": {"gpuCount": 8, "sharedMemory": "128GiB"},
        "context": {
            "priority": "urgent",
            "preemptible": False,
            **({"autoResume": True} if args.auto_resume else {}),
            **({"minRuntime": args.min_runtime} if args.min_runtime else {}),
        },
        "constraints": {"cluster": ["ai2/holmes"]},
        "timeout": args.timeout,
    }
    if args.nodes > 1:
        task.update(
            replicas=args.nodes,
            leaderSelection=True,
            hostNetworking=True,
            propagateFailure=True,
            propagatePreemption=True,
            synchronizedStartTimeout="20m",
        )
    spec = {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": args.description or f"Olmo 3.5 multimodal SFT: {args.run_name}",
        "tasks": [task],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f, indent=1)
    if args.dry:
        print(open(f.name).read())
        return
    out = subprocess.run(
        ["beaker", "experiment", "create", f.name, "--workspace", WORKSPACE, "--name", args.run_name],
        capture_output=True,
        text=True,
    )
    print(out.stdout, out.stderr, file=sys.stderr)
    os.unlink(f.name)


if __name__ == "__main__":
    main()
