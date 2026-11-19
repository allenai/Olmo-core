import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import torch
from beaker import BeakerGpuType
from beaker.exceptions import BeakerError

from olmo_core.io import is_url
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
    get_beaker_client,
)
from olmo_core.train.callbacks.beaker import BEAKER_RESULT_DIR
from olmo_core.utils import generate_uuid

log = logging.getLogger(__name__)


@lru_cache()
def get_beaker_username() -> Optional[str]:
    try:
        with get_beaker_client() as beaker:
            return beaker.user_name
    except BeakerError:
        return None


def get_root_dir(cluster: str) -> str:
    if cluster.startswith("ai2/"):
        with get_beaker_client() as beaker:
            cl = beaker.cluster.get(cluster)
            tags = set(cl.tags)
            if "storage:weka" in tags:
                return "/weka/oe-training-default/ai2-llm"
            else:
                return "gs://ai2-llm"
    else:
        return "gs://ai2-llm"


def get_work_dir(root_dir: str) -> str:
    if is_url(root_dir):
        return "./dataset-cache"
    elif (beaker_username := get_beaker_username()) is not None:
        return f"{root_dir}/checkpoints/{beaker_username.lower()}/dataset-cache"
    else:
        return f"{root_dir}/checkpoints/dataset-cache"


def build_launch_config(
    *,
    name: str,
    cmd: List[str],
    cluster: str,
    root_dir: Optional[str] = None,
    task_name: str = "train",
    workspace: str = "ai2/OLMo-core",
    budget: str = "ai2/oe-other",
    nccl_debug: Union[bool, str] = False,
    flight_recorder: bool = False,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    num_execution_units: Optional[int] = None,
    step_timeout: Optional[int] = None,
    step_soft_timeout: Optional[int] = 10 * 60,
) -> BeakerLaunchConfig:
    weka_buckets: List[BeakerWekaBucket] = []

    default_root_dir = get_root_dir(cluster)
    if root_dir is None:
        root_dir = default_root_dir
    elif root_dir != default_root_dir:
        log.warning(
            f"Overriding default root_dir for {cluster=} to {root_dir} ({default_root_dir=})."
        )

    if root_dir.startswith("/weka/"):
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))
        weka_buckets.append(BeakerWekaBucket("oe-adapt-default", "/weka/oe-adapt-default"))

    beaker_user = get_beaker_username()
    if beaker_user is None:
        raise RuntimeError(
            "Environment not configured correctly for Beaker, you may be missing the BEAKER_TOKEN env var."
        )
    beaker_user = beaker_user.upper()

    env_secrets = [
        BeakerEnvSecret(
            name="BEAKER_TOKEN",
            secret=f"{beaker_user}_BEAKER_TOKEN",
            required=True,
        ),
        BeakerEnvSecret(
            name="WANDB_API_KEY",
            secret=f"{beaker_user}_WANDB_API_KEY",
            required=True,
        ),
        BeakerEnvSecret(
            name="COMET_API_KEY",
            secret=f"{beaker_user}_COMET_API_KEY",
            required=False,
        ),
        BeakerEnvSecret(
            name="R2_ENDPOINT_URL",
            secret="R2_ENDPOINT_URL",
            required=False,
        ),
        BeakerEnvSecret(
            name="WEKA_ENDPOINT_URL",
            secret="WEKA_ENDPOINT_URL",
            required=False,
        ),
        BeakerEnvSecret(
            name="SLACK_WEBHOOK_URL",
            secret="SLACK_WEBHOOK_URL",
            required=False,
        ),
    ]

    env_vars: List[BeakerEnvVar] = []
    if isinstance(nccl_debug, str):
        env_vars.append(BeakerEnvVar(name="NCCL_DEBUG", value=nccl_debug))
    else:
        env_vars.append(BeakerEnvVar(name="NCCL_DEBUG", value="INFO" if nccl_debug else "WARN"))
    if flight_recorder:
        # https://github.com/pytorch/tutorials/blob/main/unstable_source/flight_recorder_tutorial.rst
        fr_dump_location = Path(BEAKER_RESULT_DIR) / "flightrecorder" / "nccl_trace_rank_"
        env_vars += [
            BeakerEnvVar(name="TORCH_NCCL_TRACE_BUFFER_SIZE", value="2000"),
            BeakerEnvVar(name="TORCH_NCCL_DUMP_ON_TIMEOUT", value="1"),
            BeakerEnvVar(name="TORCH_FR_DUMP_TEMP_FILE", value=str(fr_dump_location)),
        ]

    launch_config = BeakerLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget=budget,
        cmd=cmd,
        task_name=task_name,
        workspace=workspace,
        clusters=[cluster],
        weka_buckets=weka_buckets,
        beaker_image=beaker_image,
        num_nodes=num_nodes,
        num_gpus=8,
        num_execution_units=num_execution_units,
        shared_filesystem=not is_url(root_dir),
        allow_dirty=False,
        env_vars=env_vars,
        env_secrets=env_secrets,
        google_credentials_secret="GOOGLE_CREDENTIALS",
        aws_config_secret=f"{beaker_user}_AWS_CONFIG",
        aws_credentials_secret=f"{beaker_user}_AWS_CREDENTIALS",
        step_timeout=step_timeout,
        step_soft_timeout=step_soft_timeout,
    )

    return launch_config


CLUSTER_TO_GPU_TYPE = {
    "ai2/jupiter": "NVIDIA H100 80GB HBM3",
    "ai2/ceres": "NVIDIA H100 80GB HBM3",
    "ai2/titan": "NVIDIA B200",
}


def get_gpu_type(cluster: str) -> str:
    if cluster in CLUSTER_TO_GPU_TYPE:
        return CLUSTER_TO_GPU_TYPE[cluster]
    elif cluster.startswith("ai2/"):
        with get_beaker_client() as beaker:
            cl = beaker.cluster.get(cluster)
            for node in beaker.node.list(cluster=cl, limit=1):
                if (gpu_type := node.node_resources.gpu_type) > 0:
                    return BeakerGpuType(gpu_type).name.replace("_", " ")
            else:
                raise RuntimeError(f"Could not determine GPU type for cluster '{cluster}'")
    elif cluster == "local":
        return torch.get_default_device().type
    else:
        raise ValueError(f"Unknown cluster '{cluster}'")
