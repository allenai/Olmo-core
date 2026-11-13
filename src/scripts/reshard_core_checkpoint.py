import json
import logging
import os
import random
import shutil
import socket
from datetime import timedelta
from tempfile import TemporaryDirectory
from typing import Dict, Tuple

import click
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.multiprocessing as mp
from cached_path import cached_path
from torch.distributed import init_device_mesh

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import (
    OLMO_LOCAL_RANK_ENV_VAR,
    OLMO_LOCAL_WORLD_SIZE_ENV_VAR,
    OLMO_NUM_NODES_ENV_VAR,
    get_rank,
    get_world_size,
)
from olmo_core.io import file_exists, join_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def _port_in_use(host: str, port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def _get_next_port() -> int:
    """Get a random port in the range 29500-30000."""
    return random.randint(29500, 30000)


def _find_open_port(host: str = "127.0.0.1") -> int:
    """Find an available port for process group coordination."""
    port = _get_next_port()
    attempts = 0
    while _port_in_use(host, port):
        port = _get_next_port()
        attempts += 1
        if attempts >= 10:
            raise RuntimeError("Failed to find an open port after 10 attempts")
    return port


def load_config(checkpoint_input_dir: PathOrStr) -> Dict:
    if not file_exists(f"{checkpoint_input_dir}/config.json"):
        raise RuntimeError(f"Config file not found at {checkpoint_input_dir}")

    with cached_path(f"{checkpoint_input_dir}/config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return config_dict


def config_dicts_from_path(model_path: str) -> Tuple[Dict, Dict, Dict]:
    # Load and preprocess configs
    experiment_config = load_config(model_path)

    transformer_config_dict = experiment_config["model"]
    # Remove deprecated transformer config options
    if "compile" in transformer_config_dict:
        del transformer_config_dict["compile"]
    if "dp_config" in transformer_config_dict:
        del transformer_config_dict["dp_config"]
    if "tp_config" in transformer_config_dict:
        del transformer_config_dict["tp_config"]
    if "float8_config" in transformer_config_dict:
        del transformer_config_dict["float8_config"]

    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")
    optim_config_dict = experiment_config["train_module"]["optim"]

    return transformer_config_dict, tokenizer_config_dict, optim_config_dict


def model_and_optim_config_from_path(model_path: str) -> Tuple[TransformerConfig, OptimConfig]:
    transformer_config_dict, _, optim_config_dict = config_dicts_from_path(model_path)
    model_config = TransformerConfig.from_dict(transformer_config_dict)
    print(optim_config_dict)
    optim_config: OptimConfig = OptimConfig.from_dict(optim_config_dict)
    return model_config, optim_config


_state_dict_options = dist_cp_sd.StateDictOptions(
    flatten_optimizer_state_dict=True, cpu_offload=True
)


def _worker_process(
    process_rank: int,
    world_size: int,
    primary_addr: str,
    primary_port: int,
    input_path: str,
    output_path: str,
    skip_optimizer_state: bool = False,
) -> None:
    """
    Worker process that initializes the Gloo process group and prints verification messages.

    Args:
        process_rank: Rank of this process in the process group
        world_size: Total number of processes in the group
        primary_addr: Address of the primary process for coordination
        primary_port: Port for process group coordination
        input_path: Input checkpoint path
        output_path: Output checkpoint path
        skip_optimizer_state: If True, skip loading and saving optimizer state
    """
    # Set required environment variables for Olmo-core distributed utilities
    os.environ.setdefault(OLMO_NUM_NODES_ENV_VAR, "1")
    os.environ.setdefault(OLMO_LOCAL_WORLD_SIZE_ENV_VAR, str(world_size))
    os.environ.setdefault(OLMO_LOCAL_RANK_ENV_VAR, str(process_rank))

    # Initialize distributed process group with Gloo backend
    dist.init_process_group(
        "gloo",
        timeout=timedelta(minutes=5),
        init_method=f"tcp://{primary_addr}:{primary_port}",
        world_size=world_size,
        rank=process_rank,
    )

    # Print verification messages
    rank = get_rank()
    ws = get_world_size()

    log.info("Rank %d initialized, starting to load model", rank)

    # Load model
    model_config, optim_config = model_and_optim_config_from_path(input_path)
    model = model_config.build(init_device="meta")
    model.apply_fsdp(
        dp_mesh=init_device_mesh("cpu", (ws,)),
    )
    model.to_empty(device=torch.device("cpu"))

    # Conditionally create optimizer
    if skip_optimizer_state:
        optim = None
        log.info("Skipping optimizer state (--skip-optimizer-state flag set)")
    else:
        optim = optim_config.build(model, strict=True)

    with TemporaryDirectory(prefix=f"reshard_core_checkpoints-rank{get_rank()}-") as work_dir:
        model_and_optim_dir = join_path(input_path, "model_and_optim")

        # find out whether we have a flat or non-flat optimizer
        flatten_optimizer_state = False
        if optim is not None:
            checkpoint_meta = get_checkpoint_metadata(model_and_optim_dir)
            flatten_optimizer_state = (
                "optim.param_groups.0.params" not in checkpoint_meta.state_dict_metadata.keys()
            )

        log.info(f"Loading checkpoint from '{model_and_optim_dir}'")
        load_model_and_optim_state(
            model_and_optim_dir,
            model,
            optim,
            flatten_optimizer_state=flatten_optimizer_state,
            work_dir=work_dir,
            thread_count=1,
        )
        del model_and_optim_dir

    # Save model
    log.info(f"Saving resharded model to '{output_path}'")
    if rank == 0:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        def ignore_subdir(path: str, files):
            if os.path.basename(path) == "model_and_optim":
                return files
            else:
                return []

        shutil.copytree(input_path, output_path, ignore=ignore_subdir)
    dist.barrier()

    model_and_optim_dir = join_path(output_path, "model_and_optim")
    save_model_and_optim_state(
        model_and_optim_dir,
        model,
        optim,
        save_overwrite=True,
        flatten_optimizer_state=flatten_optimizer_state,
    )
    log.info(f"Saved resharded model to '{output_path}'")

    # Barrier to ensure all processes finish together
    dist.barrier()

    # Clean up
    dist.destroy_process_group()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    help="Input checkpoint path",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    help="Output checkpoint path",
)
@click.option(
    "--num-processes",
    "-n",
    type=int,
    default=1,
    help="Number of processes to use. Use 1 to run in main process without multiprocessing (default: 1)",
)
@click.option(
    "--skip-optimizer-state",
    "-s",
    is_flag=True,
    help="Skip loading and saving optimizer state (only reshard model weights)",
)
def main(input_path: str, output_path: str, num_processes: int, skip_optimizer_state: bool) -> None:
    """
    Reshard an Olmo-core checkpoint across different process group configurations.

    This script can run in single-process mode (default) or establish a Gloo process group
    with multiple processes to perform checkpoint resharding operations.

    When using a single process (n=1), the script runs directly in the main process without
    spawning additional processes. For multiple processes, it uses torch.multiprocessing
    and coordinates them using a TCP-based rendezvous on localhost.

    Examples:

    \b
    # Reshard with single process (default, no multiprocessing)
    python reshard_core_checkpoint.py -i ./input -o ./output

    \b
    # Reshard checkpoint with 4 processes
    python reshard_core_checkpoint.py \\
        --input gs://bucket/checkpoint-in \\
        --output ./checkpoint-out \\
        --num-processes 4

    \b
    # Reshard without optimizer state (model weights only)
    python reshard_core_checkpoint.py \\
        --input ./checkpoint-in \\
        --output ./checkpoint-out \\
        --skip-optimizer-state
    """
    if num_processes < 1:
        raise ValueError(f"num_processes must be at least 1, got {num_processes}")

    primary_addr = "127.0.0.1"
    primary_port = _find_open_port(host=primary_addr)

    if num_processes == 1:
        # Run directly in the main process for single process mode
        log.info("Running in single process mode (no multiprocessing)")

        # Run the worker process directly in the main process
        _worker_process(
            process_rank=0,
            world_size=1,
            primary_addr=primary_addr,
            primary_port=primary_port,
            input_path=input_path,
            output_path=output_path,
            skip_optimizer_state=skip_optimizer_state,
        )
    else:
        log.info(f"Launching {num_processes} worker processes on port {primary_port}...")

        # Launch worker processes using torch.multiprocessing
        # Use 'fork' start method for Gloo backend (CPU operations)
        mp.start_processes(
            _worker_process,
            args=(
                num_processes,
                primary_addr,
                primary_port,
                input_path,
                output_path,
                skip_optimizer_state,
            ),
            nprocs=num_processes,
            start_method="fork",
        )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
