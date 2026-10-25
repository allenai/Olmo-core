#!/usr/bin/env python3
"""
Export a multimodal OLMoDDP checkpoint as one consolidated safetensors file for evaluation.

Checkpoints written by
:class:`~olmo_core.train.train_module.transformer.multimodal_train_module.MultimodalOLMoDDPTrainModule`
hold no ``model.*`` keys: trainable parameters are stored only as flattened FP32 optimizer main
parameters (``module.<name>.main``), frozen parameters as ``frozen_model.<name>`` and persistent
buffers as ``model_buffer.<name>``. This script writes the model's state dict as
``model.safetensors`` next to ``olmo_core_config.json`` (a copy of the checkpoint's
``config.json``), the consolidated layout olmo-eval's ``olmo_core_vlm`` provider loads.

The default dtype is bfloat16, the dtype the OLMoDDP train module runs the model in, so the
export holds exactly the weights the training forward used. ``--verify`` also loads the
checkpoint through the train module's own eval-only loader into a CPU model and checks that
every exported tensor matches it bit for bit.

Usage::

    python src/scripts/export_multimodal_olmoddp.py CHECKPOINT_STEP_DIR OUTPUT_DIR [--verify]
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import click
import torch

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
from olmo_core.nn.vision.multimodal import MultimodalLM, MultimodalLMConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def checkpoint_key(name: str, checkpoint_keys: set) -> str:
    """
    Find the single checkpoint entry holding model state ``name``.

    :param name: A key of the model's state dict.
    :param checkpoint_keys: All keys of the distributed checkpoint.

    :raises KeyError: If no entry, or more than one entry, matches.
    """
    candidates = [
        f"frozen_model.{name}",
        f"module.{name}.main",
        f"{name}.main",
        f"model.{name}",
        f"model_buffer.{name}",
    ]
    found = [key for key in candidates if key in checkpoint_keys]
    if len(found) != 1:
        raise KeyError(f"Expected exactly one checkpoint entry for '{name}', found {found}")
    return found[0]


def _with_torch_attention(config: Any) -> Any:
    """Point every attention ``backend`` at the dense ``torch`` one, which runs everywhere.

    The backend holds no weights, and training backends such as ``flash_4`` only build on
    the platforms their kernels support.
    """
    if isinstance(config, dict):
        return {
            key: "torch"
            if key == "backend" and isinstance(value, str)
            else _with_torch_attention(value)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [_with_torch_attention(value) for value in config]
    return config


def build_model(config: dict, device: str) -> MultimodalLM:
    """Build the checkpoint's multimodal model on ``device``."""
    model_config = MultimodalLMConfig.from_dict(_with_torch_attention(config["model"]))
    return model_config.build(init_device=device)


def export_state_dict(step_dir: Path, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """
    Read the model state dict out of a multimodal OLMoDDP checkpoint.

    :param step_dir: The checkpoint step directory (holding ``config.json`` and
        ``model_and_optim/``).
    :param dtype: The dtype of the exported tensors.
    """
    config = json.loads((step_dir / "config.json").read_text())
    dcp_dir = str(step_dir / "model_and_optim")
    checkpoint_keys = set(get_checkpoint_metadata(dcp_dir).state_dict_metadata)

    shapes = {
        name: tensor.shape for name, tensor in build_model(config, "meta").state_dict().items()
    }
    names: List[str] = list(shapes)
    keys = [checkpoint_key(name, checkpoint_keys) for name in names]
    log.info(
        "Reading %d tensors (%d optimizer main params, %d frozen params) from %s",
        len(keys),
        sum(key.endswith(".main") for key in keys),
        sum(key.startswith("frozen_model.") for key in keys),
        dcp_dir,
    )

    state_dict: Dict[str, torch.Tensor] = {}
    for name, key, tensor in zip(names, keys, load_keys(dcp_dir, keys)):
        if key.endswith(".main"):
            tensor = tensor.view(shapes[name])
        elif tensor.shape != shapes[name]:
            raise ValueError(f"'{key}' has shape {tuple(tensor.shape)}, expected {shapes[name]}")
        state_dict[name] = tensor.to(dtype).contiguous()
    return state_dict


def verify_against_train_module(
    step_dir: Path, state_dict: Dict[str, torch.Tensor], dtype: torch.dtype
) -> None:
    """
    Check the export against the train module's own eval-only checkpoint loader.

    :raises AssertionError: If any tensor differs.
    """
    from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader
    from olmo_core.train.train_module.transformer.multimodal_train_module import (
        MultimodalOLMoDDPTrainModule,
    )

    config = json.loads((step_dir / "config.json").read_text())
    model = build_model(config, "cpu").to(dtype)
    # Only the key resolution and the load itself are needed, not a parallelized train module.
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.model_parts = [model]
    train_module.expand_shared_qk_norm_on_load = False
    dcp_dir = str(step_dir / "model_and_optim")
    reader = RemoteFileSystemReader(dcp_dir)
    train_module._load_model_state_dict_direct(reader.read_metadata(), dcp_dir, reader, None)

    reference = model.state_dict()
    if set(reference) != set(state_dict):
        raise AssertionError(f"Key mismatch: {set(reference) ^ set(state_dict)}")
    mismatched = [name for name in reference if not torch.equal(reference[name], state_dict[name])]
    if mismatched:
        raise AssertionError(f"{len(mismatched)} tensors differ, e.g. {mismatched[:5]}")
    log.info("Verified all %d tensors against the train module's loader", len(reference))


@click.command()
@click.argument("step_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--dtype", type=click.Choice(sorted(_DTYPES)), default="bfloat16", show_default=True)
@click.option("--verify", is_flag=True, help="Check the export against the train module's loader.")
@click.option("--overwrite", is_flag=True, help="Replace an existing export.")
def main(step_dir: Path, output_dir: Path, dtype: str, verify: bool, overwrite: bool):
    """Export STEP_DIR to OUTPUT_DIR/{model.safetensors,olmo_core_config.json}."""
    from safetensors.torch import save_file

    weights_path = output_dir / "model.safetensors"
    if weights_path.exists() and not overwrite:
        raise click.ClickException(f"{weights_path} exists; pass --overwrite to replace it")

    state_dict = export_state_dict(step_dir, _DTYPES[dtype])
    if verify:
        verify_against_train_module(step_dir, state_dict, _DTYPES[dtype])

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_dir / "model.safetensors.tmp"
    save_file(state_dict, str(tmp_path), metadata={"format": "pt"})
    shutil.copyfile(step_dir / "config.json", output_dir / "olmo_core_config.json")
    # Renamed last, so a complete export is exactly a directory holding model.safetensors.
    tmp_path.rename(weights_path)
    num_params = sum(tensor.numel() for tensor in state_dict.values())
    log.info("Wrote %s (%.2fB values, %s)", weights_path, num_params / 1e9, dtype)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
