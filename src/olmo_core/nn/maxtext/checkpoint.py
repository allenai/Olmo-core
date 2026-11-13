"""
Reading and writing OLMo Core (torch DCP) and MaxText (Orbax) weights, and converting between
MaxText's scanned and unscanned parameter layouts.

The OLMo Core side only needs torch. The MaxText side imports JAX, Orbax and MaxText lazily, so it
only needs them installed when it's used: install a MaxText checkout that has the model's decoder
into the same environment (CPU JAX is enough).
"""

import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, Mapping

import numpy as np
import torch
from cached_path import cached_path

from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_keys,
    save_state_dict,
)
from olmo_core.io import PathOrStr, copy_file, file_exists, join_path, normalize_path

from .config import MaxTextModelConfig, ScanLayout
from .convert import normalize_olmo_core_key, olmo_core_shapes, split_layer_key

log = logging.getLogger(__name__)

__all__ = [
    "read_json",
    "write_json",
    "load_olmo_core_config",
    "load_olmo_core_weights",
    "save_olmo_core_weights",
    "load_maxtext_params",
    "save_maxtext_params",
    "initialize_maxtext_config",
    "scan_params",
    "unscan_params",
    "is_scanned",
]


def read_json(path: PathOrStr) -> Any:
    """Read a JSON file from a local path or a cloud URL."""
    with cached_path(normalize_path(path)).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathOrStr, obj: Any, *, save_overwrite: bool = False) -> None:
    """Write ``obj`` as JSON to a local path or a cloud URL."""
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        copy_file(f.name, path, save_overwrite=save_overwrite)


def load_olmo_core_config(checkpoint_dir: PathOrStr) -> Dict[str, Any]:
    """Load the experiment ``config.json`` saved alongside an OLMo Core checkpoint."""
    path = join_path(checkpoint_dir, "config.json")
    if not file_exists(path):
        raise FileNotFoundError(f"no config.json at {checkpoint_dir}")
    config = read_json(path)
    if "model" not in config:
        raise ValueError(f"{path} is not an OLMo Core experiment config")
    return config


def load_olmo_core_weights(
    checkpoint_dir: PathOrStr, config: MaxTextModelConfig
) -> Dict[str, torch.Tensor]:
    """
    Load unsharded fp32 weights from an OLMo Core checkpoint's ``model_and_optim/`` directory.

    Handles both ``model.<name>`` entries (the FSDP train module, and the converted checkpoints
    written by :func:`save_olmo_core_weights`) and the flattened fp32 master copies
    ``module.<name>.main`` that the OLMoDDP train module saves instead. The model isn't built;
    shapes come from ``config``, so this runs on CPU without the GPU kernel packages.
    """
    model_and_optim = join_path(checkpoint_dir, "model_and_optim")
    metadata = get_checkpoint_metadata(model_and_optim).state_dict_metadata
    shapes = olmo_core_shapes(config)

    # Prefer the model weights; fall back to the fp32 master copies.
    model_keys = {normalize_olmo_core_key(k): k for k in metadata if k.startswith("model.")}
    main_keys = {normalize_olmo_core_key(k): k for k in metadata if k.endswith(".main")}
    keys = model_keys or main_keys
    if not keys:
        raise RuntimeError(f"{model_and_optim} has neither model.* nor .main entries")
    missing = sorted(set(shapes) - set(keys))
    if missing:
        raise KeyError(f"{model_and_optim} is missing parameters: {missing[:10]}")
    extra = sorted(set(keys) - set(shapes))
    if extra:
        raise KeyError(f"{model_and_optim} has parameters the config doesn't: {extra[:10]}")
    for name, shape in shapes.items():
        stored_shape = tuple(metadata[keys[name]].size)  # type: ignore[union-attr]
        if stored_shape not in (shape, (int(np.prod(shape)),)):
            raise ValueError(f"{keys[name]}: stored shape {stored_shape} doesn't fit {shape}")

    log.info(f"Loading {len(shapes)} parameters from '{model_and_optim}'")
    tensors = load_keys(model_and_optim, [keys[name] for name in shapes])
    out = {}
    for (name, shape), tensor in zip(shapes.items(), tensors):
        if tensor.dtype != torch.float32:
            log.warning(f"{name} is stored as {tensor.dtype}; upcasting to float32")
            tensor = tensor.float()
        out[name] = tensor.reshape(shape)
    return out


def save_olmo_core_weights(
    output_dir: PathOrStr,
    state: Mapping[str, torch.Tensor],
    experiment_config: Mapping[str, Any],
    *,
    save_overwrite: bool = False,
) -> None:
    """
    Write a weights-only OLMo Core checkpoint: ``model_and_optim/`` with ``model.<name>`` entries
    at full shape, plus ``config.json``. This is the layout the Hugging Face importer writes.
    Without optimizer moments the OLMoDDP train module rebuilds its fp32 master copies from
    these weights and starts the optimizer fresh.
    """
    model_and_optim = join_path(output_dir, "model_and_optim")
    log.info(f"Saving {len(state)} parameters to '{model_and_optim}'")
    save_state_dict(model_and_optim, {"model": dict(state)}, save_overwrite=save_overwrite)
    write_json(
        join_path(output_dir, "config.json"), experiment_config, save_overwrite=save_overwrite
    )


def _flatten(tree: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in tree.items():
        path = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


def _unflatten(flat: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path, v in flat.items():
        node = out
        *parents, leaf = path.split("/")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v
    return out


def load_maxtext_params(items_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load the parameters of a MaxText checkpoint as a flat ``/``-joined dict of fp32 tensors.
    ``items_dir`` is a checkpoint step's ``items`` directory, e.g. ``.../checkpoints/50/items``.
    Optimizer state, if present, is skipped.
    """
    import jax  # type: ignore
    import orbax.checkpoint as ocp  # type: ignore
    from etils import epath  # type: ignore

    path = epath.Path(items_dir)
    ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True))
    tree = ckptr.metadata(path).item_metadata.tree
    if "params" not in tree:
        raise KeyError(f"{items_dir} has no params; found {sorted(tree)}")
    tree = {"params": tree["params"]}
    # Restore to host memory: on an accelerator this would be a second copy of the model.
    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    restore_args = jax.tree_util.tree_map(
        lambda x: ocp.ArrayRestoreArgs(sharding=sharding, dtype=np.float32),
        tree,
        is_leaf=lambda x: hasattr(x, "shape"),
    )
    restored = ckptr.restore(
        path,
        args=ocp.args.PyTreeRestore(item=tree, restore_args=restore_args, partial_restore=True),
    )
    # On disk: params/params/<tree> (the item key, then Flax's params collection).
    params = restored["params"]["params"]
    return {k: torch.from_numpy(np.array(v)) for k, v in _flatten(params).items()}


def save_maxtext_params(output_dir: str, params: Mapping[str, torch.Tensor]) -> None:
    """
    Write a flat ``/``-joined parameter dict as a MaxText checkpoint at ``<output_dir>/0/items``,
    loadable with ``load_parameters_path=<output_dir>/0/items``. Uses MaxText's own checkpoint
    writer so the on-disk layout is exactly what MaxText expects.
    """
    import jax  # type: ignore
    from maxtext.checkpoint_conversion.utils.utils import (  # type: ignore
        save_weights_to_checkpoint,
    )

    save_weights_to_checkpoint(
        output_dir,
        _unflatten({k: v.numpy() for k, v in params.items()}),
        device_count=jax.device_count(),
        use_ocdbt=True,
        use_zarr3=True,
    )


def initialize_maxtext_config(
    config: MaxTextModelConfig, *overrides: str, scan_layers: bool = True
) -> Any:
    """
    MaxText's ``pyconfig`` for :attr:`config.model_name <MaxTextModelConfig.model_name>` sized to
    this model (one device, no distributed system), plus any ``key=value`` ``overrides``. This is
    the config a converted checkpoint loads into.
    """
    from maxtext.configs import pyconfig  # type: ignore
    from maxtext.utils.globals import MAXTEXT_PKG_DIR  # type: ignore

    # MaxText code paths may hand sys.argv to absl, which rejects the caller's flags.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        return pyconfig.initialize(
            [
                "",
                os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
                f"model_name={config.model_name}",
                "override_model_config=True",
                *config.overrides(),
                f"scan_layers={scan_layers}",
                "skip_jax_distributed_system=True",
                "per_device_batch_size=1",
                *overrides,
            ]
        )
    finally:
        sys.argv = argv


#: Unscanned layer parameters live under ``decoder/layers_{i}/``.
_UNSCANNED = "decoder/layers_"

#: Scanned parameters stack their cycles along this axis.
_SCAN_AXIS = 1


def _slot_prefixes(layout: ScanLayout) -> Dict[str, bool]:
    """Key prefix of each scanned slot -> whether it's the unrolled first cycle."""
    prefixes = {f"{layout.stacked}/{layout.slot}": False}
    if layout.unrolled is not None:
        prefixes[f"{layout.unrolled}/{layout.slot}"] = True
    return prefixes


def is_scanned(params: Mapping[str, Any], layout: ScanLayout) -> bool:
    """Whether ``params`` is in the ``scan_layers=True`` layout."""
    return any(k.startswith(p) for k in params for p in _slot_prefixes(layout))


def scan_params(
    params: Mapping[str, torch.Tensor], layout: ScanLayout, n_layers: int
) -> Dict[str, torch.Tensor]:
    """
    Convert an unscanned MaxText parameter tree (``decoder/layers_{i}/...``) to the layout MaxText
    uses with ``scan_layers=True``; see :class:`~olmo_core.nn.maxtext.config.ScanLayout`.
    """
    if n_layers % layout.cycle:
        raise ValueError(f"n_layers={n_layers} isn't a multiple of the scan cycle ({layout.cycle})")
    n_unrolled = 1 if layout.unrolled is not None else 0
    n_stacked = n_layers // layout.cycle - n_unrolled
    out: Dict[str, torch.Tensor] = {}
    per_slot: Dict[str, list] = {}
    for key, value in params.items():
        split = split_layer_key(key, _UNSCANNED, "/")
        if split is None:
            out[key] = value
            continue
        layer, rest = split
        cycle, slot = divmod(layer, layout.cycle)
        if cycle < n_unrolled:
            out[f"{layout.unrolled}/{layout.slot}{slot}/{rest}"] = value
        else:
            stacked_key = f"{layout.stacked}/{layout.slot}{slot}/{rest}"
            per_slot.setdefault(stacked_key, [None] * n_stacked)[cycle - n_unrolled] = value
    for key, values in per_slot.items():
        if any(v is None for v in values):
            raise KeyError(f"{key} is missing from some cycles")
        out[key] = torch.stack(values, dim=_SCAN_AXIS)
    return out


def unscan_params(
    params: Mapping[str, torch.Tensor], layout: ScanLayout
) -> Dict[str, torch.Tensor]:
    """The inverse of :func:`scan_params`."""
    n_unrolled = 1 if layout.unrolled is not None else 0
    out: Dict[str, torch.Tensor] = {}
    for key, value in params.items():
        for prefix, unrolled in _slot_prefixes(layout).items():
            split = split_layer_key(key, prefix, "/")
            if split is not None:
                break
        else:
            if split_layer_key(key, _UNSCANNED, "/") is not None:
                raise ValueError(f"{key} is already unscanned")
            out[key] = value
            continue
        slot, rest = split
        if unrolled:
            out[f"{_UNSCANNED}{slot}/{rest}"] = value
        else:
            for c in range(value.shape[_SCAN_AXIS]):
                layer = (c + n_unrolled) * layout.cycle + slot
                out[f"{_UNSCANNED}{layer}/{rest}"] = value.select(_SCAN_AXIS, c)
    return out
