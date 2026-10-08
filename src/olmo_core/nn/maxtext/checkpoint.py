"""
Reading and writing OLMo Core (torch DCP) and MaxText (Orbax) weights for OLMoE3 conversion.

The OLMo Core side only needs torch. The MaxText side imports JAX, Orbax and MaxText lazily, so it
only needs them installed when it's used: install the MaxText checkout that has the ``olmoe3``
decoder into the same environment (CPU JAX is enough).
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.distributed.checkpoint as dist_cp
from cached_path import cached_path

from olmo_core.distributed.checkpoint.filesystem import (
    RemoteFileSystemReader,
    RemoteFileSystemWriter,
)
from olmo_core.io import (
    PathOrStr,
    file_exists,
    is_url,
    join_path,
    normalize_path,
    upload,
)

from .olmoe3 import OLMoE3Geometry, normalize_olmo_core_key, olmo_core_shapes

log = logging.getLogger(__name__)

__all__ = [
    "load_olmo_core_config",
    "load_olmo_core_weights",
    "save_olmo_core_weights",
    "load_maxtext_params",
    "save_maxtext_params",
    "write_json",
]


def load_olmo_core_config(checkpoint_dir: PathOrStr) -> Dict[str, Any]:
    """Load the experiment ``config.json`` saved alongside an OLMo Core checkpoint."""
    path = f"{normalize_path(checkpoint_dir)}/config.json"
    if not file_exists(path):
        raise FileNotFoundError(f"no config.json at {checkpoint_dir}")
    with cached_path(path).open("r", encoding="utf-8") as f:
        config = json.load(f)
    if "model" not in config:
        raise ValueError(f"{path} is not an OLMo Core experiment config")
    return config


def load_olmo_core_weights(
    checkpoint_dir: PathOrStr, geometry: OLMoE3Geometry
) -> Dict[str, np.ndarray]:
    """
    Load unsharded fp32 weights from an OLMo Core checkpoint's ``model_and_optim/`` directory.

    Handles both ``model.<name>`` entries (the FSDP train module, and the converted checkpoints
    written by :func:`save_olmo_core_weights`) and the flattened fp32 master copies
    ``module.<name>.main`` that the OLMoDDP train module saves instead. The model isn't built;
    shapes come from ``geometry``, so this runs on CPU without the GPU kernel packages.
    """
    model_and_optim = join_path(checkpoint_dir, "model_and_optim")
    reader = RemoteFileSystemReader(model_and_optim)
    metadata = reader.read_metadata().state_dict_metadata
    shapes = olmo_core_shapes(geometry)

    # Prefer the model weights; fall back to the fp32 master copies.
    by_name: Dict[str, Dict[str, str]] = {"model": {}, "main": {}}
    for key in metadata:
        if key.startswith("model."):
            by_name["model"][normalize_olmo_core_key(key)] = key
        elif key.endswith(".main"):
            by_name["main"][normalize_olmo_core_key(key)] = key
    source = "model" if by_name["model"] else "main"
    keys = by_name[source]
    if not keys:
        raise RuntimeError(f"{model_and_optim} has neither model.* nor .main entries")
    missing = sorted(set(shapes) - set(keys))
    if missing:
        raise KeyError(f"{model_and_optim} is missing parameters: {missing[:10]}")
    extra = sorted(set(keys) - set(shapes))
    if extra:
        raise KeyError(f"{model_and_optim} has parameters the geometry doesn't: {extra[:10]}")

    to_load: Dict[str, torch.Tensor] = {}
    for name, shape in shapes.items():
        key = keys[name]
        md = metadata[key]
        stored_shape = tuple(md.size)  # type: ignore[union-attr]
        numel = int(np.prod(shape))
        if stored_shape not in (shape, (numel,)):
            raise ValueError(f"{key}: stored shape {stored_shape} doesn't fit {shape}")
        dtype = md.properties.dtype  # type: ignore[union-attr]
        to_load[key] = torch.empty(stored_shape, dtype=dtype)
    log.info(f"Loading {len(to_load)} parameters ({source}) from '{model_and_optim}'")
    dist_cp.load(to_load, storage_reader=reader, no_dist=True)

    out = {}
    for name, shape in shapes.items():
        tensor = to_load.pop(keys[name])
        if tensor.dtype != torch.float32:
            log.warning(f"{name} is stored as {tensor.dtype}; upcasting to float32")
            tensor = tensor.float()
        out[name] = tensor.reshape(shape).numpy()
    return out


def save_olmo_core_weights(
    output_dir: PathOrStr,
    state: Mapping[str, np.ndarray],
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
    if file_exists(f"{normalize_path(model_and_optim)}/.metadata") and not save_overwrite:
        raise FileExistsError(f"{model_and_optim} already holds a checkpoint")
    tensors = {f"model.{k}": torch.from_numpy(np.ascontiguousarray(v)) for k, v in state.items()}
    log.info(f"Saving {len(tensors)} parameters to '{model_and_optim}'")
    dist_cp.save(tensors, storage_writer=RemoteFileSystemWriter(model_and_optim), no_dist=True)

    write_json(
        f"{normalize_path(output_dir)}/config.json",
        experiment_config,
        save_overwrite=save_overwrite,
    )


def write_json(path: str, obj: Any, *, save_overwrite: bool = False) -> None:
    """Write ``obj`` as JSON to a local path or a cloud URL."""
    if not is_url(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if Path(path).exists() and not save_overwrite:
            raise FileExistsError(path)
        Path(path).write_text(json.dumps(obj, indent=2))
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        upload(f.name, path, save_overwrite=save_overwrite)


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


def load_maxtext_params(items_dir: str) -> Dict[str, np.ndarray]:
    """
    Load the parameters of a MaxText checkpoint as a flat ``/``-joined dict of fp32 numpy arrays.
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
    return {k: np.asarray(v) for k, v in _flatten(params).items()}


def save_maxtext_params(
    output_dir: str, params: Mapping[str, np.ndarray], *, device_count: Optional[int] = None
) -> None:
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
        _unflatten(params),
        device_count=device_count or jax.device_count(),
        use_ocdbt=True,
        use_zarr3=True,
    )
