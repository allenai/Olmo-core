"""
Reading and writing OLMo Core (torch DCP) and MaxText (Orbax) weights and AdamW optimizer state,
and converting between MaxText's scanned and unscanned parameter layouts.

The OLMo Core side only needs torch. The MaxText side imports JAX, Orbax and MaxText lazily, so it
only needs them installed when it's used: install a MaxText checkout that has the model's decoder
into the same environment (CPU JAX is enough).
"""

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

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
    "AdamState",
    "read_json",
    "write_json",
    "load_olmo_core_config",
    "load_olmo_core_weights",
    "load_olmo_core_optim_state",
    "save_olmo_core_weights",
    "load_maxtext_params",
    "load_maxtext_checkpoint",
    "save_maxtext_params",
    "save_maxtext_checkpoint",
    "maxtext_checkpoint_items",
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


@dataclass
class AdamState:
    """
    Unsharded fp32 AdamW optimizer state: each parameter's first and second moments, keyed like
    the weights they belong to, and the number of optimizer steps taken.

    OLMo Core's AdamW and optax's ``adamw`` (MaxText's default ``opt_type``) keep the same moments
    (``m = b1 * m + (1 - b1) * g`` and ``v = b2 * v + (1 - b2) * g**2``), count steps the same way,
    and bias-correct from that count the same way, so this state carries over between them as is.
    Every mapping between the two weight layouts is a pure re-layout, so the moments convert with
    the same mappings as the weights.
    """

    step: int
    exp_avg: Dict[str, torch.Tensor]
    exp_avg_sq: Dict[str, torch.Tensor]

    def moments(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {"exp_avg": self.exp_avg, "exp_avg_sq": self.exp_avg_sq}


#: The prefix the flattened optimizer state of the FSDP train module puts before parameter names.
_FSDP_OPTIM_STATE_PREFIX = "optim.state."

#: Per-parameter optimizer state that isn't AdamW's.
_NON_ADAM_SUFFIXES = (".muon_momentum",)


def _olmo_core_keys(metadata_keys: Iterable[str], suffix: str) -> Dict[str, str]:
    """
    Parameter name -> checkpoint key, for the keys that end in ``.<suffix>``. Covers the OLMoDDP
    train module's flat ``module.<name>.<suffix>`` entries and the FSDP train module's flattened
    ``optim.state.<name>.<suffix>`` entries.
    """
    out = {}
    for key in metadata_keys:
        if not key.endswith(f".{suffix}"):
            continue
        name = key[: -len(suffix) - 1].removeprefix(_FSDP_OPTIM_STATE_PREFIX)
        out[normalize_olmo_core_key(name)] = key
    return out


def _load_olmo_core_tensors(
    model_and_optim: str,
    metadata: Mapping[str, Any],
    keys: Mapping[str, str],
    shapes: Mapping[str, Tuple[int, ...]],
    what: str,
) -> Dict[str, torch.Tensor]:
    """
    Load ``keys`` (parameter name -> checkpoint key) as fp32 tensors at their full ``shapes``. Each
    key may be stored at full shape or flattened, as OLMoDDP stores its state.
    """
    missing = sorted(set(shapes) - set(keys))
    if missing:
        raise KeyError(f"{model_and_optim} is missing {what} for: {missing[:10]}")
    extra = sorted(set(keys) - set(shapes))
    if extra:
        raise KeyError(
            f"{model_and_optim} has {what} for parameters the config doesn't: {extra[:10]}"
        )
    for name, shape in shapes.items():
        stored_shape = tuple(metadata[keys[name]].size)
        if stored_shape not in (shape, (int(np.prod(shape)),)):
            raise ValueError(f"{keys[name]}: stored shape {stored_shape} doesn't fit {shape}")

    log.info(f"Loading {what} of {len(shapes)} parameters from '{model_and_optim}'")
    tensors = load_keys(model_and_optim, [keys[name] for name in shapes])
    out = {}
    upcast = set()
    for (name, shape), tensor in zip(shapes.items(), tensors):
        if tensor.dtype != torch.float32:
            upcast.add(str(tensor.dtype))
            tensor = tensor.float()
        out[name] = tensor.reshape(shape)
    if upcast:
        log.warning(f"Some {what} are stored as {sorted(upcast)}; upcasting them to float32")
    return out


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

    # Prefer the model weights; fall back to the fp32 master copies.
    model_keys = {normalize_olmo_core_key(k): k for k in metadata if k.startswith("model.")}
    keys = model_keys or _olmo_core_keys(metadata, "main")
    if not keys:
        raise RuntimeError(f"{model_and_optim} has neither model.* nor .main entries")
    return _load_olmo_core_tensors(
        model_and_optim, metadata, keys, olmo_core_shapes(config), "parameters"
    )


def load_olmo_core_optim_state(
    checkpoint_dir: PathOrStr, config: MaxTextModelConfig
) -> Optional[AdamState]:
    """
    Load the AdamW state from an OLMo Core checkpoint's ``model_and_optim/`` directory, or return
    ``None`` if it has no optimizer moments (e.g. a weights-only checkpoint).

    Handles the OLMoDDP train module's flat ``module.<name>.{exp_avg,exp_avg_sq,step}`` entries and
    the FSDP train module's flattened ``optim.state.<name>.{exp_avg,exp_avg_sq,step}`` entries.

    :raises ValueError: If the state isn't AdamW's or the parameters' step counts differ.
    :raises NotImplementedError: If the optimizer state was saved unflattened.
    """
    model_and_optim = join_path(checkpoint_dir, "model_and_optim")
    metadata = get_checkpoint_metadata(model_and_optim).state_dict_metadata
    if (
        f"{_FSDP_OPTIM_STATE_PREFIX}0.exp_avg" in metadata
        or "optim.param_groups.0.params" in metadata
    ):
        raise NotImplementedError(
            f"{model_and_optim} has an unflattened optimizer state (keyed by parameter index); "
            "only flattened optimizer state can be converted"
        )
    non_adam = sorted(k for k in metadata if k.endswith(_NON_ADAM_SUFFIXES))
    if non_adam:
        raise ValueError(f"{model_and_optim} has non-AdamW optimizer state, e.g. {non_adam[0]}")
    exp_avg_keys = _olmo_core_keys(metadata, "exp_avg")
    if not exp_avg_keys:
        return None

    shapes = olmo_core_shapes(config)
    step_keys = _olmo_core_keys(metadata, "step")
    missing = sorted(set(shapes) - set(step_keys))
    if missing:
        raise KeyError(f"{model_and_optim} is missing optimizer steps for: {missing[:10]}")
    steps = {int(t.item()) for t in load_keys(model_and_optim, [step_keys[n] for n in shapes])}
    if len(steps) != 1:
        raise ValueError(
            f"{model_and_optim}: parameters have taken different numbers of optimizer steps "
            f"({sorted(steps)}), but MaxText keeps a single step count"
        )
    return AdamState(
        step=steps.pop(),
        exp_avg=_load_olmo_core_tensors(
            model_and_optim, metadata, exp_avg_keys, shapes, "first moments"
        ),
        exp_avg_sq=_load_olmo_core_tensors(
            model_and_optim,
            metadata,
            _olmo_core_keys(metadata, "exp_avg_sq"),
            shapes,
            "second moments",
        ),
    )


def save_olmo_core_weights(
    output_dir: PathOrStr,
    state: Mapping[str, torch.Tensor],
    experiment_config: Mapping[str, Any],
    *,
    optim: Optional[AdamState] = None,
    optim_key_prefix: str = "module.",
    save_overwrite: bool = False,
) -> None:
    """
    Write an OLMo Core checkpoint: ``model_and_optim/`` with ``model.<name>`` entries at full
    shape, plus ``config.json``. This is the layout the Hugging Face importer writes.

    Without ``optim`` the checkpoint is weights only: the OLMoDDP train module rebuilds its fp32
    master copies from these weights and starts the optimizer fresh.

    With ``optim``, it also holds the OLMoDDP train module's flat optimizer entries
    ``<optim_key_prefix><name>.{main,exp_avg,exp_avg_sq,step}``, so that train module resumes
    with the optimizer state. The FSDP train module doesn't read these; it loads the weights and
    starts its optimizer fresh.

    :param optim_key_prefix: What the OLMoDDP train module puts before parameter names in its
        optimizer state: ``module.`` for a DDP-wrapped model, which is the usual case.
    """
    model_and_optim = join_path(output_dir, "model_and_optim")
    to_save: Dict[str, Any] = {"model": dict(state)}
    if optim is not None:
        moments = optim.moments()
        for name, weight in state.items():
            key = f"{optim_key_prefix}{name}"
            to_save[f"{key}.main"] = weight.reshape(-1)
            for suffix, values in moments.items():
                to_save[f"{key}.{suffix}"] = values[name].reshape(-1)
            to_save[f"{key}.step"] = torch.tensor(float(optim.step), dtype=torch.float32)
    log.info(
        f"Saving {len(state)} parameters{'' if optim is None else ' and their AdamW state'} "
        f"to '{model_and_optim}'"
    )
    save_state_dict(model_and_optim, to_save, save_overwrite=save_overwrite)
    write_json(
        join_path(output_dir, "config.json"), experiment_config, save_overwrite=save_overwrite
    )


def _flatten(tree: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested mappings and sequences (e.g. an optax chain's states) to ``/``-joined paths,
    indexing sequences by position. ``None`` entries (empty optax states) are dropped.
    """
    if isinstance(tree, Mapping):
        items: Iterable[Tuple[Any, Any]] = tree.items()
    elif isinstance(tree, (list, tuple)):
        items = enumerate(tree)
    else:
        return {prefix: tree}
    out = {}
    for k, v in items:
        if v is not None:
            out.update(_flatten(v, f"{prefix}/{k}" if prefix else str(k)))
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
    return load_maxtext_checkpoint(items_dir, optim=False)[0]


def load_maxtext_checkpoint(
    items_dir: str, *, optim: bool = True
) -> Tuple[Dict[str, torch.Tensor], Optional[AdamState]]:
    """
    Load a MaxText checkpoint's parameters as a flat ``/``-joined dict of fp32 tensors and, if
    ``optim`` is set, its AdamW state, keyed the same way. The state is ``None`` if the checkpoint
    has none (e.g. a parameters-only checkpoint).

    ``items_dir`` is a checkpoint step's ``items`` directory, e.g. ``.../checkpoints/50/items``.

    :raises ValueError: If the optimizer state isn't optax's ``adamw`` (MaxText's
        ``opt_type=adamw``). ``adam_pax``, for one, stores bias-corrected moments.
    """
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    import orbax.checkpoint as ocp  # type: ignore
    from etils import epath  # type: ignore

    path = epath.Path(items_dir)
    ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True))
    tree = ckptr.metadata(path).item_metadata.tree
    if "params" not in tree:
        raise KeyError(f"{items_dir} has no params; found {sorted(tree)}")
    has_optim = optim and bool(tree.get("opt_state"))
    tree = {k: tree[k] for k in (("params", "opt_state") if has_optim else ("params",))}
    # Restore to host memory: on an accelerator this would be a second copy of the model.
    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    restore_args = jax.tree_util.tree_map(
        lambda x: ocp.ArrayRestoreArgs(
            sharding=sharding,
            # Leave the step counts and flags as they are.
            dtype=np.float32 if jnp.issubdtype(x.dtype, jnp.floating) else None,
        ),
        tree,
        is_leaf=lambda x: hasattr(x, "shape"),
    )
    restored = ckptr.restore(
        path,
        args=ocp.args.PyTreeRestore(item=tree, restore_args=restore_args, partial_restore=True),
    )
    # On disk: params/params/<tree> (the item key, then Flax's params collection).
    params = {
        k: torch.from_numpy(np.array(v)) for k, v in _flatten(restored["params"]["params"]).items()
    }
    if not has_optim:
        if optim:
            log.warning(f"{items_dir} has no optimizer state")
        return params, None
    return params, _maxtext_adam_state(_flatten(restored["opt_state"]), items_dir)


def _maxtext_adam_state(opt_state: Mapping[str, Any], where: str) -> AdamState:
    """
    Find the AdamW state in a flattened MaxText ``opt_state``. On disk, optax's ``adamw`` is a
    chain whose first state holds ``count`` and the moments ``mu`` and ``nu``, each wrapped in
    Flax's ``params`` collection: ``0/mu/params/<param path>``. Wrappers such as
    ``skip_step_on_spikes`` (``inner_state/0/mu/...``) are looked through.
    """
    # The path to the state holding the moments -> "mu"/"nu" -> param path -> value.
    found: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    for path, value in opt_state.items():
        parts = path.split("/")
        i = next((i for i, p in enumerate(parts) if p in ("mu", "nu")), None)
        if i is None:
            continue
        rest = parts[i + 1 :]
        if rest[:1] == ["params"]:
            rest = rest[1:]
        moments = found.setdefault("/".join(parts[:i]), {})
        moments.setdefault(parts[i], {})["/".join(rest)] = torch.from_numpy(np.array(value))
    if len(found) != 1:
        raise ValueError(
            f"{where}: expected one AdamW state in opt_state, found {len(found)} "
            f"({sorted(found) or 'no mu/nu entries'}); only opt_type=adamw is supported"
        )
    prefix, moments = found.popitem()
    if set(moments) != {"mu", "nu"}:
        raise ValueError(f"{where}: opt_state has {sorted(moments)} but no second moments")
    if not prefix.rpartition("/")[2].isdigit():
        # optax.adamw is a chain, so its Adam state sits at an index. adam_pax's doesn't, and its
        # moments are bias-corrected, so they would need rescaling.
        raise ValueError(
            f"{where}: the Adam state at '{prefix or '<root>'}' isn't in an optax chain; only "
            "opt_type=adamw is supported"
        )
    count_key = f"{prefix}/count"
    if count_key not in opt_state:
        raise KeyError(f"{where}: opt_state has no {count_key}")
    return AdamState(
        step=int(np.array(opt_state[count_key])),
        exp_avg={k: v.float() for k, v in moments["mu"].items()},
        exp_avg_sq={k: v.float() for k, v in moments["nu"].items()},
    )


def maxtext_checkpoint_items(
    params: Mapping[str, torch.Tensor], optim: Optional[AdamState] = None
) -> Dict[str, Any]:
    """
    The ``items`` tree of a MaxText checkpoint in MaxText's (Linen) on-disk layout, as numpy
    arrays, from flat ``/``-joined parameters and, optionally, their AdamW state keyed the same way.

    Without ``optim`` this holds only the parameters, for ``load_parameters_path``. With it, it's a
    full training state for ``load_full_state_path``: ``step`` and ``opt_state`` as optax's
    ``adamw`` lays it out, ``[ScaleByAdamState(count, mu, nu), <weight decay: empty>,
    ScaleByScheduleState(count)]``. Both counts are the step, so MaxText's learning rate schedule
    resumes from it.
    """

    def tree(flat: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
        return _unflatten({k: v.numpy() for k, v in flat.items()})

    items: Dict[str, Any] = {"params": {"params": tree(params)}}
    if optim is None:
        return items
    for what, moments in (("first", optim.exp_avg), ("second", optim.exp_avg_sq)):
        if set(moments) != set(params):
            diff = sorted(set(moments) ^ set(params))
            raise KeyError(f"{what} moments and parameters have different keys: {diff[:10]}")
    count = np.asarray(optim.step, dtype=np.int32)
    items["step"] = count
    items["opt_state"] = [
        {
            "count": count,
            "mu": {"params": tree(optim.exp_avg)},
            "nu": {"params": tree(optim.exp_avg_sq)},
        },
        None,
        {"count": count},
    ]
    return items


def save_maxtext_params(output_dir: str, params: Mapping[str, torch.Tensor]) -> None:
    """
    Write a flat ``/``-joined parameter dict as a MaxText checkpoint at ``<output_dir>/0/items``,
    loadable with ``load_parameters_path=<output_dir>/0/items``. Uses MaxText's own checkpoint
    writer so the on-disk layout is exactly what MaxText expects.
    """
    save_maxtext_checkpoint(output_dir, params)


def save_maxtext_checkpoint(
    output_dir: str, params: Mapping[str, torch.Tensor], optim: Optional[AdamState] = None
) -> None:
    """
    Write a MaxText checkpoint at ``<output_dir>/0/items`` from flat ``/``-joined parameters and,
    optionally, their AdamW state keyed the same way (see :func:`maxtext_checkpoint_items`).

    Without ``optim``, load it with ``load_parameters_path=<output_dir>/0/items``. With it, load it
    with ``load_full_state_path=<output_dir>/0/items`` to resume from ``optim.step``.
    """
    if optim is None:
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
        return

    from maxtext.common import checkpointing  # type: ignore

    items = maxtext_checkpoint_items(params, optim)
    # Saved from host memory, unsharded: Orbax writes numpy arrays directly.
    manager = checkpointing.create_orbax_checkpoint_manager(
        output_dir,
        True,  # enable_checkpointing
        False,  # async_checkpointing
        1,  # save_interval_steps
        use_ocdbt=True,
        use_zarr3=True,
        enable_continuous_checkpointing=True,
    )
    if manager is None:
        raise RuntimeError("Failed to create an Orbax checkpoint manager")
    # Saved as step 0 like a parameters-only checkpoint; the training step is items["step"].
    if not checkpointing.save_checkpoint(manager, 0, items):
        raise RuntimeError(f"MaxText didn't save a checkpoint to {output_dir}")
    checkpointing.wait_until_finished(manager)


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
