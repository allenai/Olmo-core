"""
Convert OLMo Core checkpoints to MaxText checkpoints and back: the weights and, optionally, the
AdamW optimizer state, so training resumes where it left off instead of with a fresh optimizer.
"""

import logging
from typing import Any, Callable, Dict, Mapping, Optional

import torch

from olmo_core.aliases import PathOrStr
from olmo_core.doc_utils import beta_feature
from olmo_core.io import file_exists, normalize_path

from .checkpoint import (
    AdamState,
    is_scanned,
    load_maxtext_checkpoint,
    load_olmo_core_config,
    load_olmo_core_optim_state,
    load_olmo_core_weights,
    read_json,
    save_maxtext_checkpoint,
    save_olmo_core_weights,
    scan_params,
    unscan_params,
    write_json,
)
from .config import MaxTextModelConfig, get_maxtext_config
from .convert import _convert

__all__ = [
    "OLMO_CORE_CONFIG_FILE",
    "convert_checkpoint_to_maxtext",
    "convert_checkpoint_from_maxtext",
    "load_olmo_core_config_for_maxtext",
]

log = logging.getLogger(__name__)

#: Written next to a converted MaxText checkpoint (beside its ``0/`` step directory) so it can be
#: converted back without passing the OLMo Core config again.
OLMO_CORE_CONFIG_FILE = "olmo_core_config.json"


def _map_moments(
    optim: AdamState,
    fn: Callable[[Dict[str, torch.Tensor]], Mapping[str, torch.Tensor]],
) -> AdamState:
    return AdamState(
        step=optim.step, exp_avg=dict(fn(optim.exp_avg)), exp_avg_sq=dict(fn(optim.exp_avg_sq))
    )


def _convert_optim(
    config: MaxTextModelConfig, optim: AdamState, *, to_maxtext: bool, validate: bool
) -> AdamState:
    """The moments are laid out like the weights, so they convert with the same mappings."""
    return _map_moments(
        optim,
        lambda moments: _convert(
            config, moments, to_maxtext=to_maxtext, consume=True, round_trip=validate
        ),
    )


@beta_feature
def convert_checkpoint_to_maxtext(
    checkpoint_path: PathOrStr,
    output_path: str,
    *,
    scan_layers: bool = True,
    optimizer_state: bool = False,
    validate: bool = True,
) -> None:
    """
    Convert an OLMo Core checkpoint to a MaxText checkpoint at ``<output_path>/0/items``. The OLMo
    Core config is copied to ``<output_path>/olmo_core_config.json``.

    Without ``optimizer_state``, only the weights are converted: load them in MaxText with
    ``load_parameters_path=<output_path>/0/items`` and it starts its optimizer fresh.

    With ``optimizer_state``, the AdamW moments and step are converted too: load the checkpoint
    with ``load_full_state_path=<output_path>/0/items`` to resume at that step, including the
    learning rate schedule. MaxText must use ``opt_type=adamw`` without ``skip_step_on_spikes`` or
    ``trainable_parameters_mask``, and you should set its AdamW hyperparameters (``adam_b1``,
    ``adam_b2``, ``adam_eps``, ``adam_weight_decay``) and schedule to match the OLMo Core run.

    :param checkpoint_path: An OLMo Core checkpoint step directory (with ``model_and_optim/`` and
        ``config.json``).
    :param output_path: Where to write the MaxText checkpoint.
    :param scan_layers: Write the layout for MaxText's ``scan_layers=True`` (the default there).
    :param optimizer_state: Also convert the AdamW state.
    :param validate: Check that converting each layer back reproduces the original tensors.

    :raises ValueError: If ``optimizer_state`` is set and the checkpoint has no AdamW state.
    """
    experiment_config = load_olmo_core_config(checkpoint_path)
    config = get_maxtext_config(experiment_config["model"])
    log.info(f"Converting to MaxText's {config.decoder_block} decoder")
    state = load_olmo_core_weights(checkpoint_path, config)
    optim = None
    if optimizer_state:
        optim = load_olmo_core_optim_state(checkpoint_path, config)
        if optim is None:
            raise ValueError(f"{checkpoint_path} has no optimizer state to convert")
    params = _convert(config, state, to_maxtext=True, consume=True, round_trip=validate)
    if optim is not None:
        optim = _convert_optim(config, optim, to_maxtext=True, validate=validate)
    if validate:
        log.info("Converting back reproduces every tensor exactly")
    if scan_layers:
        params = scan_params(params, config.scan_layout, config.n_layers)
        if optim is not None:
            optim = _map_moments(
                optim, lambda m: scan_params(m, config.scan_layout, config.n_layers)
            )
    save_maxtext_checkpoint(output_path, params, optim)
    write_json(
        f"{normalize_path(output_path)}/{OLMO_CORE_CONFIG_FILE}",
        experiment_config,
        save_overwrite=True,
    )
    load_key = "load_parameters_path" if optim is None else "load_full_state_path"
    log.info(
        f"Wrote a scan_layers={scan_layers} MaxText checkpoint"
        f"{'' if optim is None else f' with AdamW state at step {optim.step}'}; load it with "
        f"model_name={config.model_name} override_model_config=True "
        f"{load_key}={output_path}/0/items and these overrides: {' '.join(config.overrides())}"
    )


def load_olmo_core_config_for_maxtext(maxtext_items_path: str) -> Dict[str, Any]:
    """
    Load the OLMo Core experiment config that :func:`convert_checkpoint_to_maxtext` saved next to a
    MaxText checkpoint (``<output>/0/items`` -> ``<output>/olmo_core_config.json``).
    """
    root = normalize_path(maxtext_items_path).rstrip("/").rsplit("/", 2)[0]
    path = f"{root}/{OLMO_CORE_CONFIG_FILE}"
    if not file_exists(path):
        raise FileNotFoundError(f"no {OLMO_CORE_CONFIG_FILE} next to {maxtext_items_path}")
    return read_json(path)


@beta_feature
def convert_checkpoint_from_maxtext(
    maxtext_items_path: str,
    output_path: PathOrStr,
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
    optimizer_state: bool = False,
    optim_key_prefix: str = "module.",
    validate: bool = True,
    save_overwrite: bool = False,
) -> None:
    """
    Convert a MaxText checkpoint to an OLMo Core checkpoint: ``model_and_optim/`` with
    ``model.<name>`` entries at full shape, and ``config.json``. Scanned and unscanned
    checkpoints both work.

    With ``optimizer_state``, the AdamW moments and step are converted too, and written as the
    OLMoDDP train module's optimizer entries (see
    :func:`~olmo_core.nn.maxtext.checkpoint.save_olmo_core_weights`), so that train module resumes
    with them. The MaxText run must have used ``opt_type=adamw``. Set OLMo Core's AdamW
    hyperparameters and learning rate schedule to match the MaxText run.

    :param maxtext_items_path: A MaxText checkpoint step's ``items`` directory,
        e.g. ``.../checkpoints/50/items``.
    :param output_path: Where to write the OLMo Core checkpoint.
    :param experiment_config: The OLMo Core experiment config describing the model. Defaults to the
        copy :func:`convert_checkpoint_to_maxtext` saved next to the checkpoint.
    :param optimizer_state: Also convert the AdamW state.
    :param optim_key_prefix: What the OLMoDDP train module puts before parameter names in its
        optimizer state; ``module.`` for a DDP-wrapped model.
    :param validate: Check that converting each layer back reproduces the MaxText tensors.

    :raises ValueError: If ``optimizer_state`` is set and the checkpoint has no AdamW state.
    """
    experiment_config = experiment_config or load_olmo_core_config_for_maxtext(maxtext_items_path)
    config = get_maxtext_config(experiment_config["model"])
    params, optim = load_maxtext_checkpoint(maxtext_items_path, optim=optimizer_state)
    if optimizer_state and optim is None:
        raise ValueError(f"{maxtext_items_path} has no optimizer state to convert")
    if is_scanned(params, config.scan_layout):
        params = unscan_params(params, config.scan_layout)
        if optim is not None:
            optim = _map_moments(optim, lambda m: unscan_params(m, config.scan_layout))
    state = _convert(config, params, to_maxtext=False, consume=True, round_trip=validate)
    if optim is not None:
        optim = _convert_optim(config, optim, to_maxtext=False, validate=validate)
    if validate:
        log.info("Converting back reproduces every tensor exactly")
    save_olmo_core_weights(
        output_path,
        state,
        experiment_config,
        optim=optim,
        optim_key_prefix=optim_key_prefix,
        save_overwrite=save_overwrite,
    )
    log.info(
        f"Wrote an OLMo Core checkpoint{'' if optim is None else f' with AdamW state at step {optim.step}'} "
        f"to '{output_path}'"
    )
