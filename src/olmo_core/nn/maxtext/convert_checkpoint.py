"""
Convert OLMo Core checkpoints to MaxText checkpoints and back.

This module exposes :func:`convert_checkpoint_to_maxtext` and
:func:`convert_checkpoint_from_maxtext` for programmatic use and for the example CLI scripts in
``src/examples/maxtext/``.

Only weights are converted. Optimizer state isn't, so the target framework starts its optimizer
fresh. Full-size checkpoints are held in memory in fp32, so convert large models on a high-memory
machine.
"""

import json
import logging
from typing import Any, Dict, Optional

from cached_path import cached_path

from olmo_core.aliases import PathOrStr
from olmo_core.doc_utils import beta_feature
from olmo_core.io import file_exists, normalize_path

from .checkpoint import (
    is_scanned,
    load_maxtext_params,
    load_olmo_core_config,
    load_olmo_core_weights,
    save_maxtext_params,
    save_olmo_core_weights,
    scan_params,
    unscan_params,
    write_json,
)
from .config import get_maxtext_config
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


@beta_feature
def convert_checkpoint_to_maxtext(
    checkpoint_path: PathOrStr,
    output_path: str,
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
    scan_layers: bool = True,
    validate: bool = True,
) -> None:
    """
    Convert an OLMo Core checkpoint's weights to a MaxText checkpoint at ``<output_path>/0/items``,
    loadable in MaxText with ``load_parameters_path=<output_path>/0/items``. The OLMo Core config is
    copied to ``<output_path>/olmo_core_config.json``.

    :param checkpoint_path: An OLMo Core checkpoint step directory (with ``model_and_optim/``).
    :param output_path: Where to write the MaxText checkpoint.
    :param experiment_config: The OLMo Core experiment config. Defaults to the checkpoint's
        ``config.json``.
    :param scan_layers: Write the layout for MaxText's ``scan_layers=True`` (the default there).
    :param validate: Check that converting each layer back reproduces the original weights.
    """
    experiment_config = experiment_config or load_olmo_core_config(checkpoint_path)
    config = get_maxtext_config(experiment_config["model"])
    log.info(f"Converting to MaxText's {config.decoder_block} decoder")
    state = load_olmo_core_weights(checkpoint_path, config)
    params = _convert(config, state, to_maxtext=True, consume=True, round_trip=validate)
    if validate:
        log.info("Converting back reproduces every parameter exactly")
    if scan_layers:
        params = scan_params(params, config.scan_layout, config.n_layers)
    save_maxtext_params(output_path, params)
    write_json(
        f"{normalize_path(output_path)}/{OLMO_CORE_CONFIG_FILE}",
        experiment_config,
        save_overwrite=True,
    )
    log.info(
        f"Wrote a scan_layers={scan_layers} MaxText checkpoint; load it with "
        f"model_name={config.model_name} override_model_config=True "
        f"load_parameters_path={output_path}/0/items and these overrides: {' '.join(config.overrides())}"
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
    with cached_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


@beta_feature
def convert_checkpoint_from_maxtext(
    maxtext_items_path: str,
    output_path: PathOrStr,
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
    validate: bool = True,
    save_overwrite: bool = False,
) -> None:
    """
    Convert the weights of a MaxText checkpoint to an OLMo Core checkpoint: ``model_and_optim/``
    with ``model.<name>`` entries at full shape, and ``config.json``. Scanned and unscanned
    checkpoints both work.

    :param maxtext_items_path: A MaxText checkpoint step's ``items`` directory,
        e.g. ``.../checkpoints/50/items``.
    :param output_path: Where to write the OLMo Core checkpoint.
    :param experiment_config: The OLMo Core experiment config describing the model. Defaults to the
        copy :func:`convert_checkpoint_to_maxtext` saved next to the checkpoint.
    :param validate: Check that converting each layer back reproduces the MaxText weights.
    """
    experiment_config = experiment_config or load_olmo_core_config_for_maxtext(maxtext_items_path)
    config = get_maxtext_config(experiment_config["model"])
    params = load_maxtext_params(maxtext_items_path)
    if is_scanned(params, config.scan_layout):
        params = unscan_params(params, config.scan_layout)
    state = _convert(config, params, to_maxtext=False, consume=True, round_trip=validate)
    if validate:
        log.info("Converting back reproduces every parameter exactly")
    save_olmo_core_weights(output_path, state, experiment_config, save_overwrite=save_overwrite)
    log.info(f"Wrote an OLMo Core checkpoint to '{output_path}'")
