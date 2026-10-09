"""
Utilities for converting models between OLMo Core and `MaxText <https://github.com/AI-Hypercomputer/maxtext>`_
formats. To change how OLMo Core state maps to MaxText state, you may change the mapping builders
in :mod:`olmo_core.nn.maxtext.convert`.
"""

from .checkpoint import (
    is_scanned,
    load_maxtext_params,
    load_olmo_core_config,
    load_olmo_core_weights,
    save_maxtext_params,
    save_olmo_core_weights,
    scan_params,
    unscan_params,
)
from .config import (
    MaxTextDecoderBlock,
    MaxTextModelConfig,
    ScanLayout,
    get_maxtext_config,
)
from .convert import (
    convert_state_from_maxtext,
    convert_state_to_maxtext,
    get_converter_from_maxtext,
    get_converter_to_maxtext,
    maxtext_shapes,
    olmo_core_shapes,
)
from .convert_checkpoint import (
    convert_checkpoint_from_maxtext,
    convert_checkpoint_to_maxtext,
    load_olmo_core_config_for_maxtext,
)

__all__ = [
    "MaxTextDecoderBlock",
    "MaxTextModelConfig",
    "ScanLayout",
    "convert_checkpoint_from_maxtext",
    "convert_checkpoint_to_maxtext",
    "convert_state_from_maxtext",
    "convert_state_to_maxtext",
    "get_converter_from_maxtext",
    "get_converter_to_maxtext",
    "get_maxtext_config",
    "is_scanned",
    "load_maxtext_params",
    "load_olmo_core_config",
    "load_olmo_core_config_for_maxtext",
    "load_olmo_core_weights",
    "maxtext_shapes",
    "olmo_core_shapes",
    "save_maxtext_params",
    "save_olmo_core_weights",
    "scan_params",
    "unscan_params",
]
