"""
Utilities for converting models between OLMo Core and `MaxText <https://github.com/AI-Hypercomputer/maxtext>`_
formats. To change how OLMo Core state maps to MaxText state, you may change the mapping builders
in :mod:`olmo_core.nn.maxtext.convert`.
"""

from .config import MaxTextDecoderBlock, MaxTextModelConfig, get_maxtext_config
from .convert import convert_state_from_maxtext, convert_state_to_maxtext
from .convert_checkpoint import (
    convert_checkpoint_from_maxtext,
    convert_checkpoint_to_maxtext,
)

__all__ = [
    "MaxTextDecoderBlock",
    "MaxTextModelConfig",
    "get_maxtext_config",
    "convert_state_from_maxtext",
    "convert_state_to_maxtext",
    "convert_checkpoint_from_maxtext",
    "convert_checkpoint_to_maxtext",
]
