"""
Multimodal (vision-language) training data: datasets and collation for Molmo2.

This subpackage provides a standalone, ``mm_olmo``-free pipeline for Molmo2 training data:

* :class:`~olmo_core.data.multimodal.pixmo_cap.PixMoCapDataset` — map-style dataset
  yielding packed image + caption/transcript training examples.
* :class:`~olmo_core.data.multimodal.collator.MultimodalCollator` — pads/stacks them
  into batches for :class:`~olmo_core.nn.vision.MultimodalLM`.
* :func:`~olmo_core.data.multimodal.sequence_builder.build_packed_sequence` — the
  core multi-annotation (branch-packing) sequence assembly with float loss weights.
* :class:`~olmo_core.data.multimodal.mixture_data_loader.MixtureDataLoader` — weighted,
  resumable mixing of several datasets.

Unlike the text-only :mod:`olmo_core.data.composable` pipeline (a token-stream
packer), this carries variable-shape image tensors alongside the token sequence.
"""

from .collator import MultimodalCollator, MultimodalCollatorConfig
from .data_loader import MultimodalDataLoader
from .message_weight import MessageWeight, apply_message_weight_to_loss_masks
from .mixture_data_loader import MixtureDataLoader
from .mixture_weights import DatasetSource, SubMixture, compute_flat_mixture_weights
from .packing import pack_examples
from .paths import (
    ACADEMIC_DATASETS,
    MOLMO_DATA_DIR,
    OE_ENCODER_DATA,
    OLMOCR_MIX,
    PIXMO_DATASETS,
    PIXMO_POINTS_V2,
    TORCH_DATASETS,
    TULU4_DATA,
)
from .pixmo_cap import PixMoCapDataset, PixMoCapDatasetConfig
from .sequence_builder import (
    ATTEND_ALL_SUBSEGMENT_ID,
    build_branched_sequence,
    build_packed_sequence,
)
from .sft_formatter import SftFormatter

__all__ = [
    "PixMoCapDataset",
    "PixMoCapDatasetConfig",
    "SftFormatter",
    "MessageWeight",
    "apply_message_weight_to_loss_masks",
    "DatasetSource",
    "SubMixture",
    "compute_flat_mixture_weights",
    "PIXMO_DATASETS",
    "PIXMO_POINTS_V2",
    "TULU4_DATA",
    "ACADEMIC_DATASETS",
    "OLMOCR_MIX",
    "OE_ENCODER_DATA",
    "MOLMO_DATA_DIR",
    "TORCH_DATASETS",
    "MultimodalCollator",
    "MultimodalCollatorConfig",
    "MultimodalDataLoader",
    "MixtureDataLoader",
    "build_packed_sequence",
    "build_branched_sequence",
    "ATTEND_ALL_SUBSEGMENT_ID",
    "pack_examples",
]
