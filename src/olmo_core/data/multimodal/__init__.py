"""
Multimodal (vision-language) training data: datasets and collation for Molmo2.

This subpackage provides a standalone, ``mm_olmo``-free pipeline for Molmo2 "stage 1"
caption pretraining:

* :class:`~olmo_core.data.multimodal.pixmo_cap.PixMoCapDataset` — map-style dataset
  yielding packed image + caption/transcript training examples.
* :class:`~olmo_core.data.multimodal.collator.MultimodalCollator` — pads/stacks them
  into batches for :class:`~olmo_core.nn.vision.MultimodalLM`.
* :func:`~olmo_core.data.multimodal.sequence_builder.build_packed_sequence` — the
  core multi-annotation (branch-packing) sequence assembly with float loss weights.

Unlike the text-only :mod:`olmo_core.data.composable` pipeline (a token-stream
packer), this carries variable-shape image tensors alongside the token sequence.
"""

from .academic_dataset import (
    AcademicDataset,
    AcademicDatasetConfig,
    Stage1AcademicDataset,
    Stage1AcademicDatasetConfig,
)
from .collator import MultimodalCollator, MultimodalCollatorConfig
from .data_loader import MultimodalDataLoader
from .finevision import (
    FINEVISION_ROOT,
    FineVisionDataset,
    FineVisionDatasetConfig,
    VisualWebInstructDataset,
    VisualWebInstructDatasetConfig,
)
from .gui_syn import GuiSynDataset, GuiSynDatasetConfig
from .message_weight import MessageWeight, apply_message_weight_to_loss_masks
from .mixture_data_loader import MixtureDataLoader
from .mixture_weights import DatasetSource, SubMixture, compute_flat_mixture_weights
from .mmfinereason import (
    MMFineReasonDataset,
    MMFineReasonDatasetConfig,
    extract_answer_text,
)
from .ocr_caption_tars import (
    OcrCaptionTarsDataset,
    OcrCaptionTarsDatasetConfig,
    TarShardIndex,
)
from .olmocr import OlmOcrMixDataset, OlmOcrMixDatasetConfig
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
from .pixmo_ama import PixMoAmaDataset, PixMoAmaDatasetConfig
from .pixmo_cap import PixMoCapDataset, PixMoCapDatasetConfig
from .pixmo_cap_qa import PixMoCapQaDataset, PixMoCapQaDatasetConfig
from .pixmo_points import (
    CoSynPointDataset,
    CoSynPointDatasetConfig,
    PixMoCountDataset,
    PixMoCountDatasetConfig,
    PixMoPointsDataset,
    PixMoPointsDatasetConfig,
)
from .pixmo_points_v2 import (
    PixMoCountV2Dataset,
    PixMoCountV2DatasetConfig,
    PixMoPointsV2Dataset,
    PixMoPointsV2DatasetConfig,
)
from .sequence_builder import (
    ATTEND_ALL_SUBSEGMENT_ID,
    build_branched_sequence,
    build_packed_sequence,
)
from .sft_formatter import SftFormatter
from .synthetic_ocr import (
    NvidiaSynthOcrDataset,
    NvidiaSynthOcrDatasetConfig,
    SyntheticReceiptsDataset,
    SyntheticReceiptsDatasetConfig,
)
from .text_rich_caption import TextRichCaptionDataset, TextRichCaptionDatasetConfig
from .tulu import Tulu4Dataset, Tulu4DatasetConfig

__all__ = [
    "FineVisionDataset",
    "FineVisionDatasetConfig",
    "VisualWebInstructDataset",
    "VisualWebInstructDatasetConfig",
    "FINEVISION_ROOT",
    "MMFineReasonDataset",
    "MMFineReasonDatasetConfig",
    "extract_answer_text",
    "PixMoCapDataset",
    "PixMoCapDatasetConfig",
    "PixMoPointsDataset",
    "PixMoPointsDatasetConfig",
    "PixMoCountDataset",
    "PixMoCountDatasetConfig",
    "CoSynPointDataset",
    "CoSynPointDatasetConfig",
    "PixMoPointsV2Dataset",
    "PixMoPointsV2DatasetConfig",
    "PixMoCountV2Dataset",
    "PixMoCountV2DatasetConfig",
    "GuiSynDataset",
    "GuiSynDatasetConfig",
    "OlmOcrMixDataset",
    "OlmOcrMixDatasetConfig",
    "OcrCaptionTarsDataset",
    "TextRichCaptionDataset",
    "TextRichCaptionDatasetConfig",
    "OcrCaptionTarsDatasetConfig",
    "NvidiaSynthOcrDataset",
    "NvidiaSynthOcrDatasetConfig",
    "SyntheticReceiptsDataset",
    "SyntheticReceiptsDatasetConfig",
    "TarShardIndex",
    "Tulu4Dataset",
    "Tulu4DatasetConfig",
    "AcademicDataset",
    "AcademicDatasetConfig",
    "Stage1AcademicDataset",
    "Stage1AcademicDatasetConfig",
    "PixMoAmaDataset",
    "PixMoAmaDatasetConfig",
    "PixMoCapQaDataset",
    "PixMoCapQaDatasetConfig",
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
