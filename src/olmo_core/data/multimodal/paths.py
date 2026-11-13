"""Shared on-disk data roots for Molmo2 multimodal training."""

from __future__ import annotations

import os

_DEFAULT_MOLMO_DATA = "/weka/oe-training-default/mm-olmo"
_DEFAULT_TORCH_DATASETS = f"{_DEFAULT_MOLMO_DATA}/torch_datasets"

MOLMO_DATA_DIR = os.environ.get("MOLMO_DATA_DIR", _DEFAULT_MOLMO_DATA)
TORCH_DATASETS = os.path.join(MOLMO_DATA_DIR, "torch_datasets")
PIXMO_DATASETS = os.path.join(TORCH_DATASETS, "pixmo_datasets")
TULU4_DATA = os.path.join(TORCH_DATASETS, "olmo-3-instruct-sft-no-tools-classified-v3")
ACADEMIC_DATASETS = os.path.join(TORCH_DATASETS, "academic_datasets")
# allenai/olmOCR-mix-1025 as materialised by mm_olmo's ``OlmOcrMixConfig.download``: the
# per-subset/split parquets plus the PDF tarballs expanded into ``pdfs/<chunk>/<arcname>``.
OLMOCR_MIX = os.path.join(TORCH_DATASETS, "olmocr_mix_1025")
# The oe-encoder team's webdataset-style caption tars (``<key>.jpg|png`` + ``<key>.json`` per
# sample), used for the OCR sources in :mod:`.mixtures.ocr`. Another project's directory, so
# it is overridable with the OE_ENCODER_DATA_DIR env var.
OE_ENCODER_DATA = os.environ.get("OE_ENCODER_DATA_DIR", "/weka/oe-training-default/oe-encoder")
# mm_olmo's ``text_rich_caption`` build: one HF ``DatasetDict`` per category under
# ``hf/<category>`` with a ``train`` split and a held-out ``validation`` split. The
# ``text_rich_caption_v6_tars`` are the same images with no split applied, so this build's
# ``validation`` ids are what the tar sources must exclude (see :mod:`.mixtures.ocr`).
TEXT_RICH_CAPTION = os.environ.get(
    "TEXT_RICH_CAPTION_DIR", os.path.join(MOLMO_DATA_DIR, "molmo3_datasets", "text_rich_caption")
)

# HARDCODED personal dataset (chrisc's audited, image-grouped PixMo-Points build on weka).
# mm_olmo's ``PixMoPointV2.PATH`` reads this same directory, so it is the canonical location
# for now. Override with the PIXMO_POINTS_V2_DIR env var, or per run through the dataset
# config (``--pointing_v2.dataset_path=...`` in Molmo2-Stage1.py).
PIXMO_POINTS_V2 = os.environ.get(
    "PIXMO_POINTS_V2_DIR", "/weka/oe-training-default/chrisc/pixmo-points-with-masks-v17"
)

# HARDCODED personal datasets (jasonr's downloads of two synthetic OCR sets on weka), read by
# :mod:`.synthetic_ocr`. Override with the env vars, or per run through the dataset configs
# (``--nvidia_synth.dataset_path=...`` / ``--receipts.dataset_path=...`` in Molmo2-Stage1.py).
# nvidia/OCR-Synthetic-Multilingual-v1, ``en/train/*.h5`` only:
NVIDIA_SYNTH_OCR = os.environ.get(
    "NVIDIA_SYNTH_OCR_DIR",
    "/weka/oe-training-default/jasonr/dataset/ocr/nvidia-ocr-synthetic-multilingual-v1",
)
# albertobarnabo/synthetic-receipts-ocr, ``data/train-*.parquet`` only:
SYNTH_RECEIPTS_OCR = os.environ.get(
    "SYNTH_RECEIPTS_OCR_DIR", "/weka/oe-training-default/jasonr/dataset/ocr/synthetic-receipts-ocr"
)

__all__ = [
    "MOLMO_DATA_DIR",
    "TORCH_DATASETS",
    "PIXMO_DATASETS",
    "PIXMO_POINTS_V2",
    "TULU4_DATA",
    "ACADEMIC_DATASETS",
    "OLMOCR_MIX",
    "OE_ENCODER_DATA",
    "TEXT_RICH_CAPTION",
    "NVIDIA_SYNTH_OCR",
    "SYNTH_RECEIPTS_OCR",
]
