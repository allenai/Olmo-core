"""Tests for parity-critical loader and formatter behavior."""

from __future__ import annotations

import json

import numpy as np
import pytest

from olmo_core.data.multimodal.academic.registry import build_academic_data
from olmo_core.data.multimodal.pixmo_clocks import format_pixmo_clocks_row


def test_st_qa_train_validation_split():
    train = build_academic_data("st_qa", split="train")
    val = build_academic_data("st_qa", split="validation")
    assert len(val) == 1024
    assert len(train) > 0
    train_ids = {ex["metadata"]["example_id"] for ex in train}
    val_ids = {ex["metadata"]["example_id"] for ex in val}
    assert train_ids.isdisjoint(val_ids)


def test_pixmo_clocks_aug_is_deterministic():
    clocks_jsonl = (
        "/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets/clocks/train.jsonl"
    )
    try:
        with open(clocks_jsonl) as f:
            row = json.loads(f.readline())
    except OSError:
        pytest.skip("pixmo clocks data not available")

    rng = np.random.RandomState(0)
    a = format_pixmo_clocks_row(row, rng, aug=True)
    rng = np.random.RandomState(0)
    b = format_pixmo_clocks_row(row, rng, aug=True)
    assert np.array_equal(a["image"], b["image"])
    assert a["text"] == b["text"]
