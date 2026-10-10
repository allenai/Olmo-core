from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from olmo_core.data.composable import InstanceSource, InstanceSourceConfig
from olmo_core.data.multimodal.pretraining_replay import (
    ComposableTextReplayConfig,
    ComposableTextReplayDataset,
)
from olmo_core.exceptions import OLMoConfigurationError

PAD, EOS = 127, 126


class _ListSource(InstanceSource):
    def __init__(self, instances: list[dict[str, Any]], work_dir, sequence_length: int):
        super().__init__(work_dir=work_dir, sequence_length=sequence_length)
        self.instances = instances

    @property
    def fingerprint(self) -> str:
        return f"list-{len(self.instances)}"

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def children(self):
        return ()


@dataclass
class _ListSourceConfig(InstanceSourceConfig):
    instances: list[dict[str, Any]] = field(default_factory=list)
    sequence_length: int = 8

    def build(self, work_dir):
        return _ListSource(self.instances, work_dir, self.sequence_length)


def _replay(tmp_path, *instances, **kwargs) -> ComposableTextReplayDataset:
    return ComposableTextReplayConfig(
        source=_ListSourceConfig(instances=list(instances)), work_dir=str(tmp_path), **kwargs
    ).build()


def test_unmasked_instance_is_a_plain_text_example(tmp_path):
    tokens = np.arange(1, 9)
    example = _replay(tmp_path, {"input_ids": tokens}).get(0, epoch=3)
    np.testing.assert_array_equal(example["input_ids"], tokens)
    np.testing.assert_array_equal(example["labels"], [*tokens[1:], -100])
    np.testing.assert_array_equal(example["loss_masks"], [1] * 7 + [0])
    np.testing.assert_array_equal(example["position_ids"], np.arange(8))
    assert example["images"].shape[0] == 0 and example["pooled_patches_idx"].shape[0] == 0
    assert example["metadata"] == {"instance_filter_valid": True}


def test_label_mask_marks_targets_and_chunk_boundaries(tmp_path):
    """Two concatenated 4-token chunks: the second chunk's first token is not a target."""
    tokens = np.arange(1, 9)
    mask = np.array([0, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    example = _replay(tmp_path, {"input_ids": tokens, "label_mask": mask}).get(0)
    np.testing.assert_array_equal(example["labels"], [2, 3, 4, -100, 6, 7, 8, -100])
    np.testing.assert_array_equal(example["loss_masks"], [1, 1, 1, 0, 1, 1, 1, 0])


def test_filtered_chunk_keeps_its_denominator_weight(tmp_path):
    """A filtered 4-token chunk predicts nothing but counts 4 positions, as the text run's
    ``loss_denominator_extra_tokens`` adds them to the divisor."""
    tokens = np.arange(1, 9)
    mask = np.array([0, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    instance = {"input_ids": tokens, "label_mask": mask, "loss_denominator_extra_tokens": 4}
    example = _replay(tmp_path, instance).get(0)
    np.testing.assert_array_equal(example["labels"], [2, 3, 4, -100, -100, -100, -100, -100])
    # The text run's divisor: 3 targets plus the filtered chunk's 4 tokens.
    assert example["loss_masks"].sum() == 3 + 4
    assert ((example["labels"] != -100) & (example["loss_masks"] == 0)).sum() == 0
    with pytest.raises(RuntimeError, match="extra denominator"):
        _replay(tmp_path, {**instance, "loss_denominator_extra_tokens": 7}).get(0)


def test_fully_filtered_instance_keeps_l_minus_one_weight(tmp_path):
    instance = {"input_ids": np.arange(1, 9), "instance_mask": False}
    example = _replay(tmp_path, instance).get(0)
    assert (example["labels"] == -100).all()
    np.testing.assert_array_equal(example["loss_masks"], [1] * 7 + [0])
    assert example["metadata"] == {"instance_filter_valid": False}


def test_pad_is_fed_as_eos_and_never_predicted(tmp_path):
    tokens = np.array([1, 2, PAD, PAD, 5, 6, 7, 8])
    example = _replay(tmp_path, {"input_ids": tokens}, pad_token_id=PAD, eos_token_id=EOS).get(0)
    np.testing.assert_array_equal(example["input_ids"], [1, 2, EOS, EOS, 5, 6, 7, 8])
    np.testing.assert_array_equal(example["labels"], [2, -100, -100, 5, 6, 7, 8, -100])
    np.testing.assert_array_equal(example["loss_masks"], [1, 0, 0, 1, 1, 1, 1, 0])
    # Without the ids, PAD stays an ordinary token.
    plain = _replay(tmp_path, {"input_ids": tokens}).get(0)
    np.testing.assert_array_equal(plain["input_ids"], tokens)
    with pytest.raises(OLMoConfigurationError, match="both"):
        _replay(tmp_path, {"input_ids": tokens}, pad_token_id=PAD)


def test_length_index_and_fingerprint(tmp_path):
    replay = _replay(tmp_path, {"input_ids": np.arange(8)}, {"input_ids": np.arange(8, 16)})
    assert len(replay) == 2
    np.testing.assert_array_equal(replay[-1]["input_ids"], np.arange(8, 16))
    with pytest.raises(IndexError):
        replay.get(2)
    other = _replay(
        tmp_path,
        {"input_ids": np.arange(8)},
        {"input_ids": np.arange(8, 16)},
        pad_token_id=PAD,
        eos_token_id=EOS,
    )
    assert replay.fingerprint != other.fingerprint
    with pytest.raises(RuntimeError, match="expected 8"):
        _replay(tmp_path, {"input_ids": np.arange(6)}).get(0)
