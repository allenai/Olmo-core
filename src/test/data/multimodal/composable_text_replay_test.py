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


class _InterleavedSource(_ListSource):
    """Two children concatenated, with ``locate`` as the ladder's interleaved source has."""

    def __init__(self, first, second, work_dir, sequence_length: int):
        super().__init__([*first, *second], work_dir, sequence_length)
        self.n_first = len(first)

    def locate(self, idx: int):
        return (0, idx) if idx < self.n_first else (1, idx - self.n_first)


@dataclass
class _InterleavedSourceConfig(InstanceSourceConfig):
    first: list[dict[str, Any]] = field(default_factory=list)
    second: list[dict[str, Any]] = field(default_factory=list)
    sequence_length: int = 16

    def build(self, work_dir):
        return _InterleavedSource(self.first, self.second, work_dir, self.sequence_length)


def _reference_document_lengths(input_ids, label_mask, eos_token_id, chunk):
    """scaling-ladders ``hero_mt_lc_256k._document_lengths``, verbatim but for numpy."""
    n = input_ids.shape[0]
    is_eos = input_ids == eos_token_id
    ends = np.zeros(n + 1, dtype=bool)
    ends[1:] = is_eos
    if chunk is not None:
        ends[np.arange(chunk, n, chunk)] = True
    ends[0] = ends[n] = True
    if label_mask is not None:
        pad = is_eos & ~label_mask.astype(bool)
        inside_pad_run = np.zeros(n + 1, dtype=bool)
        inside_pad_run[1:n] = pad[:-1] & pad[1:]
        if chunk is not None:
            inside_pad_run[np.arange(chunk, n, chunk)] = False
        ends &= ~inside_pad_run
    boundaries = np.flatnonzero(ends)
    return np.diff(boundaries)


@pytest.mark.parametrize("chunked", [True, False])
def test_document_masking_matches_the_text_stages_segments(tmp_path, chunked):
    # EOS mid-document, a PAD run (fed as EOS, not a target) that crosses the 8-token chunk edge,
    # and a document straddling that edge.
    tokens = np.array([5, 6, EOS, 7, 8, 9, PAD, PAD, PAD, 10, 11, EOS, 12, 13, PAD, PAD])
    mask = tokens != PAD
    mt, lc = [{"input_ids": tokens, "label_mask": mask}], [
        {"input_ids": tokens, "label_mask": mask}
    ]
    replay = ComposableTextReplayConfig(
        source=_InterleavedSourceConfig(first=mt, second=lc),
        work_dir=str(tmp_path),
        pad_token_id=PAD,
        eos_token_id=EOS,
        document_masking=True,
        chunked_source=0,
        chunk_length=8,
    ).build()
    example = replay.get(0 if chunked else 1)
    fed = np.where(tokens == PAD, EOS, tokens)
    expected = _reference_document_lengths(fed, mask, EOS, 8 if chunked else None)
    ids = example["document_ids"]
    assert np.all(np.diff(ids) >= 0) and ids[0] == 0
    np.testing.assert_array_equal(np.bincount(ids), expected)


def _reference_cap_segments(lengths, max_segments):
    """scaling-ladders ``hero_mt_lc_256k._cap_segments``, verbatim but for numpy."""
    if len(lengths) <= max_segments:
        return lengths
    min_group = -(-int(lengths.sum()) // max_segments)
    groups, current = [], 0
    for length in lengths.tolist():
        current += length
        if current >= min_group:
            groups.append(current)
            current = 0
    if current:
        groups.append(current)
    return np.asarray(groups)


@pytest.mark.parametrize("max_segments", [3, 5, 8, None])
def test_document_masking_caps_segments_as_the_text_stage(tmp_path, max_segments):
    # Eight 2-token documents (the degenerate shape: many tiny EOS-separated segments).
    tokens = np.array([5, EOS] * 8)
    replay = ComposableTextReplayConfig(
        source=_ListSourceConfig(instances=[{"input_ids": tokens}], sequence_length=16),
        work_dir=str(tmp_path),
        pad_token_id=PAD,
        eos_token_id=EOS,
        document_masking=True,
        max_document_segments=max_segments,
    ).build()
    ids = replay.get(0)["document_ids"]
    expected = _reference_document_lengths(tokens, None, EOS, None)
    if max_segments is not None:
        expected = _reference_cap_segments(expected, max_segments)
        assert len(np.bincount(ids)) <= max_segments
    np.testing.assert_array_equal(np.bincount(ids), expected)
    assert len(ids) == len(tokens)


def test_document_masking_is_off_by_default(tmp_path):
    example = _replay(tmp_path, {"input_ids": np.arange(1, 9)}).get(0)
    assert "document_ids" not in example


def test_document_masking_needs_eos(tmp_path):
    with pytest.raises(OLMoConfigurationError, match="eos_token_id"):
        ComposableTextReplayConfig(
            source=_ListSourceConfig(instances=[]), work_dir=str(tmp_path), document_masking=True
        ).build()
