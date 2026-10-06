"""EMO document pools follow the same document boundaries as attention when those come from metadata."""

import gzip
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

import olmo_core.ops.moe as ops
from olmo_core.data import (
    DataCollator,
    LongDocStrategy,
    NumpyPackedFSLDataset,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.emo import EmoRouterConfig

from ...train.train_module.transformer.ddp_train_module_test import _tiny_model_config

EOS, PAD, IM_START, IM_END, NL = 0, 1, 2, 3, 4
# A multi-turn row with an EOS after its first assistant turn, and a row that stops on a tool call.
MULTI_TURN = [
    IM_START,
    10,
    IM_END,
    NL,
    IM_START,
    11,
    EOS,
    IM_START,
    12,
    IM_END,
    NL,
    IM_START,
    13,
    EOS,
]
TOOL_CALL = [IM_START, 20, IM_END, NL, IM_START, 21, IM_END, NL]
SINGLE_TURN = [IM_START, 30, IM_END, NL, IM_START, 31, EOS]
ROWS = [MULTI_TURN, TOOL_CALL, SINGLE_TURN]


def test_segment_ids_from_doc_lens() -> None:
    doc_lens = torch.tensor([[3, 2, 3, 0], [8, 0, 0, 0], [1, 0, 7, 0]])
    torch.testing.assert_close(
        ops.segment_ids_from_doc_lens(doc_lens, 8),
        torch.tensor(
            [
                [0, 0, 0, 1, 1, 2, 2, 2],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
    )
    with pytest.raises(ValueError, match="sum to the sequence length"):
        ops.segment_ids_from_doc_lens(torch.tensor([[3, 2]]), 8)


def _batch(tmp_path: Path, use_array_if_local) -> dict:
    """One collated batch from the rows, packed the way open-instruct's SFT path packs them."""
    data = [t for row in ROWS for t in row]
    np.array(data, dtype=np.uint8).tofile(tmp_path / "token_ids_part_0000.npy")
    with gzip.open(tmp_path / "token_ids_part_0000.csv.gz", "wt") as f:
        start = 0
        for row in ROWS:
            f.write(f"{start},{start + len(row)}\n")
            start += len(row)
    ds = NumpyPackedFSLDatasetConfig(
        tokenizer=TokenizerConfig(vocab_size=128, eos_token_id=EOS, pad_token_id=PAD),
        work_dir=str(tmp_path / f"work-{use_array_if_local}"),
        paths=[str(tmp_path / "token_ids_part_*.npy")],
        expand_glob=True,
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=32,
        use_array_if_local=use_array_if_local,
    ).build()
    assert isinstance(ds, NumpyPackedFSLDataset)
    ds.prepare()
    return DataCollator(pad_token_id=PAD)([ds[i] for i in range(len(ds))])


def _segment_ids(batch: dict, segment_ids_from: str, training: bool = True, with_doc_lens=True):
    config = _tiny_model_config()
    config.block.routed_experts_router.emo = EmoRouterConfig(
        eos_token_id=EOS,
        min_document_expert_pool=2,
        max_document_expert_pool=4,
        segment_ids_from=segment_ids_from,  # type: ignore[arg-type]
    )
    model = config.build(init_device="cpu")
    model.train(training)
    kwargs = (
        {"doc_lens": batch["doc_lens"], "max_doc_lens": batch["max_doc_lens"]}
        if with_doc_lens
        else {}
    )
    per_block_kwargs = model._prepare_inputs(batch["input_ids"], **kwargs)[3]
    segment_ids = [block_kwargs["segment_ids"] for block_kwargs in per_block_kwargs.values()]
    assert len(segment_ids) == config.n_layers
    assert all(torch.equal(s, segment_ids[0]) for s in segment_ids)
    return segment_ids[0]


def _rows_by_segment(input_ids: torch.Tensor, segment_ids: torch.Tensor) -> List[List[int]]:
    out: List[List[int]] = []
    for ids, segs in zip(input_ids.tolist(), segment_ids.tolist()):
        for s in sorted(set(segs)):
            tokens = [t for t, g in zip(ids, segs) if g == s]
            if set(tokens) != {PAD}:
                out.append(tokens)
    return sorted(out)


def _doc_lens_segments(batch: dict) -> torch.Tensor:
    ids = []
    for lens in batch["doc_lens"].tolist():
        ids.append([i for i, n in enumerate(lens) for _ in range(n)])
    return torch.tensor(ids)


def test_emo_segments_follow_metadata_doc_lens_when_configured(tmp_path: Path) -> None:
    batch = _batch(tmp_path, use_array_if_local=False)
    segment_ids = _segment_ids(batch, "doc_lens")
    torch.testing.assert_close(segment_ids, _doc_lens_segments(batch))
    # Each conversation is exactly one EMO document, like its attention document.
    assert _rows_by_segment(batch["input_ids"], segment_ids) == sorted(ROWS)


@pytest.mark.parametrize("use_array_if_local", [None, False])
def test_emo_segments_from_eos_by_default(tmp_path: Path, use_array_if_local) -> None:
    batch = _batch(tmp_path, use_array_if_local=use_array_if_local)
    segment_ids = _segment_ids(batch, "eos")
    torch.testing.assert_close(segment_ids, ops.segment_ids_from_eos(batch["input_ids"], EOS))
    assert _rows_by_segment(batch["input_ids"], segment_ids) != sorted(ROWS)


def test_emo_doc_lens_segments_require_doc_lens_in_training(tmp_path: Path) -> None:
    batch = _batch(tmp_path, use_array_if_local=False)
    with pytest.raises(OLMoConfigurationError, match="doc_lens"):
        _segment_ids(batch, "doc_lens", with_doc_lens=False)
    # Evaluation batches may have no doc_lens; EOS is then the only boundary information.
    torch.testing.assert_close(
        _segment_ids(batch, "doc_lens", training=False, with_doc_lens=False),
        ops.segment_ids_from_eos(batch["input_ids"], EOS),
    )


def test_emo_segment_source_is_validated() -> None:
    config = EmoRouterConfig(
        eos_token_id=EOS,
        min_document_expert_pool=2,
        max_document_expert_pool=4,
        segment_ids_from="bos",  # type: ignore[arg-type]
    )
    with pytest.raises(OLMoConfigurationError, match="segment_ids_from"):
        config.validate_for_router(num_experts=4, top_k=2)
