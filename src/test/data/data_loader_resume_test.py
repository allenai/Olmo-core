from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from olmo_core.data import DataCollator, NumpyFSLDataLoader, NumpyFSLDataset
from olmo_core.data.composable import (
    ComposableDataLoader,
    ConcatAndChunkInstanceSource,
    InMemoryTokenSource,
)
from olmo_core.data.tokenizer import TokenizerConfig


@pytest.fixture(params=["numpy", "composable"])
def build_loader(request: pytest.FixtureRequest, tmp_path: Path):
    tokens = list(range(128))
    if request.param == "numpy":
        path = tmp_path / "tokens.npy"
        np.asarray(tokens, dtype=np.uint16).tofile(path)

    def build(batch_size, *, seed=42, shuffle=False, sequence_length=4):
        kwargs = dict(
            global_batch_size=batch_size,
            collator=DataCollator(pad_token_id=-1),
            seed=seed,
            shuffle=shuffle,
            num_threads=0,
            num_workers=0,
            work_dir=tmp_path,
        )
        if request.param == "numpy":
            return NumpyFSLDataLoader(
                NumpyFSLDataset(
                    path,
                    sequence_length=sequence_length,
                    max_target_sequence_length=8,
                    pad_token_id=-1,
                    eos_token_id=-1,
                    vocab_size=32000,
                ),
                chunk_size=8 // sequence_length,
                **kwargs,
            )
        return ComposableDataLoader(
            ConcatAndChunkInstanceSource(
                InMemoryTokenSource(tokens, work_dir=tmp_path),
                sequence_length=sequence_length,
                max_sequence_length=8,
                work_dir=tmp_path,
            ),
            tokenizer=TokenizerConfig(vocab_size=32000, eos_token_id=-1, pad_token_id=-1),
            **kwargs,
        )

    return build


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_rejects_misaligned_batch_size_without_mutation(build_loader, shuffle: bool):
    original = build_loader(16, seed=21, shuffle=shuffle)
    original.reshuffle(2)
    list(islice(original, 1))
    checkpoint = original.state_dict()
    assert checkpoint["tokens_processed"] == 16

    resumed = build_loader(32, seed=42, shuffle=shuffle)
    resumed.reshuffle(1)
    list(islice(resumed, 1))
    before = resumed.state_dict()

    with pytest.raises(RuntimeError, match="16.*32.*duplicating or losing data"):
        resumed.load_state_dict(checkpoint)

    assert resumed.state_dict() == before


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("sequence_length", [4, 8])
@pytest.mark.parametrize(
    "consumed_batches, new_batch_size",
    [(0, 32), (2, 32), (1, 16), (1, 8), (6, 32), (8, 32)],
)
def test_resume_preserves_aligned_token_stream(
    build_loader, shuffle: bool, sequence_length: int, consumed_batches: int, new_batch_size: int
):
    original = build_loader(16, shuffle=shuffle)
    original.reshuffle(2)
    list(islice(original, consumed_batches))
    checkpoint = original.state_dict()
    expected = [int(token) for batch in original for token in batch["input_ids"].flatten()]

    resumed = build_loader(new_batch_size, shuffle=shuffle, sequence_length=sequence_length)
    resumed.load_state_dict(checkpoint)
    assert resumed.tokens_processed == consumed_batches * 16
    assert resumed.batches_processed == consumed_batches * 16 // new_batch_size
    assert resumed.epoch == 2
    resumed.reshuffle(2)
    actual = [int(token) for batch in resumed for token in batch["input_ids"].flatten()]

    assert actual == expected
    assert resumed.tokens_processed == 128
    assert resumed.batches_processed == 128 // new_batch_size
