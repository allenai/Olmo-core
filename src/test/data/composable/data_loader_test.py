from itertools import islice
from pathlib import Path

import pytest

from olmo_core.data.collator import DataCollator
from olmo_core.data.composable.concat_and_chunk_instance_source import (
    ConcatAndChunkInstanceSource,
)
from olmo_core.data.composable.data_loader import (
    ComposableDataLoader,
    ShuffleStrategy,
)
from olmo_core.data.composable.token_source import InMemoryTokenSource
from olmo_core.data.tokenizer import TokenizerConfig


@pytest.mark.parametrize("num_workers", [0, 2])
def test_data_loader(tmp_path: Path, num_workers: int):
    tokens1 = InMemoryTokenSource(tokens=list(range(64)), work_dir=tmp_path)
    tokens2 = InMemoryTokenSource(tokens=list(range(64, 128)), work_dir=tmp_path)
    instances = ConcatAndChunkInstanceSource(tokens1, tokens2, sequence_length=8, work_dir=tmp_path)
    tokenizer = TokenizerConfig(vocab_size=32000, eos_token_id=0, pad_token_id=-1)
    data_loader = ComposableDataLoader(
        instances,
        collator=DataCollator(pad_token_id=tokenizer.pad_token_id),
        tokenizer=tokenizer,
        work_dir=tmp_path,
        global_batch_size=16,
        num_workers=num_workers,
    )
    assert data_loader.sequence_length == 8
    assert data_loader.max_sequence_length == 8
    assert data_loader.total_instances == 16
    assert data_loader.total_tokens == 128
    assert data_loader.total_batches == len(data_loader) == 8

    batch = data_loader.get_mock_batch()
    assert batch["input_ids"].shape == (2, 8)

    data_loader.reshuffle()
    all_tokens = set()
    n_batches = 0
    for batch in data_loader:
        assert batch["input_ids"].shape == (2, 8)
        for seq in batch["input_ids"]:
            for token in seq.tolist():
                all_tokens.add(token)
        n_batches += 1
    assert n_batches == len(data_loader) == data_loader.batches_processed
    assert all_tokens == set(range(128))

    # Iterating again before 'reset()' is called shouldn't yield any batches.
    assert len(list(data_loader)) == 0

    data_loader.reset()


def test_data_loader_save_and_restore(tmp_path: Path):
    tokens1 = InMemoryTokenSource(tokens=list(range(64)), work_dir=tmp_path)
    tokens2 = InMemoryTokenSource(tokens=list(range(64, 128)), work_dir=tmp_path)
    instances = ConcatAndChunkInstanceSource(tokens1, tokens2, sequence_length=8, work_dir=tmp_path)
    tokenizer = TokenizerConfig(vocab_size=32000, eos_token_id=0, pad_token_id=-1)

    def build_data_loader():
        return ComposableDataLoader(
            instances,
            collator=DataCollator(pad_token_id=tokenizer.pad_token_id),
            tokenizer=tokenizer,
            work_dir=tmp_path,
            global_batch_size=16,
            num_workers=0,
            num_threads=0,
        )

    data_loader = build_data_loader()
    assert data_loader.sequence_length == 8
    assert data_loader.max_sequence_length == 8
    assert data_loader.total_instances == 16
    assert data_loader.total_tokens == 128
    assert data_loader.total_batches == len(data_loader) == 8

    batch = data_loader.get_mock_batch()
    assert batch["input_ids"].shape == (2, 8)

    all_tokens = set()
    n_batches = 0

    # Go through part of the epoch...
    data_loader.reshuffle(1)
    for batch in islice(data_loader, 4):
        for seq in batch["input_ids"]:
            for token in seq.tolist():
                all_tokens.add(token)
        n_batches += 1

    # Then save and restore...
    state_dict = data_loader.state_dict()
    data_loader = build_data_loader()
    data_loader.load_state_dict(state_dict)

    # And complete the epoch.
    data_loader.reshuffle(1)
    for batch in data_loader:
        for seq in batch["input_ids"]:
            for token in seq.tolist():
                assert token not in all_tokens
                all_tokens.add(token)
        n_batches += 1

    assert n_batches == len(data_loader) == data_loader.batches_processed
    assert all_tokens == set(range(128))
    data_loader.reset()

    # Now create a new data loader with another instance source. We should be able restore
    # from the original data loader.
    state_dict = data_loader.state_dict()

    instances2 = ConcatAndChunkInstanceSource(
        InMemoryTokenSource(tokens=list(range(128, 192)), work_dir=tmp_path),
        sequence_length=8,
        work_dir=tmp_path,
    )
    data_loader = ComposableDataLoader(
        instances,
        instances2,
        sources_per_epoch=1,
        collator=DataCollator(pad_token_id=tokenizer.pad_token_id),
        tokenizer=tokenizer,
        work_dir=tmp_path,
        global_batch_size=16,
        num_workers=0,
        num_threads=0,
    )
    assert data_loader.batches_in_epoch(1) == 8
    assert data_loader.batches_in_epoch(2) == 4

    data_loader.load_state_dict(state_dict)

    all_tokens = set()
    n_batches = 0
    data_loader.reshuffle(2)
    for batch in data_loader:
        for seq in batch["input_ids"]:
            for token in seq.tolist():
                all_tokens.add(token)
        n_batches += 1

    assert n_batches == len(data_loader) == data_loader.batches_processed
    assert all_tokens == set(range(128, 192))
    data_loader.reset()


def test_interleaved_source_draws_from_the_whole_larger_source(tmp_path: Path):
    tokenizer = TokenizerConfig(vocab_size=32000, eos_token_id=0, pad_token_id=-1)
    large = ConcatAndChunkInstanceSource(
        InMemoryTokenSource(tokens=list(range(640)), work_dir=tmp_path),
        sequence_length=8,
        work_dir=tmp_path,
    )
    small = ConcatAndChunkInstanceSource(
        InMemoryTokenSource(tokens=list(range(640, 704)), work_dir=tmp_path),
        sequence_length=8,
        work_dir=tmp_path,
    )
    data_loader = ComposableDataLoader(
        large,
        small,
        collator=DataCollator(pad_token_id=tokenizer.pad_token_id),
        tokenizer=tokenizer,
        work_dir=tmp_path,
        global_batch_size=16,
        seed=0,
        shuffle_strategy=ShuffleStrategy.interleaved_source,
        num_workers=0,
        num_threads=0,
    )
    # Each source contributes as many instances as the smallest one has.
    assert data_loader.total_instances == 16

    seen_large = set()
    for epoch in range(1, 4):
        data_loader.reshuffle(epoch=epoch)
        indices = data_loader.get_global_indices()
        large_indices = set(indices[0::2].tolist())
        assert len(large_indices) == 8
        assert large_indices <= set(range(80))
        assert set(indices[1::2].tolist()) == set(range(80, 88))
        seen_large |= large_indices

    # The larger source contributes a fresh random subset each epoch, not only its first 8
    # instances every epoch.
    assert len(seen_large) > 8
