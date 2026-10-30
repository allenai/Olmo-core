"""
The replay adapter keeps per-occurrence ``metadata`` aligned with a weighted source mixture that
drops zero-allocation paths (the shared numpy dataset builder is not involved).
"""

from types import SimpleNamespace

import numpy as np

from olmo_core.data.multimodal.pretraining_replay import (
    _align_metadata_with_selected_paths,
)


def _config(metadata, tokens_per_path):
    sources = [
        SimpleNamespace(
            path_tokens=[
                SimpleNamespace(path=f"/data/{i}.npy", tokens=tokens)
                for i, tokens in enumerate(tokens_per_path)
            ]
        )
    ]
    mixture = SimpleNamespace(sources=sources)
    return SimpleNamespace(
        metadata=metadata,
        source_mixture_config=SimpleNamespace(build=lambda **kwargs: mixture),
        get_dtype=lambda: np.uint32,
        sequence_length=8,
    )


def test_metadata_follows_the_zero_token_filter():
    metadata = [{"source": "a"}, {"source": "b"}, {"source": "c"}]
    config = _config(metadata, [10, 0, 5])
    _align_metadata_with_selected_paths(config)
    assert config.metadata == [{"source": "a"}, {"source": "c"}]


def test_metadata_is_untouched_when_nothing_is_filtered_or_already_aligned():
    metadata = [{"source": "a"}, {"source": "b"}]
    config = _config(metadata, [10, 5])
    _align_metadata_with_selected_paths(config)
    assert config.metadata == metadata

    already_filtered = [{"source": "a"}, {"source": "c"}]
    config = _config(already_filtered, [10, 0, 5])
    _align_metadata_with_selected_paths(config)
    assert config.metadata == already_filtered


def test_dict_metadata_and_plain_datasets_are_left_alone():
    config = _config({"source": "all"}, [10, 0])
    _align_metadata_with_selected_paths(config)
    assert config.metadata == {"source": "all"}

    config = SimpleNamespace(metadata=[{"a": 1}, {"b": 2}], source_mixture_config=None)
    _align_metadata_with_selected_paths(config)
    assert config.metadata == [{"a": 1}, {"b": 2}]
