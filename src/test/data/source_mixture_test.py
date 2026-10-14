import logging
import math
from itertools import chain
from pathlib import Path

import numpy as np
import pytest

from olmo_core.data import NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDataset,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
    SourceMixtureOutcome,
    SourcePathTokens,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.exceptions import OLMoConfigurationError

from .utils import mk_mmaps


def test_source_mixture_config(tmp_path: Path, caplog, capsys):
    source_paths = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=2, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }

    source_configs = SourceMixtureList(
        [
            SourceMixtureConfig(
                source_name="1",
                target_ratio=0.33333,
                paths=[str(i[0]) for i in source_paths["1"]],
            ),
            SourceMixtureConfig(
                source_name="2", target_ratio=0.33333, paths=[str(i[0]) for i in source_paths["2"]]
            ),
            SourceMixtureConfig(
                source_name="3",
                target_ratio=0.33333,
                paths=[str(i[0]) for i in source_paths["3"]],
            ),
        ]
    )

    max_tokens = 5_000_000
    sequence_length = 1024
    global_batch_size = 1024 * 32

    config = SourceMixtureDatasetConfig(
        source_list=source_configs,
        requested_tokens=max_tokens,
        quiet=True,
        render_tables=True,
        global_batch_size=global_batch_size,
    )

    # NOTE: We need to disable capsys so we can override log capture as
    # we want to see the rendered tables in the case
    with capsys.disabled(), caplog.at_level(logging.DEBUG):
        config.validate()
        mixture = config.build(npdtype=np.uint32, sequence_length=sequence_length)
        assert isinstance(mixture, SourceMixtureDataset)

        requested_instances = math.ceil(max_tokens / global_batch_size) * int(
            global_batch_size / sequence_length
        )
        assert (
            sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])
            == requested_instances
        ), f"Expected {requested_instances} instances, but got {sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])}"
        #  print(caplog.text)  # uncomment if you want to see the table


def test_source_mixture_config_validation():
    with pytest.raises(OLMoConfigurationError):
        SourceMixtureConfig(
            source_name="source1", target_ratio=1.2, paths=["/path/to/source1"]
        ).validate()

    with pytest.raises(OLMoConfigurationError):
        SourceMixtureConfig(source_name="source1", target_ratio=0.5, paths=[]).validate()

    config = SourceMixtureConfig(
        source_name="source1", target_ratio=0.5, paths=["/path/to/source1"]
    )
    config.validate()


def test_dataset_mixture_config_validation():
    source_configs = SourceMixtureList(
        [
            SourceMixtureConfig(
                source_name="source1", target_ratio=0.5, paths=["/path/to/source1"]
            ),
            SourceMixtureConfig(
                source_name="source2", target_ratio=0.5, paths=["/path/to/source2"]
            ),
        ]
    )

    config = SourceMixtureDatasetConfig(
        requested_tokens=1000,
        source_list=source_configs,
        quiet=True,
        render_tables=False,
        global_batch_size=1024 * 32,
    )
    config.validate()

    source_configs_invalid = SourceMixtureList(
        [
            SourceMixtureConfig(
                source_name="source1", target_ratio=0.7, paths=["/path/to/source1"]
            ),
            SourceMixtureConfig(
                source_name="source2", target_ratio=0.5, paths=["/path/to/source2"]
            ),
        ]
    )

    config_invalid = SourceMixtureDatasetConfig(
        requested_tokens=1000,
        source_list=source_configs_invalid,
        quiet=True,
        render_tables=False,
        global_batch_size=1024 * 32,
    )

    with pytest.raises(OLMoConfigurationError):
        config_invalid.validate()


def test_dataset_mixture_build(tmp_path: Path):
    source_paths = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=2, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }

    source_configs = SourceMixtureList(
        [
            SourceMixtureConfig(
                source_name="1",
                target_ratio=0.33,
                paths=[str(i[0]) for i in source_paths["1"]],
            ),
            SourceMixtureConfig(
                source_name="2", target_ratio=0.33, paths=[str(i[0]) for i in source_paths["2"]]
            ),
            SourceMixtureConfig(
                source_name="3",
                target_ratio=0.34,
                paths=[str(i[0]) for i in source_paths["3"]],
            ),
        ]
    )

    max_tokens = 5_000_000
    sequence_length = 1024
    global_batch_size = sequence_length * 32

    config = SourceMixtureDatasetConfig(
        requested_tokens=max_tokens,
        source_list=source_configs,
        quiet=True,
        render_tables=False,
        global_batch_size=global_batch_size,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=sequence_length)
    assert isinstance(mixture, SourceMixtureDataset)

    requested_instances = math.ceil(max_tokens / global_batch_size) * int(
        global_batch_size / sequence_length
    )
    assert (
        sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])
        == requested_instances
    ), f"Expected {requested_instances} instances, but got {sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])}"


def test_dataset_mixture_build_insufficient_source_data(tmp_path: Path):
    source_paths = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=1, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }
    source_configs = SourceMixtureList(
        [
            SourceMixtureConfig(
                source_name="1",
                target_ratio=0.5,
                paths=[str(i[0]) for i in source_paths["1"]],
            ),
            SourceMixtureConfig(
                source_name="2", target_ratio=0.25, paths=[str(i[0]) for i in source_paths["2"]]
            ),
            SourceMixtureConfig(
                source_name="3",
                target_ratio=0.25,
                paths=[str(i[0]) for i in source_paths["3"]],
            ),
        ]
    )

    max_tokens = 5_000_000

    config = SourceMixtureDatasetConfig(
        requested_tokens=max_tokens,
        source_list=source_configs,
        quiet=True,
        render_tables=False,
        global_batch_size=1024 * 32,
    )

    # Should raise exception because the target ratio for source 1 @50% (2.5M) is infeasible without repetition (default max_repetition_ratio=1)
    with pytest.raises(OLMoConfigurationError):
        config.build(npdtype=np.uint32, sequence_length=1024)


def test_dataset_mixture_build_with_repetition(tmp_path: Path):
    """
    Test building a dataset with repetition of a source.

    Source 1 has a target ratio of 90% and a max repetition ratio of 4.0, so it should be possible to meet the target of 3600 tokens with 1 file of 1000 tokens repeated 4 times.
    """
    source_paths = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=1, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }

    source_configs = SourceMixtureList(
        [
            SourceMixtureConfig(
                source_name="1",
                target_ratio=0.5,
                max_repetition_ratio=3.0,  # Allow 3x repetition of source1 so that we can meet the target of 2.5M
                paths=[str(i[0]) for i in source_paths["1"]],
            ),
            SourceMixtureConfig(
                source_name="2", target_ratio=0.25, paths=[str(i[0]) for i in source_paths["2"]]
            ),
            SourceMixtureConfig(
                source_name="3",
                target_ratio=0.25,
                paths=[str(i[0]) for i in source_paths["3"]],
            ),
        ]
    )

    max_tokens = 5_000_000
    sequence_length = 1024
    global_batch_size = sequence_length * 32

    config = SourceMixtureDatasetConfig(
        requested_tokens=max_tokens,
        source_list=source_configs,
        quiet=True,
        render_tables=False,
        global_batch_size=global_batch_size,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=sequence_length)

    sources = [source for source in mixture.sources]
    all_paths = []
    for source in sources:
        all_paths.extend([item for item in source.path_tokens])

    total_tokens = sum([item.tokens for item in all_paths])
    assert isinstance(mixture, SourceMixtureDataset)

    requested_instances = math.ceil(max_tokens / global_batch_size) * int(
        global_batch_size / sequence_length
    )
    assert (
        sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])
        == requested_instances
    ), f"Expected {requested_instances} instances, but got {sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])}"

    assert total_tokens == 5013504


def test_dataset_mixture_build_insufficient_source_max_fraction(tmp_path: Path):
    source_paths = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=1, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }
    source_configs = SourceMixtureList(
        [
            SourceMixtureConfig(
                source_name="1",
                target_ratio=0.25,
                paths=[str(i[0]) for i in source_paths["1"]],
                max_source_fraction=0.10,  # Allow only 10% of source1 to be used (population is 1M tokens)
            ),
            SourceMixtureConfig(
                source_name="2",
                target_ratio=0.25,
                paths=[str(i[0]) for i in source_paths["2"]],
            ),
            SourceMixtureConfig(
                source_name="3",
                target_ratio=0.5,
                paths=[str(i[0]) for i in source_paths["3"]],
            ),
        ]
    )

    # 5 source files * 1_000_000 tokens per file
    max_tokens = len(list(chain(*source_paths.values()))) * 1_000_000

    config = SourceMixtureDatasetConfig(
        requested_tokens=max_tokens,
        source_list=source_configs,
        quiet=True,
        render_tables=False,
        global_batch_size=1024 * 32,
    )

    # Should raise exception because the target ratio for source 1 is infeasible because
    # we limit usage to 10% of the source
    with pytest.raises(OLMoConfigurationError):
        config.build(npdtype=np.uint32, sequence_length=1024)


@pytest.mark.parametrize("path_sizes", [[10, 10, 80], [10, 10, 100]])
def test_dataset_mixture_fills_batch_with_uneven_path_capacity(tmp_path: Path, path_sizes):
    paths = []
    for index, size in enumerate(path_sizes):
        path = tmp_path / f"source_{index}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        paths.append(str(path))

    config = SourceMixtureDatasetConfig(
        requested_tokens=30,
        source_list=SourceMixtureList(
            [SourceMixtureConfig(source_name="source", target_ratio=1.0, paths=paths)]
        ),
        global_batch_size=100,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    selected_path_tokens = [item.tokens for item in mixture.sources[0].path_tokens]
    assert selected_path_tokens == [10, 10, 80]
    assert all(tokens <= capacity for tokens, capacity in zip(selected_path_tokens, path_sizes))
    assert sum(selected_path_tokens) == 100
    assert sum(tokens // 10 for tokens in mixture.to_index().values()) == 10


@pytest.mark.parametrize(
    ("requested_tokens", "global_batch_size", "sequence_length", "path_sizes"),
    [(1188, 1000, 10, [1200]), (10, 10, 10, [9, 9])],
)
def test_dataset_mixture_rejects_insufficient_instance_capacity(
    tmp_path: Path, requested_tokens, global_batch_size, sequence_length, path_sizes
):
    paths = []
    for index, size in enumerate(path_sizes):
        path = tmp_path / f"source_{index}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        paths.append(str(path))
    config = SourceMixtureDatasetConfig(
        requested_tokens=requested_tokens,
        source_list=SourceMixtureList(
            [SourceMixtureConfig(source_name="source", target_ratio=1.0, paths=paths)]
        ),
        global_batch_size=global_batch_size,
        quiet=True,
        render_tables=False,
    )

    with pytest.raises(OLMoConfigurationError, match="complete requested training steps"):
        config.build(npdtype=np.uint32, sequence_length=sequence_length)


def test_dataset_mixture_preserves_balanced_sources_when_rounding_to_batch(tmp_path: Path):
    sources = []
    for name in ("left", "right"):
        path = tmp_path / f"{name}.bin"
        np.zeros(120, dtype=np.uint32).tofile(path)
        sources.append(SourceMixtureConfig(source_name=name, target_ratio=0.5, paths=[str(path)]))
    config = SourceMixtureDatasetConfig(
        requested_tokens=60,
        source_list=SourceMixtureList(sources),
        global_batch_size=100,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    token_counts = {
        outcome.name: sum(item.tokens for item in outcome.path_tokens)
        for outcome in mixture.sources
    }
    assert token_counts == {"left": 50, "right": 50}


@pytest.mark.parametrize("sizes", [(510, 500), (500, 510)])
def test_dataset_mixture_breaks_equal_source_remainders_by_capacity(tmp_path: Path, sizes):
    sources = []
    for idx, size in enumerate(sizes):
        path = tmp_path / f"source_{idx}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        sources.append(
            SourceMixtureConfig(source_name=str(idx), target_ratio=0.5, paths=[str(path)])
        )
    config = SourceMixtureDatasetConfig(
        requested_tokens=1000,
        source_list=SourceMixtureList(sources),
        global_batch_size=1010,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    assert [sum(item.tokens for item in source.path_tokens) for source in mixture.sources] == list(
        sizes
    )


@pytest.mark.parametrize("reverse_sources", [False, True])
def test_dataset_mixture_preserves_decimal_hamilton_ties(tmp_path: Path, reverse_sources: bool):
    sources = []
    for index, (size, ratio) in enumerate(((270, 0.55), (230, 0.45))):
        path = tmp_path / f"source_{index}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        sources.append(
            SourceMixtureConfig(source_name=str(index), target_ratio=ratio, paths=[str(path)])
        )
    if reverse_sources:
        sources.reverse()
    config = SourceMixtureDatasetConfig(
        requested_tokens=500,
        source_list=SourceMixtureList(sources),
        global_batch_size=500,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    assert [sum(item.tokens for item in source.path_tokens) for source in mixture.sources] == (
        [230, 270] if reverse_sources else [270, 230]
    )


@pytest.mark.parametrize(
    ("path_sizes", "expected_source"),
    [([[10], [9, 9]], 0), ([[9, 9], [10]], 1)],
)
def test_dataset_mixture_breaks_equal_remainders_by_whole_path_capacity(
    tmp_path: Path, path_sizes, expected_source
):
    sources = []
    for source_idx, source_path_sizes in enumerate(path_sizes):
        paths = []
        for path_idx, size in enumerate(source_path_sizes):
            path = tmp_path / f"source_{source_idx}_{path_idx}.bin"
            np.zeros(size, dtype=np.uint32).tofile(path)
            paths.append(str(path))
        sources.append(
            SourceMixtureConfig(source_name=str(source_idx), target_ratio=0.5, paths=paths)
        )

    config = SourceMixtureDatasetConfig(
        requested_tokens=10,
        source_list=SourceMixtureList(sources),
        global_batch_size=10,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    selected_tokens = [
        sum(item.tokens for item in source.path_tokens) for source in mixture.sources
    ]
    assert selected_tokens[expected_source] == 10
    assert sum(selected_tokens) == 10


def test_dataset_mixture_counts_allowed_repetitions_in_whole_path_capacity(tmp_path: Path):
    repeated_path = tmp_path / "repeated.bin"
    bounded_path = tmp_path / "bounded.bin"
    np.zeros(15, dtype=np.uint32).tofile(repeated_path)
    np.zeros(10, dtype=np.uint32).tofile(bounded_path)
    config = SourceMixtureDatasetConfig(
        requested_tokens=30,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="repeated",
                    target_ratio=0.5,
                    paths=[str(repeated_path)],
                    max_repetition_ratio=2.0,
                ),
                SourceMixtureConfig(
                    source_name="bounded", target_ratio=0.5, paths=[str(bounded_path)]
                ),
            ]
        ),
        global_batch_size=30,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    selected_tokens = [
        sum(item.tokens for item in source.path_tokens) for source in mixture.sources
    ]
    assert selected_tokens == [20, 10]


def test_dataset_mixture_rejects_zero_sequence_capacity_before_materializing_repetitions(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "source.bin"
    np.zeros(9, dtype=np.uint32).tofile(path)
    config = SourceMixtureDatasetConfig(
        requested_tokens=10,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=[str(path)],
                    max_repetition_ratio=1_000_000,
                )
            ]
        ),
        global_batch_size=10,
        quiet=True,
        render_tables=False,
    )

    def unexpected_materialization(**kwargs):
        pytest.fail("source paths were materialized despite zero complete-sequence capacity")

    monkeypatch.setattr(config, "get_paths_and_tokens_for_source", unexpected_materialization)

    with pytest.raises(OLMoConfigurationError, match="provide at most 0 instances"):
        config.build(npdtype=np.uint32, sequence_length=10)


@pytest.mark.parametrize("processes", [1, 2])
def test_dataset_mixture_preserves_source_ratios_when_paths_hit_capacity(
    tmp_path: Path, caplog, processes
):
    caplog.set_level(logging.INFO, logger="olmo_core.data.source_mixture")
    source_paths = []
    for name, size in (
        ("large.bin", 100),
        ("small_a.bin", 10),
        ("small_b.bin", 10),
        ("other.bin", 100),
    ):
        path = tmp_path / name
        np.zeros(size, dtype=np.uint32).tofile(path)
        source_paths.append(str(path))

    config = SourceMixtureDatasetConfig(
        requested_tokens=30,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(source_name="left", target_ratio=0.5, paths=source_paths[:1]),
                SourceMixtureConfig(source_name="right", target_ratio=0.5, paths=source_paths[1:]),
            ]
        ),
        global_batch_size=200,
        processes=processes,
        quiet=True,
        render_tables=True,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    selected_by_source = {
        outcome.name: [item.tokens for item in outcome.path_tokens] for outcome in mixture.sources
    }
    assert selected_by_source == {"left": [100], "right": [10, 10, 80]}
    assert caplog.text.count("0.5000") >= 2


def test_dataset_mixture_rejects_batch_padding_over_source_fraction_cap(tmp_path: Path):
    paths = []
    for index, size in enumerate((10, 10, 100)):
        path = tmp_path / f"source_{index}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        paths.append(str(path))

    config = SourceMixtureDatasetConfig(
        requested_tokens=30,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=paths,
                    max_source_fraction=0.6,
                )
            ]
        ),
        global_batch_size=100,
        quiet=True,
        render_tables=False,
    )

    with pytest.raises(OLMoConfigurationError, match="Insufficient tokens for source"):
        config.build(npdtype=np.uint32, sequence_length=10)


def test_dataset_mixture_materializes_repetitions_for_batch_padding(tmp_path: Path):
    path = tmp_path / "source.bin"
    np.zeros(100, dtype=np.uint32).tofile(path)

    config = SourceMixtureDatasetConfig(
        requested_tokens=90,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=[str(path)],
                    max_repetition_ratio=2.0,
                )
            ]
        ),
        global_batch_size=200,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    path_tokens = mixture.sources[0].path_tokens
    assert [item.tokens for item in path_tokens] == [100, 100]
    assert sum(item.tokens for item in path_tokens) == 200


def test_dataset_mixture_does_not_round_fractional_repetition_capacity_up(tmp_path: Path):
    paths = []
    for index, size in enumerate((9, 91)):
        path = tmp_path / f"fragment_{index}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        paths.append(str(path))

    config = SourceMixtureDatasetConfig(
        requested_tokens=100,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=paths,
                    max_repetition_ratio=1.01,
                )
            ]
        ),
        global_batch_size=100,
        quiet=True,
        render_tables=False,
    )

    with pytest.raises(OLMoConfigurationError, match="Insufficient source capacity"):
        config.build(npdtype=np.uint32, sequence_length=10)


def test_dataset_mixture_accepts_exact_decimal_repetition_capacity(tmp_path: Path):
    path = tmp_path / "source.bin"
    np.zeros(1000, dtype=np.uint32).tofile(path)
    config = SourceMixtureDatasetConfig(
        requested_tokens=1130,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=[str(path)],
                    max_repetition_ratio=1.13,
                )
            ]
        ),
        global_batch_size=1130,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    assert sum(item.tokens for item in mixture.sources[0].path_tokens) == 1130


def test_dataset_mixture_rejects_just_below_decimal_repetition_capacity(tmp_path: Path):
    path = tmp_path / "source.bin"
    np.zeros(1000, dtype=np.uint32).tofile(path)
    config = SourceMixtureDatasetConfig(
        requested_tokens=1130,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=[str(path)],
                    max_repetition_ratio=math.nextafter(1.13, 0.0),
                )
            ]
        ),
        global_batch_size=1130,
        quiet=True,
        render_tables=False,
    )

    with pytest.raises(OLMoConfigurationError, match="Insufficient"):
        config.build(npdtype=np.uint32, sequence_length=10)


def test_dataset_mixture_does_not_undercount_decimal_source_fraction_cap(tmp_path: Path):
    path = tmp_path / "source.bin"
    np.zeros(100, dtype=np.uint32).tofile(path)
    config = SourceMixtureDatasetConfig(
        requested_tokens=116,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=[str(path)],
                    max_repetition_ratio=2.0,
                    max_source_fraction=0.58,
                )
            ]
        ),
        global_batch_size=116,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=1)

    assert sum(item.tokens for item in mixture.sources[0].path_tokens) == 116


def test_dataset_mixture_caps_fractional_repetitions_per_path(tmp_path: Path):
    paths = []
    for index, size in enumerate((81, 19)):
        path = tmp_path / f"fragment_{index}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        paths.append(str(path))

    config = SourceMixtureDatasetConfig(
        requested_tokens=100,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=paths,
                    max_repetition_ratio=1.2,
                )
            ]
        ),
        global_batch_size=100,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    path_tokens = mixture.sources[0].path_tokens
    selected_by_path = {
        path: sum(item.tokens for item in path_tokens if item.path == path) for path in paths
    }
    assert selected_by_path == {paths[0]: 90, paths[1]: 10}


def test_dataset_mixture_apportions_fractional_repetitions_across_duplicate_paths(tmp_path: Path):
    path = tmp_path / "source.bin"
    np.zeros(10, dtype=np.uint32).tofile(path)
    paths = [str(path), str(path)]
    config = SourceMixtureDatasetConfig(
        requested_tokens=30,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=paths,
                    max_repetition_ratio=1.5,
                )
            ]
        ),
        global_batch_size=30,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    assert [item.tokens for item in mixture.sources[0].path_tokens] == [10, 10, 10, 0]


def test_dataset_mixture_rounds_fractional_capacity_after_aggregating_paths(tmp_path: Path):
    paths = []
    for index, size in enumerate((20, 30)):
        path = tmp_path / f"source_{index}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        paths.append(str(path))
    config = SourceMixtureDatasetConfig(
        requested_tokens=60,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=paths,
                    max_repetition_ratio=1.2,
                )
            ]
        ),
        global_batch_size=60,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    assert [item.tokens for item in mixture.sources[0].path_tokens] == [20, 30, 0, 10]


def test_dataset_mixture_materialization_respects_fractional_source_cap(tmp_path: Path):
    path = tmp_path / "source.bin"
    np.zeros(100, dtype=np.uint32).tofile(path)
    config = SourceMixtureDatasetConfig(
        requested_tokens=110,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=[str(path)],
                    max_repetition_ratio=2.0,
                    max_source_fraction=0.55,
                )
            ]
        ),
        global_batch_size=110,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    selected_tokens = sum(item.tokens for item in mixture.sources[0].path_tokens)
    assert selected_tokens == 110
    assert selected_tokens <= 100 * 0.55 * 2.0


@pytest.mark.parametrize(("max_repetition_ratio", "should_fill"), [(1.0, False), (2.0, True)])
def test_dataset_mixture_materializes_repetitions_for_fragmented_paths(
    tmp_path: Path, max_repetition_ratio: float, should_fill: bool
):
    paths = []
    for index, size in enumerate((9, 91)):
        path = tmp_path / f"fragment_{index}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        paths.append(str(path))

    config = SourceMixtureDatasetConfig(
        requested_tokens=100,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source",
                    target_ratio=1.0,
                    paths=paths,
                    max_repetition_ratio=max_repetition_ratio,
                )
            ]
        ),
        global_batch_size=100,
        quiet=True,
        render_tables=False,
    )

    if not should_fill:
        with pytest.raises(OLMoConfigurationError, match="Insufficient source capacity"):
            config.build(npdtype=np.uint32, sequence_length=10)
        return

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    assert sum(item.tokens for item in mixture.sources[0].path_tokens) == 100
    assert sum(item.tokens // 10 for item in mixture.sources[0].path_tokens) == 10


def test_dataset_mixture_rejects_fragmented_paths_without_repetition(tmp_path: Path):
    paths = []
    for index, size in enumerate((9, 9)):
        path = tmp_path / f"fragment_{index}.bin"
        np.zeros(size, dtype=np.uint32).tofile(path)
        paths.append(str(path))

    config = SourceMixtureDatasetConfig(
        requested_tokens=10,
        source_list=SourceMixtureList(
            [SourceMixtureConfig(source_name="source", target_ratio=1.0, paths=paths)]
        ),
        global_batch_size=10,
        quiet=True,
        render_tables=False,
    )

    with pytest.raises(OLMoConfigurationError, match="Insufficient source capacity"):
        config.build(npdtype=np.uint32, sequence_length=10)


@pytest.mark.parametrize("render_tables", [True, False])
def test_dataset_mixture_zero_assigned_empty_source_keeps_zero_path(
    tmp_path: Path, render_tables: bool
):
    empty_path = tmp_path / "empty.bin"
    empty_path.write_bytes(b"")
    full_path = tmp_path / "full.bin"
    np.zeros(100, dtype=np.uint32).tofile(full_path)

    config = SourceMixtureDatasetConfig(
        requested_tokens=10,
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(source_name="empty", target_ratio=0.2, paths=[str(empty_path)]),
                SourceMixtureConfig(source_name="full", target_ratio=0.8, paths=[str(full_path)]),
            ]
        ),
        global_batch_size=10,
        quiet=True,
        render_tables=render_tables,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    assert [[item.tokens for item in source.path_tokens] for source in mixture.sources] == [
        [0],
        [10],
    ]

    dataset = NumpyFSLDatasetConfig(
        source_mixture_config=config,
        sequence_length=10,
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
        dtype=NumpyDatasetDType.uint32,
        include_instance_metadata=False,
    ).build()
    dataset.prepare()
    assert len(dataset) == 1
    assert dataset[0]["input_ids"].shape == (10,)


@pytest.mark.parametrize("reverse_sources", [False, True])
@pytest.mark.parametrize("duplicate_path", [False, True])
@pytest.mark.parametrize(
    ("sizes", "requested_tokens", "max_repetition_ratio"),
    [
        ((2, 4, 4, 4), 18, 2.0),
        ((4, 2, 4, 4), 18, 2.0),
        ((4, 4, 4, 2), 18, 2.0),
        ((0, 1, 2, 4, 4, 4), 18, 2.0),
        ((2, 4, 4, 4), 32, 3.0),
        ((2, 4, 4, 4), 18, 1.5),
    ],
)
def test_dataset_mixture_trims_repetitions_before_original_paths(
    tmp_path: Path,
    reverse_sources: bool,
    duplicate_path: bool,
    sizes: tuple[int, ...],
    requested_tokens: int,
    max_repetition_ratio: float,
):
    sources = []
    for name in ("a", "b"):
        paths = []
        for index, size in enumerate(sizes):
            path = tmp_path / f"{name}_{index}.bin"
            np.full(size, index, dtype=np.uint32).tofile(path)
            paths.append(str(path))
        if duplicate_path:
            # Explicit duplicate occurrences remain part of the configured population.
            duplicate_indices = [index for index, size in enumerate(sizes) if size == 4]
            paths[duplicate_indices[1]] = paths[duplicate_indices[0]]
        sources.append(
            SourceMixtureConfig(
                source_name=name,
                target_ratio=0.5,
                paths=paths,
                max_repetition_ratio=max_repetition_ratio,
            )
        )
    if reverse_sources:
        sources.reverse()
    config = SourceMixtureDatasetConfig(
        source_list=SourceMixtureList(sources),
        requested_tokens=2 * requested_tokens,
        global_batch_size=2 * requested_tokens,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=2)

    assert [source.name for source in mixture.sources] == [source.source_name for source in sources]
    for source in mixture.sources:
        selected = source.path_tokens
        assert sum(item.tokens for item in selected) == requested_tokens
        assert [item.tokens for item in selected[: len(sizes)]] == [size // 2 * 2 for size in sizes]
        assert all(item.tokens % 2 == 0 and item.tokens <= item.max_tokens for item in selected)
        if requested_tokens == 32:
            assert [item.tokens for item in selected[len(sizes) : 2 * len(sizes)]] == list(sizes)
    assert mixture.to_paths() == [
        item.path for source in mixture.sources for item in source.path_tokens
    ]
    assert list(mixture.to_index().values()) == [
        item.tokens for source in mixture.sources for item in source.path_tokens
    ]
    assert config.build(npdtype=np.uint32, sequence_length=2).to_index() == mixture.to_index()


def test_dataset_mixture_trim_preserves_unique_shard_in_numpy_dataset(tmp_path: Path):
    paths = []
    for index, size in enumerate((2, 4, 4, 4)):
        path = tmp_path / f"source_{index}.bin"
        np.tile(np.array([index + 1, 0], dtype=np.uint32), size // 2).tofile(path)
        paths.append(str(path))
    config = SourceMixtureDatasetConfig(
        source_list=SourceMixtureList(
            [
                SourceMixtureConfig(
                    source_name="source", target_ratio=1.0, paths=paths, max_repetition_ratio=2.0
                )
            ]
        ),
        requested_tokens=18,
        global_batch_size=18,
        quiet=True,
        render_tables=False,
    )
    mixture = config.build(npdtype=np.uint32, sequence_length=2)
    assert [item.tokens for item in mixture.sources[0].path_tokens] == [2, 4, 4, 4, 0, 0, 2, 2]

    dataset = NumpyFSLDatasetConfig(
        source_mixture_config=config,
        sequence_length=2,
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
        dtype=NumpyDatasetDType.uint32,
        include_instance_metadata=False,
    ).build()
    dataset.prepare()
    assert len(dataset) == 9
    assert {int(dataset[i]["input_ids"][0]) for i in range(len(dataset))} == {
        1,
        2,
        3,
        4,
    }


@pytest.mark.parametrize("reverse_sources", [False, True])
@pytest.mark.parametrize("duplicate_path", [False, True])
@pytest.mark.parametrize("large_path_index", [0, 3, 5])
@pytest.mark.parametrize(
    ("requested_tokens", "max_repetition_ratio"), [(110, 2.0), (110, 1.13), (210, 3.0)]
)
def test_dataset_mixture_refills_originals_before_repetitions(
    tmp_path: Path,
    reverse_sources: bool,
    duplicate_path: bool,
    large_path_index: int,
    requested_tokens: int,
    max_repetition_ratio: float,
):
    sizes = [9] * 6
    sizes[large_path_index] = 100
    if duplicate_path:
        sizes.append(100)
        requested_tokens += 100
    sources = []
    for name in ("a", "b"):
        paths = []
        for index, size in enumerate(sizes):
            path = tmp_path / f"{name}_{index}.bin"
            np.zeros(size, dtype=np.uint32).tofile(path)
            paths.append(str(path))
        if duplicate_path:
            paths[-1] = paths[large_path_index]
        sources.append(
            SourceMixtureConfig(
                source_name=name,
                target_ratio=0.5,
                paths=paths,
                max_repetition_ratio=max_repetition_ratio,
            )
        )
    if reverse_sources:
        sources.reverse()
    config = SourceMixtureDatasetConfig(
        source_list=SourceMixtureList(sources),
        requested_tokens=2 * requested_tokens,
        global_batch_size=2 * requested_tokens,
        quiet=True,
        render_tables=False,
    )
    mixture = config.build(npdtype=np.uint32, sequence_length=10)
    assert [source.name for source in mixture.sources] == [source.source_name for source in sources]
    for source in mixture.sources:
        selected = source.path_tokens
        assert sum(item.tokens for item in selected) == requested_tokens
        assert [item.tokens for item in selected[: len(sizes)]] == [
            size // 10 * 10 for size in sizes
        ]
        assert all(item.tokens % 10 == 0 and item.tokens <= item.max_tokens for item in selected)
        for index, item in enumerate(selected):
            if item.tokens and index >= len(sizes):
                assert all(
                    earlier.tokens == earlier.max_tokens // 10 * 10
                    for earlier in selected[: index // len(sizes) * len(sizes)]
                )
    assert mixture.to_paths() == [
        item.path for source in mixture.sources for item in source.path_tokens
    ]
    assert config.build(npdtype=np.uint32, sequence_length=10).to_index() == mixture.to_index()


def test_dataset_mixture_refill_preserves_all_unique_sequences_in_numpy_dataset(tmp_path: Path):
    paths = []
    for index, size in enumerate((100, 9, 9, 9, 9, 9)):
        path = tmp_path / f"source_{index}.bin"
        tokens = np.ones(size, dtype=np.uint32)
        if index == 0:
            for document in range(10):
                tokens[document * 10 : (document + 1) * 10] = document + 1
                tokens[(document + 1) * 10 - 1] = 0
        else:
            tokens[-1] = 0
        tokens.tofile(path)
        paths.append(str(path))
    config = SourceMixtureDatasetConfig(
        source_list=SourceMixtureList(
            [SourceMixtureConfig("source", 1.0, paths, max_repetition_ratio=2.0)]
        ),
        requested_tokens=110,
        global_batch_size=110,
        quiet=True,
        render_tables=False,
    )
    mixture = config.build(npdtype=np.uint32, sequence_length=10)
    assert [item.tokens for item in mixture.sources[0].path_tokens] == [
        100,
        0,
        0,
        0,
        0,
        0,
        10,
        0,
        0,
        0,
        0,
        0,
    ]
    dataset = NumpyFSLDatasetConfig(
        source_mixture_config=config,
        sequence_length=10,
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
        dtype=NumpyDatasetDType.uint32,
        include_instance_metadata=False,
    ).build()
    dataset.prepare()
    assert len(dataset) == 11
    assert {int(dataset[index]["input_ids"][0]) for index in range(len(dataset))} == set(
        range(1, 11)
    )


def test_dataset_mixture_trims_per_path_rounding_to_source_target(tmp_path: Path):
    paths = []
    for index in range(101):
        path = tmp_path / f"source_{index}.bin"
        np.zeros(10, dtype=np.uint32).tofile(path)
        paths.append(str(path))

    config = SourceMixtureDatasetConfig(
        requested_tokens=1000,
        source_list=SourceMixtureList(
            [SourceMixtureConfig(source_name="source", target_ratio=1.0, paths=paths)]
        ),
        global_batch_size=1000,
        quiet=True,
        render_tables=False,
    )

    mixture = config.build(npdtype=np.uint32, sequence_length=10)

    selected = [item.tokens for item in mixture.sources[0].path_tokens]
    assert sum(selected) == 1000
    assert sum(token // 10 for token in selected) == 100
    assert all(token <= 10 for token in selected)


def test_dataset_mixture_build_duplicate_paths(tmp_path: Path):
    sources = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=1, size=500_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }

    source_configs = SourceMixtureList(
        [
            SourceMixtureConfig(
                source_name="1",
                target_ratio=0.33,  # 990k tokens
                max_repetition_ratio=2.0,
                paths=[
                    str(sources["1"][0][0]),
                    str(sources["1"][0][0]),
                ],  # Duplicate the 1 path for source 1
            ),
            SourceMixtureConfig(
                source_name="2", target_ratio=0.33, paths=[str(i[0]) for i in sources["2"]]
            ),
            SourceMixtureConfig(
                source_name="3",
                target_ratio=0.34,
                paths=[str(i[0]) for i in sources["3"]],
            ),
        ]
    )

    max_tokens = 3_000_000
    sequence_length = 1024
    global_batch_size = sequence_length * 32

    config = SourceMixtureDatasetConfig(
        requested_tokens=max_tokens,
        source_list=source_configs,
        quiet=True,
        render_tables=False,
        global_batch_size=global_batch_size,
    )

    expected = [str(sources["1"][0][0])] + [str(item[0]) for item in list(chain(*sources.values()))]
    mixture = config.build(npdtype=np.uint32, sequence_length=sequence_length)
    index = mixture.to_index()
    paths = mixture.to_paths()
    assert paths == expected
    assert len(index) == 6, "Expected 6 unique paths in the index, but got {}".format(len(index))
    assert isinstance(mixture, SourceMixtureDataset)
    assert len(mixture.sources) == 3, "Expected 3 sources, but got {}".format(len(mixture.sources))
    requested_instances = math.ceil(max_tokens / global_batch_size) * int(
        global_batch_size / sequence_length
    )
    assert (
        sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])
        == requested_instances
    ), f"Expected {requested_instances} instances, but got {sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])}"


def _mixture_with_zero_token_path() -> SourceMixtureDataset:
    return SourceMixtureDataset(
        sources=[
            SourceMixtureOutcome(
                name="s1",
                path_tokens=[
                    SourcePathTokens(path="a.npy", tokens=100, max_tokens=100),
                    SourcePathTokens(path="b.npy", tokens=0, max_tokens=50),
                    SourcePathTokens(path="c.npy", tokens=200, max_tokens=200),
                ],
            )
        ]
    )


def test_source_mixture_keeps_zero_token_paths_by_default():
    # Default container representation is stable: zero-token paths stay in the index/paths with
    # their original (path, idx) numbering.
    mixture = _mixture_with_zero_token_path()
    assert mixture.to_paths() == ["a.npy", "b.npy", "c.npy"]
    assert mixture.to_index() == {("a.npy", 0): 100, ("b.npy", 1): 0, ("c.npy", 2): 200}


def test_source_mixture_optionally_filters_zero_token_paths():
    # Opt-in filtering drops zero-token paths; to_paths()/to_index() renumber together and stay aligned.
    mixture = _mixture_with_zero_token_path()
    mixture.filter_zero_token_paths = True
    assert mixture.to_paths() == ["a.npy", "c.npy"]
    assert mixture.to_index() == {("a.npy", 0): 100, ("c.npy", 1): 200}
