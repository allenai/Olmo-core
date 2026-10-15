"""Tests for legacy Arrow dataset loading."""

import pyarrow as pa
from datasets import Dataset

from olmo_core.data.multimodal.dataset_compat import _load_arrow_split


def test_load_arrow_split_concatenates_shards_virtually(tmp_path):
    """Large list columns must not be merged with pa.concat_tables (offset overflow)."""
    schema = pa.schema(
        [("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])))]
    )
    shard0 = pa.table(
        {"messages": [[{"role": "user", "content": "hi"}]]},
        schema=schema,
    )
    shard1 = pa.table(
        {"messages": [[{"role": "assistant", "content": "hello"}]]},
        schema=schema,
    )
    paths = []
    for i, table in enumerate((shard0, shard1)):
        path = tmp_path / f"data-0000{i}-of-00002.arrow"
        with pa.ipc.new_stream(path, table.schema) as writer:
            writer.write_table(table)
        paths.append(path)

    ds = _load_arrow_split(tmp_path)
    assert len(ds) == 2
    assert isinstance(ds, Dataset)
