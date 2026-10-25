"""Tests for legacy Arrow dataset loading."""

import pyarrow as pa
from datasets import Dataset, DatasetDict

from olmo_core.data.multimodal import dataset_compat
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


def test_load_from_disk_compat_checks_every_dataset_dict_split(tmp_path, monkeypatch):
    """A later legacy split must not be hidden by an earlier ordinary split."""
    (tmp_path / "dataset_dict.json").write_text('{"splits": ["train", "validation"]}')
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    train = Dataset.from_dict({"value": [1]})
    validation = Dataset.from_dict({"value": [2]})
    loaded_paths = []

    monkeypatch.setattr(
        dataset_compat,
        "_has_list_feature",
        lambda path: path.name == "validation",
    )
    monkeypatch.setattr(dataset_compat, "_load_arrow_split", lambda path: validation)

    import datasets

    def load_from_disk(path, **kwargs):
        del kwargs
        loaded_paths.append(path)
        return train

    monkeypatch.setattr(datasets, "load_from_disk", load_from_disk)
    result = dataset_compat.load_from_disk_compat(tmp_path)

    assert isinstance(result, DatasetDict)
    assert result["train"][0]["value"] == 1
    assert result["validation"][0]["value"] == 2
    assert loaded_paths == [str(tmp_path / "train")]
