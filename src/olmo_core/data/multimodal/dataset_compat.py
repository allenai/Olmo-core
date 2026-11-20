"""Compatibility loader for HuggingFace datasets saved with deprecated ``List`` features.

PixMo and Tulu datasets on weka were materialized with ``datasets`` 5.x, which uses a
legacy ``List`` feature type that ``datasets`` 3.6 (olmo-core env) cannot parse from
``dataset_info.json``.  Loading the Arrow streams directly avoids the schema mismatch.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Union

_LIST_PATCHED = False


def _patch_list_feature_type() -> None:
    global _LIST_PATCHED
    if _LIST_PATCHED:
        return
    from datasets.features.features import _FEATURE_TYPES, Sequence

    if "List" in _FEATURE_TYPES:
        # datasets >= 5 has a native ``List`` feature — aliasing it to ``Sequence``
        # would be actively harmful: ``Sequence`` re-orders a list-of-structs into a
        # struct-of-lists, silently corrupting every later schema in the process that
        # contains a list of dicts (the patch is process-global).
        _LIST_PATCHED = True
        return
    _FEATURE_TYPES["List"] = Sequence
    _LIST_PATCHED = True


def _has_list_feature(path: Path) -> bool:
    info = path / "dataset_info.json"
    if not info.exists():
        return False
    text = info.read_text()
    return '"_type": "List"' in text or '"_type":"List"' in text


def _load_arrow_split(split_dir: Path):
    from datasets import Dataset, concatenate_datasets

    _patch_list_feature_type()
    arrow_files = sorted(glob.glob(str(split_dir / "*.arrow")))
    # Ignore HF filter/map cache shards; load the materialized data shards only.
    arrow_files = [
        f
        for f in arrow_files
        if os.path.basename(f).startswith("data-") and not os.path.basename(f).startswith("cache-")
    ]
    if not arrow_files:
        raise FileNotFoundError(f"No data-*.arrow files in {split_dir}")
    if len(arrow_files) == 1:
        return Dataset.from_file(arrow_files[0])
    # Memory-map each shard and concatenate virtually. ``pa.concat_tables`` on large
    # list-typed columns (e.g. Tulu4 messages) overflows Arrow int32 offsets.
    return concatenate_datasets([Dataset.from_file(f) for f in arrow_files])


def load_from_disk_compat(path: Union[str, os.PathLike], **kwargs: Any):
    """Load a HuggingFace dataset from disk, handling legacy ``List`` feature schemas."""
    path = Path(path)
    dict_json = path / "dataset_dict.json"
    if dict_json.exists():
        from datasets import DatasetDict

        meta = json.loads(dict_json.read_text())
        splits = meta.get("splits", [])
        legacy_splits = {split for split in splits if _has_list_feature(path / split)}
        if not legacy_splits:
            from datasets import load_from_disk

            return load_from_disk(str(path), **kwargs)

        # A DatasetDict may mix old List-authored and ordinary splits. Loading the root
        # eagerly parses every split, including the incompatible one, so load each split
        # independently whenever any split needs the Arrow fallback.
        out = {}
        for split in splits:
            split_dir = path / split
            if split in legacy_splits:
                out[split] = _load_arrow_split(split_dir)
            else:
                from datasets import load_from_disk

                out[split] = load_from_disk(str(split_dir), **kwargs)
        return DatasetDict(out)

    if _has_list_feature(path):
        return _load_arrow_split(path)

    _patch_list_feature_type()
    from datasets import load_from_disk

    return load_from_disk(str(path), **kwargs)
