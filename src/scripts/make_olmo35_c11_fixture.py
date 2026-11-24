#!/usr/bin/env python3
"""Build the open-instruct reference fixture for the olmo35 layout and the C11 cache reader.

This script imports only open-instruct's SFT stack (``c11-sft`` @ 2ba2c0044, the checkout C11
trained from) and the OLMo-core it pins (2f74ec30), so it runs in open-instruct's environment::

    HF_HUB_OFFLINE=1 HF_HUB_CACHE=<hub cache holding the tokenizer> \\
        <open-instruct>/.venv/bin/python src/scripts/make_olmo35_c11_fixture.py \\
        --open-instruct <checkout at 2ba2c0044> --olmo-core <checkout at 2f74ec30> \\
        --out src/test/fixtures/olmo35_c11

It samples rows of C11's three parquets and writes two files:

* ``rows.json.gz``: each row's messages and tools, with the token ids and label mask that
  open-instruct's ``sft_tulu_tokenize_and_truncate_v1`` gives it under C11's tokenizer and
  template. The olmo35 layout must reproduce these exactly.
* ``cache.npz``: a numpy SFT cache of the same rows written by open-instruct's
  ``convert_hf_to_numpy_sft`` with ``row_aligned_parts`` (C11's ``ROW_ALIGNED_PARTS=1``), and
  the instances, label masks and ``doc_lens`` that open-instruct's ``_numpy_dataset_config``
  gives with metadata boundaries (C11's ``DOCB=1``). The C11 cache reader must reproduce them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

C11_DATA = "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/data"
SOURCES = {
    # name -> (C11 parquet, rows to take: (multi-turn, single-turn))
    "dolci": (f"{C11_DATA}/dolci/dolci-think-clean-v1-en-dbv-5bd68837.parquet", (4, 8)),
    "nemo": (f"{C11_DATA}/nemo/nemotron-tool-oi-en-bd-825bdb83.parquet", (4, 0)),
    "sci": (f"{C11_DATA}/sci/nemotron-sci-v2-h083sel-bd-16f9aaaa.parquet", (0, 3)),
}
TOKENIZER = "allenai/dolma2-tokenizer-olmo35"
TOKENIZER_REVISION = "8b9717061fae09d5be814373d189919a62a9a00d"
MAX_SEQ_LENGTH = 65536
MAX_ROW_CHARS = 12_000
"""Keeps the fixture small; the rows still span multi-turn, reasoning and tool-call turns."""
PACK_SEQUENCE_LENGTH = 8192
"""Several rows per instance, so the boundaries between them are exercised."""
SHUFFLE_SEED = 33333


def _sample(path: str, n_multi: int, n_single: int):
    import pyarrow as pa
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    picked = []
    multi = single = 0
    for group in range(f.metadata.num_row_groups):
        table = f.read_row_group(group)
        keep = []
        for i, row in enumerate(table.to_pylist()):
            chars = len(json.dumps(row["messages"])) + len(str(row.get("tools") or ""))
            if chars > MAX_ROW_CHARS:
                continue
            is_multi = sum(m["role"] == "assistant" for m in row["messages"]) > 1 or any(
                m["role"] in ("tool", "environment") for m in row["messages"]
            )
            if is_multi and multi < n_multi:
                multi += 1
                keep.append(i)
            elif not is_multi and single < n_single:
                single += 1
                keep.append(i)
        if keep:
            picked.append(table.take(keep))
        if multi == n_multi and single == n_single:
            return pa.concat_tables(picked)
    raise RuntimeError(f"{path}: not enough short rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-instruct", required=True)
    parser.add_argument("--olmo-core", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tokenizer-path",
        default=TOKENIZER,
        help="A local snapshot of the tokenizer at TOKENIZER_REVISION, for hosts without the hub",
    )
    args = parser.parse_args()
    sys.path[:0] = [args.open_instruct, os.path.join(args.olmo_core, "src")]

    import pyarrow.parquet as pq
    from open_instruct import dataset_transformation as dt
    from open_instruct import numpy_dataset_conversion as ndc
    from open_instruct import olmo_core_finetune, olmo_core_utils

    from olmo_core.data import NumpyPackedFSLDataset

    tc = dt.TokenizerConfig(
        tokenizer_name_or_path=args.tokenizer_path,
        tokenizer_revision=TOKENIZER_REVISION if args.tokenizer_path == TOKENIZER else None,
        chat_template_name="olmo35",
    )
    tokenizer = tc.tokenizer
    fn_args = dt.sft_tokenize_fn_args(MAX_SEQ_LENGTH, dt.DEFAULT_OVER_LENGTH_STRATEGY)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        mixer = []
        for name, (path, (n_multi, n_single)) in SOURCES.items():
            table = _sample(path, n_multi, n_single)
            local = os.path.join(tmp, f"{name}.parquet")
            pq.write_table(table, local)
            mixer += [local, "1.0"]
            for row in table.to_pylist():
                tokenized = dt.sft_tulu_tokenize_and_truncate_v1(
                    {k: row[k] for k in ("messages", "tools") if k in row}, tokenizer, **fn_args
                )
                rows.append(
                    {
                        "source": name,
                        "id": row["id"],
                        "messages": row["messages"],
                        "tools": row.get("tools"),
                        "input_ids": tokenized[dt.INPUT_IDS_KEY].tolist(),
                        "label_mask": (tokenized[dt.LABELS_KEY] != -100).int().tolist(),
                    }
                )

        numpy_dir = Path(tmp) / "numpy"
        ndc.convert_hf_to_numpy_sft(
            output_dir=numpy_dir,
            dataset_mixer_list=mixer,
            dataset_mixer_list_splits=["train"],
            tc=tc,
            dataset_transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
            transform_fn_args=[fn_args, {}],
            dataset_target_columns=list(dt.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE),
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_local_cache_dir=os.path.join(tmp, "hf_cache"),
            shuffle_seed=SHUFFLE_SEED,
            row_aligned_parts=True,
        )
        files = {
            p.name: np.frombuffer(p.read_bytes(), dtype=np.uint8)
            for p in sorted(numpy_dir.iterdir())
            if p.is_file() and not p.name.startswith("_") and p.name != "tokenizer"
        }
        config = olmo_core_finetune._numpy_dataset_config(
            str(numpy_dir),
            olmo_core_utils.to_oc_tokenizer_config(tc),
            work_dir=os.path.join(tmp, "work"),
            sequence_length=PACK_SEQUENCE_LENGTH,
            document_boundary_kwargs=olmo_core_finetune._document_boundary_kwargs(True, True),
        )
        dataset = config.build()
        assert isinstance(dataset, NumpyPackedFSLDataset)
        dataset.prepare()
        instances = [dataset[i] for i in range(len(dataset))]

    with gzip.open(out / "rows.json.gz", "wt") as f:
        json.dump(
            {"tokenizer": TOKENIZER, "revision": TOKENIZER_REVISION, "rows": rows},
            f,
            sort_keys=True,
        )
    np.savez_compressed(
        out / "cache.npz",
        sequence_length=np.int64(PACK_SEQUENCE_LENGTH),
        file_names=np.array(sorted(files)),
        **{f"file:{name}": data for name, data in files.items()},
        input_ids=np.stack([x["input_ids"].numpy() for x in instances]),
        label_mask=np.stack([x["label_mask"].numpy() for x in instances]),
        doc_lens=np.concatenate([x["doc_lens"].numpy() for x in instances]),
        doc_counts=np.array([len(x["doc_lens"]) for x in instances], dtype=np.int64),
    )
    print(f"{len(rows)} rows, {len(instances)} instances of {PACK_SEQUENCE_LENGTH} -> {out}")


if __name__ == "__main__":
    main()
