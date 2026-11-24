#!/usr/bin/env python3
"""Build the open-instruct reference fixture for the olmo35 layout under the current text SFT
recipe: ``<think>`` and ``</think>`` promoted into reserved slots (open-instruct #1911).

This script imports only open-instruct's SFT tokenization (the checkout that built the frozen
text SFT cache ``bf82a8fab8-6068a350-rowaligned``, 97ac1b53a), so it runs in open-instruct's
environment::

    HF_HUB_OFFLINE=1 HF_HUB_CACHE=<hub cache holding the tokenizer> \\
        <open-instruct>/.venv/bin/python src/scripts/make_olmo35_c12_fixture.py \\
        --open-instruct <checkout at 97ac1b53a> --out src/test/fixtures/olmo35_c12

It samples short rows of the four frozen parquets and writes ``rows.json.gz``: each row's
messages and tools, with the token ids and label mask that open-instruct's
``sft_tulu_tokenize_and_truncate_v1`` gives it, and the promoted tokens with the ids they were
seeded from. The olmo35 layout must reproduce these exactly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

C12_DATA = "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12"
SOURCES = {
    # name -> (frozen parquet, rows to take: (multi-turn, single-turn))
    "dolci": (
        f"{C12_DATA}/data/parquets-a3/dolci-think-clean-v1-en-dbv-5bd68837-c12a3-9707a865.parquet",
        (3, 6),
    ),
    "nemo": (
        f"{C12_DATA}/data/parquets-a3/nemotron-tool-oi-en-bd-825bdb83-c12a3-be9ec34a.parquet",
        (3, 0),
    ),
    "sci": (
        f"{C12_DATA}/data/parquets-a3/nemotron-sci-v2-h083sel-bd-16f9aaaa-c12a3-25035bd6.parquet",
        (0, 3),
    ),
    "if": (
        f"{C12_DATA}/if/nemotron-ifchat-v3-if-decontam-1fee8794-c12-4e2daf40.parquet",
        (2, 3),
    ),
}
TOKENIZER = "allenai/dolma2-tokenizer-olmo35"
TOKENIZER_REVISION = "8b9717061fae09d5be814373d189919a62a9a00d"
RESERVED_SLOT_TOKENS = ["<think>", "</think>"]
MAX_SEQ_LENGTH = 65536
MAX_ROW_CHARS = 12_000


def _sample(path: str, n_multi: int, n_single: int):
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    picked = []
    multi = single = 0
    for group in range(f.metadata.num_row_groups):
        for row in f.read_row_group(group).to_pylist():
            chars = len(json.dumps(row["messages"])) + len(str(row.get("tools") or ""))
            if chars > MAX_ROW_CHARS:
                continue
            is_multi = sum(m["role"] == "assistant" for m in row["messages"]) > 1 or any(
                m["role"] in ("tool", "environment") for m in row["messages"]
            )
            if is_multi and multi < n_multi:
                multi += 1
                picked.append(row)
            elif not is_multi and single < n_single:
                single += 1
                picked.append(row)
            if multi == n_multi and single == n_single:
                return picked
    raise RuntimeError(f"{path}: not enough short rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-instruct", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tokenizer-path",
        default=TOKENIZER,
        help="A local snapshot of the tokenizer at TOKENIZER_REVISION, for hosts without the hub",
    )
    args = parser.parse_args()
    sys.path[:0] = [args.open_instruct]
    from open_instruct import dataset_transformation as dt

    tc = dt.TokenizerConfig(
        tokenizer_name_or_path=args.tokenizer_path,
        tokenizer_revision=TOKENIZER_REVISION if args.tokenizer_path == TOKENIZER else None,
        chat_template_name="olmo35",
        reserved_slot_tokens=RESERVED_SLOT_TOKENS,
    )
    tokenizer = tc.tokenizer
    promoted = [
        {"content": t.content, "token_id": t.token_id, "source_ids": list(t.source_ids)}
        for t in tokenizer.promoted_reserved_slot_tokens
    ]
    fn_args = dt.sft_tokenize_fn_args(MAX_SEQ_LENGTH, dt.DEFAULT_OVER_LENGTH_STRATEGY)
    rows = []
    for name, (path, (n_multi, n_single)) in SOURCES.items():
        for row in _sample(path, n_multi, n_single):
            tokenized = dt.sft_tulu_tokenize_and_truncate_v1(
                {k: row[k] for k in ("messages", "tools") if k in row}, tokenizer, **fn_args
            )
            rows.append(
                {
                    "source": name,
                    "id": row.get("id"),
                    "messages": row["messages"],
                    "tools": row.get("tools"),
                    "input_ids": tokenized[dt.INPUT_IDS_KEY].tolist(),
                    "label_mask": (tokenized[dt.LABELS_KEY] != -100).int().tolist(),
                }
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "rows.json.gz", "wt") as f:
        json.dump(
            {
                "tokenizer": TOKENIZER,
                "revision": TOKENIZER_REVISION,
                "reserved_slot_tokens": RESERVED_SLOT_TOKENS,
                "promoted": promoted,
                "rows": rows,
            },
            f,
        )
    print(f"{len(rows)} rows, promoted {promoted}")


if __name__ == "__main__":
    main()
