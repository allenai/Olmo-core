#!/usr/bin/env python3
"""Measure the stage-2 image sources in the ``olmo35`` layout: per-source rows, mean tokens and
loss weights per example (the mixture's ``mean_loss_weight``), documents per example, image-rate
statistics and one rendered example each.

The probe is stage-1's calibration probe: 128 rows per source drawn without replacement with
``numpy.random.RandomState(0)``, the dataset seed 0, epoch 0, the tokenizer of the current text SFT
recipe (``tokenizer.json`` split, think tags promoted) and 65,536-token sequences::

    PYTHONPATH=src python src/scripts/olmo35_sft_mm_probe.py --out <dir> [--sources a,b] [--workers N]

It writes ``<out>/probe.json`` (one entry per source) and ``<out>/renders/<source>.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

N_PROBE = 128
SEED = 0


def _all_sources():
    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        IMAGE_ONLY_V9_SUBMIXTURES,
    )

    return [s.name for g in IMAGE_ONLY_V9_SUBMIXTURES if g.name != "nlp" for s in g.datasets]


def _config(name: str, annotation_sampling: str):
    from olmo_core.data.multimodal.academic_dataset import AcademicDatasetConfig
    from olmo_core.data.multimodal.mixtures import olmo35_sft_mm_v1 as v1
    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        get_image_only_v9_source,
    )
    from olmo_core.data.multimodal.olmo35_layout import OLMO35_TOKEN_IDS

    if name in v1.DROPPED:  # measured for the record; not part of v1
        src = get_image_only_v9_source(name)
        assert src is not None
        return AcademicDatasetConfig(
            name=name,
            message_weight=src.message_weight,
            token_ids=OLMO35_TOKEN_IDS,
            message_format="olmo35",
        )
    return v1.build_source_config(name, annotation_sampling=annotation_sampling)


def _render(tokenizer, example, image_ids) -> str:
    ids = example["input_ids"]
    loss = example["loss_masks"]
    labels = example["labels"]
    docs = example.get("document_ids")
    out, run = [], 0
    for i, t in enumerate(ids.tolist()):
        if t in image_ids:
            run += 1
            continue
        if run:
            out.append(f"<image tokens x{run}>")
            run = 0
        piece = tokenizer.decode([t])
        if i > 0 and loss[i - 1] > 0:  # this token is a supervised target
            piece = f"[{piece}]"
        if docs is not None and i > 0 and docs[i] != docs[i - 1]:
            out.append("\n=== next document ===\n")
        out.append(piece)
    if run:
        out.append(f"<image tokens x{run}>")
    eos_targets = int(((labels == tokenizer.eos_token_id) & (loss > 0)).sum())
    return (
        "".join(out)
        + f"\n\n--- supervised tokens shown in [brackets]; EOS targets: {eos_targets}; "
        f"loss weights: {sorted(set(np.round(loss[loss > 0], 4).tolist()))}"
    )


def probe_source(name: str, annotation_sampling: str, renders: str) -> dict:
    from olmo_core.data.multimodal.olmo35_layout import (
        OLMO35_TOKEN_IDS,
        load_olmo35_sft_tokenizer,
    )

    t0 = time.time()
    tokenizer, _ = load_olmo35_sft_tokenizer()
    image_ids = set(OLMO35_TOKEN_IDS.image_token_ids)
    entry: dict = {"source": name}
    try:
        dataset = _config(name, annotation_sampling).build(tokenizer)
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"build: {type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()[-2000:]
        return entry
    n_rows = len(dataset)
    entry["rows"] = n_rows
    rng = np.random.RandomState(SEED)
    indices = rng.choice(n_rows, size=min(N_PROBE, n_rows), replace=False)
    tokens, loss_weight, loss_tokens, documents, crops, image_tokens = [], [], [], [], [], []
    over_length, errors, rendered = 0, [], False
    for index in indices.tolist():
        try:
            get = getattr(dataset, "get", None)
            example = get(index, 0) if get is not None else dataset[index]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{index}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        n = len(example["input_ids"])
        tokens.append(n)
        loss_weight.append(float(example["loss_masks"].sum()))
        loss_tokens.append(int((example["loss_masks"] > 0).sum()))
        docs = example.get("document_ids")
        documents.append(int(len(np.unique(docs))) if docs is not None else 1)
        crops.append(int(example["images"].shape[0]))
        image_tokens.append(int(example["token_type_ids"].sum()))
        over_length += int(n > 65_536)
        if not rendered:
            Path(renders, f"{name}.txt").write_text(_render(tokenizer, example, image_ids))
            rendered = True
    entry.update(
        probed=len(tokens),
        errors=len(errors),
        error_examples=errors[:5],
        mean_tokens=float(np.mean(tokens)) if tokens else None,
        mean_loss_weight=float(np.mean(loss_weight)) if tokens else None,
        mean_loss_tokens=float(np.mean(loss_tokens)) if tokens else None,
        mean_documents=float(np.mean(documents)) if tokens else None,
        mean_crops=float(np.mean(crops)) if tokens else None,
        mean_image_tokens=float(np.mean(image_tokens)) if tokens else None,
        max_tokens=int(np.max(tokens)) if tokens else None,
        over_65536=over_length,
        seconds=round(time.time() - t0, 1),
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", default=None, help="Comma-separated; default: all 42")
    parser.add_argument("--annotation-sampling", default="all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    renders = out / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    names = args.sources.split(",") if args.sources else _all_sources()
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(probe_source, n, args.annotation_sampling, str(renders)): n for n in names
        }
        for future in as_completed(futures):
            entry = future.result()
            results[entry["source"]] = entry
            print(json.dumps({k: v for k, v in entry.items() if k != "traceback"}), flush=True)
    from olmo_core.data.multimodal.mixture_weights import compute_flat_mixture_weights
    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        IMAGE_ONLY_V9_SUBMIXTURES,
    )

    lengths = {"tulu4": 1, **{n: e.get("rows", 0) for n, e in results.items()}}
    if all(lengths.get(s.name) for g in IMAGE_ONLY_V9_SUBMIXTURES for s in g.datasets):
        rates = dict(compute_flat_mixture_weights(IMAGE_ONLY_V9_SUBMIXTURES, lengths))
        for name, rate in rates.items():
            if name in results:
                results[name]["v9_example_rate"] = rate
    (out / "probe.json").write_text(
        json.dumps(
            {
                "probe": {
                    "rows_per_source": N_PROBE,
                    "seed": SEED,
                    "annotation_sampling": args.annotation_sampling,
                    "host": os.uname().nodename,
                },
                "sources": [results[n] for n in names if n in results],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
