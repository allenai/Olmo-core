#!/usr/bin/env python3
"""
Pre-render olmOCR-mix pages into a persistent render cache
(:class:`olmo_core.data.multimodal.olmocr.RenderCache`), so a training run configured with
``render_cache_dir`` decodes its pages instead of rasterising them.

The training path draws each page's render size per (row, source epoch) from
``target_longest_image_dim_range``, so a cache entry is only hit when it was rendered for that
draw. Two ways to say which pages to render:

* ``--source-epochs 0 1 ...``: every kept page of the subset(s), at the size the training path
  draws for each listed source epoch (``--seed`` and ``--dim-range`` must match the training
  config; the stage-1 v3 recipes use seed 0 and 1024-2048). ``--target-dim`` instead renders
  every page once at a fixed size (the eval split's path).
* ``--refs-file``: a TSV of ``<pdf path>\\t<longest dim>`` lines, e.g. the pages a particular run
  will touch, enumerated from the loader's deterministic reference stream.

Pages already in the cache are skipped, so the script can be re-run or sharded
(``--shard i/n``) across jobs writing into the same directory.

Examples::

    python src/scripts/prerender_olmocr_pages.py --cache-dir /weka/.../olmocr-render-cache \\
        --subset national_archives --source-epochs 0 --processes 32
    python src/scripts/prerender_olmocr_pages.py --cache-dir /weka/.../olmocr-render-cache \\
        --refs-file smoke_refs.tsv --processes 32
"""

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool
from typing import Iterator, List, Optional, Tuple

from olmo_core.data.multimodal.olmocr import (
    OLMOCR_SUBSETS,
    RENDER_CACHE_FORMATS,
    OlmOcrMixDatasetConfig,
    RenderCache,
    canonical_subset,
    render_pdf_page,
)
from olmo_core.data.multimodal.paths import OLMOCR_MIX

log = logging.getLogger("prerender_olmocr_pages")

_cache: Optional[RenderCache] = None


def _init_worker(cache_dir: str, image_format: str, root: str) -> None:
    global _cache
    _cache = RenderCache(cache_dir, image_format=image_format, root=root, write=True)


def _render_one(item: Tuple[str, int]) -> Tuple[str, int, float]:
    """Render one (pdf, dim) into the cache: returns (status, bytes written, seconds)."""
    assert _cache is not None
    pdf, dim = item
    path = _cache.path_for(pdf, 0, dim)
    if os.path.exists(path):
        return "skipped", 0, 0.0
    t0 = time.perf_counter()
    try:
        render_pdf_page(pdf, dim, cache=_cache)
    except Exception as e:  # noqa: BLE001 - report and continue; the loader skips such rows too
        log.warning("failed %s @ %d: %s", pdf, dim, e)
        return "failed", 0, time.perf_counter() - t0
    if not os.path.exists(path):
        return "failed", 0, time.perf_counter() - t0
    return "rendered", os.path.getsize(path), time.perf_counter() - t0


def _refs_from_file(path: str) -> Iterator[Tuple[str, int]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pdf, dim = line.split("\t")
            yield pdf, int(dim)


def _refs_from_subsets(args: argparse.Namespace) -> Iterator[Tuple[str, int]]:
    subsets: List[str] = (
        list(OLMOCR_SUBSETS) if args.subset == "all" else [canonical_subset(args.subset)]
    )
    languages = None if args.languages == ["all"] else tuple(args.languages)
    for subset in subsets:
        config = OlmOcrMixDatasetConfig(
            subset=subset,
            split=args.split,
            dataset_path=args.dataset_path,
            languages=languages,
            seed=args.seed,
            target_longest_image_dim_range=(args.dim_range[0], args.dim_range[1]),
            target_longest_image_dim=args.target_dim or 1536,
        )
        # The tokenizer is only used to build examples, not to pick pages or sizes.
        dataset = config.build(tokenizer=None)
        log.info("%s/%s: %d pages", subset, args.split, len(dataset))
        for i in range(len(dataset)):
            pdf = dataset.pdf_path(dataset._data[int(dataset._index[i])])
            if args.target_dim is not None:
                yield pdf, args.target_dim
            else:
                for epoch in args.source_epochs:
                    yield pdf, dataset.target_dim_for(dataset.epoch_rng(i, epoch))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-dir", required=True, help="RenderCache directory to populate")
    parser.add_argument("--format", default="webp", choices=RENDER_CACHE_FORMATS)
    parser.add_argument("--dataset-path", default=OLMOCR_MIX, help="olmOCR-mix root")
    parser.add_argument("--refs-file", help="TSV of '<pdf path>\\t<longest dim>' lines to render")
    parser.add_argument(
        "--subset",
        default="all",
        help="documents / books / loc_transcripts / national_archives / all",
    )
    parser.add_argument("--split", default="train", choices=["train", "eval"])
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="primary_language filter ('all' keeps every row)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the dataset config's seed (render-size draws)"
    )
    parser.add_argument(
        "--dim-range", type=int, nargs=2, default=[1024, 2048], metavar=("LO", "HI")
    )
    parser.add_argument(
        "--source-epochs",
        type=int,
        nargs="+",
        default=[0],
        help="source epochs whose render sizes to produce",
    )
    parser.add_argument(
        "--target-dim", type=int, help="render every page once at this fixed size instead"
    )
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument(
        "--shard", default="0/1", help="i/n: render the i-th of n interleaved slices"
    )
    parser.add_argument("--limit", type=int, help="stop after this many pages (for trials)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    shard, n_shards = (int(x) for x in args.shard.split("/"))
    refs = _refs_from_file(args.refs_file) if args.refs_file else _refs_from_subsets(args)
    items = [item for k, item in enumerate(refs) if k % n_shards == shard]
    if args.limit is not None:
        items = items[: args.limit]
    log.info(
        "%d pages to render into %s (%s, %d processes)",
        len(items),
        args.cache_dir,
        args.format,
        args.processes,
    )

    counts = {"rendered": 0, "skipped": 0, "failed": 0}
    nbytes = 0
    cpu_seconds = 0.0
    t0 = time.time()
    with Pool(
        args.processes,
        initializer=_init_worker,
        initargs=(args.cache_dir, args.format, args.dataset_path),
    ) as pool:
        for k, (status, size, seconds) in enumerate(
            pool.imap_unordered(_render_one, items, chunksize=8), 1
        ):
            counts[status] += 1
            nbytes += size
            cpu_seconds += seconds
            if k % 1000 == 0 or k == len(items):
                log.info(
                    "%d/%d: %s, %.2f GB, %.0f s wall, %.0f CPU-s",
                    k,
                    len(items),
                    counts,
                    nbytes / 1e9,
                    time.time() - t0,
                    cpu_seconds,
                )
    wall = time.time() - t0
    rendered = max(counts["rendered"], 1)
    log.info(
        "done: %s; %.3f GB written (%.0f KB/page); %.0f s wall, %.0f CPU-s (%.3f s/page)",
        counts,
        nbytes / 1e9,
        nbytes / rendered / 1024,
        wall,
        cpu_seconds,
        cpu_seconds / rendered,
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
