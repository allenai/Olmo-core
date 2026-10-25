"""Subsets of NVIDIA's Nemotron-Image-Training v3 / v3.5 for the stage-1 vision mixture.

The releases ship text only: each JSONL row is a ``messages`` conversation whose user turn names
an image by a path or key in its upstream dataset, and the images must be fetched from there.
:func:`prepare_nemotron_subset` does that once per subset: it downloads the subset's JSONL at a
pinned revision, parses and cleans the conversations, fetches every referenced image from its
upstream copy (:data:`NEMOTRON_IMAGE_SOURCES`), and saves rows of ``id`` / ``image`` (encoded
bytes, long side at most :data:`MAX_IMAGE_SIDE`) / ``prompts`` / ``responses`` with
``save_to_disk`` under ``<dataset_path>/<subset>``. Rows whose image cannot be fetched are
dropped and counted in ``<subset>/prepare_stats.json``.

The subsets are English and single-image. Two kinds are used:

* captions (``openimages_1``, ``openimages_2``, ``cc3m``: Qwen3-VL detailed captions, behind a
  ``detailed_caption`` tag; ``textcaps_commercial``: one-sentence captions that read the image's
  text, behind ``text_caption``). Filler openings ("Of course.", "This is a detailed description
  of ...") are dropped (:func:`clean_caption`).
* reasoning (``ecd`` charts, ``deepvision_reasoning`` visual logic puzzles,
  ``babyvision_reasoning`` early-vision perception puzzles), each behind a ``*_reasoning`` tag of
  its own. The ``<think>`` trace is trained, as the text mid-training mix trains its reasoning
  traces: the response is ``<think>\\n{trace}\\n</think>\\n{answer}`` in plain text
  (:func:`clean_reasoning`), and rows whose trace never closes or whose answer is empty are
  dropped.

Every source's user turn is its question behind its ``"<style>:"`` tag, as the stage-1 academic
sources do (``chart_reasoning: Which platform has ...``).
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import logging
import os
import re
import tarfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .message_sequence import encode_sft_example
from .paths import NEMOTRON_IMAGE, OE_ENCODER_DATA
from .sft_common import (
    EpochSeededExamples,
    SftMessageFormat,
    get_example_with_skip,
    truncate_for_format,
)

__all__ = [
    "NEMOTRON_RELEASES",
    "NEMOTRON_SUBSETS",
    "NEMOTRON_IMAGE_SOURCES",
    "NemotronSubset",
    "NemotronImageDatasetConfig",
    "NemotronImageDataset",
    "clean_caption",
    "clean_reasoning",
    "parse_conversation",
    "prepare_nemotron_subset",
]

log = logging.getLogger(__name__)

NEMOTRON_RELEASES: Dict[str, Tuple[str, str]] = {
    "v3": ("nvidia/Nemotron-Image-Training-v3", "7656391d4d4cb11ec3722b34f10d499435de0460"),
    "v3.5": ("nvidia/Nemotron-Image-Training-v3.5", "b82fcf11cb83b08fd11858f41a79b0cd0681f5b4"),
}
"""HF dataset repo and pinned revision of each release."""

MAX_IMAGE_SIDE = 1536
"""Images with a longer side are downscaled to it (and re-encoded as JPEG) when prepared. The
stage-1 processor tiles at most 8 crops, about 1,000 pixels a side, so nothing it sees is lost."""


@dataclass(frozen=True)
class NemotronSubset:
    """One Nemotron subset as a stage-1 source."""

    release: str
    """``"v3"`` or ``"v3.5"`` (:data:`NEMOTRON_RELEASES`)."""
    subset: str
    """The subset's directory in the release (its JSONL is ``<subset>/<subset>.jsonl``)."""
    style: str
    """The task tag in front of the user turn."""
    images: str
    """Key of the image source in :data:`NEMOTRON_IMAGE_SOURCES`."""
    reasoning: bool
    """Train the ``<think>`` trace (:func:`clean_reasoning`) rather than a caption
    (:func:`clean_caption`)."""


NEMOTRON_SUBSETS: Dict[str, NemotronSubset] = {
    "nemotron_openimages_1": NemotronSubset(
        "v3", "openimages_1", "detailed_caption", "openimages", False
    ),
    "nemotron_openimages_2": NemotronSubset(
        "v3", "openimages_2", "detailed_caption", "openimages", False
    ),
    "nemotron_cc3m": NemotronSubset("v3", "cc3m", "detailed_caption", "cc3m", False),
    "nemotron_textcaps": NemotronSubset(
        "v3", "textcaps_commercial", "text_caption", "textcaps", False
    ),
    "nemotron_ecd": NemotronSubset("v3", "ecd", "chart_reasoning", "ecd", True),
    "nemotron_deepvision": NemotronSubset(
        "v3.5", "deepvision_reasoning", "visual_logic_reasoning", "deepvision", True
    ),
    "nemotron_babyvision": NemotronSubset(
        "v3.5", "babyvision_reasoning", "visual_perception_reasoning", "babyvision", True
    ),
}
"""The stage-1 sources, keyed by their mixture source names. Left out on purpose:
``visualpuzzles_reasoning`` and ``vision_language_reasoning`` (their images are the
VisualPuzzles and SpatialEval benchmarks), ``slake_med`` (more rows than Slake's English train
split) and the subsets whose images need PDF rendering or web scraping (``hopchain_*``,
``sa_finance_*``, ``raven``, ``turing``, ``pmc_vqa``)."""


# ---------------------------------------------------------------------------
# Conversation parsing and cleaning
# ---------------------------------------------------------------------------

_CAPTION_FILLER = re.compile(
    r"^\s*(of course[.,!]?|based on the image provided,|this is a detailed description of)"
    r"[^\n]*(\n+|$)",
    re.IGNORECASE,
)


def clean_caption(text: str) -> Optional[str]:
    """A caption without a filler opening line (``"Of course. Here is ..."``,
    ``"This is a detailed description of the provided image."``), or ``None`` when nothing is
    left."""
    text = _CAPTION_FILLER.sub("", text.strip(), count=1).strip()
    return text or None


def clean_reasoning(text: str) -> Optional[str]:
    """A reasoning response as ``<think>\\n{trace}\\n</think>\\n{answer}``, or ``None`` when it
    has no closed, non-empty trace followed by a non-empty answer."""
    text = text.strip()
    if text.count("<think>") != 1 or text.count("</think>") != 1 or not text.startswith("<think>"):
        return None
    trace, answer = text[len("<think>") :].split("</think>")
    trace, answer = trace.strip(), answer.strip()
    if not trace or not answer:
        return None
    return f"<think>\n{trace}\n</think>\n{answer}"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        c if isinstance(c, str) else c.get("text", "")
        for c in content
        if isinstance(c, str) or c.get("type") == "text"
    )


def _content_images(content: Any) -> List[str]:
    if isinstance(content, str):
        return []
    return [c["image"] for c in content if isinstance(c, dict) and c.get("type") == "image"]


def parse_conversation(
    row: Dict[str, Any], clean: Callable[[str], Optional[str]]
) -> Optional[Tuple[str, List[str], List[str]]]:
    """``(image reference, prompts, responses)`` of one JSONL row, or ``None`` unless it is a
    single-image conversation of alternating user and assistant turns, with the image in the
    first user turn and every response surviving ``clean``."""
    messages = [m for m in row["messages"] if m["role"] != "system"]
    if not messages or len(messages) % 2:
        return None
    images: List[Tuple[int, str]] = []
    prompts: List[str] = []
    responses: List[str] = []
    for k, message in enumerate(messages):
        if message["role"] != ("user" if k % 2 == 0 else "assistant"):
            return None
        if k % 2 == 0:
            images.extend((k, ref) for ref in _content_images(message["content"]))
            prompts.append(_content_text(message["content"]).strip())
        else:
            response = clean(_content_text(message["content"]))
            if response is None:
                return None
            responses.append(response)
    if len(images) != 1 or images[0][0] != 0 or not prompts[0]:
        return None
    return images[0][1], prompts, responses


# ---------------------------------------------------------------------------
# Image sources (preparation only)
# ---------------------------------------------------------------------------


_SHARD_IMAGES = 250
"""Images per work item of the sources fetched by reference, small enough that decoding and
re-encoding spreads over every worker."""


def _chunks(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


@dataclass(frozen=True)
class _UrlSource:
    """Images fetched over HTTP: ``url(ref)`` for each reference."""

    url: Callable[[str], str]
    threads: int = 32

    def shards(self, needed: Sequence[str]) -> List[Any]:
        return _chunks(sorted(needed), _SHARD_IMAGES)

    def fetch(self, shard: List[str], needed: set) -> Iterator[Tuple[str, bytes]]:
        from concurrent.futures import ThreadPoolExecutor

        import requests

        session = requests.Session()

        def get(ref: str) -> Tuple[str, Optional[bytes]]:
            for _ in range(3):
                try:
                    r = session.get(self.url(ref), timeout=60)
                    if r.status_code == 200:
                        return ref, r.content
                    if r.status_code == 404:
                        break
                except requests.RequestException:
                    pass
            return ref, None

        with ThreadPoolExecutor(self.threads) as pool:
            for ref, data in pool.map(get, shard):
                if data is not None:
                    yield ref, data


@dataclass(frozen=True)
class _TarSource:
    """Images inside tar files; ``ref(member name)`` is the JSONL reference of a member, or
    ``None`` for members that are not images."""

    pattern: str
    ref: Callable[[str], Optional[str]]

    def shards(self, needed: Sequence[str]) -> List[Any]:
        files = sorted(glob.glob(self.pattern))
        if not files:
            raise OLMoConfigurationError(f"no tar files match {self.pattern!r}")
        return files

    def fetch(self, shard: str, needed: set) -> Iterator[Tuple[str, bytes]]:
        with tarfile.open(shard, "r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                ref = self.ref(member.name)
                if ref is not None and ref in needed:
                    stream = tar.extractfile(member)
                    if stream is not None:
                        yield ref, stream.read()


@dataclass(frozen=True)
class _DirectorySource:
    """Images under ``root`` at the JSONL reference's relative path."""

    root: str

    def shards(self, needed: Sequence[str]) -> List[Any]:
        return _chunks(sorted(needed), _SHARD_IMAGES)

    def fetch(self, shard: List[str], needed: set) -> Iterator[Tuple[str, bytes]]:
        for ref in shard:
            path = os.path.join(self.root, ref)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    yield ref, f.read()


@dataclass(frozen=True)
class _Sha256ParquetSource:
    """Images embedded in a parquet column, referenced as ``<prefix><sha256 of the bytes>.png``
    (NVIDIA's key for images it took from parquet releases)."""

    path: str
    column: str
    prefix: str

    def shards(self, needed: Sequence[str]) -> List[Any]:
        import pyarrow.parquet as pq

        if not os.path.isfile(self.path):
            raise OLMoConfigurationError(f"{self.path!r} does not exist")
        return list(range(pq.ParquetFile(self.path).num_row_groups))

    def fetch(self, shard: int, needed: set) -> Iterator[Tuple[str, bytes]]:
        import pyarrow.parquet as pq

        column = pq.ParquetFile(self.path).read_row_group(shard, columns=[self.column])
        for value in column.column(self.column).to_pylist():
            for image in value if isinstance(value, list) else [value]:
                data = image.get("bytes") if isinstance(image, dict) else None
                if data:
                    ref = f"{self.prefix}{hashlib.sha256(data).hexdigest()}.png"
                    if ref in needed:
                        yield ref, data


def _sources_dir(dataset_path: str) -> str:
    return os.path.join(dataset_path, "_sources")


def _cc3m_ref(name: str) -> Optional[str]:
    base = os.path.basename(name)
    return f"image_{base}" if base.endswith(".jpg") else None


def _image_ref(name: str) -> Optional[str]:
    return os.path.basename(name) if name.lower().endswith((".png", ".jpg", ".jpeg")) else None


def _babyvision_ref(name: str) -> Optional[str]:
    # Members are ``media/babyvision_synthetic_images/<file>``; the JSONL holds them under the
    # subset's directory.
    name = name[2:] if name.startswith("./") else name
    return f"babyvision_reasoning/{name}" if name.lower().endswith((".png", ".jpg")) else None


def _openimages_url(ref: str) -> str:
    # The JSONL holds ``train/data/<id>.jpg``; the CVDF mirror serves ``train/<id>.jpg``.
    return "https://s3.amazonaws.com/open-images-dataset/train/" + os.path.basename(ref)


def _openimages(root: str) -> _UrlSource:
    return _UrlSource(_openimages_url)


def _cc3m(root: str) -> _TarSource:
    # pixparse/cc3m-wds tars (``<key>.jpg``); the JSONL holds ``image_<key>.jpg``.
    return _TarSource(os.path.join(OE_ENCODER_DATA, "cc3m", "data", "cc3m-train-*.tar"), _cc3m_ref)


def _textcaps(root: str) -> _DirectorySource:
    # TextCaps' own train images; the JSONL holds ``train_images/<id>.jpg``.
    return _DirectorySource(os.path.join(OE_ENCODER_DATA, "textcaps", "images"))


def _ecd(root: str) -> _TarSource:
    # ChartFoundation/ECD-10k-Images ``images.tar``; the JSONL holds the file name.
    return _TarSource(os.path.join(_sources_dir(root), "ecd", "images.tar"), _image_ref)


def _deepvision(root: str) -> _Sha256ParquetSource:
    # skylenage-ai/DeepVision-103K ``visual_logic-26k.parquet``, keyed by SHA-256.
    return _Sha256ParquetSource(
        os.path.join(_sources_dir(root), "deepvision", "visual_logic-26k.parquet"),
        "images",
        "deepvision_reasoning/images/",
    )


def _babyvision(root: str) -> _TarSource:
    # The synthetic images bundled with the v3.5 release. Its web images (URLs in the JSONL) are
    # not fetched, so those rows are dropped.
    return _TarSource(
        os.path.join(_sources_dir(root), "nemotron", "babyvision_reasoning", "media", "*.tar"),
        _babyvision_ref,
    )


NEMOTRON_IMAGE_SOURCES: Dict[str, Callable[[str], Any]] = {
    "openimages": _openimages,
    "cc3m": _cc3m,
    "textcaps": _textcaps,
    "ecd": _ecd,
    "deepvision": _deepvision,
    "babyvision": _babyvision,
}
"""Image source of each subset, built from the prepared dataset root (where the downloaded
upstream files live under ``_sources/``)."""


def _normalize_image(data: bytes, max_side: int) -> Optional[bytes]:
    """``data`` unchanged when it is a JPEG or PNG within ``max_side``; otherwise downscaled and
    re-encoded as JPEG. ``None`` when it does not decode."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if max(image.size) <= max_side and image.format in ("JPEG", "PNG"):
                return data
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=92)
            return out.getvalue()
    except Exception:
        return None


# Rows of the subset being prepared, by image reference: inherited by the forked workers.
_ROWS: Dict[str, List[Tuple[str, List[str], List[str]]]] = {}


def _prepare_shard(args: Tuple[Any, Any, str, int]) -> Tuple[List[str], int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, shard, out_path, max_side = args
    columns: Dict[str, list] = {"id": [], "image": [], "prompts": [], "responses": []}
    images: set = set()
    bad = 0
    for ref, data in source.fetch(shard, _ROWS.keys()):
        if ref in images:
            continue
        image = _normalize_image(data, max_side)
        if image is None:
            bad += 1
            continue
        images.add(ref)
        for row_id, prompts, responses in _ROWS[ref]:
            columns["id"].append(row_id)
            columns["image"].append(image)
            columns["prompts"].append(prompts)
            columns["responses"].append(responses)
    if not columns["id"]:
        # Nothing this shard holds is referenced (most cc3m tars hold some, not all); an empty
        # parquet file would fail the final ``from_parquet``.
        return sorted(images), bad
    pq.write_table(
        pa.table(
            columns,
            schema=pa.schema(
                [
                    ("id", pa.string()),
                    ("image", pa.binary()),
                    ("prompts", pa.list_(pa.string())),
                    ("responses", pa.list_(pa.string())),
                ]
            ),
        ),
        out_path,
    )
    return sorted(images), bad


def prepare_nemotron_subset(
    name: str,
    dataset_path: str = NEMOTRON_IMAGE,
    *,
    jsonl_path: Optional[str] = None,
    num_workers: int = 32,
    max_side: int = MAX_IMAGE_SIDE,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """Prepare one subset of :data:`NEMOTRON_SUBSETS` under ``<dataset_path>/<name>``.

    :param jsonl_path: The subset's JSONL; downloaded from the pinned release into
        ``<dataset_path>/_sources/nemotron`` when not given.
    :param limit: Only the first ``limit`` JSONL rows (for a smoke test).
    :returns: Row and image counts, also written to ``<name>/prepare_stats.json``.
    """
    import multiprocessing
    import shutil

    import datasets

    subset = NEMOTRON_SUBSETS[name]
    if jsonl_path is None:
        from huggingface_hub import hf_hub_download

        repo, revision = NEMOTRON_RELEASES[subset.release]
        jsonl_path = hf_hub_download(
            repo,
            f"{subset.subset}/{subset.subset}.jsonl",
            repo_type="dataset",
            revision=revision,
            local_dir=os.path.join(_sources_dir(dataset_path), "nemotron"),
        )
    clean = clean_reasoning if subset.reasoning else clean_caption
    stats = {"jsonl_rows": 0, "parsed_rows": 0, "duplicate_rows": 0}
    _ROWS.clear()
    seen = set()
    with open(jsonl_path) as f:
        for line in f:
            if limit is not None and stats["jsonl_rows"] >= limit:
                break
            stats["jsonl_rows"] += 1
            row = json.loads(line)
            parsed = parse_conversation(row, clean)
            if parsed is None:
                continue
            ref, prompts, responses = parsed
            key = (ref, tuple(prompts), tuple(responses))
            if key in seen:
                stats["duplicate_rows"] += 1
                continue
            seen.add(key)
            stats["parsed_rows"] += 1
            _ROWS.setdefault(ref, []).append((str(row["id"]), prompts, responses))
    del seen
    source = NEMOTRON_IMAGE_SOURCES[subset.images](dataset_path)
    out = os.path.join(dataset_path, name)
    tmp = out + ".parts"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    shards = source.shards(list(_ROWS))
    jobs = [
        (source, s, os.path.join(tmp, f"{i:05d}.parquet"), max_side) for i, s in enumerate(shards)
    ]
    # A source can hold the same image more than once (identical bytes in several parquet rows),
    # so the shards may emit an image's rows twice; the copies are dropped after sorting.
    images: set = set()
    bad = 0
    with multiprocessing.get_context("fork").Pool(num_workers) as pool:
        for i, (found, n_bad) in enumerate(pool.imap_unordered(_prepare_shard, jobs)):
            images.update(found)
            bad += n_bad
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                log.info("%s: %d/%d shards, %d images", name, i + 1, len(jobs), len(images))
    stats.update(referenced_images=len(_ROWS), images=len(images), undecodable_images=bad)
    _ROWS.clear()
    if not images:
        raise OLMoConfigurationError(f"{name}: none of the referenced images could be fetched")
    # Shards are named by their index and each emits its rows in a fixed order, so reading them
    # by name gives an order that does not depend on which worker finished first. Sorting the
    # dataset instead would leave an index mapping that save_to_disk resolves by random reads over
    # every image.
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(tmp, "*.parquet")))
    seen_ids: set = set()
    keep = []
    for path in files:
        for row_id in pq.read_table(path, columns=["id"]).column("id").to_pylist():
            keep.append(row_id not in seen_ids)
            seen_ids.add(row_id)
    del seen_ids
    data = datasets.Dataset.from_parquet(files, cache_dir=os.path.join(tmp, "cache"))
    if not all(keep):
        data = data.select(np.flatnonzero(keep))
    shutil.rmtree(out, ignore_errors=True)
    data.save_to_disk(out, max_shard_size="2GB")
    stats["rows"] = len(data)
    del data
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(out, "prepare_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    log.info("%s: %s", name, stats)
    return stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class NemotronImageDatasetConfig(Config):
    """One prepared subset of :data:`NEMOTRON_SUBSETS`, as ``"<style>: <question>"`` examples."""

    subset: str
    """Source name in :data:`NEMOTRON_SUBSETS` (``nemotron_ecd``, ...)."""
    dataset_path: str = NEMOTRON_IMAGE
    """Root of the prepared subsets (:func:`prepare_nemotron_subset`)."""

    max_crops: int = 8
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens (see
    :func:`~.sft_common.truncate_example`)."""
    loss_token_weighting: str = "none"
    """``"none"`` weights every response token equally, like the other stage-1 sources."""
    message_weight: Optional[float] = None
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    """Image token IDs of the selected language-model tokenizer (set by the mixture)."""
    message_format: SftMessageFormat = "qwen3"
    """``"qwen3"`` (the released Molmo2 chat layout) or ``"document"`` (plain pretraining
    documents, for a language model trained without a chat template)."""
    seed: int = 0

    def validate(self):
        if self.subset not in NEMOTRON_SUBSETS:
            raise OLMoConfigurationError(
                f"unknown Nemotron subset {self.subset!r}; expected one of "
                f"{sorted(NEMOTRON_SUBSETS)}"
            )
        if not self.dataset_path:
            raise OLMoConfigurationError("dataset_path must point at the prepared subsets")

    def build(self, tokenizer) -> "NemotronImageDataset":
        self.validate()
        return NemotronImageDataset(self, tokenizer)


class NemotronImageDataset(EpochSeededExamples):
    """Map-style dataset over one prepared Nemotron subset."""

    def __init__(self, config: NemotronImageDatasetConfig, tokenizer):
        from .dataset_compat import load_from_disk_compat

        self.config = config
        self.tokenizer = tokenizer
        self.style = NEMOTRON_SUBSETS[config.subset].style
        prepared = os.path.join(config.dataset_path, config.subset)
        if not os.path.isdir(prepared):
            # Not built here: every rank would race to write the same directory.
            raise OLMoConfigurationError(
                f"{prepared!r} does not exist; build it once with `python -m "
                f"olmo_core.data.multimodal.nemotron_image {config.subset} "
                f"--dataset-path {config.dataset_path}`"
            )
        self._data = load_from_disk_compat(prepared)
        log.info("%s: %d rows", config.subset, len(self._data))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build row ``index``, deterministically skipping unusable rows; see
        :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`."""
        return get_example_with_skip(self, index, len(self))

    def _build(self, i: int, epoch: Optional[int] = None) -> Dict[str, np.ndarray]:
        from PIL import Image

        row = self._data[i]
        image = Image.open(io.BytesIO(row["image"])).convert("RGB")
        prompts = list(row["prompts"])
        prompts[0] = f"{self.style}: {prompts[0]}"
        cfg = self.config
        seq = encode_sft_example(
            self.tokenizer,
            image,
            list(zip(prompts, row["responses"])),
            max_crops=cfg.max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            token_ids=cfg.token_ids,
            message_format=cfg.message_format,
            shuffle_rng=self.epoch_rng(i, epoch),
        )
        return truncate_for_format(seq, cfg.max_sequence_length, cfg.token_ids, cfg.message_format)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Nemotron-Image-Training subsets.")
    parser.add_argument("subsets", nargs="+", choices=sorted(NEMOTRON_SUBSETS))
    parser.add_argument("--dataset-path", default=NEMOTRON_IMAGE)
    parser.add_argument("--jsonl-path", default=None, help="Only with a single subset.")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.jsonl_path and len(args.subsets) != 1:
        parser.error("--jsonl-path needs exactly one subset")
    for subset_name in args.subsets:
        prepare_nemotron_subset(
            subset_name,
            args.dataset_path,
            jsonl_path=args.jsonl_path,
            num_workers=args.num_workers,
            limit=args.limit,
        )
