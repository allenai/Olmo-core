"""PixMoCap caption-pretraining dataset for Molmo2 "stage 1" training.

A dependency-free (no ``mm_olmo``) map-style :class:`torch.utils.data.Dataset` that
turns PixMoCap image-caption examples into packed Molmo2 training sequences. Each
example produces a shared image prefix that branches
into one or more ``(task prompt, response)`` annotations (a long caption and/or a
spoken transcript), assembled by
:func:`~olmo_core.data.multimodal.sequence_builder.build_branched_sequence`. Each branch's user
turn is a bare style tag, ``long_caption:`` or ``transcript:`` (see :data:`CAPTION_STYLE`), with
nothing before or after it: no instruction, and no length number. That is what the stage-1
caption eval sends, so train and test share one form. mm_olmo's ``style_and_length`` families
put the response's length in the tag (``long_caption 63:``, the character count // 15 with
noise, at inference a fixed 65) so length could be requested at test time; that number was
dropped here on purpose, so the prompt has no value that differs between training and test.

Three data sources are supported via ``dataset_path``:

* ``"synthetic"`` — random RGB images + random short responses (for smoke tests).
* a ``.jsonl`` file — one object per line with ``image`` (path or URL),
  ``caption`` (str) and optional ``transcripts`` (list[str]).
* a HuggingFace Arrow directory (``datasets.load_from_disk``) — the canonical
  PixMoCap layout (``image``, ``caption``, ``transcripts``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .document_layout import branch_context_ids, image_prefix_ids, response_ids
from .sequence_builder import build_branched_sequence
from .sft_common import SftMessageFormat, sft_example_rng, validate_sft_message_format

__all__ = [
    "PixMoCapDataset",
    "PixMoCapDatasetConfig",
    "CAPTION_PROMPTS",
    "TRANSCRIPT_PROMPTS",
    "style_tag_prompt",
]

# Natural-language prompt pools mirroring mm_olmo's ``GENERAL_PROMPTS_V1`` (data_formatter.py).
# Only the un-tagged form (``style_tag=False``) samples from them. The tagged form that stage 1
# trains on does NOT: mm_olmo's stage-1 formatter runs with ``prompt_templates="none"``, under
# which a caption's question is the empty string (data_formatter.py, the ``"none"`` branch of
# ``get_user_prompt``), so its user turn is the style tag alone.
CAPTION_PROMPTS = (
    "Describe this image.",
    "Describe this image",
    "Write a long description of this image.",
    "Construct a long caption for this image",
    "Generate a caption",
    "Create a detailed caption",
    "Write a long caption",
    "Describe this image in detail",
)
TRANSCRIPT_PROMPTS = (
    "Describe this image as if you are a person speaking",
    "Imagine you are a person talking about this image. Generate a transcript of what you would say.",
    "Generate an audio transcript of a person describing this image",
    "Create a transcript of a human describing this image out load",
)

_MODES = ("caption", "transcript", "transcript_and_caption", "sft_demo")
_CONTENT_FINGERPRINT_VERSION = "pixmo-cap-adapter-v1"
_CONTENT_FINGERPRINT_DOMAIN = b"pixmo-cap-adapter-v1\0"

# Style names match mm_olmo: the caption branch is ``long_caption``, the spoken-transcript branch
# is ``transcript``. The user turn is ``"<style>:"`` and nothing else.
CAPTION_STYLE = "long_caption"
TRANSCRIPT_STYLE = "transcript"
_LENGTH_BUCKET = 15
_LENGTH_NOISE_STD = 25.0
_LENGTH_KEEP_PROB = 0.90


def style_tag_prompt(style: str) -> str:
    """The user turn of a free-text response source that has no question of its own (the OCR
    sources): the bare ``"<style>:"`` tag, with no length number (see the module docstring).
    """
    return f"{style}:"


@dataclass
class PixMoCapDatasetConfig(Config):
    """Configuration for :class:`PixMoCapDataset`."""

    dataset_path: str
    """``"synthetic"``, a ``.jsonl`` file, or a HF Arrow directory."""

    split: str = "train"
    require_split: bool = False
    """Fail instead of falling back to an unsplit dataset when ``split`` is absent."""
    mode: str = "transcript_and_caption"
    """One of ``"caption"``, ``"transcript"``, ``"transcript_and_caption"``."""

    require_transcript: bool = False
    """When ``mode="transcript"``, reject rows without a non-blank transcript instead of
    falling back to their caption."""

    image_root: Optional[str] = None
    """Optional prefix joined to relative image paths from a jsonl source."""

    max_crops: int = 8
    max_sequence_length: int = 5248
    loss_token_weighting: str = "root_subsegments"
    message_weight: Optional[float] = None
    """Optional multiplier applied to this source's response loss weights."""
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    """Image and chat token IDs for the selected language-model tokenizer."""
    message_format: SftMessageFormat = "qwen3"
    """Use released Qwen or native pretraining document layout."""
    fixed_prompt: Optional[str] = None
    """If set, always use this user prompt instead of the tag or the pools. Useful for
    deterministic parity tests. Overrides ``style_tag``."""

    style_tag: bool = True
    """Use the bare style tag, ``"long_caption:"`` / ``"transcript:"``, as each branch's whole
    user turn (see :data:`CAPTION_STYLE`). This is the stage-1 form, at training and at test
    time. When False the user turn is instead a natural-language prompt sampled from
    :data:`CAPTION_PROMPTS` / :data:`TRANSCRIPT_PROMPTS`, with no tag. Ignored when
    ``fixed_prompt`` is set."""

    style_length_conditioning: bool = False
    """Only with ``style_tag=False``: prefix each sampled prompt with mm_olmo's
    ``style_and_length_v2`` ``"<style>[ <bucket>]:"`` tag derived from the response length
    (see :meth:`PixMoCapDataset._style_length_prefix`). Ignored when ``fixed_prompt`` is set."""

    seed: int = 0
    synthetic_size: int = 64
    """Number of examples to generate when ``dataset_path == "synthetic"``."""

    def build(self, tokenizer) -> "PixMoCapDataset":
        return PixMoCapDataset(self, tokenizer)


class PixMoCapDataset:
    """Map-style dataset yielding packed Molmo2 caption-pretraining examples."""

    content_fingerprint_version = _CONTENT_FINGERPRINT_VERSION

    def __init__(self, config: PixMoCapDatasetConfig, tokenizer):
        if config.mode not in _MODES:
            raise ValueError(f"Unknown mode {config.mode!r}; expected one of {_MODES}")
        validate_sft_message_format(config.message_format)
        self.config = config
        self.tokenizer = tokenizer
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._hf = None
        self._sft_formatter = None
        if config.mode == "sft_demo":
            from .sft_formatter import SftFormatter

            self._sft_formatter = SftFormatter(seed=config.seed)

        path = config.dataset_path
        if path == "synthetic":
            if config.require_split:
                raise ValueError("Synthetic PixMoCap data does not provide named splits")
            self._kind = "synthetic"
        elif path.endswith(".jsonl"):
            if config.require_split:
                raise ValueError("A single PixMoCap JSONL file cannot prove a named split")
            self._kind = "jsonl"
            self._rows = self._load_jsonl(path)
        else:
            self._kind = "arrow"
            from .dataset_compat import load_from_disk_compat

            ds = load_from_disk_compat(path)
            if config.require_split and config.split not in ds:
                raise ValueError(f"PixMoCap dataset {path!r} lacks required split {config.split!r}")
            self._hf = ds[config.split] if config.split in ds else ds

        if config.require_split:
            if self._kind != "arrow" or self._hf is None:
                raise ValueError("Strict PixMoCap sources require a named Arrow split")
            arrow_fingerprint = getattr(self._hf, "_fingerprint", None)
            if callable(arrow_fingerprint):
                arrow_fingerprint = arrow_fingerprint()
            if not isinstance(arrow_fingerprint, str) or not arrow_fingerprint:
                raise ValueError(f"PixMoCap split {config.split!r} has no stable Arrow fingerprint")
            fingerprint_config = asdict(config)
            # The v1 identity omits an unset source-weight override.
            if config.message_weight is None:
                fingerprint_config.pop("message_weight")
            payload = {
                "adapter": type(self).__name__,
                "config": fingerprint_config,
                "source": {
                    "arrow_fingerprint": arrow_fingerprint,
                    "num_rows": len(self._hf),
                    "split": config.split,
                },
                "version": self.content_fingerprint_version,
            }
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.content_fingerprint = hashlib.sha256(
                _CONTENT_FINGERPRINT_DOMAIN + encoded
            ).hexdigest()

        self._eos_id = tokenizer.eos_token_id
        self._annotations_validated = False

    # -- length -----------------------------------------------------------------

    def __len__(self) -> int:
        if self._kind == "synthetic":
            return self.config.synthetic_size
        if self._kind == "jsonl":
            assert self._rows is not None
            return len(self._rows)
        assert self._hf is not None
        return len(self._hf)

    # -- loading helpers --------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        import json

        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _get_row(self, index: int) -> Dict[str, Any]:
        if self._kind == "jsonl":
            assert self._rows is not None
            return self._rows[index]
        assert self._hf is not None
        return self._hf[index]

    def raw_image_references(self, index: int) -> Tuple[Any, ...]:
        """Return the exact source image reference for one non-synthetic logical row.

        :param index: Dataset row index.
        :returns: A one-element tuple containing the original image path/cell.
        :raises ValueError: If called for generated synthetic data.
        """
        if self._kind == "synthetic":
            raise ValueError("Synthetic PixMoCap rows have no durable raw image reference")
        return (self._get_row(index)["image"],)

    def _load_image(self, row: Dict[str, Any]):
        from PIL import Image

        img = row["image"]
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, str):
            path = img
            if self.config.image_root is not None and not img.startswith(
                ("http://", "https://", "/")
            ):
                import os

                path = os.path.join(self.config.image_root, img)
            return Image.open(path)
        raise TypeError(f"Unsupported image field type: {type(img)}")

    def validate_required_annotations(self) -> None:
        """Validate every annotation that the configured branch selector can supervise.

        The scan never reads or decodes image fields. Captions must be non-blank whenever
        they can be selected. Transcript entries used by bundled Cap must all be non-blank;
        strict transcript-only mode requires at least one usable transcript per row.

        :raises ValueError: If strict transcript mode is enabled and any row lacks a
            non-blank transcript.
        """
        if getattr(self, "_annotations_validated", False):
            return
        if self._kind == "synthetic":
            self._annotations_validated = True
            return

        if self._kind == "jsonl":
            assert self._rows is not None
            rows: Any = iter(self._rows)
        else:
            assert self._hf is not None
            columns = ["caption", "transcripts"]
            missing = sorted(set(columns) - set(self._hf.column_names))
            if missing:
                raise ValueError(f"PixMoCap lacks required annotation columns: {missing}")
            rows = iter(self._hf.select_columns(columns))

        invalid_count = 0
        first_errors: List[str] = []
        for index, row in enumerate(rows):
            caption = row.get("caption")
            transcripts = row.get("transcripts")
            error: str | None = None
            caption_required = self.config.mode in ("caption", "transcript_and_caption")
            if caption_required and (not isinstance(caption, str) or not caption.strip()):
                error = "caption must be non-blank"
            elif not isinstance(transcripts, (list, tuple)):
                error = "transcripts must be a sequence"
            elif self.config.mode == "transcript_and_caption" and any(
                not isinstance(transcript, str) or not transcript.strip()
                for transcript in transcripts
            ):
                error = "bundled transcript entries must be non-blank strings"
            elif (
                self.config.mode == "transcript"
                and self.config.require_transcript
                and not any(
                    isinstance(transcript, str) and transcript.strip() for transcript in transcripts
                )
            ):
                error = "strict transcript mode requires a non-blank transcript"
            elif (
                self.config.mode == "transcript"
                and not self.config.require_transcript
                and not transcripts
                and (not isinstance(caption, str) or not caption.strip())
            ):
                error = "transcript fallback caption must be non-blank"
            if error is not None:
                invalid_count += 1
                if len(first_errors) < 8:
                    first_errors.append(f"{index}: {error}")

        if invalid_count:
            raise ValueError(
                f"PixMoCap has {invalid_count} invalid annotation rows out of {len(self)}; "
                f"first errors: {first_errors}"
            )
        self._annotations_validated = True

    # -- core -------------------------------------------------------------------

    def _select_branches(
        self, row: Dict[str, Any], rng: np.random.RandomState
    ) -> List[Tuple[str, str]]:
        """Pick the ``(style, response_text)`` branch(es) for this example per ``mode``.

        Mirrors mm_olmo ``PixMoCapConfig.format_example``: the caption branch carries
        :data:`CAPTION_STYLE` and the spoken-transcript branch :data:`TRANSCRIPT_STYLE`.
        """
        caption = row.get("caption", "")
        transcripts = row.get("transcripts") or []
        mode = self.config.mode
        if mode == "caption":
            return [(CAPTION_STYLE, caption)]
        if mode == "transcript":
            if self.config.require_transcript:
                transcripts = [
                    transcript
                    for transcript in transcripts
                    if isinstance(transcript, str) and transcript.strip()
                ]
            if not transcripts:
                if self.config.require_transcript:
                    raise ValueError(
                        "PixMoCap transcript mode requires at least one non-blank transcript"
                    )
                return [(CAPTION_STYLE, caption)]
            return [(TRANSCRIPT_STYLE, transcripts[rng.randint(len(transcripts))])]
        # transcript_and_caption: caption first, then a random transcript (if any).
        branches = [(CAPTION_STYLE, caption)]
        if transcripts:
            branches.append((TRANSCRIPT_STYLE, transcripts[rng.randint(len(transcripts))]))
        return branches

    def _style_length_prefix(self, style: str, text: str, rng: np.random.RandomState) -> str:
        """mm_olmo ``style_and_length_v2`` prefix: ``"<style> <bucket>:"`` (90%) or
        ``"<style>:"`` (10%), where ``bucket = (len(text) + N(0, 25)) // 15``."""
        if rng.rand() < _LENGTH_KEEP_PROB:
            n = len(text) + int(rng.normal(scale=_LENGTH_NOISE_STD))
            n = n // _LENGTH_BUCKET
            return f"{style} {n}:"
        return f"{style}:"

    def _sample_prompt(self, style: str, rng: np.random.RandomState) -> str:
        pool = TRANSCRIPT_PROMPTS if style == TRANSCRIPT_STYLE else CAPTION_PROMPTS
        return pool[rng.randint(len(pool))]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        if self.config.mode == "sft_demo":
            return self._getitem_sft_demo(index, epoch)

        from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

        cfg = self.config
        rng = sft_example_rng(cfg.seed, index, epoch, cfg.message_format)

        import torch

        if self._kind == "synthetic":
            from PIL import Image

            arr = rng.randint(0, 256, size=(64, 96, 3), dtype=np.uint8)
            pil = Image.fromarray(arr)
            n_words = rng.randint(8, 24)
            caption = " ".join(f"word{rng.randint(1000)}" for _ in range(n_words))
            transcripts = [" ".join(f"tok{rng.randint(1000)}" for _ in range(n_words))]
            row: Dict[str, Any] = {"caption": caption, "transcripts": transcripts}
        else:
            row = self._get_row(index)
            pil = self._load_image(row)

        branch_specs: List[Tuple[str, List[int]]] = []
        for style, text in self._select_branches(row, rng):
            if cfg.fixed_prompt is not None:
                prompt = cfg.fixed_prompt
            elif cfg.style_tag:
                # The tag is the entire user turn: no instruction after it, no length number.
                prompt = style_tag_prompt(style)
            else:
                base_prompt = self._sample_prompt(style, rng)
                if cfg.style_length_conditioning:
                    prompt = f"{self._style_length_prefix(style, text, rng)} {base_prompt}"
                else:
                    prompt = base_prompt
            if cfg.message_format == "qwen3":
                encoded_response = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                encoded_response = response_ids(self.tokenizer, text)
            branch_specs.append((prompt, encoded_response))

        if len(branch_specs) > 1 and cfg.message_format == "document":
            # The document serializer randomizes the branch order (the alignment adapters'
            # convention); the qwen3 stage-1 path keeps caption before transcript.
            rng.shuffle(branch_specs)

        # Molmo2 formats messages before image augmentation, sharing one RNG.
        images_t, pooling_t, image_grid = preprocess_image_molmo2(
            pil,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_crops=cfg.max_crops,
            rng=rng,
        )
        images = images_t[0].numpy()  # (n_crops, n_patches, patch_dim)
        pooled = pooling_t[0].numpy()  # (n_pool, pool_size)

        if cfg.message_format == "qwen3":
            from .qwen3_layout import branch_context_ids as qwen_branch_context_ids
            from .qwen3_layout import image_prefix_ids as qwen_image_prefix_ids

            prefix_ids = qwen_image_prefix_ids(
                self.tokenizer,
                image_grid,
                token_ids=cfg.token_ids,
            )
            multi_branch = len(branch_specs) > 1
            branch_pairs = [
                (
                    qwen_branch_context_ids(
                        self.tokenizer,
                        prompt,
                        branch_index=branch_index,
                        multi_branch=multi_branch,
                    ),
                    encoded_response,
                )
                for branch_index, (prompt, encoded_response) in enumerate(branch_specs)
            ]
        else:
            # Shared prefix = native EOS document boundary + image block. Each branch is plain
            # non-role text and its response receives Molmo's one-space message separator.
            prefix_ids = image_prefix_ids(self.tokenizer, image_grid, token_ids=cfg.token_ids)
            branch_pairs = [
                (branch_context_ids(self.tokenizer, prompt), encoded_response)
                for prompt, encoded_response in branch_specs
            ]

        seq = build_branched_sequence(
            prefix_ids,
            branch_pairs,
            eos_id=self._eos_id,
            image_token_ids=cfg.token_ids.image_token_ids,
            loss_token_weighting=cfg.loss_token_weighting,
        )
        if cfg.message_weight is not None:
            from .message_weight import (
                MessageWeight,
                apply_message_weight_to_loss_masks,
            )

            seq["loss_masks"] = apply_message_weight_to_loss_masks(
                seq["loss_masks"],
                seq.get("subsegment_ids"),
                MessageWeight.from_string(cfg.loss_token_weighting).with_overrides(
                    cfg.message_weight
                ),
                branch_scaling_already_applied=True,
            )

        # Truncate to max_sequence_length, never cutting an <im_patch> token.
        if len(seq["input_ids"]) > cfg.max_sequence_length:
            seq = _truncate(seq, cfg.max_sequence_length, cfg.token_ids.im_patch_id)

        seq["images"] = images
        seq["pooled_patches_idx"] = pooled
        return seq

    def _getitem_sft_demo(self, index: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        from .message_sequence import encode_sft_example

        row = self._get_row(index)
        pil = self._load_image(row)
        formatted = {
            "style": "long_caption",
            "caption": row.get("caption", ""),
            "text": row.get("caption", ""),
        }
        assert self._sft_formatter is not None
        rng = sft_example_rng(
            self.config.seed,
            index,
            epoch,
            self.config.message_format,
        )
        turns = self._sft_formatter.format_turns(formatted, index=index, rng=rng)
        return encode_sft_example(
            self.tokenizer,
            pil,
            turns,
            max_crops=self.config.max_crops,
            loss_token_weighting="root_subsegments_root_tokens",
            message_weight=self.config.message_weight,
            token_ids=self.config.token_ids,
            message_format=self.config.message_format,
            shuffle_rng=rng,
        )


def _truncate(
    seq: Dict[str, np.ndarray], max_len: int, image_patch_token_id: int
) -> Dict[str, np.ndarray]:
    """Right-truncate all per-token fields to ``max_len`` (asserting no image token cut).

    Follows mm_olmo ``example_preprocessor.py:260-293``: truncation that removes every
    loss token raises (the loader skips the example), and when whole branches are cut
    the surviving branches' ``1/sqrt(n_subsegments)`` scaling is recomputed over the
    *non-truncated* subsegments only.
    """
    loss_positions = np.nonzero(seq["loss_masks"][:max_len] > 0)[0]
    if len(loss_positions) == 0:
        raise ValueError(f"Truncation to {max_len} removed all loss tokens")
    n_before = None
    if "subsegment_ids" in seq:
        uniq = np.unique(seq["subsegment_ids"])
        n_before = len(uniq[uniq != 10000])  # exclude the ATTEND_ALL prefix id
    keep = max_len
    if np.any(seq["input_ids"][keep:] == image_patch_token_id):
        raise ValueError(
            "max_sequence_length too small: truncation would drop <im_patch> tokens "
            "(the image block must fit entirely within the sequence)."
        )
    out = {}
    for k, v in seq.items():
        out[k] = v[:keep] if v.ndim == 1 and len(v) >= keep else v
    if n_before is not None and "subsegment_ids" in out:
        uniq_after = np.unique(out["subsegment_ids"])
        n_after = len(uniq_after[uniq_after != 10000])
        if n_after and n_after != n_before:
            out["loss_masks"] = out["loss_masks"] * np.sqrt(n_before / n_after)
    return out
