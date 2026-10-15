"""PixMoCap caption-pretraining dataset for Molmo2 "stage 1" training.

A dependency-free (no ``mm_olmo``) map-style :class:`torch.utils.data.Dataset` that
turns PixMoCap image-caption examples into packed Molmo2 training sequences. Each
example produces a shared prefix (qwen3 user header + image block) that branches into
one or more ``(user turn, assistant response)`` annotations (a long caption and/or a
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config

from .qwen3_layout import branch_context_ids, image_prefix_ids
from .sequence_builder import build_branched_sequence, example_rng

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

# Style names match mm_olmo: the caption branch is ``long_caption``, the spoken-transcript branch
# is ``transcript``. The user turn is ``"<style>:"`` and nothing else.
CAPTION_STYLE = "long_caption"
TRANSCRIPT_STYLE = "transcript"


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
    mode: str = "transcript_and_caption"
    """One of ``"caption"``, ``"transcript"``, ``"transcript_and_caption"``."""

    image_root: Optional[str] = None
    """Optional prefix joined to relative image paths from a jsonl source."""

    max_crops: int = 8
    max_sequence_length: int = 5248
    loss_token_weighting: str = "root_subsegments"
    message_weight: Optional[float] = None
    """Example-wide multiplier on this source's loss tokens (mm_olmo
    ``kwargs_mixture[].datasets[].message_weight``). ``None`` weights every response token
    at 1, which is what the released ``Molmo2-4B-Pretrain`` uses; raising it increases
    captions' share of the ``sum(CE*w)/sum(w)`` objective relative to the other sources."""
    fixed_prompt: Optional[str] = None
    """If set, always use this user prompt instead of the tag or the pools. Useful for
    deterministic parity tests. Overrides ``style_tag``."""

    style_tag: bool = True
    """Use the bare style tag, ``"long_caption:"`` / ``"transcript:"``, as each branch's whole
    user turn (see :data:`CAPTION_STYLE`). This is the stage-1 form, at training and at test
    time. When False the user turn is instead a natural-language prompt sampled from
    :data:`CAPTION_PROMPTS` / :data:`TRANSCRIPT_PROMPTS`, with no tag. Ignored when
    ``fixed_prompt`` is set."""

    seed: int = 0
    synthetic_size: int = 64
    """Number of examples to generate when ``dataset_path == "synthetic"``."""

    def build(self, tokenizer) -> "PixMoCapDataset":
        return PixMoCapDataset(self, tokenizer)


class PixMoCapDataset:
    """Map-style dataset yielding packed Molmo2 caption-pretraining examples."""

    def __init__(self, config: PixMoCapDatasetConfig, tokenizer):
        if config.mode not in _MODES:
            raise ValueError(f"Unknown mode {config.mode!r}; expected one of {_MODES}")
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
            self._kind = "synthetic"
        elif path.endswith(".jsonl"):
            self._kind = "jsonl"
            self._rows = self._load_jsonl(path)
        else:
            self._kind = "arrow"
            from .dataset_compat import load_from_disk_compat

            ds = load_from_disk_compat(path)
            self._hf = ds[config.split] if config.split in ds else ds

        self._eos_id = tokenizer.eos_token_id

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
            if not transcripts:
                return [(CAPTION_STYLE, caption)]
            return [(TRANSCRIPT_STYLE, transcripts[rng.randint(len(transcripts))])]
        # transcript_and_caption: caption first, then a random transcript (if any).
        branches = [(CAPTION_STYLE, caption)]
        if transcripts:
            branches.append((TRANSCRIPT_STYLE, transcripts[rng.randint(len(transcripts))]))
        return branches

    def _sample_prompt(self, style: str, rng: np.random.RandomState) -> str:
        pool = TRANSCRIPT_PROMPTS if style == TRANSCRIPT_STYLE else CAPTION_PROMPTS
        return pool[rng.randint(len(pool))]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if self.config.mode == "sft_demo":
            return self._getitem_sft_demo(index)

        from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

        cfg = self.config
        rng = example_rng(cfg.seed, index)

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

        images_t, pooling_t, image_grid = preprocess_image_molmo2(
            pil, dtype=torch.float32, device=torch.device("cpu"), max_crops=cfg.max_crops
        )
        images = images_t[0].numpy()  # (n_crops, n_patches, patch_dim)
        pooled = pooling_t[0].numpy()  # (n_pool, pool_size)

        # Shared prefix = qwen3 user header + image block; each branch carries its own
        # user turn (full header when multi-branch, suffix-only when single-branch).
        prefix_ids = image_prefix_ids(self.tokenizer, image_grid)
        branch_specs: List[Tuple[str, List[int]]] = []
        for style, text in self._select_branches(row, rng):
            if cfg.fixed_prompt is not None:
                prompt = cfg.fixed_prompt
            elif cfg.style_tag:
                # The tag is the entire user turn: no instruction after it, no length number.
                prompt = style_tag_prompt(style)
            else:
                prompt = self._sample_prompt(style, rng)
            response_ids = self.tokenizer.encode(text, add_special_tokens=False)
            branch_specs.append((prompt, response_ids))

        multi_branch = len(branch_specs) > 1
        branch_pairs = [
            (
                branch_context_ids(
                    self.tokenizer, prompt, branch_index=i, multi_branch=multi_branch
                ),
                response_ids,
            )
            for i, (prompt, response_ids) in enumerate(branch_specs)
        ]

        seq = build_branched_sequence(
            prefix_ids,
            branch_pairs,
            eos_id=self._eos_id,
            loss_token_weighting=cfg.loss_token_weighting,
        )
        if cfg.message_weight is not None:
            # Same path pixmo_points uses: branch scaling is already folded into loss_masks by
            # build_branched_sequence, so only the example-wide multiplier is applied here.
            from olmo_core.data.multimodal.message_weight import (
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
            seq = _truncate(seq, cfg.max_sequence_length)

        seq["images"] = images
        seq["pooled_patches_idx"] = pooled
        return seq

    def _getitem_sft_demo(self, index: int) -> Dict[str, np.ndarray]:
        from olmo_core.data.multimodal.message_weight import MessageWeight

        from .message_sequence import encode_sft_example

        row = self._get_row(index)
        pil = self._load_image(row)
        formatted = {
            "style": "long_caption",
            "caption": row.get("caption", ""),
            "text": row.get("caption", ""),
        }
        assert self._sft_formatter is not None
        rng = example_rng(self.config.seed, index)
        turns = self._sft_formatter.format_turns(formatted, index=index, rng=rng)
        return encode_sft_example(
            self.tokenizer,
            pil,
            turns,
            max_crops=self.config.max_crops,
            loss_token_weighting="root_subsegments_root_tokens",
            # `message_weight` has to apply on this path too: stage 2 builds this source
            # with mode="sft_demo" (mixtures/image_only_v9.py), which returns before the
            # branched path's scaling, so setting the option would otherwise be silent.
            message_weight=(
                None
                if self.config.message_weight is None
                else MessageWeight.from_string("root_subsegments_root_tokens").with_overrides(
                    self.config.message_weight
                )
            ),
            shuffle_rng=rng,
        )


def _truncate(seq: Dict[str, np.ndarray], max_len: int) -> Dict[str, np.ndarray]:
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
    from olmo_core.nn.vision.molmo2_tokens import IM_PATCH_ID

    keep = max_len
    if np.any(seq["input_ids"][keep:] == IM_PATCH_ID):
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
