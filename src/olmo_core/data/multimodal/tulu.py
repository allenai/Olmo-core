"""Tulu4 text-only NLP SFT dataset for the Molmo2 stage-1 mixture.

Ports ``Tulu4FilteredConfig`` (``mm_olmo/olmo/data/academic_datasets.py``): multi-turn,
**text-only** instruction data (no image). Examples produce a standard chat sequence with
the loss on assistant turns only — no image block, no ``<im_patch>`` tokens.

The training pipeline treats these as image-less examples: ``images`` is an empty
``(0, n_patches, patch_dim)`` array, so the collator/model see no image for them (and the
batch as a whole uses ``images=None`` when nothing in it has an image).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, POOL_H, POOL_W

from .paths import TULU4_DATA

__all__ = ["Tulu4DatasetConfig", "Tulu4Dataset"]


def _format_messages(parts: List[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    """Validate/normalize raw messages into an alternating user/assistant list, folding a
    leading system message into the first user turn (matches ``format_messages``)."""
    if not parts:
        return None
    out: List[Dict[str, str]] = []
    if parts[0]["role"] == "system":
        if len(parts) < 2 or parts[1]["role"] != "user":
            return None
        out.append(
            {"role": "user", "content": f"System: {parts[0]['content']}\n{parts[1]['content']}"}
        )
        parts = parts[2:]
    elif parts[0]["role"] == "assistant":
        return None
    else:
        out.append({"role": "user", "content": parts[0]["content"]})
        parts = parts[1:]
    for ix, m in enumerate(parts):
        expected = "assistant" if ix % 2 == 0 else "user"
        if m["role"] != expected:
            return None
        out.append({"role": m["role"], "content": m["content"]})
    # need at least one user + one assistant turn
    if len(out) <= 1 or out[-1]["role"] != "assistant":
        return None
    return out


@dataclass
class Tulu4DatasetConfig(Config):
    """Tulu4 filtered text SFT (matches mm_olmo ``get_dataset('tulu4')`` filter)."""

    max_first_msg_len: int = 4096
    max_sequence_length: int = 4096
    """Truncate the full tokenized conversation to this length (Tulu can exceed
    ``max_first_msg_len`` when it has many turns)."""
    use_code: bool = False
    use_non_english: bool = False
    use_reasoning: bool = False
    use_puzzles: bool = False
    seed: int = 0

    def build(self, tokenizer) -> "Tulu4Dataset":
        return Tulu4Dataset(self, tokenizer)


class Tulu4Dataset:
    def __init__(self, config: Tulu4DatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = self._load_filtered()

    def _load_filtered(self):
        from .dataset_compat import load_from_disk_compat

        ds = load_from_disk_compat(TULU4_DATA)
        ds = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
        cfg = self.config

        def _keep(category, source, n_tokens, empty_messages, has_special_token):
            if empty_messages or has_special_token:
                return False
            if source in ("allenai/dino-hardcodes", "allenai/hardcoded-olmo"):
                return False
            if not cfg.use_puzzles and source == "allenai/puzzle_data_160k-ngram-filtered":
                return False
            if not cfg.use_reasoning and source in (
                "faezeb/verifiable-reasoning-v3-o4-mini-length-filtered-verified",
                "allenai/verifiable-reasoning-filtered-o4-mini-filtered",
            ):
                return False
            if not cfg.use_code and category == "code":
                return False
            if not cfg.use_non_english and category == "non-english":
                return False
            if cfg.max_first_msg_len and n_tokens is not None and n_tokens > cfg.max_first_msg_len:
                return False
            return True

        cols = [
            "category",
            "source",
            "first_message_qwen3_tokens",
            "empty_messages",
            "has_special_token",
        ]
        return ds.filter(_keep, input_columns=cols)

    def __len__(self) -> int:
        return len(self._data)

    def _text_sequence(self, messages: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """Tokenize the conversation as ONE sequential branch (mm_olmo ``text_sft``).

        Turn layout matches mm_olmo's qwen3 ``apply_chat_template``
        (``preprocessor_utils.py:244-291``): turn 1 is
        ``<|im_start|>user\\n{u}<|im_end|>\\n<|im_start|>assistant\\n{a}``; turns 2+
        prefix the user message with ``<|im_end|>\\n`` (closing the previous assistant
        turn). Only the conversation's final token is an EOS target — intermediate
        assistant spans carry no mid-sequence EOS loss (their last token's shifted
        weight is the following user prefix's 0) — and the root-length weight is
        ``2/sqrt(total_assistant_tokens + 1)``, counting no separator tokens.
        """
        from .qwen3_layout import followup_turn_context_ids, user_turn_ids
        from .sequence_builder import build_branched_sequence

        tok = self.tokenizer
        segments = []
        for turn_ix, ix in enumerate(range(0, len(messages) - 1, 2)):
            u, a = messages[ix]["content"], messages[ix + 1]["content"]
            if turn_ix == 0:
                ctx = user_turn_ids(tok, u)
            else:
                ctx = followup_turn_context_ids(tok, u)
            segments.append((ctx, tok.encode(a, add_special_tokens=False)))

        seq = build_branched_sequence(
            [],  # no shared prefix and no BOS (qwen3)
            [segments],
            eos_id=tok.eos_token_id,
            loss_token_weighting="root_subsegments_root_tokens",
        )
        # text-only: zero image crops / pooled rows -> no image for this example.
        seq["images"] = np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32)
        seq["pooled_patches_idx"] = np.full((0, POOL_H * POOL_W), -1, dtype=np.int64)
        return seq

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        messages = _format_messages(list(self._data[i]["messages"]))
        # The filter already drops malformed rows (mm_olmo asserts likewise); raising
        # lets the loader's skip-broken policy move on instead of training on junk.
        assert messages is not None, f"Malformed Tulu row at index {i}"
        seq = self._text_sequence(messages)
        max_len = self.config.max_sequence_length
        if max_len and len(seq["input_ids"]) > max_len:
            # mm_olmo truncates to min(max_len, last loss token + 1) and refuses
            # examples whose loss is entirely cut (example_preprocessor.py:260-270).
            loss_positions = np.nonzero(seq["loss_masks"][:max_len] > 0)[0]
            if len(loss_positions) == 0:
                raise ValueError(f"Truncation to {max_len} removed all loss tokens (index {i})")
            truncate_to = min(max_len, int(loss_positions[-1]) + 1)
            seq = {
                k: (v[:truncate_to] if v.ndim == 1 and len(v) > truncate_to else v)
                for k, v in seq.items()
            }
        return seq
