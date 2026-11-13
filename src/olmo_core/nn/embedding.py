"""Embedding tables for transformer models."""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SplitVocabEmbedding"]


class SplitVocabEmbedding(nn.Module):
    """
    An embedding table split into a **base** block and a smaller **extra** block, held as two
    separate parameters so they can be frozen, initialised and optimised independently.

    This mirrors mm_olmo's ``Embedding(embedding, new_embedding)``. The base block carries the
    language backbone's pretrained vocabulary; the extra block carries tokens added on top of
    it — for Molmo2, the 128 image-special tokens (``<im_patch>``, ``<im_col>``, …).

    Two properties make this more than a cosmetic split:

    * **Independent freezing.** Molmo2's stage-1 recipe holds the pretrained vocabulary fixed
      while learning the new image tokens. With a single fused table that is a partial-row
      freeze, which ``requires_grad`` cannot express; with this split it is just
      ``embeddings.weight.requires_grad_(False)``.
    * **A base-width tied head.** The base block is named :attr:`weight`, so weight tying
      (``lm_head.w_out.weight = embeddings.weight``) produces a head spanning the *base* vocab
      only — the extra tokens are inputs, never prediction targets, which is exactly mm_olmo's
      behaviour and removes the need to mask logit columns.

    Lookups accept IDs across the whole range ``[0, num_embeddings)``; the two blocks are
    concatenated on each forward, as in mm_olmo.

    :param num_embeddings: Size of the base vocabulary.
    :param num_extra_embeddings: Number of extra token rows appended after the base block.
    :param embedding_dim: Model dimension.
    """

    def __init__(
        self,
        num_embeddings: int,
        num_extra_embeddings: int,
        embedding_dim: int,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if num_extra_embeddings <= 0:
            raise ValueError(
                f"num_extra_embeddings must be positive, got {num_extra_embeddings}; "
                "use a plain nn.Embedding when there are no extra tokens"
            )
        # Mirrors `nn.Embedding`'s attribute so callers that do their own `F.embedding`
        # lookup (e.g. MultimodalLM, which splices image features) work unchanged.
        self.padding_idx: Optional[int] = None
        self.num_base_embeddings = num_embeddings
        self.num_extra_embeddings = num_extra_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        self.extra_weight = nn.Parameter(
            torch.empty(num_extra_embeddings, embedding_dim, dtype=dtype, device=device)
        )

    @property
    def num_embeddings(self) -> int:
        """Total vocabulary size that can be looked up (base + extra)."""
        return self.num_base_embeddings + self.num_extra_embeddings

    def full_weight(self) -> torch.Tensor:
        """The two blocks concatenated into one ``(num_embeddings, embedding_dim)`` tensor."""
        return torch.cat([self.weight, self.extra_weight], dim=0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.full_weight())

    def extra_repr(self) -> str:
        return f"{self.num_base_embeddings}(+{self.num_extra_embeddings}), {self.embedding_dim}"
