"""Message loss weighting for Molmo2 SFT (port of mm_olmo MessageWeight)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

ATTEND_ALL_SUBSEGMENT_ID = 10000

__all__ = [
    "MessageWeight",
    "ATTEND_ALL_SUBSEGMENT_ID",
    "apply_message_weight_to_loss_masks",
    "loss_token_weighting_for_build",
]


@dataclass
class MessageWeight:
    weight: Optional[float] = None
    root_subsegments: Optional[bool] = None
    root_length: Optional[bool] = None

    @staticmethod
    def from_string(loss_token_weighting: Optional[str]) -> "MessageWeight":
        if loss_token_weighting in (None, "none"):
            return MessageWeight()
        if loss_token_weighting == "root_subsegments":
            return MessageWeight(root_subsegments=True)
        if loss_token_weighting == "root_subsegments_root_tokens":
            return MessageWeight(root_subsegments=True, root_length=True)
        raise NotImplementedError(loss_token_weighting)

    def with_overrides(self, other: Union[None, float, "MessageWeight"]) -> "MessageWeight":
        if other is None:
            return self
        if isinstance(other, (int, float)):
            return MessageWeight(
                weight=float(other),
                root_subsegments=self.root_subsegments,
                root_length=self.root_length,
            )
        return MessageWeight(
            weight=self.weight if other.weight is None else other.weight,
            root_subsegments=(
                self.root_subsegments if other.root_subsegments is None else other.root_subsegments
            ),
            root_length=self.root_length if other.root_length is None else other.root_length,
        )


def loss_token_weighting_for_build(message_weight: MessageWeight) -> str:
    """Map a :class:`MessageWeight` to ``build_branched_sequence``'s weighting mode."""
    if not message_weight.root_subsegments:
        return "none"
    if message_weight.root_length:
        return "root_subsegments_root_tokens"
    return "root_subsegments"


def apply_message_weight_to_loss_masks(
    loss_masks: np.ndarray,
    subsegment_ids: Optional[np.ndarray],
    message_weight: MessageWeight,
    *,
    branch_scaling_already_applied: bool = False,
) -> np.ndarray:
    """Apply optional branch and scalar weight scaling to shifted loss masks.

    ``build_branched_sequence`` already applies ``root_subsegments`` /
    ``root_subsegments_root_tokens`` scaling (including per-branch ``1/sqrt(n)`` when
    multi-branch). Pass ``branch_scaling_already_applied=True`` after that call so only
    the scalar ``message_weight.weight`` is applied here, matching mm_olmo's split between
    ``flatten_tree`` (per-response) and the example preprocessor post-pass (per-branch).
    """
    out = loss_masks.astype(np.float32, copy=True)
    if (
        not branch_scaling_already_applied
        and message_weight.root_subsegments
        and subsegment_ids is not None
    ):
        unique = np.unique(subsegment_ids)
        branch_ids = unique[unique != ATTEND_ALL_SUBSEGMENT_ID]
        if len(branch_ids) > 1:
            out *= 1.0 / np.sqrt(len(branch_ids))
    if message_weight.weight is not None:
        out *= message_weight.weight
    return out
