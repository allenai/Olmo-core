"""Encode SFT chat turns + image into Molmo2 training tensors."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2
from olmo_core.nn.vision.molmo2_tokens import (
    DEFAULT_MOLMO2_TOKEN_IDS,
    Molmo2TokenIds,
    build_image_token_ids,
)

from .message_weight import (
    MessageWeight,
    apply_message_weight_to_loss_masks,
    loss_token_weighting_for_build,
)
from .sequence_builder import build_branched_sequence
from .sft_common import SftMessageFormat, validate_sft_message_format

__all__ = ["encode_sft_example"]


def _multi_image_prefix_ids(
    tokenizer,
    image_grids: Sequence[np.ndarray],
    *,
    message_format: SftMessageFormat,
    token_ids: Molmo2TokenIds,
) -> List[int]:
    """Build a format-specific prefix with model-specific image structural IDs."""
    if message_format == "qwen3":
        from .qwen3_layout import user_header_ids

        ids = user_header_ids(tokenizer)
    else:
        if tokenizer.eos_token_id is None:
            raise ValueError("The document layout requires an EOS token")
        ids = [int(tokenizer.eos_token_id)]

    multi = len(image_grids) > 1
    for image_index, grid in enumerate(image_grids):
        if multi:
            ids.extend(tokenizer.encode(f"Image {image_index + 1}", add_special_tokens=False))
        resized_h, resized_w, height, width = (int(grid[i]) for i in range(4))
        ids.extend(
            build_image_token_ids(
                resized_h,
                resized_w,
                height,
                width,
                token_ids=token_ids,
            )
        )
    return ids


def encode_sft_example(
    tokenizer,
    pil_image,
    turns: Union[Sequence[Tuple[str, str]], Sequence[Sequence[Tuple[str, str]]]],
    *,
    max_crops: int = 8,
    max_images: int = 5,
    p_high_res: float = 0.0,
    high_res_max_crops: int = 24,
    loss_token_weighting: str = "root_subsegments_root_tokens",
    token_ids: Optional[Molmo2TokenIds] = None,
    message_format: SftMessageFormat = "qwen3",
    message_weight: Union[None, float, MessageWeight] = None,
    seed: int = 0,
    shuffle_rng: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    """Build a branched Molmo2 SFT example from (user, assistant) turn pairs.

    ``pil_image`` may be a single image or a **list** of images (multi-image
    example). Multi-image handling follows mm_olmo's ``MultiImagePreprocessor`` +
    ``build_sequence``: at most ``max_images`` images, each preprocessed with the
    same crop budget, ``"Image {i+1}"`` text prefixes when there is more than one
    image, crops concatenated along the crop axis, and each image's pooled patch
    indices offset by the running crop-patch count.
    """
    import torch

    validate_sft_message_format(message_format)
    resolved_token_ids = token_ids or DEFAULT_MOLMO2_TOKEN_IDS

    rng = shuffle_rng if shuffle_rng is not None else np.random.RandomState(seed)

    pil_images = pil_image if isinstance(pil_image, (list, tuple)) else [pil_image]
    pil_images = list(pil_images)[:max_images]

    grids = []
    crops_list = []
    pooling_list = []
    pooled_offset = 0
    for img in pil_images:
        images_t, pooling_t, image_grid = preprocess_image_molmo2(
            img,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_crops=max_crops,
            p_high_res=p_high_res,
            high_res_max_crops=high_res_max_crops,
            is_training=True,
            rng=rng,
        )
        crops = images_t[0].numpy()  # (n_crops, n_patches, patch_dim)
        pooled = pooling_t[0].numpy()  # (n_pool, pool_size), indices local to this image
        # Offset into the concatenated (total_crops * n_patches) axis
        # (mm_olmo build_sequence: token_pooling_offset += prod(images.shape[:2])).
        pooled = np.where(pooled >= 0, pooled + pooled_offset, pooled)
        pooled_offset += int(np.prod(crops.shape[:2]))
        grids.append(image_grid)
        crops_list.append(crops)
        pooling_list.append(pooled)

    # `turns` may be flat [(q, a), ...] (each an independent branch) or a list of
    # branches, each a list of sequential (q, a) turns (SftFormatter.format_branches).
    def _is_turn(x) -> bool:
        return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str)

    if turns and _is_turn(turns[0]):
        branch_turns: List[List[Tuple[str, str]]] = [[cast(Tuple[str, str], t)] for t in turns]
    else:
        branch_turns = [list(cast(Sequence[Tuple[str, str]], b)) for b in turns]
    branch_turns = [[(q, a) for q, a in b if a] for b in branch_turns]
    branch_turns = [b for b in branch_turns if b]
    if not branch_turns:
        raise ValueError("No valid (question, answer) branches")

    if len(branch_turns) > 1:
        order = np.arange(len(branch_turns))
        rng.shuffle(order)
        branch_turns = [branch_turns[i] for i in order]

    prefix = _multi_image_prefix_ids(
        tokenizer,
        grids,
        message_format=message_format,
        token_ids=resolved_token_ids,
    )
    multi_branch = len(branch_turns) > 1

    def _branch_segments(branch: List[Tuple[str, str]]):
        segments = []
        for turn_ix, (q, a) in enumerate(branch):
            if message_format == "qwen3":
                from .qwen3_layout import branch_context_ids as qwen3_branch_context_ids
                from .qwen3_layout import followup_turn_context_ids

                if turn_ix == 0:
                    ctx = qwen3_branch_context_ids(
                        tokenizer, q, branch_index=0, multi_branch=multi_branch
                    )
                else:
                    ctx = followup_turn_context_ids(tokenizer, q)
                response = tokenizer.encode(a, add_special_tokens=False)
            else:
                from .document_layout import (
                    branch_context_ids as document_branch_context_ids,
                )
                from .document_layout import message_ids, response_ids

                ctx = (
                    document_branch_context_ids(tokenizer, q)
                    if turn_ix == 0
                    else message_ids(tokenizer, q, first=False)
                )
                response = response_ids(tokenizer, a)
            segments.append((ctx, response))
        return segments

    branches = [_branch_segments(b) for b in branch_turns]

    # `message_weight` may be a plain weight scale or a full MessageWeight override
    # (mm_olmo DatasetWithArgs.message_weight can disable root_length /
    # root_subsegments too); with_overrides handles both.
    mw = MessageWeight.from_string(loss_token_weighting).with_overrides(message_weight)
    seq = build_branched_sequence(
        prefix,
        branches,
        eos_id=tokenizer.eos_token_id,
        image_token_ids=resolved_token_ids.image_token_ids,
        loss_token_weighting=loss_token_weighting_for_build(mw),
    )
    subsegment_ids = seq.get("subsegment_ids")
    seq["loss_masks"] = apply_message_weight_to_loss_masks(
        seq["loss_masks"],
        subsegment_ids,
        mw,
        branch_scaling_already_applied=True,
    )
    if not crops_list:
        # Text-only example: zero crops / pooled rows (same shape convention as Tulu).
        from olmo_core.nn.vision.molmo2_tokens import (
            N_PATCHES_SQ,
            PATCH_DIM,
            POOL_H,
            POOL_W,
        )

        seq["images"] = np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32)
        seq["pooled_patches_idx"] = np.full((0, POOL_H * POOL_W), -1, dtype=np.int64)
        return seq
    seq["images"] = np.concatenate(crops_list, axis=0) if len(crops_list) > 1 else crops_list[0]
    seq["pooled_patches_idx"] = (
        np.concatenate(pooling_list, axis=0) if len(pooling_list) > 1 else pooling_list[0]
    )
    return seq
