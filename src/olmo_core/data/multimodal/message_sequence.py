"""Encode SFT chat turns + image into Molmo2 training tensors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

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
    image_labels: Optional[Sequence[str]] = None,
) -> List[int]:
    """Build a format-specific prefix with model-specific image structural IDs.

    Each image is preceded by its label: ``image_labels[i]`` when given (for example
    ``"Page 12:"`` for the pages of a document), else ``"Image {i + 1}"`` when there is more than
    one image and nothing for a single image.
    """
    if message_format == "qwen3":
        from .qwen3_layout import user_header_ids

        ids = user_header_ids(tokenizer)
    else:
        if tokenizer.eos_token_id is None:
            raise ValueError("The document layout requires an EOS token")
        ids = [int(tokenizer.eos_token_id)]

    multi = len(image_grids) > 1
    if image_labels is not None and len(image_labels) != len(image_grids):
        raise ValueError(f"{len(image_labels)} image labels for {len(image_grids)} images")
    for image_index, grid in enumerate(image_grids):
        if image_labels is not None:
            ids.extend(tokenizer.encode(image_labels[image_index], add_special_tokens=False))
        elif multi:
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
    image_labels: Optional[Sequence[str]] = None,
    per_image_max_crops: Optional[Sequence[int]] = None,
    p_high_res: float = 0.0,
    high_res_max_crops: int = 24,
    loss_token_weighting: str = "root_subsegments_root_tokens",
    token_ids: Optional[Molmo2TokenIds] = None,
    message_format: SftMessageFormat = "qwen3",
    message_weight: Union[None, float, MessageWeight] = None,
    seed: int = 0,
    shuffle_rng: Optional[np.random.RandomState] = None,
    branches_as_documents: bool = False,
    branches_as_sequence: bool = False,
    document_token_budget: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Build a branched Molmo2 SFT example from (user, assistant) turn pairs.

    ``pil_image`` may be a single image or a **list** of images (multi-image
    example). Multi-image handling follows mm_olmo's ``MultiImagePreprocessor`` +
    ``build_sequence``: at most ``max_images`` images, each preprocessed with the
    same crop budget, ``"Image {i+1}"`` text prefixes when there is more than one
    image, crops concatenated along the crop axis, and each image's pooled patch
    indices offset by the running crop-patch count.

    :param image_labels: Text written before each image in place of ``"Image {i+1}"``, one per
        image (for example the page numbers of a document's pages). Must not be longer than
        ``max_images``.
    :param per_image_max_crops: Crop budget of each image in place of ``max_crops``, one per
        image, so that context pages can be read at a lower resolution than a target page.
    :param branches_as_documents: Serialize each branch as its own document (prefix + that
        branch, exactly the single-branch example) instead of sibling branches of one shared
        prefix, with a per-token ``document_ids`` array numbering them. The documents share the
        image's crops: every document's pooled rows index the same crops, so the vision encoder
        runs once for all of them. For language models that isolate packed examples only by
        document boundaries (Kimi Delta Attention). Requires ``loss_token_weighting="none"``,
        under which each document weights its tokens as the sibling-branch layout does.
    :param branches_as_sequence: Serialize every branch after one copy of the prefix, in the
        shuffled branch order, as a single document: each branch's turns follow the previous
        branch's answer and an EOS token, so the image's tokens appear once instead of once per
        branch. Every branch keeps the per-token weights it has as its own document (answer
        tokens plus the EOS that ends it), so the example's loss weight and the calibrated means
        are unchanged; only later branches now also see the earlier ones. For language models
        that isolate packed examples only by document boundaries (Kimi Delta Attention).
        Requires ``loss_token_weighting="none"``.
    :param document_token_budget: With ``branches_as_documents`` or ``branches_as_sequence``,
        stop adding branches (in the shuffled branch order) once the next would take the
        example past this many tokens. The first branch is always kept.
    """
    import torch

    validate_sft_message_format(message_format)
    resolved_token_ids = token_ids or DEFAULT_MOLMO2_TOKEN_IDS

    rng = shuffle_rng if shuffle_rng is not None else np.random.RandomState(seed)

    pil_images = pil_image if isinstance(pil_image, (list, tuple)) else [pil_image]
    pil_images = list(pil_images)[:max_images]
    for name, values in (
        ("image_labels", image_labels),
        ("per_image_max_crops", per_image_max_crops),
    ):
        if values is not None and len(values) != len(pil_images):
            raise ValueError(f"{name} has {len(values)} entries for {len(pil_images)} images")

    grids = []
    crops_list = []
    pooling_list = []
    pooled_offset = 0
    for image_index, img in enumerate(pil_images):
        images_t, pooling_t, image_grid = preprocess_image_molmo2(
            img,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_crops=(
                max_crops if per_image_max_crops is None else per_image_max_crops[image_index]
            ),
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
        image_labels=image_labels,
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
    if branches_as_documents and branches_as_sequence:
        raise ValueError("branches_as_documents and branches_as_sequence are exclusive")
    if branches_as_sequence and len(branches) > 1:
        n_documents = 1
        seq = _build_sequence(
            prefix,
            branches,
            mw,
            eos_id=tokenizer.eos_token_id,
            image_token_ids=resolved_token_ids.image_token_ids,
            token_budget=document_token_budget,
        )
    elif branches_as_documents and len(branches) > 1:
        seq, n_documents = _build_documents(
            prefix,
            branches,
            mw,
            eos_id=tokenizer.eos_token_id,
            image_token_ids=resolved_token_ids.image_token_ids,
            token_budget=document_token_budget,
        )
    else:
        n_documents = 1
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
    pooled = np.concatenate(pooling_list, axis=0) if len(pooling_list) > 1 else pooling_list[0]
    # Each document's image block pools the same crops.
    seq["pooled_patches_idx"] = np.tile(pooled, (n_documents, 1)) if n_documents > 1 else pooled
    return seq


def _build_sequence(
    prefix: Sequence[int],
    branches: Sequence[Any],
    message_weight: MessageWeight,
    *,
    eos_id: int,
    image_token_ids: frozenset,
    token_budget: Optional[int],
) -> Dict[str, np.ndarray]:
    """Serialize ``prefix + branch_1 + EOS + branch_2 + EOS + ... + branch_n`` as one document.

    Each branch is the list of ``(context, response)`` turns that :func:`_build_documents` would
    put after its own copy of the prefix. Here the prefix (document boundary and image tokens)
    appears once and the branches follow it in order; an EOS token closes every branch but the
    last, whose EOS is the document's final target as usual. The EOS after a branch carries that
    branch's answer weight, as the final EOS target of its own document does, and the context
    tokens after it carry none, so each branch contributes the same loss weight as in
    :func:`_build_documents`.
    """
    if loss_token_weighting_for_build(message_weight) != "none":
        raise ValueError(
            "branches_as_sequence requires loss_token_weighting='none': branch scaling by "
            "count has no equivalent in one sequential document"
        )
    turns: List[Tuple[List[int], List[int]]] = []
    n_tokens = len(prefix)
    for branch in branches:
        segments = [(list(context), list(response)) for context, response in branch]
        # +1: the EOS that closes this branch (input token here, or the final EOS target).
        length = sum(len(c) + len(r) for c, r in segments) + 1
        if turns and token_budget is not None and n_tokens + length > token_budget:
            break
        if turns:
            context, response = turns[-1]
            turns[-1] = (context, response + [int(eos_id)])
        turns.extend(segments)
        n_tokens += length
    seq = build_branched_sequence(
        prefix,
        [turns],
        eos_id=eos_id,
        image_token_ids=image_token_ids,
        loss_token_weighting="none",
    )
    seq["loss_masks"] = apply_message_weight_to_loss_masks(
        seq["loss_masks"], None, message_weight, branch_scaling_already_applied=True
    )
    return seq


def _build_documents(
    prefix: Sequence[int],
    branches: Sequence[Any],
    message_weight: MessageWeight,
    *,
    eos_id: int,
    image_token_ids: frozenset,
    token_budget: Optional[int],
) -> Tuple[Dict[str, np.ndarray], int]:
    """Serialize each branch as ``prefix + branch``, the single-branch example, and concatenate
    them with a per-token ``document_ids`` array. Returns the example and its document count."""
    if loss_token_weighting_for_build(message_weight) != "none":
        raise ValueError(
            "branches_as_documents requires loss_token_weighting='none': the sibling-branch "
            "layout scales branches by their count, which separate documents do not reproduce"
        )
    documents: List[Dict[str, np.ndarray]] = []
    n_tokens = 0
    for branch in branches:
        document = build_branched_sequence(
            prefix,
            [branch],
            eos_id=eos_id,
            image_token_ids=image_token_ids,
            loss_token_weighting="none",
        )
        document["loss_masks"] = apply_message_weight_to_loss_masks(
            document["loss_masks"], None, message_weight, branch_scaling_already_applied=True
        )
        length = len(document["input_ids"])
        if documents and token_budget is not None and n_tokens + length > token_budget:
            break
        documents.append(document)
        n_tokens += length
    seq = {key: np.concatenate([d[key] for d in documents], axis=0) for key in documents[0]}
    seq["document_ids"] = np.concatenate(
        [np.full(len(d["input_ids"]), i, dtype=np.int64) for i, d in enumerate(documents)]
    )
    return seq, len(documents)
