"""Build packed training sequences for Molmo2 multi-annotation multimodal data.

This is a dependency-free port of the sequence-assembly performed by ``mm_olmo``'s
``RefactoredExamplePreprocessor`` (``flatten_tree`` + ``build_sequence`` in
``olmo/models/molmo2/example_preprocessor.py``) for the common case of a single
shared prefix (BOS + image block + user prompt + assistant header) that branches
into one or more assistant responses (e.g. a caption plus a transcript).

Multiple branches that share one image are packed into a single sequence where
each branch is *isolated* from its siblings via subsegment attention and the
branches share an overlapping RoPE position range (each branch continues from the
end of the shared prefix). The loss is a float per-token weight, response-only,
scaled by ``1/sqrt(n_branches)`` under ``root_subsegments`` weighting so the packed
sequence is equivalent in expectation to sampling one annotation.

See :class:`~olmo_core.nn.vision.MultimodalLM` for how ``subsegment_ids`` /
``position_ids`` / ``loss_masks`` are consumed at training time.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from olmo_core.nn.vision.molmo2_tokens import IMAGE_TOKEN_IDS

# Subsegment id of the shared prefix. Larger than any branch id so that, under the
# ``subseg[q] <= subseg[k]`` rule, prefix tokens only attend to other prefix tokens
# while every branch attends the prefix. Matches mm_olmo's ``ATTEND_ALL_SUBSEGMENT_ID``.
ATTEND_ALL_SUBSEGMENT_ID = 10000

LOSS_TOKEN_WEIGHTINGS = ("none", "root_subsegments", "root_subsegments_root_tokens")


def example_rng(
    seed: int, index: int, *, epoch: int = 0, dataset_len: int = 0
) -> np.random.RandomState:
    """Per-example rng stream (mm_olmo ``dataset.py:70-73``).

    mm_olmo derives one stream per example as ``seed * 195172 + index + len(dataset) * epoch``
    (mod 2**32-1) and threads it through the dataset's ``format_example`` AND the formatter, so
    all of an example's draws are sequential. Using the same derivation keeps our draws
    alignable with mm_olmo artifacts at every index (not just index 0).

    :param epoch: Training epoch. The default of 0 reproduces the single-epoch stream, so a
        caller that does not track epochs is unaffected. Sources whose ``__getitem__`` *samples*
        (a negative pool, a render size) must pass it, or every epoch redraws the same choice
        and the un-sampled remainder is never trained on; see
        :class:`~olmo_core.data.multimodal.sft_common.EpochSeededExamples`.
    :param dataset_len: Number of rows, the stride mm_olmo advances the stream by per epoch.
        Ignored when ``epoch`` is 0.
    """
    return np.random.RandomState((seed * 195172 + index + dataset_len * epoch) % (2**32 - 1))


def build_packed_sequence(
    prefix_ids: Sequence[int],
    response_id_lists: Sequence[Sequence[int]],
    *,
    eos_id: int,
    image_token_ids: frozenset = IMAGE_TOKEN_IDS,
    loss_token_weighting: str = "root_subsegments",
) -> Dict[str, np.ndarray]:
    """Assemble a packed training example from a shared prefix and response branches.

    :param prefix_ids: Token IDs of the shared prefix, ending with the assistant
        header (e.g. ``…<|im_start|>assistant\\n``). The final token is "carried over"
        into each branch as a non-loss token so each branch copy can predict its own
        first response token (matching mm_olmo).
    :param response_id_lists: One token-ID list per annotation (e.g. caption,
        transcript). Each is the assistant response **without** a trailing EOS — the
        EOS is supplied as the target at the branch's final position.
    :param eos_id: End-of-sequence token id used as the target at each branch end.
    :param image_token_ids: Token IDs that count as image tokens for ``token_type_ids``.
    :param loss_token_weighting: ``"none"`` (binary response weights),
        ``"root_subsegments"`` (each branch scaled by ``1/sqrt(n_branches)``), or
        ``"root_subsegments_root_tokens"`` (additionally ``2/sqrt(n_response_tokens)``
        per branch).

    :returns: A dict of 1-D ``np.ndarray`` with keys ``input_ids``, ``labels``,
        ``loss_masks`` (float32), ``position_ids``; plus ``subsegment_ids`` when there
        is more than one branch. ``labels``/``loss_masks`` are already shifted to align
        with next-token prediction. ``token_type_ids`` is also returned.

    :raises ValueError: If no responses are given or the weighting is unknown.
    """
    if loss_token_weighting not in LOSS_TOKEN_WEIGHTINGS:
        raise ValueError(
            f"Unknown loss_token_weighting {loss_token_weighting!r}; "
            f"expected one of {LOSS_TOKEN_WEIGHTINGS}"
        )
    n_branches = len(response_id_lists)
    if n_branches == 0:
        raise ValueError("`response_id_lists` must contain at least one response")

    prefix_ids = list(prefix_ids)
    if len(prefix_ids) == 0:
        raise ValueError("`prefix_ids` must be non-empty")

    root_length = loss_token_weighting == "root_subsegments_root_tokens"

    # ``parts`` mirror mm_olmo's flatten_tree output: each is
    # (tokens, weight, start_position, subsegment_id, is_segment_end_token[]).
    parts: List[dict] = []

    if n_branches == 1:
        # No branching: a single causal sequence (prefix + response), sequential
        # positions, no subsegments. Weight is binary unless root_length.
        response = list(response_id_lists[0])
        tokens = prefix_ids + response
        weight = 1.0
        if root_length:
            n_resp = len(response) + 1  # +1 for EOS
            weight = 2.0 / np.sqrt(n_resp) if n_resp else 0.0
        loss = np.zeros(len(tokens), dtype=np.float32)
        loss[len(prefix_ids) :] = weight  # response tokens get loss
        seg_end = np.zeros(len(tokens), dtype=bool)
        seg_end[-1] = True
        parts.append(
            dict(
                tokens=np.asarray(tokens, dtype=np.int64),
                loss=loss,
                position=np.arange(len(tokens), dtype=np.int64),
                subsegment_id=None,
                seg_end=seg_end,
            )
        )
    else:
        # Branching: shared prefix (minus carry-over) + N isolated branches that each
        # start with the carried-over last prefix token and continue from position
        # ``len(prefix) - 1``.
        carry_over = prefix_ids[-1]
        prefix_body = prefix_ids[:-1]
        start_position = len(prefix_ids) - 1

        prefix_tokens = np.asarray(prefix_body, dtype=np.int64)
        parts.append(
            dict(
                tokens=prefix_tokens,
                loss=np.zeros(len(prefix_body), dtype=np.float32),
                position=np.arange(len(prefix_body), dtype=np.int64),
                subsegment_id=ATTEND_ALL_SUBSEGMENT_ID,
                seg_end=np.zeros(len(prefix_body), dtype=bool),
            )
        )
        for branch_idx, response in enumerate(response_id_lists):
            response_ids = list(response)
            branch_tokens = [carry_over] + response_ids
            branch_weight = 1.0
            if root_length:
                n_resp = len(response) + 1 if n_branches == 1 else len(response)
                branch_weight = 2.0 / np.sqrt(n_resp) if n_resp else 0.0
            loss = np.zeros(len(branch_tokens), dtype=np.float32)
            loss[1:] = branch_weight  # carry-over (idx 0) is non-loss; response gets loss
            seg_end = np.zeros(len(branch_tokens), dtype=bool)
            seg_end[-1] = True
            parts.append(
                dict(
                    tokens=np.asarray(branch_tokens, dtype=np.int64),
                    loss=loss,
                    position=np.arange(
                        start_position, start_position + len(branch_tokens), dtype=np.int64
                    ),
                    subsegment_id=branch_idx,
                    seg_end=seg_end,
                )
            )

    input_ids = np.concatenate([p["tokens"] for p in parts], 0)
    loss_mask = np.concatenate([p["loss"] for p in parts], 0)
    position_ids = np.concatenate([p["position"] for p in parts], 0)
    seg_ends = np.concatenate([p["seg_end"] for p in parts], 0)

    # Labels via shift; segment ends predict EOS rather than the next segment's token.
    labels = np.zeros_like(input_ids)
    labels[:-1] = input_ids[1:]
    labels[seg_ends] = eos_id

    # Loss masks shifted to align with labels; segment ends keep the unshifted weight.
    loss_mask_shifted = np.zeros_like(loss_mask)
    loss_mask_shifted[:-1] = loss_mask[1:]
    loss_mask_shifted[seg_ends] = loss_mask[seg_ends]

    # root_subsegments: scale every loss weight by 1/sqrt(n_branches) (no-op for 1 branch).
    if loss_token_weighting in ("root_subsegments", "root_subsegments_root_tokens"):
        if n_branches > 1:
            loss_mask_shifted = loss_mask_shifted / np.sqrt(n_branches)

    token_type_ids = np.isin(input_ids, np.fromiter(image_token_ids, dtype=np.int64)).astype(
        np.int64
    )

    out: Dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_mask_shifted.astype(np.float32),
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
    }
    if n_branches > 1:
        out["subsegment_ids"] = np.concatenate(
            [np.full(len(p["tokens"]), p["subsegment_id"], dtype=np.int64) for p in parts],
            0,
        )
    return out


def build_branched_sequence(
    prefix_ids: Sequence[int],
    branches: Sequence[Tuple[Sequence[int], Sequence[int]]],
    *,
    eos_id: int,
    image_token_ids: frozenset = IMAGE_TOKEN_IDS,
    loss_token_weighting: str = "root_subsegments",
) -> Dict[str, np.ndarray]:
    """Assemble a packed example where each branch carries its OWN user turn.

    Unlike :func:`build_packed_sequence` (caption: a shared prompt in the prefix, branches
    are assistant-only and carry over the prefix's last token), this handles the
    pointing/counting layout where the shared prefix is just ``BOS + image block`` and each
    branch is a full ``(user-turn, assistant-answer)`` pair. Branches are isolated by
    subsegment and share an overlapping position range starting right after the prefix (no
    carry-over, since each branch begins with its own user turn).

    :param prefix_ids: Shared prefix token IDs (BOS + image block), all non-loss.
    :param branches: One entry per annotation. Each entry is either a single
        ``(context_ids, response_ids)`` pair — ``context_ids`` is the non-loss user turn
        (e.g. ``<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n``),
        ``response_ids`` the loss-bearing assistant answer — or a **list** of such pairs
        forming one sequential multi-turn conversation (mm_olmo keeps a
        ``{"messages": [...]}`` annotation as one branch: turn 2 attends turn 1, loss on
        every assistant span, and only the final token is a segment end / EOS target, so
        intermediate assistant spans get no mid-sequence EOS loss).
    :param eos_id: EOS token id (target at each branch end).
    :param loss_token_weighting: as in :func:`build_packed_sequence`.

    :returns: Same dict shape as :func:`build_packed_sequence`.
    """
    if loss_token_weighting not in LOSS_TOKEN_WEIGHTINGS:
        raise ValueError(f"Unknown loss_token_weighting {loss_token_weighting!r}")
    n_branches = len(branches)
    if n_branches == 0:
        raise ValueError("`branches` must be non-empty")
    prefix_ids = list(prefix_ids)
    root_length = loss_token_weighting == "root_subsegments_root_tokens"

    parts: List[dict] = []
    multi = n_branches > 1

    def _as_segments(branch):
        """Normalize a branch to a list of (context, response) turn segments."""
        if len(branch) == 2 and len(branch[0]) > 0 and isinstance(branch[0][0], (int, np.integer)):
            return [branch]
        return list(branch)

    def _branch_part(branch, subseg_id):
        segments = [(list(c), list(r)) for c, r in _as_segments(branch)]
        tokens: List[int] = []
        loss_spans: List[tuple] = []  # (start, end) of each response span
        for context, response in segments:
            tokens.extend(context)
            loss_spans.append((len(tokens), len(tokens) + len(response)))
            tokens.extend(response)
        total_resp = sum(e - s for s, e in loss_spans)
        w = 1.0
        if root_length:
            # mm_olmo flatten_tree: n = total assistant tokens (+1 for EOS when the
            # example has a single annotation).
            n_resp = total_resp + (0 if multi else 1)
            w = 2.0 / np.sqrt(n_resp) if n_resp else 0.0
        loss = np.zeros(len(tokens), dtype=np.float32)
        for s, e in loss_spans:
            loss[s:e] = w
        seg_end = np.zeros(len(tokens), dtype=bool)
        seg_end[-1] = True  # only the branch's final token is a segment end (leaf)
        return dict(loss=loss, seg_end=seg_end, subsegment_id=subseg_id, tokens=tokens)

    if not multi:
        # Single annotation: prefix + conversation, fully causal, no subsegments.
        bp = _branch_part(branches[0], None)
        tokens = prefix_ids + bp["tokens"]
        loss = np.concatenate([np.zeros(len(prefix_ids), dtype=np.float32), bp["loss"]])
        seg_end = np.concatenate([np.zeros(len(prefix_ids), dtype=bool), bp["seg_end"]])
        parts.append(
            dict(
                tokens=np.asarray(tokens, dtype=np.int64),
                loss=loss,
                position=np.arange(len(tokens), dtype=np.int64),
                subsegment_id=None,
                seg_end=seg_end,
            )
        )
    else:
        start_position = len(prefix_ids)  # branches continue from end of prefix (no carry-over)
        parts.append(
            dict(
                tokens=np.asarray(prefix_ids, dtype=np.int64),
                loss=np.zeros(len(prefix_ids), dtype=np.float32),
                position=np.arange(len(prefix_ids), dtype=np.int64),
                subsegment_id=ATTEND_ALL_SUBSEGMENT_ID,
                seg_end=np.zeros(len(prefix_ids), dtype=bool),
            )
        )
        for branch_idx, branch in enumerate(branches):
            bp = _branch_part(branch, branch_idx)
            n = len(bp["tokens"])
            parts.append(
                dict(
                    tokens=np.asarray(bp["tokens"], dtype=np.int64),
                    loss=bp["loss"],
                    position=np.arange(start_position, start_position + n, dtype=np.int64),
                    subsegment_id=branch_idx,
                    seg_end=bp["seg_end"],
                )
            )

    input_ids = np.concatenate([p["tokens"] for p in parts], 0)
    loss_mask = np.concatenate([p["loss"] for p in parts], 0)
    position_ids = np.concatenate([p["position"] for p in parts], 0)
    seg_ends = np.concatenate([p["seg_end"] for p in parts], 0)

    labels = np.zeros_like(input_ids)
    labels[:-1] = input_ids[1:]
    labels[seg_ends] = eos_id

    loss_mask_shifted = np.zeros_like(loss_mask)
    loss_mask_shifted[:-1] = loss_mask[1:]
    loss_mask_shifted[seg_ends] = loss_mask[seg_ends]
    if multi and loss_token_weighting in ("root_subsegments", "root_subsegments_root_tokens"):
        loss_mask_shifted = loss_mask_shifted / np.sqrt(n_branches)

    token_type_ids = np.isin(input_ids, np.fromiter(image_token_ids, dtype=np.int64)).astype(
        np.int64
    )
    out: Dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_mask_shifted.astype(np.float32),
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
    }
    if multi:
        out["subsegment_ids"] = np.concatenate(
            [np.full(len(p["tokens"]), p["subsegment_id"], dtype=np.int64) for p in parts], 0
        )
    return out
