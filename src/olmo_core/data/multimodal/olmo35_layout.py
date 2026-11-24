"""olmo35 chat layout for Olmo 3.5 multimodal SFT.

A conversation is rendered and labelled exactly as open-instruct's text SFT does it for C11
(``sft_tulu_tokenize_and_truncate_v1`` at ``c11-sft`` @ 2ba2c0044): the Olmo 3.5 chat template
of :data:`OLMO35_TOKENIZER` renders the whole conversation to one string, which is tokenized
once, and a token is a label when it overlaps the character span of an assistant turn. That
span runs from the end of the generation prompt (``<|im_start|>assistant\\n<think>``) to the
end of the turn, so it includes the turn's closing EOS (or ``<|im_end|>`` after a tool call).
Tokenizing the text in one piece matters: the Olmo BPE merges ``<think>``'s final ``>`` with
what follows (``></`` for an empty think block), so per-turn tokenization would give other ids.

Images go at the start of the first user turn, after ``<|im_start|>user\\n``, as in the
:mod:`~olmo_core.data.multimodal.qwen3_layout`; with several images each block is preceded by
the text ``Image {i+1}``. The text on either side of the image blocks keeps the tokens it has
in the text-only rendering, which :func:`encode_conversation` checks, so a text-only
conversation is token-identical to open-instruct's and an image conversation differs from it
only by the inserted blocks.

The example has the layout of the other multimodal sources: position ``t`` predicts token
``t + 1`` (``labels``, ``-100`` where it is not a label) with weight ``loss_masks[t]``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds, build_image_token_ids

__all__ = [
    "OLMO35_TOKENIZER",
    "OLMO35_TOKENIZER_REVISION",
    "OLMO35_TOKEN_IDS",
    "OLMO35_THINK_TAGS",
    "OLMO35_PROMOTED_THINK_IDS",
    "PromotedToken",
    "branch_segments",
    "encode_conversation",
    "image_block_ids",
    "load_olmo35_sft_tokenizer",
    "load_olmo35_tokenizer",
    "normalize_messages",
    "normalize_tools",
    "promote_tokens_into_reserved_slots",
    "require_promoted_think",
    "tokenize_conversation",
]

OLMO35_TOKENIZER = "allenai/dolma2-tokenizer-olmo35"
OLMO35_TOKENIZER_REVISION = "8b9717061fae09d5be814373d189919a62a9a00d"
"""C11's tokenizer (open-instruct ``oc_sft_hero_hillclimb_1895.sh``). Its vocabulary and merges
equal those of ``allenai/dolma2-tokenizer`` @ 5292e5d6, the mid-training tokenizer; it adds the
Olmo 3.5 chat template."""

OLMO35_TOKEN_IDS = Molmo2TokenIds(
    im_start_id=100278,
    im_end_id=100279,
    im_patch_id=100280,
    im_col_id=100281,
    low_res_im_start_id=100282,
    image_placeholder_id=100283,
    im_end_turn_id=100265,
)
"""Image token rows of the Olmo 3.5 multimodal checkpoints, as in mixed mid-training."""

_USER_HEADER = "<|im_start|>user\n"

OLMO35_THINK_TAGS: Tuple[str, str] = ("<think>", "</think>")
"""The tags the current text SFT recipe makes single tokens (open-instruct #1911,
``--reserved_slot_tokens '<think>' '</think>'``)."""

OLMO35_PROMOTED_THINK_IDS: Dict[str, int] = {"<think>": 100256, "</think>": 100266}
"""Their ids: the two lowest unused ``<|extra_id_N|>`` slots of :data:`OLMO35_TOKENIZER` at
:data:`OLMO35_TOKENIZER_REVISION` (``<|extra_id_0|>`` and ``<|extra_id_1|>``), as recorded in the
tokenizer saved beside the frozen text SFT cache (``bf82a8fab8-6068a350-rowaligned/tokenizer``)."""

_RESERVED_SLOT_RE = re.compile(r"<\|extra_id_\d+\|>")


@dataclass(frozen=True)
class PromotedToken:
    """A string made a single token by renaming an unused reserved vocabulary slot."""

    content: str
    token_id: int
    source_ids: Tuple[int, ...]
    """The ids ``content`` tokenized into before promotion; open-instruct seeds the promoted
    embedding row (and output row, when untied) from the mean of these rows before the first
    training step, never on resume."""


def promote_tokens_into_reserved_slots(tokenizer, tokens: Sequence[str]) -> List[PromotedToken]:
    """Make each of ``tokens`` a single token by renaming an unused reserved slot in place.

    A port of open-instruct's ``promote_tokens_into_reserved_slots`` (``dataset_transformation``,
    #1911), so the image rows tokenize ``<think>``/``</think>`` with exactly the ids of the text
    SFT cache. Slots are taken lowest id first among the ``<|extra_id_N|>`` added tokens that are
    neither special tokens nor referenced by the chat template. The promoted tokens are not marked
    special. Tokens that already encode to one id are skipped, so this is idempotent. The
    vocabulary size does not change.

    :param tokenizer: A fast tokenizer; its backend is replaced.
    :param tokens: Strings to promote, in slot order.

    :returns: What was promoted, lowest slot first.

    :raises ValueError: If too few unused slots remain.
    """
    from tokenizers import Tokenizer

    pending = [t for t in tokens if len(tokenizer.encode(t, add_special_tokens=False)) != 1]
    if not pending:
        return []
    source_ids = {t: tuple(tokenizer.encode(t, add_special_tokens=False)) for t in pending}
    spec = json.loads(tokenizer.backend_tokenizer.to_str())
    vocab = spec["model"]["vocab"]
    template = str(tokenizer.chat_template or "")
    in_use = set(tokenizer.all_special_tokens) | set(_RESERVED_SLOT_RE.findall(template))
    free = [
        entry
        for entry in sorted(spec["added_tokens"], key=lambda entry: entry["id"])
        if _RESERVED_SLOT_RE.fullmatch(entry["content"]) and entry["content"] not in in_use
    ]
    if len(free) < len(pending):
        raise ValueError(f"{len(free)} unused reserved slots, {len(pending)} needed for {pending}")
    sizes = (tokenizer.vocab_size, len(tokenizer))
    promoted = []
    for entry, content in zip(free, pending):
        slot_id = vocab.pop(entry["content"], None)
        if slot_id is not None:
            vocab[content] = slot_id
        entry.update(
            content=content,
            special=False,
            normalized=False,
            lstrip=False,
            rstrip=False,
            single_word=False,
        )
        promoted.append(PromotedToken(content, int(entry["id"]), source_ids[content]))
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(spec))
    if (tokenizer.vocab_size, len(tokenizer)) != sizes:
        raise RuntimeError(f"Promoting {pending} changed the vocabulary size")
    for token in promoted:
        if tokenizer.encode(token.content, add_special_tokens=False) != [token.token_id]:
            raise RuntimeError(f"{token.content!r} does not encode to slot {token.token_id}")
    return promoted


def require_promoted_think(tokenizer) -> None:
    """Raise unless ``<think>`` and ``</think>`` are the single tokens of the text SFT cache.

    :raises ValueError: If either tag is several tokens or has another id.
    """
    for tag, token_id in OLMO35_PROMOTED_THINK_IDS.items():
        ids = tokenizer.encode(tag, add_special_tokens=False)
        if ids != [token_id]:
            raise ValueError(
                f"{tag!r} encodes to {ids}, not [{token_id}]: load the tokenizer with "
                "load_olmo35_sft_tokenizer (think tags promoted, as the text SFT cache)"
            )


def load_olmo35_sft_tokenizer(
    name_or_path: str = OLMO35_TOKENIZER,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """The tokenizer of the current text SFT recipe: ``tokenizer.json`` loaded losslessly (its
    ``Split`` pre-tokenizer kept, as
    :func:`~olmo_core.nn.hf.tokenizer.load_tokenizer_losslessly` does for every multimodal
    mixture) with ``<think>`` and ``</think>`` promoted into reserved slots.

    :param name_or_path: Hub id or local directory.
    :param revision: Hub revision; defaults to :data:`OLMO35_TOKENIZER_REVISION` for the hub id.
    :param cache_dir: Hugging Face Hub cache.

    :returns: ``(tokenizer, promoted)``; ``promoted`` is empty when the source already holds the
        promoted tags (the tokenizer saved beside the text SFT cache).
    """
    from olmo_core.nn.hf.tokenizer import load_tokenizer_losslessly

    if revision is None and name_or_path == OLMO35_TOKENIZER:
        revision = OLMO35_TOKENIZER_REVISION
    tokenizer, _, _ = load_tokenizer_losslessly(
        name_or_path, revision=revision, cache_dir=cache_dir
    )
    promoted = promote_tokens_into_reserved_slots(tokenizer, OLMO35_THINK_TAGS)
    require_promoted_think(tokenizer)
    return tokenizer, promoted


def load_olmo35_tokenizer(name_or_path: str = OLMO35_TOKENIZER, *, revision: Optional[str] = None):
    """Load the tokenizer as open-instruct's ``tokenizer_utils.load_tokenizer`` does.

    Transformers 5's ``GPT2Tokenizer`` (what ``AutoTokenizer`` returns for this repo) rebuilds
    its backend with a plain ``ByteLevel`` pre-tokenizer, dropping the ``Split`` regex of
    ``tokenizer.json``, which changes the ids of whitespace runs (``"\\n\\n"`` is one token in
    ``tokenizer.json``, two after the rebuild). open-instruct then loads the file-backed
    ``PreTrainedTokenizerFast`` instead, and so does this.

    :param name_or_path: Hub id or local snapshot directory.
    :param revision: Hub revision; defaults to :data:`OLMO35_TOKENIZER_REVISION` for the hub id.
    """
    from transformers import AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizerFast
    from transformers.utils import hub

    if revision is None and name_or_path == OLMO35_TOKENIZER:
        revision = OLMO35_TOKENIZER_REVISION
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision, use_fast=True)
    if not isinstance(tokenizer, GPT2Tokenizer) or not tokenizer.is_fast:
        return tokenizer
    tokenizer_file = hub.cached_file(
        name_or_path,
        "tokenizer.json",
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    if tokenizer_file is None:
        return tokenizer
    with open(tokenizer_file) as f:
        serialized = json.load(f)
    if json.loads(tokenizer.backend_tokenizer.to_str()).get("pre_tokenizer") == serialized.get(
        "pre_tokenizer"
    ):
        return tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(name_or_path, revision=revision)
    if json.loads(tokenizer.backend_tokenizer.to_str()).get("pre_tokenizer") != serialized.get(
        "pre_tokenizer"
    ):
        raise ValueError(f"Could not keep the tokenizer.json pre-tokenizer of {name_or_path}")
    return tokenizer


def normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Decode JSON-string tool-call arguments, as open-instruct's
    ``_normalize_tool_call_arguments`` does before rendering.

    :param messages: Conversation messages.

    :returns: The messages, shallow-copied where a call's arguments were decoded.

    :raises ValueError: If arguments are not JSON, or decode to something other than an object.
    """
    out: List[Dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not tool_calls:
            out.append(message)
            continue
        new_calls = []
        changed = False
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"tool_calls[].function.arguments is not valid JSON: {arguments!r}"
                    ) from exc
                if decoded is None:
                    decoded = {}
                if not isinstance(decoded, dict):
                    raise ValueError(
                        f"tool_calls[].function.arguments must decode to an object, got: {arguments!r}"
                    )
                assert isinstance(function, dict)
                new_calls.append({**call, "function": {**function, "arguments": decoded}})
                changed = True
            else:
                new_calls.append(call)
        out.append({**message, "tool_calls": new_calls} if changed else message)
    return out


def normalize_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize a tool-schema cell, as open-instruct's ``_normalize_tools_for_chat_template``.

    :param tools: ``None``, a JSON string, a dict or a list of dicts.

    :returns: A list of tool schemas, or ``None`` when there are none.
    """
    if tools is None or tools == "" or (isinstance(tools, float) and np.isnan(tools)):
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"tools must be a JSON-encoded tool schema list, got: {tools!r}"
            ) from exc
        if tools is None or tools == "":
            return None
    if isinstance(tools, dict):
        tools = [tools]
    if not isinstance(tools, list):
        raise TypeError(
            f"tools must be a list, dict, JSON string, or None, got {type(tools).__name__}"
        )
    if not tools:
        return None
    if not all(isinstance(tool, dict) for tool in tools):
        raise TypeError(f"tools must contain JSON-schema dictionaries, got: {tools!r}")
    return tools


def _render(tokenizer, messages, tools, *, add_generation_prompt: bool) -> str:
    text = tokenizer.apply_chat_template(
        conversation=list(messages),
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    assert isinstance(text, str)
    return text


def _assistant_char_spans(tokenizer, messages, tools, rendered: str) -> List[Tuple[int, int]]:
    """Character spans of the assistant turns: from the end of the generation prompt to the end
    of the turn, each a prefix render of ``rendered``."""
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        header = (
            ""
            if index == 0
            else _render(tokenizer, messages[:index], tools, add_generation_prompt=True)
        )
        through = _render(tokenizer, messages[: index + 1], tools, add_generation_prompt=False)
        if not (
            len(header) <= len(through)
            and rendered.startswith(header)
            and rendered.startswith(through)
        ):
            # open-instruct falls back to token counts here; the Olmo 3.5 template is
            # prefix-stable, so a conversation that is not is refused rather than approximated.
            raise ValueError(
                f"assistant turn {index} does not render as a prefix of the conversation"
            )
        spans.append((len(header), len(through)))
    return spans


def tokenize_conversation(
    tokenizer, messages: Sequence[Dict[str, Any]], *, tools: Any = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Token ids and per-token label mask of a text conversation, as open-instruct gives them.

    :param tokenizer: The tokenizer from :func:`load_olmo35_tokenizer`.
    :param messages: Conversation messages (``role``, ``content``, optional ``reasoning_content``
        and ``tool_calls``).
    :param tools: Optional tool schemas (any form :func:`normalize_tools` takes).

    :returns: ``(input_ids, label_mask)``; ``label_mask[t]`` says whether token ``t`` is a label.

    :raises ValueError: If the tokenizer is not fast, or the conversation has no label.
    """
    input_ids, label_mask, _, _ = _tokenize(tokenizer, messages, tools)
    return input_ids, label_mask


def _tokenize(tokenizer, messages, tools):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("The olmo35 layout needs a fast tokenizer (character offsets)")
    messages = normalize_messages(messages)
    tools = normalize_tools(tools)
    rendered = _render(tokenizer, messages, tools, add_generation_prompt=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    offsets = encoded["offset_mapping"]
    spans = _assistant_char_spans(tokenizer, messages, tools, rendered)
    label_mask = np.zeros(len(input_ids), dtype=np.bool_)
    for index, (start, end) in enumerate(offsets):
        # Overlap, not containment, as open-instruct: a token straddling the end of the
        # generation prompt (``></`` of an empty think block) is a label.
        if start != end and any(
            start < span_end and span_start < end for span_start, span_end in spans
        ):
            label_mask[index] = True
    if not label_mask.any():
        raise ValueError("conversation has no assistant label")
    return input_ids, label_mask, offsets, (messages, tools, rendered)


def image_block_ids(
    tokenizer,
    image_grids: Sequence[np.ndarray],
    *,
    token_ids: Molmo2TokenIds = OLMO35_TOKEN_IDS,
) -> List[int]:
    """Image token blocks for the first user turn, ``Image {i+1}`` before each when there are
    several (as :func:`~olmo_core.data.multimodal.qwen3_layout.multi_image_prefix_ids`).

    :param tokenizer: The :data:`OLMO35_TOKENIZER` tokenizer.
    :param image_grids: ``(resized_h, resized_w, crop_h, crop_w)`` per image.
    :param token_ids: Image token IDs of the model.
    """
    ids: List[int] = []
    multi = len(image_grids) > 1
    for index, grid in enumerate(image_grids):
        if multi:
            ids.extend(tokenizer.encode(f"Image {index + 1}", add_special_tokens=False))
        resized_h, resized_w, height, width = (int(grid[i]) for i in range(4))
        ids.extend(build_image_token_ids(resized_h, resized_w, height, width, token_ids=token_ids))
    return ids


def _image_char_offset(tokenizer, messages, tools, rendered: str) -> int:
    """Character offset of the first user turn's content in ``rendered``."""
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if first_user is None:
        raise ValueError("an image conversation needs a user turn")
    # Render through the first user turn (transformers refuses an empty conversation, so the
    # turns before it cannot be rendered alone) and find that turn's header.
    through = _render(tokenizer, messages[: first_user + 1], tools, add_generation_prompt=False)
    start = through.rfind(_USER_HEADER)
    content = messages[first_user]["content"]
    if not isinstance(content, str):
        content = "".join(part["text"] for part in content)
    if (
        start < 0
        or not rendered.startswith(through)
        or through[start + len(_USER_HEADER) :] != content + "<|im_end|>\n"
    ):
        raise ValueError("cannot locate the first user turn's content")
    return start + len(_USER_HEADER)


def _strip_first_user_leading_whitespace(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy of ``messages`` with the first user turn's leading whitespace removed: the image
    blocks go at the start of that turn, and leading whitespace would merge with the user header
    into one token, leaving no token boundary for them."""
    out = [dict(m) for m in messages]
    for m in out:
        if m["role"] == "user":
            if isinstance(m["content"], str):
                m["content"] = m["content"].lstrip()
            elif m["content"]:
                parts = [dict(part) for part in m["content"]]
                parts[0]["text"] = parts[0]["text"].lstrip()
                m["content"] = parts
            break
    return out


def _image_token_index(tokenizer, messages, tools, rendered, input_ids, offsets) -> int:
    """Token index where the image blocks go (the start of the first user turn's content),
    checking that the text on either side keeps its tokens when the blocks split it."""
    cut = _image_char_offset(tokenizer, messages, tools, rendered)
    at = next((i for i, (start, _) in enumerate(offsets) if start >= cut), len(offsets))
    if (at > 0 and offsets[at - 1][1] != cut) or (at < len(offsets) and offsets[at][0] != cut):
        raise ValueError("no token boundary at the start of the first user turn")
    left = tokenizer(rendered[:cut], add_special_tokens=False)["input_ids"]
    right = tokenizer(rendered[cut:], add_special_tokens=False)["input_ids"]
    if left + right != np.asarray(input_ids).tolist():
        raise ValueError("the text around the image blocks tokenizes differently on its own")
    return at


def branch_segments(
    tokenizer, turns: Sequence[Tuple[str, str]]
) -> Tuple[List[int], List[Tuple[List[int], List[int]]]]:
    """One (question, answer) conversation in the olmo35 layout, cut into the pieces
    :func:`~olmo_core.data.multimodal.sequence_builder.build_branched_sequence` takes.

    The conversation is rendered by the chat template and tokenized once, with
    :func:`tokenize_conversation`'s labels (open-instruct's rule), then cut where the image blocks
    go: the head (the default system turn and ``<|im_start|>user\\n``) comes back separately so
    the caller can put the image blocks after it, and the rest alternates non-label context and
    label response runs. With no reasoning, every assistant turn renders as
    ``<|im_start|>assistant\\n<think></think>{answer}<|endoftext|>``: ``<think>`` closes the
    generation prompt and is context, ``</think>``, the answer and the EOS are labels, exactly
    as in the text SFT rows. The final EOS is dropped from the last response, because the
    builder makes EOS the target of the branch's last token itself; an earlier turn keeps its EOS
    as a label, as open-instruct trains it.

    :param tokenizer: The tokenizer from :func:`load_olmo35_sft_tokenizer`.
    :param turns: ``(question, answer)`` pairs of one sequential conversation.

    :returns: ``(head_ids, [(context_ids, response_ids), ...])``; ``head_ids`` concatenated with
        every segment, plus the final EOS, is the text-only rendering's token sequence.

    :raises ValueError: If the conversation cannot be cut as described.
    """
    messages: List[Dict[str, Any]] = []
    for index, (question, answer) in enumerate(turns):
        if index == 0:
            # The image blocks go at the start of the first question, so leading whitespace there
            # would merge with the user header into one token and leave no boundary to cut at
            # (24% of correction QA rows open with "\n"). Molmo2's layout drops it too.
            question = question.lstrip()
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    input_ids, label_mask, offsets, (messages, tools, rendered) = _tokenize(
        tokenizer, messages, None
    )
    at = _image_token_index(tokenizer, messages, tools, rendered, input_ids, offsets)
    if label_mask[:at].any():
        raise ValueError("a label precedes the first user turn")
    ids = input_ids.tolist()
    head, rest, mask = ids[:at], ids[at:], label_mask[at:].tolist()
    segments: List[Tuple[List[int], List[int]]] = []
    i = 0
    while i < len(rest):
        j = i
        while j < len(rest) and not mask[j]:
            j += 1
        k = j
        while k < len(rest) and mask[k]:
            k += 1
        if j == i or k == j:
            raise ValueError("expected alternating context and response runs")
        segments.append((rest[i:j], rest[j:k]))
        i = k
    eos = tokenizer.eos_token_id
    last_context, last_response = segments[-1]
    if last_response[-1] != eos or len(last_response) < 2:
        raise ValueError("the last assistant turn does not end with EOS")
    segments[-1] = (last_context, last_response[:-1])
    return head, segments


def encode_conversation(
    tokenizer,
    messages: Sequence[Dict[str, Any]],
    *,
    tools: Any = None,
    image_ids: Sequence[int] = (),
    token_ids: Molmo2TokenIds = OLMO35_TOKEN_IDS,
) -> Dict[str, np.ndarray]:
    """Encode a conversation, with optional image blocks, as a multimodal training example.

    :param tokenizer: The tokenizer from :func:`load_olmo35_tokenizer`.
    :param messages: Conversation messages.
    :param tools: Optional tool schemas.
    :param image_ids: Image blocks from :func:`image_block_ids`, placed at the start of the first
        user turn. Empty for a text-only conversation.
    :param token_ids: Image token IDs, for ``token_type_ids``.

    :returns: ``input_ids``, ``labels``, ``loss_masks``, ``position_ids`` and ``token_type_ids``,
        one entry per token.

    :raises ValueError: If the text around the image blocks would tokenize differently from the
        text-only rendering.
    """
    if len(image_ids):
        messages = _strip_first_user_leading_whitespace(messages)
    input_ids, label_mask, offsets, (messages, tools, rendered) = _tokenize(
        tokenizer, messages, tools
    )
    if len(image_ids):
        at = _image_token_index(tokenizer, messages, tools, rendered, input_ids, offsets)
        image = np.asarray(image_ids, dtype=np.int64)
        input_ids = np.concatenate([input_ids[:at], image, input_ids[at:]])
        label_mask = np.concatenate(
            [label_mask[:at], np.zeros(len(image), dtype=np.bool_), label_mask[at:]]
        )
    targets = label_mask[1:]
    labels = np.full(len(input_ids), -100, dtype=np.int64)
    labels[:-1] = np.where(targets, input_ids[1:], -100)
    loss_masks = np.zeros(len(input_ids), dtype=np.float32)
    loss_masks[:-1] = targets
    image_rows = np.fromiter(token_ids.image_token_ids, dtype=np.int64)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_masks,
        "position_ids": np.arange(len(input_ids), dtype=np.int64),
        "token_type_ids": np.isin(input_ids, image_rows).astype(np.int64),
    }
