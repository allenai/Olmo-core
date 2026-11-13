"""olmo35 layout under the current text SFT recipe (think tags promoted), against open-instruct
(fixture from ``src/scripts/make_olmo35_c12_fixture.py``), and the ``olmo35`` image-row
serializer of :func:`~olmo_core.data.multimodal.message_sequence.encode_sft_example`."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from olmo_core.data.multimodal.olmo35_layout import (
    OLMO35_PROMOTED_THINK_IDS,
    OLMO35_TOKEN_IDS,
    OLMO35_TOKENIZER,
    OLMO35_TOKENIZER_REVISION,
    branch_segments,
    load_olmo35_sft_tokenizer,
    load_olmo35_tokenizer,
    require_promoted_think,
    tokenize_conversation,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "olmo35_c12" / "rows.json.gz"
TEXT_SFT_CACHE_TOKENIZER = (
    "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/a7/cache-neptune/numpy_sft/"
    "bf82a8fab8-6068a350-rowaligned/tokenizer/tokenizer.json"
)
EOS = 100257
THINK, END_THINK = OLMO35_PROMOTED_THINK_IDS["<think>"], OLMO35_PROMOTED_THINK_IDS["</think>"]


@pytest.fixture(scope="module")
def fixture():
    with gzip.open(FIXTURE, "rt") as f:
        data = json.load(f)
    assert (data["tokenizer"], data["revision"]) == (OLMO35_TOKENIZER, OLMO35_TOKENIZER_REVISION)
    return data


@pytest.fixture(scope="module")
def loaded():
    pytest.importorskip("transformers")
    try:
        return load_olmo35_sft_tokenizer()
    except OSError as exc:
        pytest.skip(f"{OLMO35_TOKENIZER} unavailable: {exc}")


@pytest.fixture(scope="module")
def tokenizer(loaded):
    return loaded[0]


def test_promotion_matches_open_instruct(loaded, fixture):
    _, promoted = loaded
    assert [
        {"content": t.content, "token_id": t.token_id, "source_ids": list(t.source_ids)}
        for t in promoted
    ] == fixture["promoted"]
    assert {t.content: t.token_id for t in promoted} == OLMO35_PROMOTED_THINK_IDS


def test_vocabulary_equals_the_text_sft_cache_tokenizer(tokenizer):
    if not os.path.exists(TEXT_SFT_CACHE_TOKENIZER):
        pytest.skip("text SFT cache not mounted")
    with open(TEXT_SFT_CACHE_TOKENIZER) as f:
        cache = json.load(f)
    ours = json.loads(tokenizer.backend_tokenizer.to_str())
    assert ours["model"]["vocab"] == cache["model"]["vocab"]
    assert ours["pre_tokenizer"] == cache["pre_tokenizer"]
    assert [(e["id"], e["content"]) for e in ours["added_tokens"]] == [
        (e["id"], e["content"]) for e in cache["added_tokens"]
    ]


def test_text_rows_match_open_instruct(tokenizer, fixture):
    rows = fixture["rows"]
    assert {row["source"] for row in rows} == {"dolci", "nemo", "sci", "if"}
    for row in rows:
        input_ids, label_mask = tokenize_conversation(
            tokenizer, row["messages"], tools=row["tools"]
        )
        assert input_ids.tolist() == row["input_ids"], row["id"]
        assert label_mask.astype(int).tolist() == row["label_mask"], row["id"]


def test_unpromoted_tokenizer_is_refused():
    pytest.importorskip("transformers")
    try:
        plain = load_olmo35_tokenizer()
    except OSError as exc:
        pytest.skip(f"{OLMO35_TOKENIZER} unavailable: {exc}")
    with pytest.raises(ValueError, match="load_olmo35_sft_tokenizer"):
        require_promoted_think(plain)


TURNS = [("What is in the picture?", "A red bicycle."), ("What color is it?", "Red.")]


def test_branch_segments_reassemble_the_text_rendering(tokenizer):
    head, segments = branch_segments(tokenizer, TURNS)
    messages = []
    for q, a in TURNS:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    ids, labels = tokenize_conversation(tokenizer, messages)
    flat = list(head) + [t for c, r in segments for t in c + r] + [EOS]
    assert flat == ids.tolist()
    assert tokenizer.decode(head).endswith("<|im_start|>user\n")
    assert "You are Olmo" in tokenizer.decode(head)
    assert len(segments) == 2
    for context, response in segments:
        assert context[-1] == THINK  # the generation prompt ends with <think>
        assert response[0] == END_THINK  # thinking off: </think> opens the labelled answer
    # An earlier turn keeps its EOS as a label; the last turn's EOS is the builder's target.
    assert segments[0][1][-1] == EOS and segments[1][1][-1] != EOS
    expected = labels[len(head) :].tolist()
    assert not labels[: len(head)].any()
    got: list = []
    for c, r in segments:
        got += [0] * len(c) + [1] * len(r)
    assert got + [1] == [int(x) for x in expected]


def _image():
    from PIL import Image

    return Image.fromarray(np.random.RandomState(0).randint(0, 255, (224, 336, 3), dtype=np.uint8))


def _encode(tokenizer, turns, **kwargs):
    from olmo_core.data.multimodal.message_sequence import encode_sft_example

    return encode_sft_example(
        tokenizer,
        _image(),
        turns,
        max_crops=2,
        token_ids=OLMO35_TOKEN_IDS,
        message_format="olmo35",
        seed=0,
        **kwargs,
    )


def test_image_example_is_the_text_rendering_plus_image_blocks(tokenizer):
    example = _encode(tokenizer, [TURNS])  # one branch, two sequential turns
    ids = example["input_ids"]
    image = np.isin(ids, np.fromiter(OLMO35_TOKEN_IDS.image_token_ids, dtype=np.int64))
    assert image.any() and example["token_type_ids"].astype(bool).tolist() == image.tolist()
    head, segments = branch_segments(tokenizer, TURNS)
    text = [t for c, r in segments for t in c + r]
    assert ids[~image].tolist() == list(head) + text
    first_image = int(np.flatnonzero(image)[0])
    assert first_image == len(head)
    # Every response token weighs 1 and is predicted from the position before it; the branch's
    # last token predicts EOS.
    labels, loss = example["labels"], example["loss_masks"]
    assert set(np.unique(loss)) <= {0.0, 1.0}
    supervised = labels[loss > 0]
    assert supervised[-1] == EOS
    assert (supervised == END_THINK).sum() == 2
    assert THINK not in supervised.tolist()
    n_labels = sum(len(r) for _, r in segments) + 1
    assert int((loss > 0).sum()) == n_labels
    assert "subsegment_ids" not in example and "document_ids" not in example


def test_message_weight_scales_the_loss(tokenizer):
    example = _encode(tokenizer, [TURNS[:1]], message_weight=0.2)
    assert np.allclose(np.unique(example["loss_masks"]), [0.0, 0.2])


def test_several_annotations_become_turns_of_one_conversation(tokenizer):
    branches = [[("Point to the cat.", "<point>1</point>")], [("Count the dogs.", "3")]]
    example = _encode(tokenizer, branches)
    assert "document_ids" not in example and "subsegment_ids" not in example
    image_ids = np.fromiter(OLMO35_TOKEN_IDS.image_token_ids, dtype=np.int64)
    ids = example["input_ids"]
    image = np.isin(ids, image_ids)
    # The image once, then both annotations as sequential turns (in the shuffled order).
    assert np.flatnonzero(np.diff(image.astype(int)) != 0).size == 2
    head, segments = branch_segments(tokenizer, [b[0] for b in branches])
    text = ids[~image].tolist()
    assert text[: len(head)] == head
    supervised = example["labels"][example["loss_masks"] > 0]
    assert (supervised == END_THINK).sum() == 2
    assert (supervised == EOS).sum() == 2  # the first turn's EOS label and the final target
    assert set(np.unique(example["loss_masks"])) <= {0.0, 1.0}


def test_annotations_as_documents_share_the_crops(tokenizer):
    branches = [[("Point to the cat.", "<point>1</point>")], [("Count the dogs.", "3")]]
    example = _encode(tokenizer, branches, branches_as_documents=True)
    documents = example["document_ids"]
    assert sorted(set(documents.tolist())) == [0, 1]
    image_ids = np.fromiter(OLMO35_TOKEN_IDS.image_token_ids, dtype=np.int64)
    for doc in (0, 1):
        in_doc = documents == doc
        assert np.isin(example["input_ids"][in_doc], image_ids).any()
        assert example["labels"][in_doc][example["loss_masks"][in_doc] > 0][-1] == EOS
    pooled = example["pooled_patches_idx"]
    assert len(pooled) % 2 == 0
    np.testing.assert_array_equal(pooled[: len(pooled) // 2], pooled[len(pooled) // 2 :])


@pytest.mark.parametrize("lead", ["\n", "\n\n", " ", "\t "])
def test_leading_whitespace_in_the_first_question_is_dropped(tokenizer, lead):
    # 24% of correction QA rows open their first question with "\n", which merged with the user
    # header into one token and left no boundary for the image blocks (the first full run stopped
    # on correction_qa_multi_only_max5[30777]).
    turns = [(lead + TURNS[0][0], TURNS[0][1]), (" " + TURNS[1][0], TURNS[1][1])]
    head, segments = branch_segments(tokenizer, turns)
    stripped = [(TURNS[0][0], TURNS[0][1]), (" " + TURNS[1][0], TURNS[1][1])]
    assert (head, segments) == branch_segments(tokenizer, stripped)
    example = _encode(tokenizer, [turns])
    assert int(example["token_type_ids"].sum()) > 0
