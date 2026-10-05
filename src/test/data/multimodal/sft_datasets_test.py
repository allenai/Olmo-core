"""CPU tests for the parquet-backed image SFT datasets (MMFineReason, FineVision):
answer parsing, placeholder handling, truncation, bad-row skipping, and end-to-end
example construction from a synthetic parquet shard.

These sources assemble through the shared stage-2 encode path
(:func:`~olmo_core.data.multimodal.message_sequence.encode_sft_example`), so the
sequence layout matches the rest of the mixture: no BOS, the image token block inside
the first user turn, and multi-turn rows as ONE sequential branch with no loss at
intermediate turn ends (mm_olmo semantics).
"""

import io
import zlib
from typing import Dict, List

import numpy as np
import pytest

from olmo_core.data.multimodal import extract_answer_text
from olmo_core.data.multimodal.sft_common import (
    count_image_placeholders,
    decode_pil_image,
    get_example_with_skip,
    strip_image_placeholders,
    truncate_example,
)
from olmo_core.nn.vision.molmo2_tokens import (
    IM_PATCH_ID,
    LOW_RES_IM_START_ID,
    N_PATCHES_SQ,
    PATCH_DIM,
    Molmo2TokenIds,
)

# ---------------------------------------------------------------------------
# MMFineReason answer parsing
# ---------------------------------------------------------------------------


def test_extract_answer_think_answer_structure():
    # The dominant tagged shape: private reasoning + user-facing answer. Only the latter
    # is supervised.
    raw = "<think>Let me work this out.\nx = 9.</think><answer>The area is 9.</answer>"
    assert extract_answer_text(raw) == "The area is 9."


@pytest.mark.parametrize("sep", ["", "\n", "\n\n", " "])
def test_extract_answer_separator_variants(sep):
    raw = f"<think>reasoning</think>{sep}<answer>42</answer>"
    assert extract_answer_text(raw) == "42"


def test_extract_answer_untagged_returns_whole_text():
    # ~half the corpus has no tags at all: the entire answer is the target.
    raw = "The perimeter is 48, so the side is 11.15. Therefore the area is 65.12."
    assert extract_answer_text(raw) == raw


def test_extract_answer_multiline_answer_preserved():
    raw = "<think>t</think><answer>Step 1: a\nStep 2: b</answer>"
    assert extract_answer_text(raw) == "Step 1: a\nStep 2: b"


def test_extract_answer_uses_last_block():
    # A reasoning trace that *mentions* <answer> must not hijack the extraction.
    raw = "<think>I could write <answer>wrong</answer> here</think><answer>right</answer>"
    assert extract_answer_text(raw) == "right"


def test_extract_answer_think_only_drops_trace():
    assert extract_answer_text("<think>hidden</think>visible") == "visible"


def test_extract_answer_unpaired_tags_are_stripped():
    # Malformed (not seen in the corpus, but must not leak markup into the loss).
    assert extract_answer_text("no close <answer>tail") == "no close tail"
    assert extract_answer_text("head</answer>") == "head"


def test_extract_answer_empty_block_falls_through():
    # An empty <answer></answer> is unusable; fall back to the de-thought remainder.
    assert extract_answer_text("<think>t</think><answer>   </answer>rest") == "rest"


@pytest.mark.parametrize("raw", [None, "", "   ", "<think>only reasoning</think>"])
def test_extract_answer_empty_inputs(raw):
    assert extract_answer_text(raw) == ""


# ---------------------------------------------------------------------------
# <image> placeholder handling
# ---------------------------------------------------------------------------


def test_strip_image_placeholders():
    assert strip_image_placeholders("<image>What is shown?") == "What is shown?"
    assert strip_image_placeholders("<image>\nDescribe it.") == "Describe it."
    # mid-sentence marker (rare) is dropped, leaving the prose intact
    assert strip_image_placeholders("Look at <image> and answer.") == "Look at and answer."
    assert strip_image_placeholders("<image><image>Two.") == "Two."
    assert strip_image_placeholders(None) == ""


def test_count_image_placeholders():
    assert count_image_placeholders("<image>a<image>") == 2
    assert count_image_placeholders("none") == 0
    assert count_image_placeholders(None) == 0


# ---------------------------------------------------------------------------
# truncate_example
# ---------------------------------------------------------------------------


def _tiny_image(w=40, h=30, seed=0):
    from PIL import Image

    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8))


def test_truncate_example_trims_text_and_keeps_images():
    seq = {
        "input_ids": np.arange(10, dtype=np.int64),
        "loss_masks": np.ones(10, dtype=np.float32),
        "images": np.zeros((2, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
    }
    out = truncate_example(seq, 6)
    assert len(out["input_ids"]) == 6 and len(out["loss_masks"]) == 6
    assert out["images"].shape == (2, N_PATCHES_SQ, PATCH_DIM)  # untouched
    # no-op when already short enough
    assert truncate_example(seq, 100)["input_ids"].shape == (10,)


def test_truncate_example_refuses_to_cut_image_tokens():
    seq = {
        "input_ids": np.array([1, 2, IM_PATCH_ID, IM_PATCH_ID], dtype=np.int64),
        "loss_masks": np.ones(4, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="im_patch"):
        truncate_example(seq, 2)

    token_ids = Molmo2TokenIds()
    seq["input_ids"] = np.array([1, 2, token_ids.im_end_id, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="image-structural"):
        truncate_example(seq, 2, image_token_ids=token_ids.image_token_ids)


def test_truncate_example_refuses_to_drop_all_loss():
    seq = {
        "input_ids": np.arange(8, dtype=np.int64),
        "loss_masks": np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="loss tokens"):
        truncate_example(seq, 4)


# ---------------------------------------------------------------------------
# decode_pil_image
# ---------------------------------------------------------------------------


def test_decode_pil_image_forms(tmp_path):
    from PIL import Image

    img = _tiny_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    assert decode_pil_image(img) is img  # already decoded
    assert decode_pil_image({"bytes": buf.getvalue(), "path": None}).size == img.size
    p = tmp_path / "x.png"
    img.save(p)
    assert decode_pil_image({"bytes": None, "path": str(p)}).size == img.size
    assert decode_pil_image(str(p)).size == img.size
    with pytest.raises(ValueError):
        decode_pil_image({"bytes": None, "path": None})
    with pytest.raises(TypeError):
        decode_pil_image(123)
    assert isinstance(decode_pil_image(img), Image.Image)


# ---------------------------------------------------------------------------
# Bad-row skipping
# ---------------------------------------------------------------------------


class _FlakyDataset:
    """Fails on the rows listed in ``bad``."""

    def __init__(self, size: int, bad):
        self.size, self.bad, self._warned = size, set(bad), 0
        self.epochs = []

    def _build(self, i: int, epoch: int = 0):
        self.epochs.append(epoch)
        if i in self.bad:
            raise ValueError(f"row {i} is broken")
        return {"input_ids": np.array([i], dtype=np.int64)}


def test_get_example_with_skip_advances_over_bad_rows():
    ds = _FlakyDataset(10, bad={3, 4})
    assert get_example_with_skip(ds, 3, 10, epoch=7)["input_ids"].tolist() == [5]
    assert ds.epochs == [7, 7, 7]
    assert get_example_with_skip(ds, 0, 10)["input_ids"].tolist() == [0]


def test_get_example_with_skip_wraps_around():
    ds = _FlakyDataset(4, bad={3})
    assert get_example_with_skip(ds, 3, 4)["input_ids"].tolist() == [0]


def test_get_example_with_skip_raises_when_all_bad():
    ds = _FlakyDataset(4, bad={0, 1, 2, 3})
    with pytest.raises(RuntimeError, match="unusable"):
        get_example_with_skip(ds, 0, 4)


# ---------------------------------------------------------------------------
# End-to-end: synthetic parquet -> packed example
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal stand-in for the Molmo2 chat tokenizer (offline, deterministic).

    Encodes each whitespace-separated word as one id, kept well below the Molmo2
    special-token range so it can't collide with image/turn markers. Uses CRC32 rather than
    ``hash()`` so ids are stable across processes (``hash`` of a str is salted).
    """

    eos_token_id = 151643
    bos_token_id = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        content = messages[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [(zlib.crc32(w.encode()) % 10000) + 100 for w in text.split()]


def _write_parquet(path, rows: List[Dict], columns):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({c: [r[c] for r in rows] for c in columns})
    pq.write_table(table, path)


def _png_bytes(seed=0):
    buf = io.BytesIO()
    _tiny_image(seed=seed).save(buf, format="PNG")
    return buf.getvalue()


def test_mmfinereason_dataset_end_to_end(tmp_path):
    from olmo_core.data.multimodal import MMFineReasonDatasetConfig

    d = tmp_path / "mmfr"
    d.mkdir()
    rows = [
        {
            "question": "<image>What is the area?",
            "original_answer": "<think>long trace</think><answer>It is 9.</answer>",
            "image": {"bytes": _png_bytes(0), "path": None},
            "source": "MMR1",
            "pass_rate": 0.25,
            "is_consistent": True,
        },
        {
            "question": "<image>\nDescribe the figure.",
            "original_answer": "A triangle with a bisector.",
            "image": {"bytes": _png_bytes(1), "path": None},
            "source": "GameQA-140K",
            "pass_rate": 0.75,
            "is_consistent": False,
        },
    ]
    cols = ["question", "original_answer", "image", "source", "pass_rate", "is_consistent"]
    _write_parquet(str(d / "train-00000.parquet"), rows, cols)

    ds = MMFineReasonDatasetConfig(dataset_path=str(d), max_crops=2).build(_FakeTokenizer())
    assert len(ds) == 2

    ex = ds[0]
    for key in (
        "input_ids",
        "labels",
        "loss_masks",
        "position_ids",
        "token_type_ids",
        "images",
        "pooled_patches_idx",
    ):
        assert key in ex, key
    # the <im_patch>/pooled-feature invariant must hold for a real built example
    n_patch = int((ex["input_ids"] == IM_PATCH_ID).sum())
    assert n_patch == int((ex["pooled_patches_idx"] >= 0).any(axis=-1).sum()) > 0
    # stage-2 qwen3 layout: no BOS — the sequence must not start with an EOS/BOS token,
    # and the image block sits inside the first user turn (after the user header word).
    assert ex["input_ids"][0] != _FakeTokenizer.eos_token_id
    assert ex["input_ids"][1] == LOW_RES_IM_START_ID  # [user-header][<low_res_im_start>...
    # supervision is the parsed <answer> content only: "It is 9." -> 3 word tokens + EOS
    assert float(ex["loss_masks"].sum()) == 4.0
    assert ex["labels"][-1] == _FakeTokenizer.eos_token_id
    assert ex["images"].shape[1:] == (N_PATCHES_SQ, PATCH_DIM)


def test_mmfinereason_dataset_filters(tmp_path):
    from olmo_core.data.multimodal import MMFineReasonDatasetConfig

    d = tmp_path / "mmfr2"
    d.mkdir()
    rows = [
        {
            "question": "<image>q1",
            "original_answer": "a1",
            "image": {"bytes": _png_bytes(0), "path": None},
            "source": "MMR1",
            "pass_rate": 0.25,
            "is_consistent": True,
        },
        {
            "question": "<image>q2",
            "original_answer": "a2",
            "image": {"bytes": _png_bytes(1), "path": None},
            "source": "BMMR",
            "pass_rate": 0.75,
            "is_consistent": False,
        },
    ]
    cols = ["question", "original_answer", "image", "source", "pass_rate", "is_consistent"]
    _write_parquet(str(d / "train-00000.parquet"), rows, cols)
    tok = _FakeTokenizer()

    assert len(MMFineReasonDatasetConfig(dataset_path=str(d), sources=["MMR1"]).build(tok)) == 1
    assert len(MMFineReasonDatasetConfig(dataset_path=str(d), max_pass_rate=0.5).build(tok)) == 1
    assert (
        len(MMFineReasonDatasetConfig(dataset_path=str(d), require_consistent=True).build(tok)) == 1
    )
    assert len(MMFineReasonDatasetConfig(dataset_path=str(d)).build(tok)) == 2


def test_finevision_dataset_end_to_end(tmp_path):
    from olmo_core.data.multimodal import FineVisionDatasetConfig

    d = tmp_path / "vwi"
    d.mkdir()
    rows = [
        {
            "images": [{"bytes": _png_bytes(0), "path": None}],
            "texts": [{"user": "<image>\nDescribe the figure.", "assistant": "A triangle ABC."}],
            "relevance_min": 4,
            "visual_dependency_min": 5,
            "formatting_min": 4,
            "image_correspondence_min": 5,
        },
        {  # two images, two turns -> exercises the multi-image / multi-turn paths
            "images": [
                {"bytes": _png_bytes(1), "path": None},
                {"bytes": _png_bytes(2), "path": None},
            ],
            "texts": [
                {"user": "<image><image>Compare them.", "assistant": "They differ."},
                {"user": "Why?", "assistant": "Because of colour."},
            ],
            "relevance_min": 2,
            "visual_dependency_min": 2,
            "formatting_min": 2,
            "image_correspondence_min": 2,
        },
    ]
    cols = [
        "images",
        "texts",
        "relevance_min",
        "visual_dependency_min",
        "formatting_min",
        "image_correspondence_min",
    ]
    _write_parquet(str(d / "train-00000.parquet"), rows, cols)

    ds = FineVisionDatasetConfig(dataset_path=str(d), max_crops=2).build(_FakeTokenizer())
    assert len(ds) == 2

    ex = ds[0]
    n_patch = int((ex["input_ids"] == IM_PATCH_ID).sum())
    assert n_patch == int((ex["pooled_patches_idx"] >= 0).any(axis=-1).sum()) > 0
    # "A triangle ABC." -> 3 word tokens + EOS
    assert float(ex["loss_masks"].sum()) == 4.0

    two = ds[1]
    # both images spliced in, invariant still holds, and both replies are supervised
    assert two["images"].shape[0] == 2 * ex["images"].shape[0]
    n_patch2 = int((two["input_ids"] == IM_PATCH_ID).sum())
    assert n_patch2 == int((two["pooled_patches_idx"] >= 0).any(axis=-1).sum()) == 2 * n_patch
    # One sequential conversation branch, mm_olmo turn semantics: "They differ." (2)
    # + "Because of colour." (3) + final EOS (1); the intermediate turn end carries no
    # loss (its <|im_end|> belongs to the next turn's user context).
    assert float(two["loss_masks"].sum()) == 6.0
    # multi-image rows get "Image 1"/"Image 2" text prefixes before each block
    image_1_ids = _FakeTokenizer().encode("Image 1")
    assert two["input_ids"].tolist()[1 : 1 + len(image_1_ids)] == image_1_ids


def test_finevision_quality_filter(tmp_path):
    from olmo_core.data.multimodal import FineVisionDatasetConfig

    d = tmp_path / "vwi2"
    d.mkdir()
    rows = [
        {
            "images": [{"bytes": _png_bytes(i), "path": None}],
            "texts": [{"user": "<image>q", "assistant": "a"}],
            "relevance_min": r,
            "visual_dependency_min": v,
            "formatting_min": 3,
            "image_correspondence_min": 3,
        }
        for i, (r, v) in enumerate([(5, 5), (2, 5), (5, 1)])
    ]
    cols = [
        "images",
        "texts",
        "relevance_min",
        "visual_dependency_min",
        "formatting_min",
        "image_correspondence_min",
    ]
    _write_parquet(str(d / "train-00000.parquet"), rows, cols)
    tok = _FakeTokenizer()

    assert len(FineVisionDatasetConfig(dataset_path=str(d)).build(tok)) == 3
    assert len(FineVisionDatasetConfig(dataset_path=str(d), min_relevance=4).build(tok)) == 2
    assert (
        len(
            FineVisionDatasetConfig(
                dataset_path=str(d), min_relevance=4, min_visual_dependency=4
            ).build(tok)
        )
        == 1
    )


def test_sft_examples_pack_together(tmp_path):
    """Examples from both new sources must pack with each other (shared invariants)."""
    from olmo_core.data.multimodal import (
        FineVisionDatasetConfig,
        MMFineReasonDatasetConfig,
        pack_examples,
    )

    d = tmp_path / "both"
    d.mkdir()
    _write_parquet(
        str(d / "a.parquet"),
        [
            {
                "question": "<image>q",
                "original_answer": "<think>t</think><answer>ans here</answer>",
                "image": {"bytes": _png_bytes(0), "path": None},
                "source": "MMR1",
                "pass_rate": 0.5,
                "is_consistent": True,
            }
        ],
        ["question", "original_answer", "image", "source", "pass_rate", "is_consistent"],
    )
    dv = tmp_path / "bothv"
    dv.mkdir()
    _write_parquet(
        str(dv / "b.parquet"),
        [
            {
                "images": [{"bytes": _png_bytes(1), "path": None}],
                "texts": [{"user": "<image>u", "assistant": "reply text"}],
                "relevance_min": 4,
                "visual_dependency_min": 4,
                "formatting_min": 4,
                "image_correspondence_min": 4,
            }
        ],
        [
            "images",
            "texts",
            "relevance_min",
            "visual_dependency_min",
            "formatting_min",
            "image_correspondence_min",
        ],
    )
    tok = _FakeTokenizer()
    a = MMFineReasonDatasetConfig(dataset_path=str(d), max_crops=2).build(tok)[0]
    b = FineVisionDatasetConfig(dataset_path=str(dv), max_crops=2).build(tok)[0]

    packed = pack_examples([a, b])
    n_patch = int((packed["input_ids"] == IM_PATCH_ID).sum())
    n_pooled = int((packed["pooled_patches_idx"] >= 0).any(axis=-1).sum())
    assert n_patch == n_pooled
    assert packed["pooled_patches_idx"].max() < packed["images"].shape[0] * N_PATCHES_SQ
    assert set(packed["example_ids"].tolist()) == {0, 1}


def test_finevision_config_name_resolves_path(tmp_path):
    """`config_name` is joined onto `root`; `dataset_path` overrides both."""
    from olmo_core.data.multimodal import FineVisionDatasetConfig
    from olmo_core.data.multimodal.finevision import FINEVISION_ROOT

    cfg = FineVisionDatasetConfig(config_name="mavis_math_rule_geo")
    assert cfg.resolved_path() == f"{FINEVISION_ROOT}/mavis_math_rule_geo"
    # parentheses in a config name must survive untouched (geo170k(qa) etc.)
    assert FineVisionDatasetConfig(config_name="geo170k(qa)", root="/r").resolved_path() == (
        "/r/geo170k(qa)"
    )
    assert FineVisionDatasetConfig(dataset_path="/explicit").resolved_path() == "/explicit"
    # the default is visualwebinstruct, incl. via the pinned subclass
    from olmo_core.data.multimodal import VisualWebInstructDatasetConfig

    assert VisualWebInstructDatasetConfig().config_name == "visualwebinstruct(filtered)"


def test_finevision_without_image_placeholder(tmp_path):
    """geo170k / mavis_math_metagen rows carry NO `<image>` marker, yet still have an image.

    The image block must still be emitted (it comes from the `images` column, not the
    marker), and the invariant must hold identically to the marker-bearing layout.
    """
    from olmo_core.data.multimodal import FineVisionDatasetConfig

    d = tmp_path / "nomarker"
    d.mkdir()
    cols = [
        "images",
        "texts",
        "relevance_min",
        "visual_dependency_min",
        "formatting_min",
        "image_correspondence_min",
    ]
    rows = [
        {  # no <image> anywhere in the user text
            "images": [{"bytes": _png_bytes(0), "path": None}],
            "texts": [{"user": "Do any points other than M lie on XY?", "assistant": "Yes, A."}],
            "relevance_min": 4,
            "visual_dependency_min": 5,
            "formatting_min": 4,
            "image_correspondence_min": 1,
        },
        {  # same content but WITH a leading marker -> must produce the same shapes
            "images": [{"bytes": _png_bytes(0), "path": None}],
            "texts": [
                {"user": "<image>\nDo any points other than M lie on XY?", "assistant": "Yes, A."}
            ],
            "relevance_min": 4,
            "visual_dependency_min": 5,
            "formatting_min": 4,
            "image_correspondence_min": 1,
        },
    ]
    _write_parquet(str(d / "train-00000.parquet"), rows, cols)

    ds = FineVisionDatasetConfig(dataset_path=str(d), max_crops=2).build(_FakeTokenizer())
    no_marker, with_marker = ds[0], ds[1]

    for ex in (no_marker, with_marker):
        n_patch = int((ex["input_ids"] == IM_PATCH_ID).sum())
        assert n_patch == int((ex["pooled_patches_idx"] >= 0).any(axis=-1).sum()) > 0
        assert float(ex["loss_masks"].sum()) == 3.0  # "Yes, A." -> 2 word tokens + EOS
    # stripping the marker makes the two rows byte-identical
    for key in ("input_ids", "labels", "loss_masks", "position_ids", "token_type_ids"):
        np.testing.assert_array_equal(no_marker[key], with_marker[key], err_msg=key)


def test_finevision_image_correspondence_filter_is_opt_in(tmp_path):
    """A low `image_correspondence_min` must not be dropped unless explicitly filtered
    (it is 1 for most geo170k(qa) / mavis_math_metagen rows)."""
    from olmo_core.data.multimodal import FineVisionDatasetConfig

    d = tmp_path / "lowcorr"
    d.mkdir()
    cols = [
        "images",
        "texts",
        "relevance_min",
        "visual_dependency_min",
        "formatting_min",
        "image_correspondence_min",
    ]
    rows = [
        {
            "images": [{"bytes": _png_bytes(i), "path": None}],
            "texts": [{"user": "q", "assistant": "a"}],
            "relevance_min": 5,
            "visual_dependency_min": 5,
            "formatting_min": 4,
            "image_correspondence_min": 1,
        }
        for i in range(3)
    ]
    _write_parquet(str(d / "train-00000.parquet"), rows, cols)
    tok = _FakeTokenizer()

    assert len(FineVisionDatasetConfig(dataset_path=str(d)).build(tok)) == 3
    assert (
        len(FineVisionDatasetConfig(dataset_path=str(d), min_visual_dependency=4).build(tok)) == 3
    )
    # only an explicit correspondence floor removes them
    assert (
        len(FineVisionDatasetConfig(dataset_path=str(d), min_image_correspondence=4).build(tok))
        == 0
    )
