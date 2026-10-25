"""CPU tests for the Nemotron-Image-Training sources: response cleaning, conversation parsing, the
prepare step (image join, dropped rows, downscaling) and the task tags."""

import io
import json
import tarfile

import pytest

from olmo_core.data.multimodal import NemotronImageDatasetConfig
from olmo_core.data.multimodal import nemotron_image as nv
from olmo_core.exceptions import OLMoConfigurationError


class _PromptTok:
    """Minimal tokenizer that records the user turns it templates."""

    eos_token_id = 1
    bos_token_id = 0

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.prompts.append(messages[0]["content"])
        text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


def _png(size=(64, 48)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color=(30, 80, 140)).save(buf, format="PNG")
    return buf.getvalue()


def _row(row_id, image, turns):
    messages = []
    for k, (question, answer) in enumerate(turns):
        content = (
            [question] if k or image is None else [{"type": "image", "image": image}, question]
        )
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [answer]})
    return {"id": row_id, "messages": messages}


# ---------------------------------------------------------------------------
# Cleaning and parsing
# ---------------------------------------------------------------------------


def test_clean_caption_drops_filler_opening_only():
    text = "This is a detailed description of the provided image.\n\nA dog on a beach."
    assert nv.clean_caption(text) == "A dog on a beach."
    assert nv.clean_caption("Of course! Here it is:\nA cat.") == "A cat."
    assert nv.clean_caption("A cat. Of course, it sleeps.") == "A cat. Of course, it sleeps."
    assert nv.clean_caption("Of course.\n") is None


def test_clean_reasoning_keeps_trace_and_answer():
    assert nv.clean_reasoning("<think>\nCount: 6.\n</think>\n\\boxed{6}") == (
        "<think>\nCount: 6.\n</think>\n\\boxed{6}"
    )
    assert nv.clean_reasoning("<think>a</think>'Al Jazeera', 65%.") == (
        "<think>\na\n</think>\n'Al Jazeera', 65%."
    )


@pytest.mark.parametrize(
    "text",
    [
        "<think>never closed",
        "<think>trace</think>",
        "<think></think>answer",
        "answer without a trace",
        "<think>a</think>b<think>c</think>d",
    ],
)
def test_clean_reasoning_rejects_malformed(text):
    assert nv.clean_reasoning(text) is None


def test_parse_conversation_multi_turn():
    row = _row("r", "a.png", [("Q1?", "A1"), ("Q2?", "A2")])
    assert nv.parse_conversation(row, nv.clean_caption) == ("a.png", ["Q1?", "Q2?"], ["A1", "A2"])


def test_parse_conversation_rejects_no_image_and_bad_responses():
    assert nv.parse_conversation(_row("r", None, [("Q?", "A")]), nv.clean_caption) is None
    bad = _row("r", "a.png", [("Q?", "<think>open")])
    assert nv.parse_conversation(bad, nv.clean_reasoning) is None
    two = _row("r", "a.png", [("Q?", "A")])
    two["messages"][0]["content"].append({"type": "image", "image": "b.png"})
    assert nv.parse_conversation(two, nv.clean_caption) is None


# ---------------------------------------------------------------------------
# Prepare and build
# ---------------------------------------------------------------------------


def _prepare(tmp_path, monkeypatch, rows, images, name="nemotron_ecd"):
    """Prepare ``rows`` with ``images`` ({ref: bytes}) served from one tar."""
    jsonl = tmp_path / "in.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tar_path = tmp_path / "images.tar"
    with tarfile.open(tar_path, "w") as tar:
        for ref, data in images.items():
            info = tarfile.TarInfo(f"images/{ref}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    subset = nv.NEMOTRON_SUBSETS[name]
    monkeypatch.setitem(
        nv.NEMOTRON_IMAGE_SOURCES,
        subset.images,
        lambda root: nv._TarSource(str(tar_path), nv._image_ref),
    )
    root = tmp_path / "prepared"
    stats = nv.prepare_nemotron_subset(name, str(root), jsonl_path=str(jsonl), num_workers=2)
    return root, stats


def test_prepare_joins_images_and_drops_unusable_rows(tmp_path, monkeypatch):
    from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

    rows = [
        _row("b", "1.png", [("Q?", "<think>t</think>A")]),
        _row("a", "1.png", [("Other Q?", "<think>t</think>B")]),
        _row("c", "missing.png", [("Q?", "<think>t</think>A")]),
        _row("d", "2.png", [("Q?", "<think>never closed")]),
        _row("e", "1.png", [("Q?", "<think>t</think>A")]),  # duplicate of "b"
    ]
    root, stats = _prepare(tmp_path, monkeypatch, rows, {"1.png": _png(), "2.png": _png()})
    assert stats["jsonl_rows"] == 5
    assert stats["parsed_rows"] == 3
    assert stats["duplicate_rows"] == 1
    assert stats["rows"] == 2
    data = load_from_disk_compat(str(root / "nemotron_ecd"))
    assert data["id"] == ["b", "a"]  # an image's rows in JSONL order
    assert data[0]["responses"] == ["<think>\nt\n</think>\nA"]
    assert json.loads((root / "nemotron_ecd" / "prepare_stats.json").read_text())["rows"] == 2


def test_prepare_downscales_large_images(tmp_path, monkeypatch):
    from PIL import Image

    from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

    rows = [_row("a", "big.png", [("Q?", "<think>t</think>A")])]
    root, _ = _prepare(tmp_path, monkeypatch, rows, {"big.png": _png((4000, 1000))})
    image = Image.open(io.BytesIO(load_from_disk_compat(str(root / "nemotron_ecd"))[0]["image"]))
    assert image.format == "JPEG"
    assert max(image.size) == nv.MAX_IMAGE_SIDE


def test_example_is_tagged_with_the_subset_style(tmp_path, monkeypatch):
    rows = [
        _row(
            "a",
            "1.png",
            [("Which bar is highest?", "<think>t</think>B"), ("Why?", "<think>u</think>C")],
        )
    ]
    root, _ = _prepare(tmp_path, monkeypatch, rows, {"1.png": _png()})
    tok = _PromptTok()
    ds = NemotronImageDatasetConfig(
        subset="nemotron_ecd", dataset_path=str(root), max_crops=1
    ).build(tok)
    assert len(ds) == 1
    assert ds[0]["loss_masks"].sum() > 0
    assert "chart_reasoning: Which bar is highest?" in [p for p in tok.prompts if p]
    assert "chart_reasoning: Why?" not in tok.prompts


def test_unprepared_subset_is_a_clear_error(tmp_path):
    with pytest.raises(OLMoConfigurationError, match="nemotron_image nemotron_ecd"):
        NemotronImageDatasetConfig(subset="nemotron_ecd", dataset_path=str(tmp_path)).build(
            _PromptTok()
        )
    with pytest.raises(OLMoConfigurationError, match="unknown Nemotron subset"):
        NemotronImageDatasetConfig(subset="ecd").build(_PromptTok())


def test_every_subset_has_a_distinct_task_tag_per_kind():
    styles = {s.style for s in nv.NEMOTRON_SUBSETS.values()}
    assert {"detailed_caption", "text_caption", "chart_reasoning"} <= styles
    for subset in nv.NEMOTRON_SUBSETS.values():
        assert subset.style.endswith("_reasoning") == subset.reasoning
        assert subset.images in nv.NEMOTRON_IMAGE_SOURCES
        assert subset.release in nv.NEMOTRON_RELEASES


def test_prepare_keeps_one_copy_of_a_repeated_image(tmp_path, monkeypatch):
    """A parquet source holding the same image bytes twice emits its rows once."""
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

    data = _png()
    ref = f"deepvision_reasoning/images/{hashlib.sha256(data).hexdigest()}.png"
    parquet = tmp_path / "source.parquet"
    pq.write_table(
        pa.table({"images": [[{"bytes": data, "path": None}]] * 4}), parquet, row_group_size=1
    )
    monkeypatch.setitem(
        nv.NEMOTRON_IMAGE_SOURCES,
        "deepvision",
        lambda root: nv._Sha256ParquetSource(
            str(parquet), "images", "deepvision_reasoning/images/"
        ),
    )
    jsonl = tmp_path / "in.jsonl"
    rows = [
        _row("a", ref, [("Q?", "<think>t</think>A")]),
        _row("b", ref, [("R?", "<think>t</think>B")]),
    ]
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    root = tmp_path / "prepared"
    stats = nv.prepare_nemotron_subset(
        "nemotron_deepvision", str(root), jsonl_path=str(jsonl), num_workers=2
    )
    assert stats["images"] == 1
    assert load_from_disk_compat(str(root / "nemotron_deepvision"))["id"] == ["a", "b"]


def test_prepare_skips_shards_with_no_referenced_image(tmp_path, monkeypatch):
    """A tar holding none of the referenced images (most cc3m tars hold some) writes no shard."""
    from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

    for tar_name, refs in (("a.tar", ["1.png"]), ("b.tar", ["unused.png"])):
        with tarfile.open(tmp_path / tar_name, "w") as tar:
            for ref in refs:
                data = _png()
                info = tarfile.TarInfo(ref)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    monkeypatch.setitem(
        nv.NEMOTRON_IMAGE_SOURCES,
        "ecd",
        lambda root: nv._TarSource(str(tmp_path / "*.tar"), nv._image_ref),
    )
    jsonl = tmp_path / "in.jsonl"
    jsonl.write_text(json.dumps(_row("a", "1.png", [("Q?", "<think>t</think>A")])) + "\n")
    root = tmp_path / "prepared"
    stats = nv.prepare_nemotron_subset(
        "nemotron_ecd", str(root), jsonl_path=str(jsonl), num_workers=2
    )
    assert stats["rows"] == 1
    assert load_from_disk_compat(str(root / "nemotron_ecd"))["id"] == ["a"]


def test_compose_raven_lays_out_matrix_and_labelled_choices():
    from PIL import Image

    panel = _png((160, 160))
    image = Image.open(io.BytesIO(nv.compose_raven([panel] * 8, [panel] * 8)))
    assert image.format == "PNG"
    assert image.size[0] >= 4 * 160  # four choices a row
    assert image.size[1] >= 5 * 160  # three matrix rows and two choice rows
    with pytest.raises(ValueError, match="8 context panels"):
        nv.compose_raven([panel] * 7, [panel] * 8)


def test_combined_source_routes_references_by_predicate():
    class _Fixed:
        def __init__(self, tag):
            self.tag = tag

        def shards(self, needed):
            return [list(needed)] if needed else []

        def fetch(self, shard, needed):
            for ref in shard:
                yield ref, self.tag

    source = nv._CombinedSource(((nv._is_url, _Fixed(b"web")), (nv._is_not_url, _Fixed(b"tar"))))
    needed = {"https://example.com/a.png", "media/b.png"}
    fetched = dict(
        ref_data
        for shard in source.shards(sorted(needed))
        for ref_data in source.fetch(shard, needed)
    )
    assert fetched == {"https://example.com/a.png": b"web", "media/b.png": b"tar"}
