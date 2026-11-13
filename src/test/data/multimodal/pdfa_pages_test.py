"""PDFA pages with LightOn transcripts: text clean-up, the five tasks, and page labels."""

import io
import json

import numpy as np
import pytest

from olmo_core.data.multimodal.pdfa_pages import (
    PdfaPagesDatasetConfig,
    normalize_lighton_text,
    quotable_lines,
)

from .stage1_document_mode_test import DOLMA2_IDS, _Tok

pytest.importorskip("pypdfium2")

_TOK = _Tok()
_HEADER = "Annual report of the harbour board, printed for the members"


def _page_text(doc, page):
    return f"# {_HEADER}\n\nDocument {doc} page {page + 1} discusses item number {doc * 100 + page} at length here."


def _pdf(n_pages, shade):
    from PIL import Image

    pages = [Image.new("RGB", (120, 160), color=(shade, 20 * (i % 10), 90)) for i in range(n_pages)]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


def _build(tmp_path):
    """Four documents: 0 (12 pages, page 5 has no transcript), 1 (6 pages), 2 (2 pages), and
    3 (8 pages, held out)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "pdfa").mkdir()
    (tmp_path / "lighton").mkdir()
    specs = [(12, "train", {5}), (6, "train", set()), (2, "train", set()), (8, "heldout", set())]
    rows = []
    with open(tmp_path / "pdfa" / "pdfa-0000.tar", "wb") as tar, open(
        tmp_path / "lighton" / "lighton-000.jsonl", "wb"
    ) as lo:
        for doc, (n_pages, split, untranscribed) in enumerate(specs):
            pdf = _pdf(n_pages, 40 * doc + 30)
            tar.write(b"\0" * 512)  # stands in for a tar header
            pdf_off = tar.tell()
            tar.write(pdf)
            pages = [
                [p, _page_text(doc, p), 20, 0, 0, 0, "train"]
                for p in range(n_pages)
                if p not in untranscribed
            ]
            record = (json.dumps({"key": f"k{doc}", "pages": pages}) + "\n").encode()
            lo_off = lo.tell()
            lo.write(record)
            rows.append(
                dict(
                    key=f"k{doc}",
                    shard=0,
                    pages=n_pages,
                    pdf_off=pdf_off,
                    pdf_size=len(pdf),
                    split=split,
                    lo_bucket=0,
                    lo_off=lo_off,
                    lo_size=len(record),
                    lo_pages=len(pages),
                )
            )
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), tmp_path / "index.parquet")
    return str(tmp_path)


def _dataset(tmp_path, task, **kw):
    kw.setdefault("window_pages", (4, 8))
    return PdfaPagesDatasetConfig(
        task=task,
        dataset_path=_build(tmp_path)
        if not (tmp_path / "index.parquet").exists()
        else str(tmp_path),
        max_crops=1,
        token_ids=DOLMA2_IDS,
        message_format="document",
        **kw,
    ).build(_TOK)


def _contains(ids, text):
    needle = np.asarray(_TOK.encode(text))
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(ids), len(needle))
    return bool((windows == needle).all(axis=1).any())


def _image_blocks(example):
    return int((example["input_ids"] == DOLMA2_IDS.im_start_id).sum())


def _answer(example):
    return example["labels"][example["loss_masks"] > 0]


def test_normalize_removes_what_the_labeller_added():
    raw = (
        '<div style="text-align: center;">\n  ![image](image_1.png)\n</div>\n\n# Title\n\n'
        '<table border="1" class="dataframe">\n  <tr style="text-align: right;"><td>1</td></tr>\n</table>\n\n'
        "<!-- Image: Figure 4 -->\nNote: * Authorisation letter should be attached\n\n\n\n"
        "*Note: The image contains a logo in the top right corner, which is not transcribed.*"
    )
    out = normalize_lighton_text(raw)
    assert (
        out
        == "# Title\n\n<table>\n  <tr><td>1</td></tr>\n</table>\n\nNote: * Authorisation letter should be attached"
    )
    assert normalize_lighton_text("![image](image_1.png)\n") == ""


def test_quotable_lines_skip_repeated_and_short_lines():
    text = _page_text(0, 3) + "\nshort line"
    assert quotable_lines(text, [_page_text(0, 4), _page_text(0, 5)]) == [
        "Document 0 page 4 discusses item number 3 at length here."
    ]


def test_page_ocr_is_one_example_per_transcribed_page(tmp_path):
    ds = _dataset(tmp_path, "page_ocr")
    assert len(ds) == 11 + 6 + 2  # the held-out document's pages are not there
    assert len(_dataset(tmp_path, "page_ocr", split="heldout")) == 8
    for i, (doc, page) in [(0, (0, 0)), (5, (0, 6)), (11, (1, 0)), (18, (2, 1))]:
        example = ds[i]
        assert _image_blocks(example) == 1
        expected = _TOK.encode(" " + _page_text(doc, page)) + [_TOK.eos_token_id]
        np.testing.assert_array_equal(_answer(example), expected)
        assert not _contains(example["input_ids"], "Page ")


def test_multi_page_tasks_use_documents_with_enough_pages(tmp_path):
    assert len(_dataset(tmp_path, "transcribe_page")) == 2  # 12 and 6 pages; not the 2-page one
    assert len(_dataset(tmp_path, "transcribe_page", min_coverage=0.95)) == 1


@pytest.mark.parametrize("task", ["transcribe_page", "missing_page"])
def test_run_of_labelled_pages_and_its_target(tmp_path, task):
    ds = _dataset(tmp_path, task)
    for epoch in range(4):
        rng = ds.epoch_rng(0, epoch)
        ds.target_dim_for(rng)
        plan = ds._plan(0, rng)
        example = ds.get(0, epoch)
        target = plan["target"]
        assert target != 5  # the page without a transcript is never the target
        assert _image_blocks(example) == len(plan["pages"])
        for page in plan["pages"]:
            assert _contains(example["input_ids"], f"Page {page + 1}:")
        assert (target in plan["pages"]) == (task == "transcribe_page")
        if task == "missing_page":
            assert plan["labels"] == [f"Page {p + 1}:" for p in plan["pages"]]
            both = sorted(plan["pages"] + [target])
            assert both == list(range(both[0], both[-1] + 1))  # the run with its gap filled
        expected = _TOK.encode(" " + _page_text(0, target)) + [_TOK.eos_token_id]
        np.testing.assert_array_equal(_answer(example), expected)


def test_find_page_quotes_a_line_only_its_page_has(tmp_path):
    ds = _dataset(tmp_path, "find_page")
    rng = ds.epoch_rng(1, 0)
    ds.target_dim_for(rng)
    plan = ds._plan(1, rng)
    prompt, answer = plan["turn"]
    target = plan["target"]
    assert answer == f"Page {target + 1}"
    assert f"page {target + 1} discusses item number {100 + target}" in prompt
    assert _HEADER not in prompt
    np.testing.assert_array_equal(
        _answer(ds.get(1, 0)), _TOK.encode(" " + answer) + [_TOK.eos_token_id]
    )


def test_page_order_answer_puts_the_shuffled_images_back(tmp_path):
    ds = _dataset(tmp_path, "page_order")
    seen = set()
    for epoch in range(6):
        rng = ds.epoch_rng(0, epoch)
        ds.target_dim_for(rng)
        plan = ds._plan(0, rng)
        order = [int(k) - 1 for k in plan["turn"][1].split(", ")]
        restored = [plan["pages"][k] for k in order]
        assert restored == sorted(plan["pages"]) and restored == list(
            range(restored[0], restored[0] + len(restored))
        )
        assert plan["labels"] is None
        seen.add(tuple(plan["pages"]))
    assert len(seen) > 1  # a new run and order each epoch
    example = ds.get(0, 0)
    assert _contains(example["input_ids"], "Image 1") and not _contains(
        example["input_ids"], "Page "
    )


def test_context_pages_can_use_fewer_crops(tmp_path):
    full = _dataset(tmp_path, "transcribe_page", window_pages=(6, 6)).get(1, 0)
    ds = _dataset(tmp_path, "transcribe_page", window_pages=(6, 6), context_max_crops=1)
    ds.config.max_crops = 4
    mixed = ds.get(1, 0)
    assert _image_blocks(mixed) == _image_blocks(full) == 6
    assert len(mixed["images"]) >= len(full["images"])


def test_sequence_length_limits_the_run(tmp_path):
    ds = _dataset(
        tmp_path, "missing_page", window_pages=(4, 12), max_sequence_length=3500 + 5 * 288
    )
    assert ds.max_window() == 5
    for epoch in range(4):
        example = ds.get(0, epoch)
        assert _image_blocks(example) <= 4
        assert len(example["input_ids"]) <= ds.config.max_sequence_length
