"""Focused tests for perception-phase document datasets."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from olmo_core.data.multimodal import finevision
from olmo_core.data.multimodal import vision_alignment_perception as perception


class _Tokenizer:
    eos_token_id = 100257
    bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [(zlib.crc32(word.encode()) % 10000) + 10 for word in text.split()]


def _encoded_stub() -> Dict[str, np.ndarray]:
    return {
        "input_ids": np.array([1, 2, 3], dtype=np.int64),
        "labels": np.array([2, 3, 4], dtype=np.int64),
        "loss_masks": np.array([0.0, 1.0, 1.0], dtype=np.float32),
        "position_ids": np.arange(3, dtype=np.int64),
        "token_type_ids": np.zeros(3, dtype=np.int64),
        "images": np.zeros((1, 1, 1), dtype=np.float32),
        "pooled_patches_idx": np.zeros((1, 1), dtype=np.int64),
    }


@pytest.mark.parametrize("max_crops", [8, 12])
def test_ocr_document_uses_fixed_native_prompt_and_modal_answer(monkeypatch, tmp_path, max_crops):
    image = tmp_path / "image.png"
    image.write_bytes(b"not decoded in this test")
    rows = {
        "text_vqa": [
            {
                "image": str(image),
                "question": "  What word is visible?  ",
                "answers": ["Blue", "blue", "red"],
            }
        ],
        "doc_qa": [
            {
                "image": str(image),
                "question": "What is the total?",
                "answers": ["42"],
            }
        ],
    }
    monkeypatch.setattr(
        perception,
        "build_academic_data",
        lambda name, split: rows[name],
    )
    monkeypatch.setattr(perception, "decode_pil_image", lambda value: value)
    captured: List[Any] = []

    def encode(_tokenizer, image_value, turns, **kwargs):
        captured.append((image_value, turns, kwargs))
        return _encoded_stub()

    monkeypatch.setattr(perception, "encode_sft_example", encode)
    config = perception.VisionAlignmentOcrDocumentDatasetConfig(
        source_names=("text_vqa", "doc_qa"),
        max_sequence_length=3,
        max_crops=max_crops,
    )
    dataset = config.build(_Tokenizer())
    dataset.validate_required_annotations()

    assert len(dataset) == 2
    assert dataset[0]["input_ids"].tolist() == [1, 2, 3]
    assert captured[0][1] == [("Question: What word is visible?\nAnswer:", "Blue")]
    assert captured[0][2]["message_format"] == "document"
    assert captured[0][2]["max_images"] == 1
    assert captured[0][2]["max_crops"] == max_crops
    assert len(dataset.content_fingerprint) == 64


def test_ocr_fingerprint_binds_annotations_and_validation_fails_closed(monkeypatch, tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"x")
    rows = [
        {"image": str(image), "question": "Q", "answers": ["A"]},
        {"image": str(tmp_path / "missing.png"), "question": "", "answers": []},
    ]
    monkeypatch.setattr(perception, "build_academic_data", lambda _name, split: rows)
    config = perception.VisionAlignmentOcrDocumentDatasetConfig(source_names=("text_vqa",))
    first = config.build(_Tokenizer())
    first_fingerprint = first.content_fingerprint

    with pytest.raises(ValueError, match="2 invalid rows|1 invalid rows"):
        first.validate_required_annotations()

    rows[0] = {**rows[0], "answers": ["changed"]}
    second = config.build(_Tokenizer())
    assert second.content_fingerprint != first_fingerprint


@pytest.mark.parametrize("embedded_image", [False, True])
def test_ocr_only_computes_image_identity_at_build_time(monkeypatch, tmp_path, embedded_image):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    image = {"bytes": b"image", "path": None} if embedded_image else str(image_path)
    rows = [{"image": image, "question": "  What is visible?  ", "answers": ["Blue", "blue"]}]
    monkeypatch.setattr(perception, "build_academic_data", lambda _name, split: rows)
    monkeypatch.setattr(perception, "decode_pil_image", lambda value: value)
    encoded = []

    def encode(_tokenizer, image_value, turns, **kwargs):
        encoded.append((image_value, turns))
        return _encoded_stub()

    monkeypatch.setattr(perception, "encode_sft_example", encode)
    dataset = perception.VisionAlignmentOcrDocumentDatasetConfig(
        source_names=("text_vqa",),
    ).build(_Tokenizer())
    fingerprint = dataset.content_fingerprint

    def unexpected_identity(_image):
        raise AssertionError("Image identity is unnecessary for validation and encoding")

    monkeypatch.setattr(perception, "_image_identity", unexpected_identity)
    dataset.validate_required_annotations()
    dataset.get(0, 3)

    assert dataset.content_fingerprint == fingerprint
    assert encoded == [(image, [("Question: What is visible?\nAnswer:", "Blue")])]


@pytest.mark.parametrize("loss_token_weighting", ["none", "root_subsegments_root_tokens"])
@pytest.mark.parametrize("message_weight", [None, 1.0, 16.0])
def test_ocr_message_weight_only_scales_response_loss_masks(
    monkeypatch, loss_token_weighting, message_weight
):
    from PIL import Image

    rows = [
        {
            "image": Image.new("RGB", (32, 32), "white"),
            "question": "What words are visible?",
            "answers": ["blue sky"],
        }
    ]
    monkeypatch.setattr(perception, "build_academic_data", lambda _name, split: rows)
    tokenizer = _Tokenizer()
    config = perception.VisionAlignmentOcrDocumentDatasetConfig(
        source_names=("text_vqa",),
        max_crops=1,
        max_sequence_length=8192,
        loss_token_weighting=loss_token_weighting,
    )
    baseline = config.build(tokenizer).get(0, 7)
    weighted = replace(config, message_weight=message_weight).build(tokenizer).get(0, 7)

    assert baseline.keys() == weighted.keys()
    for key in baseline:
        if key == "loss_masks":
            multiplier = 1.0 if message_weight is None else message_weight
            np.testing.assert_array_equal(weighted[key], baseline[key] * multiplier)
        elif isinstance(baseline[key], np.ndarray):
            np.testing.assert_array_equal(weighted[key], baseline[key])
        else:
            assert weighted[key] == baseline[key]
    response_mask = baseline["loss_masks"] > 0
    assert baseline["labels"][response_mask].tolist() == [
        *tokenizer.encode(" blue sky", add_special_tokens=False),
        tokenizer.eos_token_id,
    ]
    assert np.count_nonzero(~response_mask) > 0
    assert np.count_nonzero(weighted["loss_masks"][~response_mask]) == 0
    assert baseline["images"].shape[0] > 0
    assert not baseline["metadata"]["truncated"]


def test_ocr_fingerprint_preserves_unweighted_identity(monkeypatch):
    rows = [{"image": "/synthetic/image.png", "question": "Q", "answers": ["A"]}]
    monkeypatch.setattr(perception, "build_academic_data", lambda _name, split: rows)
    identities = []
    canonical_sha256 = perception._canonical_sha256

    def capture_identity(value):
        identities.append(value)
        return canonical_sha256(value)

    monkeypatch.setattr(perception, "_canonical_sha256", capture_identity)
    config = perception.VisionAlignmentOcrDocumentDatasetConfig(source_names=("text_vqa",))
    baseline = config.build(_Tokenizer())
    weighted = replace(config, message_weight=16).build(_Tokenizer())
    weighted_float = replace(config, message_weight=16.0).build(_Tokenizer())

    assert "message_weight" not in identities[0]
    assert identities[1] == {**identities[0], "message_weight": 16.0}
    assert baseline.content_fingerprint != weighted.content_fingerprint
    assert weighted.content_fingerprint == weighted_float.content_fingerprint


@pytest.mark.parametrize("message_weight", [None, 1.0, 16, 16.0])
def test_ocr_message_weight_config_round_trip(message_weight):
    config = perception.VisionAlignmentOcrDocumentDatasetConfig(message_weight=message_weight)
    assert config.from_dict(config.as_config_dict()) == config
    legacy_config = config.as_config_dict()
    legacy_config.pop("message_weight", None)
    assert config.from_dict(legacy_config).message_weight is None


@pytest.mark.parametrize(
    "message_weight", [0, -1.0, float("nan"), float("inf"), -float("inf"), True, False, "16", []]
)
def test_ocr_rejects_invalid_message_weights(message_weight):
    with pytest.raises(ValueError, match="message_weight must be finite and positive"):
        perception.VisionAlignmentOcrDocumentDatasetConfig(message_weight=message_weight).build(
            _Tokenizer()
        )


def test_ocr_rejects_chat_layout_and_unreviewed_sources(monkeypatch):
    monkeypatch.setattr(perception, "build_academic_data", lambda _name, split: [])
    with pytest.raises(ValueError, match="message_format='document'"):
        perception.VisionAlignmentOcrDocumentDatasetConfig(
            source_names=("text_vqa",), message_format="qwen3"
        ).build(_Tokenizer())
    with pytest.raises(ValueError, match="selected from"):
        perception.VisionAlignmentOcrDocumentDatasetConfig(source_names=("science_qa_img",)).build(
            _Tokenizer()
        )


def _finevision_arrow(*, texts, images, formatting=4, visual=4, relevance=4):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "texts": texts,
            "images": images,
            "formatting_min": [formatting] * len(texts),
            "visual_dependency_min": [visual] * len(texts),
            "image_correspondence_min": [4] * len(texts),
            "relevance_min": [relevance] * len(texts),
        }
    )


def test_finevision_strict_annotations_and_fingerprint(monkeypatch):
    valid = _finevision_arrow(
        texts=[[{"user": "Describe it.", "assistant": "A triangle."}]],
        images=[[{"bytes": b"image", "path": None}]],
    )
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: valid)
    config = finevision.FineVisionDatasetConfig(
        dataset_path="/synthetic/reviewed",
        message_format="document",
        min_formatting=4,
        min_visual_dependency=4,
        min_relevance=4,
        require_quality_columns=True,
        strict_annotations=True,
    )
    dataset = config.build(_Tokenizer())
    dataset.validate_required_annotations()
    assert len(dataset.content_fingerprint) == 64

    invalid = _finevision_arrow(
        texts=[[{"user": "", "assistant": "answer"}]],
        images=[[]],
    )
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: invalid)
    dataset = config.build(_Tokenizer())
    with pytest.raises(ValueError, match="exactly one image"):
        dataset.validate_required_annotations()


@pytest.fixture
def finevision_materialization(tmp_path, monkeypatch):
    import pyarrow as pa

    root = tmp_path / "materialized" / "visualwebinstruct-filtered"
    root.mkdir(parents=True)
    arrow = _finevision_arrow(
        texts=[[{"user": "Describe it.", "assistant": "A triangle."}]] * 2,
        images=[[{"bytes": b"image", "path": None}]] * 2,
    ).add_column("unused", [0, 1])
    info = {"features": arrow.features.to_dict()}
    # Match the List-authored materializations loaded through the Arrow compatibility path.
    # ``datasets`` < 4 serializes a list feature as ``[inner]``; newer versions already emit
    # ``{"_type": "List", "feature": inner}``.
    texts_feature = info["features"]["texts"]
    info["features"]["texts"] = {
        "_type": "List",
        "feature": texts_feature[0]
        if isinstance(texts_feature, list)
        else texts_feature["feature"],
    }
    info_path = root / "dataset_info.json"
    info_path.write_text(json.dumps(info))
    table = arrow.data.table
    shards = []
    for i in range(2):
        path = root / f"data-{i:05d}-of-00002.arrow"
        with pa.ipc.new_stream(str(path), table.schema) as writer:
            writer.write_table(table.slice(i, 1))
        shards.append(
            {
                "path": str(path.relative_to(root.parent)),
                "rows": 1,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    output = {
        "name": "visualwebinstruct(filtered)",
        "path": root.name,
        "rows": 2,
        "physical_schema_sha256": hashlib.sha256(
            table.schema.remove_metadata().serialize().to_pybytes()
        ).hexdigest(),
        "dataset_info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "shards": shards,
    }
    identity = {
        "version": "vision-alignment-finevision-arrow-content-v1",
        "source_name": output["name"],
        "rows": output["rows"],
        "physical_schema_sha256": output["physical_schema_sha256"],
        "dataset_info_sha256": output["dataset_info_sha256"],
        "shards": [{"rows": item["rows"], "sha256": item["sha256"]} for item in shards],
    }
    output["dataset_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root.parent / "vision-alignment-finevision-materialization.json").write_text(
        json.dumps(
            {
                "format": "vision_alignment_finevision_materialization",
                "version": 1,
                "status": "verified",
                "outputs": [output],
            }
        )
    )
    monkeypatch.setenv("OLMO_CORE_DATA_VERIFICATION_CACHE_DIR", str(tmp_path / "verified-cache"))
    return finevision.FineVisionDatasetConfig(
        dataset_path=str(root),
        expected_materialized_fingerprint=output["dataset_fingerprint"],
        strict_annotations=True,
        message_format="document",
    )


def test_finevision_materialization_pin_is_not_projected_hf_fingerprint(finevision_materialization):
    config = finevision_materialization
    raw = finevision.load_hf_dataset(config.resolved_path())
    dataset = config.build(_Tokenizer())
    assert "unused" in raw.column_names and "unused" not in dataset._data.column_names
    assert len(raw._fingerprint) == len(dataset._data._fingerprint) == 16
    assert raw._fingerprint != dataset._data._fingerprint
    assert config.expected_materialized_fingerprint not in (
        raw._fingerprint,
        dataset._data._fingerprint,
    )
    assert len(dataset) == 2
    dataset.validate_required_annotations()
    # Byte verification preserves the established pin-based resume identity.
    assert dataset.content_fingerprint == config.build(_Tokenizer()).content_fingerprint


@pytest.mark.parametrize("strict", [False, True])
def test_finevision_materialization_rejects_wrong_pin(finevision_materialization, strict):
    config = replace(
        finevision_materialization,
        expected_materialized_fingerprint="0" * 64,
        strict_annotations=strict,
    )
    with pytest.raises(ValueError, match="materialization fingerprint differs"):
        config.build(_Tokenizer())


@pytest.mark.parametrize("change", ["missing", "extra", "reordered_loaded"])
def test_finevision_materialization_requires_exact_loaded_shards(
    finevision_materialization, monkeypatch, change
):
    from datasets import Dataset, concatenate_datasets

    config = finevision_materialization
    root = Path(config.resolved_path())
    shards = sorted(root.glob("data-*.arrow"))
    if change == "missing":
        shards[-1].unlink()
    elif change == "extra":
        (root / "data-unexpected.arrow").write_bytes(shards[0].read_bytes())
    else:
        wrong = concatenate_datasets([Dataset.from_file(str(path)) for path in reversed(shards)])
        monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: wrong)
    with pytest.raises(ValueError, match="loaded shard inventory differs"):
        config.build(_Tokenizer())


def test_finevision_materialization_cache_detects_same_size_edits_and_does_not_cache_failure(
    finevision_materialization, monkeypatch
):
    config = finevision_materialization
    calls = []
    file_digest = hashlib.file_digest

    def count_hash(stream, algorithm):
        calls.append(Path(stream.name))
        return file_digest(stream, algorithm)

    monkeypatch.setattr(finevision.hashlib, "file_digest", count_hash)
    config.build(_Tokenizer())
    assert len(calls) == 3  # Two Arrow shards and dataset_info.json.
    config.build(_Tokenizer())
    assert len(calls) == 3

    path = sorted(Path(config.resolved_path()).glob("data-*.arrow"))[0]
    before = path.stat()
    original = path.read_bytes()
    changed = original.replace(b"triangle", b"squarexx", 1)
    assert changed != original and len(changed) == len(original)
    path.write_bytes(changed)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.stat().st_ctime_ns != before.st_ctime_ns
    for expected_calls in (4, 5):
        with pytest.raises(ValueError, match="file checksum differs"):
            config.build(_Tokenizer())
        assert len(calls) == expected_calls


@pytest.mark.parametrize("cached", [False, True])
def test_finevision_materialization_rejects_changes_during_verification(
    finevision_materialization, monkeypatch, cached
):
    config = finevision_materialization
    path = sorted(Path(config.resolved_path()).glob("data-*.arrow"))[0]
    if cached:
        config.build(_Tokenizer())
        verify_file = finevision._verify_file

        def mutate_after_cache_hit(filename, *args):
            result = verify_file(filename, *args)
            if Path(filename) == path:
                path.touch()
            return result

        monkeypatch.setattr(finevision, "_verify_file", mutate_after_cache_hit)
    else:
        file_digest = hashlib.file_digest

        def mutate_during_hash(stream, algorithm):
            result = file_digest(stream, algorithm)
            if Path(stream.name) == path:
                path.touch()
            return result

        monkeypatch.setattr(finevision.hashlib, "file_digest", mutate_during_hash)
    with pytest.raises(ValueError, match="changed during verification"):
        config.build(_Tokenizer())


def test_finevision_materialization_rejects_stale_mapping_after_atomic_replacement(
    finevision_materialization, monkeypatch
):
    config = finevision_materialization
    path = sorted(Path(config.resolved_path()).glob("data-*.arrow"))[0]
    original = path.read_bytes()
    changed = original.replace(b"triangle", b"squarexx", 1)
    assert changed != original and len(changed) == len(original)
    path.write_bytes(changed)
    old_inode = path.stat().st_ino
    load = finevision.load_hf_dataset

    def load_then_replace(*args, **kwargs):
        stale = load(*args, **kwargs)
        replacement = path.with_name(f".{path.name}.replacement")
        replacement.write_bytes(original)
        replacement.replace(path)
        assert path.stat().st_ino != old_inode
        # The path now has the pinned bytes, but the mapped table still reads the old inode.
        assert stale[0]["texts"][0]["assistant"] == "A squarexx."
        return stale

    monkeypatch.setattr(finevision, "load_hf_dataset", load_then_replace)
    with pytest.raises(ValueError, match="changed while loading"):
        config.build(_Tokenizer())


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("<image>\n", ""),
        ("  <image>  ", ""),
        ("<image>\nDescribe it.", "Describe it."),
        ("Describe it.", "Describe it."),
    ],
)
def test_finevision_validation_and_runtime_share_prompt_rules(monkeypatch, prompt, expected):
    arrow = _finevision_arrow(
        texts=[[{"user": prompt, "assistant": " A triangle. "}]],
        images=[[{"bytes": b"image", "path": None}]],
    )
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: arrow)
    monkeypatch.setattr(finevision, "decode_pil_image", lambda image: image)
    captured = []

    def encode(_tokenizer, images, turns, **kwargs):
        captured.append((images, turns))
        return _encoded_stub()

    monkeypatch.setattr(finevision, "encode_sft_example", encode)
    dataset = finevision.FineVisionDatasetConfig(
        message_format="document", strict_annotations=True, skip_bad_rows=False
    ).build(_Tokenizer())
    fingerprint = dataset.content_fingerprint
    dataset.validate_required_annotations()
    dataset.get(0, 0)

    assert len(dataset) == 1
    assert dataset.content_fingerprint == fingerprint
    assert captured == [([{"bytes": b"image", "path": None}], [[(expected, "A triangle.")]])]


@pytest.mark.parametrize(
    "prompt,answer,images",
    [
        ("<image>", "Answer", []),
        ("<image>", "Answer", [None]),
        ("", "Answer", [{"bytes": b"image"}]),
        ("  ", "Answer", [{"bytes": b"image"}]),
        (None, "Answer", [{"bytes": b"image"}]),
        ("<image>", "  ", [{"bytes": b"image"}]),
        ("<image>", None, [{"bytes": b"image"}]),
    ],
)
def test_finevision_image_only_prompt_does_not_admit_missing_data(
    monkeypatch, prompt, answer, images
):
    arrow = _finevision_arrow(texts=[[{"user": prompt, "assistant": answer}]], images=[images])
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: arrow)
    dataset = finevision.FineVisionDatasetConfig(
        message_format="document", strict_annotations=True, skip_bad_rows=False
    ).build(_Tokenizer())

    with pytest.raises(ValueError, match="exactly one image"):
        dataset.validate_required_annotations()
    with pytest.raises(ValueError, match=r"no usable \(user, assistant\) turn"):
        dataset.get(0, 0)


@pytest.mark.parametrize("first_prompt", ["Describe it.", ""])
def test_finevision_image_only_prompt_is_only_allowed_in_first_turn(monkeypatch, first_prompt):
    arrow = _finevision_arrow(
        texts=[
            [
                {"user": first_prompt, "assistant": "First answer."},
                {"user": "<image>\n", "assistant": "Second answer."},
            ]
        ],
        images=[[{"bytes": b"image", "path": None}]],
    )
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: arrow)
    monkeypatch.setattr(finevision, "decode_pil_image", lambda image: image)
    captured = []

    def encode(_tokenizer, images, turns, **kwargs):
        captured.append(turns)
        return _encoded_stub()

    monkeypatch.setattr(finevision, "encode_sft_example", encode)
    dataset = finevision.FineVisionDatasetConfig(
        message_format="document", skip_bad_rows=False
    ).build(_Tokenizer())
    if first_prompt:
        dataset.get(0, 0)
        assert captured == [[[(first_prompt, "First answer.")]]]
    else:
        with pytest.raises(ValueError, match=r"no usable \(user, assistant\) turn"):
            dataset.get(0, 0)


def test_finevision_image_only_document_encoding(monkeypatch):
    from PIL import Image

    image_bytes = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(image_bytes, format="PNG")
    answer = "A triangle."
    arrow = _finevision_arrow(
        texts=[[{"user": "<image>\n", "assistant": answer}]],
        images=[[{"bytes": image_bytes.getvalue(), "path": None}]],
    )
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: arrow)
    tokenizer = _Tokenizer()
    config = finevision.FineVisionDatasetConfig(
        message_format="document",
        strict_annotations=True,
        skip_bad_rows=False,
        max_crops=1,
        max_images=1,
        max_sequence_length=8192,
    )
    dataset = config.build(tokenizer)
    dataset.validate_required_annotations()
    example = dataset.get(0, 0)

    assert example["input_ids"][0] == tokenizer.eos_token_id
    assert (example["input_ids"] == config.token_ids.im_patch_id).any()
    assert example["images"].shape[0] > 0
    assert example["labels"][example["loss_masks"] > 0].tolist() == [
        *tokenizer.encode(" " + answer, add_special_tokens=False),
        tokenizer.eos_token_id,
    ]
    assert not example["metadata"]["truncated"]


def test_finevision_required_quality_column_fails_closed(monkeypatch):
    arrow = _finevision_arrow(
        texts=[[{"user": "Q", "assistant": "A"}]],
        images=[[{"bytes": b"image", "path": None}]],
    ).remove_columns("relevance_min")
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: arrow)
    config = finevision.FineVisionDatasetConfig(
        dataset_path="/synthetic/reviewed",
        min_relevance=4,
        require_quality_columns=True,
    )
    with pytest.raises(ValueError, match="relevance_min"):
        config.build(_Tokenizer())


class _AlignmentChild:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.content_fingerprint = f"fingerprint-{name}"
        self.validated = False

    def __len__(self):
        return self.size

    def validate_required_annotations(self):
        self.validated = True

    def get(self, index: int, epoch: int = 0):
        return {"source": self.name, "index": index, "epoch": epoch}


@pytest.mark.parametrize("max_crops", [8, 12])
def test_audited_alignment_combines_reviewed_sources(monkeypatch, max_crops):
    built = []

    def build(config, _tokenizer):
        child = _AlignmentChild(config.config_name, 2 if "visualweb" in config.config_name else 1)
        built.append((config, child))
        return child

    monkeypatch.setattr(finevision.FineVisionDatasetConfig, "build", build)
    config = perception.VisionAlignmentAuditedAlignmentDatasetConfig(max_crops=max_crops)
    dataset = config.build(_Tokenizer())
    dataset.validate_required_annotations()

    assert len(dataset) == 3
    assert dataset.get(0, 7)["source"] == "visualwebinstruct(filtered)"
    assert dataset.get(2, 7)["source"] == "geo170k(align)"
    assert all(child.validated for _, child in built)
    assert all(child_config.strict_annotations for child_config, _ in built)
    assert all(child_config.message_format == "document" for child_config, _ in built)
    assert all(child_config.max_images == 1 for child_config, _ in built)
    assert all(child_config.max_crops == max_crops for child_config, _ in built)
    assert len(dataset.content_fingerprint) == 64


@pytest.mark.parametrize(
    "config_type",
    [
        perception.VisionAlignmentOcrDocumentDatasetConfig,
        perception.VisionAlignmentAuditedAlignmentDatasetConfig,
    ],
)
@pytest.mark.parametrize("max_crops", [0, -1, True, False, 1.5, "12"])
def test_alignment_sources_require_positive_integer_crop_budgets(config_type, max_crops):
    with pytest.raises(ValueError, match="max_crops must be a positive integer"):
        config_type(max_crops=max_crops).build(_Tokenizer())
