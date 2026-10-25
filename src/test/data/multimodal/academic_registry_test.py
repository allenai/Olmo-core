"""Focused tests for academic dataset path resolution."""

from __future__ import annotations

import json

import pytest

from olmo_core.data.multimodal.academic import registry


@pytest.mark.parametrize(
    ("requested_split", "annotation_split", "image_directory"),
    [
        ("train", "train", "train_images"),
        ("validation", "val", "train_images"),
        ("test", "test", "test_images"),
    ],
)
def test_text_vqa_split_paths(
    monkeypatch,
    tmp_path,
    requested_split: str,
    annotation_split: str,
    image_directory: str,
):
    image_id = f"{annotation_split}-image"
    annotation_path = tmp_path / f"TextVQA_0.5.1_{annotation_split}.json"
    annotation_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "image_id": image_id,
                        "question": f"{annotation_split} question",
                        "answers": [f"{annotation_split} answer"],
                        "question_id": f"{annotation_split}-question-id",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(registry, "TEXT_VQA_SOURCE", str(tmp_path))

    rows = registry._load_text_vqa(requested_split)

    assert rows == [
        {
            "image": str(tmp_path / image_directory / f"{image_id}.jpg"),
            "question": f"{annotation_split} question",
            "answers": [f"{annotation_split} answer"],
            "metadata": {
                "image_url": None,
                "image_id": image_id,
                "example_id": f"{annotation_split}-question-id",
            },
        }
    ]
