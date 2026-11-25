"""Unit tests for the environment-independent parity artifact comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_parity_script():
    script = Path(__file__).resolve().parents[4] / "src" / "scripts" / "compare_sft_example.py"
    spec = importlib.util.spec_from_file_location("compare_sft_example", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save(module, path: Path, example: dict) -> None:
    module.save_artifact(
        path,
        example,
        source="test",
        dataset="text_vqa",
        index=0,
        seed=7,
        seq_len=16_384,
    )


def test_artifact_comparison_accepts_equal_normalized_tensors(tmp_path):
    module = _load_parity_script()
    mm_path = tmp_path / "mm.npz"
    oc_path = tmp_path / "oc.npz"
    _save(
        module,
        mm_path,
        {
            "input_tokens": np.array([1, 2], dtype=np.int64),
            "target_tokens": np.array([2, 3], dtype=np.int64),
            "token_pooling": np.array([[0, 1]], dtype=np.int64),
            "loss_masks": np.array([0.0, 1.0], dtype=np.float32),
        },
    )
    _save(
        module,
        oc_path,
        {
            "input_ids": np.array([1, 2], dtype=np.int64),
            "labels": np.array([2, 3], dtype=np.int64),
            "pooled_patches_idx": np.array([[0, 1]], dtype=np.int64),
            "loss_masks": np.array([0.0, 1.0], dtype=np.float32),
        },
    )

    assert module.compare_artifacts(mm_path, oc_path) == []


def test_artifact_comparison_reports_first_tensor_difference(tmp_path):
    module = _load_parity_script()
    mm_path = tmp_path / "mm.npz"
    oc_path = tmp_path / "oc.npz"
    _save(module, mm_path, {"input_ids": np.array([1, 2], dtype=np.int64)})
    _save(module, oc_path, {"input_ids": np.array([1, 9], dtype=np.int64)})

    differences = module.compare_artifacts(mm_path, oc_path)

    assert differences == ["input_ids: first mismatch at (1,): mm_olmo=2, olmo_core=9"]


def test_image_only_v9_registry_has_43_datasets():
    module = _load_parity_script()
    # 4 demo + 33 academic (incl. 3 mantis + 6 multidoc) + 5 pointing + tulu4
    assert len(module.image_only_v9_dataset_names()) == 43


def test_image_diagnostics_accepts_numpy_array():
    module = _load_parity_script()
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[0, 0] = [255, 0, 0]
    diag = module.image_diagnostics(arr)
    assert diag["size"] == [4, 4]
    assert len(diag["rgb_sha256"]) == 64
