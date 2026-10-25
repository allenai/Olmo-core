#!/usr/bin/env python3
"""Export and compare isolated mm_olmo / OLMo-core SFT example artifacts.

The ``export-mm`` and ``export-olmo-core`` commands intentionally import only one
training stack. They can therefore run in their respective Conda environments.
The ``compare`` command is NumPy-only and compares the resulting ``.npz`` files.
Use ``scripts/compare_sft_example.sh`` to run the three commands together.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

_ALIASES = {
    "input_tokens": "input_ids",
    "target_tokens": "labels",
    "token_pooling": "pooled_patches_idx",
}
_PARITY_KEYS = (
    "input_ids",
    "labels",
    "loss_masks",
    "position_ids",
    "subsegment_ids",
    "images",
    "pooled_patches_idx",
)
# HARDCODED personal paths (donovanc's mm_olmo checkout / conda / activate script on
# weka), used only by this offline parity harness — never by training. Override via the
# MM_OLMO_ROOT env var and the --conda / --mm-env / --mm-activate-script CLI flags to run
# the mm_olmo export side from your own environment.
_MM_OLMO_ROOT = os.environ.get(
    "MM_OLMO_ROOT", "/weka/oe-training-default/donovanc/molmofication/mm_olmo"
)
_DEFAULT_CONDA = "/weka/oe-training-default/donovanc/miniconda3/bin/conda"
_DEFAULT_MM_OLMO_ACTIVATE = "/weka/oe-training-default/donovanc/mm_olmo-activate.sh"
_MM_OLMO_ENV_KEYS = (
    "HF_DATASETS_OFFLINE",
    "OLMO_SHARED_FS",
    "OMP_NUM_THREADS",
    "HF_HOME",
    "HF_DATASETS_CACHE",
)
_SCRIPT_PATH = Path(__file__).resolve()


@dataclass(frozen=True)
class ParityResult:
    dataset: str
    ok: bool
    differences: tuple[str, ...] = ()
    error: str | None = None
    artifact_dir: str | None = None


def image_only_v9_dataset_names() -> list[str]:
    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        IMAGE_ONLY_V9_SUBMIXTURES,
    )

    return [src.name for group in IMAGE_ONLY_V9_SUBMIXTURES for src in group.datasets]


def _artifact_stem(dataset: str, index: int, seed: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset)
    return f"{safe}-{index}-{seed}"


@lru_cache(maxsize=4)
def _mm_olmo_subprocess_env(activate_script: str) -> dict[str, str]:
    """Return a subprocess env with ``mm_olmo-activate.sh`` applied."""
    script = Path(activate_script)
    if not script.is_file():
        raise FileNotFoundError(f"mm_olmo activate script not found: {script}")
    proc = subprocess.run(
        [
            "bash",
            "-lc",
            f"source {shlex.quote(str(script))} && "
            "python -c 'import json, os; print(json.dumps(dict(os.environ)))'",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    env = os.environ.copy()
    env.update(json.loads(proc.stdout))
    return env


def _ensure_mm_olmo_env(activate_script: str | None = None) -> None:
    """Apply mm_olmo cache/offline env vars in the current process."""
    script = activate_script or _DEFAULT_MM_OLMO_ACTIVATE
    env = _mm_olmo_subprocess_env(script)
    for key in _MM_OLMO_ENV_KEYS:
        if key in env:
            os.environ[key] = env[key]


def _run_conda(
    conda: str,
    env: str,
    script: Path,
    command: str,
    args: list[str],
    *,
    subprocess_env: dict[str, str] | None = None,
) -> None:
    subprocess.run(
        [conda, "run", "-n", env, "python", str(script), command, *args],
        check=True,
        env=subprocess_env,
    )


def compare_one_dataset(
    *,
    dataset: str,
    index: int,
    seed: int,
    seq_len: int,
    artifact_dir: Path,
    conda: str,
    mm_env: str,
    olmo_core_env: str,
    script: Path,
    inspect: bool,
    diff_tokens: bool,
    mm_activate_script: str,
) -> ParityResult:
    """Export both stacks in their native envs and compare normalized artifacts."""
    mm_env_vars = _mm_olmo_subprocess_env(mm_activate_script)
    stem = _artifact_stem(dataset, index, seed)
    work_dir = artifact_dir / stem
    work_dir.mkdir(parents=True, exist_ok=True)
    mm_artifact = work_dir / "mm_olmo.npz"
    oc_artifact = work_dir / "olmo_core.npz"
    common = [
        "--dataset",
        dataset,
        "--index",
        str(index),
        "--seed",
        str(seed),
        "--seq_len",
        str(seq_len),
    ]
    try:
        _run_conda(
            conda,
            mm_env,
            script,
            "export-mm",
            [*common, "--output", str(mm_artifact), "--mm-activate-script", mm_activate_script],
            subprocess_env=mm_env_vars,
        )
        _run_conda(
            conda,
            olmo_core_env,
            script,
            "export-olmo-core",
            [*common, "--output", str(oc_artifact)],
        )
        if inspect:
            _run_conda(
                conda,
                olmo_core_env,
                script,
                "inspect",
                ["--mm-artifact", str(mm_artifact), "--olmo-core-artifact", str(oc_artifact)],
            )
        if diff_tokens:
            _run_conda(
                conda,
                mm_env,
                script,
                "diff-tokens",
                ["--mm-artifact", str(mm_artifact), "--olmo-core-artifact", str(oc_artifact)],
                subprocess_env=mm_env_vars,
            )
        differences = compare_artifacts(mm_artifact, oc_artifact)
        return ParityResult(
            dataset=dataset,
            ok=not differences,
            differences=tuple(differences),
            artifact_dir=str(work_dir),
        )
    except subprocess.CalledProcessError as exc:
        return ParityResult(
            dataset=dataset,
            ok=False,
            error=f"subprocess failed with exit code {exc.returncode}",
            artifact_dir=str(work_dir),
        )
    except Exception as exc:
        return ParityResult(
            dataset=dataset,
            ok=False,
            error=str(exc),
            artifact_dir=str(work_dir),
        )


def _compare_one_task(kwargs: dict[str, Any]) -> ParityResult:
    return compare_one_dataset(**kwargs)


def _print_summary(results: Sequence[ParityResult]) -> None:
    width = max((len(result.dataset) for result in results), default=7)
    print(f"\n{'dataset'.ljust(width)}  status  details")
    print(f"{'-' * width}  ------  -------")
    for result in sorted(results, key=lambda item: item.dataset):
        if result.ok:
            detail = "OK"
        elif result.error:
            detail = result.error
        elif result.differences:
            detail = result.differences[0]
            if len(result.differences) > 1:
                detail += f" (+{len(result.differences) - 1} more)"
        else:
            detail = "MISMATCH"
        status = "PASS" if result.ok else "FAIL"
        print(f"{result.dataset.ljust(width)}  {status:<6}  {detail}")


def _run_parity(args: argparse.Namespace) -> int:
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    datasets = list(args.dataset or [])
    if args.sweep:
        if datasets:
            raise SystemExit("--dataset and --sweep cannot be combined")
        datasets = image_only_v9_dataset_names()
    if not datasets:
        raise SystemExit("Either --dataset or --sweep is required")

    artifact_dir = (
        Path(args.artifact_dir)
        if args.artifact_dir
        else Path(tempfile.mkdtemp(prefix="molmo2-parity."))
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    script = Path(args.script) if args.script else _SCRIPT_PATH
    task_kwargs = [
        {
            "dataset": name,
            "index": args.index,
            "seed": args.seed,
            "seq_len": args.seq_len,
            "artifact_dir": artifact_dir,
            "conda": args.conda,
            "mm_env": args.mm_env,
            "olmo_core_env": args.olmo_core_env,
            "script": script,
            "inspect": args.inspect,
            "diff_tokens": args.diff_tokens,
            "mm_activate_script": args.mm_activate_script,
        }
        for name in datasets
    ]

    print(
        f"Running parity on {len(datasets)} dataset(s) "
        f"(index={args.index}, seed={args.seed}, seq_len={args.seq_len}, jobs={args.jobs})"
    )
    results: list[ParityResult] = []
    if args.jobs <= 1:
        for kwargs in task_kwargs:
            result = compare_one_dataset(**kwargs)
            results.append(result)
            status = "OK" if result.ok else "FAIL"
            print(f"[{status}] {result.dataset}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(_compare_one_task, kwargs): kwargs["dataset"]
                for kwargs in task_kwargs
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "OK" if result.ok else "FAIL"
                print(f"[{status}] {result.dataset}")

    _print_summary(results)
    failed = [result for result in results if not result.ok]
    if failed:
        print(f"\n{len(failed)} of {len(results)} dataset(s) failed.")
        print(f"Artifacts retained at: {artifact_dir}")
        return 1

    print(f"\nAll {len(results)} dataset(s) passed.")
    if args.keep_artifacts:
        print(f"Artifacts retained at: {artifact_dir}")
    elif args.artifact_dir is None:
        shutil.rmtree(artifact_dir)
    return 0


def _json_safe(value: Any) -> Any:
    """Return a compact JSON-compatible view of a formatted dataset example."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "image"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def image_diagnostics(image: Any) -> Dict[str, Any]:
    """Fingerprint decoded RGB pixels before either stack applies crop transforms."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype("uint8")).convert("RGB")
        source = None
    else:
        source = str(image) if isinstance(image, (str, os.PathLike)) else None
        img = Image.open(image) if source is not None else image
        img = img.convert("RGB")
    pixels = np.asarray(img)
    return {
        "source": source,
        "size": list(img.size),
        "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }


def _safe_image_diagnostics(formatted: Dict[str, Any]) -> Dict[str, Any] | None:
    if "image" not in formatted:
        return None
    return image_diagnostics(formatted["image"])


def _message_tree_text(tree: Any) -> Dict[str, Any]:
    """Render mm_olmo's pre-tokenization message tree without multimodal pixels."""
    return {
        "prefix": [
            {"role": message.role.value, "text": message.content.text} for message in tree.prefix
        ],
        "branches": [_message_tree_text(branch) for branch in tree.branches],
    }


def normalize_example(example: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Keep tensor fields and normalize equivalent mm_olmo / OLMo-core names."""
    normalized: Dict[str, np.ndarray] = {}
    for key, value in example.items():
        key = _ALIASES.get(key, key)
        if key not in _PARITY_KEYS:
            continue
        if isinstance(value, np.ndarray):
            normalized[key] = value
        elif hasattr(value, "detach") and hasattr(value, "cpu"):
            normalized[key] = value.detach().cpu().numpy()
        elif hasattr(value, "numpy"):
            normalized[key] = value.numpy()
    return normalized


def save_artifact(
    output: Path,
    example: Dict[str, Any],
    *,
    source: str,
    dataset: str,
    index: int,
    seed: int,
    seq_len: int,
    diagnostics: Dict[str, Any] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "source": source,
            "dataset": dataset,
            "index": index,
            "seed": seed,
            "seq_len": seq_len,
            "diagnostics": diagnostics,
        },
        sort_keys=True,
    )
    np.savez_compressed(output, __metadata__=np.array(metadata), **normalize_example(example))


def load_artifact(path: Path) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as artifact:
        if "__metadata__" not in artifact:
            raise ValueError(f"{path} is missing parity metadata")
        metadata = json.loads(str(artifact["__metadata__"].item()))
        arrays = {key: artifact[key] for key in artifact.files if key != "__metadata__"}
    return metadata, arrays


def _first_mismatch(a: np.ndarray, b: np.ndarray) -> tuple[int, ...] | None:
    if a.shape != b.shape:
        return None
    equal = np.equal(a, b)
    if np.issubdtype(a.dtype, np.inexact) and np.issubdtype(b.dtype, np.inexact):
        equal |= np.isnan(a) & np.isnan(b)
    mismatches = np.argwhere(~equal)
    return tuple(int(i) for i in mismatches[0]) if len(mismatches) else None


def compare_artifacts(mm_path: Path, oc_path: Path) -> list[str]:
    """Return exact tensor-level differences between two normalized artifacts."""
    mm_metadata, mm = load_artifact(mm_path)
    oc_metadata, oc = load_artifact(oc_path)
    differences: list[str] = []
    for key in ("dataset", "index", "seed", "seq_len"):
        if mm_metadata.get(key) != oc_metadata.get(key):
            differences.append(
                f"metadata {key}: mm_olmo={mm_metadata.get(key)!r} "
                f"olmo_core={oc_metadata.get(key)!r}"
            )

    for key in _PARITY_KEYS:
        a, b = mm.get(key), oc.get(key)
        if a is None and b is None:
            continue
        if a is None or b is None:
            if key in ("images", "pooled_patches_idx"):

                def _empty(arr):
                    return arr is not None and getattr(arr, "shape", (None,))[0] == 0

                if (a is None and _empty(b)) or (b is None and _empty(a)):
                    continue
            differences.append(f"{key}: mm_olmo={a is not None}, olmo_core={b is not None}")
            continue
        if a.shape != b.shape:
            differences.append(f"{key}: shape mm_olmo={a.shape}, olmo_core={b.shape}")
            continue
        if a.dtype != b.dtype:
            if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
                if np.array_equal(a.astype(np.int64), b.astype(np.int64)):
                    continue
            differences.append(f"{key}: dtype mm_olmo={a.dtype}, olmo_core={b.dtype}")
            continue
        mismatch = _first_mismatch(a, b)
        if mismatch is not None:
            a_value = a[mismatch].item()
            b_value = b[mismatch].item()
            differences.append(
                f"{key}: first mismatch at {mismatch}: "
                f"mm_olmo={a_value!r}, olmo_core={b_value!r}"
            )
    return differences


def inspect_artifacts(mm_path: Path, oc_path: Path) -> None:
    """Print pre-tokenization text fields and raw-image fingerprints side-by-side."""
    mm_metadata, _ = load_artifact(mm_path)
    oc_metadata, _ = load_artifact(oc_path)
    print("mm_olmo diagnostics:")
    print(json.dumps(mm_metadata.get("diagnostics"), indent=2, sort_keys=True))
    print("olmo_core diagnostics:")
    print(json.dumps(oc_metadata.get("diagnostics"), indent=2, sort_keys=True))


def _decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)])
    if text == "":
        return repr(tokenizer.convert_ids_to_tokens(int(token_id)))
    return repr(text)


def _token_label(tokenizer: Any, token_id: int) -> str:
    token_id = int(token_id)
    return f"{token_id}={_decode_token(tokenizer, token_id)}"


def diff_token_sequences(
    mm_ids: Sequence[int],
    oc_ids: Sequence[int],
    *,
    mm_tokenizer: Any,
    oc_tokenizer: Any,
    context: int = 3,
    max_diffs: int = 20,
) -> list[str]:
    """Return human-readable differences between two token-id sequences."""
    mm = [int(token_id) for token_id in mm_ids]
    oc = [int(token_id) for token_id in oc_ids]
    lines = [f"lengths: mm_olmo={len(mm)}, olmo_core={len(oc)}"]
    matcher = difflib.SequenceMatcher(None, mm, oc, autojunk=False)
    diff_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        diff_count += 1
        if diff_count > max_diffs:
            lines.append(f"... truncated after {max_diffs} diff regions")
            break
        mm_slice = mm[max(0, i1 - context) : min(len(mm), i2 + context)]
        oc_slice = oc[max(0, j1 - context) : min(len(oc), j2 + context)]
        lines.append(f"diff {diff_count}: {tag} mm[{i1}:{i2}] vs oc[{j1}:{j2}]")
        if tag == "replace":
            for offset, (mm_id, oc_id) in enumerate(zip(mm[i1:i2], oc[j1:j2])):
                lines.append(
                    f"  @{i1 + offset}: "
                    f"mm {_token_label(mm_tokenizer, mm_id)} | "
                    f"oc {_token_label(oc_tokenizer, oc_id)}"
                )
            if len(mm[i1:i2]) != len(oc[j1:j2]):
                longer = "mm_olmo" if len(mm[i1:i2]) > len(oc[j1:j2]) else "olmo_core"
                lines.append(
                    f"  extra in {longer}: "
                    f"{len(mm[i1:i2]) - len(oc[j1:j2])} token(s) in this region"
                )
        elif tag == "delete":
            for offset, mm_id in enumerate(mm[i1:i2]):
                lines.append(f"  @{i1 + offset}: mm only {_token_label(mm_tokenizer, mm_id)}")
        elif tag == "insert":
            for offset, oc_id in enumerate(oc[j1:j2]):
                lines.append(f"  @{j1 + offset}: oc only {_token_label(oc_tokenizer, oc_id)}")
        lines.append(
            "  mm context: "
            + ", ".join(_token_label(mm_tokenizer, token_id) for token_id in mm_slice)
        )
        lines.append(
            "  oc context: "
            + ", ".join(_token_label(oc_tokenizer, token_id) for token_id in oc_slice)
        )
    if diff_count == 0 and len(mm) == len(oc):
        lines.append("sequences are identical")
    return lines


def diff_artifacts(
    mm_path: Path,
    oc_path: Path,
    *,
    context: int = 3,
    max_diffs: int = 20,
) -> None:
    """Decode and diff ``input_ids`` from two parity artifacts."""
    if _MM_OLMO_ROOT not in sys.path:
        sys.path.insert(0, _MM_OLMO_ROOT)
    from olmo.model_configs import QWEN3_4B_INSTRUCT
    from transformers import AutoTokenizer

    _, mm = load_artifact(mm_path)
    _, oc = load_artifact(oc_path)
    if "input_ids" not in mm or "input_ids" not in oc:
        raise ValueError("both artifacts must contain input_ids")

    mm_tokenizer = QWEN3_4B_INSTRUCT.build_tokenizer()
    oc_tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-4B", trust_remote_code=True)
    for line in diff_token_sequences(
        mm["input_ids"],
        oc["input_ids"],
        mm_tokenizer=mm_tokenizer,
        oc_tokenizer=oc_tokenizer,
        context=context,
        max_diffs=max_diffs,
    ):
        print(line)


# image-only-v9 pointing sources use flat message_weight=0.2 (see mixtures/image_only_v9.py).
_IMAGE_ONLY_V9_POINTING_DATASETS = frozenset(
    {
        "pixmo_points_train",
        "pixmo_count_train",
        "pixmo_points_high_freq_train",
        "cosyn_point",
    }
)
_IMAGE_ONLY_V9_POINTING_WEIGHT = 0.2


def _image_only_v9_example_weight(dataset: str):
    """Mixture-level ``MessageWeight`` override (matches mm ``DeterministicDataset``)."""
    if dataset not in _IMAGE_ONLY_V9_POINTING_DATASETS:
        return None
    from olmo.preprocessing.text_preprocessor import MessageWeight

    return MessageWeight(
        weight=_IMAGE_ONLY_V9_POINTING_WEIGHT,
        root_length=False,
        root_subsegments=False,
    )


def _export_mm(args: argparse.Namespace) -> None:
    _ensure_mm_olmo_env(getattr(args, "mm_activate_script", None))
    if _MM_OLMO_ROOT not in sys.path:
        sys.path.insert(0, _MM_OLMO_ROOT)
    os.environ.setdefault("MOLMO_DATA_DIR", "/weka/oe-training-default/mm-olmo")

    from olmo.data.data_formatter import DataFormatter
    from olmo.data.get_dataset import get_dataset_config_by_name
    from olmo.model_configs import QWEN3_4B_INSTRUCT, VISION_BACKBONES
    from olmo.models.molmo2.example_preprocessor import Molmo2ExamplePreprocessor
    from olmo.models.molmo2.grounding_formatter import GroundingPreprocessor
    from olmo.nn.vision_backbone import MolmoVisionBackboneConfig
    from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
    from olmo.preprocessing.text_preprocessor import MessageWeight

    cfg = get_dataset_config_by_name(args.dataset)
    if args.dataset == "pixmo_clocks":
        from olmo.data.pixmo_datasets import PixMoClocksConfig

        cfg = PixMoClocksConfig(aug=True)
    data = cfg.build(split="train")
    tokenizer = QWEN3_4B_INSTRUCT.build_tokenizer()
    formatter = DataFormatter(
        prompt_templates="uber_model_v2",
        system_prompt="demo_or_style_v2",
        select_answer="best",
    )
    image_config = MultiCropConfig(
        use_single_crop_col_tokens=False,
        use_single_crop_start_token=True,
        crop_mode="overlap-and-resize-c2",
        max_crops=8,
        overlap_margins=(4, 4),
    )
    preprocessor = Molmo2ExamplePreprocessor(
        formatter,
        tokenizer,
        grounding_preprocessor=GroundingPreprocessor(),
        image_preprocessor=image_config.build_image_preprocessor(
            tokenizer,
            MolmoVisionBackboneConfig(vit=VISION_BACKBONES["siglip2"]).build_preprocessor(),
            None,
        )[0],
        video_preprocessor=None,
        max_sequence_len=args.seq_len,
        message_format="qwen3",
        default_message_weight=MessageWeight.from_string("root_subsegments_root_tokens"),
        is_training=True,
    )
    # mm_olmo per-example stream (dataset.py:68, epoch 0): one rng threads the
    # dataset's format_example AND the formatter — identical to olmo-core's
    # example_rng derivation, so artifacts align at every index.
    rng = np.random.RandomState((args.seed * 195172 + args.index) % (2**32 - 1))
    formatted = data.get(args.index, rng)
    example_weight = _image_only_v9_example_weight(args.dataset)
    if example_weight is not None:
        formatted = dict(formatted)
        formatted["weight"] = example_weight
    message_tree, _ = formatter(
        copy.deepcopy(formatted),
        is_training=True,
        for_inference=False,
        rng=rng,
    )
    diagnostics = {
        "formatted": _json_safe(formatted),
        "message_tree": _message_tree_text(message_tree),
    }
    image_diag = _safe_image_diagnostics(formatted)
    if image_diag is not None:
        diagnostics["image"] = image_diag
    example = preprocessor(formatted, rng)
    save_artifact(
        Path(args.output),
        example,
        source="mm_olmo",
        dataset=args.dataset,
        index=args.index,
        seed=args.seed,
        seq_len=args.seq_len,
        diagnostics=diagnostics,
    )


def _export_olmo_core(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from olmo_core.data.multimodal.academic.registry import (
        ACADEMIC_REGISTRY,
        build_academic_data,
        format_academic_example,
    )
    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        build_image_only_v9_datasets,
    )
    from olmo_core.data.multimodal.sft_formatter import SftFormatter

    tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-4B", trust_remote_code=True)
    datasets = build_image_only_v9_datasets(tokenizer, seed=args.seed)
    example = datasets[args.dataset][args.index]
    diagnostics = None
    if args.dataset in ACADEMIC_REGISTRY:
        from olmo_core.data.multimodal.sequence_builder import example_rng

        diag_rng = example_rng(args.seed, args.index)
        formatted = format_academic_example(
            args.dataset,
            build_academic_data(args.dataset, split="train")[args.index],
            diag_rng,
        )
        diagnostics = {
            "formatted": _json_safe(formatted),
            "turns": SftFormatter(seed=args.seed).format_branches(
                formatted, index=args.index, rng=diag_rng
            ),
        }
        image_diag = _safe_image_diagnostics(formatted)
        if image_diag is not None:
            diagnostics["image"] = image_diag
    elif args.dataset == "tulu4":
        from olmo_core.data.multimodal.tulu import Tulu4DatasetConfig

        row = Tulu4DatasetConfig(seed=args.seed).build(tokenizer)._data[args.index]
        diagnostics = {
            "formatted": _json_safe({"messages": row["messages"], "metadata": {"id": row["id"]}}),
        }
    save_artifact(
        Path(args.output),
        example,
        source="olmo_core",
        dataset=args.dataset,
        index=args.index,
        seed=args.seed,
        seq_len=args.seq_len,
        diagnostics=diagnostics,
    )


def _add_example_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument(
        "--mm-activate-script",
        default=_DEFAULT_MM_OLMO_ACTIVATE,
        help="bash script sourced for mm_olmo HF cache/offline env (default: mm_olmo-activate.sh)",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, handler in (("export-mm", _export_mm), ("export-olmo-core", _export_olmo_core)):
        export_parser = subparsers.add_parser(command)
        _add_example_arguments(export_parser)
        export_parser.add_argument("--output", required=True)
        export_parser.set_defaults(handler=handler)
    run_parser = subparsers.add_parser(
        "run",
        help="export mm_olmo + olmo-core artifacts and compare (supports --sweep and --jobs)",
    )
    run_parser.add_argument(
        "--dataset", action="append", help="image-only-v9 dataset name (repeatable)"
    )
    run_parser.add_argument(
        "--sweep", action="store_true", help="compare index 0 of all image-only-v9 datasets"
    )
    run_parser.add_argument("--index", type=int, default=0)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--seq_len", type=int, default=16384)
    run_parser.add_argument(
        "--jobs", "-j", type=int, default=1, help="parallel dataset workers (default: 1)"
    )
    run_parser.add_argument(
        "--inspect", action="store_true", help="print formatted text + image fingerprints"
    )
    run_parser.add_argument("--diff-tokens", action="store_true", help="decode and diff input_ids")
    run_parser.add_argument(
        "--keep-artifacts", action="store_true", help="retain artifacts even on success"
    )
    run_parser.add_argument(
        "--artifact-dir",
        help="directory for per-dataset artifact folders (default: temp dir, removed on full success)",
    )
    run_parser.add_argument("--conda", default=_DEFAULT_CONDA, help="path to conda executable")
    run_parser.add_argument("--mm-env", default="mm_olmo", help="conda env for export-mm")
    run_parser.add_argument(
        "--olmo-core-env", default="olmo-core", help="conda env for export-olmo-core"
    )
    run_parser.add_argument(
        "--mm-activate-script",
        default=_DEFAULT_MM_OLMO_ACTIVATE,
        help="bash script sourced for mm_olmo HF cache/offline env (default: mm_olmo-activate.sh)",
    )
    run_parser.add_argument("--script", help="path to this script (defaults to the invoked file)")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--mm-artifact", required=True)
    compare_parser.add_argument("--olmo-core-artifact", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--mm-artifact", required=True)
    inspect_parser.add_argument("--olmo-core-artifact", required=True)
    diff_parser = subparsers.add_parser("diff-tokens")
    diff_parser.add_argument("--mm-artifact", required=True)
    diff_parser.add_argument("--olmo-core-artifact", required=True)
    diff_parser.add_argument("--context", type=int, default=3)
    diff_parser.add_argument("--max-diffs", type=int, default=20)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _run_parity(args)
    if args.command == "inspect":
        inspect_artifacts(Path(args.mm_artifact), Path(args.olmo_core_artifact))
        return 0
    if args.command == "diff-tokens":
        diff_artifacts(
            Path(args.mm_artifact),
            Path(args.olmo_core_artifact),
            context=args.context,
            max_diffs=args.max_diffs,
        )
        return 0
    if args.command == "compare":
        differences = compare_artifacts(Path(args.mm_artifact), Path(args.olmo_core_artifact))
        if differences:
            print("MISMATCH")
            print("\n".join(f"  {difference}" for difference in differences))
            return 1
        print("OK")
        return 0
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
