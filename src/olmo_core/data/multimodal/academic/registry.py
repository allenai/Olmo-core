"""Registry and data loading for image-only-v9 academic datasets."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from os.path import join
from typing import Any, Callable, Dict, FrozenSet, List, Optional

import numpy as np
from PIL import Image

from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat
from olmo_core.data.multimodal.paths import (
    ACADEMIC_DATASETS,
    PIXMO_DATASETS,
    TORCH_DATASETS,
)
from olmo_core.exceptions import OLMoConfigurationError

from ..pixmo_clocks import format_pixmo_clocks_row
from .formatters import (
    format_cosyn,
    format_mc,
    format_message_list,
    format_vqa2_multi,
    format_vqa_short,
)

__all__ = [
    "ACADEMIC_REGISTRY",
    "STAGE2_EVAL_TRAIN_SETS",
    "build_academic_data",
    "format_academic_example",
    "coco_val2017_ids",
    "vg_coco_ids",
]

log = logging.getLogger(__name__)

VQA2_SOURCE = join(TORCH_DATASETS, "vqa2")
TEXT_VQA_SOURCE = join(TORCH_DATASETS, "text_vqa")
CHARTQA_SOURCE = join(TORCH_DATASETS, "chartqa")
DOCQA_SOURCE = join(TORCH_DATASETS, "docqa")
INFOQA_SOURCE = join(TORCH_DATASETS, "info_qa")
A_OKVQA_SOURCE = join(TORCH_DATASETS, "a_okvqa")
SCIENCE_QA_SOURCE = join(TORCH_DATASETS, "science_qa_img_only")
ST_QA_SOURCE = join(TORCH_DATASETS, "scene-text")
TALLY_QA_SOURCE = join(TORCH_DATASETS, "tally_qa")
COSYN_IMAGES = join(TORCH_DATASETS, "cosyn_images")
PLOT_QA_SOURCE = join(TORCH_DATASETS, "plot_qa")
#: Visual Genome's official image metadata; its ``coco_id`` column names the 51,498 VG images
#: that are COCO images.
VG_IMAGE_DATA = join(TORCH_DATASETS, "locca", "vg", "image_data.json")
#: COCO's official val2017 captions file; its ``images`` list is the 5,000 val2017 images.
COCO_VAL2017_ANNOTATIONS = join(TORCH_DATASETS, "coco", "annotations", "captions_val2017.json")

#: Training sets of the benchmarks the stage-2 checkpoints are evaluated on (mm_olmo
#: ``eval_molmo2.py`` ``IMAGE_TASKS`` / ``TEST_IMAGE_TASKS``), each mapped to that benchmark. They
#: stay stage-2 data: a stage-1 source is never one of them. ``tally_qa`` is on the list for its
#: images rather than its task: 36,327 of the 127,867 images it trains on are in the VQAv2 eval's
#: val2014 pool. Stage 2 keeps them (see :func:`_load_tally_qa`); stage 1 does not add them.
STAGE2_EVAL_TRAIN_SETS: Dict[str, str] = {
    "coco_2014_vqa_multi": "VQAv2",
    "text_vqa": "TextVQA",
    "chart_qa_weighted": "ChartQA",
    "doc_qa": "DocVQA",
    "info_qa": "InfographicVQA",
    "ai2_diagram_v2_mix_transparent": "AI2D",
    "a_okvqa_mc": "A-OKVQA",
    "a_okvqa_da": "A-OKVQA",
    "tally_qa": "VQAv2 (shared val2014 images)",
}

_TRAILING_ID = re.compile(r"(\d+)\.jpg$")


def _open_image(path: Any) -> Image.Image:
    if isinstance(path, Image.Image):
        return path
    if isinstance(path, np.ndarray):
        return Image.fromarray(path.astype("uint8"))
    return Image.open(path).convert("RGB")


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_vqa2_multi(split: str) -> List[Dict[str, Any]]:
    split = "val" if split == "validation" else split
    return _load_json(join(VQA2_SOURCE, f"molmo_{split}.json"))


def _load_text_vqa(split: str) -> List[Dict[str, Any]]:
    split = "val" if split == "validation" else split
    data = _load_json(join(TEXT_VQA_SOURCE, f"TextVQA_0.5.1_{split}.json"))
    out = []
    for ex in data["data"]:
        out.append(
            {
                "image": join(TEXT_VQA_SOURCE, f"{split}_images", ex["image_id"] + ".jpg"),
                "question": ex["question"],
                "answers": ex["answers"],
                "metadata": {
                    "image_url": ex.get("image_url"),
                    "image_id": ex["image_id"],
                    "example_id": ex["question_id"],
                },
            }
        )
    return out


def _load_a_okvqa(split: str, *, direct_answer: bool) -> List[Dict[str, Any]]:
    split = "val" if split == "validation" else split
    data = _load_json(join(A_OKVQA_SOURCE, f"aokvqa_v1p0_{split}.json"))
    out = []
    for ex in data:
        image = join(A_OKVQA_SOURCE, f"{split}2017", f"{ex['image_id']:0>12}.jpg")
        if direct_answer:
            if ex.get("difficult_direct_answer") and split in ("val", "test"):
                continue
            out.append(
                {
                    "image": image,
                    "question": ex["question"],
                    "answers": ex.get("direct_answers"),
                    "metadata": {"example_id": ex["question_id"]},
                }
            )
        else:
            item = {
                "image": image,
                "question": ex["question"],
                "options": ex["choices"],
                "metadata": {"example_id": ex["question_id"]},
            }
            if ex.get("correct_choice_idx") is not None:
                item["answer_idx"] = ex["correct_choice_idx"]
            out.append(item)
    return out


def _load_chart_qa(split: str) -> List[Dict[str, Any]]:
    split = "val" if split == "validation" else split
    out = []
    for kind in ("human", "augmented"):
        path = join(CHARTQA_SOURCE, split, f"{split}_{kind}.json")
        if not os.path.exists(path):
            continue
        for ex in _load_json(path):
            out.append(
                {
                    "image": join(CHARTQA_SOURCE, split, "png", ex["imgname"]),
                    "question": ex["query"],
                    "answers": ex[
                        "label"
                    ],  # bare string (mm_olmo parity: select_vqa_answer short-circuits on str)
                    "metadata": {"is_human": kind == "human", "example_id": ex.get("id")},
                }
            )
    return out


def _load_doc_qa(split: str) -> List[Dict[str, Any]]:
    split = "val" if split == "validation" else split
    suffix = "" if split == "test" else "_withQT"
    data = _load_json(join(DOCQA_SOURCE, f"{split}_v1.0{suffix}.json"))
    out = []
    for ex in data["data"]:
        answers = ex.get("answers") or [""]
        out.append(
            {
                "image": join(DOCQA_SOURCE, ex["image"]),
                "question": ex["question"],
                "answers": answers,
                "metadata": {
                    "doc_id": ex["docId"],
                    "question_types": ex.get("question_types"),
                    "example_id": ex["questionId"],
                },
            }
        )
    return out


def _load_info_qa(split: str) -> List[Dict[str, Any]]:
    if split == "validation":
        filename = "infographicsVQA_val_v1.0_withQT.json"
    else:
        filename = f"infographicsVQA_{split}_v1.0.json"
    data = _load_json(join(INFOQA_SOURCE, filename))
    out = []
    for ex in data["data"]:
        out.append(
            {
                "image": join(INFOQA_SOURCE, "images", ex["image_local_name"]),
                "question": ex["question"],
                "answers": ex.get("answers", [""]),
                "metadata": {"example_id": ex["questionId"]},
            }
        )
    return out


def _load_science_qa(split: str) -> List[Dict[str, Any]]:
    split = "val" if split == "validation" else split
    data = _load_json(join(SCIENCE_QA_SOURCE, f"{split}.json"))
    out = []
    for ex in data:
        question = ex["question"]
        if ex.get("hint"):
            question = ex["hint"].strip() + "\n" + question
        out.append(
            {
                "image": join(SCIENCE_QA_SOURCE, "images", ex["image"]),
                "question": question,
                "options": ex["choices"],
                "answer_idx": ex["answer"],
                "metadata": {"example_id": ex.get("id")},
            }
        )
    return out


def _load_st_qa(split: str) -> List[Dict[str, Any]]:
    if split == "val":
        split = "validation"
    file_split = "train" if split == "validation" else split
    path = join(ST_QA_SOURCE, f"{file_split}_task_3.json")
    data = _load_json(path)
    out = []
    for ex in data["data"]:
        out.append(
            {
                "image": join(ST_QA_SOURCE, ex["file_path"]),
                "question": ex["question"],
                "answers": ex.get("answers", []),
                "metadata": {"example_id": ex["question_id"]},
            }
        )
    if split in ("train", "validation"):
        # Synthetic val split (mm_olmo SceneTextQaConfig): no official validation split.
        out.sort(key=lambda x: x["metadata"]["example_id"])
        np.random.RandomState(63069).shuffle(out)
        if split == "train":
            return out[1024:]
        return out[:1024]
    return out


def _resolve_tally_image(image_id: str) -> str:
    for sub in ("train2014", "val2014", "VG_100K", "VG_100K_2"):
        path = join(TALLY_QA_SOURCE, sub, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return join(TALLY_QA_SOURCE, "VG_100K", f"{image_id}.jpg")


def _read_heldout_json(path: str) -> Any:
    if not os.path.exists(path):
        raise OLMoConfigurationError(
            f"held-out set {path!r} not found, so train/eval image overlap cannot be checked"
        )
    return _load_json(path)


@lru_cache(maxsize=1)
def coco_val2017_ids() -> FrozenSet[int]:
    """COCO ids of the 5,000 val2017 images.

    COCO keeps an image's id across its 2014 and 2017 splits, so these ids also find val2017
    images filed under a 2014 name (all 5,000 are val2014 images).

    :raises OLMoConfigurationError: If the annotation file is missing.
    """
    return frozenset(int(im["id"]) for im in _read_heldout_json(COCO_VAL2017_ANNOTATIONS)["images"])


@lru_cache(maxsize=1)
def vg_coco_ids() -> Dict[int, int]:
    """Visual Genome image id -> COCO id, for the VG images that are COCO images.

    :raises OLMoConfigurationError: If VG's ``image_data.json`` is missing.
    """
    return {
        int(ex["image_id"]): int(ex["coco_id"])
        for ex in _read_heldout_json(VG_IMAGE_DATA)
        if ex.get("coco_id")
    }


def _tally_coco_id(image: str) -> Optional[int]:
    """The COCO id of a TallyQA image (``train2014/...`` / ``val2014/...`` / ``VG_100K*/<id>.jpg``),
    or None for a VG image that is not a COCO image."""
    src, fname = image.split("/")
    image_id = int(_TRAILING_ID.search(fname).group(1))  # type: ignore[union-attr]
    return vg_coco_ids().get(image_id) if src.startswith("VG_") else image_id


def _load_tally_qa(split: str) -> List[Dict[str, Any]]:
    """TallyQA, grouped by image.

    The train split holds 5,114 COCO val2017 images (:func:`coco_val2017_ids`; 9,578 of its 249,318
    questions): 4,052 filed as COCO val2014 images, and 1,062 Visual Genome images that are val2017
    images under a VG file name. They are dropped, leaving 127,867 images. The rest of val2014 is
    kept. Other splits are returned whole.
    """
    split = "val" if split == "validation" else split
    data = _load_json(join(TALLY_QA_SOURCE, f"{split}.json"))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ex in data:
        grouped.setdefault(ex["image"], []).append(ex)
    if split == "train":
        heldout = coco_val2017_ids()
        n_before = len(grouped)
        grouped = {
            image: qs for image, qs in grouped.items() if _tally_coco_id(image) not in heldout
        }
        log.info(
            "tally_qa train: dropped %d of %d images that are COCO val2017 images",
            n_before - len(grouped),
            n_before,
        )
    image_sources = {
        "train2014": VQA2_SOURCE,
        "val2014": VQA2_SOURCE,
        "VG_100K": TALLY_QA_SOURCE,
        "VG_100K_2": TALLY_QA_SOURCE,
    }
    out = []
    for image, questions in grouped.items():
        image_id = questions[0]["image_id"]
        image_src, path = image.split("/")
        image_path = join(image_sources[image_src], image_src, path)
        out.append(
            {
                "image": image_path,
                "questions": questions,
                "metadata": {"image_id": image_id},
            }
        )
    return out


def _load_plot_qa(split: str) -> List[Dict[str, Any]]:
    split = "val" if split == "validation" else split
    return _load_json(join(PLOT_QA_SOURCE, f"molmo_{split}.json"))


def _load_hf_academic(name: str, split: str) -> Any:
    ds = load_from_disk_compat(join(ACADEMIC_DATASETS, name))
    split = "validation1" if name == "figure_qa" and split == "validation" else split
    split = "dev" if name == "tabwmp" and split == "validation" else split
    return ds[split] if hasattr(ds, "keys") and split in ds else ds


def _load_cosyn(doc_type: str, split: str) -> Any:
    path = join(PIXMO_DATASETS, f"cosyn-{doc_type}")
    ds = load_from_disk_compat(path)
    return ds[split] if hasattr(ds, "keys") and split in ds else ds


def _load_clocks(split: str) -> List[Dict[str, Any]]:
    path = join(PIXMO_DATASETS, "clocks", f"{split}.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def _format_clocks_row(row: Dict[str, Any], rng: np.random.RandomState) -> Dict[str, Any]:
    return format_pixmo_clocks_row(row, rng, aug=True)


@dataclass(frozen=True)
class AcademicSpec:
    name: str
    loader: Callable[[str], Any]
    formatter: Callable[[Any, np.random.RandomState, str], Dict[str, Any]]


def _format_chart_qa_weighted(ex, _rng, _split):
    formatted = format_vqa_short(
        {**ex, "example_id": ex["metadata"].get("example_id")},
        style="chart_qa",
    )
    is_human = ex["metadata"]["is_human"]
    formatted["weight"] = (2 * 20901 / 28299) if is_human else (2 * 7398 / 28299)
    return formatted


def _format_okvqa(ex, _rng, _split):
    return format_vqa_short(
        {
            "image": ex["image"],
            "question": ex["question"],
            "answers": [x["raw_answer"] for x in ex["answers"]],
            "example_id": ex["question_id"],
        },
        style="okvqa",
    )


def _format_ai2d(ex, _rng, _split):
    options = ex["answer_texts"]
    item = {
        "image": ex["image"],
        "question": ex["question"],
        "answer_idx": ex["correct_answer"],
        "metadata": {
            "example_id": ex["question_id"],
            "image_id": ex["image_id"],
            "abc_label": ex["abc_label"],
            "has_transparent_box": ex["has_transparent_box"],
        },
    }
    if ex["abc_label"] and sum(ex["option_is_abc"]) >= (len(options) - 1):
        item["unlabelled_options"] = [
            opt.upper() if abc else opt for opt, abc in zip(options, ex["option_is_abc"])
        ]
        style = "ai2_diagram_no_letter"
    else:
        item["options"] = options
        style = "ai2_diagram"
    return format_mc(item, style=style)


def _format_tabwmp(ex, _rng, _split):
    return format_vqa_short(
        {
            "image": ex["image"],
            "question": ex["question"],
            "answer": ex["answer"],
            "example_id": ex["example_id"],
        },
        style="tabwmp_da",
    )


def _format_tally_qa(ex, _rng, _split):
    messages = [
        {"question": q["question"], "answer": str(q["answer"]), "style": "tally_qa"}
        for q in ex["questions"]
    ]
    return format_message_list(
        {"image": ex["image"], "message_list": messages, "metadata": ex.get("metadata", {})},
        style="tally_qa",
    )


def _format_plot_qa_molmo(ex, _rng, _split):
    messages = [
        {"question": q, "answer": str(a), "style": "plot_qa"}
        for q, a in zip(ex["questions"], ex["answers"])
    ]
    return format_message_list(
        {
            "image": ex["image"],
            "message_list": messages,
            "metadata": {"image_id": ex.get("image_index")},
        },
        style="plot_qa",
    )


def _format_multi_qa(ex, _rng, _split, *, style: str):
    qas = ex["questions"]
    messages = [
        {"question": q, "answer": str(a), "style": style}
        for q, a in zip(qas["question"], qas["answer"])
    ]
    return format_message_list(
        {
            "image": ex["image"],
            "message_list": messages,
            "metadata": {"image_id": ex.get("image_id")},
        },
        style=style,
    )


ACADEMIC_REGISTRY: Dict[str, AcademicSpec] = {
    "coco_2014_vqa_multi": AcademicSpec(
        "coco_2014_vqa_multi", _load_vqa2_multi, lambda e, r, s: format_vqa2_multi(e)
    ),
    "text_vqa": AcademicSpec(
        "text_vqa",
        _load_text_vqa,
        lambda e, r, s: format_vqa_short(
            {**e, "example_id": e["metadata"]["example_id"]}, style="text_vqa"
        ),
    ),
    "okvqa": AcademicSpec("okvqa", lambda s: _load_hf_academic("okvqa", s), _format_okvqa),
    "chart_qa_weighted": AcademicSpec(
        "chart_qa_weighted",
        _load_chart_qa,
        _format_chart_qa_weighted,
    ),
    "doc_qa": AcademicSpec(
        "doc_qa",
        _load_doc_qa,
        lambda e, r, s: format_vqa_short(
            {**e, "example_id": e["metadata"]["example_id"]}, style="doc_qa"
        ),
    ),
    "info_qa": AcademicSpec(
        "info_qa",
        _load_info_qa,
        lambda e, r, s: format_vqa_short(
            {**e, "example_id": e["metadata"]["example_id"]}, style="info_qa"
        ),
    ),
    "ai2_diagram_v2_mix_transparent": AcademicSpec(
        "ai2_diagram_v2_mix_transparent", lambda s: _load_hf_academic("ai2d", s), _format_ai2d
    ),
    "a_okvqa_mc": AcademicSpec(
        "a_okvqa_mc",
        lambda s: _load_a_okvqa(s, direct_answer=False),
        lambda e, r, s: format_mc({**e, "metadata": e["metadata"]}, style="a_okvqa_mc"),
    ),
    "a_okvqa_da": AcademicSpec(
        "a_okvqa_da",
        lambda s: _load_a_okvqa(s, direct_answer=True),
        lambda e, r, s: format_vqa_short(
            {**e, "example_id": e["metadata"]["example_id"]}, style="a_okvqa_da"
        ),
    ),
    "science_qa_img": AcademicSpec(
        "science_qa_img",
        _load_science_qa,
        lambda e, r, s: format_mc({**e, "metadata": e["metadata"]}, style="science_qa"),
    ),
    "tabwmp_da": AcademicSpec(
        "tabwmp_da", lambda s: _load_hf_academic("tabwmp", s), _format_tabwmp
    ),
    "st_qa": AcademicSpec(
        "st_qa",
        _load_st_qa,
        lambda e, r, s: format_vqa_short(
            {**e, "example_id": e["metadata"]["example_id"]}, style="st_qa"
        ),
    ),
    "tally_qa": AcademicSpec("tally_qa", _load_tally_qa, _format_tally_qa),
    "pixmo_clocks": AcademicSpec(
        "pixmo_clocks",
        _load_clocks,
        lambda e, r, s: _format_clocks_row(e, r),
    ),
    "dv_qa": AcademicSpec(
        "dv_qa",
        lambda s: _load_hf_academic("dv_qa", s),
        lambda e, r, s: _format_multi_qa(e, r, s, style="dv_qa"),
    ),
    "figure_qa": AcademicSpec(
        "figure_qa",
        lambda s: _load_hf_academic("figure_qa", s),
        lambda e, r, s: _format_multi_qa(e, r, s, style="figure_qa"),
    ),
    "plot_qa": AcademicSpec("plot_qa", _load_plot_qa, _format_plot_qa_molmo),
}


for _doc_type, _use_exp in [
    ("chart", True),
    ("chemical", True),
    ("diagram", True),
    ("document", False),
    ("math", True),
    ("music", True),
    ("table", True),
]:
    _name = f"cosyn_{_doc_type}{'_exp' if _use_exp else ''}"
    if _name == "cosyn_document":
        _name = "cosyn_document"

    def _make_cosyn_loader(dt=_doc_type):
        return lambda s: _load_cosyn(dt, s)

    def _make_cosyn_formatter(dt=_doc_type, ue=_use_exp):
        return lambda e, r, s: format_cosyn(e, doc_type=dt, use_exp=ue)

    ACADEMIC_REGISTRY[_name] = AcademicSpec(_name, _make_cosyn_loader(), _make_cosyn_formatter())


@lru_cache(maxsize=64)
def build_academic_data(name: str, split: str = "train"):
    if name not in ACADEMIC_REGISTRY:
        raise KeyError(f"Unknown academic dataset: {name}")
    return ACADEMIC_REGISTRY[name].loader(split)


def format_academic_example(name: str, example: Any, rng, split: str = "train") -> Dict[str, Any]:
    """Format one raw row. ``rng`` is a RandomState (or an int seed for one-off use)."""
    spec = ACADEMIC_REGISTRY[name]
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    formatted = spec.formatter(example, rng, split)
    if "image" in formatted:
        img = formatted["image"]
        if isinstance(img, np.ndarray):
            formatted = dict(formatted)
            formatted["image"] = Image.fromarray(img.astype("uint8"))
        elif not isinstance(img, Image.Image):
            formatted = dict(formatted)
            formatted["image"] = _open_image(img)
    return formatted
