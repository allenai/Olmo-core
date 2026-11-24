#!/usr/bin/env python3
"""Image overlap of the stage-2 SFT image sources with the stage-1 (v6) visual sources the Olmo 3.5
multimodal mid-training checkpoints trained on.

Images are matched by the identifiers the datasets carry, not by pixels:

* PixMo sources (captions, AMA, CapQA, correction QA, points, counts, multi-image points) by
  ``image_url``, against the union of stage-1's PixMo-Cap (caption and transcript, the
  content-disjoint split), PixMo-Points v2 and PixMo-Count v2;
* CoSyn pointing by row ``id``, against stage-1's CoSyn-Point v2;
* TextVQA by OpenImages id, ST-VQA by file name, against stage-1's TextOCR images;
* PixMo-Clocks by ``image_id``: stage 1 drew from the same 800K-image pool.

The stage-1 pools are whole training splits, so a count is an upper bound on what stage 1
actually drew. Sources with no stage-1 pool in common (COCO-based VQA, DocVQA, InfoVQA, ChartQA,
AI2D, ScienceQA, TabMWP, TallyQA, Mantis, CoSyn multi-document) are listed with
``"checked": false`` and the reason::

    PYTHONPATH=src python src/scripts/olmo35_sft_mm_overlap.py --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tarfile
from pathlib import Path

PIXMO = "/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets"
STAGE1 = {
    "pixmo_cap_content_disjoint": "/weka/oe-training-default/jasonr/olmo35-vision-alignment/"
    "artifacts/pixmo-cap-content-disjoint-v1/dataset",
    "pixmo_points_v2": "/weka/oe-training-default/chrisc/pixmo-points-with-masks-v17",
    "pixmo_count_v2": f"{PIXMO}/count-v2",
    "cosyn_point_v2": f"{PIXMO}/cosyn-point-v2-masks",
    "textocr": "/weka/oe-training-default/oe-encoder/textocr_v6_tars",
}
UNCHECKED = {
    "coco_2014_vqa_multi": "COCO images; no stage-1 source uses COCO",
    "okvqa": "COCO images; no stage-1 source uses COCO",
    "a_okvqa_mc": "COCO images; no stage-1 source uses COCO",
    "a_okvqa_da": "COCO images; no stage-1 source uses COCO",
    "tally_qa": "Visual Genome / COCO images; no stage-1 source uses them",
    "chart_qa_weighted": "ChartQA charts; stage-1 charts are synthetic (CoSyn, text-rich)",
    "doc_qa": "DocVQA (UCSF industry documents); stage-1 documents are olmOCR PDFs and synthetic",
    "info_qa": "InfographicVQA; no stage-1 source draws from it",
    "ai2_diagram_v2_mix_transparent": "AI2D diagrams; no stage-1 source draws from it",
    "science_qa_img": "ScienceQA; no stage-1 source draws from it",
    "tabwmp_da": "TabMWP rendered tables; no stage-1 source draws from it",
    "mantis_instruct_llava_665k_multi_multi_only": "Mantis-Instruct; no stage-1 source",
    "mantis_instruct_nlvr2_multi_only": "Mantis-Instruct; no stage-1 source",
    "mantis_instruct_spot-the-diff_multi_only": "Mantis-Instruct; no stage-1 source",
    **{
        f"cosyn_multidoc_{t}_exp": "CoSyn multi-document sets, generated separately from the "
        "single-document sets stage 1 used"
        for t in ("chart", "chemical", "diagram", "doc", "music", "table")
    },
}


def _split(path: str, split: str = "train"):
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

    ds = load_from_disk_compat(path)
    if hasattr(ds, "keys"):
        return ds[split] if split in ds else ds[list(ds.keys())[0]]
    return ds


def _column(path: str, column: str, split: str = "train") -> list:
    return _split(path, split).data.column(column).to_pylist()


def _urls(path: str, split: str = "train") -> set:
    return {u for u in _column(path, "image_url", split) if u}


def _report(name: str, ids: set, pool: set, pool_name: str, how: str) -> dict:
    shared = ids & pool
    return {
        "source": name,
        "checked": True,
        "matched_by": how,
        "stage1_pool": pool_name,
        "images": len(ids),
        "shared_with_stage1": len(shared),
        "share": round(len(shared) / len(ids), 4) if ids else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from olmo_core.data.multimodal.academic.registry import build_academic_data

    s1_pixmo = (
        _urls(STAGE1["pixmo_cap_content_disjoint"])
        | {u for u in _column(STAGE1["pixmo_points_v2"], "image_url") if u}
        | _urls(STAGE1["pixmo_count_v2"])
    )
    pixmo_pool = "stage-1 PixMo-Cap + PixMo-Points v2 + PixMo-Count v2 (image_url)"
    rows = []
    for name, ids in (
        ("pixmo_cap", _urls(f"{PIXMO}/cap")),
        ("pixmo_ask_model_anything", _urls(f"{PIXMO}/ask-model-anything")),
        ("pixmo_cap_qa_as_user_qa", _urls(f"{PIXMO}/cap-qa")),
        ("pixmo_count_train", _urls(f"{PIXMO}/count")),
    ):
        rows.append(_report(name, ids, s1_pixmo, pixmo_pool, "image_url"))
    for name, sub in (
        ("pixmo_points_train", "points-pointing"),
        ("pixmo_points_high_freq_train", "points-counting"),
    ):
        path = f"{PIXMO}/{sub}"
        if os.path.isdir(path):
            rows.append(_report(name, _urls(path), s1_pixmo, pixmo_pool, "image_url"))
        else:
            rows.append({"source": name, "checked": False, "reason": f"{path} not found"})
    multi = {u for urls in _column(f"{PIXMO}/pixmo-multi-points", "image_urls") for u in urls}
    rows.append(_report("pixmo_multi_points", multi, s1_pixmo, pixmo_pool, "image_url"))
    with open(f"{PIXMO}/correction-qa/train-records.json") as f:
        records = json.load(f)
    correction = {u for r in records for u in r.get("imageUrls", [r.get("imageUrl")]) if u}
    rows.append(
        _report("correction_qa_multi_only_max5", correction, s1_pixmo, pixmo_pool, "image_url")
    )
    rows.append(
        _report(
            "cosyn_point",
            set(_column(f"{PIXMO}/cosyn-point", "id")),
            set(_column(STAGE1["cosyn_point_v2"], "id")),
            "stage-1 CoSyn-Point v2 (id)",
            "id",
        )
    )
    textocr = set()
    for tar in sorted(glob.glob(os.path.join(STAGE1["textocr"], "*.tar"))):
        with tarfile.open(tar) as t:
            textocr |= {Path(m).stem for m in t.getnames() if m.endswith(".jpg")}
    textvqa = {Path(str(r["image"])).stem for r in build_academic_data("text_vqa")}
    rows.append(
        _report("text_vqa", textvqa, textocr, "stage-1 TextOCR (OpenImages id)", "image id")
    )
    stqa = {Path(str(r["image"])).stem for r in build_academic_data("st_qa")}
    rows.append(_report("st_qa", stqa, textocr, "stage-1 TextOCR (file name)", "file name"))
    clocks = build_academic_data("pixmo_clocks")
    rows.append(
        {
            "source": "pixmo_clocks",
            "checked": True,
            "matched_by": "dataset",
            "stage1_pool": "stage-1 PixMo-Clocks: the same dataset",
            "images": len({r["image_id"] for r in clocks}),
            "shared_with_stage1": "all (same pool)",
            "share": 1.0,
        }
    )
    for name, reason in UNCHECKED.items():
        rows.append({"source": name, "checked": False, "reason": reason})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap.json").write_text(json.dumps({"stage1": STAGE1, "sources": rows}, indent=1))
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
