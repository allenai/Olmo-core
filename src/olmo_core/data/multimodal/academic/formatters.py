"""Format functions for image-only-v9 academic datasets (ported from mm_olmo)."""

from __future__ import annotations

from typing import Any, Dict

__all__ = [
    "format_vqa2_multi",
    "format_vqa_short",
    "format_mc",
    "format_message_list",
    "format_cosyn",
    "format_clocks",
]


def format_vqa2_multi(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": {"image_id": example["image_id"]},
        "image": example["image"],
        "message_list": example["messages"],
    }


def format_vqa_short(example: Dict[str, Any], *, style: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "style": style,
        "metadata": {
            k: example[k] for k in ("image_id", "example_id", "question_id") if k in example
        },
        "image": example["image"],
        "question": example["question"],
    }
    if example.get("answers") is not None:
        out["answers"] = example["answers"]
    if "answer" in example:
        out["answer"] = example["answer"]
    if "options" in example:
        out["options"] = example["options"]
    return out


def format_mc(example: Dict[str, Any], *, style: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "image": example["image"],
        "question": example["question"],
        "style": style,
        "metadata": example.get("metadata", {}),
    }
    if "options" in example:
        out["options"] = example["options"]
    if "unlabelled_options" in example:
        out["unlabelled_options"] = example["unlabelled_options"]
    if "answer_idx" in example:
        out["answer_idx"] = example["answer_idx"]
    if "answers" in example:
        out["answers"] = example["answers"]
    return out


def format_message_list(example: Dict[str, Any], *, style: str) -> Dict[str, Any]:
    return {
        "image": example["image"],
        "message_list": example["message_list"],
        "metadata": example.get("metadata", {}),
    }


def format_cosyn(example: Dict[str, Any], *, doc_type: str, use_exp: bool) -> Dict[str, Any]:
    style = f"cosyn_{doc_type}"
    if use_exp:
        style += "_exp"
    qeas = example["qa_pairs"]
    if use_exp:
        message_list = [
            {"question": q, "explanation": e, "answer": a, "style": style}
            for q, e, a in zip(qeas["question"], qeas["explanation"], qeas["answer"])
        ]
    else:
        message_list = [
            {"question": q, "answer": a, "style": style}
            for q, a in zip(qeas["question"], qeas["answer"])
        ]
    return {
        "image": example["image"],
        "message_list": message_list,
        "metadata": {"image_id": example["id"]},
    }


def format_clocks(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image": example["image"],
        "prompt": "What time is being shown?",
        "text": example["text"],
        "style": "clocks",
        "metadata": {
            "hour": example.get("hour"),
            "minute": example.get("minute"),
            "second": example.get("second"),
        },
    }
