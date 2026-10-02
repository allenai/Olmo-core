"""Greedy multimodal decoding and explicit counting, pointing, OCR and lexical diagnostics.

These diagnostics retain invalid and unfinished outputs. They are not official benchmark
implementations. Distributed decoding keeps every expert-parallel rank participating until
all ranks finish.
"""

import json
import logging
import re
import time
from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.utils import move_to_device

DEFAULT_SOURCES = ("scalar_count", "pixmo_points_basic", "ocr_document", "pixmo_caption")
TOKEN_LIMITS = {
    "scalar_count": 16,
    "pixmo_points_basic": 256,
    "pixmo_points_high_frequency": 256,
    "cosyn_point": 256,
    "ocr_document": 64,
    "pixmo_caption": 512,
}
log = logging.getLogger(__name__)


def _collective_error(error: str | None) -> None:
    errors = [error]
    if is_distributed():
        errors = [None] * get_world_size()
        dist.all_gather_object(errors, error)
    if any(item is not None for item in errors):
        raise RuntimeError(f"Decoded evaluation failed collectively: {errors}")


def source_indices(evaluator, batch_index: int, rank_batch_size: int) -> list[dict[str, int]]:
    """Resolve rank-local panel positions back to the source's original row indices."""
    loader = evaluator.batches
    first = (batch_index * loader.dp_world_size + loader.dp_rank) * rank_batch_size
    subset = loader.dataset
    indices = getattr(subset.dataset, "indices", None)
    return [
        {
            "panel_index": index,
            "selected_source_index": int(subset.indices[index]),
            "base_source_index": (
                int(indices[int(subset.indices[index])])
                if indices is not None
                else int(subset.indices[index])
            ),
        }
        for index in range(first, first + rank_batch_size)
    ]


def prepare_prompts(batch, tokenizer, identities, source):
    """Clone exact first-annotation prompts and remove every gold response/other branch."""
    inputs, labels, weights = (batch[key] for key in ("input_ids", "labels", "loss_masks"))
    if inputs.ndim != 2 or labels.shape != inputs.shape or weights.shape != inputs.shape:
        raise ValueError("Expected equally shaped input IDs, shifted labels, and loss masks")
    if len(identities) != len(inputs):
        raise ValueError("Each input needs an explicit source identity")
    model_inputs = {
        key: batch[key].clone()
        for key in ("input_ids", "position_ids", "token_type_ids", "subsegment_ids", "example_ids")
        if key in batch
    }
    for key in ("position_ids", "token_type_ids"):
        if key not in model_inputs or model_inputs[key].shape != inputs.shape:
            raise ValueError(f"Saved multimodal inputs require {key}")
    model_inputs["router_token_mask"] = torch.zeros_like(inputs, dtype=torch.bool)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    records = []
    lengths = []
    for index, identity in enumerate(identities):
        positions = (weights[index] > 0).nonzero(as_tuple=True)[0].tolist()
        if not positions:
            raise ValueError(f"No supervised response at {source} row {identity}")
        start = positions[0]
        end = start
        while end + 1 < inputs.shape[1] and weights[index, end + 1] > 0:
            if labels[index, end].item() == tokenizer.eos_token_id:
                break
            end += 1
        targets = labels[index, start : end + 1].tolist()
        complete = (
            targets[-1] == tokenizer.eos_token_id and tokenizer.eos_token_id not in targets[:-1]
        )
        if any(token < 0 for token in targets):
            raise ValueError("Supervised reference contains an ignored/negative token ID")
        length = start + 1
        # The last prompt token predicts the first gold response token. That response
        # token and every subsequent original token must be removed before decoding.
        model_inputs["input_ids"][index, length:] = pad
        model_inputs["token_type_ids"][index, length:] = 0
        model_inputs["router_token_mask"][index, :length] = True
        last_position = int(model_inputs["position_ids"][index, start])
        model_inputs["position_ids"][index, length:] = torch.arange(
            last_position + 1,
            last_position + 1 + inputs.shape[1] - length,
            device=inputs.device,
        )
        for key in ("subsegment_ids", "example_ids"):
            if key in model_inputs:
                model_inputs[key][index, length:] = model_inputs[key][index, start]
        records.append(
            {
                **identity,
                "source": source,
                "annotation_index": 0,
                "prompt_token_ids": inputs[index, :length].tolist(),
                "text_prompt": tokenizer.decode(
                    inputs[index, :length][batch["token_type_ids"][index, :length] == 0].tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "reference_token_ids": targets,
                "reference_complete": complete,
                "reference": tokenizer.decode(
                    targets[:-1] if complete else targets,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "generated_token_ids": [],
                "stop_reason": "context_limit" if length == inputs.shape[1] else None,
            }
        )
        lengths.append(length)
    return model_inputs, lengths, records


def _globally_finished(finished: bool, device) -> bool:
    value = torch.tensor(int(finished), device=device)
    if is_distributed():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(value)


def decode_batch(
    module,
    batch,
    tokenizer,
    token_ids,
    identities,
    source,
    *,
    score_source: str | None = None,
    max_new_tokens: int | None = None,
):
    """Decode equal-shaped rank-local rows while every rank participates until global stop."""
    scoring_source = score_source or source
    token_limit = TOKEN_LIMITS[scoring_source] if max_new_tokens is None else max_new_tokens
    if token_limit <= 0:
        raise ValueError("max_new_tokens must be positive")
    started = time.perf_counter()
    error = None
    try:
        model_inputs, lengths, records = prepare_prompts(batch, tokenizer, identities, source)
        pooled = batch["pooled_patches_idx"]
        if pooled.ndim != 3 or pooled.shape[0] != len(records):
            raise ValueError("Expected one pooled image-patch map per input row")
        expected = (pooled >= 0).any(dim=-1).sum(dim=-1).cpu()
        present = model_inputs["input_ids"].eq(token_ids.im_patch_id).sum(dim=-1).cpu()
        if not torch.equal(expected, present):
            raise ValueError("The original prompt must retain exactly every pooled image patch")
        model_inputs = move_to_device(model_inputs, module.device)
    except Exception as exc:  # noqa: BLE001 - synchronize preparation before model collectives
        error = f"Prompt preparation: {type(exc).__name__}: {exc}"
    _collective_error(error)
    image_special_ids = set(token_ids.image_token_ids) | {token_ids.image_placeholder_id}
    invalid_special_ids = set(tokenizer.all_special_ids) - {tokenizer.eos_token_id}
    for part in module.model_parts:
        part.eval()
    with torch.compiler.set_stance("force_eager"), module._multimodal_eval_batch_context():
        # Keep the proven grad-enabled attention regime. Detaching projected features
        # avoids retaining the vision graph across autoregressive forwards.
        features = module.multimodal_model.encode_images(
            batch["images"], batch["pooled_patches_idx"]
        ).detach()
        for step in range(token_limit):
            if _globally_finished(
                all(row["stop_reason"] is not None for row in records), module.device
            ):
                break
            positions = torch.tensor(lengths, device=module.device).unsqueeze(1) - 1
            logits = module.model_forward_no_pipeline(
                **model_inputs,
                encoded_image_features=features,
                logits_to_keep=positions,
            )
            error = None
            try:
                if not isinstance(logits, torch.Tensor) or logits.shape[:2] != (len(records), 1):
                    raise ValueError("Expected one vocabulary distribution per input row")
                usable = (
                    (
                        ~(torch.isnan(logits) | torch.isposinf(logits)).any(dim=-1)
                        & torch.isfinite(logits).any(dim=-1)
                    )
                    .squeeze(1)
                    .cpu()
                    .tolist()
                )
                predictions = logits.detach().argmax(dim=-1).squeeze(1).cpu().tolist()
                for index, (row, prediction) in enumerate(zip(records, predictions)):
                    if row["stop_reason"] is not None:
                        continue
                    if not usable[index]:
                        row["stop_reason"] = "nonfinite_logits"
                        continue
                    row["generated_token_ids"].append(prediction)
                    if prediction == tokenizer.eos_token_id:
                        row["stop_reason"] = "eos"
                    elif prediction in image_special_ids:
                        row["stop_reason"] = "image_special_token"
                    elif prediction in invalid_special_ids or prediction >= len(tokenizer):
                        row["stop_reason"] = "invalid_token"
                    elif lengths[index] >= model_inputs["input_ids"].shape[1]:
                        row["stop_reason"] = "context_limit"
                    elif step + 1 == token_limit:
                        row["stop_reason"] = "max_tokens"
                    else:
                        model_inputs["input_ids"][index, lengths[index]] = prediction
                        model_inputs["router_token_mask"][index, lengths[index]] = True
                        lengths[index] += 1
            except Exception as exc:  # noqa: BLE001 - synchronize before next forward
                error = f"Decode step {step}: {type(exc).__name__}: {exc}"
            del logits
            _collective_error(error)
            if get_rank() == 0 and (step == 0 or (step + 1) % 16 == 0):
                log.info(
                    "Decoded %s token step %d/%d, %.1fs elapsed",
                    source,
                    step + 1,
                    token_limit,
                    time.perf_counter() - started,
                )
        del features
    error = None
    try:
        for row in records:
            generated = row["generated_token_ids"]
            text_ids = generated[:-1] if row["stop_reason"] == "eos" else generated
            # Invalid model-vocabulary IDs cannot be decoded by the tokenizer; preserve
            # them in generated_token_ids instead of silently dropping them from scoring.
            row["prediction"] = (
                tokenizer.decode(
                    text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                if all(token < len(tokenizer) for token in text_ids)
                else None
            )
            row["output_tokens"] = len(generated)
            row["batch_decode_seconds"] = time.perf_counter() - started
            row["metrics"] = score_prediction(scoring_source, row)
    except Exception as exc:  # noqa: BLE001 - synchronize before next batch
        error = f"Decoded scoring: {type(exc).__name__}: {exc}"
    _collective_error(error)
    return records


def _normalized_text(text):
    return " ".join(text.lower().split())


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, value in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (value != other))
            )
        previous = current
    return previous[-1]


def parse_points(text):
    """Parse the existing single-image html-v2 answer, without repairing malformed outputs."""
    text = text.strip()
    if text == "There are none.":
        return np.empty((0, 2), dtype=np.float64)
    match = re.fullmatch(r'<points coords="([0-9 ]+)">.*?</points>', text, flags=re.DOTALL)
    if match is None:
        return None
    coordinates = [int(value) for value in match.group(1).split()]
    if len(coordinates) < 4 or coordinates[0] != 1 or (len(coordinates) - 1) % 3:
        return None
    triplets = np.asarray(coordinates[1:]).reshape(-1, 3)
    if not np.array_equal(triplets[:, 0], np.arange(1, len(triplets) + 1)):
        return None
    if np.any(triplets[:, 1:] > 1000):
        return None
    return triplets[:, 1:] / 1000.0


def score_prediction(source, row):
    """Score explicit decoded diagnostics; failed or unfinished predictions never disappear."""
    reference, prediction = row["reference"], row["prediction"]
    valid = row["stop_reason"] == "eos" and prediction is not None
    result: dict[str, Any] = {
        "reference_supported": row["reference_complete"],
        "valid_output": valid,
    }
    if source == "scalar_count":
        gold_count = re.fullmatch(r"\s*([0-9]+)\s*", reference)
        parsed_count = re.fullmatch(r"\s*([0-9]+)\s*", prediction or "") if valid else None
        result.update(
            reference_supported=row["reference_complete"] and gold_count is not None,
            parsed_count=int(parsed_count.group(1)) if parsed_count else None,
            strict_count_accuracy=float(
                bool(
                    row["reference_complete"]
                    and gold_count
                    and parsed_count
                    and int(gold_count.group(1)) == int(parsed_count.group(1))
                )
            ),
        )
    elif source in ("pixmo_points_basic", "pixmo_points_high_frequency", "cosyn_point"):
        from scipy.optimize import linear_sum_assignment

        gold, parsed = parse_points(reference), parse_points(prediction or "") if valid else None
        supported = row["reference_complete"] and gold is not None
        matches = 0
        if supported and parsed is not None and len(gold) and len(parsed):
            distances = np.linalg.norm(parsed[:, None, :] - gold[None, :, :], axis=-1)
            # Maximize the number of pairs under threshold, then minimize distance.
            costs = (distances > 0.05) * 2 * (min(len(gold), len(parsed)) + 1) + distances
            predicted_indices, gold_indices = linear_sum_assignment(costs)
            matches = int((distances[predicted_indices, gold_indices] <= 0.05).sum())
        score = 0.0
        if supported and parsed is not None:
            denominator = len(gold) + len(parsed)
            score = 2 * matches / denominator if denominator else 1.0
        result.update(
            reference_supported=supported,
            point_parse_valid=parsed is not None,
            reference_points=len(gold) if gold is not None else None,
            predicted_points=len(parsed) if parsed is not None else None,
            matched_points=matches,
            point_f1_at_005=score,
        )
    elif source == "ocr_document":
        gold = _normalized_text(reference)
        parsed = _normalized_text(prediction or "")
        distance = _edit_distance(gold, parsed)
        result.update(
            normalized_exact_match=float(
                bool(row["reference_complete"] and valid and gold == parsed)
            ),
            normalized_character_error_rate=distance / max(1, len(gold)),
            edit_distance=distance,
            reference_characters=len(gold),
        )
    elif source == "pixmo_caption":
        gold = Counter(re.findall(r"\w+", reference.lower()))
        parsed = Counter(re.findall(r"\w+", (prediction or "").lower())) if valid else Counter()
        overlap = sum((gold & parsed).values())
        denominator = sum(gold.values()) + sum(parsed.values())
        result["lexical_token_f1"] = (
            2 * overlap / denominator if row["reference_complete"] and denominator else 0.0
        )
    return result


def check_scorers():
    """Exercise every scorer and JSON serialization on CPU before loading a model."""
    fixtures = {
        "scalar_count": ("7", "strict_count_accuracy"),
        "pixmo_points_basic": ('<points coords="1 1 100 200">dog</points>', "point_f1_at_005"),
        "ocr_document": ("Red car", "normalized_exact_match"),
        "pixmo_caption": ("A red car", "lexical_token_f1"),
    }
    for source, (answer, metric) in fixtures.items():
        result = score_prediction(
            source,
            {
                "reference": answer,
                "prediction": answer,
                "stop_reason": "eos",
                "reference_complete": True,
            },
        )
        if result[metric] != 1.0:
            raise RuntimeError(f"Decoded scorer check failed for {source}: {result}")
        json.dumps(result, allow_nan=False)


def summarize(rows, sources, examples):
    """Require exact row coverage and average diagnostics over every selected example."""
    result = {}
    for source in sources:
        selected = [row for row in rows if row["source"] == source]
        if sorted(row["panel_index"] for row in selected) != list(range(examples)):
            raise ValueError(f"Missing or duplicate selected rows for {source}")
        metrics = [row["metrics"] for row in selected]
        names = {
            "scalar_count": ("strict_count_accuracy",),
            "ocr_document": ("normalized_exact_match", "normalized_character_error_rate"),
            "pixmo_caption": ("lexical_token_f1",),
        }.get(source, ("point_f1_at_005",))
        result[source] = {
            "examples": len(selected),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in selected)),
            "unsupported_references": sum(not row["reference_supported"] for row in metrics),
            "invalid_or_unfinished_outputs": sum(not row["valid_output"] for row in metrics),
            **{name: sum(row[name] for row in metrics) / len(metrics) for name in names},
        }
    return result
