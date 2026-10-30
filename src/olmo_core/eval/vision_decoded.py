"""Evaluate vision checkpoints on prepared, source-balanced decoded diagnostic panels."""

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, cast

import torch.distributed as dist

from olmo_core.data.multimodal.data_loader import MultimodalDataLoader
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.eval.multimodal_decoding import (
    _collective_error,
    check_scorers,
    decode_batch,
    source_indices,
    summarize,
)
from olmo_core.eval.multimodal_panel import (
    SCORE_SOURCES,
    load_frozen_evaluator,
    validate_frozen_panel_indices,
    validate_panel_rows,
)
from olmo_core.nn.vision import MultimodalLMConfig
from olmo_core.train import (
    Trainer,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
)
from olmo_core.utils import seed_all

TOKEN_LIMITS = {
    "scalar_count": 16,
    "pixmo_points_basic": 256,
    "ocr_document": 256,
    "pixmo_caption": 768,
}


def checkpoint_configs(checkpoint, evaluator, panel_checkpoint, world_size):
    """Load current or historical metadata with identical evaluation-only execution settings."""
    for path in (checkpoint, panel_checkpoint):
        if not (path / ".metadata.json").is_file():
            raise ValueError(f"Expected a completed checkpoint: {path}")
    saved = json.loads((checkpoint / "config.json").read_text())
    panel = json.loads((panel_checkpoint / "config.json").read_text())
    dataset = saved.get("dataset", {})
    artifacts = saved.get("artifacts", {})
    identity = (
        dataset.get("tokenizer", {}).get("identifier", artifacts.get("tokenizer_id")),
        dataset.get("tokenizer_revision", artifacts.get("tokenizer_revision")),
    )
    expected = (
        evaluator.eval_dataset.tokenizer.identifier,
        evaluator.eval_dataset.tokenizer_revision,
    )
    if identity != expected:
        raise ValueError("Checkpoint and evaluation tokenizer identifier/revision differ")
    ancestry = saved.get("pretraining_checkpoint", artifacts.get("base_checkpoint"))
    if ancestry != panel.get("pretraining_checkpoint"):
        raise ValueError("Checkpoint and evaluation panel pretraining ancestry differ")
    model = MultimodalLMConfig.from_dict(saved["model"])
    module = MultimodalOLMoDDPTrainModuleConfig.from_dict(saved["train_module"])
    if model.lm.vocab_size != evaluator.eval_dataset.model_vocab_size:
        raise ValueError("Checkpoint and evaluation vocabulary sizes differ")
    if module.max_sequence_length < evaluator.sequence_length:
        raise ValueError("Checkpoint context is shorter than the evaluation inputs")
    if any(getattr(module, name, None) for name in ("cp_config", "tp_config", "pp_config")):
        raise ValueError("Decoded diagnostics currently support EP/DP only")
    degree = module.ep_config.degree if module.ep_config is not None else 1
    if world_size <= 0 or world_size % degree:
        raise ValueError("Evaluation world size must be divisible by the expert-parallel degree")
    if evaluator.examples_per_source % (evaluator.rank_batch_size * world_size):
        raise ValueError("Each panel must divide into complete distributed evaluation batches")
    for block in [model.lm.block, *(model.lm.block_overrides or {}).values()]:
        if getattr(block, "routed_experts_router", None) is not None:
            ep = getattr(block, "ep", None)
            if ep is None:
                raise ValueError("Expected expert-parallel routing configuration")
            ep.capacity_factor = float(degree)
    module.rank_microbatch_size = evaluator.rank_batch_size * evaluator.sequence_length
    module.max_sequence_length = evaluator.sequence_length
    module.response_logits_only = True
    return (
        model,
        module,
        {
            "checkpoint": str(checkpoint),
            "checkpoint_phase": saved.get("recipe", {}).get("phase", saved.get("phase")),
            "evaluation_model": model.as_config_dict(),
            "evaluation_train_module": module.as_config_dict(),
            "init_seed": saved.get("init_seed", 12536),
            "model_load": "model_only_preserving_saved_trainability_and_architecture",
            "execution_overrides": {
                "rank_microbatch_size": module.rank_microbatch_size,
                "sequence_length": module.max_sequence_length,
                "destination_capacity_factor": degree,
                "decoding": "greedy_full_vocabulary_eager_grad_enabled_no_kv_cache",
            },
        },
    )


def output_diagnostics(row, token_limit):
    """Report length and repeated word four-grams without repairing predictions."""
    words = re.findall(r"\w+", (row["prediction"] or "").lower())
    grams = [tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    return {
        "word_fourgram_repeat_fraction": 1 - len(set(grams)) / len(grams) if grams else 0.0,
        "reference_tokens": len(row["reference_token_ids"]),
        "reference_exceeds_generation_limit": len(row["reference_token_ids"]) > token_limit,
        "generation_limit": token_limit,
    }


def source_summary(rows, source, examples):
    """Keep primary all-row metrics and separate localization, absence, and termination."""
    score_source = SCORE_SOURCES[source]
    canonical = [{**row, "source": score_source} for row in rows]
    summary = summarize(canonical, [score_source], examples)[score_source]
    summary.update(
        completed_outputs=sum(row["stop_reason"] == "eos" for row in rows),
        reference_exceeds_generation_limit=sum(
            row["output_diagnostics"]["reference_exceeds_generation_limit"] for row in rows
        ),
        mean_word_fourgram_repeat_fraction=sum(
            row["output_diagnostics"]["word_fourgram_repeat_fraction"] for row in rows
        )
        / len(rows),
    )
    if score_source == "pixmo_points_basic":
        supported = [row["metrics"] for row in rows if row["metrics"]["reference_supported"]]
        positive = [row for row in supported if row["reference_points"] > 0]
        empty = [row for row in supported if row["reference_points"] == 0]
        summary.update(
            positive_examples=len(positive),
            positive_point_f1_at_005=(
                sum(row["point_f1_at_005"] for row in positive) / len(positive)
                if positive
                else None
            ),
            reference_points=sum(row["reference_points"] for row in positive),
            matched_points=sum(row["matched_points"] for row in positive),
            empty_reference_examples=len(empty),
            correct_empty_predictions=sum(row["predicted_points"] == 0 for row in empty),
            parsed_outputs=sum(row["metrics"]["point_parse_valid"] for row in rows),
        )
    return summary


def write_result(path, payload):
    """Atomically save one completed source so later-source failures do not lose its rows."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def run(
    checkpoint,
    panel_checkpoint,
    panel_file,
    output_dir,
    *,
    dry_run=False,
    check_complete=False,
    world_size=None,
):
    """Evaluate saved model weights, resuming only matching completed source outputs."""
    metadata_only = dry_run or check_complete
    if world_size is None:
        world_size = 8 if metadata_only else get_world_size()
    if world_size <= 0:
        raise ValueError("Evaluation world size must be positive")
    if not metadata_only and world_size != get_world_size():
        raise ValueError("Requested evaluation world size differs from the active process group")
    evaluator_config, panel_manifest = load_frozen_evaluator(panel_file)
    model_config, module_config, model_manifest = checkpoint_configs(
        checkpoint, evaluator_config, panel_checkpoint, world_size
    )
    manifest: dict[str, Any] = {
        "receipt_version": 1,
        "benchmark_definition": {
            "version": 2,
            "scope": "source_balanced_first_annotation_diagnostics_not_official_benchmarks",
            "panel": panel_manifest,
            "selected_evaluator": evaluator_config.as_config_dict(),
            "score_sources": SCORE_SOURCES,
            "token_limits": TOKEN_LIMITS,
        },
        "execution": {
            "implementation": "olmo_core.eval.vision_decoded",
            **model_manifest,
            "world_size": world_size,
        },
    }
    # Compare JSON representations because config tuples become lists on disk.
    manifest = json.loads(json.dumps(manifest))
    if dry_run:
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return
    error = None
    try:
        validate_frozen_panel_indices(evaluator_config, panel_manifest)
    except Exception as exc:  # noqa: BLE001 - synchronize before model collectives
        if check_complete:
            raise
        error = f"Panel selections: {type(exc).__name__}: {exc}"
    if not check_complete:
        _collective_error(error)
    if any(
        output_dir.is_relative_to(path) or path.is_relative_to(output_dir)
        for path in (checkpoint, panel_checkpoint)
    ):
        raise ValueError("Output must be separate from both checkpoints")
    result_path = output_dir / "results.json"
    if check_complete or result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("manifest") != manifest or result.get("completed") is not True:
            raise ValueError("Existing result does not match this checkpoint and definition")
        for source in SCORE_SOURCES:
            rows = [row for row in result["rows"] if row["source"] == source]
            validate_panel_rows(rows, source, panel_manifest)
            source_summary(rows, source, evaluator_config.examples_per_source)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(manifest["execution"]["init_seed"])
    module = module_config.build(model_config.build(init_device="meta"), eval_only=True)
    module.load_state_dict_direct(
        checkpoint / "model_and_optim",
        process_group=dist.group.WORLD,
        load_optim_state=False,
    )
    context = SimpleNamespace(
        device=module.device,
        dp_process_group=module.dp_process_group,
        work_dir=output_dir / "work",
    )
    total_batches = evaluator_config.examples_per_source // (
        evaluator_config.rank_batch_size * get_world_size()
    )
    error = None
    try:
        evaluators = evaluator_config.build(cast(Trainer, context)).evaluators
        tokenizer, token_ids = evaluator_config.eval_dataset.build_tokenizer()
        sources = list(evaluator_config.eval_dataset.sources)
        if sources != list(SCORE_SOURCES) or [e.name for e in evaluators] != [
            f"{source}-validation" for source in sources
        ]:
            raise ValueError("Unexpected decoded panel source order")
        if any(
            cast(MultimodalDataLoader, evaluator.batches).total_batches != total_batches
            for evaluator in evaluators
        ):
            raise ValueError("Unexpected decoded panel batch count")
        for source, evaluator in zip(sources, evaluators):
            for index in range(total_batches):
                validate_panel_rows(
                    source_indices(evaluator, index, evaluator_config.rank_batch_size),
                    source,
                    panel_manifest,
                )
    except Exception as exc:  # noqa: BLE001 - synchronize before model collectives
        error = f"Panel preparation: {type(exc).__name__}: {exc}"
    _collective_error(error)
    all_rows: list[dict[str, Any]] = []
    for source, evaluator in zip(sources, evaluators):
        path = output_dir / f"{source}-rank{get_rank()}.json"
        cached = None
        error = None
        try:
            if path.exists():
                cached = json.loads(path.read_text())
                if cached.get("manifest") != manifest or cached.get("source") != source:
                    raise ValueError("Saved source result belongs to another evaluation")
                batch_size = evaluator_config.rank_batch_size
                expected_indices = [
                    (batch * get_world_size() + get_rank()) * batch_size + index
                    for batch in range(
                        evaluator_config.examples_per_source // (batch_size * get_world_size())
                    )
                    for index in range(batch_size)
                ]
                if sorted(row["panel_index"] for row in cached["rows"]) != expected_indices:
                    raise ValueError("Incomplete cached rank panel")
                validate_panel_rows(cached["rows"], source, panel_manifest)
        except Exception as exc:  # noqa: BLE001 - synchronize before model collectives
            error = f"Saved source: {type(exc).__name__}: {exc}"
        _collective_error(error)
        cached_ranks = [cached is not None] * get_world_size()
        if is_distributed():
            dist.all_gather_object(cached_ranks, cached is not None)
        if all(cached_ranks):
            assert cached is not None
            rows = cached["rows"]
        else:
            rows = []
            loader = cast(MultimodalDataLoader, evaluator.batches)
            iterator = iter(evaluator)
            try:
                for index in range(total_batches):
                    error = None
                    try:
                        batch = next(iterator)
                        identities = source_indices(
                            evaluator, index, evaluator_config.rank_batch_size
                        )
                    except Exception as exc:  # noqa: BLE001 - synchronize before decoding
                        error = f"Batch preparation: {type(exc).__name__}: {exc}"
                    _collective_error(error)
                    decoded = decode_batch(
                        module,
                        batch,
                        tokenizer,
                        token_ids,
                        identities,
                        source,
                        score_source=SCORE_SOURCES[source],
                        max_new_tokens=TOKEN_LIMITS[SCORE_SOURCES[source]],
                    )
                    error = None
                    try:
                        for row in decoded:
                            row["output_diagnostics"] = output_diagnostics(
                                row, TOKEN_LIMITS[SCORE_SOURCES[source]]
                            )
                    except Exception as exc:  # noqa: BLE001 - synchronize before next batch
                        error = f"Output diagnostics: {type(exc).__name__}: {exc}"
                    _collective_error(error)
                    rows.extend(decoded)
            finally:
                if callable(close := getattr(iterator, "close", None)):
                    close()
                loader.reset()
            error = None
            try:
                write_result(path, {"manifest": manifest, "source": source, "rows": rows})
            except Exception as exc:  # noqa: BLE001 - synchronize before next source
                error = f"Source output: {type(exc).__name__}: {exc}"
            _collective_error(error)
        packets: list[Any] = [rows]
        if is_distributed():
            packets = [None] * get_world_size()
            dist.all_gather_object(packets, rows)
        all_rows.extend(row for packet in packets for row in packet)
    summary = {
        source: source_summary(
            [row for row in all_rows if row["source"] == source],
            source,
            evaluator_config.examples_per_source,
        )
        for source in sources
    }
    error = None
    if get_rank() == 0:
        try:
            write_result(
                result_path,
                {
                    "manifest": manifest,
                    "completed": True,
                    "summary": summary,
                    "rows": all_rows,
                },
            )
        except Exception as exc:  # noqa: BLE001 - synchronize completion failures
            error = f"Summary output: {type(exc).__name__}: {exc}"
    _collective_error(error)


def main(argv: Sequence[str] | None = None) -> None:
    """Run a common diagnostic panel under torchrun, or check metadata/completed outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel-checkpoint", type=Path, required=True)
    parser.add_argument("--panel-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--world-size",
        type=int,
        help="GPU ranks: defaults to 8 for metadata checks, or the active torchrun world size.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    args = parser.parse_args(argv)
    check_scorers()
    if not args.dry_run and not args.check_complete:
        prepare_training_environment(backend="cpu:gloo,cuda:nccl")
    try:
        run(
            args.checkpoint.resolve(strict=True),
            args.panel_checkpoint.resolve(strict=True),
            args.panel_file.resolve(strict=True),
            args.output_dir.resolve(),
            dry_run=args.dry_run,
            check_complete=args.check_complete,
            world_size=args.world_size,
        )
    finally:
        if not args.dry_run and not args.check_complete:
            teardown_training_environment()


if __name__ == "__main__":
    main()
