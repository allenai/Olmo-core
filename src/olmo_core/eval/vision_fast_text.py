"""Evaluate the complete 26-task OLMES-fast panel with resumable native checkpoints.

Run with eight torchrun ranks. The benchmark uses 2K completion contexts, 8192 tokens
per EP-DP rank, and the pinned Dolma2 tokenizer. Per-task results are written atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist

from olmo_core.data.utils import get_labels
from olmo_core.distributed.utils import get_rank
from olmo_core.eval.multimodal_checkpoint import (
    build_model_and_module_config,
    checkpoint_state_dir,
    native_checkpoint_load_coverage_distributed,
)
from olmo_core.eval.task_groups import FAST_TASKS
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks.evaluator_callback import DownstreamEvaluator
from olmo_core.train.train_module import EvalBatchSpec
from olmo_core.utils import gc_cuda, move_to_device

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-group", choices=["fast"], default="fast")
    parser.add_argument("--ep-degree", type=int, choices=[8], default=8)
    parser.add_argument("--max-sequence-length", type=int, choices=[2048], default=2048)
    parser.add_argument("--rank-batch-size", type=int, choices=[8192], default=8192)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pad-token-id", type=int, choices=[100277], default=100277)
    parser.add_argument("--eos-token-id", type=int, choices=[100257], default=100257)
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--check-complete", action="store_true")
    return parser.parse_args(argv)


def _identity(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    checkpoint = args.checkpoint.resolve()
    tokenizer = args.tokenizer.resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    token_config = config["dataset"]["tokenizer"]
    if (
        token_config["identifier"] != "allenai/dolma2-tokenizer"
        or token_config["pad_token_id"] != args.pad_token_id
        or token_config["eos_token_id"] != args.eos_token_id
    ):
        raise ValueError("Checkpoint tokenizer differs from the pinned completion protocol")
    if _sha256(tokenizer) != "969c214487b744f1457d8a4f2055fd4ad348edff5322d4f21f2906ceb158a636":
        raise ValueError("Tokenizer file is not the pinned Dolma2 tokenizer")
    marker = json.loads((checkpoint / ".metadata.json").read_text())
    if marker.get("ephemeral") is not False:
        raise ValueError("Only permanent checkpoints are admitted")
    tasks = list(FAST_TASKS)
    if len(tasks) != 26 or len(set(tasks)) != 26:
        raise ValueError("The complete fast protocol must contain 26 distinct tasks")
    identity = {
        "format": "olmo_core_vision_fast_text_v2",
        "checkpoint": str(checkpoint),
        "config_sha256": _sha256(checkpoint / "config.json"),
        "checkpoint_marker_sha256": _sha256(checkpoint / ".metadata.json"),
        "dcp_metadata_sha256": _sha256(checkpoint_state_dir(checkpoint) / ".metadata"),
        "checkpoint_tensor_hashing": False,
        "tokenizer": {
            "file": str(tokenizer),
            "sha256": _sha256(tokenizer),
            "pad_token_id": args.pad_token_id,
            "eos_token_id": args.eos_token_id,
        },
        "protocol": {
            "harness": "ai2-olmo-eval",
            "harness_version": "0.9.0",
            "task_group": "fast",
            "tasks": tasks,
            "partial": False,
            "max_batches_per_task": None,
            "max_sequence_length": args.max_sequence_length,
            "rank_batch_size_tokens": args.rank_batch_size,
            "world_size": 8,
            "ep_degree": 8,
            "ep_dp_degree": 1,
            "attention_backend": "flex",
            "expert_parallel_path": "rowwise_nvshmem",
            "interface": "native_completion_no_images_no_chat_template",
            "router_lb_policy": "preserve_checkpoint_config",
            "rms_repair": False,
        },
    }
    return identity, tasks


def _validate_result(task: str, result: Any) -> None:
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("metrics"), dict)
        or not result["metrics"]
    ):
        raise ValueError(f"Missing task metrics: {task}")
    for value in result["metrics"].values():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Non-finite or malformed task metric: {task}")
    batches = result.get("batches_per_ep_dp_rank")
    if (
        type(batches) is not int
        or batches <= 0
        or batches != result.get("total_batches_per_ep_dp_rank")
    ):
        raise ValueError(f"Incomplete task: {task}")
    instances = result.get("instances_per_ep_dp_rank")
    if type(instances) is not int or instances <= 0:
        raise ValueError(f"Missing instances: {task}")


def _git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _read_task(path: Path, task: str, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    if cached.get("identity") != identity or cached.get("task") != task:
        raise ValueError(f"Task cache belongs to a different checkpoint or protocol: {path}")
    _validate_result(task, cached.get("result"))
    return cached["result"]


def _validate_complete(path: Path, identity: dict[str, Any], tasks: list[str]) -> None:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 2 or value.get("identity") != identity:
        raise ValueError("Completed results use a different checkpoint, protocol, or result format")
    if set(value.get("results", {})) != set(tasks):
        raise ValueError("Completed results lack the complete 26-task panel")
    coverage = value.get("native_checkpoint_load", {})
    if coverage.get("complete") is not True or coverage.get("load_completed") is not True:
        raise ValueError("Completed results lack complete native checkpoint loading")
    for task in tasks:
        _validate_result(task, value["results"][task])


def _rank_zero_call(function: Callable[..., Any], *args: Any) -> Any:
    packet: list[Any] = [None]
    if get_rank() == 0:
        try:
            packet[0] = {"ok": True, "result": function(*args)}
        except Exception as error:
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    if not packet[0]["ok"]:
        raise RuntimeError(packet[0]["error"])
    return packet[0]["result"]


def main(argv: list[str] | None = None) -> None:
    """Run or inspect a complete OLMES-fast evaluation with per-task resumption."""
    args = _parse_args(argv)
    if args.checkpoint_load_threads <= 0:
        raise ValueError("--checkpoint-load-threads must be positive")
    identity, tasks = _identity(args)
    output = args.output.resolve()
    cache_root = output.with_suffix(output.suffix + ".tasks")
    if args.check_complete:
        _validate_complete(output, identity, tasks)
        print(json.dumps({"complete": True, "output": str(output), "tasks": len(tasks)}))
        return
    if args.dry_run:
        cached = [
            task
            for task in tasks
            if _read_task(cache_root / f"{task}.json", task, identity) is not None
        ]
        print(
            json.dumps(
                {"identity": identity, "output": str(output), "cached_tasks": cached}, indent=2
            )
        )
        return
    if output.exists():
        _validate_complete(output, identity, tasks)
        print(f"Already complete: {output}")
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise ValueError("Run with exactly eight torchrun ranks")
    if version("ai2-olmo-eval") != "0.9.0":
        raise ValueError("The benchmark requires ai2-olmo-eval==0.9.0")

    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()
    try:
        from olmo_eval import HFTokenizer

        checkpoint = args.checkpoint.resolve()
        config = json.loads((checkpoint / "config.json").read_text())
        model, module_config, checkpoint_kind = build_model_and_module_config(
            config,
            ep_degree=8,
            max_sequence_length=args.max_sequence_length,
            rank_batch_size=args.rank_batch_size,
        )
        train_module = module_config.build(model, eval_only=True)
        state_dir = checkpoint_state_dir(checkpoint)
        coverage = native_checkpoint_load_coverage_distributed(train_module, state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )
        coverage["load_completed"] = True
        # The coverage digest describes the pre-load mapping, not the execution status.
        coverage.pop("sha256", None)
        tokenizer = HFTokenizer(
            str(args.tokenizer), pad_token_id=args.pad_token_id, eos_token_id=args.eos_token_id
        )
        results = {}
        for task in tasks:
            cache_path = cache_root / f"{task}.json"
            result = _rank_zero_call(_read_task, cache_path, task, identity)
            if result is None:
                result = evaluate_tasks(train_module, [task], tokenizer, max_batches=None)[task]
                _validate_result(task, result)
                _rank_zero_call(
                    _atomic_json,
                    cache_path,
                    {"identity": identity, "task": task, "result": result},
                )
            elif get_rank() == 0:
                log.info("Reusing complete task: %s", task)
            results[task] = result
            dist.barrier()
        final_identity, _ = _identity(args)
        if final_identity != identity:
            raise ValueError("Checkpoint or tokenizer changed during evaluation")
        payload = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_state_dir": str(state_dir),
            "checkpoint_kind": checkpoint_kind,
            "config": str(checkpoint / "config.json"),
            "execution": {"git": _git_revision(), "harness_version": version("ai2-olmo-eval")},
            "protocol": identity["protocol"],
            "identity": identity,
            "native_checkpoint_load": coverage,
            "results": results,
        }
        _rank_zero_call(_atomic_json, output, payload)
        _rank_zero_call(_validate_complete, output, identity, tasks)
        if get_rank() == 0:
            log.info("Wrote complete OLMES-fast results: %s", output)
    finally:
        teardown_training_environment()


def _float_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu().item()) for name, value in metrics.items()}


def evaluate_tasks(
    train_module,
    tasks: Iterable[str],
    tokenizer,
    *,
    max_batches: int | None,
) -> dict[str, Any]:
    """Evaluate native completion tasks with the benchmark harness batch and score definitions."""
    results: dict[str, Any] = {}
    device = train_module.device
    for task in tasks:
        evaluator = DownstreamEvaluator(
            name="downstream",
            task=task,
            batch_spec=cast(EvalBatchSpec, train_module.eval_batch_spec),
            tokenizer=tokenizer,
            device=device,
            dp_process_group=train_module.dp_process_group,
        )
        evaluator.reset_metrics()
        started = time.monotonic()
        batches = 0
        instances = 0
        for batch in evaluator:
            batches += 1
            instances += int(batch["input_ids"].shape[0])
            batch = move_to_device(batch, device)
            labels = get_labels(batch)
            with torch.no_grad():
                output = train_module.eval_batch(dict(batch), labels=labels)
            if not isinstance(output, LMOutputWithLoss):
                raise TypeError(f"Expected LMOutputWithLoss, got {type(output).__name__}")
            logits, _, ce_loss, _ = output
            evaluator.update_metrics(batch, ce_loss, logits)
            if get_rank() == 0 and (batches == 1 or batches % 20 == 0):
                log.info(
                    "[%s] batch %d/%s",
                    task,
                    batches,
                    evaluator.total_batches if evaluator.total_batches is not None else "?",
                )
            if max_batches is not None and batches >= max_batches:
                break

        metrics = _float_metrics(evaluator.compute_metrics())
        results[task] = {
            "metrics": metrics,
            "batches_per_ep_dp_rank": batches,
            "instances_per_ep_dp_rank": instances,
            "total_batches_per_ep_dp_rank": evaluator.total_batches,
            "elapsed_seconds": time.monotonic() - started,
        }
        if get_rank() == 0:
            log.info("Finished %s: %s", task, metrics)
        del evaluator
        gc_cuda()
    return results


if __name__ == "__main__":
    main()
