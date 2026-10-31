#!/usr/bin/env python3
"""
Check that a multimodal OLMoDDP checkpoint gives the same forward loss under two OLMo-core
versions.

The batches are drawn once, by the run's own data loader, and saved; each version then loads the
checkpoint through its own train module and scores the same saved batches with
:meth:`~olmo_core.train.train_module.transformer.multimodal_train_module.MultimodalOLMoDDPTrainModule.eval_batch`,
the training loss path without the backward. Run every step under the OLMo-core version it
names, with this file passed by path::

    # Once, under the code that trained the checkpoint:
    torchrun --nproc-per-node=1 multimodal_load_parity.py make-batches STEP_DIR OUT_DIR
    # Under each version:
    torchrun --nproc-per-node=1 multimodal_load_parity.py score STEP_DIR OUT_DIR --tag old
    torchrun --nproc-per-node=1 multimodal_load_parity.py score STEP_DIR OUT_DIR --tag new
    python multimodal_load_parity.py compare OUT_DIR old new

``make-batches`` keeps the first batch with images and the first without, each one instance of
the run's sequence length, and writes their sha256 to ``batches.json``. ``score`` refuses a batch
whose hash differs. ``compare`` passes when the token-weighted CE of every batch agrees within
``--tolerance`` (default 1e-3); the largest logit difference over a fixed sample of response
positions is reported beside it as a diagnostic, for batches with at most ``LOGIT_MAX_POSITIONS``
supervised positions (a text instance supervises nearly every position, and its response logits
alone would take ~26 GB in fp32).
"""

import dataclasses
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import click
import torch

log = logging.getLogger(__name__)

BATCH_NAMES = ("images", "text")
LOGIT_SAMPLE = 1024
LOGIT_MAX_POSITIONS = 16384


def _load_config(step_dir: Path):
    from olmo_core.internal.vision_midtraining import MixedMidtrainingExperimentConfig

    raw = json.loads((step_dir / "config.json").read_text())
    return MixedMidtrainingExperimentConfig.from_dict(raw)


def _tensors(batch: Dict[str, Any]) -> Iterator[Tuple[str, torch.Tensor]]:
    for key in sorted(batch):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            yield key, value


def batch_sha256(batch: Dict[str, Any]) -> str:
    """Hash every tensor of a batch: key, dtype, shape and bytes, keys in sorted order."""
    digest = hashlib.sha256()
    for key, tensor in _tensors(batch):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _to_cpu(batch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Shared-memory or pinned tensors become ordinary owned tensors.
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


@click.group()
def cli():
    pass


@cli.command("make-batches")
@click.argument("step_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--max-batches", default=256, show_default=True)
def make_batches(step_dir: Path, out_dir: Path, max_batches: int):
    """Draw the first image batch and the first text-only batch from the run's data loader."""
    from olmo_core.internal.vision_alignment import export_hf_datasets_cache

    config = _load_config(step_dir)
    export_hf_datasets_cache(config.recipe.hf_datasets_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # One instance per batch, a private work dir (the run's own stays untouched) and no
    # background processes; the sources, packing and seed are the run's.
    loader_config = dataclasses.replace(
        config.data_loader,
        global_batch_size=config.data_loader.sequence_length,
        work_dir=str(out_dir / "data_loader"),
        prefetch_workers=2,
        prefetch_processes=False,
        batch_prefetch_process=False,
    )
    data_loader = loader_config.build(config.dataset.build())
    data_loader.reshuffle(epoch=1)

    found: Dict[str, Dict[str, Any]] = {}
    for index, batch in enumerate(data_loader):
        if index >= max_batches:
            break
        images = batch.get("images")
        name = "images" if isinstance(images, torch.Tensor) and images.numel() > 0 else "text"
        if name in found:
            continue
        found[name] = {"index": index, "batch": _to_cpu(batch)}
        log.info("Batch %d is the first %s batch", index, name)
        if len(found) == len(BATCH_NAMES):
            break
    missing = [name for name in BATCH_NAMES if name not in found]
    if missing:
        raise click.ClickException(f"No {missing} batch in the first {max_batches} batches")

    manifest = {"step_dir": str(step_dir), "batches": {}}
    for name in BATCH_NAMES:
        batch = found[name]["batch"]
        torch.save(batch, out_dir / f"batch_{name}.pt")
        loss_masks = batch.get("loss_masks")
        manifest["batches"][name] = {
            "loader_index": found[name]["index"],
            "sha256": batch_sha256(batch),
            "keys": sorted(batch),
            "loss_weight": float((loss_masks * (loss_masks > 0)).sum()) if loss_masks is not None else None,
            "crops": int(batch["images"].shape[0]) if name == "images" else 0,
        }
    (out_dir / "batches.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


@cli.command("score")
@click.argument("step_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--tag", required=True, help="Name of this OLMo-core version, e.g. old or new.")
@click.option("--compile/--no-compile", "compile_model", default=False, show_default=True)
def score(step_dir: Path, out_dir: Path, tag: str, compile_model: bool):
    """Load STEP_DIR's model through this version's train module and score the saved batches."""
    import olmo_core
    from olmo_core.utils import seed_all

    manifest = json.loads((out_dir / "batches.json").read_text())
    config = _load_config(step_dir)
    config.train_module.compile_model = compile_model
    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    train_module.load_state_dict_direct(str(step_dir / "model_and_optim"))

    results: Dict[str, Any] = {"tag": tag, "olmo_core": str(Path(olmo_core.__file__).parent)}
    for name in BATCH_NAMES:
        batch = torch.load(out_dir / f"batch_{name}.pt", weights_only=False)
        expected = manifest["batches"][name]["sha256"]
        if (actual := batch_sha256(batch)) != expected:
            raise click.ClickException(f"batch_{name}.pt hash {actual} != manifest {expected}")
        loss_masks = batch["loss_masks"].float()
        weight = float((loss_masks * (loss_masks > 0)).sum())
        positions = int((loss_masks > 0).sum())
        with_logits = positions <= LOGIT_MAX_POSITIONS
        with torch.no_grad():
            output = train_module.eval_batch(batch, return_response_logits=with_logits)
        if with_logits:
            logits = output.logits
            assert logits is not None
            # A fixed, evenly spaced sample of the supervised positions, the same on every version.
            rows = torch.linspace(0, logits.shape[0] - 1, min(LOGIT_SAMPLE, logits.shape[0])).long()
            torch.save(
                logits[rows.to(logits.device)].float().cpu(), out_dir / f"logits_{tag}_{name}.pt"
            )
        results[name] = {
            "sha256": expected,
            "ce": float(output.ce_loss) / weight,
            "z_loss": float(output.z_loss) / weight if output.z_loss is not None else None,
            "loss_weight": weight,
            "response_positions": positions,
            "logits_saved": with_logits,
        }
        del output
        torch.cuda.empty_cache()
        log.info("%s %s: CE %.8f over weight %.1f", tag, name, results[name]["ce"], weight)
    (out_dir / f"score_{tag}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


@cli.command("compare")
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("tag_a")
@click.argument("tag_b")
@click.option("--tolerance", default=1e-3, show_default=True)
def compare(out_dir: Path, tag_a: str, tag_b: str, tolerance: float):
    """Pass when every batch's CE agrees within TOLERANCE; exit 1 otherwise."""
    a = json.loads((out_dir / f"score_{tag_a}.json").read_text())
    b = json.loads((out_dir / f"score_{tag_b}.json").read_text())
    report: Dict[str, Any] = {"tolerance": tolerance, "batches": {}}
    passed = True
    for name in BATCH_NAMES:
        if a[name]["sha256"] != b[name]["sha256"]:
            raise click.ClickException(f"{name}: the two scores read different batches")
        logit_diff = None
        if a[name].get("logits_saved") and b[name].get("logits_saved"):
            logits_a = torch.load(out_dir / f"logits_{tag_a}_{name}.pt")
            logits_b = torch.load(out_dir / f"logits_{tag_b}_{name}.pt")
            logit_diff = float((logits_a - logits_b).abs().max())
        gap = abs(a[name]["ce"] - b[name]["ce"])
        ok = gap <= tolerance
        passed &= ok
        report["batches"][name] = {
            f"ce_{tag_a}": a[name]["ce"],
            f"ce_{tag_b}": b[name]["ce"],
            "abs_gap": gap,
            "pass": ok,
            "max_abs_logit_diff": logit_diff,
        }
    report["pass"] = passed
    (out_dir / f"compare_{tag_a}_{tag_b}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "compare":
        from olmo_core.train import prepare_training_environment

        prepare_training_environment()
    else:
        from olmo_core.utils import prepare_cli_environment

        prepare_cli_environment()
    cli()
