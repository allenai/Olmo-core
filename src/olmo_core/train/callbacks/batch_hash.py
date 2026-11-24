"""Record each step's per-rank batch hash, for data-order gates (replay, resume)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Tuple

import numpy as np
import torch

from olmo_core.distributed.utils import get_rank

from .callback import Callback

__all__ = ["BatchHashCallback", "hash_batch"]

HASH_KEYS: Tuple[str, ...] = (
    "input_ids",
    "labels",
    "loss_masks",
    "example_ids",
    "document_ids",
    "position_ids",
    "image_crop_counts",
    "pooled_patches_idx",
)


def hash_batch(batch: Dict[str, Any], keys: Tuple[str, ...] = HASH_KEYS) -> Dict[str, str]:
    """sha256 of every present batch key, and of all of them together (``all``)."""
    out: Dict[str, str] = {}
    total = hashlib.sha256()
    for key in keys:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        data = np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()
        out[key] = hashlib.sha256(data).hexdigest()
        total.update(key.encode())
        total.update(data)
    out["all"] = total.hexdigest()
    return out


@dataclass
class BatchHashCallback(Callback):
    """Append ``{step, rank, hashes, index}`` for every step to ``out_dir/rank{R}.jsonl``.

    With ``step1_reference``, step 1 also (before its update) saves each rank's batch to
    ``out_dir/step1/rankNN.npz`` (``input_ids``, ``loss_masks``, ``document_ids``) for an HF
    reference forward, and scores it with the train module in eval mode (``eval_batch``: no
    dropout, an EMo router's eval pool), summed over ranks, to ``out_dir/step1_eval.json``.
    """

    priority: ClassVar[int] = 100
    out_dir: str = ""
    max_step: int = 100
    step1_reference: bool = False

    def pre_step(self, batch: Dict[str, Any]):
        step = self.trainer.global_step  # already advanced to this batch's step
        if not self.out_dir or step > self.max_step:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        record = {"step": step, "rank": get_rank(), "hashes": hash_batch(batch)}
        index = batch.get("index")
        if isinstance(index, torch.Tensor):
            record["index"] = index.tolist()
        with open(os.path.join(self.out_dir, f"rank{get_rank()}.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.step1_reference and step == 1:
            self._step1(batch)

    def _step1(self, batch: Dict[str, Any]):
        import torch.distributed as dist

        from olmo_core.distributed.utils import get_world_size, is_distributed

        rank = get_rank()
        os.makedirs(os.path.join(self.out_dir, "step1"), exist_ok=True)
        arrays = {
            key: batch[key].detach().cpu().numpy()
            for key in ("input_ids", "loss_masks", "document_ids", "labels")
            if isinstance(batch.get(key), torch.Tensor)
        }
        np.savez(os.path.join(self.out_dir, "step1", f"rank{rank:02d}.npz"), **arrays)
        train_module = self.trainer.train_module
        with torch.no_grad():
            output = train_module.eval_batch(dict(batch))  # type: ignore[attr-defined]
        loss_masks = batch["loss_masks"].float()
        sums = torch.tensor(
            [float(output.ce_loss), float((loss_masks * (loss_masks > 0)).sum())],
            dtype=torch.float64,
            device=train_module.device,  # type: ignore[attr-defined]
        )
        if is_distributed():
            dist.all_reduce(sums)
        # eval_batch left the model in eval mode; the step's train_batch needs it training.
        set_mode = getattr(train_module, "_set_model_mode", None)
        if callable(set_mode):
            set_mode("train")
        else:
            for part in train_module.model_parts:  # type: ignore[attr-defined]
                part.train()
        if rank == 0:
            result = {
                "eval_mode_ce": float(sums[0] / sums[1]),
                "loss_weight": float(sums[1]),
                "ranks": get_world_size(),
            }
            with open(os.path.join(self.out_dir, "step1_eval.json"), "w") as f:
                json.dump(result, f, indent=1)
