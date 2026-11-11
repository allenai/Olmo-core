"""Text-only SFT batches in open-instruct's order, for the multimodal trainer.

open-instruct's GPU text SFT (``olmo_core_finetune.py``) builds a
:class:`~olmo_core.data.numpy_dataset.NumpyPackedFSLDataset` over its numpy cache (plus any
``--extra_numpy_dirs``) and iterates it with OLMo-core's own
:class:`~olmo_core.data.data_loader.NumpyFSLDataLoader`: one global permutation from
``get_rng(seed + epoch)``, sliced into global batches, each rank taking every
``dp_world_size``-th instance of a batch.

:class:`SftTextDataLoaderConfig` builds that same loader over the same packed dataset, with a
collator that turns each instance into the multimodal trainer's batch format (one packed
example per sequence, ``document_ids`` from the row metadata, no images). With the same cache,
seed and global batch, step ``k`` holds exactly the instances open-instruct's step ``k`` holds,
so the multimodal trainer can be compared step for step with the text trainer, and a text-only
control can be trained in the multimodal trainer in the text run's own order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from olmo_core.data.data_loader import (
    DataLoaderConfig,
    NumpyDataLoaderBase,
    NumpyFSLDataLoader,
)
from olmo_core.distributed.utils import get_fs_local_rank, get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError

from .collator import MultimodalCollator
from .packing import pack_examples
from .pretraining_replay import MAX_DOCUMENT_SEGMENTS
from .sft_numpy_cache import SftNumpyCacheConfig, check_row_aligned_cache, instance_to_example

__all__ = ["SftTextCollator", "SftTextDataLoaderConfig", "batch_sha256"]


class SftTextCollator:
    """Collate packed SFT instances into multimodal text batches.

    Each instance becomes one packed example (as the mixture loader packs a full-length text
    example alone), so the batch carries ``example_ids`` and ``document_ids`` exactly as a
    text sequence from :class:`~olmo_core.data.multimodal.mixture_data_loader.MixtureDataLoader`
    does. ``index`` keeps the instance indices for order checks.
    """

    def __init__(
        self,
        *,
        pad_token_id: int,
        sequence_length: int,
        batch_metadata: bool = True,
        compact_images: bool = True,
        image_dtype: Optional[torch.dtype] = torch.bfloat16,
        document_masking: bool = True,
        max_document_segments: Optional[int] = MAX_DOCUMENT_SEGMENTS,
    ):
        self.document_masking = document_masking
        self.max_document_segments = max_document_segments
        self.collator = MultimodalCollator(
            pad_token_id=pad_token_id,
            pad_sequence_length=sequence_length,
            batch_metadata=batch_metadata,
            image_dtype=image_dtype,
            compact_images=compact_images,
        )

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        packs = []
        for instance in instances:
            example = instance_to_example(
                instance,
                index=int(instance["index"]),
                document_masking=self.document_masking,
                max_document_segments=self.max_document_segments,
            )
            packs.append(pack_examples([example]))
        batch = self.collator(packs)
        batch["index"] = torch.tensor([int(i["index"]) for i in instances], dtype=torch.int64)
        return batch


def batch_sha256(batch: Dict[str, Any], keys=("input_ids", "labels", "loss_masks", "document_ids")) -> str:
    """Hash the token-level tensors of a rank batch (instance order included)."""
    digest = hashlib.sha256()
    for key in keys:
        value = batch.get(key)
        if value is None:
            continue
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.cpu().numpy()).tobytes())
    return digest.hexdigest()


@DataLoaderConfig.register("sft_text")
@dataclass
class SftTextDataLoaderConfig(DataLoaderConfig["NumpyFSLDataLoader"]):
    """open-instruct's text SFT loader over an SFT numpy cache, in multimodal batch format."""

    cache: SftNumpyCacheConfig
    """The cache (and replay dirs) open-instruct trains from."""
    global_batch_size: int
    """Global batch size in tokens (instances x sequence length)."""
    seed: int
    """open-instruct's ``--data_loader_seed``; epoch ``e`` shuffles with ``get_rng(seed + e)``."""
    work_dir: str
    num_workers: int = 4
    """As open-instruct; the order does not depend on it."""
    pad_token_id: int = 100277
    batch_metadata: bool = True
    compact_images: bool = True

    def build(self, dataset: Any = None, *, dp_process_group: Optional[dist.ProcessGroup] = None):  # type: ignore[override]
        """Build the loader; ``dataset`` is ignored (the cache config builds its own)."""
        del dataset
        check_row_aligned_cache(self.cache.numpy_dir)
        packed = self.cache.dataset_config().build()
        packed.prepare()
        if self.global_batch_size % packed.sequence_length:
            raise OLMoConfigurationError("global_batch_size must be whole sequences")
        collator = SftTextCollator(
            pad_token_id=self.pad_token_id,
            sequence_length=packed.sequence_length,
            batch_metadata=self.batch_metadata,
            compact_images=self.compact_images,
            document_masking=self.cache.document_masking,
            max_document_segments=self.cache.max_document_segments,
        )
        # The construction open-instruct's NumpyDataLoaderConfig.build performs.
        return NumpyDataLoaderBase.wrap_numpy_dataset(
            packed,
            global_batch_size=self.global_batch_size,
            collator=collator,  # type: ignore[arg-type]
            work_dir=self.work_dir,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            seed=self.seed,
            num_workers=self.num_workers,
            target_device_type="cuda" if torch.cuda.is_available() else "cpu",
        )


def open_instruct_step_indices(
    num_instances: int, *, seed: int, epoch: int = 1, instances_per_step: int, steps: int
) -> np.ndarray:
    """The global instance indices of steps ``1..steps``, as open-instruct's loader draws them.

    ``NumpyFSLDataLoader`` with ``chunk_size == 1``: ``get_rng(seed + epoch)`` shuffles
    ``arange(num_instances, dtype=uint32)``; step ``k`` is ``order[(k-1)*n:k*n]``.
    """
    from olmo_core.data.utils import get_rng

    order = np.arange(num_instances, dtype=np.uint32)
    get_rng(seed + epoch).shuffle(order)
    return order[: steps * instances_per_step].reshape(steps, instances_per_step)
