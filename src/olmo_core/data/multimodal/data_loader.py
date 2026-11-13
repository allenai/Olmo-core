"""A :class:`~olmo_core.data.data_loader.DataLoaderBase` adapter for multimodal data.

Wraps a map-style multimodal dataset (e.g.
:class:`~olmo_core.data.multimodal.pixmo_cap.PixMoCapDataset`) plus a
:class:`~olmo_core.data.multimodal.collator.MultimodalCollator` so it can drive the
:class:`~olmo_core.train.Trainer`. Batches are **instance**-based but reported in
*tokens* (``instances × pad_sequence_length``) so they fit the Trainer's token-based
bookkeeping; this requires the collator to pad to a fixed ``pad_sequence_length``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from ..data_loader import DataLoaderBase
from .collator import MultimodalCollator

__all__ = ["MultimodalDataLoader"]


class MultimodalDataLoader(DataLoaderBase):
    """Instance-based data loader over a map-style multimodal dataset.

    :param dataset: A map-style dataset whose ``__getitem__`` returns the per-example
        dict expected by :class:`MultimodalCollator`.
    :param collator: The collator, which **must** have a fixed ``pad_sequence_length``.
    :param global_batch_size: Global batch size *in tokens* (= global instances ×
        ``pad_sequence_length``).
    """

    def __init__(
        self,
        dataset,
        collator: MultimodalCollator,
        *,
        work_dir,
        global_batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
    ):
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        if collator.pad_sequence_length is None:
            raise OLMoConfigurationError(
                "MultimodalDataLoader requires the collator to have a fixed "
                "`pad_sequence_length` so every batch has a constant token count."
            )
        self.dataset = dataset
        self.collator = collator
        self.seed = seed
        self.shuffle = shuffle
        self.seq_len = collator.pad_sequence_length
        self._order: Optional[np.ndarray] = None

    @property
    def _global_instances(self) -> int:
        return self.global_batch_size // self.seq_len

    @property
    def _rank_instances(self) -> int:
        return self.rank_batch_size // self.seq_len

    @property
    def total_batches(self) -> Optional[int]:
        return len(self.dataset) // self._global_instances

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        if epoch is not None:
            self._epoch = epoch
        epoch = self._epoch if self._epoch is not None else 1
        # Hand the epoch to a source that samples per example, as MixtureDataLoader does.
        set_epoch = getattr(self.dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        order = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.RandomState(self.seed + epoch).shuffle(order)
        self._order = order

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        if self._order is None:
            raise RuntimeError("call reshuffle() before iterating")
        order = self._order
        n_batches = self.total_batches or 0
        gi = self._global_instances
        ri = self._rank_instances
        start_batch = self.batches_processed
        for b in range(start_batch, n_batches):
            global_slice = order[b * gi : (b + 1) * gi]
            rank_slice = global_slice[self.dp_rank * ri : (self.dp_rank + 1) * ri]
            examples = [self.dataset[int(i)] for i in rank_slice]
            yield self.collator(examples)

    def get_mock_batch(self) -> Dict[str, Any]:
        ri = max(self._rank_instances, 1)
        n = min(ri, len(self.dataset))
        examples = [self.dataset[i] for i in range(n)]
        while len(examples) < ri:
            examples.append(examples[-1])
        return self.collator(examples)

    def global_num_tokens_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        del batch
        return self.global_batch_size

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batches_processed": self.batches_processed,
            "epoch": self._epoch,
            "seed": self.seed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.batches_processed = state_dict.get("batches_processed", 0)
        self._epoch = state_dict.get("epoch")
        self.seed = state_dict.get("seed", self.seed)
