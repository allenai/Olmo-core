from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ..config import StrEnum

__all__ = ["DataCollator"]


class PaddingDirection(StrEnum):
    """
    Specifies the direction to pad instances when needed.
    """

    left = "left"
    right = "right"


@dataclass
class DataCollator:
    """
    The default data collator used by :class:`~olmo_core.data.data_loader.TextDataLoaderBase` subclasses.

    :param pad_token_id: The token ID to use for padding.
    :param pad_direction: The direction to pad instances.
    :param label_ignore_index: The index to use for ignored labels.
    :param vocab_size: If set, validate that all token IDs in the collated batch are
        in ``[0, vocab_size)``. This catches out-of-range IDs early with a clear error
        message, which is especially useful when using ``torch.compile`` where the
        resulting CUDA error would otherwise be opaque.
    """

    pad_token_id: int
    pad_direction: PaddingDirection = PaddingDirection.right
    label_ignore_index: int = -100
    vocab_size: Optional[int] = None

    def __call__(
        self, items: Union[Sequence[Dict[str, Any]], Sequence[torch.Tensor]]
    ) -> Dict[str, Any]:
        """
        Create a batch from a sequence of instances.
        """
        assert items
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))
        all_input_ids = []
        all_attention_mask = []
        all_attention_bias = []
        all_label_mask = []
        all_loss_valid_mask = []
        all_indices = []
        all_metadata = []
        all_instance_mask = []
        all_doc_lens = []
        all_max_doc_lens = []
        max_docs = max(
            (len(x["doc_lens"]) if isinstance(x, dict) and "doc_lens" in x else 0 for x in items)
        )

        for x in items:
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)

            pad_shape = (
                (max_len - len(input_ids), 0)
                if self.pad_direction == PaddingDirection.left
                else (0, max_len - len(input_ids))
            )

            # Pad input IDs.
            all_input_ids.append(
                F.pad(
                    input_ids.to(dtype=torch.long),
                    pad_shape,
                    value=self.pad_token_id,
                )
            )

            # Pad attention mask.
            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is None and isinstance(x, dict) and "loss_valid_mask" in x:
                if max_len != len(input_ids):
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                elif any(len(item["input_ids"]) != max_len for item in items):
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                all_attention_mask.append(
                    F.pad(
                        attention_mask.to(dtype=torch.float),
                        pad_shape,
                        value=0.0,
                    )
                )

            # Pad attention bias.
            attention_bias = x.get("attention_bias") if isinstance(x, dict) else None
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                all_attention_bias.append(
                    F.pad(
                        attention_bias,
                        pad_shape + pad_shape,
                        value=pad_value,
                    )
                )

            # Pad label mask.
            label_mask = x.get("label_mask") if isinstance(x, dict) else None
            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(
                    F.pad(
                        label_mask.to(dtype=torch.bool),
                        pad_shape,
                        value=False,
                    )
                )

            # Indices.
            valid_mask = x.get("loss_valid_mask") if isinstance(x, dict) else None
            if valid_mask is not None:
                all_loss_valid_mask.append(
                    F.pad(torch.as_tensor(valid_mask, dtype=torch.bool), pad_shape, value=False)
                )

            # Indices.
            index = x.get("index") if isinstance(x, dict) else None
            if index is not None:
                all_indices.append(torch.tensor(index))

            # Instance mask.
            instance_mask = x.get("instance_mask") if isinstance(x, dict) else None
            if instance_mask is not None:
                all_instance_mask.append(torch.tensor(instance_mask))

            # Document lengths.
            doc_lens = x.get("doc_lens") if isinstance(x, dict) else None
            if doc_lens is not None:
                doc_pad_shape = (0, max_docs - len(doc_lens))
                all_doc_lens.append(F.pad(doc_lens, doc_pad_shape, value=0))
                all_max_doc_lens.append(int(doc_lens.max()))

            # Metadata.
            metadata = x.get("metadata") if isinstance(x, dict) else None
            if metadata is not None:
                all_metadata.append(metadata)

        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}

        if self.vocab_size is not None:
            input_ids_batch = out["input_ids"]
            invalid = (input_ids_batch < 0) | (input_ids_batch >= self.vocab_size)
            if invalid.any():
                bad_ids = input_ids_batch[invalid].unique().tolist()
                positions = invalid.nonzero(as_tuple=False).tolist()
                raise ValueError(
                    f"Token IDs {bad_ids} outside valid range [0, {self.vocab_size}). "
                    f"Found at (batch_idx, pos): {positions[:10]}"
                )

        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        if all_attention_bias:
            out["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            out["label_mask"] = torch.stack(all_label_mask)
        if all_loss_valid_mask:
            if len(all_loss_valid_mask) != len(items):
                raise ValueError("loss_valid_mask must be supplied for every instance in a batch")
            out["loss_valid_mask"] = torch.stack(all_loss_valid_mask)
        if all_indices:
            out["index"] = torch.stack(all_indices)
        if all_instance_mask:
            out["instance_mask"] = torch.stack(all_instance_mask)
        if all_doc_lens:
            out["doc_lens"] = torch.stack(all_doc_lens)
        if all_max_doc_lens:
            out["max_doc_lens"] = all_max_doc_lens
        if all_metadata:
            out["metadata"] = all_metadata

        return out
