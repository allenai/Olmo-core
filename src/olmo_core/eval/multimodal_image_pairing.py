"""Bounded same-geometry image-content diagnostics for in-loop evaluation."""

import hashlib
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Tuple

import numpy as np
import torch

from olmo_core.exceptions import OLMoConfigurationError


def _array_descriptor(value: Any, *, field_name: str) -> Dict[str, Any]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        try:
            raw = tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
        except RuntimeError as error:
            raise OLMoConfigurationError(
                f"Could not describe tensor field {field_name!r} for wrong-image evaluation"
            ) from error
        return {
            "kind": "torch",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise OLMoConfigurationError(
            f"Wrong-image evaluation field {field_name!r} has an object dtype"
        )
    dtype = array.dtype
    if dtype.itemsize > 1:
        little_dtype = dtype.newbyteorder("<")
        if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big"):
            array = array.byteswap().view(little_dtype)
        else:
            array = array.astype(little_dtype, copy=False)
    array = np.ascontiguousarray(array)
    return {
        "kind": "numpy",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _get_example(dataset: Any, index: int, epoch: int) -> Mapping[str, Any]:
    get = getattr(dataset, "get", None)
    example = get(index, epoch) if callable(get) else dataset[index]
    if not isinstance(example, Mapping):
        raise OLMoConfigurationError(f"Wrong-image validation row {index} is not a mapping")
    return example


def _geometry(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    example = row["example"]
    images = example["images"]
    pooled = example["pooled_patches_idx"]
    return (
        images["kind"],
        images["dtype"],
        tuple(images["shape"]),
        pooled["kind"],
        pooled["dtype"],
        tuple(pooled["shape"]),
        pooled["sha256"],
    )


def _arrays_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    return False


def build_bounded_image_pairs(
    dataset: Any, *, examples: int, max_candidates: int, seed: int = 0
) -> list[tuple[int, int]]:
    """Pair distinct image tensors within exact geometry groups in a bounded validation prefix.

    Only the first ``min(max_candidates, len(dataset))`` rows are preprocessed, at source
    epoch zero. Image digests reject duplicate pixel tensors; geometry requires matching
    array types, dtypes, image shapes, and exact pooling indices.
    No training population, annotation inventory, or source files are scanned.

    :param examples: Required number of paired recipients.
    :param max_candidates: Maximum validation rows to inspect for donors and recipients.
    :param seed: Seed for reproducible recipient and donor ordering.
    :returns: Recipient/donor dataset indices, with unique recipients and donors.
    """
    if examples <= 0 or max_candidates < examples or seed < 0:
        raise OLMoConfigurationError("Invalid bounded image-pairing size or seed")
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    seen: set[tuple[tuple[Any, ...], str]] = set()
    for index in range(min(max_candidates, len(dataset))):
        example = _get_example(dataset, index, 0)
        descriptors = {}
        for name in ("images", "pooled_patches_idx"):
            value = example.get(name)
            if not isinstance(value, (np.ndarray, torch.Tensor)) or 0 in value.shape:
                raise OLMoConfigurationError(
                    f"Image-pairing row {index} requires a nonempty {name!r} array"
                )
            descriptors[name] = _array_descriptor(value, field_name=name)
        geometry = _geometry({"example": descriptors})
        identity = (geometry, descriptors["images"]["sha256"])
        if identity not in seen:
            seen.add(identity)
            groups[geometry].append(index)

    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        rng.shuffle(group)
        pairs.extend(zip(group, group[1:] + group[:1]))
    if len(pairs) < examples:
        raise OLMoConfigurationError(
            f"Only {len(pairs)} validation rows have distinct exact-geometry image donors "
            f"within {min(max_candidates, len(dataset))} candidates; requested {examples}. "
            "Increase matched_image_candidates or reduce matched_image_examples."
        )
    rng.shuffle(pairs)
    return pairs[:examples]


class MultimodalImagePairDataset:
    """Read fixed correct-image or wrong-image recipients for an in-loop diagnostic.

    Pairing indices are prepared once by :func:`build_bounded_image_pairs`. Every replay
    reads source epoch zero. Wrong-image examples replace only the image tensor, retaining
    the recipient's labels, loss weights, geometry, and attention metadata.
    """

    def __init__(self, dataset: Any, pairs: Sequence[tuple[int, int]], *, wrong_images: bool):
        self.dataset = dataset
        self.pairs = tuple(pairs)
        self.wrong_images = wrong_images

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        """Read a recipient, checking donor geometry and distinct pixels before replacement."""
        if epoch != 0:
            raise OLMoConfigurationError("Image-pair evaluation requires source epoch zero")
        recipient_index, donor_index = self.pairs[index]
        recipient = dict(_get_example(self.dataset, recipient_index, 0))
        if self.wrong_images:
            donor = _get_example(self.dataset, donor_index, 0)
            equal = _arrays_equal
            if (
                recipient["images"].shape != donor["images"].shape
                or recipient["images"].dtype != donor["images"].dtype
                or not equal(recipient["pooled_patches_idx"], donor["pooled_patches_idx"])
                or equal(recipient["images"], donor["images"])
            ):
                raise OLMoConfigurationError("Matched image geometry or distinctness changed")
            images = donor["images"]
            recipient["images"] = (
                images.clone() if isinstance(images, torch.Tensor) else images.copy()
            )
        return recipient
