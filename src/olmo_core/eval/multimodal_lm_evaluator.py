from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.distributed as dist

from ..exceptions import OLMoConfigurationError
from .evaluator import Evaluator
from .metrics import MeanMetric


class MultimodalLMEvaluator(Evaluator):
    """Response-token loss and perplexity for multimodal batches.

    This evaluator pairs with a multimodal train module whose ``eval_batch()`` returns
    the summed, per-token-weighted CE loss. Keeping that loss reduced avoids materializing
    full-sequence vocabulary logits during Stage 1 evaluation.

    :param response_prefix_tokens: Optionally score only the first N supervised positions
        of each sequence, retaining the full teacher-forced input.
    :param reference_evaluator: An already evaluated correct-image control with identical
        recipients and loss weights. Adds ``CE gap`` (this CE minus the reference CE), so
        a positive wrong-image gap indicates useful image content.
    """

    def __init__(
        self,
        *,
        name: str,
        batches: Iterable[dict[str, Any]],
        device: torch.device | None = None,
        process_group: dist.ProcessGroup | None = None,
        deterministic: bool = True,
        response_prefix_tokens: int | None = None,
        reference_evaluator: "MultimodalLMEvaluator | None" = None,
    ):
        super().__init__(name=name, batches=batches, device=device, deterministic=deterministic)
        if response_prefix_tokens is not None and response_prefix_tokens <= 0:
            raise OLMoConfigurationError("response_prefix_tokens must be positive or None")
        self.ce_loss = MeanMetric(device=device, process_group=process_group)
        self.response_prefix_tokens = response_prefix_tokens
        self.reference_evaluator = reference_evaluator

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Optionally score only the first N supervised positions of each sequence.

        The full teacher-forced input and original labels are retained. Only loss weights
        are masked, so the prefix diagnostic uses exactly the same attention context.
        """
        for batch in super().__iter__():
            if self.response_prefix_tokens is not None:
                weights = batch["loss_masks"]
                valid = (weights > 0) & (batch["labels"] != -100)
                prefix = valid & (valid.long().cumsum(dim=-1) <= self.response_prefix_tokens)
                batch = dict(batch)
                batch["loss_masks"] = weights * prefix
            yield batch

    def update_metrics(
        self,
        batch: dict[str, Any],
        ce_loss: torch.Tensor | None,
        logits: torch.Tensor | None,
    ) -> None:
        del logits
        if ce_loss is None:
            return
        if ce_loss.numel() != 1:
            raise OLMoConfigurationError(
                "MultimodalLMEvaluator expects a scalar summed CE loss, "
                f"got shape {tuple(ce_loss.shape)}"
            )

        loss_weights = batch["loss_masks"].to(device=self.ce_loss.device).float()
        valid = loss_weights > 0
        if (labels := batch.get("labels")) is not None:
            valid &= labels.to(device=self.ce_loss.device) != -100
        weight = loss_weights.masked_select(valid).sum()
        self.ce_loss.update(ce_loss.detach() / weight.clamp_min(1.0), weight)

    def compute_metrics(self) -> dict[str, torch.Tensor]:
        """Return response CE and, when configured, its gap from the preceding paired control."""
        ce_loss = self.ce_loss.compute()
        metrics = {"CE loss": ce_loss, "PPL": torch.exp(ce_loss)}
        if self.reference_evaluator is not None:
            metrics["CE gap"] = ce_loss - self.reference_evaluator.ce_loss.compute()
        return metrics

    def reset_metrics(self) -> None:
        self.ce_loss.reset()


class MultimodalBlankImageEvaluator(MultimodalLMEvaluator):
    """Evaluate response loss after replacing normalized image patches with zeros.

    The token sequence, response labels, crop geometry, and image-placement indices remain
    unchanged. Because image preprocessing normalizes pixel channels, zeros represent a
    mean-color blank control without introducing crop-count or padding mismatches. Comparing its
    CE against the ordinary evaluator is a content-reliance diagnostic; it is not by itself a
    complete measure of visual understanding.

    :param args: Positional arguments forwarded to :class:`MultimodalLMEvaluator`.
    :param kwargs: Keyword arguments forwarded to :class:`MultimodalLMEvaluator`.

    :raises OLMoConfigurationError: If a batch lacks an image tensor.
    """

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in super().__iter__():
            images = batch.get("images")
            if not isinstance(images, torch.Tensor):
                raise OLMoConfigurationError(
                    "Image-ablation evaluation requires a tensor-valued 'images' batch field"
                )
            transformed = dict(batch)
            transformed["images"] = torch.zeros_like(images)
            yield transformed
