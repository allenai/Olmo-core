import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import olmo_core.ops.moe as ops
from olmo_core.exceptions import OLMoConfigurationError

from ..emo import EmoRouterConfig
from ..router import MoERouterGatingFunction
from .router import MoERouterV2


class EmoRouterV2(MoERouterV2):
    """Document-level expert-pool routing for the v2 routed-expert path.

    This router only controls routed experts. The v2 ``SharedExperts`` branch is
    independent and remains active exactly as configured by the transformer block.
    """

    def __init__(self, *, emo: EmoRouterConfig, **kwargs):
        super().__init__(**kwargs)
        self.emo = emo
        self._profile_document_pool = os.environ.get("OLMO_PROFILE_EMO_DOCUMENT_POOL", "0") == "1"
        self._profile_top16 = os.environ.get("OLMO_PROFILE_EMO_TOP16", "0") == "1"
        self.emo.validate_for_router(num_experts=self.num_experts, top_k=self.top_k)

        unsupported = {
            "uniform_expert_assignment": self.uniform_expert_assignment,
            "random_expert_assignment": self.random_expert_assignment,
            "bias_gamma": self.bias_gamma is not None,
            "score_correction_bias": self.score_correction_bias,
            "grouped routing": self.n_group is not None or self.topk_group is not None,
            "recompute fp32 cast": self.use_recompute_fp32_cast,
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise OLMoConfigurationError(f"EMO routing does not support: {', '.join(enabled)}")

    @property
    def requires_segment_ids(self) -> bool:
        return not self.emo.full_pool and getattr(self, "replay_expert_indices", None) is None

    @property
    def eos_token_id(self) -> int:
        return self.emo.eos_token_id

    def _pool_sizes(self, segment_ids: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return torch.full_like(segment_ids, self.emo.eval_pool_size())

        per_document = torch.randint(
            self.emo.min_document_expert_pool,
            self.emo.max_document_expert_pool + 1,
            segment_ids.shape,
            device=segment_ids.device,
        )
        return per_document.gather(1, segment_ids)

    def forward(
        self,
        x: torch.Tensor,
        scores_only: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        if scores_only:
            raise OLMoConfigurationError(
                "EMO is only supported for the routed-expert router; configure shared-expert "
                "mixing with a separate standard router"
            )
        if self.emo.full_pool or getattr(self, "replay_expert_indices", None) is not None:
            # Ordinary forward owns the gating-specific replay weights, gradients
            # and auxiliary counts. Never sample a document pool on this path.
            return super().forward(x, scores_only=False, loss_div_factor=loss_div_factor)
        if segment_ids is None:
            raise OLMoConfigurationError("EMO routing requires per-token segment_ids")
        if segment_ids.shape != x.shape[:2]:
            raise OLMoConfigurationError(
                f"segment_ids shape {tuple(segment_ids.shape)} must match token shape "
                f"{tuple(x.shape[:2])}"
            )

        x = self.jitter(x)
        logits = self.get_expert_logits(x.float()).float()
        if self.gating_function in (
            MoERouterGatingFunction.softmax,
            MoERouterGatingFunction.topk_softmax,
        ):
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = logits.sigmoid()
            if self.sigmoid_stability_epsilon:
                scores = scores + self.sigmoid_stability_epsilon
        else:
            raise NotImplementedError(self.gating_function)

        if self._profile_document_pool and scores.is_cuda and self.num_experts == 512:
            from olmo_core.ops.emo_document_pool import document_pool_keep_mask

            keep = document_pool_keep_mask(scores, segment_ids, self._pool_sizes(segment_ids))
        else:
            document_scores = ops.doc_sum_scatter(scores, segment_ids)
            keep = ops.pool_keep_mask(document_scores, self._pool_sizes(segment_ids))

        if self.gating_function == MoERouterGatingFunction.topk_softmax:
            selection_logits = logits.masked_fill(~keep, float("-inf"))
            _, expert_indices = selection_logits.topk(self.top_k, dim=-1)
            selected_logits = logits.gather(-1, expert_indices)
            expert_weights = selected_logits.softmax(dim=-1)
        else:
            selection_scores = scores.masked_fill(~keep, float("-inf"))
            if self._profile_top16 and self.top_k == 16 and self.num_experts == 512:
                from olmo_core.ops.moe_top16 import top16_native_indices

                expert_indices = top16_native_indices(selection_scores)
            else:
                _, expert_indices = selection_scores.topk(self.top_k, dim=-1)
            expert_weights = scores.gather(-1, expert_indices)

        if self.normalize_expert_weights is not None:
            expert_weights = F.normalize(expert_weights, p=self.normalize_expert_weights, dim=-1)
        if self.restore_weight_scale:
            expert_weights = expert_weights * self.top_k
        if self.expert_weight_scale is not None:
            expert_weights = expert_weights * self.expert_weight_scale
        if self.original_top_k is not None and self.top_k != self.original_top_k:
            expert_weights = expert_weights * (self.original_top_k / self.top_k) ** 0.5

        with torch.no_grad():
            batched_counts = ops.batched_histc(expert_indices, self.num_experts).sum(dim=1)
            counts = batched_counts.sum(dim=0)

        aux_loss_info = (scores, logits, counts, batched_counts, loss_div_factor)
        return expert_weights, expert_indices, counts, aux_loss_info

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, emo_pool="
            f"[{self.emo.min_document_expert_pool}, {self.emo.max_document_expert_pool}]"
        )
