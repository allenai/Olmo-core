from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

import olmo_core.ops.moe as ops
from olmo_core._nvtx import nvtx
from olmo_core.config import Config, DType
from olmo_core.distributed.utils import (
    _HiddenTensor,
    distribute_like,
    get_local_tensor,
    hide_from_torch,
    is_distributed,
    unhide_from_torch,
)
from olmo_core.exceptions import OLMoConfigurationError

from ...output_discard_checkpoint import OutputDiscardCheckpoint
from ..emo import EmoRouterConfig
from ..loss import MoELoadBalancingLossGranularity, load_balancing_loss, router_z_loss
from ..router import MoERouterGatingFunction, _uniform_expert_assignment

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType


def _cast_to_fp32(x: torch.Tensor) -> torch.Tensor:
    return x.float()


@dataclass
class MoERouterConfigV2(Config):
    """
    A configuration class for easily building any of the different MoE router modules.
    """

    d_model: int

    num_experts: int

    top_k: int
    bias: bool = False
    jitter_eps: Optional[float] = None
    normalize_expert_weights: Optional[float] = None
    uniform_expert_assignment: bool = False
    random_expert_assignment: bool = False
    bias_gamma: Optional[float] = None
    gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax
    dtype: Optional[DType] = None
    record_routing_batch_size: bool = False
    lb_loss_weight: Optional[float] = None
    lb_loss_count_source: Literal["executed", "router_selected"] = field(
        default="executed", kw_only=True
    )
    """Use executed assignments (including replay), or router-selected top-k for balancing only."""
    lb_loss_granularity: MoELoadBalancingLossGranularity = (
        MoELoadBalancingLossGranularity.local_batch
    )
    z_loss_weight: Optional[float] = None
    orth_loss_weight: Optional[float] = None
    restore_weight_scale: bool = False  # if True, multiply the router weights by topK so that the scores have similar scale as dense models.
    expert_weight_scale: Optional[float] = None
    original_top_k: Optional[
        int
    ] = None  # for restoring weight scales to match a model trained with a different top_k
    use_recompute_fp32_cast: bool = False  # whether to use an OutputDiscardCheckpoint to save the fp32 cast of the router input for recomputation in backward, which can save memory at the cost of extra compute in backward.
    score_correction_bias: bool = False
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    sigmoid_stability_epsilon: float = 1e-7
    global_load_balancing: bool = False
    """Compute load-balancing loss from assignment counts averaged across the DP group."""
    emo: Optional[EmoRouterConfig] = None
    """Optional EMO document-pool routing policy."""

    def num_params(self) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        num_params = 0

        num_params += self.d_model * self.num_experts
        if self.bias:
            num_params += self.num_experts

        return num_params

    def build(
        self,
        init_device: str = "cpu",
    ) -> "MoERouterV2":
        """
        Build the corresponding MoE router module.

        :param d_model: The model dimensionality.
        :param num_experts: The number of experts.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)

        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()

        if self.emo is not None:
            if self.lb_loss_count_source == "router_selected":
                raise OLMoConfigurationError(
                    "Router-selected balancing counts do not support EMO routing"
                )
            from .emo_router import EmoRouterV2

            return EmoRouterV2(**kwargs, init_device=init_device)
        return MoERouterV2(**kwargs, init_device=init_device)


class MoERouterV2(nn.Module):
    """
    A base class for MoE router modules.

    :param d_model: The model dimensionality (hidden size).
    :param num_experts: The total number of experts.
    :param top_k: The number of experts to assign to each item/token.
    :param jitter_eps: Controls the amount of noise added to the input during training.
    :param normalize_expert_weights: The type of norm (e.g. ``2.0`` for L2 norm) to use to normalize
        the expert weights.
    :param uniform_expert_assignment: Force uniform assignment. Useful for benchmarking.
    :param bias_gamma: If set to a positive float, experts scores for top-k routing will be adjusted
        by a bias following the "auxiliary-loss-free load balancing" strategy from DeepSeek-v3.
        A reasonable value is on the order of 0.0001.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        top_k: int,
        bias: bool = False,
        jitter_eps: Optional[float] = None,
        normalize_expert_weights: Optional[float] = None,
        uniform_expert_assignment: bool = False,
        random_expert_assignment: bool = False,
        bias_gamma: Optional[float] = None,
        gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax,
        lb_loss_weight: Optional[float] = None,
        lb_loss_count_source: Literal["executed", "router_selected"] = "executed",
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        orth_loss_weight: Optional[float] = None,
        init_device: str = "cpu",
        record_routing_batch_size: bool = False,
        restore_weight_scale: bool = False,
        expert_weight_scale: Optional[float] = None,
        original_top_k: Optional[int] = None,
        use_recompute_fp32_cast=False,
        score_correction_bias: bool = False,
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        sigmoid_stability_epsilon: float = 1e-7,
        global_load_balancing: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts

        self.top_k = top_k
        self.use_bias = bias
        self.jitter_eps = jitter_eps
        self.normalize_expert_weights = normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment
        self.random_expert_assignment = random_expert_assignment
        self.bias_gamma = bias_gamma
        self.gating_function = gating_function
        self.lb_loss_weight = lb_loss_weight
        self.lb_loss_count_source = lb_loss_count_source
        self.lb_loss_granularity = lb_loss_granularity
        self.z_loss_weight = z_loss_weight
        self.orth_loss_weight = orth_loss_weight
        self.group: Optional[dist.ProcessGroup] = None
        self.cp_mesh: Optional[DeviceMesh] = None
        self.tp_mesh: Optional[DeviceMesh] = None
        self.record_routing_batch_size = record_routing_batch_size
        self.restore_weight_scale = restore_weight_scale
        self.expert_weight_scale = expert_weight_scale
        self.original_top_k = original_top_k
        self.use_recompute_fp32_cast = use_recompute_fp32_cast
        self.score_correction_bias = score_correction_bias
        self.n_group = n_group
        self.topk_group = topk_group
        self.sigmoid_stability_epsilon = sigmoid_stability_epsilon
        self.global_load_balancing = global_load_balancing
        self.lb_process_group: Optional[dist.ProcessGroup] = None

        if self.lb_loss_count_source not in ("executed", "router_selected"):
            raise OLMoConfigurationError(
                "lb_loss_count_source must be 'executed' or 'router_selected'"
            )
        if self.lb_loss_count_source == "router_selected" and (
            self.global_load_balancing
            or self.gating_function != MoERouterGatingFunction.softmax
            or self.bias_gamma is not None
            or self.score_correction_bias
            or self.n_group is not None
            or self.topk_group is not None
            or self.uniform_expert_assignment
            or self.random_expert_assignment
        ):
            raise OLMoConfigurationError(
                "Router-selected-count balancing currently requires plain softmax routing and local "
                "balancing; global counts, routing biases, groups and assignment overrides "
                "are not supported"
            )

        if (
            self.global_load_balancing
            and self.lb_loss_granularity == MoELoadBalancingLossGranularity.instance
        ):
            raise OLMoConfigurationError(
                "global_load_balancing does not support instance-granularity load-balancing "
                "loss because per-instance assignments cannot be combined across unrelated "
                "data-parallel batches"
            )

        if (self.n_group is None) != (self.topk_group is None):
            # Grouped routing needs both knobs; with only one set, the group-masking branch
            # (which checks both) never runs and routing silently falls back to global top-k.
            raise OLMoConfigurationError(
                "Grouped routing requires both n_group and topk_group to be set (or neither); "
                f"got n_group={self.n_group}, topk_group={self.topk_group}."
            )

        if self.gating_function == MoERouterGatingFunction.topk_softmax and (
            self.n_group is not None or self.topk_group is not None
        ):
            raise OLMoConfigurationError(
                "Grouped routing (n_group/topk_group) is not supported with "
                "gating_function='topk_softmax': group masking is only applied on the "
                "softmax/sigmoid selection path (get_top_k), so the topk_softmax path would "
                "silently route to experts outside the allowed group."
            )

        if (
            self.gating_function == MoERouterGatingFunction.topk_softmax
            and self.random_expert_assignment
        ):
            raise OLMoConfigurationError(
                "random_expert_assignment is not supported with gating_function='topk_softmax': "
                "the topk_softmax path selects experts from the raw logits, so it would ignore the "
                "randomized scores and route deterministically."
            )

        if (
            self.n_group is not None
            and self.topk_group is not None
            and self.num_experts % self.n_group == 0
        ):
            experts_per_group = self.num_experts // self.n_group
            if self.top_k > self.topk_group * experts_per_group:
                raise OLMoConfigurationError(
                    f"top_k ({self.top_k}) exceeds the number of experts reachable under grouped "
                    f"routing (topk_group * experts_per_group = {self.topk_group} * "
                    f"{experts_per_group}); top-k would otherwise select masked out-of-group experts."
                )

        if self.bias_gamma is not None or self.score_correction_bias:
            if self.bias_gamma is not None:
                assert self.bias_gamma > 0
            self.register_buffer("score_bias", torch.zeros(self.num_experts, device=init_device))
        else:
            self.register_buffer("score_bias", None)

        # NOTE: historical FSDP/composable-DDP workaround. These mutable metric
        # tensors are hidden so wrappers do not register/manage/move/reduce them
        # as buffers. MultiGroupDDP may no longer require all of this, but
        # torch.compile() still dislikes BufferCache-style mutation here.
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=init_device)
        )
        self._global_batch_size_per_expert: Optional[_HiddenTensor] = None
        self._score_bias_batch_size_per_expert: Optional[_HiddenTensor] = None
        self._load_balancing_loss: Optional[_HiddenTensor] = None
        self._z_loss: Optional[_HiddenTensor] = None
        self._orth_loss: Optional[_HiddenTensor] = None

        self.weight = nn.Parameter(
            torch.empty(self.num_experts * self.d_model, device=init_device, dtype=dtype)
        )
        self.bias = (
            nn.Parameter(torch.empty(self.num_experts, device=init_device, dtype=dtype))
            if bias
            else None
        )
        self.reset_parameters()

        self._debug_index = None
        self._recompute_cache = None
        self.use_recompute_cache = False

    def reset_parameters(self):
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=self.device)
        )
        if self.global_load_balancing:
            self._global_batch_size_per_expert = hide_from_torch(
                torch.zeros(self.num_experts, device=self.device)
            )

        if self.score_bias is not None:
            score_bias = cast(torch.Tensor, self.score_bias)
            score_bias.zero_()

        if self.bias_gamma is not None:
            assert self.score_bias is not None
            self._score_bias_batch_size_per_expert = hide_from_torch(
                torch.zeros(self.num_experts, device=self.device)
            )

        if self.lb_loss_weight is not None:
            self._load_balancing_loss = hide_from_torch(torch.zeros([], device=self.device))

        if self.z_loss_weight is not None:
            self._z_loss = hide_from_torch(torch.zeros([], device=self.device))

        if self.orth_loss_weight is not None:
            self._orth_loss = hide_from_torch(torch.zeros([], device=self.device))

        nn.init.trunc_normal_(self.weight, std=0.02, a=-3 * 0.02, b=3 * 0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02, a=-3 * 0.02, b=3 * 0.02)

    @property
    def device(self) -> torch.device:
        return self.weight.device if self.weight.device.type != "meta" else torch.device("cpu")

    def extra_repr(self):
        return f"in_features={self.d_model}, num_experts={self.num_experts}"

    @property
    def score_bias_batch_size_per_expert(self) -> Optional[torch.Tensor]:
        if self.bias_gamma is not None:
            if self._score_bias_batch_size_per_expert is None:
                self._score_bias_batch_size_per_expert = hide_from_torch(
                    torch.zeros(self.num_experts, device=self.device)
                )
            elif self._score_bias_batch_size_per_expert.device != self.device:
                self._score_bias_batch_size_per_expert = self._score_bias_batch_size_per_expert.to(
                    self.device
                )
        return (
            None
            if self._score_bias_batch_size_per_expert is None
            else unhide_from_torch(self._score_bias_batch_size_per_expert)
        )

    @score_bias_batch_size_per_expert.setter
    def score_bias_batch_size_per_expert(self, value: torch.Tensor):
        self._score_bias_batch_size_per_expert = hide_from_torch(value)

    @property
    def batch_size_per_expert(self) -> torch.Tensor:
        if self._batch_size_per_expert.device != self.device:
            self._batch_size_per_expert = self._batch_size_per_expert.to(self.device)
        return unhide_from_torch(self._batch_size_per_expert)

    @batch_size_per_expert.setter
    def batch_size_per_expert(self, value: torch.Tensor):
        self._batch_size_per_expert = hide_from_torch(value)

    @property
    def global_batch_size_per_expert(self) -> Optional[torch.Tensor]:
        if not self.global_load_balancing:
            return None
        if self._global_batch_size_per_expert is None:
            self._global_batch_size_per_expert = hide_from_torch(
                torch.zeros(self.num_experts, device=self.device)
            )
        elif self._global_batch_size_per_expert.device != self.device:
            self._global_batch_size_per_expert = self._global_batch_size_per_expert.to(self.device)
        return unhide_from_torch(self._global_batch_size_per_expert)

    @global_batch_size_per_expert.setter
    def global_batch_size_per_expert(self, value: torch.Tensor):
        self._global_batch_size_per_expert = hide_from_torch(value)

    @property
    def load_balancing_loss(self) -> Optional[torch.Tensor]:
        if self.lb_loss_weight is not None:
            if self._load_balancing_loss is None:
                self._load_balancing_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._load_balancing_loss.device != self.device:
                self._load_balancing_loss = self._load_balancing_loss.to(self.device)
        return (
            None
            if self._load_balancing_loss is None
            else unhide_from_torch(self._load_balancing_loss)
        )

    @load_balancing_loss.setter
    def load_balancing_loss(self, value: torch.Tensor):
        self._load_balancing_loss = hide_from_torch(value)

    @property
    def z_loss(self) -> Optional[torch.Tensor]:
        if self.z_loss_weight is not None:
            if self._z_loss is None:
                self._z_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._z_loss.device != self.device:
                self._z_loss = self._z_loss.to(self.device)
        return None if self._z_loss is None else unhide_from_torch(self._z_loss)

    @z_loss.setter
    def z_loss(self, value: torch.Tensor):
        self._z_loss = hide_from_torch(value)

    @property
    def orth_loss(self) -> Optional[torch.Tensor]:
        if self.orth_loss_weight is not None:
            if self._orth_loss is None:
                self._orth_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._orth_loss.device != self.device:
                self._orth_loss = self._orth_loss.to(self.device)
        return None if self._orth_loss is None else unhide_from_torch(self._orth_loss)

    @orth_loss.setter
    def orth_loss(self, value: torch.Tensor):
        self._orth_loss = hide_from_torch(value)

    @torch.no_grad()
    def post_batch(self, dry_run: bool = False):
        if self.bias_gamma is None or not self.training:
            return

        assert self.score_bias is not None
        assert self.score_bias_batch_size_per_expert is not None
        score_bias = cast(torch.Tensor, self.score_bias)
        batch_size_per_expert = self.score_bias_batch_size_per_expert

        # Maybe reduce across the process group.
        if is_distributed():
            dist.all_reduce(batch_size_per_expert, group=self.group)

        ideal_batch_size_per_expert = batch_size_per_expert.mean(
            dim=0, keepdim=True, dtype=torch.float32
        )
        bias_delta = self.bias_gamma * (ideal_batch_size_per_expert - batch_size_per_expert).sign()
        # NOTE: have to be careful here to manage the case where `score_bias` is a DTensor.
        bias_delta = distribute_like(score_bias, bias_delta)

        if not dry_run:
            get_local_tensor(score_bias).add_(get_local_tensor(bias_delta))

        # Reset the accumulator.
        batch_size_per_expert.zero_()

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.jitter_eps is None or not self.training:
            return x
        else:
            low = 1.0 - self.jitter_eps
            high = 1.0 + self.jitter_eps
            noise = torch.rand_like(x)
            return x * (low + noise * (high - low))

    @nvtx.annotate("MoERouter.get_top_k", color="blue")
    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_weights: torch.Tensor
        expert_indices: torch.Tensor
        selection_scores = scores
        if self.score_bias is not None:
            selection_scores = scores + self.score_bias.unsqueeze(0)  # type: ignore[union-attr]

        if self.n_group is not None and self.topk_group is not None:
            if self.num_experts % self.n_group != 0:
                raise OLMoConfigurationError(
                    f"num_experts ({self.num_experts}) must be divisible by n_group ({self.n_group})"
                )
            if not (0 < self.topk_group <= self.n_group):
                raise OLMoConfigurationError(
                    f"topk_group ({self.topk_group}) must be in [1, n_group={self.n_group}]"
                )
            experts_per_group = self.num_experts // self.n_group
            group_scores = (
                selection_scores.view(*selection_scores.shape[:-1], self.n_group, experts_per_group)
                .topk(min(2, experts_per_group), dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(-1, group_idx, True)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(*selection_scores.shape[:-1], self.n_group, experts_per_group)
                .reshape_as(selection_scores)
            )
            selection_scores = selection_scores.masked_fill(~score_mask, float("-inf"))

        with torch.no_grad() if self.score_bias is not None else torch.enable_grad():
            if self.top_k == 1:
                _, expert_indices = selection_scores.max(dim=-1, keepdim=True)
            else:
                _, expert_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        expert_weights = scores.gather(-1, expert_indices)

        if self.uniform_expert_assignment:
            expert_indices = _uniform_expert_assignment(expert_indices, self.num_experts)
            expert_weights = scores.gather(-1, expert_indices)

        return expert_weights, expert_indices

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x.float(),
            get_local_tensor(self.weight).view(self.num_experts, self.d_model).float(),
            None if self.bias is None else get_local_tensor(self.bias).float(),
        )

    @torch.no_grad()
    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}

        # Load imbalance.
        batch_size_per_expert = self.batch_size_per_expert
        out["load imbalance"] = (
            batch_size_per_expert.max() / batch_size_per_expert.mean(dtype=torch.float),
            ReduceType.max,
        )

        if self.global_load_balancing:
            global_batch_size_per_expert = self.global_batch_size_per_expert
            assert global_batch_size_per_expert is not None
            out["global load imbalance"] = (
                global_batch_size_per_expert.max()
                / global_batch_size_per_expert.mean(dtype=torch.float),
                # All DP ranks accumulate the same globally reduced counts.
                ReduceType.mean,
            )

        # record the number of tokens routed to each routed expert.
        if self.record_routing_batch_size:
            for i in range(self.num_experts):
                out[f"expert {i} assigned tokens"] = (
                    self.batch_size_per_expert[i],
                    ReduceType.mean,
                )

        # Load balancing loss.
        if self.lb_loss_weight is not None:
            assert self.load_balancing_loss is not None
            out["load balancing loss"] = (
                self.lb_loss_weight * self.load_balancing_loss,
                ReduceType.mean,
            )
            out["load balancing loss unscaled"] = (
                self.load_balancing_loss.clone(),
                ReduceType.mean,
            )

        # Router Z loss.
        if self.z_loss_weight is not None:
            assert self.z_loss is not None
            out["router Z loss"] = (self.z_loss_weight * self.z_loss, ReduceType.mean)
            out["router Z loss unscaled"] = (self.z_loss.clone(), ReduceType.mean)

        if self.orth_loss_weight is not None:
            assert self.orth_loss is not None
            out["router orthogonal loss"] = (
                self.orth_loss_weight * self.orth_loss,
                ReduceType.mean,
            )
            out["router orthogonal loss unscaled"] = (self.orth_loss.clone(), ReduceType.mean)

        if reset:
            self.reset_metrics()

        return out

    def reset_metrics(self):
        if (bz_per_expert := self.batch_size_per_expert) is not None:
            bz_per_expert.zero_()
        if (global_bz_per_expert := self.global_batch_size_per_expert) is not None:
            global_bz_per_expert.zero_()
        if (lb_loss := self.load_balancing_loss) is not None:
            lb_loss.zero_()
        if (z_loss := self.z_loss) is not None:
            z_loss.zero_()
        if (orth_loss := self.orth_loss) is not None:
            orth_loss.zero_()

    def _break_ties(self, logits):
        eid = torch.arange(logits.size(-1), device=logits.device, dtype=torch.float32)
        logits = logits + eid * 1e-7  # small constant to break ties
        return logits

    def _quantize_scores(self, logits):
        q = 2**14  # tune;
        # NOTE: smaller q (coarser) tolerates larger drift
        #       larger q (finer) tolerates smaller drift
        # q = 2**13 = 8192 → Δ ≈ 1/8192 ≈ 1.22e-4
        # q = 2**14 = 16384 → Δ ≈ 1/16384 ≈ 6.10e-5
        # q = 2**15 = 32768 → Δ ≈ 1/32768 ≈ 3.05e-5
        logits = torch.round(logits.float() * q) / q
        return logits

    @nvtx.annotate("MoERouter.forward", color="blue")
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
        """
        Given the input ``x`` of shape ``(B, S, d_model)``, compute the experts assignment.

        :returns: The expert weights of shape ``(B, S, top_k)``,
            the expert indices of shape ``(B, S, top_k)``,
            the total number of items routed to each expert, with shape ``(num_experts,)``,
            and optionally the auxiliary losses.
        """
        del segment_ids

        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # Keep activation in bf16/fp16 in forward graph, and only materialize fp32
        # router input through OutputDiscardCheckpoint so backward can recompute it.
        cast_checkpoint: Optional[OutputDiscardCheckpoint] = None
        if torch.is_grad_enabled() and x.requires_grad and self.use_recompute_fp32_cast:
            cast_checkpoint = OutputDiscardCheckpoint()
            x_fp32 = cast(torch.Tensor, cast_checkpoint.checkpoint(_cast_to_fp32, x))
        else:
            x_fp32 = x.float()

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x_fp32).float()
        if cast_checkpoint is not None:
            # Recompute fp32 cast before linear backward consumes the saved input.
            cast_checkpoint.discard_output_and_register_recompute(logits)

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function in (
            MoERouterGatingFunction.softmax,
            MoERouterGatingFunction.topk_softmax,
        ):
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(logits)
            # to avoid NaNs in the load balancing loss
            # if all logits of a token are very negative for all experts, sigmoid gives 0 for all experts, causing NaNs when we div by the sum.
            if self.sigmoid_stability_epsilon:
                scores = scores + self.sigmoid_stability_epsilon
        else:
            raise NotImplementedError(self.gating_function)

        if self.random_expert_assignment:
            scores = scores * 0 + torch.rand_like(scores)  # random, but keep the autograd graph

        if scores_only:
            if self.gating_function == MoERouterGatingFunction.topk_softmax:
                # The scores-only path (e.g. shared-expert mixing) returns the dense score
                # vector over all experts; topk_softmax has no top-k masking here, so it would
                # silently mix in non-selected experts. Reject rather than route incorrectly.
                raise NotImplementedError(
                    "gating_function='topk_softmax' is not supported for scores-only routing "
                    "(the dense score vector would ignore top_k)."
                )
            if self.normalize_expert_weights is not None:
                scores = scores.div(
                    torch.norm(
                        scores,
                        p=self.normalize_expert_weights,
                        dim=-1,
                        keepdim=True,
                    )
                )
            # If we only need the scores, return them directly.
            return scores, None, None, None

        UES_QUANT_SCORES = False
        replay_indices = getattr(self, "replay_expert_indices", None)
        if replay_indices is not None:
            if replay_indices.shape != (*scores.shape[:-1], self.top_k):
                raise ValueError("Replay expert indices do not match the router input shape")
            expert_indices = replay_indices.to(device=scores.device, dtype=torch.long)
            if self.gating_function == MoERouterGatingFunction.topk_softmax:
                expert_weights = logits.gather(-1, expert_indices).softmax(dim=-1)
            else:
                expert_weights = scores.gather(-1, expert_indices)

        elif UES_QUANT_SCORES:
            # TODO: merge into get_top_k
            scores_sel = self._quantize_scores(scores)
            scores_sel = self._break_ties(scores_sel)
            expert_indices = self.get_top_k(scores_sel)[1]

            # weights from original scores/logits (not quantized), to keep smooth grads
            expert_weights = scores.gather(-1, expert_indices)

        elif self.gating_function == MoERouterGatingFunction.topk_softmax:
            selection_logits = logits
            if self.score_bias is not None:
                selection_logits = logits + self.score_bias.unsqueeze(0)  # type: ignore[union-attr]

            with torch.no_grad() if self.score_bias is not None else torch.enable_grad():
                _, expert_indices = torch.topk(selection_logits, self.top_k, dim=-1)
            if self.uniform_expert_assignment:
                expert_indices = _uniform_expert_assignment(expert_indices, self.num_experts)

            expert_logits = logits.gather(-1, expert_indices)
            expert_weights = expert_logits.softmax(dim=-1, dtype=expert_logits.dtype)
        else:
            # shape: (batch_size, seq_len, top_k)
            expert_weights, expert_indices = self.get_top_k(scores)

        # TODO: verify the recompute-index cache is correct under activation checkpointing;
        # known-broken under pipeline parallelism (the cached indices don't survive PP microbatching).
        if self.use_recompute_cache and replay_indices is None:
            if self._recompute_cache is None:  # first forward
                if torch.is_grad_enabled():
                    self._recompute_cache = expert_indices.detach()  # save for recompute
            else:  # recompute
                expert_indices = self._recompute_cache  # use saved expert indices
                self._recompute_cache = None  #
        else:
            pass

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        # if used together normalize_expert_weights=True,
        # then the weights for each token sum to TOP_K instead of 1.0
        if self.restore_weight_scale:
            expert_weights = expert_weights * self.top_k

        if self.expert_weight_scale is not None:
            expert_weights = expert_weights * self.expert_weight_scale

        if self.original_top_k is not None and self.top_k != self.original_top_k:
            expert_weights = expert_weights * (self.original_top_k / self.top_k) ** 0.5

        with torch.no_grad():
            # Histogram the expert ids to identify the number of items/tokens routed to each expert.
            # shape: (batch_size, seq_len, num_experts)
            batched_batch_size_per_expert = ops.batched_histc(expert_indices, self.num_experts)
            # shape: (batch_size, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts,)
            batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)

        # with torch.no_grad():
        #     if self._debug_index is None: # first forward
        #         self._debug_index = expert_indices.detach().cpu()
        #     else: # recompute
        #         recomputed = expert_indices.detach().cpu()
        #         if not (self._debug_index == recomputed).all():
        #             print("Debug expert indices mismatch!")
        #             with open(f"/workspace/{dist.get_rank()}_ori_index.pt", "wb") as f:
        #                 torch.save(self._debug_index, f)
        #             with open(f"/workspace/{dist.get_rank()}_recomputed_index.pt", "wb") as f:
        #                 torch.save(recomputed, f)
        #             debug_info = {
        #                 'scores': scores.detach().cpu(),
        #                 'expert_weights': expert_weights.detach().cpu(),
        #                 'logits': logits.detach().cpu(),
        #                 'weight': self.weight.data.detach().cpu(),
        #                 'x': x.detach().cpu(),
        #             }
        #             with open(f"/workspace/{dist.get_rank()}_debug_info.pt", "wb") as f:
        #                 torch.save(debug_info, f)

        #             raise ValueError("Debug expert indices mismatch!")
        #         else:
        #             self._debug_index = None #

        aux_loss_info = (
            scores,
            logits,
            batch_size_per_expert,
            batched_batch_size_per_expert,
            loss_div_factor,
        )
        return expert_weights, expert_indices, batch_size_per_expert, aux_loss_info

    @nvtx.annotate("MoERouter.compute_aux_loss")
    def compute_aux_loss(
        self,
        scores,
        logits,
        batch_size_per_expert,
        batched_batch_size_per_expert,
        loss_div_factor,
        *,
        accumulate_metrics: bool = True,
    ) -> Optional[torch.Tensor]:
        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            global_batch_size_per_expert = None
            if self.global_load_balancing:
                if self.lb_process_group is None:
                    raise RuntimeError(
                        "global_load_balancing requires a load-balancing process group; "
                        "apply data parallelism before training"
                    )
                # DDP averages parameter gradients, and the training module supplies a
                # loss_div_factor normalized by the DP world size. Average the global counts
                # to preserve the single-rank-equivalent auxiliary-loss scale.
                global_batch_size_per_expert = batch_size_per_expert.float().clone()
                # TODO: This is a tiny but synchronous collective for every routed MoE layer
                # and microbatch, so latency can dominate its bandwidth cost, especially across
                # nodes. Consider coalescing count reductions across layers/microbatches or
                # overlapping asynchronous reductions with other work.
                dist.all_reduce(
                    global_batch_size_per_expert,
                    op=dist.ReduceOp.SUM,
                    group=self.lb_process_group,
                )
                global_batch_size_per_expert.div_(dist.get_world_size(self.lb_process_group))

            if self.lb_loss_weight is not None:
                assert self.load_balancing_loss is not None

                # Make sure scores are normalized, otherwise load balancing loss doesn't work well.
                if self.gating_function == MoERouterGatingFunction.sigmoid:
                    scores = scores / scores.sum(dim=-1, keepdim=True)

                lb_batch_size_per_expert = (
                    global_batch_size_per_expert
                    if global_batch_size_per_expert is not None
                    else batch_size_per_expert
                )
                lb_batched_batch_size_per_expert = batched_batch_size_per_expert
                if self.lb_loss_count_source == "router_selected":
                    # Keep replayed dispatch, selected weights, load metrics and z-loss intact.
                    # Compute fresh nondifferentiable counts from this forward's scores, not
                    # router-local replay state that may have changed before recomputation.
                    with torch.no_grad():
                        current_indices = self.get_top_k(scores.detach())[1]
                        lb_batched_batch_size_per_expert = ops.batched_histc(
                            current_indices, self.num_experts
                        ).sum(dim=1)
                        lb_batch_size_per_expert = lb_batched_batch_size_per_expert.sum(dim=0)

                lb_loss = load_balancing_loss(
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    expert_scores=scores,
                    batch_size_per_expert=lb_batch_size_per_expert,
                    batched_batch_size_per_expert=lb_batched_batch_size_per_expert,
                    granularity=self.lb_loss_granularity,
                    loss_div_factor=loss_div_factor,
                    tp_mesh=self.tp_mesh,
                    cp_mesh=self.cp_mesh,
                )
                if accumulate_metrics:
                    # NOTE: under TP the loss is a replicated-scalar DTensor; the metric
                    # accumulator is a plain tensor, so accumulate its local value to avoid
                    # a plain-tensor += DTensor in-place error. The training path below keeps
                    # the DTensor for correct autograd.
                    self.load_balancing_loss += get_local_tensor(lb_loss.detach())

                scaled_lb_loss = self.lb_loss_weight * lb_loss
                aux_loss = scaled_lb_loss

            if self.z_loss_weight is not None:
                assert self.z_loss is not None

                z_loss = router_z_loss(
                    expert_logits=logits,
                    loss_div_factor=loss_div_factor,
                    tp_mesh=self.tp_mesh,
                    cp_mesh=self.cp_mesh,
                )
                if accumulate_metrics:
                    self.z_loss += get_local_tensor(z_loss.detach())

                scaled_z_loss = self.z_loss_weight * z_loss
                aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

            if self.orth_loss_weight is not None:
                # TODO: should only compute orthogonal loss on the last micro batch because the loss is only computed on the router weights.
                assert self.orth_loss is not None

                # NOTE: loss_div_factor is the total number of tokens in the global batch.
                # orth_loss_div_factor  is approximately the number of micro batches in the global batch.
                # since the orthogonal loss is computed `num_micro_batches` times (should have the same loss each time since it does not chagne wrt data), we need to scale it down.
                if loss_div_factor is None:
                    raise NotImplementedError(
                        "Orthogonal loss requires a loss_div_factor to be set."
                    )
                # orth_loss_factor = (logits.size(0) * logits.size(1)) / loss_div_factor  # --> divide by num_micro_batches
                # or
                orth_loss_factor = 1 / loss_div_factor  # --> divide by num_tokens in micro batch

                orth_loss = self.compute_orthogonal_loss() * orth_loss_factor

                if accumulate_metrics:
                    self.orth_loss += get_local_tensor(orth_loss.detach())

                scaled_orth_loss = self.orth_loss_weight * orth_loss
                aux_loss = scaled_orth_loss if aux_loss is None else aux_loss + scaled_orth_loss
            if accumulate_metrics:
                self.batch_size_per_expert += batch_size_per_expert
                if global_batch_size_per_expert is not None:
                    assert self.global_batch_size_per_expert is not None
                    self.global_batch_size_per_expert += global_batch_size_per_expert
                if self.bias_gamma is not None:
                    assert self.score_bias_batch_size_per_expert is not None
                    self.score_bias_batch_size_per_expert += batch_size_per_expert

        return aux_loss

    def set_load_balancing_process_group(self, group: dist.ProcessGroup) -> None:
        self.lb_process_group = group

    def compute_orthogonal_loss(self) -> torch.Tensor:
        """
        Computes the orthogonal loss for the router.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Orthogonal loss computation is not implemented.")

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        del float8_enabled
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Shard(1),),
                use_local_output=True,
            ),
        )
        self.tp_mesh = tp_mesh
        self.register_parameter(
            "weight", nn.Parameter(distribute_tensor(self.weight, tp_mesh, [Replicate()]))
        )
        if self.bias is not None:
            self.register_parameter(
                "bias", nn.Parameter(distribute_tensor(self.bias, tp_mesh, [Replicate()]))
            )

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.cp_mesh = cp_mesh
