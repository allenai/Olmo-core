"""
Nested-width FFN mixture ("flexible-compute FFN", learned router).

The role-gated FFN (:mod:`olmo_core.nn.role_gated_ffn`) gives every context-document token the
same binary choice -- full MLP or nothing -- from a deterministic role rule. That failed: gating
from layer 4 could not fit the training data at all (CE wall 0.95) and gating from layer 12 fit
but evaluated at 0.316. Doc tokens need *some* MLP compute, and a role-blind all-or-nothing rule
cannot allocate it.

This module replaces the binary choice with a **ladder of nested widths** plus a learned per-token
router. Each block's FFN exposes several granularities that share one set of weights by prefix-
slicing the hidden dimension::

    rung 0: y = w2[:, :9728] (act(w1[:9728] x) * w3[:9728] x)   cost 1.0    (the original FFN)
    rung 1: y = w2[:, :2432] (act(w1[:2432] x) * w3[:2432] x)   cost 1/4
    rung 2: ... 608 units                                        cost 1/16
    rung 3: ... 152 units                                        cost 1/64
    rung 4: y = 0                                                cost 0      (null / AdaMoE)

A per-layer linear router scores the rungs from the token's hidden state; the token runs on
exactly ONE rung, chosen by argmax. Tokens are grouped by rung and each group runs a narrow
matmul, so the saving is a genuinely smaller GEMM, not a masked multiply.

Relation to the literature
--------------------------
- **AdaMoE** (Zeng et al. 2024) contributes the *null expert* -- a zero-output, zero-FLOP entry in
  the routing pool -- and the idea of decoupling load balancing from a budget term that dials
  average compute. But its experts are all the SAME size, so its dynamic range is the top-k value
  (~14% FLOP savings on Mixtral), and it presupposes an existing MoE. Qwen3-4B is dense.
- **MatFormer** (Devvrit et al. 2023) contributes nested prefix-sliced FFN granularities sharing
  one weight matrix -- but chooses a granularity per *deployment*, not per token.
- **MoNE** (Jain et al. 2024) is the routed version of nesting under a compute budget; this module
  is closest to it.
- **Mixture-of-Depths** (Raposo et al. 2024) contributes the static-shape capacity trick.

The synthesis here is: nested widths (MatFormer/MoNE geometry) + a null rung (AdaMoE) + a budget
hinge loss. Crucially it adds NO expert parameters -- only a tiny router and per-rung gains -- so
it can be fitted onto an already-trained checkpoint (e.g. the v25 pooled-doc-KV model) instead of
requiring a fresh pretrain.

Why it should not repeat the v24/v26 failure
--------------------------------------------
1. The bottom rung is not the only cheap option: a token that needs a little MLP can take 1/64
   instead of being forced to choose between full and zero.
2. The router is **zero-initialized to pick the full rung with probability ~1**, so at step 0 the
   model is bit-identical to its base. Compute is given up only as the budget loss pushes it.
3. The budget term is a **hinge on the batch mean**, not a per-token pull toward zero, so the
   model is free to spend full compute on a few tokens and nothing on the rest -- which is the
   whole point. Below the target the term is exactly 0 and only CE is optimized.

Objective
---------
``loss = CE + lambda_budget * relu(mean_cost - target)**p
             + w_recon * relative_mse(chosen_rung_output, full_ffn_output)   [optional]
             - w_entropy * router_entropy                                    [optional]``

``mean_cost`` is the router-probability-weighted expected cost averaged over tokens and gated
layers (differentiable in the router). The optional reconstruction term is a *local* distillation
of the full FFN by the small rungs, measured on a small random subset of tokens (default 2%), and
is the literal form of "keep the output similar to the original with the smallest FFN". It runs
only the FFN on that subset -- it involves no attention and no full-context forward, so it is
outside the no-full-attention-at-32k constraint.

Parameter baking
----------------
The router and gain parameters are new state-dict keys, so a base checkpoint must be re-saved with
them present before training (distributed-checkpoint loads are driven by the destination model's
keys). Use ``research/ffn_compute/ffnmoe_train/bake_ffn_moe_into_base.py``, exactly as the
soft-token projector is baked by ``bake_projector_into_base.py``.

Enabled via :meth:`~olmo_core.nn.transformer.model.Transformer.enable_nested_ffn_moe`.
"""

import logging
import math
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = [
    "NestedFFNHolder",
    "NestedFFNRouter",
    "install_nested_ffn_moe",
    "resolve_rung_widths",
    "ffn_importance_permutation",
    "apply_ffn_permutation",
    "post_build_hook_from_config",
]

def post_build_hook_from_config(model_path: str):
    """
    Return a ``post_build_hook`` for
    :meth:`~olmo_core.generate.generation_module.TransformerGenerationModuleConfig.build` that
    enables nested-FFN routing exactly as recorded in ``<model_path>/config.json`` (the ``ffn_moe``
    block the CTC-suite / ffnmoe trainers write), or ``None`` for a checkpoint without one.

    Evaluators call this so a routed checkpoint is scored WITH its router (the deployed model)
    without per-run flags. A DCP load reads only the keys the MODEL asks for, so a checkpoint whose
    extra parameters (router, adapters) are not installed would load WITHOUT error and silently score
    the bare backbone: the returned hook therefore always ends with a key guard that refuses any
    checkpoint holding model keys the built model lacks (``pooled_projector`` excepted -- every
    soft-token run carries it and the evaluated model never uses it). It is ``None`` only when there
    is no config.json.
    """
    import json
    import os

    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    block = cfg.get("ffn_moe")
    kv_block = cfg.get("kv_route")  # the attention-side router; same "enable before load" need
    bs_block = cfg.get("block_skip")  # per-token block skipping
    lora_block = cfg.get("early_lora")  # low-rank early stack (olmo_core.nn.early_lora)
    if not lora_block:
        # configs written before 2026-10-24 dropped the block: rebuild it from the checkpoint (or refuse)
        from .early_lora import lora_block_from_checkpoint

        lora_block = lora_block_from_checkpoint(model_path)

    def _enable(model):
        if not (block or kv_block or bs_block or lora_block):
            return
        if lora_block:
            from .early_lora import enable_from_config_block as enable_lora

            enable_lora(model, lora_block)
        if kv_block:
            from .attention.kv_route import enable_from_config_block

            enable_from_config_block(model, kv_block)
        if bs_block:
            from .block_skip import enable_from_config_block as enable_bs

            enable_bs(model, bs_block)
        if not block:
            return
        model.enable_nested_ffn_moe(
            start_layer=int(block["start_layer"]),
            divisors=[float(x) for x in str(block["divisors"]).split(",")],
            include_null=bool(block.get("include_null", True)),
            width_multiple=int(block.get("width_multiple", 8)),
            trainable_width=int(block.get("trainable_width", 0) or 0),
        )
        print(f"[ffn-moe] routing enabled from config.json: rungs {model._nested_ffn_moe['widths']}", flush=True)
        log.info(
            "[ffn-moe] routing enabled from config.json: start_layer=%s divisors=%s rungs=%s",
            block["start_layer"],
            block["divisors"],
            model._nested_ffn_moe["widths"],
        )

    def hook(model):
        _enable(model)
        guard_checkpoint_keys(model, model_path)

    return hook


def guard_checkpoint_keys(model, model_path: str) -> None:
    """Refuse to load a checkpoint whose ``model.*`` keys include parameters the built model lacks
    (``pooled_projector`` excepted): a DCP load would silently drop them."""
    import os

    try:
        from torch.distributed.checkpoint import FileSystemReader

        keys = FileSystemReader(os.path.join(model_path, "model_and_optim")).read_metadata().state_dict_metadata
    except Exception:  # no / unreadable DCP metadata (e.g. an HF directory): nothing to check
        return
    ck = {k[len("model."):] for k in keys if k.startswith("model.")}
    have = set(model.state_dict().keys())
    extra = sorted(k for k in ck - have if "pooled_projector" not in k)
    if extra:
        raise RuntimeError(
            f"{model_path}: the checkpoint holds {len(extra)} model keys the eval model lacks (e.g. "
            f"{extra[:3]}); loading would silently drop them. Record the variant in config.json "
            "(variants.export_record) or install it before the load."
        )


#: Router bias on the full rung at init, in logit units. Large enough that ``softmax`` puts
#: ~1 - 5e-5 on the full rung, so an un-trained router reproduces the base model exactly.
FULL_RUNG_INIT_BIAS = 10.0


def resolve_rung_widths(
    hidden_size: int, divisors: Sequence[float], *, include_null: bool = True, multiple_of: int = 8
) -> Tuple[List[int], List[float]]:
    """
    Turn a list of divisors into concrete nested hidden widths and their relative costs.

    :param hidden_size: The FFN's full hidden dimension.
    :param divisors: Cost divisors, e.g. ``(1, 4, 16, 64)`` for full, 1/4, 1/16, 1/64. Must be
        strictly increasing and >= 1.
    :param include_null: Append a zero-width (null / AdaMoE) rung with cost 0.
    :param multiple_of: Round each width down to a multiple of this (GEMM friendliness), min 8.

    :returns: ``(widths, costs)`` where ``widths[0] == hidden_size`` and ``costs`` are the widths
        as fractions of ``hidden_size`` (the exact FLOP ratio, since FFN cost is linear in width).

    :raises ValueError: If the divisors are not strictly increasing, or any is < 1.
    """
    if not divisors:
        raise ValueError("need at least one divisor")
    if any(d < 1 for d in divisors):
        raise ValueError(f"divisors must be >= 1, got {list(divisors)}")
    if any(b <= a for a, b in zip(divisors, divisors[1:])):
        raise ValueError(f"divisors must be strictly increasing, got {list(divisors)}")

    widths: List[int] = []
    for d in divisors:
        w = (
            hidden_size
            if d == 1
            else max(multiple_of, int(hidden_size / d) // multiple_of * multiple_of)
        )
        widths.append(min(w, hidden_size))
    if len(set(widths)) != len(widths):
        raise ValueError(f"divisors {list(divisors)} collapse to duplicate widths {widths}")
    if include_null:
        widths.append(0)
    return widths, [w / hidden_size for w in widths]


class NestedFFNHolder:
    """
    Per-forward router state shared by every nested FFN in a model.

    Accumulates the differentiable expected cost (for the budget loss) and hard usage counts (for
    logging) across gated layers, and owns the target/exploration schedules.

    .. note::
        Under activation checkpointing every routed block's forward runs twice (once in forward,
        once in the backward recompute), so :meth:`accumulate` is called twice per block. Both the
        cost sum and the token count double, so every quantity :meth:`metrics` reports -- all of
        which are ratios -- is unaffected. The duplicate loss tensors appended during recompute
        are never read: the regularization loss is built during the forward and the accumulators
        are cleared by the next :meth:`begin_forward`.

    :param costs: Relative cost of each rung (fractions of the full FFN).
    :param target_cost: Budget target -- the mean per-token FFN cost the hinge allows for free.
        ``0.05`` asks for a ~20x average FFN reduction on gated layers.
    :param budget_weight: ``lambda`` on the hinge term.
    :param hinge_power: 1 for a linear hinge (constant pressure above target), 2 for squared.
    :param two_sided: Penalize ``|mean_cost - target|`` instead of the one-sided hinge, so the
        router is pulled back UP when it undershoots the budget. With the one-sided hinge the
        only routing pressure is downward: on a 30-step long-context SFT (fs35 nq 16M) the
        mean cost sank to 0.0026 against a 0.01 target, half the tokens in the routed layers
        got no FFN at all, and the 32k rung collapsed (f1 0.14 vs 0.76 dense). Two-sided keeps
        the realized cost AT the target, which is also what makes the FLOP accounting exact.
    :param target_start: Target at call 0; annealed linearly to ``target_cost`` over
        ``target_anneal_calls`` forwards. Starting at 1.0 means no compute pressure at all until
        the router has begun to differentiate, which avoids the v24-style early CE wall.
    :param target_anneal_calls: Length of that anneal, in forward calls (0 disables).
    :param explore_prob: Probability that a *training* token ignores the argmax and takes a
        uniformly random rung. Gives the small rungs gradient before the router ever prefers them
        (the nesting chicken-and-egg). Annealed to 0 over ``explore_anneal_calls``.
    :param explore_anneal_calls: Length of the exploration anneal, in forward calls.
    :param recon_frac: Fraction of tokens on which to compute the local full-FFN reconstruction
        target (0 disables).
    :param recon_weight: Weight of the reconstruction term.
    :param entropy_weight: Weight of a router-entropy BONUS (subtracted from the loss); keeps the
        router from collapsing to one rung early. 0 disables.
    :param seed: Base seed for the exploration / reconstruction-subset draws. These are seeded per
        ``(seed, call index, layer)`` rather than taken from the ambient RNG so that activation
        checkpointing's recompute reproduces the SAME routing -- see :func:`_forward_generator`.
    """

    def __init__(
        self,
        costs: Sequence[float],
        *,
        target_cost: float = 0.05,
        budget_weight: float = 1.0,
        hinge_power: int = 1,
        two_sided: bool = False,
        target_start: float = 1.0,
        target_anneal_calls: int = 0,
        explore_prob: float = 0.0,
        explore_anneal_calls: int = 0,
        recon_frac: float = 0.0,
        recon_weight: float = 0.0,
        entropy_weight: float = 0.0,
        seed: int = 0,
        start_layer: int = 0,
        n_layers: int = 0,
        layer_curriculum_calls: int = 0,
    ):
        self.seed = int(seed)
        self.costs = list(costs)
        #: Layer curriculum: routing opens from the LAST layer downward to ``start_layer`` over
        #: ``layer_curriculum_calls`` forwards (0 = every routed layer from the first call).
        #: Layers below :meth:`current_min_layer` run the full FFN and are excluded from the
        #: budget mean, exactly as layers below ``start_layer`` are.
        self.start_layer = int(start_layer)
        self.n_layers = int(n_layers)
        self.layer_curriculum_calls = int(layer_curriculum_calls)
        self.n_rungs = len(self.costs)
        self._choice_cache: Dict[tuple, torch.Tensor] = {}
        self.last_exp_per_layer: Dict[int, float] = {}
        self.last_exp_mean: Optional[float] = None
        self.target_cost = float(target_cost)
        self.budget_weight = float(budget_weight)
        self.hinge_power = int(hinge_power)
        self.two_sided = bool(two_sided)
        self.target_start = float(target_start)
        self.target_anneal_calls = int(target_anneal_calls)
        self.explore_prob = float(explore_prob)
        self.explore_anneal_calls = int(explore_anneal_calls)
        self.recon_frac = float(recon_frac)
        self.recon_weight = float(recon_weight)
        self.entropy_weight = float(entropy_weight)

        self.enabled = True
        self.calls = 0
        #: Set by the model each forward; when False (e.g. a decode step) routing still runs but
        #: no loss terms are accumulated.
        self.collect_loss = True
        #: Snapshot of the most recently COMPLETED forward (see :meth:`begin_forward`).
        self._last_cost_sum = 0.0
        self._last_tokens = 0
        self._last_usage = [0 for _ in self.costs]
        self._reset_accumulators()
        #: Cumulative hard-usage counts per rung, for logging across a whole run.
        self.usage_total = [0 for _ in self.costs]
        #: Cumulative hard-usage counts per rung PER LAYER (never reset): where the compute went.
        self.usage_by_layer: Dict[int, List[int]] = {}
        #: Same, for the last completed forward only.
        self._usage_layer: Dict[int, List[int]] = {}
        self._last_usage_layer: Dict[int, List[int]] = {}

    def _reset_accumulators(self) -> None:
        self._exp_costs: List[torch.Tensor] = []
        self._exp_layers: List[int] = []
        self._entropies: List[torch.Tensor] = []
        self._recons: List[torch.Tensor] = []
        self._hard_cost_sum = 0.0
        self._hard_tokens = 0
        self._usage = [0 for _ in self.costs]

    def begin_forward(self, *, collect_loss: bool = True) -> None:
        """Snapshot the finished forward's routing stats, then clear and advance the schedules.

        The snapshot is what :meth:`metrics` falls back on. A trainer callback reads metrics in
        ``post_step``, which is not guaranteed to land between the last routed forward and the
        next reset (activation-checkpoint recompute and multi-microbatch steps both interleave),
        so relying on the live accumulators alone silently reports nothing.
        """
        if self._hard_tokens > 0:
            self._last_cost_sum = self._hard_cost_sum
            self._last_tokens = self._hard_tokens
            self._last_usage = list(self._usage)
            self._last_usage_layer = {k: list(v) for k, v in self._usage_layer.items()}
        if self._exp_costs:  # lagged expectations for budget_attach (previous forward's values)
            self.last_exp_per_layer = {
                int(li): float(c.detach()) for li, c in zip(self._exp_layers, self._exp_costs)
            }
            self.last_exp_mean = sum(self.last_exp_per_layer.values()) / len(self.last_exp_per_layer)
        self._usage_layer = {}
        self._reset_accumulators()
        self.collect_loss = collect_loss
        self.calls += 1
        # Routing decisions of THIS forward, per layer, replayed by the activation-checkpoint
        # recompute (see _nested_forward): a fresh argmax in the recompute can flip a token when
        # the upstream kernels are not bit-reproducible, and torch then aborts with
        # "Recomputed values ... have different metadata" (27B, 2026-09-04).
        self._choice_cache = {}

    def set_calls(self, calls: int) -> None:
        """
        Pin the schedule clock to an externally tracked value (the trainer's global step times
        forwards-per-step).

        ``calls`` is in-memory only, so without this every crash-resume restarted the target and
        exploration anneals from scratch: a 3000-step run that resumed twice ended with
        ``target=0.84, explore=0.08`` instead of ``0.05, 0``, and the three routed 4B arms got
        three different schedules. :class:`~olmo_core.train.callbacks.NestedFFNMoECallback`
        calls this in ``pre_step``, which makes the anneals a pure function of the global step.
        """
        self.calls = int(calls)

    def per_layer_cost(self, *, last_forward: bool = False) -> Dict[int, float]:
        """Mean per-token FFN cost per routed layer (cumulative, or the last forward only)."""
        store = self._last_usage_layer if last_forward else self.usage_by_layer
        out: Dict[int, float] = {}
        for layer, usage in sorted(store.items()):
            total = sum(usage)
            if total:
                out[layer] = sum(c * u for c, u in zip(self.costs, usage)) / total
        return out

    def cumulative_metrics(self) -> Dict[str, float]:
        """
        Hard routing statistics accumulated over EVERY forward since construction (never reset).

        This is the number to quote for a deployed model: ``mean_cost`` from :meth:`metrics` is
        one training microbatch and includes whatever exploration noise was active, whereas this
        covers e.g. a whole eval (prefill and decode) under argmax routing.
        """
        total = sum(self.usage_total)
        out: Dict[str, float] = {"ffn_moe/total_tokens": float(total)}
        if total == 0:
            return out
        cost_sum = sum(c * u for c, u in zip(self.costs, self.usage_total))
        out["ffn_moe/mean_cost"] = cost_sum / total
        out["ffn_moe/speedup"] = total / cost_sum if cost_sum > 0 else float("inf")
        for i, u in enumerate(self.usage_total):
            out[f"ffn_moe/frac_rung{i}"] = u / total
        return out

    def current_min_layer(self) -> int:
        """
        The lowest layer index that routes on this call.

        With ``layer_curriculum_calls == 0`` this is ``start_layer``. Otherwise routing opens one
        layer at a time from the last layer down to ``start_layer``, linearly over the
        curriculum -- a token's early-layer FFN is the last thing the router is allowed to
        touch, after the late layers have already learned to live on the cheap rungs.
        """
        if self.layer_curriculum_calls <= 0 or self.n_layers <= 0:
            return self.start_layer
        frac = min(1.0, self.calls / self.layer_curriculum_calls)
        span = max(0, self.n_layers - 1 - self.start_layer)
        return self.start_layer + int(math.ceil((1.0 - frac) * span))

    def current_target(self) -> float:
        """The budget target for this call (linearly annealed from ``target_start``)."""
        if self.target_anneal_calls <= 0:
            return self.target_cost
        frac = min(1.0, self.calls / self.target_anneal_calls)
        return self.target_start + frac * (self.target_cost - self.target_start)

    def current_explore(self) -> float:
        """The exploration probability for this call (linearly annealed to 0)."""
        if self.explore_prob <= 0:
            return 0.0
        if self.explore_anneal_calls <= 0:
            return self.explore_prob
        frac = min(1.0, self.calls / self.explore_anneal_calls)
        return self.explore_prob * (1.0 - frac)

    def accumulate(
        self,
        *,
        exp_cost: Optional[torch.Tensor],
        entropy: Optional[torch.Tensor],
        hard_cost_sum: float,
        n_tokens: int,
        usage: Sequence[int],
        recon: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        in_budget: bool = True,
    ) -> None:
        """
        Record one gated layer's routing statistics (called from the shadowed forward).

        :param in_budget: ``False`` for a layer the curriculum has not opened yet: its (full-rung)
            usage is still counted for the cost report, but it contributes nothing to the budget
            mean and its loss terms are not recorded.
        """
        if in_budget and self.collect_loss and exp_cost is not None:
            self._exp_costs.append(exp_cost)
            self._exp_layers.append(int(layer_idx) if layer_idx is not None else -1)
            self._entropies.append(entropy)
            if recon is not None:
                self._recons.append(recon)
        if in_budget:
            self._hard_cost_sum += hard_cost_sum
            self._hard_tokens += n_tokens
            for i, c in enumerate(usage):
                self._usage[i] += int(c)
        for i, c in enumerate(usage):
            self.usage_total[i] += int(c)
        if layer_idx is not None:
            for store in (self.usage_by_layer, self._usage_layer):
                row = store.setdefault(int(layer_idx), [0 for _ in self.costs])
                for i, c in enumerate(usage):
                    row[i] += int(c)

    def regularization_loss(self) -> Optional[torch.Tensor]:
        """
        The budget (+ optional reconstruction and entropy) term for the current forward, or
        ``None`` if nothing was accumulated.

        The hinge is applied to the mean cost across gated layers, so the model may spend a full
        FFN on a few tokens as long as the average stays under target.
        """
        if not self._exp_costs:
            return None
        mean_cost = torch.stack(self._exp_costs).mean()
        gap = mean_cost - self.current_target()
        dev = gap.abs() if self.two_sided else torch.clamp(gap, min=0.0)
        loss = self.budget_weight * dev**self.hinge_power
        if self.entropy_weight > 0 and self._entropies:
            loss = loss - self.entropy_weight * torch.stack(self._entropies).mean()
        if self.recon_weight > 0 and self._recons:
            loss = loss + self.recon_weight * torch.stack(self._recons).mean()
        return loss

    def metrics(self) -> Dict[str, float]:
        """
        Hard (actually-executed) routing statistics for the current forward.

        Always returns the schedule values and ``ffn_moe/tokens``, even when no routed forward has
        been recorded. Returning an empty dict instead would make an inert router indistinguishable
        from a broken monitor -- which cost a live run's worth of debugging: the routing was
        working and only the reporting was silent.
        """
        out: Dict[str, float] = {
            "ffn_moe/target": self.current_target(),
            "ffn_moe/explore": self.current_explore(),
            "ffn_moe/tokens": float(self._hard_tokens),
            "ffn_moe/calls": float(self.calls),
            "ffn_moe/min_layer": float(self.current_min_layer()),
        }
        # Live accumulators when the caller lands mid-forward, else the last completed forward.
        if self._hard_tokens > 0:
            cost_sum, tokens, usage = self._hard_cost_sum, self._hard_tokens, self._usage
        elif self._last_tokens > 0:
            cost_sum, tokens, usage = self._last_cost_sum, self._last_tokens, self._last_usage
        else:
            return out
        out["ffn_moe/mean_cost"] = cost_sum / tokens
        out["ffn_moe/speedup"] = tokens / cost_sum if cost_sum > 0 else float("inf")
        total = max(1, sum(usage))
        for i, c in enumerate(usage):
            out[f"ffn_moe/frac_rung{i}"] = c / total
        return out


class _RouterLinear(nn.Linear):
    """``nn.Linear`` whose default reset is the router's one-hot "full rung" init (see
    :class:`NestedFFNRouter`); keeps the ``_nffn_router.w.{weight,bias}`` state-dict keys."""

    def reset_parameters(self) -> None:
        if self.weight.device.type == "meta":
            return
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()
            self.bias[0] = FULL_RUNG_INIT_BIAS


class NestedFFNRouter(nn.Module):
    """
    Per-layer linear router over the rungs.

    Initialized so that rung 0 (the full FFN) wins with probability ~1: weights are zeroed and the
    bias is a one-hot :data:`FULL_RUNG_INIT_BIAS`. An enabled-but-untrained model therefore
    reproduces its base exactly, which is what makes it safe to fit this onto an existing
    checkpoint.
    """

    def __init__(
        self,
        d_model: int,
        n_rungs: int,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.n_rungs = n_rungs
        # _RouterLinear (not a plain nn.Linear): ``Transformer.init_weights`` walks EVERY module
        # and calls its ``reset_parameters`` -- parent first, then children -- so a plain Linear
        # child re-ran torch's kaiming init right after this router's one-hot init. On every
        # meta-built (FSDP/Beaker) run the router therefore started RANDOM: tokens on random
        # rungs at step 1, Qwen3.5-4B CE 8.86 instead of 0.73 (FLOP-scaling grid, 2026-09-02).
        self.w = _RouterLinear(d_model, n_rungs, bias=True, dtype=dtype, device=init_device)
        if init_device != "meta":
            self.reset_parameters()

    def reset_parameters(self) -> None:
        self.w.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


def _ffn_weights(ff: nn.Module):
    """The FFN's effective ``(w1, w3, w2)`` weights.

    "Train what you route to" (``trainable_width`` > 0): the base FFN weights are FROZEN and the
    first ``k`` hidden units live in separate trainable parameters ``_nffnp_w1/_nffnp_w3`` (k x d)
    and ``_nffnp_w2`` (d x k); the effective weight is their concatenation with the frozen tail.
    The frozen rows carry no gradient and no optimizer state, which is what makes a 27B routed
    fine-tune fit one 80GB node. Without ``trainable_width`` the plain weights are returned.
    """
    k = getattr(ff, "_nffn_trainable_width", 0)
    w1, w2, w3 = ff.w1.weight, ff.w2.weight, ff.w3.weight  # type: ignore[union-attr]
    if not k:
        return w1, w3, w2
    return (
        torch.cat([ff._nffnp_w1, w1[k:]], dim=0),  # type: ignore[attr-defined]
        torch.cat([ff._nffnp_w3, w3[k:]], dim=0),  # type: ignore[attr-defined]
        torch.cat([ff._nffnp_w2, w2[:, k:]], dim=1),  # type: ignore[attr-defined]
    )


def _full_ffn(ff: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The un-routed gated MLP on the EFFECTIVE weights (identical to ``ff.forward`` unless a
    trainable prefix is installed, in which case ``ff.forward`` would read the frozen base)."""
    if not getattr(ff, "_nffn_trainable_width", 0):
        return ff._nffn_orig_forward(x)  # type: ignore[attr-defined]
    w1, w3, w2 = _ffn_weights(ff)
    h = ff.activation_fn(F.linear(x, w1, ff.w1.bias)) * F.linear(x, w3, ff.w3.bias)  # type: ignore[union-attr]
    return F.linear(h, w2, ff.w2.bias)  # type: ignore[union-attr]


def _slice_ffn(ff: nn.Module, x: torch.Tensor, width: int) -> torch.Tensor:
    """Run ``ff``'s gated MLP using only its first ``width`` hidden units."""
    w1, w3, w2 = _ffn_weights(ff)
    b1, b3, b2 = ff.w1.bias, ff.w3.bias, ff.w2.bias  # type: ignore[union-attr]
    h1 = F.linear(x, w1[:width], None if b1 is None else b1[:width])
    h3 = F.linear(x, w3[:width], None if b3 is None else b3[:width])
    h = ff.activation_fn(h1) * h3  # type: ignore[union-attr]
    return F.linear(h, w2[:, :width], b2)


#: Set to ``False`` to route through plain autograd over weight slices (the reference path).
#: The fused path is used whenever the FFN is a bias-free SiLU gated MLP.
USE_FUSED_LADDER = True


#: Row chunk for the ladder backward (see ``_NestedLadderFn.backward``): 8192 rows x 9728 width x
#: six bf16 intermediates ~ 1 GB transient instead of ~10 GB for a 65k row.
_BWD_CHUNK = 8192


class _NestedLadderFn(torch.autograd.Function):
    """
    Every rung of one nested FFN as ONE autograd node.

    The reference path (``_slice_ffn`` under autograd) is correct but slow in backward for a
    reason that has nothing to do with FLOPs: each ``w1.weight[:width]`` slice gets its own
    ``SliceBackward``, which allocates a FULL-size zero gradient for the weight, copies the
    narrow gradient into it, and autograd then ``add_``s four full-size (50 MB) gradients per
    weight matrix per layer. Profiled on an A100 (``debug/ffnmoe/profile_routed_layer.py``) that
    bookkeeping was 2x the GEMM time. Here each weight gradient is written once: the full rung
    assigns it and the narrow rungs ``+=`` into its prefix.

    The forward is also written to need ONE host sync per layer (the rung counts); the reference
    path had five (``nonzero`` per rung), each of which drains the GPU queue and turns the
    CPU-side launch cost of the many narrow kernels into wall-clock.

    Gains and the straight-through coefficient are applied OUTSIDE this function (a per-token
    scale), so it computes the un-gained ``out``.
    """

    @staticmethod
    def forward(ctx, widths, x, w1, w3, w2, *idxs):  # type: ignore[override]
        n_tokens = x.shape[0]
        out = torch.zeros_like(x)
        saved: List[Optional[torch.Tensor]] = []
        for width, idx in zip(widths, idxs):
            n = idx.shape[0]
            if width == 0 or n == 0:
                saved.extend([None, None])
                continue
            whole = n == n_tokens
            xs = x if whole else x.index_select(0, idx)
            h1 = F.linear(xs, w1[:width])
            h3 = F.linear(xs, w3[:width])
            ys = F.linear(F.silu(h1) * h3, w2[:, :width])
            if whole:
                out = ys
            else:
                out.index_copy_(0, idx, ys)
            saved.extend([h1, h3])
        ctx.widths = tuple(widths)
        ctx.n_saved = len(saved)
        ctx.save_for_backward(x, w1, w3, w2, *idxs, *[t for t in saved if t is not None])
        ctx.saved_mask = [t is not None for t in saved]
        return out

    @staticmethod
    def backward(ctx, dout):  # type: ignore[override]
        widths = ctx.widths
        n_rungs = len(widths)
        tensors = ctx.saved_tensors
        x, w1, w3, w2 = tensors[:4]
        idxs = tensors[4 : 4 + n_rungs]
        it = iter(tensors[4 + n_rungs :])
        saved = [next(it) if m else None for m in ctx.saved_mask]
        n_tokens = x.shape[0]
        full_width = widths[0]

        dx = torch.zeros_like(x)
        d1: Optional[torch.Tensor] = None
        d3: Optional[torch.Tensor] = None
        d2: Optional[torch.Tensor] = None
        # Full rung first (widths are descending) so its gradient is ASSIGNED and the narrow
        # rungs accumulate into a prefix of it -- no full-size zero fills, no full-size adds.
        for g, (width, idx) in enumerate(zip(widths, idxs)):
            h1, h3 = saved[2 * g], saved[2 * g + 1]
            if h1 is None:
                continue
            n = idx.shape[0]
            whole = n == n_tokens
            xs = x if whole else x.index_select(0, idx)
            dys = dout if whole else dout.index_select(0, idx)
            # Chunk over token rows: the un-chunked version materialised six (n, width) bf16
            # intermediates at once -- ~10 GB for a 65k row at width 9728 -- which is what pushed
            # the three-router Qwen3-4B runs over 80 GB (2026-09-05). Weight grads accumulate in
            # fp32 across chunks, the input grad is concatenated.
            dw1 = dw3 = dw2 = None
            dx_parts = []
            for s0 in range(0, n, _BWD_CHUNK):
                sl = slice(s0, min(n, s0 + _BWD_CHUNK))
                h1c, h3c, xc, dyc = h1[sl], h3[sl], xs[sl], dys[sl]
                a = F.silu(h1c)
                h = a * h3c
                dh = F.linear(dyc, w2[:, :width].t())  # (c, width)
                dw2c = dyc.t().mm(h)  # (D, width)
                sig = torch.sigmoid(h1c)
                dh1 = dh * h3c * (sig * (1 + h1c * (1 - sig)))
                dh3 = dh * a
                dw1c = dh1.t().mm(xc)  # (width, D)
                dw3c = dh3.t().mm(xc)
                dx_parts.append(dh1.mm(w1[:width]).add_(dh3.mm(w3[:width])))
                if dw1 is None:
                    dw1, dw3, dw2 = dw1c.float(), dw3c.float(), dw2c.float()
                else:
                    dw1.add_(dw1c.float()); dw3.add_(dw3c.float()); dw2.add_(dw2c.float())
                del a, h, dh, sig, dh1, dh3, dw1c, dw3c, dw2c
            dw1, dw3, dw2 = dw1.to(w1.dtype), dw3.to(w3.dtype), dw2.to(w2.dtype)
            dxs = dx_parts[0] if len(dx_parts) == 1 else torch.cat(dx_parts, 0)
            if whole:
                dx = dxs
            else:
                dx.index_copy_(0, idx, dxs)
            if width == full_width:
                d1, d3, d2 = dw1, dw3, dw2
            else:
                if d1 is None:
                    d1, d3, d2 = torch.zeros_like(w1), torch.zeros_like(w3), torch.zeros_like(w2)
                d1[:width] += dw1
                d3[:width] += dw3
                d2[:, :width] += dw2
        if d1 is None:  # every token was null: the weights got no signal this call
            d1, d3, d2 = torch.zeros_like(w1), torch.zeros_like(w3), torch.zeros_like(w2)
        return (None, dx, d1, d3, d2) + (None,) * n_rungs


def _fused_ladder_ok(ff: nn.Module) -> bool:
    return (
        USE_FUSED_LADDER
        and ff.activation_fn is F.silu  # type: ignore[union-attr]
        and ff.w1.bias is None  # type: ignore[union-attr]
        and ff.w2.bias is None  # type: ignore[union-attr]
        and ff.w3.bias is None  # type: ignore[union-attr]
    )


def _forward_generator(self: nn.Module, device: torch.device, stream: int = 0) -> torch.Generator:
    """
    A generator seeded deterministically from ``(holder.seed, holder.calls, layer index)``.

    Routing MUST be a pure function of the forward pass. Activation checkpointing re-runs the
    block in backward, and if the exploration draw came from the ambient RNG the recompute would
    put a different number of tokens on each rung -- torch then aborts with
    ``CheckpointError: Recomputed values ... have different metadata`` (observed: 5653 vs 5643
    tokens on one rung). ``holder.calls`` advances only in ``begin_forward``, which the recompute
    does not re-enter, so seeding off it makes forward and recompute agree exactly.
    """
    holder: NestedFFNHolder = self._nffn_holder  # type: ignore[attr-defined]
    gen: Optional[torch.Generator] = getattr(self, "_nffn_gen", None)
    if gen is None or gen.device != device:
        gen = torch.Generator(device=device)
        self._nffn_gen = gen  # type: ignore[assignment]
    seed = (holder.seed * 1_000_003 + holder.calls) * 1_000_003 + self._nffn_layer_idx  # type: ignore[attr-defined]
    gen.manual_seed(seed * 7 + stream)
    return gen


def _nested_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    holder: NestedFFNHolder = self._nffn_holder  # type: ignore[attr-defined]
    orig = lambda t: _full_ffn(self, t)  # noqa: E731  (effective weights, see _ffn_weights)
    if not holder.enabled:
        return orig(x)

    widths: List[int] = self._nffn_widths  # type: ignore[attr-defined]
    costs: List[float] = self._nffn_costs  # type: ignore[attr-defined]
    full_width = widths[0]
    layer_idx: int = self._nffn_layer_idx  # type: ignore[attr-defined]

    if layer_idx < holder.current_min_layer():
        # Layer curriculum: not opened yet -- dense FFN, counted as full-rung usage for the cost
        # report, outside the budget mean, router untouched.
        n_all = x.numel() // x.shape[-1]
        usage_full = [n_all] + [0] * (len(widths) - 1)
        holder.accumulate(
            exp_cost=None,
            entropy=None,
            hard_cost_sum=float(n_all),
            n_tokens=n_all,
            usage=usage_full,
            layer_idx=layer_idx,
            in_budget=False,
        )
        return orig(x)

    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    n_tokens = flat.shape[0]

    logits = self._nffn_router(flat)  # type: ignore[operator]
    probs = torch.softmax(logits.float(), dim=-1)
    # Routing must be a pure function of the forward: the activation-checkpoint recompute re-enters
    # this layer within the same ``holder.calls`` and MUST reproduce the same token->rung split
    # (every downstream index tensor's shape depends on it). Replay the cached decision.
    cache_key = (int(self._nffn_layer_idx), int(n_tokens))  # type: ignore[attr-defined]
    cache = getattr(holder, "_choice_cache", None)
    if self.training and cache is not None and cache_key in cache:
        choice = cache[cache_key]
    else:
        choice = probs.argmax(dim=-1)
        explore = holder.current_explore()
        if self.training and explore > 0:
            gen = _forward_generator(self, flat.device)
            rand_rung = torch.randint(0, len(widths), (n_tokens,), device=flat.device, generator=gen)
            take = torch.rand(n_tokens, device=flat.device, generator=gen) < explore
            choice = torch.where(take, rand_rung, choice)
        if self.training and cache is not None:
            cache[cache_key] = choice

    gain = self._nffn_gain  # type: ignore[attr-defined]
    # ONE host sync per layer: the rung counts. Tokens are then grouped by a stable argsort and
    # split on the host counts, which gives every rung a contiguous index tensor without further
    # syncs (the first version called ``nonzero`` per rung -- five GPU drains per layer).
    counts = torch.bincount(choice, minlength=len(widths)).tolist()
    usage: List[int] = [int(c) for c in counts]
    hard_cost_sum = float(sum(costs[g] * usage[g] for g in range(len(widths))))
    order = torch.argsort(choice, stable=True)
    idxs = torch.split(order, usage)

    if _fused_ladder_ok(self):
        w1_eff, w3_eff, w2_eff = _ffn_weights(self)
        out = _NestedLadderFn.apply(tuple(widths), flat, w1_eff, w3_eff, w2_eff, *idxs)
    else:
        # Reference path: plain autograd over weight slices. Correct, but see _NestedLadderFn for
        # why its backward is slow.
        out = torch.zeros_like(flat)
        for g, width in enumerate(widths):
            if width == 0 or usage[g] == 0:
                continue
            if usage[g] == n_tokens:
                out = orig(flat) if width == full_width else _slice_ffn(self, flat, width)
                continue
            idx = idxs[g]
            xs = flat.index_select(0, idx)
            ys = orig(xs) if width == full_width else _slice_ffn(self, xs, width)
            out.index_copy_(0, idx, ys.to(out.dtype))

    # Optional local reconstruction target: teach the small rungs to reproduce the full FFN on a
    # small random subset of tokens. FFN-only -- no attention, no full-context forward.
    recon: Optional[torch.Tensor] = None
    if self.training and holder.collect_loss and holder.recon_frac > 0 and holder.recon_weight > 0:
        k = max(1, int(n_tokens * holder.recon_frac))
        # Same determinism requirement as the exploration draw above.
        sub = torch.randperm(
            n_tokens, device=flat.device, generator=_forward_generator(self, flat.device, stream=1)
        )[:k]
        with torch.no_grad():
            target = orig(flat.index_select(0, sub)).float()
        got = out.index_select(0, sub).float()
        denom = target.pow(2).sum(-1).mean().clamp(min=1e-6)
        recon = (got - target).pow(2).sum(-1).mean() / denom

    # Straight-through router coefficient: exactly 1.0 in the forward (so the base model's
    # behaviour is untouched at init) while d loss / d p_chosen flows to the router logits.
    #
    # NOTE: the obvious form ``p / p.detach()`` also has value 1 but gradient ``1/p``, which
    # explodes when a token is routed to a rung the router gives ~0 probability -- exactly what
    # exploration does on purpose. That produced NaN CE within ~75 steps in the CPU smoke test
    # (debug/ffnmoe/smoke_router_learns.py). The additive form has constant gradient 1 and is
    # numerically safe for any p.
    p_sel = probs.gather(1, choice[:, None]).squeeze(1)
    coef = 1.0 + p_sel - p_sel.detach()
    # Per-rung gain and the ST coefficient are one per-token scale (a single pass over ``out``).
    # The gain is gathered with a one-hot matmul rather than ``gain[choice]``: advanced indexing
    # backpropagates through a sort-based ``index_put`` kernel that cost 0.7 ms per call (35 ms
    # per training step) in the 2-GPU profile; the one-hot form's backward is a tiny GEMM.
    one_hot = F.one_hot(choice, num_classes=len(widths)).to(probs.dtype)
    scale = (one_hot @ gain.float()) * coef
    out = out * scale.to(out.dtype)[:, None]

    cost_vec = torch.tensor(costs, device=flat.device, dtype=probs.dtype)
    exp_cost = (probs * cost_vec).sum(-1).mean()
    entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(-1).mean()
    holder.accumulate(
        exp_cost=exp_cost,
        entropy=entropy,
        hard_cost_sum=hard_cost_sum,
        n_tokens=n_tokens,
        usage=usage,
        recon=recon,
        layer_idx=layer_idx,
    )
    return out.reshape(shape)


def install_nested_ffn_moe(
    blocks: nn.ModuleDict,
    holder: NestedFFNHolder,
    *,
    start_layer: int,
    widths: Sequence[int],
    costs: Sequence[float],
    init_device: str = "cpu",
    trainable_width: int = 0,
) -> List[str]:
    """
    Attach a router + per-rung gains to every block at or after ``start_layer`` and shadow its
    ``feed_forward.forward`` with the nested-rung version.

    ``trainable_width`` > 0 = "train what you route to": the routed FFN's base weights are frozen
    and only the first ``trainable_width`` hidden units train, as separate parameters
    ``_nffnp_w1/_nffnp_w3/_nffnp_w2`` (see :func:`_ffn_weights`). New state-dict keys; the frozen
    base keys are unchanged, so a base checkpoint still loads.

    The FFN's own weights and state-dict keys are untouched; the only new keys are
    ``<block>.feed_forward.{_nffn_router.w.weight,_nffn_router.w.bias,_nffn_gain}``.

    :param blocks: The model's block dict (keys are layer indices as strings).
    :param holder: Shared router state.
    :param start_layer: First layer to route (earlier layers keep the full FFN for every token --
        early layers build the token identity that later layers redistribute, and both role-gate
        experiments showed low-layer FFN removal is what breaks trainability).
    :param widths: Nested hidden widths, descending, ``widths[0]`` = full.
    :param costs: Matching relative costs.

    :returns: The block keys routed.

    :raises ValueError: If ``widths`` is not descending or exceeds a block's hidden size.
    """
    if list(widths) != sorted(widths, reverse=True):
        raise ValueError(f"widths must be descending, got {list(widths)}")
    routed = []
    for key, block in blocks.items():
        if int(key) < start_layer:
            continue
        ff = getattr(block, "feed_forward", None)
        if ff is None or hasattr(ff, "_nffn_orig_forward"):
            continue
        if not all(hasattr(ff, a) for a in ("w1", "w2", "w3", "activation_fn")):
            raise ValueError(f"block {key}: feed_forward is not a gated MLP, cannot nest it")
        if widths[0] != ff.w1.out_features:
            raise ValueError(
                f"block {key}: full rung {widths[0]} != hidden size {ff.w1.out_features}"
            )
        dtype, device = ff.w1.weight.dtype, ff.w1.weight.device
        ff._nffn_router = NestedFFNRouter(
            ff.w1.in_features,
            len(widths),
            dtype=dtype,
            init_device="meta" if device.type == "meta" else init_device,
        ).to(device)
        ff._nffn_gain = nn.Parameter(
            torch.ones(len(widths), dtype=dtype, device=device)
            if device.type != "meta"
            else torch.empty(len(widths), dtype=dtype, device=device)
        )
        ff._nffn_holder = holder
        ff._nffn_widths = list(widths)
        ff._nffn_costs = list(costs)
        ff._nffn_layer_idx = int(key)
        ff._nffn_trainable_width = int(trainable_width or 0)
        if trainable_width:
            k = int(trainable_width)
            if not (0 < k < ff.w1.out_features):
                raise ValueError(f"block {key}: trainable_width {k} must be in (0, {ff.w1.out_features})")
            for lin in (ff.w1, ff.w2, ff.w3):
                lin.weight.requires_grad_(False)
                if lin.bias is not None:
                    lin.bias.requires_grad_(False)
            ff._nffnp_w1 = nn.Parameter(ff.w1.weight.detach()[:k].clone())
            ff._nffnp_w3 = nn.Parameter(ff.w3.weight.detach()[:k].clone())
            ff._nffnp_w2 = nn.Parameter(ff.w2.weight.detach()[:, :k].clone())
        ff._nffn_orig_forward = ff.forward
        ff.forward = MethodType(_nested_forward, ff)
        # Built on ``meta`` (every FSDP/Beaker run), the gains above are an EMPTY tensor and the
        # router skipped its one-hot init; ``Transformer.init_weights`` only re-initializes
        # modules exposing ``reset_parameters``. Without this hook a routed layer started with
        # whatever ``to_empty`` left in memory: Qwen3.5-4B went from CE 0.73 to 8.86 at step 1
        # on every FFN arm of the FLOP-scaling grid (2026-09-02) before anything was learned.
        ff._nffn_orig_reset = getattr(ff, "reset_parameters", None)
        ff.reset_parameters = MethodType(_nested_reset_parameters, ff)
        routed.append(key)
    return routed


def _nested_reset_parameters(self: nn.Module) -> None:
    """Deterministic init of the nested-FFN extras: gains to 1, router to the full rung."""
    if self._nffn_orig_reset is not None:  # type: ignore[attr-defined]
        self._nffn_orig_reset()  # type: ignore[attr-defined]
    reset_nested_ffn_extras(self)


def reset_nested_ffn_extras(ff: nn.Module, parts: Optional[Sequence[str]] = None) -> None:
    """Deterministic init of the nested-FFN extras: router to the full rung, gains to 1, and the
    trainable prefix copied from the (already loaded) frozen base weights, so the effective FFN
    equals the base exactly. ``parts`` restricts this to a subset of {"router", "gain", "prefix"}
    (a stage-2 warm start loads its router and must not have it reset)."""
    parts = set(parts) if parts is not None else {"router", "gain", "prefix"}
    with torch.no_grad():
        if "gain" in parts:
            ff._nffn_gain.fill_(1.0)  # type: ignore[attr-defined]
        if "router" in parts:
            ff._nffn_router.reset_parameters()  # type: ignore[attr-defined]
        if "prefix" in parts and getattr(ff, "_nffn_trainable_width", 0):
            k = ff._nffn_trainable_width  # type: ignore[attr-defined]
            for name, src, sl in (("_nffnp_w1", ff.w1.weight, (slice(None, k),)),  # type: ignore[union-attr]
                                  ("_nffnp_w3", ff.w3.weight, (slice(None, k),)),  # type: ignore[union-attr]
                                  ("_nffnp_w2", ff.w2.weight, (slice(None), slice(None, k)))):  # type: ignore[union-attr]
                dst = getattr(ff, name)
                full = src.full_tensor() if hasattr(src, "full_tensor") else src
                val = full[sl]
                if hasattr(dst, "device_mesh"):  # FSDP DTensor: re-shard the prefix like the param
                    from torch.distributed.tensor import distribute_tensor

                    val = distribute_tensor(val.to(dst.dtype), dst.device_mesh, dst.placements)
                dst.copy_(val)


@torch.no_grad()
def ffn_importance_permutation(
    ff: nn.Module, act_stats: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Rank an FFN's hidden units by importance, most important first.

    Nesting takes the FIRST ``width`` hidden units, so which units land in the prefix decides
    whether a 1/64 rung is a useful approximation of the full FFN or an arbitrary 1/64 of it.
    Reordering the hidden units is an **exactly output-preserving reparameterization** (permute
    ``w1``/``w3`` rows and ``w2`` columns together), so this can be applied to a trained
    checkpoint for free -- see :func:`apply_ffn_permutation`.

    :param ff: The feed-forward module.
    :param act_stats: Optional per-unit mean absolute activation of ``act(w1 x) * (w3 x)``,
        measured on real data, shape ``(hidden_size,)``. When given, importance is
        ``act_stats * ||w2[:, j]||`` -- the unit's actual mean contribution norm. When ``None`` a
        data-free proxy ``||w1[j]|| * ||w3[j]|| * ||w2[:, j]||`` is used.

    :returns: A permutation index of shape ``(hidden_size,)``.
    """
    w2_norm = ff.w2.weight.float().norm(dim=0)  # type: ignore[union-attr]
    if act_stats is not None:
        score = act_stats.float().to(w2_norm.device) * w2_norm
    else:
        w1_norm = ff.w1.weight.float().norm(dim=1)  # type: ignore[union-attr]
        w3_norm = ff.w3.weight.float().norm(dim=1)  # type: ignore[union-attr]
        score = w1_norm * w3_norm * w2_norm
    return torch.argsort(score, descending=True)


@torch.no_grad()
def apply_ffn_permutation(ff: nn.Module, perm: torch.Tensor) -> None:
    """
    Permute an FFN's hidden units in place. The module's output is unchanged (bit-identical up to
    floating-point summation order), because the hidden dimension is summed over in ``w2``.

    :param ff: The feed-forward module.
    :param perm: Permutation index from :func:`ffn_importance_permutation`.
    """
    ff.w1.weight.copy_(ff.w1.weight[perm])  # type: ignore[union-attr]
    ff.w3.weight.copy_(ff.w3.weight[perm])  # type: ignore[union-attr]
    ff.w2.weight.copy_(ff.w2.weight[:, perm])  # type: ignore[union-attr]
    if ff.w1.bias is not None:  # type: ignore[union-attr]
        ff.w1.bias.copy_(ff.w1.bias[perm])  # type: ignore[union-attr]
    if ff.w3.bias is not None:  # type: ignore[union-attr]
        ff.w3.bias.copy_(ff.w3.bias[perm])  # type: ignore[union-attr]


def flop_summary(widths: Sequence[int], costs: Sequence[float], usage: Sequence[int]) -> str:
    """A one-line human-readable summary of where the FFN FLOPs went."""
    total = max(1, sum(usage))
    parts = [f"w{w}({c:.4f}): {u / total:.1%}" for w, c, u in zip(widths, costs, usage)]
    mean = sum(c * u for c, u in zip(costs, usage)) / total
    return f"mean_cost={mean:.4f} ({1 / mean if mean else math.inf:.1f}x) | " + " ".join(parts)
