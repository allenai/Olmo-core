"""
The optimizer side of the CompletedP parameterization: per-parameter AdamW learning rates,
epsilons and weight decays, plus batch-size and training-duration scaling.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch.nn as nn

from ..config import Config
from ..exceptions import OLMoConfigurationError
from ..nn.transformer.completedp import (
    CompletedPRole,
    find_completedp_model,
    named_completedp_params,
)
from .mup import split_groups_by_multipliers

if TYPE_CHECKING:
    from .config import OptimGroupOverride

__all__ = [
    "CompletedPOptimConfig",
    "completedp_multipliers",
    "completedp_betas",
    "apply_completedp",
]


@dataclass
class CompletedPOptimConfig(Config):
    """
    Per-parameter AdamW hyperparameters following the CompletedP parameterization of
    `Mlodozeniec et al., 2025 <https://arxiv.org/abs/2512.22382>`_ (Table 1). Use it with a model
    whose :class:`~olmo_core.nn.transformer.completedp.CompletedPConfig` is set; the width and
    depth settings are read from the model.

    The optimizer's ``lr``, ``eps``, ``weight_decay`` and ``betas`` (or a group override's) are the
    base values, tuned at the base width, depth, batch size and training duration. With
    ``r = fan_in / base_width`` for a matrix, ``m_N = d_model / base_width``,
    ``m_L = n_layers / base_depth`` and ``s = sqrt(m_B / m_D)``, where ``m_B`` and ``m_D`` are the
    batch-size and training-token ratios, each parameter group gets:

    - hidden weight matrices: ``lr * m_L**(alpha - 1) * s / r``,
      ``eps * m_L**-alpha / (m_N * s)``, ``weight_decay * r * s``; in-block matrices with a fixed
      output size (MoE routers, KDA ``f_proj_1`` / ``g_proj_1``) use the output-weight epsilon,
      ``eps * m_L**-alpha / s``;
    - hidden vectors (norm gains, biases, ...): ``lr * m_L**(alpha - 1) * s``,
      ``eps * m_L**-alpha / (m_N * s)``, ``weight_decay * s``; vectors shared across attention
      heads (e.g. head-wise QK-norm gains) use ``eps * m_L**-alpha / s``;
    - the input embedding and embedding norm: ``lr * s``, ``eps / (m_N * s)``,
      ``weight_decay * s``;
    - the LM head weight: ``lr * s / r``, ``eps / s``, ``weight_decay * r * s``; its norm and bias:
      ``lr * s``, ``eps / s``, ``weight_decay * s``;
    - every group: ``1 - beta`` scaled by ``m_B / m_D``.

    Weight decay follows the PyTorch convention (decay ``lr * weight_decay`` per step).
    """

    base_batch_size: Optional[int] = None
    """
    The batch size, in tokens per optimizer step, that the base hyperparameters were tuned with.
    """

    batch_size: Optional[int] = None
    """
    The batch size of this run, in tokens per optimizer step.
    """

    base_train_tokens: Optional[int] = None
    """
    The number of training tokens that the base hyperparameters were tuned with.
    """

    train_tokens: Optional[int] = None
    """
    The number of training tokens of this run.
    """

    def __post_init__(self):
        for base, value, name in (
            (self.base_batch_size, self.batch_size, "batch_size"),
            (self.base_train_tokens, self.train_tokens, "train_tokens"),
        ):
            if (base is None) != (value is None):
                raise OLMoConfigurationError(
                    f"Set both 'base_{name}' and '{name}' or neither, got {base} and {value}"
                )
            if base is not None and value is not None and (base <= 0 or value <= 0):
                raise OLMoConfigurationError(
                    f"'base_{name}' and '{name}' must be positive, got {base} and {value}"
                )

    @property
    def batch_size_ratio(self) -> float:
        """
        The batch-size ratio ``m_B``, 1 if unset.
        """
        if self.base_batch_size is None or self.batch_size is None:
            return 1.0
        return self.batch_size / self.base_batch_size

    @property
    def duration_ratio(self) -> float:
        """
        The training-token ratio ``m_D``, 1 if unset.
        """
        if self.base_train_tokens is None or self.train_tokens is None:
            return 1.0
        return self.train_tokens / self.base_train_tokens


def completedp_multipliers(
    model: nn.Module, config: CompletedPOptimConfig
) -> Dict[str, Dict[str, float]]:
    """
    Compute the CompletedP ``lr``, ``eps`` and ``weight_decay`` multipliers of every parameter.

    :param model: The model, or a wrapper around it. Names match ``model.named_parameters()``.
    :param config: The optimizer-side CompletedP config.

    :returns: A mapping from parameter name to ``{"lr": ..., "eps": ..., "weight_decay": ...}``.

    :raises OLMoConfigurationError: If the model does not use CompletedP.
    """
    transformer = find_completedp_model(model)
    if transformer is None or transformer.completedp is None:
        raise OLMoConfigurationError(
            "CompletedP optimizer settings require a model built with "
            "TransformerConfig.completedp set"
        )
    model_config = transformer.completedp
    alpha = model_config.alpha
    m_n = model_config.width_ratio(transformer.d_model)
    m_l = model_config.depth_ratio(transformer.n_layers)
    s = (config.batch_size_ratio / config.duration_ratio) ** 0.5

    hidden_lr = m_l ** (alpha - 1)
    hidden_eps = m_l**-alpha / m_n
    multipliers: Dict[str, Dict[str, float]] = {}
    for name, param in named_completedp_params(model).items():
        r = 1.0
        if param.fan_in is not None:
            r = model_config.width_ratio(param.fan_in)

        if param.role in (CompletedPRole.input_embedding, CompletedPRole.input_vector):
            lr, eps, weight_decay = 1.0, 1 / m_n, 1.0
        elif param.role == CompletedPRole.hidden_matrix:
            lr, eps, weight_decay = hidden_lr / r, hidden_eps, r
        elif param.role == CompletedPRole.hidden_readout_matrix:
            lr, eps, weight_decay = hidden_lr / r, m_l**-alpha, r
        elif param.role == CompletedPRole.hidden_vector:
            lr, eps, weight_decay = hidden_lr, hidden_eps, 1.0
        elif param.role == CompletedPRole.hidden_head_shared_vector:
            lr, eps, weight_decay = hidden_lr, m_l**-alpha, 1.0
        elif param.role == CompletedPRole.output_matrix:
            lr, eps, weight_decay = 1 / r, 1.0, r
        elif param.role == CompletedPRole.output_vector:
            lr, eps, weight_decay = 1.0, 1.0, 1.0
        else:
            raise NotImplementedError(param.role)
        multipliers[name] = {"lr": lr * s, "eps": eps / s, "weight_decay": weight_decay * s}
    return multipliers


def completedp_betas(betas: Sequence[float], config: CompletedPOptimConfig) -> Tuple[float, float]:
    """
    Scale ``1 - beta`` of each AdamW beta by the batch-size ratio over the training-token ratio.

    :param betas: The base ``(beta1, beta2)``.
    :param config: The optimizer-side CompletedP config.

    :returns: The scaled betas.

    :raises OLMoConfigurationError: If a scaled beta falls outside ``[0, 1)``.
    """
    beta1, beta2 = betas
    ratio = config.batch_size_ratio / config.duration_ratio
    if ratio == 1.0:
        return (beta1, beta2)
    scaled = (1 - (1 - beta1) * ratio, 1 - (1 - beta2) * ratio)
    for base, beta in zip((beta1, beta2), scaled):
        if not 0.0 <= beta < 1.0:
            raise OLMoConfigurationError(
                f"CompletedP scales beta {base} to {beta}, which is outside [0, 1); the batch-size "
                f"to training-token ratio ({ratio}) is too large for it"
            )
    return scaled


def apply_completedp(
    groups: List["OptimGroupOverride"],
    model_parts: Sequence[nn.Module],
    config: Optional[CompletedPOptimConfig],
    *,
    optimizer_name: str,
    supported: bool,
    mup_enabled: bool,
    lr: Optional[float] = None,
    eps: Optional[float] = None,
    weight_decay: Optional[float] = None,
    betas: Optional[Sequence[float]] = None,
) -> List["OptimGroupOverride"]:
    """
    Split parameter groups by their CompletedP multipliers, then set each group's ``lr``, ``eps``,
    ``weight_decay`` and ``betas``. Also checks that the model and optimizer agree on whether
    CompletedP is used.

    :param groups: The groups, with parameter names already expanded.
    :param model_parts: The model parts being optimized.
    :param config: The optimizer's CompletedP config, if set.
    :param optimizer_name: The optimizer config's class name, used in error messages.
    :param supported: Whether the optimizer supports CompletedP.
    :param mup_enabled: Whether muP learning rates are also enabled.
    :param lr: The optimizer's learning rate.
    :param eps: The optimizer's AdamW epsilon.
    :param weight_decay: The optimizer's weight decay.
    :param betas: The optimizer's betas.

    :returns: The split groups (or the input groups if CompletedP is not used).

    :raises OLMoConfigurationError: If only one of the model and optimizer uses CompletedP, if the
        optimizer does not support it, or if muP is also enabled.
    """
    if config is None:
        if any(find_completedp_model(part) is not None for part in model_parts):
            raise OLMoConfigurationError(
                f"The model uses CompletedP, so {optimizer_name} needs 'completedp' set"
            )
        return groups
    if not supported:
        raise OLMoConfigurationError(f"CompletedP is not supported for {optimizer_name}")
    if mup_enabled:
        raise OLMoConfigurationError("'completedp' and 'mup' can't both be set")
    if betas is None:
        raise OLMoConfigurationError(f"CompletedP requires {optimizer_name} to have 'betas'")

    multipliers: Dict[str, Dict[str, float]] = {}
    for part in model_parts:
        multipliers.update(completedp_multipliers(part, config))
    split = split_groups_by_multipliers(
        groups,
        multipliers,
        {"lr": lr, "eps": eps, "weight_decay": weight_decay},
        parameterization="CompletedP",
    )
    for group in split:
        group.opts["betas"] = completedp_betas(group.opts.get("betas", betas), config)
    return split
