"""
muP (maximal update parameterization) learning rates for Adam-type optimizers.
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch.nn as nn

from ..config import Config
from ..exceptions import OLMoConfigurationError
from ..nn.transformer.init import InitMethod
from ..nn.transformer.mup import ParamKind, named_param_shapes
from ..utils import log_once

if TYPE_CHECKING:
    from .config import OptimGroupOverride

__all__ = [
    "MuPConfig",
    "mup_lr_multipliers",
    "split_groups_by_lr_multiplier",
    "split_groups_by_multipliers",
]

log = logging.getLogger(__name__)

LR_MULTIPLIER_FIELD = "lr_multiplier"


@dataclass
class MuPConfig(Config):
    """
    Per-parameter learning rates following muP for Adam-type optimizers, from the spectral
    condition for feature learning (`Yang et al., 2023 <https://arxiv.org/abs/2310.17813>`_).
    Use it together with :data:`~olmo_core.nn.transformer.init.InitMethod.nanochat`, which is the
    matching initialization.

    A weight matrix with fan-in ``fan_in`` gets the learning rate ``lr * base_width / fan_in``,
    embeddings get ``lr * embedding_lr_multiplier``, and element-wise parameters (norm gains,
    biases, per-head scalars, depthwise convolutions) get ``lr``. Here ``lr`` is the optimizer's
    learning rate, or a group override's ``lr`` if one is set.

    Keep ``base_width`` fixed across the models being compared: that is what lets a learning rate
    tuned at one width transfer to other widths.
    """

    base_width: int
    """
    The fan-in at which a weight matrix gets exactly the base learning rate.
    """

    embedding_lr_multiplier: float = 1.0
    """
    A width-independent multiplier for the embedding learning rate.
    """

    scale_weight_decay: bool = False
    """
    If ``True``, divide each group's weight decay by its learning-rate multiplier, so that
    ``lr * weight_decay`` stays at its base value for every parameter. Otherwise weight decay is
    applied as ``lr * weight_decay`` with the scaled learning rate.
    """

    def __post_init__(self):
        if self.base_width <= 0:
            raise OLMoConfigurationError(f"base_width must be positive, got {self.base_width}")


def mup_lr_multipliers(model: nn.Module, config: MuPConfig) -> Dict[str, float]:
    """
    Compute the muP learning-rate multiplier of every parameter of a model.

    :param model: The model, or a wrapper around it. Names match ``model.named_parameters()``.
    :param config: The muP config.

    :returns: A mapping from parameter name to learning-rate multiplier.
    """
    init_method = next(
        (
            m.init_method
            for m in model.modules()
            if isinstance(getattr(m, "init_method", None), InitMethod)
        ),
        None,
    )
    if init_method is not None and init_method != InitMethod.nanochat:
        log_once(
            log,
            f"muP learning rates are enabled but the model uses init method '{init_method}'; "
            f"muP expects '{InitMethod.nanochat}'",
            level=logging.WARNING,
        )

    multipliers: Dict[str, float] = {}
    for name, shape in named_param_shapes(model).items():
        if shape.kind == ParamKind.matrix:
            assert shape.fan_in is not None
            multipliers[name] = config.base_width / shape.fan_in
        elif shape.kind == ParamKind.embedding:
            multipliers[name] = config.embedding_lr_multiplier
        else:
            multipliers[name] = 1.0
    return multipliers


def split_groups_by_multipliers(
    groups: List["OptimGroupOverride"],
    multipliers: Dict[str, Dict[str, float]],
    base_values: Dict[str, Optional[float]],
    *,
    parameterization: str,
) -> List["OptimGroupOverride"]:
    """
    Split parameter groups so that every parameter in a group shares the same multipliers, and
    set each multiplied option of a group to its base value times the multiplier. The base value is
    the group's own option if it sets one, otherwise the optimizer's. Each group records each
    multiplier under ``"<option>_multiplier"`` (e.g. ``"lr_multiplier"``); other group options are
    kept.

    :param groups: The groups, with parameter names already expanded.
    :param multipliers: For every parameter, a mapping from option name to multiplier. Every
        parameter must have the same option names, in the same order.
    :param base_values: The optimizer's value of each option.
    :param parameterization: The name of the parameterization, used in error messages.

    :returns: The split groups, in order. Sub-groups of a group are in order of first appearance.

    :raises OLMoConfigurationError: If a parameter has no multipliers, or an option to scale has no
        base value.
    """
    split: List["OptimGroupOverride"] = []
    for group in groups:
        names_by_multipliers: Dict[Tuple[Tuple[str, float], ...], List[str]] = {}
        for name in group.params:
            if name not in multipliers:
                raise OLMoConfigurationError(
                    f"{parameterization} has no multiplier for optimizer parameter '{name}'"
                )
            names_by_multipliers.setdefault(tuple(multipliers[name].items()), []).append(name)

        for key, names in names_by_multipliers.items():
            opts = dict(group.opts)
            for option, multiplier in key:
                base_value = group.opts.get(option, base_values.get(option))
                if base_value is None:
                    raise OLMoConfigurationError(
                        f"{parameterization} needs a value for '{option}' to scale"
                    )
                opts[option] = base_value * multiplier
                opts[f"{option}_multiplier"] = multiplier
            split.append(dataclasses.replace(group, params=names, opts=opts))
    return split


def split_groups_by_lr_multiplier(
    groups: List["OptimGroupOverride"],
    multipliers: Dict[str, float],
    config: MuPConfig,
    *,
    lr: float,
    weight_decay: Optional[float] = None,
) -> List["OptimGroupOverride"]:
    """
    Split parameter groups so that every parameter in a group shares one muP learning-rate
    multiplier, and set each group's ``lr`` (and ``weight_decay``, with
    :data:`MuPConfig.scale_weight_decay`) accordingly. Other group options are kept. Each group
    also records its multiplier under ``"lr_multiplier"``.

    :param groups: The groups, with parameter names already expanded.
    :param multipliers: The multiplier of every parameter, from :func:`mup_lr_multipliers`.
    :param config: The muP config.
    :param lr: The optimizer's learning rate, used for groups that don't override it.
    :param weight_decay: The optimizer's weight decay, used for groups that don't override it.

    :returns: The split groups, in order.

    :raises OLMoConfigurationError: If a parameter has no multiplier, or if weight decay must be
        scaled but is unknown.
    """
    split = split_groups_by_multipliers(
        groups,
        {name: {"lr": multiplier} for name, multiplier in multipliers.items()},
        {"lr": lr},
        parameterization="muP",
    )
    if config.scale_weight_decay:
        for group in split:
            group_weight_decay = group.opts.get("weight_decay", weight_decay)
            if group_weight_decay is None:
                raise OLMoConfigurationError(
                    "MuPConfig.scale_weight_decay requires a weight decay to scale"
                )
            group.opts["weight_decay"] = group_weight_decay / group.opts[LR_MULTIPLIER_FIELD]
    return split
