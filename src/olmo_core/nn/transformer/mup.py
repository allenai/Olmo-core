"""
Per-parameter shapes used by the muP learning-rate rules.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch.nn as nn

from olmo_core.config import StrEnum

__all__ = ["ParamKind", "ParamShape", "named_param_shapes"]


class ParamKind(StrEnum):
    """
    How muP treats a parameter.
    """

    matrix = "matrix"
    """
    A weight matrix whose fan-in grows with width, such as a linear layer, an expert weight, or a
    router.
    """

    embedding = "embedding"
    """
    An embedding table. Its one-hot input has a fixed size, so it does not scale with width.
    """

    vector = "vector"
    """
    An element-wise parameter: norm gains, biases, per-head or per-channel scalars, and depthwise
    convolution taps.
    """


@dataclass(frozen=True)
class ParamShape:
    """
    The muP role of a parameter and, for matrices, its logical fan-in and fan-out.
    """

    kind: ParamKind
    fan_in: Optional[int] = None
    fan_out: Optional[int] = None


def _module_param_shapes(module: nn.Module) -> Dict[str, ParamShape]:
    from ..layer_norm import LayerNorm
    from ..moe import MoERouter
    from ..moe.mlp import MoEMLPBase
    from ..moe.v2.routed_experts import RoutedExperts
    from ..moe.v2.router import MoERouterV2
    from ..moe.v2.shared_experts import SharedExperts

    matrix = ParamKind.matrix
    if isinstance(module, nn.Embedding):
        return {
            "weight": ParamShape(
                ParamKind.embedding, fan_in=module.num_embeddings, fan_out=module.embedding_dim
            )
        }
    if isinstance(module, nn.Linear):
        return {"weight": ParamShape(matrix, module.in_features, module.out_features)}
    if isinstance(module, nn.Conv1d):
        if module.groups == module.in_channels == module.out_channels:
            return {"weight": ParamShape(ParamKind.vector)}
        return {}
    if isinstance(module, LayerNorm):
        # Includes per-head gains, which are 2-D.
        return {
            name: ParamShape(ParamKind.vector) for name, _ in module.named_parameters(recurse=False)
        }
    # The fused ``w_up_gate`` holds a gate and an up projection, each with ``hidden_size`` outputs.
    if isinstance(module, (RoutedExperts, SharedExperts)):
        return {
            "w_up_gate": ParamShape(matrix, module.d_model, module.hidden_size),
            "w_down": ParamShape(matrix, module.hidden_size, module.d_model),
            "b_up_gate": ParamShape(ParamKind.vector),
            "b_down": ParamShape(ParamKind.vector),
        }
    # Router weights are stored flattened to 1-D but act as a ``d_model -> num_experts`` matrix.
    if isinstance(module, (MoERouterV2, MoERouter)):
        return {"weight": ParamShape(matrix, module.d_model, module.num_experts)}
    if isinstance(module, MoEMLPBase):
        return {
            "w1": ParamShape(matrix, module.d_model, module.hidden_size),
            "w2": ParamShape(matrix, module.hidden_size, module.d_model),
            "w3": ParamShape(matrix, module.d_model, module.hidden_size),
        }
    return {}


def named_param_shapes(model: nn.Module) -> Dict[str, ParamShape]:
    """
    Classify every parameter of a model for muP, reading fan-in and fan-out from module
    attributes rather than tensor shapes (expert weights are 3-D and router weights are
    flattened).

    Names match ``model.named_parameters()``, including any wrapper prefix such as ``module.``,
    so they can be looked up directly from optimizer parameter names. A parameter shared between
    modules (e.g. tied embeddings) gets the shape of the first module that owns it, as in
    ``named_parameters()``.

    :param model: The model, or a wrapper around it.

    :returns: A mapping from parameter name to :class:`ParamShape`.

    :raises NotImplementedError: If a parameter with two or more dimensions is not recognized,
        so that no parameter silently gets the wrong learning rate.
    """
    shapes: Dict[str, ParamShape] = {}
    seen: set = set()
    for module_name, module in model.named_modules():
        known = _module_param_shapes(module)
        for name, param in module.named_parameters(recurse=False):
            if id(param) in seen:
                continue
            seen.add(id(param))
            full_name = f"{module_name}.{name}" if module_name else name
            if name in known:
                shapes[full_name] = known[name]
            elif param.ndim <= 1:
                shapes[full_name] = ParamShape(ParamKind.vector)
            else:
                raise NotImplementedError(
                    f"muP has no rule for parameter '{full_name}' with shape "
                    f"{tuple(param.shape)} in {type(module).__name__}"
                )
    return shapes
