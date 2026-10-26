"""
Per-parameter shapes used by the muP learning-rate rules.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch.nn as nn

from olmo_core.config import StrEnum
from olmo_core.exceptions import OLMoConfigurationError

if TYPE_CHECKING:
    from .model import Transformer

__all__ = [
    "ParamKind",
    "ParamShape",
    "named_param_shapes",
    "named_matrix_roles",
    "matrix_fan_ins_by_role",
]

# Name segments added by wrappers (activation checkpointing, ``torch.compile``) that don't change
# what a parameter is.
_WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module", "_orig_mod")


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


def _find_transformer(model: nn.Module) -> Tuple["Transformer", str]:
    from .model import Transformer

    for module_name, module in model.named_modules():
        if isinstance(module, Transformer):
            return module, f"{module_name}." if module_name else ""
    raise OLMoConfigurationError("muP base-model fan-ins require a Transformer model")


def _strip_wrappers(path: str) -> str:
    return ".".join(segment for segment in path.split(".") if segment not in _WRAPPER_SEGMENTS)


def _unwrap(module: nn.Module) -> nn.Module:
    for segment in _WRAPPER_SEGMENTS:
        while isinstance(getattr(module, segment, None), nn.Module):
            module = getattr(module, segment)
    return module


def _class_name(module: nn.Module) -> str:
    # FSDP's ``fully_shard`` swaps a module's class for a generated ``FSDP<Class>`` subclass.
    for cls in type(module).__mro__:
        if not cls.__module__.startswith("torch.distributed"):
            return cls.__name__
    return type(module).__name__


def _block_kind(block: nn.Module) -> str:
    from ..layer_norm import LayerNorm

    parts = []
    for name, child in _unwrap(block).named_children():
        child = _unwrap(child)
        if isinstance(child, LayerNorm) or next(child.parameters(), None) is None:
            continue
        # The sequence mixer's type tells e.g. linear-attention blocks from full-attention ones.
        parts.append(f"{name}:{_class_name(child)}" if name == "attention" else name)
    return "+".join(parts)


def named_matrix_roles(model: nn.Module) -> Dict[str, str]:
    """
    Give every weight matrix of a transformer a role that names the same matrix in a model of
    another size, so that muP can scale its learning rate relative to a base model.

    Matrices outside the blocks keep their name (e.g. ``lm_head.w_out.weight``). A matrix inside a
    block is named by the kind of block and its path within the block, e.g.
    ``blocks[attention:KimiDeltaAttention+feed_forward].attention.w_out.weight``. The block kind
    lists the block's non-norm submodules, with the sequence mixer's type, so blocks of different
    kinds (e.g. linear vs. full attention, dense vs. MoE) don't share roles, while models of
    different depths do.

    Wrapper prefixes (``module.``), activation-checkpointing and ``torch.compile`` wrappers, and
    FSDP's class swaps don't change roles.

    :param model: A :class:`~olmo_core.nn.transformer.Transformer`, or a wrapper around one.

    :returns: A mapping from matrix parameter name (as in ``model.named_parameters()``) to role.

    :raises OLMoConfigurationError: If the model has no transformer.
    """
    transformer, prefix = _find_transformer(model)
    block_kinds = {idx: _block_kind(block) for idx, block in transformer.blocks.items()}
    roles: Dict[str, str] = {}
    for name, shape in named_param_shapes(model).items():
        if shape.kind != ParamKind.matrix:
            continue
        local = name[len(prefix) :] if name.startswith(prefix) else name
        if local.startswith("blocks."):
            idx, path = local[len("blocks.") :].split(".", 1)
            roles[name] = f"blocks[{block_kinds[idx]}].{_strip_wrappers(path)}"
        else:
            roles[name] = _strip_wrappers(local)
    return roles


def matrix_fan_ins_by_role(model: nn.Module) -> Dict[str, int]:
    """
    The fan-in of every weight-matrix role in a model, keyed by :func:`named_matrix_roles`. This is
    the table that :class:`~olmo_core.optim.MuPConfig` uses as its base model. The model can be
    built on the ``meta`` device.

    :param model: The base model: a :class:`~olmo_core.nn.transformer.Transformer`, or a wrapper
        around one.

    :returns: A mapping from role to fan-in, sorted by role.

    :raises OLMoConfigurationError: If the model has no transformer, or if matrices sharing a role
        have different fan-ins, which would make the role ambiguous.
    """
    shapes = named_param_shapes(model)
    fan_ins: Dict[str, int] = {}
    for name, role in named_matrix_roles(model).items():
        fan_in = shapes[name].fan_in
        assert fan_in is not None
        if fan_ins.setdefault(role, fan_in) != fan_in:
            raise OLMoConfigurationError(
                f"muP role '{role}' has fan-ins {fan_ins[role]} and {fan_in} (at '{name}') in the "
                "base model"
            )
    return dict(sorted(fan_ins.items()))
