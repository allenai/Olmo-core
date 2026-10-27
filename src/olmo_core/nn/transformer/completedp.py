"""
The model side of the CompletedP parameterization: depth-scaled residual branches,
width-scaled initialization, and the per-parameter roles that the optimizer side uses.
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError

from .init import InitMethod
from .mup import ParamKind, named_param_shapes

if TYPE_CHECKING:
    from .config import TransformerBlockConfig, TransformerConfig
    from .model import Transformer

__all__ = [
    "CompletedPConfig",
    "CompletedPRole",
    "CompletedPParam",
    "named_completedp_params",
    "find_completedp_model",
    "apply_completedp_init_",
]

log = logging.getLogger(__name__)


@dataclass
class CompletedPConfig(Config):
    """
    Width and depth settings for the CompletedP parameterization from
    `Mlodozeniec et al., 2025 <https://arxiv.org/abs/2512.22382>`_ (Table 1), which extends
    CompleteP (`Dey et al., 2025 <https://arxiv.org/abs/2505.01618>`_).

    Set it as :data:`~olmo_core.nn.transformer.TransformerConfig.completedp` on the model config,
    and set :class:`~olmo_core.optim.CompletedPOptimConfig` on an AdamW optimizer config, which
    reads these settings from the model. On the model this:

    - scales every residual branch by ``(n_layers / base_depth) ** -alpha``;
    - multiplies the :data:`~olmo_core.nn.transformer.init.InitMethod.normal` init of each hidden
      weight matrix by ``sqrt(1 / r)``, and of the LM head and of each in-block matrix with a fixed
      output size (routers, KDA ``f_proj_1`` / ``g_proj_1``) by ``1 / r``, so the init variance
      scales as ``1 / width`` and ``1 / width**2``.

    ``r = fan_in / base_fan_in`` is a matrix's own width ratio: its fan-in over the fan-in of the
    same matrix in the base model. Dimensions that grow at different rates (or not at all, like a
    fixed low-rank projection) each get the right ratio, and every matrix of the base model has
    ``r == 1``. Build the config with :meth:`from_base_model`, which records the base fan-ins.

    The base hyperparameters (``init_std``, and the optimizer's ``lr``, ``eps``, ``weight_decay``
    and ``betas``) are the values for the base model, which gets exactly them. Keep the config
    fixed across the models being compared.
    """

    base_width: int
    """
    The ``d_model`` of the base model, the model the base hyperparameters were tuned on.
    """

    base_depth: int
    """
    The number of blocks of the base model.
    """

    alpha: float = 1.0
    """
    The depth exponent, in ``[0.5, 1]``. ``1.0`` is CompleteP.
    """

    base_block_fan_ins: Optional[List[Dict[str, int]]] = None
    """
    The fan-in of every weight matrix in the base model's blocks: one mapping per distinct block
    structure, from a matrix's name inside its block (e.g. ``"attention.w_out.weight"``) to its
    fan-in. A block of another model uses the mapping with exactly its set of matrix names.
    Set by :meth:`from_base_model`.

    If unset, every matrix uses ``base_width`` as its base fan-in. Matrices whose fan-in isn't
    ``d_model`` (e.g. an MLP's down projection) then don't get the base hyperparameters even in the
    base model. This is kept only so that runs started with it keep their parameterization.
    """

    def __post_init__(self):
        if self.base_width <= 0:
            raise OLMoConfigurationError(f"base_width must be positive, got {self.base_width}")
        if self.base_depth <= 0:
            raise OLMoConfigurationError(f"base_depth must be positive, got {self.base_depth}")
        if not 0.5 <= self.alpha <= 1.0:
            raise OLMoConfigurationError(f"alpha must be in [0.5, 1], got {self.alpha}")
        if self.base_block_fan_ins is not None:
            seen: Set[FrozenSet[str]] = set()
            for fan_ins in self.base_block_fan_ins:
                names = frozenset(fan_ins)
                if not names:
                    raise OLMoConfigurationError("base_block_fan_ins has an empty mapping")
                if names in seen:
                    raise OLMoConfigurationError(
                        "base_block_fan_ins has two mappings with the same matrix names: "
                        f"{sorted(names)}"
                    )
                seen.add(names)
                for name, fan_in in fan_ins.items():
                    if fan_in <= 0:
                        raise OLMoConfigurationError(
                            f"base_block_fan_ins['{name}'] must be positive, got {fan_in}"
                        )

    @classmethod
    def from_base_model(
        cls, base_model: "TransformerConfig", alpha: float = 1.0
    ) -> "CompletedPConfig":
        """
        Create the config that transfers hyperparameters tuned on ``base_model``. It records the
        base model's ``d_model``, number of blocks, and the fan-in of every weight matrix in its
        blocks.

        :param base_model: The config of the base model. It is built on the meta device, without
            its own ``completedp`` setting.
        :param alpha: The depth exponent.

        :raises OLMoConfigurationError: If two blocks of the base model have the same weight
            matrices with different fan-ins, so a block of another model can't be matched to one.
        """
        model = base_model.replace(completedp=None).build(init_device="meta")
        transformer, prefix, params = _classify_params(model)
        by_names: Dict[FrozenSet[str], Tuple[str, Dict[str, int]]] = {}
        for block_key, fan_ins in _block_matrix_fan_ins(params, prefix).items():
            names = frozenset(fan_ins)
            if names not in by_names:
                by_names[names] = (block_key, fan_ins)
            elif by_names[names][1] != fan_ins:
                raise OLMoConfigurationError(
                    f"Blocks {by_names[names][0]} and {block_key} of the base model have the same "
                    "weight matrices with different fan-ins, so CompletedP can't tell which one a "
                    "block of another model corresponds to"
                )
        return cls(
            base_width=transformer.d_model,
            base_depth=transformer.n_layers,
            alpha=alpha,
            base_block_fan_ins=[fan_ins for _, fan_ins in by_names.values()],
        )

    def width_ratio(self, d_model: int) -> float:
        """
        The residual-stream width ratio ``m_N`` of a model with the given ``d_model``.
        """
        return d_model / self.base_width

    def depth_ratio(self, n_layers: int) -> float:
        """
        The depth ratio of a model with ``n_layers`` blocks.
        """
        return n_layers / self.base_depth

    def residual_multiplier(self, n_layers: int) -> float:
        """
        The multiplier on every residual branch of a model with ``n_layers`` blocks.
        """
        return self.depth_ratio(n_layers) ** -self.alpha


class CompletedPRole(StrEnum):
    """
    The role of a parameter in CompletedP, which determines its init and optimizer
    rules.
    """

    input_embedding = "input_embedding"
    """
    The input embedding table.
    """

    input_vector = "input_vector"
    """
    An element-wise parameter of the input layer, such as the embedding norm gain.
    """

    hidden_matrix = "hidden_matrix"
    """
    A weight matrix inside a block (projections, experts).
    """

    hidden_readout_matrix = "hidden_readout_matrix"
    """
    A weight matrix inside a block whose output size does not grow with width: MoE routers (one
    output per expert) and the projections into KDA's fixed-width gate bottlenecks (``f_proj_1``,
    ``g_proj_1``). It follows the output-weight rules for width (init std and learning rate
    ``1 / r``, epsilon not width-scaled) and the hidden rules for depth.
    """

    hidden_vector = "hidden_vector"
    """
    An element-wise parameter inside a block whose elements each act on one coordinate or head of
    a width-scaled dimension: norm gains, biases, per-head scalars, depthwise convolution taps, and
    per-head or full-width QK-norm gains.
    """

    hidden_head_shared_vector = "hidden_head_shared_vector"
    """
    An element-wise parameter inside a block that is shared across attention heads, such as
    head-wise QK-norm gains without per-head gains, and the output norm gain of linear attention.
    Its gradient sums over the heads, so its AdamW epsilon does not scale with width.
    """

    output_matrix = "output_matrix"
    """
    The LM head (unembedding) weight.
    """

    output_vector = "output_vector"
    """
    An element-wise parameter of the LM head, such as its norm gain or bias.
    """


@dataclass(frozen=True)
class CompletedPParam:
    """
    The CompletedP role of a parameter and, for matrices, its logical fan-in and the fan-in of the
    same matrix in the base model.
    """

    role: CompletedPRole
    fan_in: Optional[int] = None
    base_fan_in: Optional[int] = None

    @property
    def width_ratio(self) -> float:
        """
        The matrix's own width ratio ``r = fan_in / base_fan_in``.
        """
        assert self.fan_in is not None and self.base_fan_in is not None
        return self.fan_in / self.base_fan_in


_SUPPORTED_BLOCK_NAMES = ("default", "reordered_norm", "peri_norm", "moe_fused_v2")


def completedp_block_configs(
    block_configs: List["TransformerBlockConfig"], config: CompletedPConfig, n_layers: int
) -> List["TransformerBlockConfig"]:
    """
    Copy each block config with its residual multipliers set for CompletedP.

    :param block_configs: The resolved config of every block.
    :param config: The CompletedP config.
    :param n_layers: The number of blocks.

    :returns: The new block configs, in order.

    :raises OLMoConfigurationError: If a block type is not supported, or if a block config already
        sets a residual multiplier.
    """
    from ..moe.v2.nemotron import NemotronBlockConfig

    multiplier = config.residual_multiplier(n_layers)
    resolved = []
    for block_idx, block_config in enumerate(block_configs):
        if (
            isinstance(block_config, NemotronBlockConfig)
            or block_config.name not in _SUPPORTED_BLOCK_NAMES
        ):
            raise OLMoConfigurationError(
                f"CompletedP does not support block {block_idx} of type "
                f"'{block_config.name}' ({type(block_config).__name__})"
            )
        if (
            block_config.attention_residual_alpha is not None
            or block_config.feed_forward_residual_alpha is not None
        ):
            raise OLMoConfigurationError(
                f"CompletedP sets the residual multipliers itself, but block {block_idx} already "
                "sets 'attention_residual_alpha' or 'feed_forward_residual_alpha'"
            )
        resolved.append(
            dataclasses.replace(
                block_config,
                attention_residual_alpha=multiplier,
                feed_forward_residual_alpha=multiplier,
            )
        )
    return resolved


def validate_completedp_model(model: "Transformer") -> None:
    """
    Check that a model built with CompletedP only uses supported components.

    :raises OLMoConfigurationError: If the model uses an unsupported init method, block, sequence
        mixer or LM head, ties its embeddings, or has a block whose weight matrices don't match any
        block of the base model.
    """
    from ..attention import Attention, FusedAttentionV2
    from ..attention.kda import KimiDeltaAttention
    from ..attention.recurrent import GatedDeltaNet
    from ..ddp.block import OLMoDDPTransformerBlock
    from ..lm_head import LMHead
    from .block import (
        PeriNormTransformerBlock,
        ReorderedNormTransformerBlock,
        TransformerBlock,
    )

    if model.init_method != InitMethod.normal:
        raise OLMoConfigurationError(
            f"CompletedP scales the '{InitMethod.normal}' init, but the model uses init method "
            f"'{model.init_method}'"
        )
    if model.tie_word_embeddings:
        raise OLMoConfigurationError("CompletedP does not support tied word embeddings")
    if model.lm_head is not None and type(model.lm_head) is not LMHead:
        raise OLMoConfigurationError(
            f"CompletedP does not support LM head type {type(model.lm_head).__name__}"
        )

    block_types = (
        TransformerBlock,
        ReorderedNormTransformerBlock,
        PeriNormTransformerBlock,
        OLMoDDPTransformerBlock,
    )
    mixer_types = (Attention, FusedAttentionV2, KimiDeltaAttention, GatedDeltaNet)
    for block_key, block in model.blocks.items():
        if type(block) not in block_types:
            raise OLMoConfigurationError(
                f"CompletedP does not support block {block_key} of type {type(block).__name__}"
            )
        mixer = getattr(block, "attention")
        if type(mixer) not in mixer_types:
            raise OLMoConfigurationError(
                f"CompletedP does not support the sequence mixer of block {block_key} "
                f"({type(mixer).__name__})"
            )

    if model.embed_scale is not None and model.embedding_norm is None:
        log.warning(
            "CompletedP is enabled with 'embed_scale' and no embedding norm. A width-dependent "
            "embedding scale breaks the width transfer of the input embedding."
        )
    assert model.completedp is not None
    if model.completedp.base_block_fan_ins is None:
        log.warning(
            "CompletedP has no 'base_block_fan_ins', so every weight matrix uses 'base_width' as "
            "its base fan-in and matrices whose fan-in isn't d_model are rescaled even in the base "
            "model. Build the config with CompletedPConfig.from_base_model() instead."
        )
    # Fails early if a block doesn't match any block of the base model.
    named_completedp_params(model)


def find_completedp_model(model: nn.Module) -> Optional["Transformer"]:
    """
    Find the transformer inside a model (or a wrapper around it) that uses
    CompletedP.

    :returns: The first :class:`~olmo_core.nn.transformer.Transformer` module whose
        ``completedp`` is set, or ``None``.
    """
    from .model import Transformer

    for module in model.modules():
        if isinstance(module, Transformer) and module.completedp is not None:
            return module
    return None


def _head_shared_param_names(transformer: "Transformer", prefix: str) -> Set[str]:
    from ..attention import Attention
    from ..attention.kda import KimiDeltaAttention
    from ..attention.recurrent import GatedDeltaNet

    names: Set[str] = set()
    for module_name, module in transformer.named_modules():
        if isinstance(module, Attention) and module.use_head_qk_norm:
            # Head-wise QK norms share a [head_dim] gain across heads unless each head has its
            # own gains, in which case the gain is 2-D.
            for norm_name in ("q_norm", "k_norm"):
                norm = getattr(module, norm_name)
                if norm is None:
                    continue
                for param_name, param in norm.named_parameters():
                    if param.ndim == 1:
                        names.add(f"{prefix}{module_name}.{norm_name}.{param_name}")
        elif isinstance(module, (KimiDeltaAttention, GatedDeltaNet)):
            # The output norm normalizes each head with one [head_v_dim] gain.
            for param_name, _ in module.o_norm.named_parameters():
                names.add(f"{prefix}{module_name}.o_norm.{param_name}")
    return names


def _readout_param_names(transformer: "Transformer", prefix: str) -> Set[str]:
    from ..attention.kda import KimiDeltaAttention
    from ..moe import MoERouter
    from ..moe.v2.router import MoERouterV2

    names: Set[str] = set()
    for module_name, module in transformer.named_modules():
        if isinstance(module, (MoERouterV2, MoERouter)):
            names.add(f"{prefix}{module_name}.weight")
        elif isinstance(module, KimiDeltaAttention):
            # These project into the fixed head_v_dim-wide bottlenecks of the decay and output
            # gates.
            for linear in ("f_proj_1", "g_proj_1"):
                names.add(f"{prefix}{module_name}.{linear}.weight")
    return names


def named_completedp_params(model: nn.Module) -> Dict[str, CompletedPParam]:
    """
    Classify every parameter of a model for CompletedP.

    Names match ``model.named_parameters()``, including any wrapper prefix such as ``module.``.
    Fan-ins come from :func:`~olmo_core.nn.transformer.mup.named_param_shapes`. If the model uses
    CompletedP, each matrix also gets its base fan-in.

    :param model: A :class:`~olmo_core.nn.transformer.Transformer`, or a wrapper around one.

    :returns: A mapping from parameter name to :class:`CompletedPParam`.

    :raises OLMoConfigurationError: If the model has no transformer, a parameter is not part of
        the embeddings, embedding norm, blocks or LM head, or a block's weight matrices don't match
        any block of the base model.
    """
    transformer, prefix, params = _classify_params(model)
    config = transformer.completedp
    if config is None:
        return params

    base_fan_ins: Dict[str, int] = {}
    templates = {frozenset(fan_ins): fan_ins for fan_ins in (config.base_block_fan_ins or [])}
    for block_key, fan_ins in _block_matrix_fan_ins(params, prefix).items():
        if config.base_block_fan_ins is None:
            template = {name: config.base_width for name in fan_ins}
        else:
            match = templates.get(frozenset(fan_ins))
            if match is None:
                raise OLMoConfigurationError(
                    f"The weight matrices of block {block_key} ({sorted(fan_ins)}) don't match "
                    "any block of the CompletedP base model"
                )
            template = match
        for name in fan_ins:
            base_fan_ins[f"{prefix}blocks.{block_key}.{name}"] = template[name]

    resolved: Dict[str, CompletedPParam] = {}
    for name, param in params.items():
        if param.role == CompletedPRole.output_matrix:
            # The LM head reads the residual stream.
            param = dataclasses.replace(param, base_fan_in=config.base_width)
        elif name in base_fan_ins:
            param = dataclasses.replace(param, base_fan_in=base_fan_ins[name])
        resolved[name] = param
    return resolved


def _block_matrix_fan_ins(
    params: Dict[str, CompletedPParam], prefix: str
) -> Dict[str, Dict[str, int]]:
    """
    The fan-in of every weight matrix in each block, keyed by block key and then by the matrix's
    name inside the block.
    """
    blocks: Dict[str, Dict[str, int]] = {}
    for name, param in params.items():
        if param.role not in (CompletedPRole.hidden_matrix, CompletedPRole.hidden_readout_matrix):
            continue
        assert param.fan_in is not None
        _, block_key, inner = name[len(prefix) :].split(".", 2)
        blocks.setdefault(block_key, {})[inner] = param.fan_in
    return blocks


def _classify_params(model: nn.Module) -> Tuple["Transformer", str, Dict[str, CompletedPParam]]:
    """
    Find the transformer inside ``model`` and classify every parameter, without base fan-ins.

    :returns: The transformer, the prefix of its parameter names, and each parameter's
        :class:`CompletedPParam`.
    """
    from .model import Transformer

    transformer: Optional[Transformer] = None
    prefix = ""
    for module_name, module in model.named_modules():
        if isinstance(module, Transformer):
            transformer = module
            prefix = f"{module_name}." if module_name else ""
            break
    if transformer is None:
        raise OLMoConfigurationError("CompletedP requires a Transformer model")

    head_shared = _head_shared_param_names(transformer, prefix)
    readouts = _readout_param_names(transformer, prefix)
    params: Dict[str, CompletedPParam] = {}
    for name, shape in named_param_shapes(model).items():
        location = name[len(prefix) :].split(".", 1)[0] if name.startswith(prefix) else None
        is_matrix = shape.kind == ParamKind.matrix
        param: Optional[CompletedPParam] = None
        if location == "embeddings" and shape.kind == ParamKind.embedding:
            param = CompletedPParam(CompletedPRole.input_embedding)
        elif location == "embedding_norm" and shape.kind == ParamKind.vector:
            param = CompletedPParam(CompletedPRole.input_vector)
        elif location == "blocks" and is_matrix:
            role = (
                CompletedPRole.hidden_readout_matrix
                if name in readouts
                else CompletedPRole.hidden_matrix
            )
            param = CompletedPParam(role, shape.fan_in)
        elif location == "blocks" and shape.kind == ParamKind.vector:
            role = (
                CompletedPRole.hidden_head_shared_vector
                if name in head_shared
                else CompletedPRole.hidden_vector
            )
            param = CompletedPParam(role)
        elif location == "lm_head" and is_matrix:
            param = CompletedPParam(CompletedPRole.output_matrix, shape.fan_in)
        elif location == "lm_head" and shape.kind == ParamKind.vector:
            param = CompletedPParam(CompletedPRole.output_vector)
        if param is None:
            raise OLMoConfigurationError(
                f"CompletedP has no rule for parameter '{name}' ({shape.kind})"
            )
        params[name] = param
    return transformer, prefix, params


@torch.no_grad()
def apply_completedp_init_(model: "Transformer") -> None:
    """
    Scale a model's :data:`~olmo_core.nn.transformer.init.InitMethod.normal` init in place to the
    CompletedP init: hidden weight matrices by ``sqrt(1 / r)``, and the LM head and in-block
    matrices with a fixed output size by ``1 / r``, where ``r`` is the matrix's own width ratio.
    Other parameters are unchanged.

    :param model: A model whose ``completedp`` is set, right after its standard init.
    """
    assert model.completedp is not None
    named_params = dict(model.named_parameters())
    for name, param in named_completedp_params(model).items():
        if param.role == CompletedPRole.hidden_matrix:
            assert param.fan_in is not None and param.base_fan_in is not None
            factor = (param.base_fan_in / param.fan_in) ** 0.5
        elif param.role in (CompletedPRole.output_matrix, CompletedPRole.hidden_readout_matrix):
            assert param.fan_in is not None and param.base_fan_in is not None
            factor = param.base_fan_in / param.fan_in
        else:
            continue
        if factor != 1.0:
            named_params[name].mul_(factor)
