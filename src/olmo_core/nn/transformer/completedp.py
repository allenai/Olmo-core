"""
The model side of the CompletedP parameterization: depth-scaled residual branches,
width-scaled initialization, and the per-parameter roles that the optimizer side uses.
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch
import torch.nn as nn

from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError

from .init import InitMethod
from .mup import ParamKind, named_param_shapes

if TYPE_CHECKING:
    from .config import TransformerBlockConfig
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
      weight matrix by ``sqrt(base_width / fan_in)``, and of the LM head and of each in-block
      matrix with a fixed output size (routers, KDA ``f_proj_1`` / ``g_proj_1``) by
      ``base_width / fan_in``, so the init variance scales as ``1 / width`` and ``1 / width**2``.

    Each matrix uses its own fan-in, so dimensions that don't grow with ``d_model`` (e.g. a fixed
    low-rank projection) keep their settings. The base hyperparameters (``init_std``, and the
    optimizer's ``lr``, ``eps``, ``weight_decay`` and ``betas``) are the values for a model with
    ``d_model == base_width`` and ``n_layers == base_depth``. Keep ``base_width``, ``base_depth`` and
    ``alpha`` fixed across the models being compared.
    """

    base_width: int
    """
    The fan-in at which a weight matrix gets exactly the base init std and learning rate,
    typically the ``d_model`` of the model the base hyperparameters were tuned on.
    """

    base_depth: int
    """
    The number of blocks of the model the base hyperparameters were tuned on.
    """

    alpha: float = 1.0
    """
    The depth exponent, in ``[0.5, 1]``. ``1.0`` is CompleteP.
    """

    def __post_init__(self):
        if self.base_width <= 0:
            raise OLMoConfigurationError(f"base_width must be positive, got {self.base_width}")
        if self.base_depth <= 0:
            raise OLMoConfigurationError(f"base_depth must be positive, got {self.base_depth}")
        if not 0.5 <= self.alpha <= 1.0:
            raise OLMoConfigurationError(f"alpha must be in [0.5, 1], got {self.alpha}")

    def width_ratio(self, fan_in: int) -> float:
        """
        The width ratio of a weight matrix with the given fan-in.
        """
        return fan_in / self.base_width

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
    The CompletedP role of a parameter and, for matrices, its logical fan-in.
    """

    role: CompletedPRole
    fan_in: Optional[int] = None


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
        mixer or LM head, or ties its embeddings.
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
    Fan-ins come from :func:`~olmo_core.nn.transformer.mup.named_param_shapes`.

    :param model: A :class:`~olmo_core.nn.transformer.Transformer`, or a wrapper around one.

    :returns: A mapping from parameter name to :class:`CompletedPParam`.

    :raises OLMoConfigurationError: If the model has no transformer, or a parameter is not part of
        the embeddings, embedding norm, blocks or LM head.
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
    return params


@torch.no_grad()
def apply_completedp_init_(model: "Transformer") -> None:
    """
    Scale a model's :data:`~olmo_core.nn.transformer.init.InitMethod.normal` init in place to the
    CompletedP init: hidden weight matrices by ``sqrt(base_width / fan_in)``, and the LM head and
    in-block matrices with a fixed output size by ``base_width / fan_in``. Other parameters are
    unchanged.

    :param model: A model whose ``completedp`` is set, right after its standard init.
    """
    config = model.completedp
    assert config is not None
    named_params = dict(model.named_parameters())
    for name, param in named_completedp_params(model).items():
        if param.role == CompletedPRole.hidden_matrix:
            assert param.fan_in is not None
            factor = (config.base_width / param.fan_in) ** 0.5
        elif param.role in (CompletedPRole.output_matrix, CompletedPRole.hidden_readout_matrix):
            assert param.fan_in is not None
            factor = config.base_width / param.fan_in
        else:
            continue
        if factor != 1.0:
            named_params[name].mul_(factor)
