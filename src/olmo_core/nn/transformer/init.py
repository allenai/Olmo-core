from typing import TYPE_CHECKING, Optional, Sequence, Union, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import distribute_like, get_local_tensor

if TYPE_CHECKING:
    from ..attention import SequenceMixer
    from ..feed_forward import FeedForward
    from ..moe import MoEBase


def _apply_init(init_fun, x: torch.Tensor, *args, **kwargs):
    if not isinstance(x, DTensor):
        init_fun(x, *args, **kwargs)
        return

    # Initialize full version of x locally, then apply init to that.
    full_x = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    init_fun(full_x, *args, **kwargs)
    full_x = distribute_like(x, full_x)

    # Now copy over the corresponding shard of `full_x` into `x`.
    get_local_tensor(x).copy_(get_local_tensor(full_x))


def init_linear(
    m: nn.Linear | nn.Conv1d, *, std: float = 0.02, generator: Optional[torch.Generator] = None
):
    _apply_init(
        nn.init.trunc_normal_,
        m.weight,
        mean=0.0,
        std=std,
        a=-3 * std,
        b=3 * std,
        generator=generator,
    )
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _init_trunc_normal(
    x: torch.Tensor, *, std: float, generator: Optional[torch.Generator] = None
):
    _apply_init(
        nn.init.trunc_normal_,
        x,
        mean=0.0,
        std=std,
        a=-3 * std,
        b=3 * std,
        generator=generator,
    )


def nanochat_std(fan_in: int, fan_out: int) -> float:
    """
    The standard deviation that :data:`InitMethod.nanochat` uses for a weight matrix that maps
    ``fan_in`` features to ``fan_out`` features.

    :param fan_in: The number of input features.
    :param fan_out: The number of output features.

    :returns: ``1/√fan_in · min(1, √(fan_out/fan_in))``.
    """
    return fan_in**-0.5 * min(1.0, (fan_out / fan_in) ** 0.5)


def init_linear_nanochat(m: nn.Linear | nn.Conv1d, *, generator: Optional[torch.Generator] = None):
    """
    Initialize a linear layer or a depthwise 1D convolution following :data:`InitMethod.nanochat`.

    A depthwise convolution maps each channel to itself from ``kernel_size`` taps, so it is
    treated as a square matrix with ``fan_in = fan_out = kernel_size``.

    :param m: The module to initialize.
    :param generator: The random generator.

    :raises NotImplementedError: If ``m`` is a convolution that is not depthwise.
    """
    if isinstance(m, nn.Conv1d):
        if not (m.groups == m.in_channels == m.out_channels):
            raise NotImplementedError(
                f"init method '{InitMethod.nanochat}' only supports depthwise convolutions"
            )
        std = nanochat_std(m.kernel_size[0], m.kernel_size[0])
    else:
        std = nanochat_std(m.in_features, m.out_features)
    init_linear(m, std=std, generator=generator)


def init_fused_linear_nanochat(
    m: nn.Linear, out_features: Sequence[int], *, generator: Optional[torch.Generator] = None
):
    """
    Initialize a linear layer that fuses several projections along its output dimension
    following :data:`InitMethod.nanochat`, using a separate standard deviation for each
    projection.

    :param m: The fused linear layer.
    :param out_features: The output size of each fused projection, in order.
    :param generator: The random generator.
    """
    if sum(out_features) != m.out_features:
        raise ValueError(
            f"fused output sizes {tuple(out_features)} do not add up to {m.out_features}"
        )

    def init_fun(w: torch.Tensor, *, generator: Optional[torch.Generator] = None):
        for rows in w.split(list(out_features), dim=0):
            std = nanochat_std(m.in_features, rows.shape[0])
            nn.init.trunc_normal_(rows, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)

    _apply_init(init_fun, m.weight, generator=generator)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class InitMethod(StrEnum):
    normal = "normal"
    """
    Every linear and embedding layer and initialized from a truncated normal distributed
    with standard deviation 0.02.
    """

    normalized = "normalized"
    """
    Follow the nGPT initialization scheme.
    """

    llama = "llama"
    """
    Like :data:`normal`, but "output" layers are initialized with a standard deviation that's
    dependent on either ``d_model`` or the number of layers.
    """

    llama_depth = "llama_depth"
    """
    Like :data:`normal`, but "output" layers are initialized with a standard deviation that's
    dependent on either ``d_model`` or the layer index.
    """

    fan_in = "fan_in"
    """
    Per-layer fan-in initialization where each weight matrix is initialized with
    ``std = 1/√d_in`` where ``d_in`` is the fan-in (number of input features) of that
    specific layer. Embeddings use ``std = 1.0`` with normal distribution.
    This provides forward-pass variance-preserving initialization adapted to each layer's
    specific dimensions, with no depth scaling.
    """

    nanochat = "nanochat"
    """
    The initialization used by `nanochat <https://github.com/karpathy/nanochat>`_, which follows
    the spectral condition for feature learning (`Yang et al., 2023
    <https://arxiv.org/abs/2310.17813>`_). Every weight matrix mapping ``fan_in`` features to
    ``fan_out`` features is initialized from a truncated normal distribution with
    ``std = 1/√fan_in · min(1, √(fan_out/fan_in))`` (see :func:`nanochat_std`), with no depth
    scaling. Fused matrices use the shape of each fused projection, depthwise convolutions use
    ``std = 1/√kernel_size``, and embeddings use a normal distribution with ``std = 1.0``.
    The base ``init_std`` and ``embedding_init_std`` are ignored.
    """

    def init_embeddings(
        self,
        m: nn.Embedding,
        *,
        d_model: int,
        embed_scale: Optional[float] = None,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self in (InitMethod.llama, InitMethod.llama_depth, InitMethod.nanochat):
            _apply_init(nn.init.normal_, m.weight, generator=generator)
        elif self == InitMethod.normalized:
            _apply_init(nn.init.normal_, m.weight, generator=generator, std=d_model**-0.5)
        elif self == InitMethod.fan_in:
            # Fan-in init uses std = 1.0 for embeddings, scaled down by embed_scale if set
            emb_std = 1.0 / embed_scale if embed_scale is not None else 1.0
            _apply_init(nn.init.normal_, m.weight, generator=generator, std=emb_std)
        else:
            _apply_init(
                nn.init.trunc_normal_,
                m.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )

    def init_final_w_out(
        self,
        m: nn.Linear,
        *,
        d_model: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self in (
            InitMethod.llama,
            InitMethod.llama_depth,
            InitMethod.normalized,
            InitMethod.fan_in,
        ):
            std = d_model**-0.5
        elif self == InitMethod.nanochat:
            std = nanochat_std(m.in_features, m.out_features)
        init_linear(m, std=std, generator=generator)

    def init_attention(
        self,
        m: "SequenceMixer",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        m.init_weights(
            init_method=self,
            d_model=d_model,
            block_idx=block_idx,
            num_blocks=num_blocks,
            std=std,
            generator=generator,
        )

    def init_feed_forward(
        self,
        m: "FeedForward",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self == InitMethod.nanochat:
            for w in (m.w1, m.w3, m.w2):
                init_linear_nanochat(w, generator=generator)
            return

        # Compute std for w1 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w1 uses 1/√d_in where d_in = d_model (ignores base std parameter)
            std = m.w1.in_features**-0.5
        elif self == InitMethod.normalized:
            std = d_model**-0.5

        init_linear(m.w1, std=std, generator=generator)

        # Compute std for w3 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w3 uses 1/√d_in where d_in = d_model
            std = m.w3.in_features**-0.5
        elif self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        init_linear(m.w3, std=std, generator=generator)

        # Compute std for w2 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w2 uses 1/√d_in where d_in = hidden_size
            std = m.w2.in_features**-0.5
        elif self == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(m.w2, std=std, generator=generator)

    def init_feed_forward_moe(
        self,
        m: "MoEBase",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        from ..moe import DroplessMoEMLP, MoELinearRouter, MoEMLP

        if self == InitMethod.nanochat:
            router = cast(MoELinearRouter, m.router)
            _init_trunc_normal(
                router.weight,
                std=nanochat_std(router.d_model, router.num_experts),
                generator=generator,
            )
            for projection in (m.latent_down_proj, m.latent_up_proj):
                if projection is not None:
                    init_linear_nanochat(projection, generator=generator)

            mlp = cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp)
            up_std = nanochat_std(mlp.d_model, mlp.hidden_size)
            _init_trunc_normal(mlp.w1, std=up_std, generator=generator)
            _init_trunc_normal(
                mlp.w2, std=nanochat_std(mlp.hidden_size, mlp.d_model), generator=generator
            )
            _init_trunc_normal(mlp.w3, std=up_std, generator=generator)

            if m.shared_mlp is not None:
                self.init_feed_forward(
                    m.shared_mlp,
                    d_model=d_model,
                    block_idx=block_idx,
                    num_blocks=num_blocks,
                    generator=generator,
                )
            return

        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif self == InitMethod.fan_in:
            # Routing decisions are always computed from model-space activations.
            std = m.router.d_model**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            cast(MoELinearRouter, m.router).weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        for projection in (m.latent_down_proj, m.latent_up_proj):
            if projection is not None:
                projection_std = (
                    projection.in_features**-0.5 if self == InitMethod.fan_in else std
                )
                init_linear(projection, std=projection_std, generator=generator)

        mlp = cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp)

        # Initialize w1 (maps d_model -> hidden_size, fan-in = d_model)
        if self == InitMethod.fan_in:
            std = mlp.d_model**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            mlp.w1,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        # Initialize w2 (maps hidden_size -> d_model, fan-in = hidden_size)
        if self == InitMethod.fan_in:
            std = mlp.hidden_size**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            mlp.w2,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        # Initialize w3 (maps d_model -> hidden_size, fan-in = d_model)
        if self == InitMethod.fan_in:
            std = mlp.d_model**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            mlp.w3,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

    def init_moe_v2(
        self,
        b,
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
        ep_generator: Optional[torch.Generator] = None,
    ):
        """
        Initialize the weights of an :class:`~olmo_core.nn.ddp.block.OLMoDDPTransformerBlock`.

        :param ep_generator: An optional separate generator for expert-parallel-sharded weights
            (the routed expert weights). Falls back to ``generator`` when expert parallelism is
            not enabled.
        """
        from ..ddp.block import OLMoDDPTransformerBlock

        b = cast(OLMoDDPTransformerBlock, b)
        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        if ep_generator is None:
            assert b.ep_enabled is False, "ep_generator should be provided when ep_enabled is True"
            # Expert parallelism is not in use, so the routed expert weights are not sharded and
            # can share the default generator.
            ep_generator = generator

        if self == InitMethod.nanochat:
            for router in (b.shared_experts_router, b.routed_experts_router):
                if router is not None:
                    _init_trunc_normal(
                        router.weight,
                        std=nanochat_std(router.d_model, router.num_experts),
                        generator=generator,
                    )
                    if router.bias is not None:
                        _apply_init(nn.init.zeros_, router.bias)

            for projection in (b.latent_down_proj, b.latent_up_proj):
                if projection is not None:
                    init_linear_nanochat(projection, generator=generator)

            # The fused ``w_up_gate`` holds a gate and an up projection, each with
            # ``hidden_size`` outputs.
            if b.routed_experts:
                routed = b.routed_experts
                _init_trunc_normal(
                    routed.w_up_gate,
                    std=nanochat_std(routed.d_model, routed.hidden_size),
                    generator=ep_generator,
                )
                _init_trunc_normal(
                    routed.w_down,
                    std=nanochat_std(routed.hidden_size, routed.d_model),
                    generator=ep_generator,
                )
                if routed.b_up_gate is not None:
                    _apply_init(nn.init.zeros_, routed.b_up_gate)
                if routed.b_down is not None:
                    _apply_init(nn.init.zeros_, routed.b_down)

            if b.shared_experts:
                shared = b.shared_experts
                _init_trunc_normal(
                    shared.w_up_gate,
                    std=nanochat_std(shared.d_model, shared.hidden_size),
                    generator=generator,
                )
                _init_trunc_normal(
                    shared.w_down,
                    std=nanochat_std(shared.hidden_size, shared.d_model),
                    generator=generator,
                )
            return

        if b.shared_experts_router:
            _apply_init(
                nn.init.trunc_normal_,
                b.shared_experts_router.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )
            if b.shared_experts_router.bias is not None:
                _apply_init(nn.init.zeros_, b.shared_experts_router.bias)
        if b.routed_experts_router:
            routed_router_std = (
                b.routed_experts_router.d_model**-0.5 if self == InitMethod.fan_in else std
            )
            _apply_init(
                nn.init.trunc_normal_,
                b.routed_experts_router.weight,
                mean=0.0,
                std=routed_router_std,
                a=-3 * routed_router_std,
                b=3 * routed_router_std,
                generator=generator,
            )
            if b.routed_experts_router.bias is not None:
                _apply_init(nn.init.zeros_, b.routed_experts_router.bias)

        for projection in (b.latent_down_proj, b.latent_up_proj):
            if projection is not None:
                projection_std = (
                    projection.in_features**-0.5 if self == InitMethod.fan_in else std
                )
                init_linear(projection, std=projection_std, generator=generator)

        if b.routed_experts:
            # The routed expert weights may be sharded across the expert-parallel mesh, so use the
            # EP generator to keep initialization consistent across shards.
            routed_up_gate_std = (
                b.routed_experts.d_model**-0.5 if self == InitMethod.fan_in else std
            )
            _apply_init(
                nn.init.trunc_normal_,
                b.routed_experts.w_up_gate,
                mean=0.0,
                std=routed_up_gate_std,
                a=-3 * routed_up_gate_std,
                b=3 * routed_up_gate_std,
                generator=ep_generator,
            )
            routed_down_std = (
                b.routed_experts.hidden_size**-0.5 if self == InitMethod.fan_in else std
            )
            _apply_init(
                nn.init.trunc_normal_,
                b.routed_experts.w_down,
                mean=0.0,
                std=routed_down_std,
                a=-3 * routed_down_std,
                b=3 * routed_down_std,
                generator=ep_generator,
            )
            if b.routed_experts.b_up_gate is not None:
                _apply_init(nn.init.zeros_, b.routed_experts.b_up_gate)
            if b.routed_experts.b_down is not None:
                _apply_init(nn.init.zeros_, b.routed_experts.b_down)

        if b.shared_experts:
            _apply_init(
                nn.init.trunc_normal_,
                b.shared_experts.w_up_gate,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )
            _apply_init(
                nn.init.trunc_normal_,
                b.shared_experts.w_down,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )
