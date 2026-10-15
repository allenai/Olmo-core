import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import Placement, Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.ops import attach_auxiliary_loss

from ..attention.base import SequenceMixerConfig
from ..attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForward, FeedForwardConfig
from ..functional import l2_normalize
from ..layer_norm import LayerNormConfig
from ..moe import MoEConfig, MoERouter
from ..moe.parallel_mlp import ParallelMLPBase
from ..residual_stream import ResidualStream
from .config import TransformerDataParallelWrappingStrategy

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType


class TransformerBlockBase(nn.Module):
    """
    Base class for transformer block implementations.
    """

    def __init__(self, *, n_layers: int):
        super().__init__()
        self.n_layers = n_layers

    @property
    def is_moe(self) -> bool:
        return False

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the block on the input ``x``.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        """
        raise NotImplementedError

    def apply_pp(self, pp_mesh: DeviceMesh):
        del pp_mesh

    @abstractmethod
    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        raise NotImplementedError

    @abstractmethod
    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        raise NotImplementedError

    def apply_compile(self):
        self.compile(fullgraph=False)

    @abstractmethod
    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def num_flops_per_token(self, seq_len: int) -> int:
        raise NotImplementedError


class TransformerBlock(TransformerBlockBase):
    """
    A typical "Llama-style" transformer block implementation.

    :param d_model: The model dimensionality.
    :param block_idx: The index/position of the block within the model. Ranges from 0 to ``n_layers - 1``.
    :param sequence_mixer: The sequence mixer module config (e.g. attention, recurrent, convolution, etc.).
    :param feed_forward: The feed forward module config.
    :param layer_norm: The layer norm config for both the attention LN and the feed forward LN.
    :param dropout: Dropout probability.
    :param init_device: The device used when initializing parameters.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        masked_dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx

        # NOTE: The `self.attention` naming is kept for backwards compatibility with old checkpoints.
        # `self.attention` could contain any `SequenceMixer` implementation, such as a `GatedDeltaNet`.
        # Generally it's ok to think of these as "attention" modules at the block level.
        self.attention = sequence_mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.attention_residual_stream = ResidualStream(
            alpha=attention_residual_alpha, dropout=dropout, masked_dropout=masked_dropout
        )
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)
        self.feed_forward_residual_stream = ResidualStream(
            alpha=feed_forward_residual_alpha, dropout=dropout, masked_dropout=masked_dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        drop_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        :param drop_mask: Optional ``(batch_size, seq_len)`` 0/1 tensor selecting the tokens
            that receive :data:`ResidualStream.masked_dropout`. Consumed here rather than
            forwarded into the sequence mixer. Ignored unless a residual stream was built
            with ``masked_dropout > 0``.
        """
        del loss_div_factor
        h = self.attention_residual_stream(
            x, self.attention(self.attention_norm(x), **kwargs), drop_mask=drop_mask
        )
        return self.feed_forward_residual_stream(
            h, self.feed_forward(self.feed_forward_norm(h)), drop_mask=drop_mask
        )

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(input_layout,),
                desired_input_layouts=(Shard(1),),
            ),
        )

        parallelize_module(
            self.attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )
        parallelize_module(
            self.attention_residual_stream.dropout,
            device_mesh=tp_mesh,
            parallelize_plan=SequenceParallel(),
        )

        self.attention.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )
        parallelize_module(
            self.feed_forward_residual_stream.dropout,
            device_mesh=tp_mesh,
            parallelize_plan=SequenceParallel(),
        )

        self.feed_forward.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_mlp = cast(FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

    def num_flops_per_token(self, seq_len: int) -> int:
        attn_flops = self.attention.num_flops_per_token(seq_len)
        ff_flops = self.feed_forward.num_flops_per_token(seq_len)
        return attn_flops + ff_flops


class LayerNormScaledTransformerBlock(TransformerBlock):
    """
    A variant of ``TransformerBlock`` that applies
    `LayerNorm Scaling (LNS) <https://github.com/lmsdss/LayerNorm-Scaling>`_.

    Each LayerNorm output is multiplied by ``1 / sqrt(layer_id)`` where ``layer_id`` is the
    1-based position of the block inside the transformer. Keeping this logic in a dedicated
    subclass ensures that the vanilla ``TransformerBlock`` remains simple and easy to reason
    about.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            feed_forward=feed_forward,
            layer_norm=layer_norm,
            dropout=dropout,
            attention_residual_alpha=attention_residual_alpha,
            feed_forward_residual_alpha=feed_forward_residual_alpha,
            init_device=init_device,
            cache=cache,
        )

        # LayerNorm scaling factor 1/sqrt(layer_id), where layer_id is 1-based.
        ln_scale_value = 1.0 / math.sqrt(block_idx + 1)
        self.register_buffer("ln_scale", torch.tensor(ln_scale_value, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        scale = self.ln_scale.to(dtype=x.dtype, device=x.device)
        h = self.attention_residual_stream(
            x, self.attention(self.attention_norm(x) * scale, **kwargs)
        )
        return self.feed_forward_residual_stream(
            h, self.feed_forward(self.feed_forward_norm(h) * scale)
        )


class ReorderedNormTransformerBlock(TransformerBlock):
    """
    Like :class:`TransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the output
    of the feed-forward instead of the input.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        h = self.attention_residual_stream(x, self.attention_norm(self.attention(x, **kwargs)))
        return self.feed_forward_residual_stream(h, self.feed_forward_norm(self.feed_forward(h)))


class PeriNormTransformerBlock(TransformerBlock):
    """
    A transformer block in the style of `Peri-LN <https://arxiv.org/pdf/2502.02732>`_.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            feed_forward=feed_forward,
            layer_norm=layer_norm,
            dropout=dropout,
            attention_residual_alpha=attention_residual_alpha,
            feed_forward_residual_alpha=feed_forward_residual_alpha,
            init_device=init_device,
            cache=cache,
        )
        self.post_attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.post_feed_forward_norm = layer_norm.build(d_model, init_device=init_device)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        h = self.attention_residual_stream(
            x, self.post_attention_norm(self.attention(self.attention_norm(x), **kwargs))
        )
        return self.feed_forward_residual_stream(
            h, self.post_feed_forward_norm(self.feed_forward(self.feed_forward_norm(h)))
        )

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        super().apply_tp(tp_mesh, input_layout=input_layout, float8_enabled=float8_enabled)
        parallelize_module(
            self.post_feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )
        parallelize_module(
            self.post_attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )


@beta_feature
class NormalizedTransformerBlock(TransformerBlockBase):
    """
    An nGPT block implementation to be used with the :class:`~olmo_core.nn.attention.NormalizedAttention`
    attention type and :class:`~olmo_core.nn.feed_forward.NormalizedFeedForward` feed-forward type.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx

        # NOTE: The `self.attention` naming is kept for backwards compatibility with old checkpoints.
        # `self.attention` could contain any `SequenceMixer` implementation, such as a `GatedDeltaNet`.
        # Generally it's ok to think of these as "attention" modules at the block level.
        self.attention = sequence_mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.attn_alpha = nn.Parameter(
            torch.empty(d_model, dtype=torch.float32, device=init_device)
        )

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.mlp_alpha = nn.Parameter(torch.empty(d_model, dtype=torch.float32, device=init_device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.attn_alpha)
        nn.init.ones_(self.mlp_alpha)
        with torch.no_grad():
            self.attn_alpha.mul_(self.attn_alpha_init_scaling)
            self.mlp_alpha.mul_(self.mlp_alpha_init_scaling)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        h = l2_normalize(
            torch.lerp(
                x,
                l2_normalize(self.attention(x, **kwargs)),
                (
                    self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
                ).abs(),
            )
        )

        return l2_normalize(
            torch.lerp(
                h,
                l2_normalize(self.feed_forward(h)),
                (self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)).abs(),
            )
        )

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        del tp_mesh, input_layout, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer block variant"
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
            fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)

        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

        if (
            wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained
            and prefetch_factor > 0
        ):
            cast(FSDPModule, self).set_modules_to_forward_prefetch(
                [cast(FSDPModule, self.attention)]
            )
            cast(FSDPModule, self.attention).set_modules_to_forward_prefetch(
                [cast(FSDPModule, self.feed_forward)]
            )

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        if hasattr(self.attention, "normalize_matrices"):
            self.attention.normalize_matrices()  # type: ignore

        if hasattr(self.feed_forward, "normalize_matrices"):
            self.feed_forward.normalize_matrices()  # type: ignore

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))

    def num_flops_per_token(self, seq_len: int) -> int:
        attn_flops = self.attention.num_flops_per_token(seq_len)
        ff_flops = self.feed_forward.num_flops_per_token(seq_len)
        return attn_flops + ff_flops


@beta_feature
class MoETransformerBlock(TransformerBlockBase):
    """
    Like :class:`TransformerBlock` except that the dense :class:`~olmo_core.nn.feed_forward.FeedForward`
    module is replaced with a mixture-of-experts (MoE).
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward_moe: MoEConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx

        # NOTE: The `self.attention` naming is kept for backwards compatibility with old checkpoints.
        # `self.attention` could contain any `SequenceMixer` implementation, such as a `GatedDeltaNet`.
        # Generally it's ok to think of these as "attention" modules at the block level.
        self.attention = sequence_mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.feed_forward_moe = feed_forward_moe.build(
            d_model=d_model, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._ep_enabled = False
        self._tp_enabled = False

    @property
    def is_moe(self) -> bool:
        return True

    @property
    def router(self) -> MoERouter:
        return self.feed_forward_moe.router

    @property
    def shared_mlp(self) -> Optional[FeedForward]:
        return self.feed_forward_moe.shared_mlp

    @property
    def experts(self) -> ParallelMLPBase:
        return self.feed_forward_moe.experts

    @property
    def top_k(self) -> int:
        return self.feed_forward_moe.top_k

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        return self.feed_forward_moe.compute_metrics(reset=reset)

    def reset_metrics(self):
        self.feed_forward_moe.reset_metrics()

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(
            self.feed_forward_moe(self.feed_forward_norm(h), loss_div_factor=loss_div_factor)
        )

    def apply_pp(self, pp_mesh: DeviceMesh):
        self.feed_forward_moe.apply_pp(pp_mesh)

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        self.feed_forward_moe.apply_ep(ep_mesh, **kwargs)
        self._ep_enabled = True

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(input_layout,),
                desired_input_layouts=(Shard(1),),
            ),
        )

        parallelize_module(
            self.attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.attention.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.feed_forward_moe.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(self.dropout, device_mesh=tp_mesh, parallelize_plan=SequenceParallel())

        self._tp_enabled = True

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)
        self.feed_forward_moe.apply_cp(cp_mesh)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_moe = cast(
                FSDPModule, fully_shard(self.feed_forward_moe, mesh=dp_mesh, **fsdp_kwargs)
            )
            fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_moe])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

    def num_flops_per_token(self, seq_len: int) -> int:
        attn_flops = self.attention.num_flops_per_token(seq_len)
        moe_flops = self.feed_forward_moe.num_flops_per_token(seq_len)
        return attn_flops + moe_flops


@beta_feature
class MoEReorderedNormTransformerBlock(MoETransformerBlock):
    """
    Like :class:`MoETransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the
    output of the feed-forward MoE instead of the input.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(
            self.feed_forward_norm(self.feed_forward_moe(h, loss_div_factor=loss_div_factor))
        )

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_moe = cast(
                FSDPModule, fully_shard(self.feed_forward_moe, mesh=dp_mesh, **fsdp_kwargs)
            )
            fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_moe])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


@beta_feature
class MoEHybridTransformerBlockBase(MoETransformerBlock):
    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        layer_norm: LayerNormConfig,
        feed_forward: FeedForwardConfig,
        init_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            layer_norm=layer_norm,
            init_device=init_device,
            **kwargs,
        )
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)
        self.feed_forward_moe_norm = layer_norm.build(d_model, init_device=init_device)
        self._use_combined_forward: Optional[bool] = None

    @property
    def use_combined_forward(self) -> bool:
        if self._use_combined_forward is not None:
            return self._use_combined_forward
        elif not self.ep_enabled and not self.tp_enabled:
            return False
        else:
            return True

    @use_combined_forward.setter
    def use_combined_forward(self, should_use: bool):
        if should_use and not (self.tp_enabled or self.ep_enabled):
            raise RuntimeError(
                "combined forward can only be used when expert parallelism is enabled"
            )
        self._use_combined_forward = should_use

    @abstractmethod
    def dense_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sparse_forward(
        self, x: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.use_combined_forward:
            return self.sparse_forward(x, loss_div_factor=loss_div_factor) + self.dense_forward(
                x, **kwargs
            )
        else:
            # NOTE: alternatively could do something like this, but even with an extra stream it's
            # not as fast as the hand-crafted 'combined_forward()'.
            # stream = get_or_init_stream()
            # stream.wait_stream(torch.cuda.default_stream())
            # h_sparse = self._fwd_sparse(x)
            # with torch.cuda.stream(stream):
            #     h_dense = self._fwd_dense(x, **kwargs)
            # torch.cuda.default_stream().wait_stream(stream)
            # return h_sparse + h_dense
            return self.combined_forward(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        super().apply_tp(tp_mesh, input_layout=input_layout, float8_enabled=float8_enabled)

        self.feed_forward.apply_tp(
            tp_mesh,
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_moe_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        from torch.distributed.fsdp import MixedPrecisionPolicy

        # Force router to be full-precision.
        fsdp_router = cast(
            FSDPModule,
            fully_shard(
                self.feed_forward_moe.router,
                mesh=dp_mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32),
            ),
        )

        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            if not self.use_combined_forward:
                fsdp_att = cast(
                    FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_mlp = cast(
                    FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_moe = cast(
                    FSDPModule, fully_shard(self.feed_forward_moe, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
                if prefetch_factor > 0:
                    fsdp_root.set_modules_to_forward_prefetch([fsdp_router, fsdp_moe, fsdp_att])
                    fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
            else:
                fsdp_att = cast(
                    FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_mlp = cast(
                    FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)
                )
                #  fsdp_moe = cast(
                #      FSDPModule,
                #      fully_shard(self.feed_forward_moe.experts.mlp, mesh=dp_mesh, **fsdp_kwargs),
                #  )
                fsdp_shared_mlp = (
                    None
                    if self.feed_forward_moe.shared_mlp is None
                    else cast(
                        FSDPModule,
                        fully_shard(self.feed_forward_moe.shared_mlp, mesh=dp_mesh, **fsdp_kwargs),
                    )
                )
                fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))

                if prefetch_factor > 0:
                    #  fsdp_root.set_modules_to_forward_prefetch([fsdp_att, fsdp_moe])
                    fsdp_root.set_modules_to_forward_prefetch([fsdp_att, fsdp_router])
                    if fsdp_shared_mlp is not None:
                        fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp, fsdp_shared_mlp])
                    else:
                        fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


@beta_feature
class MoEHybridTransformerBlock(MoEHybridTransformerBlockBase):
    def dense_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))

    def sparse_forward(
        self, x: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        return self.dropout(
            self.feed_forward_moe(self.feed_forward_moe_norm(x), loss_div_factor=loss_div_factor)
        )

    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # NOTE: this follows the same code path as the MoE's forward pass, except that we run
        # dense operations while we wait on expert parallel all-to-all comms.
        B, _, D = x.shape

        model_moe_input = get_local_tensor(self.feed_forward_moe_norm(x))
        x_moe = self.feed_forward_moe.prepare_routed_input(model_moe_input)
        routed_dim = x_moe.shape[-1]

        expert_weights, expert_indices, batch_size_per_expert, router_aux_loss = self.router(
            model_moe_input, loss_div_factor=loss_div_factor
        )

        if router_aux_loss is not None:
            x_moe = attach_auxiliary_loss(x_moe, router_aux_loss)

        # shape: (batch_size * seq_len, d_model)
        x_moe = x_moe.view(-1, routed_dim)
        # shape: (batch_size * top_k,)
        expert_weights = expert_weights.flatten()
        # shape: (batch_size * top_k,)
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins = self.experts.indices_and_bins(
                expert_indices, batch_size_per_expert
            )

        (
            parallel_x,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            parallel_batch_size_per_expert,
            recv_counts,
            send_counts,
            expert_capacity,
            handle,
        ) = self.experts.permute_and_all_to_all(
            x_moe,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
            batch_size_per_expert=batch_size_per_expert,
        )

        # Compute attention while all-to-all is in progress.
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))

        # Maybe compute MoE shared out while all-to-all is in progress.
        moe_shared_out: Optional[torch.Tensor] = None
        if self.shared_mlp is not None:
            # NOTE: -1 on seq dim in case of TP
            moe_shared_out = self.shared_mlp(model_moe_input.view(B, -1, D))

        handle.wait()
        parallel_x = self.experts.compute_local_experts(
            parallel_x,
            parallel_indices=parallel_indices,
            parallel_bin_ids=parallel_bin_ids,
            parallel_bins=parallel_bins,
            parallel_batch_size_per_expert=parallel_batch_size_per_expert,
            expert_capacity=expert_capacity,
        )

        x_moe, handle = self.experts.reverse_all_to_all(
            parallel_x, send_counts=send_counts, recv_counts=recv_counts
        )

        # Compute feed-forward while all-to-all is in progress.
        h = h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))

        handle.wait()
        x_moe = self.experts.unpermute(
            x_moe,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
        ).view(B, -1, routed_dim)
        x_moe = self.feed_forward_moe.restore_routed_output(x_moe)

        if moe_shared_out is not None:
            moe_shared_out = moe_shared_out / (self.top_k + 1)
            x_moe = moe_shared_out.add(x_moe, alpha=self.top_k / (self.top_k + 1))

        return h + self.dropout(x_moe)


@beta_feature
class MoEHybridReorderedNormTransformerBlock(MoEHybridTransformerBlockBase):
    def dense_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))

    def sparse_forward(
        self, x: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        return self.dropout(
            self.feed_forward_moe_norm(self.feed_forward_moe(x, loss_div_factor=loss_div_factor))
        )

    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # NOTE: this follows the same code path as the MoE's forward pass, except that we run
        # dense operations while we wait on expert parallel all-to-all comms.
        B, _, D = x.shape

        model_moe_input = get_local_tensor(x)
        x_moe = self.feed_forward_moe.prepare_routed_input(model_moe_input)
        routed_dim = x_moe.shape[-1]

        expert_weights, expert_indices, batch_size_per_expert, router_aux_loss = self.router(
            model_moe_input, loss_div_factor=loss_div_factor
        )

        if router_aux_loss is not None:
            x_moe = attach_auxiliary_loss(x_moe, router_aux_loss)

        # shape: (batch_size * seq_len, d_model)
        x_moe = x_moe.view(-1, routed_dim)
        # shape: (batch_size * seq_len * top_k,)
        expert_weights = get_local_tensor(expert_weights).flatten()
        # shape: (batch_size * seq_len * top_k,)
        expert_indices = get_local_tensor(expert_indices).flatten()

        with torch.no_grad():
            indices, bin_ids, bins = self.experts.indices_and_bins(
                expert_indices, batch_size_per_expert
            )

        (
            parallel_x,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            parallel_batch_size_per_expert,
            recv_counts,
            send_counts,
            expert_capacity,
            handle,
        ) = self.experts.permute_and_all_to_all(
            x_moe,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
            batch_size_per_expert=batch_size_per_expert,
        )

        # Compute attention while all-to-all is in progress.
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))

        # Maybe compute MoE shared out while all-to-all is in progress.
        moe_shared_out: Optional[torch.Tensor] = None
        if self.shared_mlp is not None:
            # NOTE: -1 on seq dim in case of TP
            moe_shared_out = self.shared_mlp(model_moe_input.view(B, -1, D))

        handle.wait()
        parallel_x = self.experts.compute_local_experts(
            parallel_x,
            parallel_indices=parallel_indices,
            parallel_bin_ids=parallel_bin_ids,
            parallel_bins=parallel_bins,
            parallel_batch_size_per_expert=parallel_batch_size_per_expert,
            expert_capacity=expert_capacity,
        )

        x_moe, handle = self.experts.reverse_all_to_all(
            parallel_x, send_counts=send_counts, recv_counts=recv_counts
        )

        # Compute feed-forward while all-to-all is in progress.
        h = h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))

        handle.wait()
        x_moe = self.experts.unpermute(
            x_moe,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
        ).view(B, -1, routed_dim)
        x_moe = self.feed_forward_moe.restore_routed_output(x_moe)

        if moe_shared_out is not None:
            moe_shared_out = moe_shared_out / (self.top_k + 1)
            x_moe = moe_shared_out.add(x_moe, alpha=self.top_k / (self.top_k + 1))

        return h + self.dropout(self.feed_forward_moe_norm(x_moe))
