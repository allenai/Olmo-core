from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from olmo_core.config import DType, StrEnum
from olmo_core.nn.config import ModuleConfig
from olmo_core.nn.vision.config import VisionEncoderConfig
from olmo_core.nn.vision.sdpa import vision_scaled_dot_product_attention

__all__ = [
    "ImagePoolingType",
    "ImageProjectorType",
    "VisionConnectorConfig",
    "VisionConnector",
]


class ImagePoolingType(StrEnum):
    """
    How to pool patch groups before projection.

    Each pool group is defined by the data preprocessor's ``pooled_patches_idx``
    tensor (a per-group list of patch indices, with ``-1`` marking padded
    slots). The same pool size is applied to every group in a batch.
    """

    attention_meanq = "attention_meanq"
    """
    For each group, use the mean of its patch features as the query and
    cross-attend over all patches in the group as keys/values. With a 4-patch
    group it reduces patch count 4×.
    """

    none = "none"
    """
    No pooling — pass the gathered features directly to the projector. ``pool_size``
    must be 1 for this mode.
    """


class ImageProjectorType(StrEnum):
    """
    How to project pooled vision features into the LM embedding space.
    """

    mlp = "mlp"
    """
    SwiGLU two-stream MLP: ``w2(silu(w1(x)) * w3(x))``.
    """

    linear = "linear"
    """
    Single bias-free linear layer.
    """


class _PoolingCrossAttention(nn.Module):
    """Cross-attention used for patch-group pooling.

    Q is the mean of the group's patches; K/V are all patches in the group.
    Maps ``num_input_layers * image_emb_dim`` → ``image_emb_dim``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        emb_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        initializer_range: float = 0.02,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.emb_dim = emb_dim
        self.initializer_range = initializer_range

        self.wq = nn.Linear(
            input_dim, num_heads * head_dim, bias=True, device=init_device, dtype=dtype
        )
        self.wk = nn.Linear(
            input_dim, num_kv_heads * head_dim, bias=True, device=init_device, dtype=dtype
        )
        self.wv = nn.Linear(
            input_dim, num_kv_heads * head_dim, bias=True, device=init_device, dtype=dtype
        )
        self.wo = nn.Linear(
            num_heads * head_dim, emb_dim, bias=True, device=init_device, dtype=dtype
        )

    def reset_parameters(self):
        for w in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(w.weight, std=self.initializer_range)
            nn.init.zeros_(w.bias)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param query: ``(B, 1, input_dim)`` — mean of the group's patch features.
        :param context: ``(B, pool_size, input_dim)`` — all patches in the group.
        :param attn_mask: Optional bool mask ``(B, 1, 1, pool_size)`` where ``True``
            allows attention. Used to ignore padded patches (where the
            ``pooled_patches_idx`` entry was ``-1``).
        :returns: ``(B, 1, emb_dim)``
        """
        B = query.shape[0]
        q = self.wq(query).reshape(B, 1, self.num_heads, self.head_dim)
        k = self.wk(context).reshape(B, context.shape[1], self.num_kv_heads, self.head_dim)
        v = self.wv(context).reshape(B, context.shape[1], self.num_kv_heads, self.head_dim)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        out = vision_scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=False,
        ).transpose(
            1, 2
        )  # (B, 1, num_heads * head_dim)

        out = out.reshape(B, 1, self.num_heads * self.head_dim)
        return self.wo(out)


class _ConnectorMLP(nn.Module):
    """SwiGLU MLP that projects vision features to LM embedding dimension.

    Computes ``w2(silu(w1(x)) * w3(x))``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        initializer_range: float = 0.02,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.initializer_range = initializer_range
        # SwiGLU: two parallel gates → one output
        self.w1 = nn.Linear(input_dim, hidden_size, bias=False, device=init_device, dtype=dtype)
        self.w3 = nn.Linear(input_dim, hidden_size, bias=False, device=init_device, dtype=dtype)
        self.w2 = nn.Linear(hidden_size, output_dim, bias=False, device=init_device, dtype=dtype)

    def reset_parameters(self):
        for w in (self.w1, self.w2, self.w3):
            nn.init.normal_(w.weight, std=self.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@dataclass
class VisionConnectorConfig(ModuleConfig):
    """
    Configuration for the vision-to-language connector.

    The connector sits between the vision encoder and the LM backbone. For each
    group described by ``pooled_patches_idx`` it gathers a fixed-size patch
    set, applies attention pooling, and projects the result into the LM's
    embedding space.

    Pool size and spatial layout are determined entirely by the data
    preprocessor (which builds ``pooled_patches_idx``). The connector only
    needs to know the pooling and projection style.

    Use :meth:`from_vision_encoder` to build a config from a
    :class:`~olmo_core.nn.vision.VisionEncoderConfig` and an LM ``d_model``.
    """

    image_emb_dim: int = 1024
    """Vision encoder hidden dimension (from :attr:`VisionEncoderConfig.image_emb_dim`)."""

    image_num_heads: int = 16
    """Number of attention heads for pooling cross-attention."""

    image_num_key_value_heads: int = 16
    """Number of KV heads for pooling cross-attention."""

    image_head_dim: int = 64
    """Per-head dimension for pooling cross-attention."""

    output_dim: int = 4096
    """LM ``d_model`` — target dimension after projection."""

    num_input_layers: int = 1
    """
    Number of ViT layer outputs concatenated before pooling (e.g. ``2`` for
    ``vit_layers=[-2, -9]``). The pooling cross-attention's input dimension
    becomes ``num_input_layers * image_emb_dim``.
    """

    pooling_type: ImagePoolingType = ImagePoolingType.attention_meanq
    """Pooling strategy applied before projection."""

    pooling_attention_mask: bool = False
    """
    If ``True``, mask out patches with ``pooled_patches_idx == -1`` from the
    pooling attention so they don't contribute. Recommended whenever the data
    preprocessor pads groups (e.g. multi-crop with non-aligned grids).
    """

    projector_type: ImageProjectorType = ImageProjectorType.mlp
    """Projection architecture."""

    mlp_hidden_size: Optional[int] = None
    """
    Hidden size for the SwiGLU MLP projector. Defaults to
    ``4 * image_emb_dim`` when ``None``.
    """

    initializer_range: float = 0.02
    """Standard deviation for normal-distribution weight initialisation."""

    dtype: DType = DType.float32
    """Default parameter dtype."""

    @classmethod
    def from_vision_encoder(
        cls,
        vision_cfg: VisionEncoderConfig,
        output_dim: int,
        num_input_layers: int = 1,
        **kwargs,
    ) -> "VisionConnectorConfig":
        """
        Convenience factory that copies attention hyperparameters from a
        :class:`~olmo_core.nn.vision.VisionEncoderConfig`.

        :param vision_cfg: Vision encoder configuration.
        :param output_dim: LM ``d_model``.
        :param num_input_layers: Number of ViT layer outputs to concatenate
            before pooling (default 1 = last layer only).
        """
        return cls(
            image_emb_dim=vision_cfg.image_emb_dim,
            image_num_heads=vision_cfg.image_num_heads,
            image_num_key_value_heads=vision_cfg.image_num_key_value_heads,
            image_head_dim=vision_cfg.image_head_dim,
            output_dim=output_dim,
            num_input_layers=num_input_layers,
            dtype=vision_cfg.dtype,
            **kwargs,
        )

    @property
    def pooling_input_dim(self) -> int:
        """Input dim to pooling cross-attention = ``num_input_layers * image_emb_dim``."""
        return self.num_input_layers * self.image_emb_dim

    @property
    def projector_input_dim(self) -> int:
        """Input dim to the MLP projector.

        Equals ``image_emb_dim`` after pooling, or ``pooling_input_dim`` when
        :attr:`pooling_type` is ``none``.
        """
        if self.pooling_type == ImagePoolingType.none:
            return self.pooling_input_dim
        return self.image_emb_dim

    def build(self, init_device: str = "cpu") -> "VisionConnector":
        """
        Instantiate the connector on ``init_device``.

        :param init_device: Device string (e.g. ``"cpu"``, ``"meta"``).
        :returns: A :class:`VisionConnector`.
        """
        return VisionConnector(self, init_device=init_device)


class VisionConnector(nn.Module):
    """
    Vision-to-language connector: gather-based group pooling + projection.

    Each group of patches (defined by a row of ``pooled_patches_idx``) is
    gathered from the per-image patch features, optionally attention-pooled
    into a single token, and projected into LM embedding space.

    :param cfg: Connector configuration.
    :param init_device: Device on which to initialise parameters.
    """

    def __init__(self, cfg: VisionConnectorConfig, init_device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        dtype = cfg.dtype.as_pt()

        self.pooling: Optional[_PoolingCrossAttention] = None
        if cfg.pooling_type == ImagePoolingType.attention_meanq:
            self.pooling = _PoolingCrossAttention(
                input_dim=cfg.pooling_input_dim,
                emb_dim=cfg.image_emb_dim,
                num_heads=cfg.image_num_heads,
                num_kv_heads=cfg.image_num_key_value_heads,
                head_dim=cfg.image_head_dim,
                initializer_range=cfg.initializer_range,
                dtype=dtype,
                init_device=init_device,
            )

        proj_in = cfg.projector_input_dim
        if cfg.projector_type == ImageProjectorType.mlp:
            hidden = cfg.mlp_hidden_size or 4 * cfg.image_emb_dim
            self.projector: nn.Module = _ConnectorMLP(
                input_dim=proj_in,
                output_dim=cfg.output_dim,
                hidden_size=hidden,
                initializer_range=cfg.initializer_range,
                dtype=dtype,
                init_device=init_device,
            )
        elif cfg.projector_type == ImageProjectorType.linear:
            self.projector = nn.Linear(
                proj_in, cfg.output_dim, bias=False, device=init_device, dtype=dtype
            )
        else:
            raise NotImplementedError(f"Unsupported projector type: {cfg.projector_type}")

        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialise all parameters."""
        if self.pooling is not None:
            self.pooling.reset_parameters()
        if isinstance(self.projector, _ConnectorMLP):
            self.projector.reset_parameters()
        elif isinstance(self.projector, nn.Linear):
            nn.init.normal_(self.projector.weight, std=self.cfg.initializer_range)

    def apply_activation_checkpointing(self) -> None:
        """Checkpoint pooling + projector (mm_olmo ``connector_activation_checkpointing``)."""
        if self.pooling is not None:
            self.pooling = checkpoint_wrapper(self.pooling)
        self.projector = checkpoint_wrapper(self.projector)

    def apply_compile(self) -> None:
        """``torch.compile`` pooling + projector (mm_olmo ``compile_connector: dynamic``).

        Compiled with ``dynamic=True``: the number of pooled groups varies per batch (it
        follows the crop count), so a static compile would recompile on every new shape.

        .. warning::
            Call after :meth:`apply_activation_checkpointing` and before FSDP wrapping.
        """
        if self.pooling is not None:
            self.pooling = torch.compile(self.pooling, dynamic=True)  # type: ignore[assignment]
        self.projector = torch.compile(self.projector, dynamic=True)  # type: ignore[assignment]

    def forward(
        self,
        image_features: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool and project patch features by gathering through ``pooled_patches_idx``.

        :param image_features: Float tensor of shape
            ``(B, n_total_patches, pooling_input_dim)``. ``n_total_patches``
            covers all crops for each batch element flattened together
            (e.g. ``n_crops * n_patches_per_crop``).
        :param pooled_patches_idx: Long tensor of shape
            ``(B, n_pooled, pool_size)``. Each row lists ``pool_size`` indices
            into ``image_features`` (along the second axis) that should be
            pooled into a single output token. Entries equal to ``-1`` mark
            padding within a group and are zeroed out (and masked from the
            pooling attention if :attr:`cfg.pooling_attention_mask` is set).
        :returns: ``(B, n_pooled, output_dim)``.
        """
        cfg = self.cfg
        B, _, dim = image_features.shape
        n_pooled, pool_size = pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]
        valid = pooled_patches_idx >= 0

        # Gather pool_size patches per group.
        batch_idx = (
            torch.arange(B, device=pooled_patches_idx.device)
            .view(B, 1, 1)
            .expand(B, n_pooled, pool_size)
        )
        # (B, n_pooled, pool_size, dim)
        to_pool = image_features[batch_idx, pooled_patches_idx.clamp(min=0)]
        # Zero out features at padded slots (idx == -1).
        to_pool = to_pool * valid.unsqueeze(-1).to(to_pool.dtype)

        if cfg.pooling_type == ImagePoolingType.attention_meanq:
            assert self.pooling is not None
            # Flatten group dim into the cross-attention batch.
            flat = to_pool.reshape(B * n_pooled, pool_size, dim)
            attn_mask: Optional[torch.Tensor] = None
            if cfg.pooling_attention_mask:
                attn_mask = valid.reshape(B * n_pooled, 1, 1, pool_size)
                # Build query as masked mean over valid slots only.
                denom = valid.reshape(B * n_pooled, pool_size).sum(-1).clamp(min=1).unsqueeze(-1)
                query = flat.sum(dim=1, keepdim=True) / denom.unsqueeze(-1).to(flat.dtype)
            else:
                query = flat.mean(dim=1, keepdim=True)
            pooled = self.pooling(query, flat, attn_mask=attn_mask)  # (B*n_pooled, 1, emb_dim)
            pooled = pooled.reshape(B, n_pooled, cfg.image_emb_dim)
        elif cfg.pooling_type == ImagePoolingType.none:
            # No pooling: pool_size must be 1, just squeeze.
            assert pool_size == 1, "pool_size must be 1 when pooling_type=none"
            pooled = to_pool.reshape(B, n_pooled, dim)
        else:
            raise NotImplementedError(f"Unsupported pooling type: {cfg.pooling_type}")

        return self.projector(pooled)
