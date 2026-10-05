import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from olmo_core.nn.vision.config import VisionEncoderConfig
from olmo_core.nn.vision.sdpa import vision_scaled_dot_product_attention

__all__ = [
    "ViTAttention",
    "ViTMLP",
    "ViTBlock",
    "VisionTransformer",
    "siglip_state_dict_to_vision_encoder",
]


def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
    # OpenAI CLIP's "QuickGELU"; not available as a PyTorch built-in.
    return x * torch.sigmoid(1.702 * x)


# Activations used by the supported vision encoders, implemented with plain
# PyTorch ops. Names follow the HF convention so configs map directly onto
# HF checkpoint configs.
_ACTIVATIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "quick_gelu": _quick_gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu": F.gelu,
}


def _get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve an activation function by name."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}.")
    return _ACTIVATIONS[name]


def _vit_activation_checkpoint_function(cfg: VisionEncoderConfig) -> Callable:
    preserve_rng_state = (cfg.attention_dropout != 0.0) or (cfg.residual_dropout != 0.0)
    return partial(checkpoint, preserve_rng_state=preserve_rng_state, use_reentrant=False)


class ViTAttention(nn.Module):
    """Multi-head dot-product attention for a vision transformer block."""

    def __init__(self, cfg: VisionEncoderConfig, init_device: str = "cpu"):
        super().__init__()
        self.embed_dim = cfg.image_emb_dim
        self.num_heads = cfg.image_num_heads
        self.head_dim = cfg.image_head_dim
        self.num_kv_heads = cfg.image_num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.initializer_range = cfg.initializer_range
        dtype = cfg.dtype.as_pt()

        self.wq = nn.Linear(
            self.embed_dim,
            self.num_heads * self.head_dim,
            bias=True,
            device=init_device,
            dtype=dtype,
        )
        self.wk = nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=True,
            device=init_device,
            dtype=dtype,
        )
        self.wv = nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=True,
            device=init_device,
            dtype=dtype,
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=True,
            device=init_device,
            dtype=dtype,
        )

        self.attn_drop = nn.Dropout(cfg.attention_dropout) if cfg.attention_dropout > 0 else None
        self.resid_drop = nn.Dropout(cfg.residual_dropout)

    def reset_parameters(self):
        for w in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(w.weight, std=self.initializer_range)
            nn.init.zeros_(w.bias)

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        src = x if kv is None else kv

        q = self.wq(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.wk(src).reshape(B, src.shape[1], self.num_kv_heads, self.head_dim)
        v = self.wv(src).reshape(B, src.shape[1], self.num_kv_heads, self.head_dim)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        out = vision_scaled_dot_product_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            is_causal=False,
            dropout_p=self.attn_drop.p if self.attn_drop is not None and self.training else 0.0,
        ).transpose(1, 2)

        out = out.reshape(B, N, self.embed_dim)
        out = self.wo(out)
        out = self.resid_drop(out)
        return out


class ViTMLP(nn.Module):
    """Two-layer feed-forward network used inside each ViT block."""

    def __init__(self, cfg: VisionEncoderConfig, init_device: str = "cpu"):
        super().__init__()
        dtype = cfg.dtype.as_pt()
        self.w1 = nn.Linear(
            cfg.image_emb_dim, cfg.image_mlp_dim, bias=True, device=init_device, dtype=dtype
        )
        self.w2 = nn.Linear(
            cfg.image_mlp_dim, cfg.image_emb_dim, bias=True, device=init_device, dtype=dtype
        )
        self.act = _get_activation(cfg.image_mlp_activations)
        self._emb_dim = cfg.image_emb_dim
        self._mlp_dim = cfg.image_mlp_dim

    def reset_parameters(self):
        nn.init.trunc_normal_(self.w1.weight, std=math.sqrt(1 / self._emb_dim), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w2.weight, std=math.sqrt(1 / self._mlp_dim), a=-2.0, b=2.0)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class ViTBlock(nn.Module):
    """Standard pre-LN ViT residual block (attention + MLP)."""

    def __init__(self, cfg: VisionEncoderConfig, init_device: str = "cpu"):
        super().__init__()
        dtype = cfg.dtype.as_pt()
        self.attn_norm = nn.LayerNorm(
            cfg.image_emb_dim, eps=cfg.image_norm_eps, device=init_device, dtype=dtype
        )
        self.ffn_norm = nn.LayerNorm(
            cfg.image_emb_dim, eps=cfg.image_norm_eps, device=init_device, dtype=dtype
        )
        self.attn = ViTAttention(cfg, init_device=init_device)
        self.ffn = ViTMLP(cfg, init_device=init_device)

    def reset_parameters(self):
        self.attn_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _interpolate_pos_emb(
    pos_emb: torch.Tensor,  # (num_pos, emb_dim)
    target_patch_num: Tuple[int, int],
    num_prefix: int = 1,
) -> torch.Tensor:
    """Bicubic-interpolate positional embeddings to a different spatial resolution."""
    cls_emb = pos_emb[:num_prefix]  # (num_prefix, emb_dim)
    patch_emb = pos_emb[num_prefix:]  # (H0*W0, emb_dim)

    h0 = w0 = int(math.sqrt(patch_emb.shape[0]))
    h1, w1 = target_patch_num

    if h0 == h1 and w0 == w1:
        return pos_emb

    patch_emb = patch_emb.reshape(1, h0, w0, -1).permute(0, 3, 1, 2)  # (1, D, H0, W0)
    patch_emb = F.interpolate(
        patch_emb, size=(h1, w1), mode="bicubic", align_corners=False, antialias=True
    )
    patch_emb = patch_emb.permute(0, 2, 3, 1).reshape(h1 * w1, -1)  # (H1*W1, D)

    return torch.cat([cls_emb, patch_emb], dim=0)


class VisionTransformer(nn.Module):
    """
    Configurable vision transformer encoder.

    A single implementation that covers the supported encoder families; the
    architectural variant is selected entirely through :class:`VisionEncoderConfig`:

    - :attr:`~VisionEncoderConfig.use_cls_token` — prepend a learnable CLS token
      (CLIP-style) vs. patches only (SigLIP-style).
    - :attr:`~VisionEncoderConfig.patch_embedding_bias` — whether the patch-embedding
      projection has a bias term.
    - :attr:`~VisionEncoderConfig.use_pre_ln` — apply a LayerNorm to the embeddings
      before the transformer blocks (CLIP-style).

    The defaults match OpenAI CLIP ViT-L/14-336; the
    :meth:`VisionEncoderConfig.siglip_so400m` family of factories configures the
    SigLIP variants. Use whichever factory matches your checkpoint.

    The forward pass accepts pre-patchified images and returns the hidden states from
    every block, so callers can select specific layers (e.g. ``[-2, -9]``).

    :param cfg: Vision encoder configuration.
    :param init_device: Device on which to initialise parameters.
    """

    def __init__(self, cfg: VisionEncoderConfig, init_device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        dtype = cfg.dtype.as_pt()

        self.num_prefix_tokens: int = 1 if cfg.use_cls_token else 0

        # Patch embedding: flattened patch pixels → emb_dim.
        patch_pixels = cfg.image_patch_size * cfg.image_patch_size * 3
        self.patch_embedding = nn.Linear(
            patch_pixels,
            cfg.image_emb_dim,
            bias=cfg.patch_embedding_bias,
            device=init_device,
            dtype=dtype,
        )

        # Optional prepended CLS / prefix token (CLIP-style).
        self.class_embedding: Optional[nn.Parameter] = None
        if cfg.use_cls_token:
            self.class_embedding = nn.Parameter(
                torch.zeros(cfg.image_emb_dim, device=init_device, dtype=dtype)
            )

        self.positional_embedding = nn.Parameter(
            torch.zeros(cfg.image_num_pos, cfg.image_emb_dim, device=init_device, dtype=dtype)
        )

        # Optional pre-LayerNorm applied before the blocks (CLIP-style).
        self.pre_ln: Optional[nn.LayerNorm] = None
        if cfg.use_pre_ln:
            self.pre_ln = nn.LayerNorm(
                cfg.image_emb_dim, eps=cfg.image_norm_eps, device=init_device, dtype=dtype
            )

        self.blocks = nn.ModuleList(
            [cfg.block.build(cfg, init_device=init_device) for _ in range(cfg.image_num_layers)]
        )
        self._activation_checkpoint_fn: Optional[Callable] = None

        self.reset_parameters()

    def apply_activation_checkpointing(self) -> None:
        """Per-block activation checkpointing (mm_olmo ``VitConfig.activation_checkpointing``)."""
        self._activation_checkpoint_fn = _vit_activation_checkpoint_function(self.cfg)

    def apply_compile(self) -> None:
        """``torch.compile`` each ViT block (mm_olmo ``compile_vit: blocks``).

        Per-block compilation keeps compile times low thanks to the repeated structure, the
        same strategy :meth:`olmo_core.nn.transformer.Transformer.apply_compile` uses.

        .. warning::
            Call after :meth:`apply_activation_checkpointing` and before FSDP wrapping.
        """
        for idx, block in enumerate(self.blocks):
            self.blocks[idx] = torch.compile(block)  # type: ignore[assignment]

    def reset_parameters(self):
        """Re-initialise all parameters."""
        scale = self.cfg.image_emb_dim**-0.5
        if self.class_embedding is not None:
            nn.init.normal_(self.class_embedding, std=scale)
        nn.init.normal_(self.positional_embedding, std=scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)
        if self.pre_ln is not None:
            self.pre_ln.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()

    def _add_pos_emb(self, x: torch.Tensor, patch_num: Tuple[int, int]) -> torch.Tensor:
        pos_emb = _interpolate_pos_emb(
            self.positional_embedding, patch_num, num_prefix=self.num_prefix_tokens
        )
        return x + pos_emb[None, :, :].to(x.dtype)

    def forward(
        self, x: torch.Tensor, patch_num: Optional[Tuple[int, int]] = None
    ) -> List[torch.Tensor]:
        """
        Encode pre-patchified images.

        :param x: Float tensor of shape ``(batch, n_patches, patch_size**2 * 3)``.
        :param patch_num: Spatial grid ``(H, W)`` of patches. Defaults to
            ``cfg.image_num_patch``.
        :returns: List of length ``image_num_layers``, each element a tensor of shape
            ``(batch, num_prefix_tokens + n_patches, image_emb_dim)``.
        """
        if patch_num is None:
            patch_num = self.cfg.image_num_patch

        B, N, _ = x.shape
        x = self.patch_embedding(x)

        # Prepend CLS / prefix token when configured.
        if self.class_embedding is not None:
            cls = self.class_embedding[None, None, :].expand(B, 1, -1).to(x.dtype)
            x = torch.cat([cls, x], dim=1)  # (B, num_prefix + N, D)

        x = self._add_pos_emb(x, patch_num)

        if self.pre_ln is not None:
            x = self.pre_ln(x)

        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(block, x)
            else:
                x = block(x)
            hidden_states.append(x)
        return hidden_states


def siglip_state_dict_to_vision_encoder(
    hf_state: Dict[str, torch.Tensor],
    *,
    n_blocks: Optional[int] = None,
    prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """
    Map a HuggingFace SigLIP / SigLIP2 vision-tower state dict onto
    :class:`VisionTransformer` parameter names.

    Accepts either a ``SiglipVisionTransformer`` state dict or a ``SiglipVisionModel``
    one (whose keys carry a ``vision_model.`` prefix, stripped automatically).
    ``post_layernorm`` and ``head.*`` are skipped: our encoder returns per-block hidden
    states and does not model SigLIP's attention-pooling head.

    :param hf_state: The HF vision-tower state dict.
    :param n_blocks: Number of transformer blocks to emit. Defaults to every block found
        in ``hf_state``. Pass a smaller value to load into a truncated encoder — e.g.
        Molmo2 keeps blocks ``0..24`` of SigLIP2-SO400M's 27.
    :param prefix: Prepended to every output key, e.g. ``"vision."`` when loading into a
        :class:`~olmo_core.nn.vision.MultimodalLM` rather than a bare encoder.

    :returns: A state dict suitable for ``load_state_dict``.

    :raises KeyError: If an expected SigLIP key is absent.
    """
    hf_state = {k.removeprefix("vision_model."): v for k, v in hf_state.items()}

    available = (
        max((int(k.split(".")[2]) for k in hf_state if k.startswith("encoder.layers.")), default=-1)
        + 1
    )
    if n_blocks is None:
        n_blocks = available
    elif n_blocks > available:
        raise KeyError(f"requested {n_blocks} blocks but the state dict only has {available}")

    patch_w = hf_state["embeddings.patch_embedding.weight"]
    out: Dict[str, torch.Tensor] = {
        # Conv2d (D, 3, p, p) -> our linear projection (D, 3 * p * p), C-first flatten.
        f"{prefix}patch_embedding.weight": patch_w.reshape(patch_w.shape[0], -1),
        f"{prefix}patch_embedding.bias": hf_state["embeddings.patch_embedding.bias"],
        f"{prefix}positional_embedding": hf_state["embeddings.position_embedding.weight"],
    }
    for i in range(n_blocks):
        src, dst = f"encoder.layers.{i}", f"{prefix}blocks.{i}"
        for hf_name, ours in (("layer_norm1", "attn_norm"), ("layer_norm2", "ffn_norm")):
            for suffix in ("weight", "bias"):
                out[f"{dst}.{ours}.{suffix}"] = hf_state[f"{src}.{hf_name}.{suffix}"]
        for hf_name, ours in (
            ("q_proj", "wq"),
            ("k_proj", "wk"),
            ("v_proj", "wv"),
            ("out_proj", "wo"),
        ):
            for suffix in ("weight", "bias"):
                out[f"{dst}.attn.{ours}.{suffix}"] = hf_state[f"{src}.self_attn.{hf_name}.{suffix}"]
        for hf_name, ours in (("fc1", "w1"), ("fc2", "w2")):
            for suffix in ("weight", "bias"):
                out[f"{dst}.ffn.{ours}.{suffix}"] = hf_state[f"{src}.mlp.{hf_name}.{suffix}"]
    return out
