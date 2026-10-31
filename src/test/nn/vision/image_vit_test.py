import copy
import dataclasses

import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.vision import (
    VisionEncoderConfig,
    VisionEncoderType,
    VisionTransformer,
)
from olmo_core.nn.vision.image_vit import ViTAttention, ViTBlock
from olmo_core.testing import requires_gpu

# ---------------------------------------------------------------------------
# Tiny configs for fast CPU tests
# ---------------------------------------------------------------------------


def _tiny_clip_cfg(**kwargs) -> VisionEncoderConfig:
    """Minimal CLIP-style config: 2 layers, small dims."""
    return VisionEncoderConfig(
        name=VisionEncoderType.openai,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=64,
        image_num_heads=4,
        image_num_key_value_heads=4,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=128,
        image_mlp_activations="quick_gelu",
        image_num_pos=5,  # 2*2 patches + 1 CLS
        **kwargs,
    )


def _tiny_siglip_cfg(**kwargs) -> VisionEncoderConfig:
    """Minimal SigLIP-style config: 2 layers, small dims."""
    return VisionEncoderConfig(
        name=VisionEncoderType.siglip,
        use_cls_token=False,
        patch_embedding_bias=True,
        use_pre_ln=False,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=64,
        image_num_heads=4,
        image_num_key_value_heads=4,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=128,
        image_mlp_activations="gelu_pytorch_tanh",
        image_num_pos=4,  # 2*2 patches, no CLS
        **kwargs,
    )


# ---------------------------------------------------------------------------
# VisionTransformer (CLIP-style)
#
# NOTE: config-layer concerns (factory fields, build() dispatch, CLS-token
# presence, image_num_patch, serialization round-trip) live in config_test.py.
# These tests cover only forward-pass behavior of the encoder module.
# ---------------------------------------------------------------------------


class TestVisionTransformer:
    def setup_method(self):
        self.cfg = _tiny_clip_cfg()
        self.vit = VisionTransformer(self.cfg, init_device="cpu")

    def _make_input(self, batch: int = 2) -> torch.Tensor:
        n_patches = 2 * 2  # 28px / 14px = 2 per side
        patch_pixels = 14 * 14 * 3
        return torch.randn(batch, n_patches, patch_pixels)

    def test_forward_returns_list_of_length_num_layers(self):
        x = self._make_input()
        out = self.vit(x)
        assert len(out) == self.cfg.image_num_layers

    def test_output_shape_includes_cls_token(self):
        batch = 3
        x = self._make_input(batch)
        out = self.vit(x)
        n_patches = 2 * 2
        expected = (batch, 1 + n_patches, self.cfg.image_emb_dim)
        for layer_out in out:
            assert layer_out.shape == expected, f"Got {layer_out.shape}, expected {expected}"

    def test_forward_deterministic(self):
        x = self._make_input()
        self.vit.eval()
        out1 = self.vit(x)
        out2 = self.vit(x)
        assert torch.allclose(out1[-1], out2[-1])

    def test_patch_num_override(self):
        """Positional embeddings should interpolate to a different resolution."""
        cfg = _tiny_clip_cfg()
        cfg.image_default_input_size = (42, 42)  # 3x3 patches
        cfg.image_num_pos = 5  # still sized for 2x2 — must interpolate
        vit = VisionTransformer(cfg, init_device="cpu")

        n_patches = 3 * 3
        patch_pixels = 14 * 14 * 3
        x = torch.randn(1, n_patches, patch_pixels)
        out = vit(x, patch_num=(3, 3))
        assert out[-1].shape == (1, 1 + n_patches, cfg.image_emb_dim)

    def test_meta_device_build(self):
        vit = VisionTransformer(_tiny_clip_cfg(), init_device="meta")
        assert next(iter(vit.parameters())).device.type == "meta"

    @pytest.mark.parametrize("batch", [1, 4])
    def test_various_batch_sizes(self, batch: int):
        x = self._make_input(batch)
        out = self.vit(x)
        assert out[-1].shape[0] == batch


# ---------------------------------------------------------------------------
# SiglipVisionTransformer
# ---------------------------------------------------------------------------


class TestSiglipVisionTransformer:
    def setup_method(self):
        self.cfg = _tiny_siglip_cfg()
        self.vit = VisionTransformer(self.cfg, init_device="cpu")

    def _make_input(self, batch: int = 2) -> torch.Tensor:
        n_patches = 2 * 2
        patch_pixels = 14 * 14 * 3
        return torch.randn(batch, n_patches, patch_pixels)

    def test_forward_returns_list_of_length_num_layers(self):
        x = self._make_input()
        out = self.vit(x)
        assert len(out) == self.cfg.image_num_layers

    def test_output_shape_no_cls_token(self):
        batch = 2
        x = self._make_input(batch)
        out = self.vit(x)
        n_patches = 2 * 2
        expected = (batch, n_patches, self.cfg.image_emb_dim)
        for layer_out in out:
            assert layer_out.shape == expected

    def test_meta_device_build(self):
        vit = VisionTransformer(_tiny_siglip_cfg(), init_device="meta")
        assert next(iter(vit.parameters())).device.type == "meta"


# ---------------------------------------------------------------------------
# Block internals: fused self-attention projection and LayerNorm
# ---------------------------------------------------------------------------


def _separate_projection_attention(attn: ViTAttention, x: torch.Tensor) -> torch.Tensor:
    """``attn`` with separate q, k and v projections and explicit softmax attention."""
    B, N, _ = x.shape
    q, k, v = (
        proj(x).reshape(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
        for proj in (attn.wq, attn.wk, attn.wv)
    )
    probs = (q @ k.transpose(-1, -2) * attn.head_dim**-0.5).softmax(-1)
    return attn.wo((probs @ v).transpose(1, 2).reshape(B, N, attn.embed_dim))


def test_self_attention_matches_separate_projections():
    torch.manual_seed(0)
    attn = ViTAttention(_tiny_siglip_cfg())
    attn.reset_parameters()
    x = torch.randn(3, 4, 64, requires_grad=True)
    params = [x, *attn.parameters()]
    out = attn(x)
    ref = _separate_projection_attention(attn, x)
    torch.testing.assert_close(out, ref)
    for a, b in zip(
        torch.autograd.grad(out.square().sum(), params),
        torch.autograd.grad(ref.square().sum(), params),
    ):
        torch.testing.assert_close(a, b)


@requires_gpu
def test_self_attention_is_as_accurate_as_the_unfused_path():
    """At SigLIP's sizes the fused path (cuDNN first on SM100) is as close to float32 as the
    separate projections on flash or memory-efficient attention (``kv=x`` takes that path)."""
    torch.manual_seed(0)
    cfg = dataclasses.replace(
        _tiny_siglip_cfg(),
        image_emb_dim=1152,
        image_num_heads=16,
        image_num_key_value_heads=16,
        image_head_dim=72,
    )
    attn = ViTAttention(cfg, init_device="cuda")
    attn.reset_parameters()
    reference = copy.deepcopy(attn)
    attn = attn.to(torch.bfloat16)
    x = torch.randn(4, 729, 1152, device="cuda")
    ref = _separate_projection_attention(reference, x)
    with torch.no_grad():
        fused = attn(x.bfloat16()).float()
        unfused = attn(x.bfloat16(), kv=x.bfloat16()).float()
    fused_error = (fused - ref).abs().mean()
    unfused_error = (unfused - ref).abs().mean()
    assert fused_error <= 1.25 * unfused_error


def test_block_layer_norms_are_torch_layer_norms():
    block = ViTBlock(_tiny_siglip_cfg())
    assert isinstance(block.attn_norm, torch.nn.LayerNorm)
    assert isinstance(block.ffn_norm, torch.nn.LayerNorm)
    assert {"attn_norm.weight", "attn_norm.bias", "ffn_norm.weight", "ffn_norm.bias"} <= set(
        block.state_dict()
    )
    x = torch.randn(2, 4, 64)
    norm = block.attn_norm
    torch.testing.assert_close(norm(x), F.layer_norm(x, (64,), norm.weight, norm.bias, norm.eps))
