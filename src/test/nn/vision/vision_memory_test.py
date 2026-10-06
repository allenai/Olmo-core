"""Memory-related vision tests (activation checkpointing, SDPA backend)."""

import torch

from olmo_core.nn.vision.config import VisionEncoderConfig
from olmo_core.nn.vision.image_vit import VisionTransformer
from olmo_core.nn.vision.sdpa import (
    vision_scaled_dot_product_attention,
    vision_sdpa_context,
)


def test_vit_apply_activation_checkpointing_sets_fn():
    cfg = VisionEncoderConfig.siglip_b16_224()
    cfg.image_num_layers = 2
    model = cfg.build(init_device="cpu")
    assert isinstance(model, VisionTransformer)
    assert model._activation_checkpoint_fn is None
    model.apply_activation_checkpointing()
    assert model._activation_checkpoint_fn is not None


def test_vision_sdpa_runs_on_cpu():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    with vision_sdpa_context():
        out = vision_scaled_dot_product_attention(q, k, v)
    assert out.shape == q.shape


def test_vit_forward_with_activation_checkpointing():
    cfg = VisionEncoderConfig.siglip_b16_224()
    cfg.image_num_layers = 2
    model = cfg.build(init_device="cpu")
    model.apply_activation_checkpointing()
    model.train()
    n_patches = cfg.image_num_patch[0] * cfg.image_num_patch[1]
    x = torch.randn(2, n_patches, cfg.image_patch_size**2 * 3)
    hidden = model(x)
    assert len(hidden) == 2
    loss = hidden[-1].sum()
    loss.backward()
