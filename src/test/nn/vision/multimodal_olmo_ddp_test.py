"""
Tests for :class:`~olmo_core.nn.vision.MultimodalOLMoDDPModel`: it is built for an OLMoDDP
language model, keeps the plain multimodal forward, and computes the float-weighted response-only
objective through the language model's ``input_embeddings`` / ``router_loss_div_factor`` hooks.
"""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.functional import weighted_cross_entropy_loss
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMOutputWithLoss
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.nn.vision import (
    MultimodalLM,
    MultimodalLMConfig,
    MultimodalOLMoDDPModel,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)

_D_MODEL = 16
_VOCAB = 64
_IMAGE_PATCH_TOKEN = 1


def _lm_config() -> OLMoDDPModelConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=DType.float32)
    block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=2,
            bias=False,
            backend=AttentionBackendName.torch,
            dtype=DType.float32,
        ),
        layer_norm=layer_norm,
        routed_experts=None,
        routed_experts_router=None,
        shared_experts=SharedExpertsConfig(
            d_model=_D_MODEL, hidden_size=32, num_experts=1, bias=False, dtype=DType.float32
        ),
        shared_experts_router=None,
    )
    return OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=_D_MODEL,
        vocab_size=_VOCAB,
        n_layers=2,
        lm_head=LMHeadConfig(bias=False, dtype=DType.float32),
        block=block,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
    )


def _vision_config() -> VisionEncoderConfig:
    return VisionEncoderConfig(
        name=VisionEncoderType.openai,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=32,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=64,
        image_num_pos=5,
        image_norm_eps=1e-5,
    )


def _config() -> MultimodalLMConfig:
    vision = _vision_config()
    return MultimodalLMConfig(
        lm=_lm_config(),
        vision=vision,
        connector=VisionConnectorConfig.from_vision_encoder(
            vision, output_dim=_D_MODEL, mlp_hidden_size=32
        ),
        image_patch_token_id=_IMAGE_PATCH_TOKEN,
    )


def _model() -> MultimodalOLMoDDPModel:
    torch.manual_seed(0)
    model = _config().build(init_device="cpu")
    assert isinstance(model, MultimodalOLMoDDPModel)
    model.init_weights(max_seq_len=8, device=torch.device("cpu"))
    return model.eval()


def _text_batch(batch: int = 2, seq_len: int = 8):
    torch.manual_seed(1)
    input_ids = torch.randint(2, _VOCAB, (batch, seq_len))
    labels = torch.randint(2, _VOCAB, (batch, seq_len))
    loss_masks = torch.zeros(batch, seq_len)
    loss_masks[0, 2:6] = 1.0
    loss_masks[1, 1:3] = 0.5
    loss_masks[1, 5] = 2.0
    labels[loss_masks == 0] = -100
    return input_ids, labels, loss_masks


def _image_batch(batch: int = 2, seq_len: int = 8):
    input_ids = torch.randint(2, _VOCAB, (batch, seq_len))
    input_ids[:, 0] = _IMAGE_PATCH_TOKEN  # one pooled feature per sequence (4 patches / pool 4)
    images = torch.randn(batch, 1, 4, 14 * 14 * 3)
    pooled = torch.arange(4, dtype=torch.long).view(1, 1, 4).expand(batch, -1, -1).contiguous()
    return input_ids, images, pooled


def test_config_builds_the_olmo_ddp_wrapper_for_an_olmo_ddp_lm():
    model = _model()
    assert isinstance(model, MultimodalLM)
    assert model._olmo_ddp_compatible is True
    assert model.is_moe is model.lm.is_moe
    assert model.tbo is False and model.recompute_each_block is False
    assert model.device == model.lm.device
    assert isinstance(model.vision.parameters().__next__(), torch.nn.Parameter)


def test_forward_without_labels_is_the_plain_multimodal_forward():
    model = _model()
    input_ids, images, pooled = _image_batch()
    with torch.no_grad():
        logits = model(input_ids, images=images, pooled_patches_idx=pooled)
        expected = MultimodalLM.forward(model, input_ids, images=images, pooled_patches_idx=pooled)
    assert logits.shape == (2, 8, _VOCAB)
    torch.testing.assert_close(logits, expected)


def test_weighted_loss_matches_the_reference_on_response_positions():
    model = _model()
    input_ids, labels, loss_masks = _text_batch()
    divisor = torch.tensor(3.5)
    out = model(
        input_ids,
        labels=labels,
        loss_masks=loss_masks,
        loss_reduction="sum",
        z_loss_multiplier=1e-4,
        loss_div_factor=torch.tensor(99.0),  # the LM-only divisor is superseded
        loss_weight_div_factor=divisor,
        return_logits=True,
    )
    assert isinstance(out, LMOutputWithLoss)
    with torch.no_grad():
        full_logits = model(input_ids)
    mask = loss_masks > 0
    ce, z = weighted_cross_entropy_loss(
        full_logits[mask],
        labels[mask],
        loss_masks[mask],
        compute_z_loss=True,
        z_loss_multiplier=1e-4,
    )
    assert out.logits is not None and out.logits.shape == (int(mask.sum()), _VOCAB)
    torch.testing.assert_close(out.ce_loss, ce / divisor)
    assert out.z_loss is not None
    torch.testing.assert_close(out.z_loss, z / divisor)
    torch.testing.assert_close(out.loss, (ce + z) / divisor)
    # Gradients flow into the language model through the embeddings hook.
    out.loss.backward()
    assert model.lm.embeddings.weight.grad is not None
    assert model.lm.embeddings.weight.grad.abs().sum() > 0


def test_loss_falls_back_to_the_lm_divisor_and_skips_z_loss_when_unset():
    model = _model()
    input_ids, labels, loss_masks = _text_batch()
    with torch.no_grad():
        out = model(
            input_ids,
            labels=labels,
            loss_masks=loss_masks,
            loss_reduction="sum",
            loss_div_factor=torch.tensor(7.0),
        )
        full_logits = model(input_ids)
    mask = loss_masks > 0
    ce, _ = weighted_cross_entropy_loss(full_logits[mask], labels[mask], loss_masks[mask])
    torch.testing.assert_close(out.ce_loss, ce / 7.0)
    assert out.z_loss is None and out.logits is None
    torch.testing.assert_close(out.loss, out.ce_loss)


def test_text_only_labels_use_the_plain_lm_loss_and_the_router_mask_is_dropped():
    model = _model()
    input_ids, labels, loss_masks = _text_batch()
    # Labels without loss masks (a text-only batch) take the language model's plain loss path.
    with torch.no_grad():
        text_only = model(input_ids, labels=labels, loss_reduction="none", return_logits=True)
        logits = model(input_ids)
    assert isinstance(text_only, LMOutputWithLoss)
    assert text_only.ce_loss.shape == (2, 8) and text_only.logits is not None
    reference = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, _VOCAB), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape(2, 8)
    torch.testing.assert_close(text_only.ce_loss.float(), reference, rtol=2e-2, atol=2e-2)
    with pytest.raises(ValueError, match="loss_reduction"):
        model(input_ids, labels=labels, loss_masks=loss_masks, loss_reduction="mean")
    with torch.no_grad():
        plain = model(input_ids, labels=labels, loss_masks=loss_masks, loss_reduction="sum")
        masked = model(
            input_ids,
            labels=labels,
            loss_masks=loss_masks,
            loss_reduction="sum",
            router_token_mask=torch.zeros_like(input_ids, dtype=torch.bool),
        )
    torch.testing.assert_close(plain.loss, masked.loss)


def test_router_divisor_reaches_the_language_model_blocks(monkeypatch):
    model = _model()
    input_ids, labels, loss_masks = _text_batch()
    seen = {}
    original = model.lm._forward_blocks

    def spy(h, all_block_kwargs, per_block_kwargs):
        seen["block"] = all_block_kwargs.get("loss_div_factor")
        return original(h, all_block_kwargs, per_block_kwargs)

    monkeypatch.setattr(model.lm, "_forward_blocks", spy)
    with torch.no_grad():
        model(
            input_ids,
            labels=labels,
            loss_masks=loss_masks,
            loss_reduction="sum",
            loss_weight_div_factor=torch.tensor(3.5),
            router_loss_div_factor=torch.tensor(16.0),
        )
    assert float(seen["block"]) == 16.0


def test_encoded_image_features_reproduce_the_direct_forward():
    model = _model()
    input_ids, images, pooled = _image_batch()
    with torch.no_grad():
        direct = model(input_ids, images=images, pooled_patches_idx=pooled)
        features = model.encode_images(images, pooled)
        cached = model(input_ids, encoded_image_features=features)
    assert features.shape == (2, _D_MODEL)
    torch.testing.assert_close(cached, direct)
    with pytest.raises(ValueError, match="not both"):
        model(input_ids, images=images, pooled_patches_idx=pooled, encoded_image_features=features)


def test_encode_images_casts_pixels_to_the_vision_tower_dtype():
    # The OLMoDDP train module casts the whole wrapped model to bf16 while the collator
    # delivers float32 pixels (the first bridge smoke failed on exactly this matmul).
    model = _model().to(torch.bfloat16)
    input_ids, images, pooled = _image_batch()
    assert images.dtype == torch.float32
    with torch.no_grad():
        features = model.encode_images(images, pooled)
        logits = model(input_ids, images=images, pooled_patches_idx=pooled)
    assert features.dtype == torch.bfloat16 and features.shape == (2, _D_MODEL)
    assert logits.shape[0] == 2 and torch.isfinite(logits.float()).all()


def test_frozen_vision_tower_stays_in_eval_mode_during_training():
    model = _model()
    for param in model.vision.parameters():
        param.requires_grad_(False)
    model.train()
    assert model.training and not model.vision.training and model.connector.training
    for param in model.vision.parameters():
        param.requires_grad_(True)
    model.train()
    assert model.vision.training
