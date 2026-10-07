"""
Tests for :class:`~olmo_core.nn.vision.MultimodalOLMoDDPModel`: it is built for an OLMoDDP
language model, keeps the plain multimodal forward, and computes the float-weighted response-only
objective through the language model's ``input_embeddings`` / ``router_loss_div_factor`` hooks.
"""

import pytest
import torch

import olmo_core.nn.vision.chunked_loss as chunked_loss
from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
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


def _model_with_loss_chunk_size(chunk_size: int) -> MultimodalOLMoDDPModel:
    """Same weights as :func:`_model` (same seed), scoring ``chunk_size`` tokens at a time."""
    torch.manual_seed(0)
    config = _config()
    config.loss_chunk_size = chunk_size
    model = config.build(init_device="cpu")
    assert isinstance(model, MultimodalOLMoDDPModel)
    model.init_weights(max_seq_len=8, device=torch.device("cpu"))
    return model.train()


def _record_projected_rows(model: MultimodalOLMoDDPModel, monkeypatch) -> list:
    """Row counts of every LM-head projection (the logits materialized at once), through the
    head itself (one-shot path) or the chunked loss's projection."""
    rows: list = []
    w_out = model.lm.lm_head.w_out
    original = w_out.forward

    def forward(x):
        rows.append(int(x.shape[0]))
        return original(x)

    w_out.forward = forward  # type: ignore[method-assign]
    project = chunked_loss._project

    def recording_project(hidden, weight, bias):
        rows.append(int(hidden.shape[0]))
        return project(hidden, weight, bias)

    monkeypatch.setattr(chunked_loss, "_project", recording_project)
    return rows


def _mixed_batch():
    """Two sequences, one leading pooled image token each, weighted response positions."""
    input_ids, images, pooled = _image_batch()
    torch.manual_seed(2)
    labels = torch.randint(2, _VOCAB, input_ids.shape)
    loss_masks = torch.zeros_like(labels, dtype=torch.float32)
    loss_masks[0, 1:5] = 1.0
    loss_masks[1, 2:4] = 0.5
    loss_masks[1, 6] = 2.0
    labels[loss_masks == 0] = -100
    return input_ids, labels, loss_masks, images, pooled


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_loss_matches_the_one_shot_loss_and_gradients(
    with_images: bool, dtype: torch.dtype, monkeypatch
):
    kwargs = {}
    if with_images:
        input_ids, labels, loss_masks, images, pooled = _mixed_batch()
        kwargs = dict(images=images, pooled_patches_idx=pooled)
    else:
        input_ids, labels, loss_masks = _text_batch()
    n_response = int((loss_masks > 0).sum())
    assert n_response == 7
    # bfloat16 gradients can land on either side of a rounding boundary when the float32
    # sums differ in order; float32 is the strict check.
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-2, atol=1e-3)

    one_shot = _model_with_loss_chunk_size(0).to(dtype)
    chunked = _model_with_loss_chunk_size(3).to(dtype)  # 7 response tokens -> chunks of 3, 3, 1
    rows = _record_projected_rows(one_shot, monkeypatch)
    _record_projected_rows(chunked, monkeypatch)
    outputs = []
    projected = []
    for model in (one_shot, chunked):
        seen = len(rows)
        out = model(
            input_ids,
            labels=labels,
            loss_masks=loss_masks,
            loss_reduction="sum",
            z_loss_multiplier=1e-4,
            loss_weight_div_factor=torch.tensor(3.5),
            **kwargs,
        )
        assert isinstance(out, LMOutputWithLoss) and out.logits is None
        out.loss.backward()
        outputs.append(out)
        projected.append(rows[seen:])
    reference, result = outputs
    one_shot_rows, chunked_rows = projected

    # The one-shot path projects every response token at once; the chunked path never more
    # than a chunk, in the forward and again in the backward recomputation.
    assert one_shot_rows == [n_response]
    assert max(chunked_rows) == 3 and sum(chunked_rows) == 2 * n_response
    torch.testing.assert_close(result.ce_loss, reference.ce_loss, rtol=1e-5, atol=1e-6)
    assert result.z_loss is not None and reference.z_loss is not None
    torch.testing.assert_close(result.z_loss, reference.z_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result.loss, reference.loss, rtol=1e-5, atol=1e-6)
    grads = 0
    for (name, p_ref), (_, p_new) in zip(one_shot.named_parameters(), chunked.named_parameters()):
        if p_ref.grad is None:
            assert p_new.grad is None, name
            continue
        assert p_new.grad is not None, name
        torch.testing.assert_close(p_new.grad, p_ref.grad, msg=name, **tolerance)
        grads += 1
    assert grads > 0
    assert one_shot.lm.lm_head.w_out.weight.grad is not None
    if with_images:
        assert next(chunked.vision.parameters()).grad is not None


def test_chunked_loss_without_z_loss_and_with_the_lm_divisor():
    input_ids, labels, loss_masks = _text_batch()
    model = _model_with_loss_chunk_size(2)
    with torch.no_grad():
        out = model(
            input_ids,
            labels=labels,
            loss_masks=loss_masks,
            loss_reduction="sum",
            loss_div_factor=torch.tensor(7.0),
        )
        full_logits = _model_with_loss_chunk_size(0)(input_ids)
    mask = loss_masks > 0
    ce, _ = weighted_cross_entropy_loss(full_logits[mask], labels[mask], loss_masks[mask])
    assert out.z_loss is None and out.logits is None
    torch.testing.assert_close(out.ce_loss, ce / 7.0, rtol=1e-5, atol=1e-6)


def test_requesting_logits_keeps_the_one_shot_path(monkeypatch):
    input_ids, labels, loss_masks = _text_batch()
    model = _model_with_loss_chunk_size(2)
    rows = _record_projected_rows(model, monkeypatch)
    out = model(
        input_ids,
        labels=labels,
        loss_masks=loss_masks,
        loss_reduction="sum",
        return_logits=True,
    )
    assert out.logits is not None and out.logits.shape == (7, _VOCAB)
    assert rows == [7]


def test_lm_head_returns_its_own_logits_outside_the_hidden_states_block():
    model = _model_with_loss_chunk_size(2)
    input_ids, _, _ = _text_batch()
    with torch.no_grad():
        logits = model(input_ids)
        head_in = torch.randn(2, 8, _D_MODEL).to(model.lm.lm_head.w_out.weight.dtype)
        head_out = model.lm.lm_head(head_in)
    assert logits.shape == (2, 8, _VOCAB) and head_out.shape == (2, 8, _VOCAB)
    assert model._lm_head_hidden_states_only is False


def _dummy_crop_batch(batch: int = 2):
    """What the collator hands over for an all-text micro-batch: one zero crop nobody refers to."""
    return torch.zeros(batch, 1, 4, 14 * 14 * 3), torch.full((batch, 1, 4), -1, dtype=torch.long)


def _loss_kwargs(labels, loss_masks):
    return dict(
        labels=labels, loss_masks=loss_masks, loss_reduction="sum", loss_weight_div_factor=4.0
    )


def test_skip_vision_on_text_leaves_the_vision_path_out_of_all_text_batches(monkeypatch):
    model = _model().train()
    model.sync_vit_crops = False
    input_ids, labels, loss_masks = _text_batch()
    images, no_crops = _dummy_crop_batch()
    kwargs = _loss_kwargs(labels, loss_masks)

    # The dummy crop's contribution: exactly 0 to the loss and to every vision/connector grad.
    reference = model(input_ids, images=images, pooled_patches_idx=no_crops, **kwargs)
    reference.loss.backward()
    assert all(
        param.grad is not None and not param.grad.any()
        for module in (model.vision, model.connector)
        for param in module.parameters()
    )
    lm_grads = {name: param.grad.clone() for name, param in model.lm.named_parameters()}
    model.zero_grad(set_to_none=True)

    model.skip_vision_on_text = True
    calls = []
    monkeypatch.setattr(model, "encode_images", lambda *args, **kw: calls.append(1))
    skipped = model(input_ids, images=images, pooled_patches_idx=no_crops, **kwargs)
    skipped.loss.backward()
    assert not calls
    torch.testing.assert_close(skipped.loss, reference.loss, rtol=0, atol=0)
    for name, param in model.lm.named_parameters():
        torch.testing.assert_close(param.grad, lm_grads[name], rtol=0, atol=0, msg=name)
    assert all(
        param.grad is None
        for module in (model.vision, model.connector)
        for param in module.parameters()
    )


def test_skip_vision_on_text_keeps_image_batches_unchanged(monkeypatch):
    model = _model()
    model.sync_vit_crops = False
    input_ids, images, pooled = _image_batch()
    with torch.no_grad():
        reference = model(input_ids, images=images, pooled_patches_idx=pooled)
    model.skip_vision_on_text = True
    calls = []
    encode = model.encode_images
    monkeypatch.setattr(
        model, "encode_images", lambda *args, **kw: (calls.append(1), encode(*args, **kw))[1]
    )
    with torch.no_grad():
        logits = model(input_ids, images=images, pooled_patches_idx=pooled)
    assert calls == [1]
    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
    # A row of indices is enough: the batch is text-only only when no entry refers to a crop.
    mixed = pooled.clone()
    mixed[1] = -1
    assert not model.skips_vision(mixed) and model.skips_vision(torch.full_like(pooled, -1))


def test_skip_vision_on_text_needs_a_wrapper_without_forward_collectives():
    model = _model()
    model.skip_vision_on_text = True
    input_ids, labels, loss_masks = _text_batch()
    images, no_crops = _dummy_crop_batch()
    assert model.sync_vit_crops
    with pytest.raises(OLMoConfigurationError, match="sync_vit_crops=False"):
        model(input_ids, images=images, pooled_patches_idx=no_crops)
    model.sync_vit_crops = False
    # Skipping must not hide ids that still expect features.
    input_ids[0, 0] = _IMAGE_PATCH_TOKEN
    with pytest.raises(ValueError, match="refers to no crop"):
        model(input_ids, images=images, pooled_patches_idx=no_crops)
