import pytest
import torch

from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalLM,
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LM_VOCAB = 256
_LM_D_MODEL = 12  # olmo2_1M d_model
_IMAGE_PATCH_TOKEN = 1  # arbitrary reserved ID for <im_patch>


def _tiny_lm_cfg() -> TransformerConfig:
    return TransformerConfig.olmo2_1M(vocab_size=_LM_VOCAB)


def _tiny_vision_cfg() -> VisionEncoderConfig:
    """Tiny CLIP-style ViT: 2×2 patch grid (28px / 14px), 2 layers, emb_dim=32."""
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
        image_num_pos=5,  # 1 CLS + 4 patches
        image_norm_eps=1e-5,
    )


def _tiny_multimodal_cfg(vit_layers=(-1,)) -> MultimodalLMConfig:
    lm_cfg = _tiny_lm_cfg()
    vis_cfg = _tiny_vision_cfg()
    conn_cfg = VisionConnectorConfig.from_vision_encoder(
        vis_cfg,
        output_dim=_LM_D_MODEL,
        mlp_hidden_size=32,
    )
    return MultimodalLMConfig(
        lm=lm_cfg,
        vision=vis_cfg,
        connector=conn_cfg,
        image_patch_token_id=_IMAGE_PATCH_TOKEN,
        vit_layers=vit_layers,
    )


def _make_inputs(
    batch: int,
    seq_len: int,
    n_crops: int = 1,
    n_patches_per_crop: int = 4,
    pool_size: int = 4,
):
    """Build (input_ids, images, pooled_patches_idx) with the right counts."""
    total_patches = n_crops * n_patches_per_crop
    assert total_patches % pool_size == 0
    n_pooled = total_patches // pool_size

    # Place `n_pooled` <im_patch> tokens at the front; fill the rest with random non-image tokens.
    image_token_ids = torch.full((batch, n_pooled), _IMAGE_PATCH_TOKEN, dtype=torch.long)
    text_ids = torch.randint(2, _LM_VOCAB, (batch, seq_len - n_pooled))
    input_ids = torch.cat([image_token_ids, text_ids], dim=1)

    images = torch.randn(batch, n_crops, n_patches_per_crop, 14 * 14 * 3)

    # Row-major non-overlapping pools across the total patch axis.
    idx = (
        torch.arange(total_patches, dtype=torch.long)
        .view(n_pooled, pool_size)
        .unsqueeze(0)
        .expand(batch, -1, -1)
        .contiguous()
    )
    return input_ids, images, idx


# ---------------------------------------------------------------------------
# MultimodalLMConfig
# ---------------------------------------------------------------------------


def test_build_returns_multimodal_lm():
    cfg = _tiny_multimodal_cfg()
    model = cfg.build(init_device="cpu")
    assert isinstance(model, MultimodalLM)


def test_config_has_expected_fields():
    cfg = _tiny_multimodal_cfg()
    assert cfg.vit_layers == (-1,)
    assert cfg.lm.d_model == _LM_D_MODEL
    assert cfg.vision.image_emb_dim == 32
    assert cfg.connector.output_dim == _LM_D_MODEL
    assert cfg.image_patch_token_id == _IMAGE_PATCH_TOKEN


# ---------------------------------------------------------------------------
# Text-only forward pass
# ---------------------------------------------------------------------------


class TestTextOnlyForward:
    def setup_method(self):
        self.cfg = _tiny_multimodal_cfg()
        self.model = self.cfg.build(init_device="cpu")
        self.model.eval()

    def test_output_shape(self):
        input_ids = torch.randint(2, _LM_VOCAB, (2, 16))
        out = self.model(input_ids)
        assert out.shape == (2, 16, _LM_VOCAB)

    def test_batch_size_1(self):
        input_ids = torch.randint(2, _LM_VOCAB, (1, 8))
        out = self.model(input_ids)
        assert out.shape == (1, 8, _LM_VOCAB)

    def test_deterministic_eval(self):
        input_ids = torch.randint(2, _LM_VOCAB, (2, 8))
        out1 = self.model(input_ids)
        out2 = self.model(input_ids)
        assert torch.allclose(out1, out2)

    def test_with_labels_returns_loss(self):
        from olmo_core.nn.lm_head import LMOutputWithLoss

        input_ids = torch.randint(2, _LM_VOCAB, (2, 8))
        labels = torch.randint(2, _LM_VOCAB, (2, 8))
        out = self.model(input_ids, labels=labels)
        assert isinstance(out, LMOutputWithLoss)
        assert out.loss.shape == ()


# ---------------------------------------------------------------------------
# Image forward pass (1 crop, 2×2 pool → 1 pooled token per batch element)
# ---------------------------------------------------------------------------


class TestImageForward:
    def setup_method(self):
        self.cfg = _tiny_multimodal_cfg()
        self.model = self.cfg.build(init_device="cpu")
        self.model.eval()

    def test_output_shape(self):
        input_ids, images, idx = _make_inputs(batch=2, seq_len=16)
        out = self.model(input_ids, images=images, pooled_patches_idx=idx)
        assert out.shape == (2, 16, _LM_VOCAB)

    def test_image_injection_changes_output(self):
        """Output differs from text-only when image features are injected."""
        torch.manual_seed(42)
        input_ids, images, idx = _make_inputs(batch=1, seq_len=16)
        out_text = self.model(input_ids)
        out_img = self.model(input_ids, images=images, pooled_patches_idx=idx)
        assert not torch.allclose(out_text, out_img)

    def test_missing_idx_raises(self):
        input_ids, images, _ = _make_inputs(batch=1, seq_len=16)
        with pytest.raises(ValueError, match="pooled_patches_idx"):
            self.model(input_ids, images=images)

    def test_mismatched_image_patch_count_raises(self):
        """Wrong number of <im_patch> tokens in input_ids should raise."""
        input_ids, images, idx = _make_inputs(batch=1, seq_len=16)
        # Drop one <im_patch>: replace with a non-image token.
        input_ids[0, 0] = 7
        with pytest.raises(ValueError, match="match the number of projected image features"):
            self.model(input_ids, images=images, pooled_patches_idx=idx)


# ---------------------------------------------------------------------------
# Multi-crop image input
# ---------------------------------------------------------------------------


def test_multi_crop():
    cfg = _tiny_multimodal_cfg()
    model = cfg.build(init_device="cpu")
    model.eval()
    # 2 crops × 4 patches = 8 total, pool=4 → 2 pooled tokens per batch element.
    input_ids, images, idx = _make_inputs(
        batch=2, seq_len=16, n_crops=2, n_patches_per_crop=4, pool_size=4
    )
    out = model(input_ids, images=images, pooled_patches_idx=idx)
    assert out.shape == (2, 16, _LM_VOCAB)


# ---------------------------------------------------------------------------
# Meta device
# ---------------------------------------------------------------------------


def test_meta_device():
    cfg = _tiny_multimodal_cfg()
    model = cfg.build(init_device="meta")
    for p in model.parameters():
        assert p.device.type == "meta"
        break


# ---------------------------------------------------------------------------
# Multi-layer ViT extraction
# ---------------------------------------------------------------------------


def test_multi_layer_vit():
    """Two-layer extraction: connector must have num_input_layers=2."""
    lm_cfg = _tiny_lm_cfg()
    vis_cfg = _tiny_vision_cfg()
    conn_cfg = VisionConnectorConfig.from_vision_encoder(
        vis_cfg,
        output_dim=_LM_D_MODEL,
        num_input_layers=2,
        mlp_hidden_size=32,
    )
    cfg = MultimodalLMConfig(
        lm=lm_cfg,
        vision=vis_cfg,
        connector=conn_cfg,
        image_patch_token_id=_IMAGE_PATCH_TOKEN,
        vit_layers=(-1, -2),
    )
    model = cfg.build(init_device="cpu")
    model.eval()
    input_ids, images, idx = _make_inputs(batch=2, seq_len=16)
    out = model(input_ids, images=images, pooled_patches_idx=idx)
    assert out.shape == (2, 16, _LM_VOCAB)


# ---------------------------------------------------------------------------
# Sequence packing: a packed example must be isolated (its logits unchanged vs
# running it standalone), for both the dense and FlexAttention backends.
# ---------------------------------------------------------------------------


def _packing_example(seed: int, n_text: int, n_crops: int):
    import numpy as np

    rng = np.random.RandomState(seed)
    L = n_crops + n_text
    return dict(
        input_ids=np.array(
            [_IMAGE_PATCH_TOKEN] * n_crops + list(rng.randint(2, _LM_VOCAB, n_text))
        ),
        labels=np.full(L, 2, dtype=np.int64),
        loss_masks=np.ones(L, dtype=np.float32),
        position_ids=np.arange(L, dtype=np.int64),
        token_type_ids=np.array([1] * n_crops + [0] * n_text, dtype=np.int64),
        images=rng.randn(n_crops, 4, 14 * 14 * 3).astype("float32"),
        pooled_patches_idx=np.arange(n_crops * 4).reshape(n_crops, 4).astype("int64"),
    )


@pytest.mark.parametrize("backend", ["torch", "flex"])
def test_packed_examples_are_isolated(backend):
    from olmo_core.data.multimodal.collator import MultimodalCollatorConfig
    from olmo_core.data.multimodal.packing import pack_examples
    from olmo_core.nn.attention import AttentionBackendName

    torch.manual_seed(0)
    lm_cfg = _tiny_lm_cfg()
    # set the attention backend
    lm_cfg = TransformerConfig.olmo2_1M(
        vocab_size=_LM_VOCAB, attn_backend=AttentionBackendName(backend)
    )
    vis_cfg = _tiny_vision_cfg()
    conn_cfg = VisionConnectorConfig.from_vision_encoder(
        vis_cfg, output_dim=_LM_D_MODEL, mlp_hidden_size=32
    )
    cfg = MultimodalLMConfig(
        lm=lm_cfg,
        vision=vis_cfg,
        connector=conn_cfg,
        image_patch_token_id=_IMAGE_PATCH_TOKEN,
        vit_layers=(-1,),
    )
    model = cfg.build(init_device="cpu").eval()
    coll = MultimodalCollatorConfig(pad_token_id=0).build()

    a = _packing_example(1, n_text=6, n_crops=1)
    b = _packing_example(2, n_text=4, n_crops=2)

    def run(example_list):
        batch = coll(example_list)
        kw = {k: v for k, v in batch.items() if k not in ("input_ids", "labels", "loss_masks")}
        with torch.no_grad():
            return model(batch["input_ids"], **kw)[0]

    la, lb, lp = run([a]), run([b]), run([pack_examples([a, b])])
    na, nb = len(a["input_ids"]), len(b["input_ids"])
    torch.testing.assert_close(lp[:na], la, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lp[na : na + nb], lb, atol=1e-4, rtol=1e-4)
