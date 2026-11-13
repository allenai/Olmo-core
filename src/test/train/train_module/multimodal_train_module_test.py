"""Regression tests for ``MultimodalTransformerTrainModule.train_batch``.

These exercise the *train module*, not just ``MultimodalLM.forward``. That distinction caught a
real bug: response residual dropout needs ``loss_masks`` to build its per-token drop mask, and
the dense-logits path (``response_logits_only=False``, which is the released Molmo2 recipe) was
not forwarding it — so training raised on the first batch even though calling the model directly
with ``loss_masks`` worked fine.
"""

from typing import Any, Dict, List, cast

import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)
from olmo_core.optim import AdamWConfig, SkipStepAdamWConfig
from olmo_core.train.train_module import MultimodalTransformerTrainModuleConfig

BASE_VOCAB, EXTRA_VOCAB = 200, 8
IMAGE_PATCH_ID = BASE_VOCAB + 2
SEQ_LEN, N_POOLED = 16, 4


def _model_config(masked_dropout: float) -> MultimodalLMConfig:
    lm = TransformerConfig.llama_like(
        d_model=16,
        vocab_size=BASE_VOCAB,
        n_layers=2,
        n_heads=2,
        n_extra_vocab=EXTRA_VOCAB,
        attn_backend=AttentionBackendName.torch,
    )
    if masked_dropout:
        cast(Any, lm.block).masked_dropout = masked_dropout
    vision = VisionEncoderConfig(
        name=VisionEncoderType.siglip,
        use_cls_token=False,
        patch_embedding_bias=True,
        use_pre_ln=False,
        image_default_input_size=(56, 56),
        image_patch_size=14,
        image_emb_dim=32,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=64,
        image_num_pos=16,
        image_norm_eps=1e-5,
    )
    connector = VisionConnectorConfig.from_vision_encoder(
        vision, output_dim=lm.d_model, mlp_hidden_size=64
    )
    return MultimodalLMConfig(
        lm=lm, vision=vision, connector=connector, image_patch_token_id=IMAGE_PATCH_ID
    )


class _StubTrainer:
    """Minimal stand-in: ``train_batch`` only reaches the trainer to record metrics."""

    def __init__(self) -> None:
        self.ce_losses: List[float] = []

    def record_metric(self, *args, **kwargs) -> None:
        pass

    def record_ce_loss(self, value, *args, **kwargs) -> None:
        self.ce_losses.append(float(value))


def _train_module(masked_dropout: float, response_logits_only: bool, optim=None):
    config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LEN,
        max_sequence_length=SEQ_LEN,
        optim=optim or AdamWConfig(lr=1e-4),
        max_grad_norm=1.0,
        z_loss_multiplier=1e-4,
        compile_model=False,
        response_logits_only=response_logits_only,
    )
    model = _model_config(masked_dropout).build(init_device="cpu")
    # `build` allocates parameters without filling them, and `MultimodalLM` has no single
    # `init_weights` — the training script initialises the three components separately (LM and
    # ViT from pretrained checkpoints, connector randomly). Skipping this leaves the model running
    # on whatever was in memory, which produced NaN logits in roughly half of all runs.
    # `device` defaults to the default device, which is CUDA wherever a GPU is visible; this test
    # is CPU-only, so pin it.
    model.lm.init_weights(max_seq_len=SEQ_LEN, device=torch.device("cpu"))
    model.vision.reset_parameters()
    model.connector.reset_parameters()
    train_module = config.build(model)
    train_module._trainer = _StubTrainer()  # type: ignore[assignment]
    return train_module


def _batch() -> Dict[str, torch.Tensor]:
    # Seeded: an unseeded random 2-layer model occasionally produces non-finite logits, which
    # the trainer's finite-loss guard then reports. That is a tiny-random-model artifact (the
    # real model starts from pretrained Qwen3 + SigLIP2 weights) and unrelated to what these
    # tests check, so pin it rather than let the suite flake.
    torch.manual_seed(0)
    input_ids = torch.randint(0, BASE_VOCAB, (1, SEQ_LEN))
    input_ids[0, 2 : 2 + N_POOLED] = IMAGE_PATCH_ID
    loss_masks = torch.zeros(1, SEQ_LEN)
    loss_masks[:, SEQ_LEN // 2 :] = 1.0  # second half = response tokens
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "loss_masks": loss_masks,
        "images": torch.randn(1, 1, 16, 3 * 14**2),
        "pooled_patches_idx": torch.arange(16).view(1, N_POOLED, 4),
    }


@pytest.mark.parametrize("response_logits_only", [False, True])
def test_train_batch_with_response_residual_dropout(response_logits_only: bool):
    """Both logit paths must supply ``loss_masks`` to the model.

    The dense path (``False``) is the released Molmo2 recipe and previously raised
    ``ValueError: loss_masks is required when the LM uses masked_dropout``.
    """
    torch.manual_seed(0)
    train_module = _train_module(masked_dropout=0.1, response_logits_only=response_logits_only)
    train_module.train_batch(_batch())


def _single_token_batch(weight: float) -> Dict[str, torch.Tensor]:
    batch = _batch()
    batch["loss_masks"] = torch.zeros(1, SEQ_LEN)
    batch["loss_masks"][0, SEQ_LEN - 2] = weight
    return batch


def _run_single_token(weight: float):
    torch.manual_seed(0)
    train_module = _train_module(masked_dropout=0.0, response_logits_only=False)
    train_module.train_batch(_single_token_batch(weight))
    grad_norm = torch.sqrt(
        sum(
            p.grad.float().pow(2).sum()
            for p in train_module.model.parameters()
            if p.grad is not None
        )
    )
    return float(grad_norm), cast(_StubTrainer, train_module._trainer).ce_losses[-1]


def test_fractional_weights_are_normalised_not_clamped():
    """One supervised token with weight 0.25 must give the same gradient and metric as weight 1."""
    grad_full, ce_full = _run_single_token(1.0)
    grad_frac, ce_frac = _run_single_token(0.25)
    assert grad_full > 0
    assert grad_frac == pytest.approx(grad_full, rel=1e-4)
    assert ce_frac == pytest.approx(ce_full, rel=1e-4)


def test_skip_step_optimizer_receives_loss_and_grad_norm():
    torch.manual_seed(0)
    train_module = _train_module(
        masked_dropout=0.0,
        response_logits_only=False,
        optim=SkipStepAdamWConfig(lr=1e-4, rolling_interval_length=4),
    )
    train_module.train_batch(_batch())
    train_module.optim_step()
    optim = cast(Any, train_module.optim)
    assert optim.latest_loss is not None
    assert optim.latest_grad_norm is not None
