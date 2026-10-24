"""Activation checkpointing must replay dropout masks, not redraw them.

Checkpointing runs the forward twice: once to get the output, once during backward to rebuild the
activations it discarded. If the recomputed pass draws *fresh* dropout masks, the gradients are
taken with respect to a different sample than the loss was, which silently biases training. The
fix is to preserve the RNG state across recomputation whenever dropout is active — the same
condition mm_olmo uses in `llm_activation_checkpoint_function`.

This was not academic: with Molmo2's `response_residual_dropout=0.1` and full activation
checkpointing, a 150-step benchmark landed 0.3 nats *worse* than the identical run with
checkpointing off, which should be mathematically neutral.
"""


import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
from olmo_core.nn.transformer.config import TransformerConfig

VOCAB, SEQ_LEN = 64, 8


def _model(masked_dropout: float = 0.0, dropout: float = 0.0):
    config = TransformerConfig.llama_like(
        d_model=32,
        vocab_size=VOCAB,
        n_layers=2,
        n_heads=2,
        attn_backend=AttentionBackendName.torch,
    )
    config.block.masked_dropout = masked_dropout
    config.block.dropout = dropout
    return config.build(init_device="cpu")


@pytest.mark.parametrize(
    "masked_dropout, dropout, expected",
    [
        pytest.param(0.0, 0.0, False, id="no-dropout"),
        pytest.param(0.1, 0.0, True, id="masked-dropout"),
        pytest.param(0.0, 0.1, True, id="plain-dropout"),
    ],
)
def test_preserve_rng_state_tracks_whether_dropout_is_active(
    masked_dropout: float, dropout: float, expected: bool
):
    """RNG preservation costs something per block, so it should be on only when it matters."""
    model = _model(masked_dropout=masked_dropout, dropout=dropout)
    assert model._dropout_is_active() is expected

    model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    wrapped = next(iter(model.blocks.values()))
    assert wrapped.checkpoint_fn.keywords["preserve_rng_state"] is expected  # type: ignore[union-attr]


def _grads(masked_dropout: float, ac: bool, seed: int = 0) -> list[torch.Tensor]:
    """Gradients from one forward/backward, with the model built and run under a fixed seed."""
    torch.manual_seed(seed)
    model = _model(masked_dropout=masked_dropout)
    if ac:
        model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)

    input_ids = torch.randint(0, VOCAB, (1, SEQ_LEN))
    loss_masks = torch.zeros(1, SEQ_LEN)
    loss_masks[:, SEQ_LEN // 2 :] = 1.0

    torch.manual_seed(seed + 1)  # same dropout draws for both the AC and non-AC paths
    logits = model(input_ids, drop_mask=loss_masks > 0)
    logits.sum().backward()
    return [p.grad.clone() for p in model.parameters() if p.grad is not None]


def test_checkpointed_grads_match_uncheckpointed_grads_under_dropout():
    """The real property: checkpointing must not change the gradient.

    With the masks replayed, the checkpointed backward sees exactly the forward's masks, so the
    gradients match. Redrawing them instead produces visibly different gradients — this test fails
    with ``preserve_rng_state=False``.
    """
    without_ac = _grads(masked_dropout=0.1, ac=False)
    with_ac = _grads(masked_dropout=0.1, ac=True)

    assert len(without_ac) == len(with_ac)
    for expected, actual in zip(without_ac, with_ac):
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)


def test_checkpointing_is_still_neutral_without_dropout():
    """Sanity check on the harness: with no dropout there is no RNG to preserve either way."""
    for expected, actual in zip(
        _grads(masked_dropout=0.0, ac=False), _grads(masked_dropout=0.0, ac=True)
    ):
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
