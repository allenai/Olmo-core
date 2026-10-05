import pytest
import torch

from olmo_core.nn.functional import cross_entropy_loss, weighted_cross_entropy_loss
from olmo_core.testing import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_cross_entropy_loss(device, reduction):
    vocab_size = 50257
    N = 32

    logits = torch.randn(N, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (N,), device=device)

    ce_loss, z_loss = cross_entropy_loss(logits, labels, reduction=reduction, compute_z_loss=True)
    assert ce_loss.shape == tuple()
    assert ce_loss.numel() == 1
    assert z_loss is not None
    assert z_loss.shape == tuple()
    assert z_loss.numel() == 1

    # Now add some masked values to logits and labels and make sure we get the same result.
    logits_padded = torch.cat([logits, torch.rand(3, vocab_size, device=device)], dim=0)
    labels_padded = torch.cat([labels, torch.tensor([-100] * 3, device=device)], dim=0)
    ce_loss1, z_loss1 = cross_entropy_loss(
        logits_padded, labels_padded, reduction=reduction, compute_z_loss=True
    )
    torch.testing.assert_close(ce_loss, ce_loss1)
    torch.testing.assert_close(z_loss, z_loss1)


@pytest.mark.parametrize("device", DEVICES)
def test_weighted_cross_entropy_loss_matches_manual(device):
    vocab_size = 1000
    N = 16

    logits = torch.randn(N, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (N,), device=device)
    weights = torch.rand(N, device=device)

    ce_loss, z_loss = weighted_cross_entropy_loss(
        logits, labels, weights, compute_z_loss=True, z_loss_multiplier=1e-4
    )

    # Manual reference: per-token CE dotted with weights, weighted z-loss.
    per_token = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")
    expected_ce = torch.dot(per_token, weights.float())
    z_squared = logits.float().logsumexp(-1).pow(2)
    expected_z = torch.dot(z_squared, weights.float()) * 1e-4

    assert ce_loss.shape == tuple()
    assert z_loss is not None
    torch.testing.assert_close(ce_loss, expected_ce)
    torch.testing.assert_close(z_loss, expected_z)


@pytest.mark.parametrize("device", DEVICES)
def test_weighted_cross_entropy_loss_ignores_zero_weight_tokens(device):
    """Padded tokens (weight 0, label -100) must not change the loss."""
    vocab_size = 1000
    N = 16

    logits = torch.randn(N, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (N,), device=device)
    weights = torch.rand(N, device=device)

    ce_loss, z_loss = weighted_cross_entropy_loss(logits, labels, weights, compute_z_loss=True)

    # Append padding rows: weight 0 and ignore_index labels.
    logits_pad = torch.cat([logits, torch.randn(4, vocab_size, device=device)], dim=0)
    labels_pad = torch.cat([labels, torch.full((4,), -100, device=device)], dim=0)
    weights_pad = torch.cat([weights, torch.zeros(4, device=device)], dim=0)
    ce_pad, z_pad = weighted_cross_entropy_loss(
        logits_pad, labels_pad, weights_pad, compute_z_loss=True
    )

    torch.testing.assert_close(ce_loss, ce_pad)
    assert z_loss is not None and z_pad is not None
    torch.testing.assert_close(z_loss, z_pad)
