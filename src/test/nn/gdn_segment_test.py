"""Sequence-segmented GatedDeltaNet training forward (--gdn-segment-tokens): same outputs and gradients as one pass."""
import torch

from olmo_core.nn.ffn_chunk import install_gdn_segments
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla


def _model():
    from olmo_core.nn.attention import GatedDeltaNetConfig
    from olmo_core.nn.feed_forward import FeedForwardConfig
    from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
    from olmo_core.nn.lm_head import LMHeadConfig
    from olmo_core.nn.transformer import TransformerBlockConfig, TransformerBlockType, TransformerConfig

    torch.manual_seed(0)
    ln = LayerNormConfig(name=LayerNormType.rms, bias=False)
    gdn = TransformerBlockConfig(name=TransformerBlockType.default, sequence_mixer=GatedDeltaNetConfig(n_heads=4),
                                 layer_norm=ln, feed_forward=FeedForwardConfig(hidden_size=256, bias=False))
    m = TransformerConfig(d_model=128, vocab_size=512, n_layers=2, block=gdn,
                          lm_head=LMHeadConfig(layer_norm=ln, bias=False)).build(init_device="cuda")
    m.init_weights()
    return m


def _run(m, ids):
    m.zero_grad(set_to_none=True)
    out = m(ids)
    logits = out if torch.is_tensor(out) else out.logits
    (logits.float() ** 2).mean().backward()
    return logits.detach().float(), {n: p.grad.detach().float().clone() for n, p in m.named_parameters()
                                     if p.grad is not None}


@requires_gpu
@requires_fla
def test_segmented_gdn_matches_one_pass():
    m = _model()
    m.train()
    ids = torch.randint(0, 512, (2, 640), device="cuda")
    ref, ref_g = _run(m, ids)
    assert install_gdn_segments(m, 128) == 2
    out, g = _run(m, ids)
    assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3), (out - ref).abs().max()
    for n in ref_g:
        err = (g[n] - ref_g[n]).norm() / ref_g[n].norm().clamp_min(1e-12)
        assert err < 2e-3, (n, float(err))
