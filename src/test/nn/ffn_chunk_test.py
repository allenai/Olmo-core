"""Sequence-chunked FFN sublayer (olmo_core.nn.ffn_chunk): exact outputs and gradients, eval / short rows untouched."""
import pytest
import torch

from olmo_core.nn.ffn_chunk import install_ffn_chunk
from olmo_core.nn.transformer import TransformerConfig


def _model():
    torch.manual_seed(0)
    m = TransformerConfig.llama_like(vocab_size=64, d_model=32, n_layers=3, n_heads=2).build(init_device="cpu")
    m.init_weights(device=torch.device("cpu"))
    return m.float()


def _run(m, ids):
    m.zero_grad(set_to_none=True)
    out = m(ids)
    logits = out if torch.is_tensor(out) else out.logits
    (logits.float() ** 2).mean().backward()
    return logits.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def test_chunked_ffn_matches_unchunked_outputs_and_grads():
    m = _model()
    m.train()
    ids = torch.randint(0, 64, (2, 11))
    ref_out, ref_g = _run(m, ids)
    layers = install_ffn_chunk(m, 4, "0-1")
    assert layers == [0, 1]
    out, g = _run(m, ids)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)
    assert ref_g.keys() == g.keys()
    for n in ref_g:
        assert torch.allclose(g[n], ref_g[n], atol=1e-5, rtol=1e-4), n
    blocks = list(m.blocks.values())
    assert blocks[0]._ffn_chunk_calls == 1 and blocks[1]._ffn_chunk_calls == 1
    assert not hasattr(blocks[2], "_ffn_chunk")


def test_short_rows_and_eval_take_the_normal_path():
    m = _model()
    install_ffn_chunk(m, 16)
    m.train()
    _run(m, torch.randint(0, 64, (1, 9)))  # T <= chunk
    m.eval()
    with torch.no_grad():
        m(torch.randint(0, 64, (1, 40)))
    assert all(b._ffn_chunk_calls == 0 for b in m.blocks.values())


def test_rejects_bad_args():
    m = _model()
    with pytest.raises(ValueError):
        install_ffn_chunk(m, 0)
