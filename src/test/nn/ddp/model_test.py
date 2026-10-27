"""Tests for ``OLMoDDPModel`` construction and FLOP accounting."""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)


def _build_model_config(*, d_model: int = 64, n_layers: int = 2) -> OLMoDDPModelConfig:
    dtype = DType.float32
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    return OLMoDDPModelConfig(
        init_seed=0,
        d_model=d_model,
        recompute_each_block=False,
        vocab_size=128,
        n_layers=n_layers,
        name=TransformerType.moe_fused_v2,
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=4,
                bias=False,
                backend=AttentionBackendName.torch,
                dtype=dtype,
            ),
            routed_experts=RoutedExpertsConfig(
                d_model=d_model, hidden_size=128, num_experts=4, bias=False, dtype=dtype
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model, num_experts=4, top_k=2, dtype=dtype
            ),
            shared_experts=None,
            layer_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
    )


def test_moe_v2_model_builds():
    model = _build_model_config(n_layers=2).build(init_device="cpu")

    assert len(model.blocks) == 2
    assert any(p.numel() > 0 for p in model.parameters())
    assert model.num_flops_per_token(seq_len=512) > 0


def test_deepep_rejects_chunk_recompute():
    config = _build_model_config(n_layers=2)
    config.recompute_all_blocks_by_chunk = True
    assert isinstance(config.block, OLMoDDPTransformerBlockConfig)
    config.block.ep = ExpertParallelConfig(path=ExpertParallelPath.deepep_v2)
    with pytest.raises(OLMoConfigurationError, match="recompute_all_blocks_by_chunk"):
        config.build(init_device="cpu")


def test_conditioned_embeddings_preserve_lookup_gradients_and_skip_normalization():
    model = _build_model_config(n_layers=1).build(init_device="cpu")
    # Isolate the embedding/head path from CUDA-only routed expert kernels.
    model.blocks.clear()
    model.init_weights()
    model.embed_scale = 8.0
    model.embedding_norm = torch.nn.LayerNorm(model.d_model)
    tokens = torch.tensor([[1, 2, 3, 4]])
    ordinary = model(tokens)
    embedded = model.forward_embed(tokens)
    torch.testing.assert_close(model(tokens, input_embeddings=embedded), ordinary)
    features = torch.randn_like(embedded)
    conditioned = model(tokens, input_embeddings=embedded + features)
    assert not torch.allclose(conditioned, ordinary)
    conditioned.square().mean().backward()
    assert model.embeddings.weight.grad[1:5].abs().sum() > 0
    assert model.embedding_norm.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["cp", "tbo", "pipeline", "shape", "dtype"])
def test_conditioned_embeddings_reject_unqualified_layouts(mode):
    model = _build_model_config(n_layers=1).build(init_device="cpu")
    tokens = torch.tensor([[1, 2]])
    embedded = torch.zeros(1, 2, model.d_model)
    if mode == "cp":
        model._cp_load_balancer = object()
    elif mode == "tbo":
        model.tbo = True
    elif mode == "pipeline":
        model.embeddings = None
    elif mode == "shape":
        embedded = embedded[:, :1]
    else:
        embedded = embedded.long()
    with pytest.raises(ValueError, match="input_embeddings"):
        model(tokens, input_embeddings=embedded)
