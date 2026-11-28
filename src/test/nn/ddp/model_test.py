"""Tests for ``OLMoDDPModel`` construction and FLOP accounting."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    KimiDeltaAttention,
    KimiDeltaAttentionConfig,
)
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
from olmo_core.testing.distributed import run_distributed_test
from olmo_core.testing.utils import requires_fla


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


def _run_kda_decay_params_stay_fp32():
    config = _build_model_config(n_layers=1)
    assert isinstance(config.block, OLMoDDPTransformerBlockConfig)
    config.block.sequence_mixer = KimiDeltaAttentionConfig(n_heads=4, dtype=DType.float32)
    model = config.build(init_device="meta")
    model.init_weights(device=torch.device("cpu"))
    attention = model.blocks["0"].attention
    assert isinstance(attention, KimiDeltaAttention)
    assert attention.A_log.dtype == attention.dt_bias.dtype == torch.float32
    assert attention.w_q.weight.dtype == attention.q_conv1d.weight.dtype == torch.bfloat16

    with torch.no_grad():
        attention.A_log.fill_(1.234567)
        attention.dt_bias.fill_(0.1234567)
    expected = (attention.A_log.detach().clone(), attention.dt_bias.detach().clone())
    mesh = init_device_mesh("cpu", (2,))
    model.apply_dp(dp_mesh=mesh, ep_mesh=None)
    for param, value in zip((attention.A_log, attention.dt_bias), expected):
        torch.testing.assert_close(param, value, rtol=0, atol=0)


@requires_fla
def test_kda_decay_params_stay_fp32_during_materialization_and_dp():
    run_distributed_test(
        _run_kda_decay_params_stay_fp32, world_size=2, backend="gloo", start_method="spawn"
    )


def test_deepep_rejects_chunk_recompute():
    config = _build_model_config(n_layers=2)
    config.recompute_all_blocks_by_chunk = True
    assert isinstance(config.block, OLMoDDPTransformerBlockConfig)
    config.block.ep = ExpertParallelConfig(path=ExpertParallelPath.deepep_v2)
    with pytest.raises(OLMoConfigurationError, match="recompute_all_blocks_by_chunk"):
        config.build(init_device="cpu")
