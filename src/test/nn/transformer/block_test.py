from typing import Any, Dict, Type

import pytest
import torch
from torch.distributed.tensor import Shard, init_device_mesh

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.transformer.block import (
    PeriNormTransformerBlock,
    ReorderedNormTransformerBlock,
    TransformerBlock,
)
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device, seed_all


def _build_block(
    block_cls: Type[TransformerBlock],
    *,
    d_model: int,
    init_device: str,
    attn_kwargs: Dict[str, Any],
) -> TransformerBlock:
    attn_cfg = AttentionConfig(**attn_kwargs)
    ff_cfg = FeedForwardConfig(hidden_size=4 * d_model)
    ln_cfg = LayerNormConfig()
    return block_cls(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
        sequence_mixer=attn_cfg,
        feed_forward=ff_cfg,
        attention_norm=ln_cfg,
        feed_forward_norm=ln_cfg,
        init_device=init_device,
    )


def _run_tensor_parallel_block(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    block_cls: Type[TransformerBlock],
    d_model: int,
    attn_kwargs: Dict[str, Any],
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    block = _build_block(
        block_cls, d_model=d_model, init_device=device.type, attn_kwargs=attn_kwargs
    )

    # Shard sequence dim in/out like the transformer model does.
    block.apply_tp(mesh["tp"], input_layout=Shard(1))
    load_model_and_optim_state(checkpoint_dir, block)

    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk = x.size(1) // world_size
    x_local = x[:, rank * chunk : (rank + 1) * chunk, :]
    y_local = block(x_local)

    # Backward to exercise graph in TP mode.
    y_local.sum().backward()

    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk : (rank + 1) * chunk, :]
    torch.testing.assert_close(y_ref_local, y_local.to_local())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "attn_kwargs",
    [
        pytest.param(dict(n_heads=8), id="default"),
        pytest.param(dict(n_heads=8, rope=None, bias=False), id="no-bias"),
    ],
)
@pytest.mark.parametrize(
    "block_cls", [TransformerBlock, ReorderedNormTransformerBlock, PeriNormTransformerBlock]
)
def test_tensor_parallel_transformer_block(
    backend: str, block_cls: Type[TransformerBlock], attn_kwargs: Dict[str, Any], tmp_path
):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")

    seed_all(0)
    d_model = 128
    attn_kwargs = {
        **attn_kwargs,
        "name": AttentionType.default,
        "backend": AttentionBackendName.torch,
    }

    block = _build_block(
        block_cls, d_model=d_model, init_device=device.type, attn_kwargs=attn_kwargs
    )

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, d_model, device=device)
    y = block(x)

    outputs_path = tmp_path / "block_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "block_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, block)

    run_distributed_test(
        _run_tensor_parallel_block,
        backend=backend,
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, block_cls, d_model, attn_kwargs),
    )
