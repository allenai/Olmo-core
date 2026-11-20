from typing import Any, Dict, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import Shard, init_device_mesh

from olmo_core.data.utils import attention_mask_to_cache_leftpad
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    FusedAttentionV2,
    GateConfig,
    GateGranularity,
    NormalizedAttention,
    RingAttentionLoadBalancerType,
    SlidingWindowAttentionConfig,
    _causal_attention_positions,
)
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import (
    BACKENDS,
    DEVICES,
    FLASH_2_MARKS,
    FLASH_3_MARKS,
    FLASH_4_MARKS,
    GPU_MARKS,
    TE_MARKS,
    requires_flash_attn_2,
    requires_gpu,
    requires_multi_gpu,
    run_distributed_test,
)
from olmo_core.testing.utils import compute_capability, requires_compute_capability
from olmo_core.utils import get_default_device, seed_all

BF16_RTOL = 1e-5
BF16_ATOL = 5e-3


@pytest.mark.parametrize(
    "window_size",
    [
        pytest.param((-1, -1), id="full"),
        pytest.param((8, 8), id="SWA"),
    ],
)
@pytest.mark.parametrize("n_kv_heads", [None, 4])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize(
    "backend_name",
    [
        AttentionBackendName.flash_2,
        AttentionBackendName.flash_3,
        pytest.param(AttentionBackendName.flash_4, id="flash_4", marks=FLASH_4_MARKS),
        AttentionBackendName.te,
    ],
)
@requires_gpu
def test_attention_backend(
    backend_name: AttentionBackendName,
    head_dim: int,
    n_heads: int,
    n_kv_heads: Optional[int],
    window_size: Tuple[int, int],
    dtype: torch.dtype = torch.bfloat16,
):
    if backend_name == AttentionBackendName.te and (
        compute_capability is None or compute_capability < 9
    ):
        pytest.skip("TransformerEngine fused attention requires Hopper (sm_90+)")
    try:
        backend_name.assert_supported()
        backend = backend_name.build(
            head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, window_size=window_size
        )
        default = AttentionBackendName.torch.build(
            head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, window_size=window_size
        )
    except RuntimeError as e:
        pytest.skip(str(e))

    seed_all(0)
    B, T = 2, 16

    q = torch.randn(B, T, n_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(B, T, n_kv_heads or n_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(B, T, n_kv_heads or n_heads, head_dim, device="cuda", dtype=dtype)

    att = backend((q, k, v)).view(B, T, -1)
    att_reference = default((q, k, v)).view(B, T, -1)
    torch.testing.assert_close(att, att_reference)


@pytest.mark.parametrize("attention_cls", [Attention, NormalizedAttention])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16", marks=GPU_MARKS),
        pytest.param(torch.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "n_kv_heads",
    [pytest.param(None, id="MHA"), pytest.param(1, id="MQA"), pytest.param(2, id="GQA")],
)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("flash_2", id="flash-attn-2", marks=FLASH_2_MARKS),
        pytest.param("flash_3", id="flash-attn-3", marks=FLASH_3_MARKS),
        pytest.param("flash_4", id="flash-attn-4", marks=FLASH_4_MARKS),
        pytest.param("torch", id="torch-SDPA"),
        pytest.param("te", id="te-attn", marks=TE_MARKS),
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"clip_qkv": 8.0}, id="QKV-clip"),
        pytest.param({"rope": RoPEConfig()}, id="rope"),
        pytest.param({"rope": RoPEConfig(name=RoPEType.complex)}, id="complex-rope"),
        pytest.param({"qk_norm": LayerNormConfig()}, id="qk-norm"),
        pytest.param({"qk_norm": LayerNormConfig(), "use_head_qk_norm": True}, id="head-qk-norm"),
        pytest.param(
            {
                "qk_norm": LayerNormConfig(),
                "use_head_qk_norm": True,
                "qk_norm_per_head_gains": True,
            },
            id="head-qk-norm-per-head-gains",
        ),
    ],
)
def test_attention(
    attention_cls,
    dtype: torch.dtype,
    device: torch.device,
    n_kv_heads: Optional[int],
    backend: str,
    kwargs: Dict[str, Any],
):
    if backend in ("flash_2", "flash_3", "flash_4") and dtype == torch.float32:
        pytest.skip("flash-attn requires a low precision dtype")
    if dtype == torch.bfloat16 and device.type == "cpu":
        pytest.skip("bf16 requires GPU")
    if backend == "te" and device.type != "cuda":
        pytest.skip("TransformerEngine attention requires a CUDA device")
    if backend == "te" and (compute_capability is None or compute_capability < 9):
        pytest.skip("TransformerEngine fused attention requires Hopper (sm_90+)")
    if attention_cls is NormalizedAttention:
        if "clip_qkv" in kwargs:
            pytest.skip("clip_qkv is not supported for NormalizedAttention")
        if "use_head_qk_norm" in kwargs:
            pytest.skip("use_head_qk_norm is not supported for NormalizedAttention")
        if backend in ("flash_2", "flash_3", "flash_4", "te"):
            pytest.xfail(
                f"NormalizedAttention is broken with '{backend}' backend because it creates activation tensors in fp32"
            )

    seed_all(0)

    d_model = 128
    seq_len = 32

    attention = attention_cls(
        d_model=d_model,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        backend=backend,
        init_device=device.type,
        **kwargs,
    )

    x1 = torch.randn(1, seq_len, d_model, dtype=dtype, device=device)
    x2 = torch.randn(1, seq_len, d_model, dtype=dtype, device=device)
    x = torch.cat([x1, x2])

    # Make sure batch outputs match individual outputs.
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        y1 = attention(x1)
        y2 = attention(x2)
        y = attention(x)

    torch.testing.assert_close(y[0:1, :, :], y1)
    torch.testing.assert_close(y[1:, :, :], y2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(AttentionBackendName.flash_2, id="flash-attn-2", marks=FLASH_2_MARKS),
        pytest.param(AttentionBackendName.flash_3, id="flash-attn-2", marks=FLASH_3_MARKS),
        pytest.param(AttentionBackendName.torch, id="torch-SDPA"),
        pytest.param(AttentionBackendName.te, id="te-attn", marks=TE_MARKS),
    ],
)
@pytest.mark.parametrize(
    "window_size",
    [pytest.param(None, id="full"), pytest.param(16, id="sliding")],
)
@pytest.mark.parametrize(
    "intra_doc_masking",
    [pytest.param(False, id="no-doc-masking"), pytest.param(True, id="doc-masking")],
)
def test_sdpa(
    device: torch.device,
    dtype: torch.dtype,
    backend_name: AttentionBackendName,
    window_size: Optional[int],
    intra_doc_masking: bool,
):
    if (
        backend_name in (AttentionBackendName.flash_2, AttentionBackendName.flash_3)
        and dtype == torch.float32
    ):
        pytest.skip("flash-attn requires a low precision dtype")
    if (
        backend_name
        in (AttentionBackendName.flash_2, AttentionBackendName.flash_3, AttentionBackendName.te)
        and device.type == "cpu"
    ):
        pytest.skip(f"{backend_name} backend requires GPU")
    if backend_name == AttentionBackendName.te and (
        compute_capability is None or compute_capability < 9
    ):
        pytest.skip("TransformerEngine fused attention requires Hopper (sm_90+)")
    if backend_name == AttentionBackendName.torch and intra_doc_masking:
        pytest.skip("intra-document masking is not supported by torch backend")

    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32
    batch_size = 2
    n_heads = 8
    if intra_doc_masking:
        doc_lens = torch.tensor([[0, 4, 16, 12], [8, 8, 8, 8]], dtype=torch.int32, device=device)
        max_doc_len = int(torch.max(doc_lens))
        cu_doc_lens = torch.cumsum(doc_lens.flatten(), dim=0, dtype=torch.int32)
        assert int(cu_doc_lens[-1]) == batch_size * seq_len
    else:
        doc_lens = None
        max_doc_len = None
        cu_doc_lens = None

    kwargs: Dict[str, Any] = dict(
        d_model=d_model,
        n_heads=8,
        init_device=device.type,
        window_size=window_size,
    )

    attention = Attention(backend=backend_name, **kwargs)

    q = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)

    with torch.no_grad():
        mask_len = batch_size * seq_len if intra_doc_masking else seq_len
        attn_mask = torch.ones(mask_len, mask_len, dtype=torch.bool, device=device).tril(diagonal=0)
        is_causal = False

        if window_size is not None:
            attn_mask = torch.logical_and(
                attn_mask,
                torch.ones(mask_len, mask_len, dtype=torch.bool, device=device).triu(
                    diagonal=1 - window_size
                ),
            )
        if intra_doc_masking:
            assert doc_lens is not None
            attn_mask = torch.logical_and(
                attn_mask,
                torch.block_diag(
                    *[
                        torch.ones(int(doc_len), int(doc_len), dtype=torch.bool, device=device)
                        for doc_len in doc_lens.flatten()
                    ]
                ),
            )

        if window_size is None and not intra_doc_masking:
            attn_mask = None
            is_causal = True

        # PyTorch's SDPA expects the head dimension to come before the sequence dimension.
        y1 = (
            F.scaled_dot_product_attention(
                q.view(q.shape[0] * q.shape[1] // mask_len, mask_len, *q.shape[2:]).transpose(1, 2),
                k.view(k.shape[0] * k.shape[1] // mask_len, mask_len, *k.shape[2:]).transpose(1, 2),
                v.view(v.shape[0] * v.shape[1] // mask_len, mask_len, *v.shape[2:]).transpose(1, 2),
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            .transpose(1, 2)
            .contiguous()
        )
        try:
            y2 = attention.sdpa(
                q,
                k,
                v,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            ).view_as(y1)
        except RuntimeError:
            if backend_name == AttentionBackendName.te and intra_doc_masking:
                pytest.xfail("intra-document masking is currently broken in te backend")
            raise

    torch.testing.assert_close(y1, y2)


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16")])
@pytest.mark.parametrize("backend", [AttentionBackendName.flash_2, AttentionBackendName.torch])
def test_fused_attention_v2_against_non_fused(dtype: torch.dtype, backend: AttentionBackendName):
    seed_all(0)

    d_model = 128
    seq_len = 32
    batch_size = 2
    kwargs: Dict[str, Any] = dict(
        d_model=d_model,
        n_heads=8,
        init_device="cuda",
    )

    attention = Attention(backend=backend, **kwargs)
    fused_att = FusedAttentionV2(backend=AttentionBackendName.flash_2, **kwargs)

    # Make sure weights match.
    with torch.no_grad():
        fused_att.w_out.load_state_dict(attention.w_out.state_dict())
        fused_att.w_qkv.weight.copy_(
            torch.cat([attention.w_q.weight, attention.w_k.weight, attention.w_v.weight])
        )
        fused_att.w_qkv.bias.copy_(
            torch.cat([attention.w_q.bias, attention.w_k.bias, attention.w_v.bias])
        )

    x1 = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")
    x2 = x1.clone()

    with torch.autocast("cuda", dtype=dtype, enabled=True):
        y1 = attention(x1)
        y2 = fused_att(x2)

    torch.testing.assert_close(y1, y2)


@requires_gpu
@requires_flash_attn_2
def test_fused_attention_v2_with_rope():
    seed_all(0)

    d_model = 128
    seq_len = 32

    fused_att = FusedAttentionV2(
        d_model=d_model,
        n_heads=8,
        rope=RoPEConfig(),
        init_device="cuda",
        backend=AttentionBackendName.flash_2,
    )

    x1 = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device="cuda")
    x2 = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device="cuda")
    x = torch.cat([x1, x2])

    # Make sure batch outputs match individual outputs.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y1 = fused_att(x1)
        y2 = fused_att(x2)
        y = fused_att(x)

    torch.testing.assert_close(y[0:1, :, :], y1)
    torch.testing.assert_close(y[1:, :, :], y2)


@requires_gpu
@requires_flash_attn_2
def test_attention_with_intra_document_masking():
    seed_all(0)

    d_model = 128
    seq_len = 32

    attention = Attention(
        d_model=d_model, n_heads=8, init_device="cuda", backend=AttentionBackendName.flash_2
    )
    fused_att = FusedAttentionV2(
        d_model=d_model, n_heads=8, init_device="cuda", backend=AttentionBackendName.flash_2
    )

    # Make sure weights match.
    with torch.no_grad():
        fused_att.w_out.load_state_dict(attention.w_out.state_dict())
        fused_att.w_qkv.weight.copy_(
            torch.cat([attention.w_q.weight, attention.w_k.weight, attention.w_v.weight])
        )
        fused_att.w_qkv.bias.copy_(
            torch.cat([attention.w_q.bias, attention.w_k.bias, attention.w_v.bias])
        )

    x = torch.randn(2, seq_len, d_model, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y1 = attention(x.clone())
        y2 = attention(
            x.clone(),
            max_doc_len=seq_len,
            cu_doc_lens=torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device="cuda"),
        )

        y1_fused = fused_att(x.clone())
        y2_fused = fused_att(
            x.clone(),
            max_doc_len=seq_len,
            cu_doc_lens=torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device="cuda"),
        )

    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(y1_fused, y2_fused)
    torch.testing.assert_close(y1, y1_fused)
    torch.testing.assert_close(y2, y2_fused)


@requires_gpu
@requires_compute_capability(min_cc=9)  # flash-attn bf16 precision is worse on A100s (cc=8)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "n_kv_heads",
    [pytest.param(None, id="MHA"), pytest.param(2, id="GQA")],
)
@pytest.mark.parametrize(
    "use_rope",
    [pytest.param(True, id="rope"), pytest.param(False, id="no-rope")],
)
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(AttentionBackendName.flash_2, id="flash-attn-2", marks=FLASH_2_MARKS),
        pytest.param(AttentionBackendName.flash_4, id="flash-attn-4", marks=FLASH_4_MARKS),
    ],
)
def test_attention_kv_caching(
    batch_size: int,
    n_kv_heads: Optional[int],
    use_rope: bool,
    backend_name: AttentionBackendName,
):
    seed_all(0)

    d_model = 512
    n_heads = 8
    max_seq_len = 512
    prefill_len = 508
    decode_steps = 1
    total_len = prefill_len + decode_steps
    assert total_len <= max_seq_len

    # Initialize attention module
    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope=RoPEConfig() if use_rope else None,
        backend=backend_name,
        init_device="cuda",
        dtype=torch.float32,
    )

    # Input tensor
    x = torch.randn(batch_size, total_len, d_model, dtype=torch.bfloat16, device="cuda")

    # 1. Combined forward pass (for comparison)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_combined = attention(x)

    # 2. Prefill + multiple decode steps with KV cache
    attention.init_kv_cache_manager(batch_size, max_seq_len)
    x_prefill = x[:, :prefill_len, :]
    attention_mask = torch.ones(batch_size, prefill_len, dtype=torch.bool, device="cuda")
    cache_leftpad = attention_mask_to_cache_leftpad(attention_mask)

    # First pass with allocated KV cache - this will populate the cache
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_prefill = attention(x_prefill, cache_leftpad=cache_leftpad)

    # Multiple decode steps
    y_decode_steps = []
    for step in range(decode_steps):
        x_decode = x[:, prefill_len + step : prefill_len + step + 1, :]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y_decode = attention(x_decode, cache_leftpad=None)
        y_decode_steps.append(y_decode)
    y_decode_combined = torch.cat(y_decode_steps, dim=1)

    # 3. Compare results
    # Check that prefill output matches the corresponding part of the combined output
    torch.testing.assert_close(
        y_combined[:, :prefill_len, :],
        y_prefill,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
        msg="Prefill outputs don't match",
    )

    # Check that decode outputs match the corresponding part of the combined output
    torch.testing.assert_close(
        y_combined[:, prefill_len:, :],
        y_decode_combined,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
        msg="Decode outputs with KV-cache don't match",
    )


@requires_gpu
@requires_compute_capability(min_cc=9)
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(AttentionBackendName.flash_2, id="flash-attn-2", marks=FLASH_2_MARKS),
        pytest.param(AttentionBackendName.flash_4, id="flash-attn-4", marks=FLASH_4_MARKS),
    ],
)
def test_attention_kv_cache_update(backend_name: AttentionBackendName):
    seed_all(0)

    d_model = 512
    n_heads = 8
    n_kv_heads = 2
    batch_size = 2
    max_seq_len = 64
    prefill_len = 30
    decode_steps = 5
    dtype = torch.bfloat16

    # Initialize attention module
    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        backend=backend_name,
        init_device="cuda",
        dtype=torch.float32,
    )

    # Initialize cache
    attention.init_kv_cache_manager(batch_size, max_seq_len)
    assert attention.kv_cache_manager is not None

    # Prefill
    prefill_input = torch.randn(batch_size, prefill_len, d_model, dtype=dtype, device="cuda")
    attention_mask = torch.ones(batch_size, prefill_len, dtype=torch.bool, device="cuda")
    cache_leftpad = attention_mask_to_cache_leftpad(attention_mask)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        attention(prefill_input, cache_leftpad=cache_leftpad)

    k_at_prev_write_pos: Optional[torch.Tensor] = None
    v_at_prev_write_pos: Optional[torch.Tensor] = None

    for step in range(decode_steps):
        # Store cache state before the decode step.
        k_cache_before = attention.kv_cache_manager.k_cache.clone()
        v_cache_before = attention.kv_cache_manager.v_cache.clone()
        cache_seqlens_before = attention.kv_cache_manager.cache_seqlens.clone()

        decode_input = torch.randn(batch_size, 1, d_model, dtype=dtype, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            attention(decode_input, cache_leftpad=None)

        # Check that cache has been updated.
        assert not torch.equal(k_cache_before, attention.kv_cache_manager.k_cache)
        assert not torch.equal(v_cache_before, attention.kv_cache_manager.v_cache)
        torch.testing.assert_close(
            attention.kv_cache_manager.cache_seqlens, cache_seqlens_before + 1
        )

        # Check that the update happened at the right position.
        current_write_pos = int(cache_seqlens_before.item())
        k_cache_after = attention.kv_cache_manager.k_cache
        v_cache_after = attention.kv_cache_manager.v_cache

        # Check that the cache at the new token position is not all zeros.
        for b in range(batch_size):
            assert not torch.all(k_cache_after[b, current_write_pos] == 0)
            assert not torch.all(v_cache_after[b, current_write_pos] == 0)

        # Check that the cache *before* the new token is unchanged.
        torch.testing.assert_close(
            k_cache_before[:, :current_write_pos, :, :],
            k_cache_after[:, :current_write_pos, :, :],
        )
        torch.testing.assert_close(
            v_cache_before[:, :current_write_pos, :, :],
            v_cache_after[:, :current_write_pos, :, :],
        )

        # Check that the cache *after* the new token is unchanged.
        torch.testing.assert_close(
            k_cache_before[:, current_write_pos + 1 :, :, :],
            k_cache_after[:, current_write_pos + 1 :, :, :],
        )
        torch.testing.assert_close(
            v_cache_before[:, current_write_pos + 1 :, :, :],
            v_cache_after[:, current_write_pos + 1 :, :, :],
        )

        # Ensure previous write is untouched.
        if step > 0:
            assert k_at_prev_write_pos is not None and v_at_prev_write_pos is not None
            prev_write_pos = current_write_pos - 1
            for b in range(batch_size):
                torch.testing.assert_close(
                    k_at_prev_write_pos[b],
                    k_cache_after[b, prev_write_pos],
                    msg=f"step {step}, batch {b}",
                )
                torch.testing.assert_close(
                    v_at_prev_write_pos[b],
                    v_cache_after[b, prev_write_pos],
                    msg=f"step {step}, batch {b}",
                )

        # Store the written slice for the next iteration's check.
        k_at_prev_write_pos = torch.stack(
            [k_cache_after[b, current_write_pos] for b in range(batch_size)]
        ).clone()
        v_at_prev_write_pos = torch.stack(
            [v_cache_after[b, current_write_pos] for b in range(batch_size)]
        ).clone()


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("batch_size", [1, 8])
def test_attention_prefill_forward_pass(batch_size: int):
    seed_all(0)

    d_model = 64
    n_heads = 4
    max_seq_len = 128
    seq_len = 124
    dtype = torch.bfloat16
    attention = Attention(
        d_model=d_model, n_heads=n_heads, backend=AttentionBackendName.flash_2, init_device="cuda"
    )

    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")

    # Standard forward pass without KV cache
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_standard = attention(x)

    # Forward pass with KV cache allocated
    attention.init_kv_cache_manager(batch_size, max_seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device="cuda")
    cache_leftpad = attention_mask_to_cache_leftpad(attention_mask)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_with_cache = attention(x, cache_leftpad=cache_leftpad)

    torch.testing.assert_close(y_standard, y_with_cache)


@requires_gpu
@requires_flash_attn_2
def test_attention_kv_cache_write_position():
    """Test KV caching with left-padded attention masks."""
    seed_all(0)

    batch_size = 2
    d_model = 128
    n_heads = 8
    max_seq_len = 100
    dtype = torch.bfloat16

    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        backend=AttentionBackendName.flash_2,
        init_device="cuda",
        dtype=torch.float32,
    )

    # Create inputs with different sequence lengths (simulated with left padding)
    # Sequence 1: 3 padding tokens + 7 real tokens
    # Sequence 2: 5 padding tokens + 5 real tokens
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")

    # Create attention mask with left padding
    attention_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # 3 padding tokens
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 5 padding tokens
        ],
        dtype=torch.bool,
        device="cuda",
    )

    # Convert to cache_leftpad
    cache_leftpad = attention_mask_to_cache_leftpad(attention_mask)
    assert cache_leftpad.tolist() == [3, 5]

    # 1. Test prefill with KV cache
    attention.init_kv_cache_manager(batch_size, max_seq_len)
    assert attention.kv_cache_manager is not None

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_prefill = attention(x, cache_leftpad=cache_leftpad)

    assert y_prefill.shape == (batch_size, seq_len, d_model)
    torch.testing.assert_close(attention.kv_cache_manager.cache_leftpad, cache_leftpad)

    # Check zero/non-zero structure in the cache after prefill
    k_cache = attention.kv_cache_manager.k_cache
    v_cache = attention.kv_cache_manager.v_cache
    for i in range(batch_size):
        lp = int(cache_leftpad[i].item())
        content_len = seq_len - lp
        # Padded region should be zero
        if lp > 0:
            assert torch.all(k_cache[i, :lp] == 0)
            assert torch.all(v_cache[i, :lp] == 0)
        # Content region should be non-zero
        assert not torch.all(k_cache[i, lp : lp + content_len] == 0)
        assert not torch.all(v_cache[i, lp : lp + content_len] == 0)
        # Region after content should be zero
        assert torch.all(k_cache[i, lp + content_len :] == 0)
        assert torch.all(v_cache[i, lp + content_len :] == 0)

    # 2. Test incremental decoding
    new_token = torch.randn(batch_size, 1, d_model, dtype=dtype, device="cuda")
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()
    seqlens_before = attention.kv_cache_manager.cache_seqlens.clone()

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y_decode = attention(new_token, cache_leftpad=None)

    assert y_decode.shape == (batch_size, 1, d_model)
    assert torch.all(attention.kv_cache_manager.cache_seqlens == (seqlens_before + 1))

    # Verify that only the new write position per batch changed
    for i in range(batch_size):
        write_pos = int(seqlens_before)
        k_cache_new = attention.kv_cache_manager.k_cache[i]
        v_cache_new = attention.kv_cache_manager.v_cache[i]

        # Regions before the write position should be unchanged
        if write_pos > 0:
            torch.testing.assert_close(k_cache_before[i, :write_pos], k_cache_new[:write_pos])
            torch.testing.assert_close(v_cache_before[i, :write_pos], v_cache_new[:write_pos])

        # The write position must now be non-zero
        assert not torch.all(k_cache_new[write_pos] == 0)
        assert not torch.all(v_cache_new[write_pos] == 0)

        # Region after the write position should remain zeros (unchanged)
        torch.testing.assert_close(k_cache_before[i, write_pos + 1 :], k_cache_new[write_pos + 1 :])
        torch.testing.assert_close(v_cache_before[i, write_pos + 1 :], v_cache_new[write_pos + 1 :])


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("use_rope", [True, False], ids=["rope", "no-rope"])
def test_attention_leftpad_shift_equivalence(use_rope):
    """The same content, presented with different left-padding, should produce identical outputs on the valid region."""
    seed_all(0)

    d_model = 128
    n_heads = 8
    dtype = torch.bfloat16
    kv_cache_max_len = 100

    # Shared content of length L
    len_content = 7
    x_shared = torch.randn(1, len_content, d_model, dtype=dtype, device="cuda")
    x_next_shared = torch.randn(1, 1, d_model, dtype=dtype, device="cuda")

    # Two different left-padding amounts for the same content
    pad_a = 3
    pad_b = 8

    # Build masks to derive correct cache_leftpad and seq_lens
    max_len_a = pad_a + len_content
    mask_a = torch.tensor([[0] * pad_a + [1] * len_content], dtype=torch.bool, device="cuda")
    cache_leftpad_a = attention_mask_to_cache_leftpad(mask_a)

    max_len_b = pad_b + len_content
    mask_b = torch.tensor([[0] * pad_b + [1] * len_content], dtype=torch.bool, device="cuda")
    cache_leftpad_b = attention_mask_to_cache_leftpad(mask_b)

    # Build left-padded inputs so padding tokens are present and must be ignored by the kernel
    x_a = torch.zeros(1, max_len_a, d_model, dtype=dtype, device="cuda")
    x_b = torch.zeros(1, max_len_b, d_model, dtype=dtype, device="cuda")
    x_a[:, -len_content:, :] = x_shared
    x_b[:, -len_content:, :] = x_shared

    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        rope=RoPEConfig() if use_rope else None,
        backend=AttentionBackendName.flash_2,
        init_device="cuda",
        dtype=torch.float32,
    )

    # Run with leftpad A
    attention.init_kv_cache_manager(1, kv_cache_max_len)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        # Prefill
        y_a = attention(x_a, cache_leftpad=cache_leftpad_a)

        # Decode one more token using the KV cache
        y_a_next = attention(x_next_shared)

    # Run with leftpad B
    attention.init_kv_cache_manager(1, kv_cache_max_len)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        # Prefill
        y_b = attention(x_b, cache_leftpad=cache_leftpad_b)

        # Decode one more token using the KV cache (same next token content)
        y_b_next = attention(x_next_shared)

    # Without RoPE, leftpad shift should not change outputs on the valid region.
    torch.testing.assert_close(
        y_a[:, -len_content:, :], y_b[:, -len_content:, :], rtol=BF16_RTOL, atol=BF16_ATOL
    )

    # Also validate the decode step equivalence (single-token outputs should match)
    torch.testing.assert_close(y_a_next, y_b_next, rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.parametrize(
    "attn_config",
    [
        AttentionConfig(name=AttentionType.default, n_heads=8, n_kv_heads=1, bias=True),
        AttentionConfig(name=AttentionType.default, n_heads=8, n_kv_heads=1, bias=False),
        AttentionConfig(
            name=AttentionType.default, n_heads=8, bias=False, qk_norm=LayerNormConfig()
        ),
        # GQA with QK norm - regression test for k_norm size calculation
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=8,
                n_kv_heads=2,
                bias=False,
                qk_norm=LayerNormConfig(),
            ),
            id="GQA-qk-norm",
        ),
        # MQA with QK norm - regression test for k_norm size calculation
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=8,
                n_kv_heads=1,
                bias=False,
                qk_norm=LayerNormConfig(),
            ),
            id="MQA-qk-norm",
        ),
        # OLMo 3 32B-like config (scaled down) - regression test for k_norm size calculation
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=40,
                n_kv_heads=8,
                bias=False,
                qk_norm=LayerNormConfig(),
            ),
            id="OLMo3-32B-like-qk-norm",
        ),
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=8,
                bias=False,
                qk_norm=LayerNormConfig(),
                gate=GateConfig(granularity=GateGranularity.headwise),
            ),
            id="headwise-gating",
        ),
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=8,
                bias=False,
                scalable_softmax=True,
            ),
            id="scalable-softmax",
        ),
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=8,
                n_kv_heads=2,
                bias=False,
                qk_norm=LayerNormConfig(),
                use_head_qk_norm=True,
            ),
            id="GQA-head-qk-norm",
        ),
        pytest.param(
            AttentionConfig(
                name=AttentionType.default,
                n_heads=8,
                n_kv_heads=2,
                bias=False,
                qk_norm=LayerNormConfig(),
                use_head_qk_norm=True,
                qk_norm_per_head_gains=True,
            ),
            id="GQA-head-qk-norm-per-head-gains",
        ),
    ],
)
def test_attention_builder_config(attn_config: AttentionConfig):
    # Use d_model that's divisible by max n_heads in our test configs (40)
    d_model = 160

    attn = attn_config.build(d_model, layer_idx=0, n_layers=1)

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in attn.parameters())
    assert attn_config.num_params(d_model) == n_params


def test_qk_norm_per_head_gains():
    d_model, n_heads, n_kv_heads = 128, 8, 2
    head_dim = d_model // n_heads

    attn = Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        qk_norm=LayerNormConfig(),
        use_head_qk_norm=True,
        qk_norm_per_head_gains=True,
    )
    assert attn.q_norm is not None and attn.k_norm is not None
    assert attn.q_norm.weight.shape == (n_heads, head_dim)
    assert attn.k_norm.weight.shape == (n_kv_heads, head_dim)
    # Statistics should still be computed per-head over 'head_dim'.
    assert attn.q_norm.normalized_shape == (head_dim,)
    assert attn.k_norm.normalized_shape == (head_dim,)

    with pytest.raises(OLMoConfigurationError, match="use_head_qk_norm"):
        Attention(
            d_model=d_model,
            n_heads=n_heads,
            qk_norm=LayerNormConfig(),
            qk_norm_per_head_gains=True,
        )


@pytest.mark.parametrize(
    "cu_doc_lens,expected_lengths",
    [
        pytest.param(None, [[1, 2, 3, 4], [1, 2, 3, 4]], id="causal"),
        pytest.param(
            torch.tensor([0, 2, 4, 7, 8], dtype=torch.int32),
            [[1, 2, 1, 2], [1, 2, 3, 1]],
            id="packed-documents",
        ),
    ],
)
def test_scalable_softmax_query_scaling(cu_doc_lens, expected_lengths):
    attention = Attention(
        d_model=8,
        n_heads=2,
        head_dim=4,
        bias=False,
        scalable_softmax=True,
    )
    assert attention.ssmax_scale is not None
    with torch.no_grad():
        attention.ssmax_scale.copy_(torch.tensor([0.5, 2.0]))

    q = torch.ones(2, 4, 2, 4)
    scaled_q = attention._apply_scalable_softmax(q, cu_doc_lens)
    expected = torch.tensor(expected_lengths, dtype=q.dtype).log()
    expected = expected[:, :, None, None] * torch.tensor([0.5, 2.0])[None, None, :, None]

    torch.testing.assert_close(scaled_q, expected.expand_as(q))
    scaled_q.sum().backward()
    assert attention.ssmax_scale.grad is not None
    assert torch.all(attention.ssmax_scale.grad > 0)


def test_scalable_softmax_disabled_is_identity():
    attention = Attention(d_model=8, n_heads=2, head_dim=4, bias=False)
    q = torch.randn(2, 4, 2, 4)

    assert attention.ssmax_scale is None
    assert attention._apply_scalable_softmax(q, None) is q


def test_scalable_softmax_scale_is_initialized_to_one():
    attention = Attention(
        d_model=8,
        n_heads=2,
        head_dim=4,
        bias=False,
        scalable_softmax=True,
    )
    assert attention.ssmax_scale is not None
    with torch.no_grad():
        attention.ssmax_scale.fill_(float("nan"))

    attention.init_weights(
        init_method=InitMethod.normal,
        d_model=8,
        block_idx=0,
        num_blocks=1,
    )

    torch.testing.assert_close(attention.ssmax_scale, torch.ones(2))


@pytest.mark.parametrize("flag", ["scalable_softmax", "qk_norm_per_head_gains"])
@pytest.mark.parametrize("name", [AttentionType.normalized, AttentionType.fused_v2])
def test_attention_rejects_unsupported_hybrid_extensions(flag, name):
    config = AttentionConfig(name=name, n_heads=2)
    setattr(config, flag, True)
    with pytest.raises(OLMoConfigurationError, match=f"not supported with {name} attention"):
        config.build(d_model=8, layer_idx=0, n_layers=1)


@pytest.mark.parametrize("mode", ["context-parallel", "kv-cache"])
def test_scalable_softmax_rejects_unsupported_setup(mode):
    attention = Attention(d_model=8, n_heads=2, scalable_softmax=True)
    with pytest.raises(OLMoConfigurationError, match="Scalable-Softmax"):
        if mode == "context-parallel":
            attention.apply_cp(None)
        else:
            attention.init_kv_cache_manager(1, 8)


def test_scalable_softmax_is_applied_after_qk_norm(monkeypatch):
    class ConstantNorm(nn.Module):
        def forward(self, x):
            return torch.full_like(x, 3.0)

    attention = Attention(
        d_model=8,
        n_heads=2,
        head_dim=4,
        bias=False,
        qk_norm=LayerNormConfig(),
        use_head_qk_norm=True,
        scalable_softmax=True,
    )
    attention.q_norm = ConstantNorm()
    attention.k_norm = ConstantNorm()
    assert attention.ssmax_scale is not None
    with torch.no_grad():
        attention.ssmax_scale.fill_(2.0)

    captured_q = None

    def capture_sdpa(q, k, v, **kwargs):
        nonlocal captured_q
        captured_q = q
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "sdpa", capture_sdpa)
    attention(torch.randn(1, 3, 8))

    assert captured_q is not None
    expected = 6.0 * torch.arange(1, 4, dtype=captured_q.dtype).log()
    expected = expected.view(1, 3, 1, 1).expand_as(captured_q)
    torch.testing.assert_close(captured_q, expected)


@pytest.mark.parametrize("unsupported_mode", ["kv-cache", "context-parallel"])
def test_scalable_softmax_rejects_unsupported_runtime_modes(unsupported_mode):
    attention = Attention(
        d_model=8,
        n_heads=2,
        head_dim=4,
        bias=False,
        scalable_softmax=True,
    )
    if unsupported_mode == "kv-cache":
        attention.kv_cache_manager = nn.Identity()
        match = "KV caching"
    else:
        attention.backend.cp_enabled = True
        match = "context parallelism"

    with pytest.raises(NotImplementedError, match=match):
        attention._apply_scalable_softmax(torch.ones(1, 2, 2, 4), None)


def test_scalable_softmax_rejects_sliding_window_attention():
    config = AttentionConfig(
        n_heads=2,
        scalable_softmax=True,
        sliding_window=SlidingWindowAttentionConfig(
            pattern=[128],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        ),
    )

    with pytest.raises(OLMoConfigurationError, match="scalable_softmax"):
        config.build(d_model=8, layer_idx=0, n_layers=2)


@pytest.mark.parametrize(
    "no_global_rope,expected_rope_enabled",
    [
        pytest.param(True, False, id="no_global_rope=True-disables"),
        pytest.param(False, True, id="no_global_rope=False-enables"),
        pytest.param(None, True, id="no_global_rope-default-enables"),
    ],
)
def test_no_global_rope_on_global_layers(
    no_global_rope: Optional[bool], expected_rope_enabled: bool
):
    """Test that no_global_rope controls RoPE on global (non-SWA) layers."""
    d_model = 128
    rope_config = (
        RoPEConfig(no_global_rope=no_global_rope) if no_global_rope is not None else RoPEConfig()
    )
    attn_config = AttentionConfig(
        name=AttentionType.default,
        n_heads=8,
        rope=rope_config,
    )

    # Build a global layer (no sliding window)
    attn = attn_config.build(d_model, layer_idx=0, n_layers=1)

    if expected_rope_enabled:
        assert attn.rope is not None
    else:
        assert attn.rope is None


@pytest.mark.parametrize(
    "force_full_attention,expected_rope_enabled",
    [
        pytest.param(False, True, id="swa-layer-preserves-rope"),
        pytest.param(True, False, id="forced-full-attention-disables-rope"),
    ],
)
def test_no_global_rope_with_sliding_window(
    force_full_attention: bool, expected_rope_enabled: bool
):
    """Test that no_global_rope=True only affects global layers, not SWA layers."""
    d_model = 128
    rope_config = RoPEConfig(no_global_rope=True)
    sliding_window_config = SlidingWindowAttentionConfig(
        pattern=[1024, 2048, -1],
        force_full_attention_on_first_layer=force_full_attention,
        force_full_attention_on_last_layer=False,
    )
    attn_config = AttentionConfig(
        name=AttentionType.default,
        n_heads=8,
        rope=rope_config,
        sliding_window=sliding_window_config,
    )

    # Build layer_idx=0
    # - If force_full_attention=False: uses SWA (window_size=1024), so RoPE should be preserved
    # - If force_full_attention=True: forced to full attention (global), so RoPE should be disabled
    attn = attn_config.build(d_model, layer_idx=0, n_layers=12)

    if expected_rope_enabled:
        assert attn.rope is not None
    else:
        assert attn.rope is None


@pytest.mark.parametrize(
    "force_first, force_last, layer_idx, expected_window_size, expected_should_use_swa",
    [
        # Test with forcing full attention on neither first nor last layer.
        (False, False, 0, 1024, True),  # Pattern start
        (False, False, 1, 2048, True),  # Pattern middle
        (False, False, 2, -1, False),  # Pattern end
        (False, False, 11, -1, False),  # Last layer, pattern end
        (True, False, 1, 1024, True),  # Effective layer=0
        (True, False, 11, 2048, True),  # Effective layer=10
        # Test with forcing full attention only on the last layer.
        (False, True, 0, 1024, True),  # First layer, not forced
        (False, True, 11, -1, False),  # Forced last
        # Test with forcing full attention on both first and last layers.
        (True, True, 0, -1, False),  # Forced first
        (True, True, 1, 1024, True),  # Effective layer=0
        (True, True, 11, -1, False),  # Forced last
    ],
)
def test_sliding_window_attention_config_window_size(
    force_first: bool,
    force_last: bool,
    layer_idx: int,
    expected_window_size: int,
    expected_should_use_swa: bool,
):
    n_layers = 12
    pattern = [1024, 2048, -1]

    config = SlidingWindowAttentionConfig(
        pattern=pattern,
        force_full_attention_on_first_layer=force_first,
        force_full_attention_on_last_layer=force_last,
    )

    assert config._get_window_size(layer_idx, n_layers) == expected_window_size
    assert config.should_use_swa(layer_idx, n_layers) == expected_should_use_swa


def test_sliding_window_attention_config_get_window_size_error():
    n_layers = 12
    pattern = [1024, 2048, -1]
    config = SlidingWindowAttentionConfig(
        pattern=pattern,
        force_full_attention_on_first_layer=True,
        force_full_attention_on_last_layer=True,
    )

    assert config.get_window_size(1, n_layers) == 1024  # This layer uses SWA
    with pytest.raises(ValueError):
        config.get_window_size(0, n_layers)  # This layer uses full attention


def test_sliding_window_attention_config_invalid_pattern_error():
    with pytest.raises(OLMoConfigurationError):
        bad_config = SlidingWindowAttentionConfig(
            pattern=[0], force_full_attention_on_first_layer=False
        )
        bad_config._get_window_size(0, n_layers=12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16", marks=GPU_MARKS),
        pytest.param(torch.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "gate_granularity", [GateGranularity.headwise, GateGranularity.elementwise]
)
@pytest.mark.parametrize("full_precision", [True, False])
def test_attention_gating(
    device: torch.device,
    dtype: torch.dtype,
    gate_granularity: GateGranularity,
    full_precision: bool,
):
    seed_all(0)

    d_model = 64
    n_heads = 4
    seq_len = 8
    batch_size = 2

    attention = Attention(
        d_model=d_model,
        n_heads=n_heads,
        gate=GateConfig(granularity=gate_granularity, full_precision=full_precision),
        backend=AttentionBackendName.torch,
        init_device=device.type,
    )

    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)

    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        y = attention(x)
        y.sum().backward()


def _run_tensor_parallel_attention(
    checkpoint_dir: str, inputs_path: str, outputs_path: str, attn_kwargs: Dict[str, Any]
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    attn = Attention(init_device=device.type, **attn_kwargs)

    # Shard sequence dim in/out like the transformer block does.
    attn.apply_tp(mesh["tp"], input_layout=Shard(1), output_layout=Shard(1), use_local_output=False)
    load_model_and_optim_state(checkpoint_dir, attn)

    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk = x.size(1) // world_size
    x_local = x[:, rank * chunk : (rank + 1) * chunk, :]
    y_local = attn(x_local)

    # Backward to exercise graph in TP mode.
    y_local.sum().backward()

    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk : (rank + 1) * chunk, :]
    torch.testing.assert_close(y_ref_local, y_local.to_local())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "attn_kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"rope": RoPEConfig()}, id="rope"),
        pytest.param({"qk_norm": LayerNormConfig()}, id="qk-layernorm"),
        pytest.param({"qk_norm": LayerNormConfig(), "rope": RoPEConfig()}, id="qk-layernorm-rope"),
        pytest.param(
            {"qk_norm": LayerNormConfig(), "use_head_qk_norm": True},
            id="headwise-qk-layernorm",
        ),
        pytest.param(
            {"qk_norm": LayerNormConfig(), "use_head_qk_norm": True, "rope": RoPEConfig()},
            id="headwise-qk-layernorm-rope",
        ),
        pytest.param(
            {
                "qk_norm": LayerNormConfig(),
                "use_head_qk_norm": True,
                "qk_norm_per_head_gains": True,
            },
            id="headwise-qk-layernorm-per-head-gains",
        ),
        pytest.param(
            {"gate": GateConfig(granularity=GateGranularity.headwise)},
            id="headwise-gating",
        ),
        pytest.param(
            {
                "qk_norm": LayerNormConfig(),
                "use_head_qk_norm": True,
                "scalable_softmax": True,
            },
            id="scalable-softmax",
        ),
    ],
)
def test_tensor_parallel_attention(backend: str, attn_kwargs: Dict[str, Any], tmp_path):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")

    seed_all(0)
    attn_kwargs.update({"d_model": 128, "n_heads": 8, "backend": AttentionBackendName.torch})
    attn = Attention(init_device=device.type, **attn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, attn_kwargs["d_model"], device=device)
    y = attn(x)

    outputs_path = tmp_path / "attn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "attn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, attn)

    run_distributed_test(
        _run_tensor_parallel_attention,
        backend=backend,
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, attn_kwargs),
    )


def _run_context_parallel_attention_ring(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    attn_kwargs: Dict[str, Any],
    load_balancer_type,
    head_stride: int,
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    attn = Attention(init_device=device.type, **attn_kwargs)
    ring_style = RingContextParallelStyle(load_balancer=load_balancer_type, head_stride=head_stride)
    attn.apply_cp(mesh["cp"], ring=ring_style)
    load_model_and_optim_state(checkpoint_dir, attn)

    # Load the input and split it across ranks on the sequence dimension.
    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk_size = x.size(1) // world_size
    x_local = x[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    with torch.autocast(device.type, dtype=x_local.dtype):
        y_local = attn(x_local)

    # Backward to exercise graph in CP mode.
    y_local.sum().backward()

    # Load the reference output and split it across ranks on the sequence dimension.
    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    # Compare the local output with the reference output.
    torch.testing.assert_close(y_ref_local, y_local, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_multi_gpu
@requires_flash_attn_2
@pytest.mark.parametrize(
    "load_balancer_type",
    [pytest.param(RingAttentionLoadBalancerType.zig_zag, id="zig_zag")],
)
@pytest.mark.parametrize("head_stride", [pytest.param(1), pytest.param(8)])
@pytest.mark.skip("known precision issues with ring-flash-attn")
def test_context_parallel_attention(load_balancer_type, head_stride: int, tmp_path):
    seed_all(0)
    device = torch.device("cuda")

    # CP requires flash-attn and low precision dtypes.
    attn_kwargs: Dict[str, Any] = {
        "d_model": 128,
        "n_heads": 8,
        "backend": AttentionBackendName.flash_2,
    }
    attn = Attention(init_device=device.type, **attn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, attn_kwargs["d_model"], device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = attn(x)

    outputs_path = tmp_path / "attn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "attn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, attn)

    run_distributed_test(
        _run_context_parallel_attention_ring,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            inputs_path,
            outputs_path,
            attn_kwargs,
            load_balancer_type,
            head_stride,
        ),
    )


def _run_context_parallel_attention_ulysses(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    attn_kwargs: Dict[str, Any],
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    attn = Attention(init_device=device.type, **attn_kwargs)
    attn.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    load_model_and_optim_state(checkpoint_dir, attn)

    # Load the input and split it across ranks on the sequence dimension.
    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk_size = x.size(1) // world_size
    x_local = x[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    with torch.autocast(device.type, dtype=x_local.dtype):
        y_local = attn(x_local)

    # Backward to exercise graph in CP mode.
    y_local.sum().backward()

    # Load the reference output and split it across ranks on the sequence dimension.
    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    # Compare the local output with the reference output.
    tol_scale = 2  # requires slightly more tolerance than default
    torch.testing.assert_close(
        y_ref_local, y_local, rtol=BF16_RTOL * tol_scale, atol=BF16_ATOL * tol_scale
    )


@requires_multi_gpu
@pytest.mark.parametrize(
    "attn_backend,scalable_softmax",
    [
        pytest.param(AttentionBackendName.torch, False, id="torch-SDPA"),
        pytest.param(AttentionBackendName.flash_2, False, id="flash-attn-2", marks=FLASH_2_MARKS),
        pytest.param(
            AttentionBackendName.flash_2,
            True,
            id="flash-attn-2-scalable-softmax",
            marks=FLASH_2_MARKS,
        ),
        pytest.param(AttentionBackendName.flash_3, False, id="flash-attn-3", marks=FLASH_3_MARKS),
        pytest.param(
            AttentionBackendName.te,
            False,
            id="te-attn",
            marks=(
                *TE_MARKS,
                # te-attn Ulysses CP is flaky on Hopper (intermittent numerical/collective
                # failure); strict=False so it stays green whether it fails or passes.
                pytest.mark.xfail(
                    reason="te-attn Ulysses CP is flaky on Hopper",
                    strict=False,
                ),
            ),
        ),
    ],
)
def test_context_parallel_attention_ulysses(
    tmp_path, attn_backend: AttentionBackendName, scalable_softmax: bool
):
    """
    Test Ulysses-style context parallelism.

    Unlike ring attention, Ulysses-style CP uses all-to-all communication to gather the full
    sequence while partitioning heads, then runs standard flash attention, and finally uses
    all-to-all to restore the sequence-partitioned layout. This doesn't require a load balancer
    or ring-flash-attn.
    """
    seed_all(0)
    device = get_default_device()

    if attn_backend == AttentionBackendName.te and (
        compute_capability is None or compute_capability < 9
    ):
        pytest.skip("TransformerEngine fused attention requires Hopper (sm_90+)")

    # n_heads must be divisible by CP degree (world_size).
    attn_kwargs: Dict[str, Any] = {
        "d_model": 128,
        "n_heads": 8,
        "backend": attn_backend,
        "scalable_softmax": scalable_softmax,
    }
    attn = Attention(init_device=device.type, **attn_kwargs)
    if device.type == "cpu":
        attn = attn.to(dtype=torch.bfloat16)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, attn_kwargs["d_model"], device=device, dtype=torch.bfloat16)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"
    ):
        y = attn(x)

    outputs_path = tmp_path / "attn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "attn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, attn)

    run_distributed_test(
        _run_context_parallel_attention_ulysses,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            inputs_path,
            outputs_path,
            attn_kwargs,
        ),
    )


def test_attention_num_flops_per_token():
    d_model = 128
    n_heads = 8
    seq_len = 32

    def _flops_per_token(
        *,
        n_kv_heads: Optional[int],
        window_size: Optional[int],
        gate: Optional[GateConfig],
    ) -> int:
        attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            window_size=window_size,
            gate=gate,
            backend=AttentionBackendName.torch,
            init_device="cpu",
        )
        return attn.num_flops_per_token(seq_len)

    mha_full = _flops_per_token(n_kv_heads=None, window_size=None, gate=None)
    mha_swa = _flops_per_token(n_kv_heads=None, window_size=16, gate=None)
    gqa_full = _flops_per_token(n_kv_heads=2, window_size=None, gate=None)
    gqa_swa = _flops_per_token(n_kv_heads=2, window_size=16, gate=None)
    mha_full_gated = _flops_per_token(
        n_kv_heads=None,
        window_size=None,
        gate=GateConfig(granularity=GateGranularity.headwise),
    )

    # NOTE: we use relative comparisons here rather than direct comparisons to record_flops() for
    # the Attention estimate, since the idealized estimate used by Attention is different from the
    # actual FLOPs used by SDPA/flash-attn due to the use of recomputation in the backward pass.

    # full attention should be more expensive than SWA.
    assert mha_full > mha_swa
    assert gqa_full > gqa_swa

    # MHA should be more expensive than GQA.
    assert mha_full > gqa_full
    assert mha_swa > gqa_swa

    # Gating should add additional compute.
    assert mha_full_gated > mha_full


def test_fused_attention_v2_num_flops_per_token():
    n_heads = 8

    fused_small = FusedAttentionV2(d_model=128, n_heads=n_heads, init_device="cpu")

    # Compare against the basic Attention estimate for the same configuration.
    attn_small = Attention(
        d_model=128,
        n_heads=n_heads,
        backend=AttentionBackendName.torch,
        init_device="cpu",
    )
    assert fused_small.num_flops_per_token(32) == attn_small.num_flops_per_token(32)

    # Longer sequences should be more expensive.
    assert fused_small.num_flops_per_token(64) > fused_small.num_flops_per_token(32)

    # Larger models should be more expensive.
    fused_large = FusedAttentionV2(d_model=256, n_heads=n_heads, init_device="cpu")
    assert fused_large.num_flops_per_token(32) > fused_small.num_flops_per_token(32)


def test_attention_sinks_num_params_and_build():
    config = AttentionConfig(
        name=AttentionType.default,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        bias=True,
        attention_sinks=True,
    )
    without = AttentionConfig(
        name=AttentionType.default,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        bias=True,
    )
    # Sinks add exactly one learnable logit per head.
    assert config.num_params(32) == without.num_params(32) + 4

    attention = config.build(32, layer_idx=0, n_layers=2)
    assert isinstance(attention, Attention)
    assert attention.sinks is not None
    assert attention.sinks.shape == (4,)


def test_attention_sinks_rejected_for_non_default_attention():
    config = AttentionConfig(name=AttentionType.normalized, n_heads=4, attention_sinks=True)
    with pytest.raises(OLMoConfigurationError, match="attention_sinks"):
        config.build(32, layer_idx=0, n_layers=2)


def test_attention_sinks_rejected_for_non_torch_backend_at_construction():
    # An explicitly requested non-torch backend must fail while building, not on the first forward.
    config = AttentionConfig(
        name=AttentionType.default,
        n_heads=4,
        backend=AttentionBackendName.flash_2,
        attention_sinks=True,
    )
    with pytest.raises(OLMoConfigurationError, match="torch attention backend"):
        config.build(32, layer_idx=0, n_layers=2)


def test_attention_sinks_softmax_matches_sdpa_when_inactive():
    from olmo_core.nn.transformer.init import InitMethod

    seed_all(0)
    d_model, seq_len = 32, 16

    with_sinks = Attention(
        d_model=d_model,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        backend=AttentionBackendName.torch,
        attention_sinks=True,
    )
    with_sinks.init_weights(
        init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2
    )

    without_sinks = Attention(
        d_model=d_model,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        backend=AttentionBackendName.torch,
    )
    without_sinks.load_state_dict(
        {k: v for k, v in with_sinks.state_dict().items() if k != "sinks"}
    )

    x = torch.randn(2, seq_len, d_model)

    assert with_sinks.sinks is not None
    # A very negative sink logit makes the extra softmax column vanish, so the manual sink softmax
    # must collapse to the plain SDPA result.
    with torch.no_grad():
        with_sinks.sinks.fill_(-1e4)
        sink_out = with_sinks(x)
        plain_out = without_sinks(x)
    torch.testing.assert_close(sink_out, plain_out, rtol=1e-4, atol=1e-4)


def test_fused_attention_v2_matches_standard_attention():
    from olmo_core.nn.mxfp8_linear import MXFP8Linear

    seed_all(0)
    d_model = 64
    fused = FusedAttentionV2(
        d_model=d_model,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        bias=False,
        backend=AttentionBackendName.torch,
    )
    # Non-MXFP8 projections should be plain Linear layers.
    assert isinstance(fused.w_qkv, torch.nn.Linear)
    assert not isinstance(fused.w_qkv, MXFP8Linear)

    standard = Attention(
        d_model=d_model,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        bias=False,
        backend=AttentionBackendName.torch,
    )
    # Copy the packed QKV projection into the standard attention's separate Q/K/V projections.
    q_dim, kv_dim = 4 * 16, 2 * 16
    with torch.no_grad():
        standard.w_q.weight.copy_(fused.w_qkv.weight[:q_dim])
        standard.w_k.weight.copy_(fused.w_qkv.weight[q_dim : q_dim + kv_dim])
        standard.w_v.weight.copy_(fused.w_qkv.weight[q_dim + kv_dim :])
        standard.w_out.weight.copy_(fused.w_out.weight)

    x = torch.randn(2, 8, d_model)
    with torch.no_grad():
        torch.testing.assert_close(fused(x), standard(x))


def test_attention_config_rejects_mxfp8_on_non_fused_v2():
    config = AttentionConfig(
        name=AttentionType.default, n_heads=4, head_dim=16, mxfp8_projections=True
    )
    with pytest.raises(OLMoConfigurationError, match="fused_v2"):
        config.build(64, layer_idx=0, n_layers=1)


def test_attention_use_recompute_qkv_prep_is_transparent():
    from olmo_core.nn.transformer.init import InitMethod

    def run(recompute: bool):
        seed_all(0)
        attention = FusedAttentionV2(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            bias=False,
            backend=AttentionBackendName.torch,
            use_recompute_qkv_prep=recompute,
        )
        attention.init_weights(init_method=InitMethod.normal, d_model=64, block_idx=0, num_blocks=2)
        x = torch.randn(2, 8, 64, requires_grad=True)
        seed_all(123)
        out = attention(x)
        out.sum().backward()
        assert x.grad is not None
        assert attention.w_qkv.weight.grad is not None
        return out.detach(), x.grad.detach(), attention.w_qkv.weight.grad.detach().clone()

    # Recomputing Q/K/V in backward must not change the forward output or the gradients.
    out0, grad_x0, grad_w0 = run(False)
    out1, grad_x1, grad_w1 = run(True)
    torch.testing.assert_close(out0, out1)
    torch.testing.assert_close(grad_x0, grad_x1)
    torch.testing.assert_close(grad_w0, grad_w1)


def test_attention_config_rejects_recompute_on_unsupported_attention():
    config = AttentionConfig(
        name=AttentionType.normalized, n_heads=4, head_dim=16, use_recompute_qkv_prep=True
    )
    with pytest.raises(OLMoConfigurationError, match="use_recompute_qkv_prep"):
        config.build(64, layer_idx=0, n_layers=1)


def test_attention_config_allows_disabled_fused_v2_flags_on_other_types():
    # A disabled (falsy) fused_v2-only flag is a no-op and must not break other attention types,
    # even though as_dict keeps the explicit False.
    config = AttentionConfig(
        name=AttentionType.default,
        n_heads=4,
        head_dim=16,
        mxfp8_projections=False,
        mxfp8_qkv_projection=False,
        use_recompute_qkv_prep=False,
    )
    attention = config.build(64, layer_idx=0, n_layers=1)
    assert isinstance(attention, Attention)


def test_mxfp8_saved_qkv_hooks_match_saved_tensors_by_storage():
    # The Torch backend transposes (and, for GQA, repeats) q/k/v before SDPA, producing new tensor
    # objects that autograd saves. The pack hook must recognize those via shared storage; matching
    # by tensor identity would miss all of them and silently no-op.
    from olmo_core.nn.attention import _MXFP8SavedQKVHooks
    from olmo_core.nn.attention.backend import _repeat_kv

    q = torch.randn(2, 8, 4, 32)
    k = torch.randn(2, 8, 2, 32)
    v = torch.randn(2, 8, 2, 32)
    hooks = _MXFP8SavedQKVHooks(q, k, v, pack_counter=[0])

    def matched(t: torch.Tensor):
        return hooks.target_names.get(t.untyped_storage().data_ptr())

    # transpose is a view -> shares storage -> matched.
    assert matched(q.transpose(1, 2)) == "attention.q"
    # a non-repeating kv path (n_rep=1) stays a view -> matched.
    assert matched(_repeat_kv(k, 1).transpose(1, 2)) == "attention.k"
    # a GQA repeat copies -> fresh storage -> not matched (acceptable; never mis-packs).
    assert matched(_repeat_kv(k, 3)) is None
    # unrelated tensors are never matched.
    assert matched(torch.randn(2, 8, 4, 32)) is None


def test_causal_attention_positions():
    # Full causal attention: the triangle sum, including the diagonal (self-attention).
    assert _causal_attention_positions(1) == 1
    assert _causal_attention_positions(4) == 10  # 4 + 3 + 2 + 1
    assert _causal_attention_positions(32) == 32 * 33 // 2

    # A window at least as large as the sequence is equivalent to full attention.
    assert _causal_attention_positions(32, 32) == _causal_attention_positions(32)
    assert _causal_attention_positions(32, 100) == _causal_attention_positions(32)

    # Sliding window: an early triangle plus a flat ``window_size`` per remaining query.
    assert _causal_attention_positions(32, 8) == 8 * 9 // 2 + (32 - 8) * 8


def test_attention_num_flops_per_token_applies_causal_discount():
    d_model, n_heads, seq_len = 64, 4, 32
    attn = AttentionConfig(name=AttentionType.default, n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1
    )
    head_dim = attn.head_dim
    param_flops = 6 * sum(p.numel() for p in attn.parameters())
    expected_attn = 12 * n_heads * head_dim * _causal_attention_positions(seq_len) // seq_len

    assert attn.num_flops_per_token(seq_len) == param_flops + expected_attn
    # The causal triangle roughly halves the attention-compute term relative to the pre-change
    # formula, which counted a full ``seq_len`` of keys per query.
    old_style_attn = 12 * n_heads * head_dim * seq_len
    assert expected_attn < old_style_attn
    assert expected_attn * 2 > old_style_attn  # ~half, not an order of magnitude off


def test_attention_num_flops_per_token_sliding_window_is_cheaper():
    d_model, n_heads, seq_len = 64, 4, 32
    sliding_window = SlidingWindowAttentionConfig(
        pattern=[8],
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=False,
    )
    swa_attn = AttentionConfig(
        name=AttentionType.default, n_heads=n_heads, sliding_window=sliding_window
    ).build(d_model, layer_idx=0, n_layers=4)
    full_attn = AttentionConfig(name=AttentionType.default, n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=4
    )

    assert swa_attn.window_size == 8
    # Same parameters, so any difference is purely the windowed attention-compute term.
    assert swa_attn.num_flops_per_token(seq_len) < full_attn.num_flops_per_token(seq_len)

    head_dim = swa_attn.head_dim
    param_flops = 6 * sum(p.numel() for p in swa_attn.parameters())
    expected_attn = 12 * n_heads * head_dim * _causal_attention_positions(seq_len, 8) // seq_len
    assert swa_attn.num_flops_per_token(seq_len) == param_flops + expected_attn


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("use_rope", [False, True])
def test_legacy_fused_attention_checkpoint(bias: bool, use_rope: bool):
    """Load the original packed parameter and optimizer layout through a serialized legacy config."""
    config = AttentionConfig.from_dict(
        {
            "name": "fused",
            "n_heads": 2,
            "bias": bias,
            "backend": "torch",
            "use_flash": True,  # Ignored by the original implementation, even with a backend set.
            "rope": {"name": "fused", "theta": 12345, "full_precision": False}
            if use_rope
            else None,
        }
    )
    original_config = config.as_config_dict()
    # The original module registered exactly these projections, in this order.
    legacy = torch.nn.ModuleDict(
        {
            "w_qkv": torch.nn.Linear(32, 96, bias=bias),
            "w_out": torch.nn.Linear(32, 32, bias=bias),
        }
    )
    legacy_optim = torch.optim.AdamW(legacy.parameters())
    sum(p.square().sum() for p in legacy.parameters()).backward()
    legacy_optim.step()

    with pytest.warns(UserWarning, match="not numerically identical"):
        attention = config.build(32, layer_idx=0, n_layers=1)
    assert isinstance(attention, FusedAttentionV2)
    assert config.as_config_dict() == original_config
    assert list(attention.state_dict()) == list(legacy.state_dict())
    attention.load_state_dict(legacy.state_dict(), strict=True)
    optim = torch.optim.AdamW(attention.parameters())
    optim.load_state_dict(legacy_optim.state_dict())
    for old_param, new_param in zip(legacy.parameters(), attention.parameters()):
        torch.testing.assert_close(new_param, old_param, rtol=0, atol=0)
        for key, value in legacy_optim.state[old_param].items():
            torch.testing.assert_close(optim.state[new_param][key], value, rtol=0, atol=0)
    if use_rope:
        from olmo_core.nn.rope import RotaryEmbedding

        assert isinstance(attention.rope, RotaryEmbedding)
        assert attention.rope.theta == 12345
        assert not attention.rope.full_precision
    else:
        assert attention.rope is None
    attention(torch.randn(2, 8, 32)).square().mean().backward()
    optim.step()
    assert all(torch.isfinite(p).all() for p in attention.parameters())


@pytest.mark.parametrize("name", list(AttentionType))
@pytest.mark.parametrize(
    "use_flash,backend",
    [
        (None, None),
        (False, None),
        (True, None),
        (False, "flash_3"),
        (True, "flash_2"),
        (True, "torch"),
        (True, "flash_3"),
    ],
)
def test_legacy_use_flash_config_backend(name, use_flash, backend, monkeypatch):
    """Old serialized flags preserve backend selection and reject conflicting explicit backends."""
    import warnings

    config = AttentionConfig.from_dict(
        {"name": name, "n_heads": 2, "use_flash": use_flash, "backend": backend}
    )
    selected = []
    original_build = AttentionBackendName.build

    def record_backend(backend_name, **kwargs):
        selected.append(backend_name)
        return original_build(AttentionBackendName.torch, **kwargs)

    # Observe backend selection through real module construction without requiring GPU kernels.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(AttentionBackendName, "assert_supported", lambda self: None)
    monkeypatch.setattr(AttentionBackendName, "build", record_backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        if name != AttentionType.fused and use_flash and backend in ("torch", "flash_3"):
            with pytest.raises(OLMoConfigurationError, match="only compatible with 'flash_2'"):
                config.build(64, layer_idx=0, n_layers=1)
            assert not selected
        else:
            config.build(64, layer_idx=0, n_layers=1)
            expected = backend or (
                "flash_2" if use_flash or name == AttentionType.fused else "torch"
            )
            assert selected == [expected]


@pytest.mark.parametrize("name", [AttentionType.default, AttentionType.fused_v2])
def test_legacy_use_flash_false_preserves_swa_backend(name, monkeypatch):
    """False historically leaves an unspecified SWA backend free to select FlashAttention."""
    from olmo_core.nn.attention import flash_attn_api

    config = AttentionConfig(
        name=name,
        n_heads=2,
        use_flash=False,
        sliding_window=SlidingWindowAttentionConfig(
            pattern=[4],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        ),
    )
    selected = []
    original_build = AttentionBackendName.build

    def record_backend(backend_name, **kwargs):
        selected.append(backend_name)
        return original_build(AttentionBackendName.torch, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flash_attn_api, "has_flash_attn_2", lambda: True)
    monkeypatch.setattr(AttentionBackendName, "assert_supported", lambda self: None)
    monkeypatch.setattr(AttentionBackendName, "build", record_backend)
    config.build(64, layer_idx=0, n_layers=1)
    assert selected == [AttentionBackendName.flash_2]
