import importlib.metadata
import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from .ring import RingAttentionLoadBalancerType

_flash_attn_pkgs = importlib.metadata.packages_distributions().get("flash_attn", [])

flash_attn_2 = None
try:
    _is_flash_attn_2_available = "flash-attn" in [pkg.replace("_", "-") for pkg in _flash_attn_pkgs]

    if _is_flash_attn_2_available:
        import flash_attn as flash_attn_2  # type: ignore
except ImportError:
    pass

try:
    # NOTE: The flash-attn 3 API might be available under the name 'flash_attn_3.flash_attn_interface'
    # or just 'flash_attn_interface', depending on how it was built.
    import flash_attn_3.flash_attn_interface as flash_attn_3  # type: ignore
except ImportError:
    try:
        import flash_attn_interface as flash_attn_3  # type: ignore
    except ImportError:
        flash_attn_3 = None  # type: ignore

# match transformers fa4 detection - cute submodule available in both fa2/fa4
flash_attn_4 = None
try:
    _is_flash_attn_4_available = "flash-attn-4" in [
        pkg.replace("_", "-") for pkg in _flash_attn_pkgs
    ]

    if _is_flash_attn_4_available:
        import flash_attn.cute as flash_attn_4  # type: ignore
except ImportError:
    pass

try:
    import ring_flash_attn  # type: ignore
except ImportError:
    ring_flash_attn = None  # type: ignore

log = logging.getLogger(__name__)


def has_flash_attn_2() -> bool:
    return flash_attn_2 is not None


def has_flash_attn_3() -> bool:
    if flash_attn_3 is not None:
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            is_supported = (9, 0) <= compute_capability < (10, 0)  # H100 / H800
            return is_supported
        return True
    return False


def has_flash_attn_4() -> bool:
    if flash_attn_4 is not None:
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            return compute_capability >= (10, 0) and compute_capability <= (
                12,
                0,
            )  # Blackwell, Blackwell Ultra
        return True
    return False


def has_ring_flash_attn() -> bool:
    return ring_flash_attn is not None


def _flatten_batch_dim(x: torch.Tensor) -> torch.Tensor:
    B, T, *other = x.shape
    return x.view(B * T, *other)


def dispatch_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn_2 is None:
        raise RuntimeError("flash-attn 2 is required!")

    if cu_seqlens is not None:
        cu_seqlens_q = cu_seqlens if cu_seqlens_q is None else cu_seqlens_q
        cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
    if max_seqlen is not None:
        max_seqlen_q = max_seqlen if max_seqlen_q is None else max_seqlen_q
        max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k

    varlen = all(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k))

    if varlen:
        return flash_attn_2.flash_attn_varlen_func(
            _flatten_batch_dim(q),
            _flatten_batch_dim(k),
            _flatten_batch_dim(v),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    else:
        return flash_attn_2.flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )


def dispatch_flash_attn_3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn_3 is None:
        raise RuntimeError("flash-attn 3 is required!")

    if cu_seqlens is not None:
        cu_seqlens_q = cu_seqlens if cu_seqlens_q is None else cu_seqlens_q
        cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
    if max_seqlen is not None:
        max_seqlen_q = max_seqlen if max_seqlen_q is None else max_seqlen_q
        max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k

    varlen = all(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k))

    if varlen:
        return flash_attn_3.flash_attn_varlen_func(
            _flatten_batch_dim(q),
            _flatten_batch_dim(k),
            _flatten_batch_dim(v),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    else:
        return flash_attn_3.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )


def dispatch_flash_attn_qkvpacked(
    qkv: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn_2 is None:
        raise RuntimeError("flash-attn 2 is required!")

    if cu_seqlens is not None and max_seqlen is not None:
        return flash_attn_2.flash_attn_varlen_qkvpacked_func(
            _flatten_batch_dim(qkv),
            cu_seqlens,
            max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    else:
        return flash_attn_2.flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )


def dispatch_flash_attn_3_qkvpacked(
    qkv: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn_3 is None:
        raise RuntimeError("flash-attn 3 is required!")

    if cu_seqlens is not None and max_seqlen is not None:
        return flash_attn_3.flash_attn_varlen_qkvpacked_func(
            _flatten_batch_dim(qkv),
            cu_seqlens,
            max_seqlen,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    else:
        return flash_attn_3.flash_attn_qkvpacked_func(
            qkv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )


@torch._dynamo.disable()
def dispatch_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn_2 is None:
        raise RuntimeError("flash-attn 2 is required!")

    return flash_attn_2.flash_attn_with_kvcache(
        q,
        k_cache,  # updated in-place if k/v are provided
        v_cache,  # updated in-place if k/v are provided
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        cache_leftpad=cache_leftpad,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )


@torch._dynamo.disable()
def dispatch_flash_attn_3_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn_3 is None:
        raise RuntimeError("flash-attn 3 is required!")

    return flash_attn_3.flash_attn_with_kvcache(
        q,
        k_cache,  # updated in-place if k/v are provided
        v_cache,  # updated in-place if k/v are provided
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        cache_leftpad=cache_leftpad,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )


@torch._dynamo.disable()
def dispatch_ring_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    strategy: RingAttentionLoadBalancerType,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    heads_k_stride: Optional[int] = None,
    local_k_slice: Optional[slice] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if ring_flash_attn is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")

    if strategy == RingAttentionLoadBalancerType.zig_zag:
        if any(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)):
            raise RuntimeError(
                f"{strategy} load balancing strategy requires unified QK doc lengths"
            )

        if local_k_slice is not None:
            raise RuntimeError(f"'local_k_slice' is invalid for {strategy} load balancing strategy")

        if window_size != (-1, -1):
            # See https://github.com/zhuzilin/ring-flash-attention/issues/35
            raise RuntimeError(
                f"SWA is currently not supported for {strategy} load balancing strategy"
            )

        if cu_seqlens is not None and max_seqlen is not None:
            out = ring_flash_attn.zigzag_ring_flash_attn_varlen_func(
                _flatten_batch_dim(q),
                _flatten_batch_dim(k),
                _flatten_batch_dim(v),
                cu_seqlens,
                max_seqlen,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
        else:
            out = ring_flash_attn.zigzag_ring_flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
    elif strategy == RingAttentionLoadBalancerType.llama3:
        if any(x is not None for x in (cu_seqlens, max_seqlen)):
            raise RuntimeError(
                f"{strategy} load balancing strategy requires separate QK doc lengths"
            )

        if (
            cu_seqlens_q is None
            or cu_seqlens_k is None
            or max_seqlen_q is None
            or max_seqlen_k is None
            or heads_k_stride is None
            or local_k_slice is None
        ):
            raise RuntimeError(
                f"{strategy} load balancing strategy is only implemented for 'varlen' variant.\n"
                "The following arguments are required: 'cu_seqlens_(q|k)', 'max_seqlen_(q|k)', "
                "'heads_k_stride', and 'local_k_slice'."
            )

        out = ring_flash_attn.llama3_flash_attn_varlen_func(
            _flatten_batch_dim(q),
            _flatten_batch_dim(k),
            _flatten_batch_dim(v),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            heads_k_stride,
            local_k_slice,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            group=group,
            window_size=window_size,
        )
    else:
        raise NotImplementedError(strategy)

    return out  # type: ignore


@torch._dynamo.disable()
def dispatch_ring_flash_attn_qkvpacked(
    qkv: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    strategy: RingAttentionLoadBalancerType,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if ring_flash_attn is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")

    if strategy == RingAttentionLoadBalancerType.zig_zag:
        if cu_seqlens is not None and max_seqlen is not None:
            out = ring_flash_attn.zigzag_ring_flash_attn_varlen_qkvpacked_func(
                _flatten_batch_dim(qkv),
                cu_seqlens,
                max_seqlen,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
        else:
            out = ring_flash_attn.zigzag_ring_flash_attn_qkvpacked_func(
                qkv,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
    else:
        raise NotImplementedError(strategy)

    return out  # type: ignore


@torch._dynamo.disable()
def dispatch_flash_attn_4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    seqused_k: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if flash_attn_4 is None:
        raise RuntimeError("flash-attn 4 (CUTE implementation) is required!")

    if window_size == (-1, -1) or (window_size == (-1, 0) and causal):
        window_size = (None, None)  # type: ignore

    # KV cache path: use varlen API with seqused_k to limit the valid cache length.
    if seqused_k is not None:
        B = q.shape[0]
        T = q.shape[1]
        cu_seqlens_q_cache = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
        return flash_attn_4.flash_attn_varlen_func(
            _flatten_batch_dim(q),
            k,
            v,
            cu_seqlens_q=cu_seqlens_q_cache,
            cu_seqlens_k=None,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )[0]

    if cu_seqlens is not None:
        cu_seqlens_q = cu_seqlens if cu_seqlens_q is None else cu_seqlens_q
        cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
    if max_seqlen is not None:
        max_seqlen_q = max_seqlen if max_seqlen_q is None else max_seqlen_q
        max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k

    varlen = all(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k))

    if varlen:
        return flash_attn_4.flash_attn_varlen_func(
            _flatten_batch_dim(q),
            _flatten_batch_dim(k),
            _flatten_batch_dim(v),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )[0]
    else:
        return flash_attn_4.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )[0]
