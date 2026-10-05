from abc import abstractmethod
from typing import Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh

from olmo_core.config import StrEnum
from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_cp2hp_qkvpacked,
    all_to_all_single_hp2cp,
)
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.buffer_cache import BufferCache

from .flash_attn_api import (
    dispatch_flash_attn,
    dispatch_flash_attn_3,
    dispatch_flash_attn_3_qkvpacked,
    dispatch_flash_attn_3_with_kvcache,
    dispatch_flash_attn_4,
    dispatch_flash_attn_qkvpacked,
    dispatch_flash_attn_with_kvcache,
    dispatch_ring_flash_attn,
    dispatch_ring_flash_attn_qkvpacked,
    has_flash_attn_2,
    has_flash_attn_3,
    has_flash_attn_4,
    has_ring_flash_attn,
)
from .ring import (
    RingAttentionLoadBalancerType,
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from .te_attn_api import TEDotProductAttention, has_te_attn


class AttentionBackendName(StrEnum):
    """
    An enumeration of the different attention backends.
    """

    torch = "torch"
    """
    PyTorch's built-in SDPA. Works on all devices. ➡️ :class:`TorchAttentionBackend`
    """
    flash_2 = "flash_2"
    """
    Flash attention 2 from the `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ library.
    Supports Ampere (SM 8.0+) and newer NVIDIA GPUs.
    To use this with context-parallelism, `ring-flash-attn <https://github.com/zhuzilin/ring-flash-attention>`_
    is also required. ➡️ :class:`FlashAttention2Backend`
    """
    flash_3 = "flash_3"
    """
    Flash attention 3 (beta) from the `flash-attn <https://github.com/Dao-AILab/flash-attention>`_
    library ``hopper/`` subdirectory. Supports Hopper (SM 9.0) GPUs only (H100/H800).
    ➡️ :class:`FlashAttention3Backend`
    """
    flash_4 = "flash_4"
    """
    Flash attention 4, the CUTE implementation from `flash-attn <https://github.com/Dao-AILab/flash-attention>`_
    in the ``flash_attn/cute`` subdirectory. Supports Blackwell (SM 10.0, e.g. B200) GPUs only.
    ➡️ :class:`FlashAttention4Backend`
    """
    te = "te"
    """
    Transformer Engine attention. Supports Hopper (SM 9.0+) and newer NVIDIA GPUs.
    ➡️ :class:`TEAttentionBackend`.
    """
    flex = "flex"
    """
    PyTorch's `FlexAttention <https://pytorch.org/blog/flexattention/>`_ (a fused, compiled
    kernel). Like :class:`TorchAttentionBackend` it supports the custom ``or_mask`` /
    ``and_mask`` (via a ``mask_mod`` + ``BlockMask``) — but as a fused kernel rather than
    dense SDPA, so it is much faster for the bidirectional-image / subsegment masks used by
    multimodal training. ➡️ :class:`FlexAttentionBackend`.
    """

    def get_class(self) -> Type["AttentionBackend"]:
        if self == self.torch:
            return TorchAttentionBackend
        elif self in self.flash_2:
            return FlashAttention2Backend
        elif self == self.flash_3:
            return FlashAttention3Backend
        elif self == self.flash_4:
            return FlashAttention4Backend
        elif self == self.te:
            return TEAttentionBackend
        elif self == self.flex:
            return FlexAttentionBackend
        else:
            raise NotImplementedError(self)

    def build(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ) -> "AttentionBackend":
        return self.get_class()(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=(cache.with_namespace(f"attn_backend={self}") if cache else None),
        )

    def assert_supported(self):
        self.get_class().assert_supported()

    def assert_supports_swa(self):
        self.get_class().assert_supports_swa()

    def assert_supports_ring_cp(self):
        self.get_class().assert_supports_ring_cp()

    def assert_supports_ulysses_cp(self):
        self.get_class().assert_supports_ulysses_cp()

    def assert_supports_packed_qkv(self):
        self.get_class().assert_supports_packed_qkv()

    def assert_supports_kv_cache(self):
        self.get_class().assert_supports_kv_cache()


class AttentionBackend(nn.Module):
    """
    Encapsulates a backend for the scaled dot-product attention (SDPA) operation.
    """

    SUPPORTS_OR_MASK: bool = False
    """Whether :meth:`forward` honors the ``or_mask`` argument (a boolean allow-mask
    OR'd onto the causal/sliding base, e.g. for bidirectional image-token attention).
    Only the dense SDPA backend supports it; flash/TE backends do not."""

    SUPPORTS_AND_MASK: bool = False
    """Whether :meth:`forward` honors the ``and_mask`` argument (a boolean keep-mask
    AND'd onto the (causal | or_mask) base, e.g. for subsegment / branch isolation in
    packed multi-annotation multimodal data). Only the dense SDPA backend supports it;
    flash/TE backends do not."""

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        self.assert_supported()
        if window_size != (-1, -1):
            self.assert_supports_swa()
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.scale = scale
        self.dropout_p = dropout_p
        self.window_size = window_size
        self.cache = cache
        self.cp_pg: Optional[dist.ProcessGroup] = None
        self.cp_enabled = False
        self.head_stride: int = 1

    @classmethod
    @abstractmethod
    def assert_supported(cls):
        """
        Validates that this backend is supported on the current system.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_swa(cls):
        """
        Validates that this backend supports sliding window attention (SWA).
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_ring_cp(cls):
        """
        Validates that this backend supports ring context parallelism.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_ulysses_cp(cls):
        """
        Validates that this backend supports ulysses context parallelism.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_packed_qkv(cls):
        """
        Validates that this backend supports taking QKV as a single packed tensor.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_kv_cache(cls):
        """
        Validates that this backend supports KV caching.
        Raises an error if not supported.
        """
        pass

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        """
        Warmup the buffer cache.
        """
        del max_seq_len, device

    @abstractmethod
    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the attention operation.

        :param sinks: Optional per-head attention-sink logits (learnable "no-op" attention
            targets). Only the :class:`TorchAttentionBackend` consumes these.
        """
        raise NotImplementedError

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """
        Apply context parallelism if supported by the backend.
        """
        if ring is not None:
            self.assert_supports_ring_cp()
        elif uly is not None:
            self.assert_supports_ulysses_cp()
        else:
            raise ValueError("One of ring or uly must be specified")

        self.cp_pg = cp_mesh.get_group()
        self.ring = ring
        self.uly = uly
        self.cp_enabled = True


class TorchAttentionBackend(AttentionBackend):
    """
    PyTorch's built-in scaled dot-product attention (SDPA) backend.
    """

    SUPPORTS_OR_MASK = True
    SUPPORTS_AND_MASK = True

    @classmethod
    def assert_supported(cls):
        pass

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support ring context parallelism")

    @classmethod
    def assert_supports_ulysses_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support KV caching")

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_sliding_window_mask(
            seq_len_q=max_seq_len,
            seq_len_kv=max_seq_len,
            device=device,
            window_size=self.window_size,
        )

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del local_k_slice

        if isinstance(qkv, torch.Tensor):
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support packed QKV")

        q, k, v = qkv

        if kv_cache_manager is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support KV caching")

        attn_mask: Optional[torch.Tensor] = None
        if self.window_size != (-1, -1):
            attn_mask = self._get_sliding_window_mask(
                seq_len_q=q.shape[1],
                seq_len_kv=k.shape[1],
                device=q.device,
                window_size=self.window_size,
            )
        elif sinks is not None:
            # The sink path applies softmax manually (see below), so it needs an explicit causal
            # mask rather than relying on SDPA's ``is_causal``.
            attn_mask = self._get_sliding_window_mask(
                seq_len_q=q.shape[1],
                seq_len_kv=k.shape[1],
                device=q.device,
                window_size=(-1, -1),
            )

        if or_mask is not None:
            # OR an extra boolean allow-mask (e.g. bidirectional attention among
            # image tokens) onto the causal/sliding-window base, mirroring HF's
            # `or_mask_function`. ``True`` entries are additionally allowed to
            # attend regardless of causal order. Build an explicit causal base
            # when we don't already have a sliding-window one.
            base = attn_mask
            if base is None:
                base = torch.ones(q.shape[1], k.shape[1], device=q.device, dtype=torch.bool).tril()
            attn_mask = base | or_mask.to(device=q.device, dtype=torch.bool)

        if and_mask is not None:
            # AND a boolean keep-mask onto the (causal | or_mask) base, mirroring
            # mm_olmo's `attention_mask & subsegment_mask`: a query may only attend
            # to keys allowed by *both* masks. Used for subsegment / branch isolation
            # in packed multi-annotation multimodal data. Build an explicit causal
            # base first when we don't already have one.
            base = attn_mask
            if base is None:
                base = torch.ones(q.shape[1], k.shape[1], device=q.device, dtype=torch.bool).tril()
            attn_mask = base & and_mask.to(device=q.device, dtype=torch.bool)

        if any(
            opt is not None
            for opt in (
                cu_doc_lens,
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
            )
        ):
            raise RuntimeError(
                f"'{self.__class__.__name__}' doesn't support intra-document masking"
            )

        if self.cp_enabled and self.uly is not None:
            assert self.cp_pg is not None
            # Transform from context-parallel to head-parallel partitioning
            # [B, T/CP, H, D] -> [B, T, H/CP, D]
            q = all_to_all_single_cp2hp(q, self.cp_pg)
            k, v = all_to_all_cp2hp([k, v], self.cp_pg)

        # NOTE: PyTorch's SDPA doesn't support GQA, so we have to do this.
        n_rep = self.n_heads // self.n_kv_heads
        # shape: (batch_size, seq_len, n_heads, head_dim)
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)

        # PyTorch's SDPA expects the head dimension to come before the sequence dimension.
        # shape: (batch_size, n_heads, seq_len, head_dim),
        #        (batch_size, n_kv_heads, seq_len, head_dim),
        #        (batch_size, n_kv_heads, seq_len, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # shape: (batch_size, n_heads, seq_len, head_dim)
        if sinks is None:
            att = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                is_causal=attn_mask is None,
                scale=self.scale,
            )
        else:
            # Attention sinks add a per-head learnable logit as an extra softmax column. Since SDPA
            # can't express that, run the softmax explicitly: concatenate the sink logit, normalize
            # over the widened last dim, then drop the sink column before the value matmul.
            scale = self.scale if self.scale is not None else self.head_dim**-0.5
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
            assert attn_mask is not None
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(
                    ~attn_mask.view(1, 1, attn_mask.shape[0], attn_mask.shape[1]),
                    torch.finfo(attn_weights.dtype).min,
                )
            else:
                attn_weights = attn_weights + attn_mask

            sink_logits = sinks.to(dtype=attn_weights.dtype, device=attn_weights.device)
            sink_logits = sink_logits.view(1, -1, 1, 1).expand(
                attn_weights.shape[0],
                -1,
                attn_weights.shape[-2],
                -1,
            )
            combined_logits = torch.cat((attn_weights, sink_logits), dim=-1)
            combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
            probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)[..., :-1]
            probs = F.dropout(probs, p=self.dropout_p, training=self.training).to(v.dtype)
            att = torch.matmul(probs, v)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        att = att.transpose(1, 2)

        if self.cp_enabled and self.uly is not None:
            assert self.cp_pg is not None
            # Transform back from head-parallel to context-parallel partitioning
            # [B, T, H/CP, D] -> [B, T/CP, H, D]
            att = all_to_all_single_hp2cp(att, self.cp_pg)

        return att.contiguous()

    def _get_sliding_window_mask(
        self,
        seq_len_q: int,
        seq_len_kv: int,
        device: torch.device,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        key = f"seq_len_q={seq_len_q},seq_len_kv={seq_len_kv},window_size={window_size}"
        if self.cache is not None:
            if (mask := self.cache.get_for_device(key, device)) is not None:
                return mask

            attn_mask = self._build_sliding_window_mask(
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                device=device,
                window_size=window_size,
            )
            self.cache[key] = attn_mask

            return attn_mask

        return self._build_sliding_window_mask(
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            device=device,
            window_size=window_size,
        )

    @classmethod
    def _build_sliding_window_mask(
        cls,
        seq_len_q: int,
        seq_len_kv: int,
        device: torch.device,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_kv, device=device, dtype=torch.bool))

        if window_size != (-1, -1):
            sliding_window_left_mask = torch.ones_like(
                causal_mask, dtype=torch.bool, device=device
            ).triu(diagonal=-window_size[0])
            sliding_window_right_mask = torch.ones_like(
                causal_mask, dtype=torch.bool, device=device
            ).tril(diagonal=window_size[1])
            sliding_window_mask = torch.logical_and(
                sliding_window_left_mask,
                sliding_window_right_mask,
            )

            attn_mask = torch.logical_and(
                causal_mask,
                sliding_window_mask,
            )
        else:
            attn_mask = causal_mask

        return attn_mask


_compiled_flex_attention = None


def _get_flex_attention(device: torch.device):
    """Return ``flex_attention``, compiled on CUDA (where the fused kernel lives) and
    eager on CPU (compiling FlexAttention on CPU is slow / unnecessary — used only for
    correctness tests)."""
    from torch.nn.attention.flex_attention import flex_attention

    if device.type != "cuda":
        return flex_attention
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention


class FlexAttentionBackend(AttentionBackend):
    """`FlexAttention <https://pytorch.org/blog/flexattention/>`_ backend.

    Reproduces the dense :class:`TorchAttentionBackend` masking semantics —
    ``(causal | or_mask) & and_mask`` plus optional sliding window — but expresses them as
    a FlexAttention ``mask_mod`` over a :class:`~torch.nn.attention.flex_attention.BlockMask`
    so attention runs as a single fused, block-sparse kernel instead of materializing the
    full ``(B, H, S, S)`` scores. This is the only *fast* backend that supports the custom
    ``or_mask`` / ``and_mask`` used by multimodal training (flash / TE cannot).
    """

    SUPPORTS_OR_MASK = True
    SUPPORTS_AND_MASK = True

    @classmethod
    def assert_supported(cls):
        try:
            import torch.nn.attention.flex_attention  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "FlexAttention is not available in this PyTorch build (needs torch>=2.5)"
            ) from e

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support ring context parallelism")

    @classmethod
    def assert_supports_ulysses_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support ulysses context parallelism")

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support KV caching")

    @staticmethod
    def _per_token_from_masks(
        or_mask: Optional[torch.Tensor], and_mask: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Recover the per-token vectors behind the multimodal masks.

        FlexAttention's ``mask_mod`` must index per-token vectors (``v[b, q_idx]``), not the
        ``(q, kv)`` matrix (which is not ``vmap``-able). Our masks are structured, so the
        vectors are recoverable exactly:

        * ``or_mask[b, q, kv] = is_image[q] & is_image[kv]`` (an outer product), so
          ``is_image`` is its diagonal.
        * ``and_mask[b, q, kv] = same_example & (seg[q] <= seg[kv])``. ``same_example``
          (block-diagonal, identity within an example, from sequence packing) is
          ``and_mask | and_mask.T`` (within an example one of the two preorder directions
          always holds; across examples both are false). Packed examples are contiguous, so
          the per-token ``example_id`` is the running count of example boundaries
          (``¬same_example`` between adjacent tokens). The within-example subsegment rank
          ``seg_code[q] = #{kv : seg[kv] < seg[q]}`` (``= and_mask[kv,q] & ¬and_mask[q,kv]``,
          which is automatically example-local) preserves ``seg[q] <= seg[kv]`` exactly.

        For a single unpacked example ``same_example`` is all-true, so ``example_id`` is all
        zero (a no-op) and ``seg_code`` reduces to the plain subsegment rank.
        """
        is_image = None
        if or_mask is not None:
            is_image = or_mask[:, 0].diagonal(dim1=-2, dim2=-1).contiguous()  # (B, S) bool
        seg_code = None
        example_id = None
        if and_mask is not None:
            a = and_mask[:, 0]  # (B, S, S): a[b, q, kv] = same_example & (seg[q] <= seg[kv])
            strictly_less = a.transpose(-1, -2) & (~a)  # [b, q, kv] = seg[kv] < seg[q]
            seg_code = strictly_less.sum(dim=-1)  # (B, S) long, example-local rank
            same = a | a.transpose(-1, -2)  # (B, S, S) equivalence (block-diagonal)
            B, S = a.shape[0], a.shape[1]
            example_id = torch.zeros(B, S, dtype=torch.int64, device=a.device)
            if S > 1:
                adj_same = same.diagonal(offset=-1, dim1=-2, dim2=-1)  # (B, S-1): same[p, p-1]
                example_id[:, 1:] = torch.cumsum((~adj_same).to(torch.int64), dim=-1)
        return is_image, seg_code, example_id

    def _build_mask_mod(
        self,
        is_image: Optional[torch.Tensor],
        subsegment_ids: Optional[torch.Tensor],
        seg_code: Optional[torch.Tensor],
        example_id: Optional[torch.Tensor],
    ):
        ws_left, ws_right = self.window_size
        has_window = self.window_size != (-1, -1)

        def mask_mod(b, h, q_idx, kv_idx):
            del h
            # Causal base (+ optional sliding window), matching the dense backend.
            allow = kv_idx <= q_idx
            if has_window:
                if ws_left >= 0:
                    allow = allow & (q_idx - kv_idx <= ws_left)
                if ws_right >= 0:
                    allow = allow & (kv_idx - q_idx <= ws_right)
            # OR the bidirectional image allow-mask onto the base, then AND the subsegment
            # keep-mask and the packed-example block-diagonal — exactly
            # `(causal | or_mask) & and_mask` from TorchAttentionBackend.
            if is_image is not None:
                allow = allow | (is_image[b, q_idx] & is_image[b, kv_idx])
            if subsegment_ids is not None:
                allow = allow & (subsegment_ids[b, q_idx] <= subsegment_ids[b, kv_idx])
            elif seg_code is not None:
                allow = allow & (seg_code[b, q_idx] <= seg_code[b, kv_idx])
            if example_id is not None:
                allow = allow & (example_id[b, q_idx] == example_id[b, kv_idx])
            return allow

        return mask_mod

    @staticmethod
    def build_block_mask_from_vectors(
        B: int,
        S: int,
        device: torch.device,
        *,
        is_image: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        example_id: Optional[torch.Tensor] = None,
        window_size: Tuple[int, int] = (-1, -1),
    ):
        """Build a :class:`~torch.nn.attention.flex_attention.BlockMask` from O(S) vectors.

        Call once per forward (before the vision encoder) and reuse across all attention
        layers — matching mm_olmo's ``build_block_mask`` timing and avoiding a peak when
        ViT activations are already resident.
        """
        from torch.nn.attention.flex_attention import create_block_mask

        helper = FlexAttentionBackend(
            head_dim=1, n_heads=1, n_kv_heads=1, scale=1.0, window_size=window_size
        )
        mask_mod = helper._build_mask_mod(is_image, subsegment_ids, None, example_id)
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device)

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        flex_attn_is_image: Optional[torch.Tensor] = None,
        flex_attn_subsegment_ids: Optional[torch.Tensor] = None,
        flex_attn_example_ids: Optional[torch.Tensor] = None,
        flex_attn_block_mask: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from torch.nn.attention.flex_attention import create_block_mask

        del local_k_slice
        if sinks is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support attention sinks")
        if isinstance(qkv, torch.Tensor):
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support packed QKV")
        if kv_cache_manager is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support KV caching")
        if self.dropout_p:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support attention dropout")
        if self.cp_enabled:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support context parallelism")
        if any(
            opt is not None
            for opt in (
                cu_doc_lens,
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
            )
        ):
            raise RuntimeError(
                f"'{self.__class__.__name__}' doesn't support intra-document masking"
            )

        q, k, v = qkv
        # PyTorch SDPA-style GQA expansion + (B, S, H, D) -> (B, H, S, D).
        n_rep = self.n_heads // self.n_kv_heads
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        B, _, S_q, _ = q.shape
        S_kv = k.shape[2]
        if flex_attn_block_mask is not None:
            block_mask = flex_attn_block_mask
        else:
            if (
                flex_attn_is_image is not None
                or flex_attn_subsegment_ids is not None
                or flex_attn_example_ids is not None
            ):
                # O(S) per-token vectors from MultimodalLM — avoids materializing (B, S, S) masks
                # at 16k sequence length.
                is_image = flex_attn_is_image
                subsegment_ids = flex_attn_subsegment_ids
                seg_code = None
                example_id = flex_attn_example_ids
            else:
                om = or_mask.to(device=q.device, dtype=torch.bool) if or_mask is not None else None
                am = (
                    and_mask.to(device=q.device, dtype=torch.bool) if and_mask is not None else None
                )
                is_image, seg_code, example_id = self._per_token_from_masks(om, am)
                subsegment_ids = None
            mask_mod = self._build_mask_mod(is_image, subsegment_ids, seg_code, example_id)
            # `B`/`H=None` so the mask may depend on the batch index (image / subsegment ids are
            # per-example) but broadcasts over heads. Rebuilt each step since the masks are data
            # dependent; block-sparsity makes this cheap.
            block_mask = create_block_mask(
                mask_mod, B=B, H=None, Q_LEN=S_q, KV_LEN=S_kv, device=q.device
            )
        flex = _get_flex_attention(q.device)
        att = flex(q, k, v, block_mask=block_mask, scale=self.scale)

        # (B, H, S, D) -> (B, S, H, D)
        return att.transpose(1, 2).contiguous()


class FlashAttention2Backend(AttentionBackend):
    """
    SDPA from the flash-attn package. Additionally, ring-flash-attn is required for context parallelism.
    """

    @classmethod
    def assert_supported(cls):
        if not has_flash_attn_2():
            raise RuntimeError(
                f"'{cls.__name__}' is missing the flash-attn package or is not supported on this platform."
            )

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        if not has_ring_flash_attn():
            raise RuntimeError(
                f"'{cls.__name__}' requires the ring-flash-attn package for context parallelism."
            )

    @classmethod
    def assert_supports_ulysses_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        pass

    @classmethod
    def assert_supports_kv_cache(cls):
        pass

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(qkv, torch.Tensor):
            if kv_cache_manager is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with KV caching"
                )

            if self.window_size != (-1, -1):
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with sliding window attention"
                )

            if self.cp_enabled:
                assert self.cp_pg is not None
                if self.ring is not None:
                    return dispatch_ring_flash_attn_qkvpacked(
                        qkv,
                        group=self.cp_pg,
                        strategy=self.ring.load_balancer,
                        cu_seqlens=cu_doc_lens,
                        max_seqlen=max_doc_len,
                        dropout_p=self.dropout_p,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.window_size,
                    )
                elif self.uly is not None:
                    # Transform packed qkv from context-parallel to head-parallel partitioning
                    # [B, T/CP, 3, H, D] -> [B, T, 3, H/CP, D]
                    qkv = all_to_all_single_cp2hp_qkvpacked(qkv, self.cp_pg)
                    B, T, _, H_local, D = qkv.shape

                    # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                    # (same on all CP ranks), so we use them directly after gathering the full sequence.

                    # Run attention with full sequence, partitioned heads
                    out = dispatch_flash_attn_qkvpacked(
                        qkv,
                        cu_seqlens=cu_doc_lens,
                        max_seqlen=max_doc_len,
                        dropout_p=self.dropout_p,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.window_size,
                    )

                    # Transform back from head-parallel to context-parallel partitioning
                    # [B, T, H/CP, D] -> [B, T/CP, H, D]
                    return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
                else:
                    raise RuntimeError("One of ring or uly must be specified")
            else:
                return dispatch_flash_attn_qkvpacked(
                    qkv,
                    cu_seqlens=cu_doc_lens,
                    max_seqlen=max_doc_len,
                    dropout_p=self.dropout_p,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.window_size,
                )

        q, k, v = qkv

        if kv_cache_manager:
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support KV caching with context parallelism"
                )

            return dispatch_flash_attn_with_kvcache(
                q,
                k=k,
                v=v,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
                k_cache=kv_cache_manager.k_cache,  # updated in-place
                v_cache=kv_cache_manager.v_cache,  # updated in-place
                cache_leftpad=kv_cache_manager.cache_leftpad,
                cache_seqlens=kv_cache_manager.cache_seqlens.expand(
                    kv_cache_manager.cache_leftpad.shape[0]
                ).contiguous(),
            )

        if self.cp_enabled:
            assert self.cp_pg is not None
            if self.ring is not None:
                return dispatch_ring_flash_attn(
                    q,
                    k,
                    v,
                    group=self.cp_pg,
                    strategy=self.ring.load_balancer,
                    cu_seqlens=cu_doc_lens,
                    cu_seqlens_q=cu_doc_lens_q,
                    cu_seqlens_k=cu_doc_lens_k,
                    max_seqlen=max_doc_len,
                    max_seqlen_q=max_doc_len_q,
                    max_seqlen_k=max_doc_len_k,
                    heads_k_stride=self.ring.head_stride,
                    local_k_slice=local_k_slice,
                    dropout_p=self.dropout_p,
                    causal=True,
                    softmax_scale=self.scale,
                    window_size=self.window_size,
                )
            elif self.uly is not None:
                # Transform from context-parallel to head-parallel partitioning
                # [B, T/CP, H, D] -> [B, T, H/CP, D]
                q = all_to_all_single_cp2hp(q, self.cp_pg)
                k, v = all_to_all_cp2hp([k, v], self.cp_pg)
                B, T, H_local, D = q.shape

                # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                # (same on all CP ranks), so we use them directly after gathering the full sequence.
                # This is the default state of cu_doc_lens and max_doc_len before a load balancer is applied.

                # Run attention with full sequence, partitioned heads
                out = dispatch_flash_attn(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_doc_lens,
                    cu_seqlens_q=cu_doc_lens_q,
                    cu_seqlens_k=cu_doc_lens_k,
                    max_seqlen=max_doc_len,
                    max_seqlen_q=max_doc_len_q,
                    max_seqlen_k=max_doc_len_k,
                    dropout_p=self.dropout_p,
                    causal=True,
                    softmax_scale=self.scale,
                    window_size=self.window_size,
                )

                # Transform back from head-parallel to context-parallel partitioning
                # [B, T, H/CP, D] -> [B, T/CP, H, D]
                return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
            else:
                raise RuntimeError("One of ring or uly must be specified")

        return dispatch_flash_attn(
            q,
            k,
            v,
            cu_seqlens=cu_doc_lens,
            cu_seqlens_q=cu_doc_lens_q,
            cu_seqlens_k=cu_doc_lens_k,
            max_seqlen=max_doc_len,
            max_seqlen_q=max_doc_len_q,
            max_seqlen_k=max_doc_len_k,
            dropout_p=self.dropout_p,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.window_size,
        )


class FlashAttention3Backend(AttentionBackend):
    """
    SDPA from the flash-attn 3 package. Does not currently support context parallelism.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        if dropout_p > 0.0:
            raise RuntimeError("dropout_p > 0.0 is not supported for flash-attn 3")
        super().__init__(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=cache,
        )

    @classmethod
    def assert_supported(cls):
        if not has_flash_attn_3():
            raise RuntimeError(
                f"'{cls.__name__}' is missing the flash-attn 3 package or is not supported on this platform."
            )

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support ring context parallelism")

    @classmethod
    def assert_supports_ulysses_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        pass

    @classmethod
    def assert_supports_kv_cache(cls):
        pass

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(qkv, torch.Tensor):
            if kv_cache_manager is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with KV caching"
                )

            if self.window_size != (-1, -1):
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with sliding window attention"
                )

            if self.cp_enabled:
                assert self.cp_pg is not None
                if self.ring is not None:
                    raise RuntimeError(
                        f"'{self.__class__.__name__}' doesn't support ring context parallelism"
                    )
                elif self.uly is not None:
                    # Transform packed qkv from context-parallel to head-parallel partitioning
                    # [B, T/CP, 3, H, D] -> [B, T, 3, H/CP, D]
                    qkv = all_to_all_single_cp2hp_qkvpacked(qkv, self.cp_pg)
                    B, T, _, H_local, D = qkv.shape

                    # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                    # (same on all CP ranks), so we use them directly after gathering the full sequence.

                    # Run attention with full sequence, partitioned heads
                    out = dispatch_flash_attn_3_qkvpacked(
                        qkv,
                        cu_seqlens=cu_doc_lens,
                        max_seqlen=max_doc_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.window_size,
                    )

                    # Transform back from head-parallel to context-parallel partitioning
                    # [B, T, H/CP, D] -> [B, T/CP, H, D]
                    return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
                else:
                    raise RuntimeError("One of ring or uly must be specified")

            return dispatch_flash_attn_3_qkvpacked(
                qkv,
                cu_seqlens=cu_doc_lens,
                max_seqlen=max_doc_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
            )

        q, k, v = qkv

        if kv_cache_manager:
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support KV caching with context parallelism"
                )
            return dispatch_flash_attn_3_with_kvcache(
                q,
                k=k,
                v=v,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
                k_cache=kv_cache_manager.k_cache,  # updated in-place
                v_cache=kv_cache_manager.v_cache,  # updated in-place
                cache_leftpad=kv_cache_manager.cache_leftpad,
                cache_seqlens=kv_cache_manager.cache_seqlens.expand(
                    kv_cache_manager.cache_leftpad.shape[0]
                ).contiguous(),
            )

        if self.cp_enabled:
            if self.ring is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support ring context parallelism"
                )
            elif self.uly is not None:
                assert self.cp_pg is not None

                # Transform from context-parallel to head-parallel partitioning
                # [B, T/CP, H, D] -> [B, T, H/CP, D]
                q = all_to_all_single_cp2hp(q, self.cp_pg)
                k, v = all_to_all_cp2hp([k, v], self.cp_pg)
                B, T, H_local, D = q.shape

                # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                # (same on all CP ranks), so we use them directly after gathering the full sequence.
                # This is the default state of cu_doc_lens and max_doc_len before a load balancer is applied.

                # Run attention with full sequence, partitioned heads
                out = dispatch_flash_attn_3(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_doc_lens,
                    cu_seqlens_q=cu_doc_lens_q,
                    cu_seqlens_k=cu_doc_lens_k,
                    max_seqlen=max_doc_len,
                    max_seqlen_q=max_doc_len_q,
                    max_seqlen_k=max_doc_len_k,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.window_size,
                )

                # Transform back from head-parallel to context-parallel partitioning
                # [B, T, H/CP, D] -> [B, T/CP, H, D]
                return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
            else:
                raise RuntimeError("One of ring or uly must be specified")

        return dispatch_flash_attn_3(
            q,
            k,
            v,
            cu_seqlens=cu_doc_lens,
            cu_seqlens_q=cu_doc_lens_q,
            cu_seqlens_k=cu_doc_lens_k,
            max_seqlen=max_doc_len,
            max_seqlen_q=max_doc_len_q,
            max_seqlen_k=max_doc_len_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.window_size,
        )


class FlashAttention4Backend(AttentionBackend):
    """
    SDPA from flash-attn 4 (CUTE implementation).
    """

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        if dropout_p > 0.0:
            raise RuntimeError("dropout_p > 0.0 is not supported for flash-attn 4")
        super().__init__(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=cache,
        )

    @classmethod
    def assert_supported(cls):
        if not has_flash_attn_4():
            raise RuntimeError(
                f"'{cls.__name__}' is missing the flash-attn CUTE implementation or is not supported on this platform."
            )

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_ring_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support ring context parallelism")

    @classmethod
    def assert_supports_ulysses_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        pass

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert isinstance(qkv, tuple), f"'{self.__class__.__name__}' requires unpacked QKV"
        assert local_k_slice is None, f"'{self.__class__.__name__}' doesn't support local_k_slice"

        q, k, v = qkv

        if kv_cache_manager is not None:
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support KV caching with context parallelism"
                )
            pos = int(kv_cache_manager.cache_seqlens.item())
            T_new = k.shape[1]
            kv_cache_manager.k_cache[:, pos : pos + T_new] = k
            kv_cache_manager.v_cache[:, pos : pos + T_new] = v
            seqused_k = torch.full((q.shape[0],), pos + T_new, dtype=torch.int32, device=q.device)
            return dispatch_flash_attn_4(
                q,
                kv_cache_manager.k_cache,
                kv_cache_manager.v_cache,
                seqused_k=seqused_k,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
            )

        if self.cp_enabled:
            if self.ring is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support ring context parallelism"
                )
            elif self.uly is not None:
                assert self.cp_pg is not None

                # Transform from context-parallel to head-parallel partitioning
                # [B, T/CP, H, D] -> [B, T, H/CP, D]
                q = all_to_all_single_cp2hp(q, self.cp_pg)
                k, v = all_to_all_cp2hp([k, v], self.cp_pg)
                B, T, H_local, D = q.shape

                # NOTE: cu_doc_lens and max_doc_len are assumed to describe the FULL sequence
                # (same on all CP ranks), so we use them directly after gathering the full sequence.
                # This is the default state of cu_doc_lens and max_doc_len before a load balancer is applied.

                # Run attention with full sequence, partitioned heads
                out = dispatch_flash_attn_4(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_doc_lens,
                    cu_seqlens_q=cu_doc_lens_q,
                    cu_seqlens_k=cu_doc_lens_k,
                    max_seqlen=max_doc_len,
                    max_seqlen_q=max_doc_len_q,
                    max_seqlen_k=max_doc_len_k,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.window_size,
                )

                # Transform back from head-parallel to context-parallel partitioning
                # [B, T, H/CP, D] -> [B, T/CP, H, D]
                return all_to_all_single_hp2cp(out.view(B, T, H_local, D), self.cp_pg)
            else:
                raise RuntimeError("One of ring or uly must be specified")

        return dispatch_flash_attn_4(
            q,
            k,
            v,
            cu_seqlens=cu_doc_lens,
            cu_seqlens_q=cu_doc_lens_q,
            cu_seqlens_k=cu_doc_lens_k,
            max_seqlen=max_doc_len,
            max_seqlen_q=max_doc_len_q,
            max_seqlen_k=max_doc_len_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.window_size,
        )


class TEAttentionBackend(AttentionBackend):
    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=cache,
        )
        if not has_te_attn():
            raise RuntimeError("TransformerEngine attention is not available")
        assert TEDotProductAttention is not None
        self.te_attn = TEDotProductAttention(
            self.n_heads,
            self.head_dim,
            num_gqa_groups=self.n_kv_heads,
            attention_dropout=self.dropout_p,
            attn_mask_type="causal",
            window_size=(self.window_size[0], 0),  # be explicit about causal mask
            qkv_format="bshd",
            softmax_scale=self.scale,
        )

    @classmethod
    def assert_supported(cls):
        if not has_te_attn():
            raise RuntimeError(
                f"'{cls.__name__}' is missing the TransformerEngine package or is not supported on this platform."
            )

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support KV caching")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        super().apply_cp(cp_mesh, ring=ring, uly=uly)
        if self.ring is not None:
            if self.ring.load_balancer == RingAttentionLoadBalancerType.zig_zag:
                cp_comm_type = "p2p"  # Note: zig-zag/p2p is preferred bc it overlaps with the attention computation
            elif self.ring.load_balancer == RingAttentionLoadBalancerType.llama3:
                cp_comm_type = "all_gather"
            else:
                raise ValueError(self.ring.load_balancer)

            self.te_attn.set_context_parallel_group(
                cp_group=cp_mesh.get_group(),
                cp_global_ranks=dist.get_process_group_ranks(cp_mesh.get_group()),
                cp_stream=torch.cuda.default_stream(),
                #  cp_stream=get_or_init_stream("cp"),  # this doesn't seem to help
                cp_comm_type=cp_comm_type,
            )
        elif self.uly is not None:
            self.te_attn.set_context_parallel_group(
                cp_group=cp_mesh.get_group(),
                cp_global_ranks=dist.get_process_group_ranks(cp_mesh.get_group()),
                cp_stream=torch.cuda.default_stream(),
                #  cp_stream=get_or_init_stream("cp"),  # this doesn't seem to help
                cp_comm_type="a2a",
            )
        else:
            raise ValueError("One of ring or uly must be specified")

    @torch.compiler.disable(
        reason="Transformer Engine attention uses Python/pybind setup that Dynamo should not trace"
    )
    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        or_mask: Optional[torch.Tensor] = None,
        and_mask: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del local_k_slice

        if kv_cache_manager is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support KV caching")

        if isinstance(qkv, torch.Tensor):
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support packed QKV")

        if any(
            opt is not None
            for opt in (
                cu_doc_lens,
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
            )
        ):
            raise RuntimeError(
                f"'{self.__class__.__name__}' doesn't currently support intra-document masking"
            )

        q, k, v = qkv
        return self.te_attn(
            q,
            k,
            v,
            cu_seqlens_q=cu_doc_lens if cu_doc_lens is not None else cu_doc_lens_q,
            cu_seqlens_kv=cu_doc_lens if cu_doc_lens is not None else cu_doc_lens_k,
            max_seqlen_q=max_doc_len if max_doc_len is not None else max_doc_len_q,
            max_seqlen_kv=max_doc_len if max_doc_len is not None else max_doc_len_k,
        )


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
