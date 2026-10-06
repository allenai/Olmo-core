import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config, StrEnum
from ..exceptions import OLMoConfigurationError
from .buffer_cache import BufferCache
from .config import ModuleConfig

__all__ = [
    "RoPEType",
    "RoPEConfig",
    "RoPEScalingConfig",
    "ABFRoPEScalingConfig",
    "PIRoPEScalingConfig",
    "StepwiseRoPEScalingConfig",
    "YaRNRoPEScalingConfig",
    "RotaryEmbeddingBase",
    "RotaryEmbedding",
    "FusedRotaryEmbedding",
    "ComplexRotaryEmbedding",
]


class RoPEType(StrEnum):
    """
    An enumeration of the different RoPE implementations.
    """

    default = "default"
    """
    ➡️ :class:`RotaryEmbedding`
    """
    fused = "fused"
    """
    ➡️ :class:`FusedRotaryEmbedding`
    """
    complex = "complex"
    """
    ➡️ :class:`ComplexRotaryEmbedding`
    """


def compute_inv_freqs(theta: int, dim: int, device: torch.device) -> "torch.Tensor":
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
    return inv_freq


@dataclass
class RoPEScalingConfig(Config):
    """
    Base class for RoPE scaling configs. Defines a strategy for scaling RoPE to longer sequences.
    """

    @abstractmethod
    def compute_scaled_inv_freq(
        self, theta: int, dim: int, device: torch.device
    ) -> tuple["torch.Tensor", float]:
        """Compute the scaled inverse frequencies for RoPE, and the attention rescaling factor."""
        raise NotImplementedError

    @abstractmethod
    def to_hf_config(self) -> dict:
        """Convert to HuggingFace rope_scaling format."""
        raise NotImplementedError


@dataclass
class ABFRoPEScalingConfig(RoPEScalingConfig):
    """Absolute base frequency scaling (ABF). Simply uses a new base frequency parameter."""

    attention_rescale_factor: float = 1.0
    """
    Factor to rescale attention scores by when using scaled RoPE. Can be used to compensate for
    the larger effective context. 1.0 means no rescaling.
    """

    new_theta: int = 8_000_000

    def compute_scaled_inv_freq(
        self, theta: int, dim: int, device: torch.device
    ) -> tuple["torch.Tensor", float]:
        del theta  # unused
        inv_freq = compute_inv_freqs(self.new_theta, dim, device)
        return inv_freq, self.attention_rescale_factor

    def to_hf_config(self) -> dict:
        """ABF scaling doesn't have a direct HF equivalent (just modify the config's base frequency)."""
        raise NotImplementedError


@dataclass
class PIRoPEScalingConfig(RoPEScalingConfig):
    """
    Position-Interpolation (PI) RoPE scaling from Chen et al. (https://arxiv.org/pdf/2306.15595)

    Interpolate the rotary angles instead of extrapolating them when the context window at
    inference time exceeds the window used during training. In practice, this amounts to linearly
    *compressing* the original position indices by a constant factor ``factor``.
    """

    attention_rescale_factor: float = 1.0
    """
    Factor to rescale attention scores by when using scaled RoPE. Can be used to compensate for
    the larger effective context. 1.0 means no rescaling.
    """

    factor: float = 2.0
    """Context expansion multiplier. If factor = 1, reduces to vanilla RoPE."""

    def compute_scaled_inv_freq(
        self, theta: int, dim: int, device: torch.device
    ) -> tuple["torch.Tensor", float]:
        inv_freq = compute_inv_freqs(theta, dim, device)

        # Positional-interpolation scales the *positions* by 1/factor. This is
        # equivalent to scaling the inverse frequencies by the same amount.
        if self.factor != 1.0:
            inv_freq = inv_freq / self.factor

        return inv_freq, self.attention_rescale_factor

    def to_hf_config(self) -> dict:
        """PI scaling corresponds to HF's linear scaling."""
        return {"rope_type": "linear", "factor": self.factor}


@dataclass
class StepwiseRoPEScalingConfig(RoPEScalingConfig):
    """Step-wise RoPE scaling (aka "Per-frequency" scaling or Llama-3.1 scaling).

    Reference: `Llama-3.1-8B README <https://huggingface.co/meta-llama/Llama-3.1-8B/blob/refs%2Fpr%2F3/README.md>`_

    Scales RoPE to longer sequence lengths by interpolating between high- and low-frequency components.

    1. **High-frequency band** (short wavelengths) – keeps the original frequencies unchanged.
        These correspond to the very first dimensions of the rotary embedding and already encode
        short-range ordering well.
    2. **Low-frequency band** (long wavelengths) – divides the original inverse frequency by
        ``factor`` (equivalently, multiplies the wavelength by ``factor``).  This has the effect of
        spreading the very low frequencies across a longer context window (similar to PI scaling).
    3. **Medium-frequency band** – linearly interpolates (in inverse-frequency space) between the
        unscaled and the fully-scaled value so that the full spectrum changes smoothly.
    """

    attention_rescale_factor: float = 1.0
    """
    Factor to rescale attention scores by when using scaled RoPE. Can be used to compensate for
    the larger effective context. 1.0 means no rescaling.
    """

    factor: float = 32.0
    """Context expansion multiplier applied to the long-wavelength part of the spectrum."""

    low_freq_proportion: float = 0.0
    """
    Proportion of the spectrum that is considered *low-frequency*. Is translated into a concrete
    wavelength that represents the upper bound of the *low-frequency* band.
    """

    high_freq_proportion: float = 0.25
    """
    Proportion of the spectrum that is considered *high-frequency*. Is translated into a concrete
    wavelength that represents the lower bound of the *high-frequency* band.
    """

    old_context_len: int = 8192
    """Maximum sequence length the *base* model was originally trained with."""

    def compute_scaled_inv_freq(
        self, theta: int, dim: int, device: torch.device
    ) -> tuple["torch.Tensor", float]:
        inv_freq = compute_inv_freqs(theta, dim, device)

        # Convert the low/high-frequency *denominators* into concrete wavelength thresholds
        low_freq_factor = 1.0 / (1 - self.low_freq_proportion)
        high_freq_factor = 1.0 / self.high_freq_proportion
        low_band_threshold = self.old_context_len / low_freq_factor
        high_band_threshold = self.old_context_len / high_freq_factor

        # Current (un-scaled) wavelengths associated with each inverse-frequency component
        wavelen = 2 * math.pi / inv_freq

        # 1. Low-frequency band  (wavelen > low_band_threshold) -> fully scaled.
        # 2. High-frequency band (wavelen < high_band_threshold) -> unchanged.
        inv_freq = torch.where(wavelen > low_band_threshold, inv_freq / self.factor, inv_freq)

        # 3. Mid-frequency band  (between the two thresholds) -> smoothly interpolated.
        interp_weight = (self.old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (1 - interp_weight) * inv_freq / self.factor + interp_weight * inv_freq
        is_mid_band = (wavelen <= low_band_threshold) & (wavelen >= high_band_threshold)

        return torch.where(is_mid_band, smoothed_inv_freq, inv_freq), self.attention_rescale_factor

    def to_hf_config(self) -> dict:
        """Stepwise scaling corresponds to HF's llama3 scaling."""
        return {
            "rope_type": "llama3",
            "factor": self.factor,
            "original_max_position_embeddings": self.old_context_len,
            "low_freq_factor": 1.0 / (1 - self.low_freq_proportion),
            "high_freq_factor": 1.0 / self.high_freq_proportion,
        }


@dataclass
class YaRNRoPEScalingConfig(RoPEScalingConfig):
    """Yet-another RoPE interpolatioN (YaRN) scaling.

    Reference: https://arxiv.org/abs/2309.00071

    Extends a model’s context window by *blending* two sets of inverse frequencies:

    1. **Interpolation frequencies** – the original RoPE frequencies divided
       by ``factor``.  These allow the model to *compress* positions and hence
       attend across a longer sequence.
    2. **Extrapolation frequencies** – the unmodified RoPE frequencies the
       model was trained with.

    A *linear ramp* (controlled by ``beta_fast`` / ``beta_slow``) determines
    which of the two spectra dominates for each dimension so that high-
    frequency bands remain intact while very low frequencies are fully scaled.

    Besides re-mapping the rotary angles, YaRN rescales the attention logits by
    ``attention_factor`` (computed via *m-scale*) to compensate for the larger
    effective context.
    """

    factor: float = 8.0
    """Context expansion multiplier. (e.g. 8× gives ≈8-times longer context length)."""

    beta_fast: int = 32
    """Dimensional cut-off that delimits the start (high-freq) of the ramp region."""

    beta_slow: int = 1
    """Dimensional cut-off that delimits the end (low-freq) of the ramp region."""

    old_context_len: int = 8192
    """Maximum sequence length that the *base* model was originally trained with."""

    _IGNORE_FIELDS: ClassVar[Tuple[str, ...]] = ("attention_rescale_factor",)

    def compute_scaled_inv_freq(
        self, theta: int, dim: int, device: torch.device
    ) -> tuple["torch.Tensor", float]:
        # 1. Base (un-scaled) inverse frequencies and purely scaled copy
        inv_freq_extrapolation = compute_inv_freqs(theta, dim, device)
        inv_freq_interpolation = inv_freq_extrapolation / self.factor

        # 2. Identify the start/end of the linear-ramp blend region
        half_dim = inv_freq_extrapolation.shape[0]
        idx = torch.arange(half_dim, device=device, dtype=torch.float32)  # 0 … dim/2-1

        def _dim_from_rot(n_rot: int) -> float:
            return (
                dim
                * math.log(self.old_context_len / (n_rot * 2.0 * math.pi))
                / (2.0 * math.log(theta))
            )

        low = max(int(math.floor(_dim_from_rot(self.beta_fast))), 0)
        high = min(int(math.ceil(_dim_from_rot(self.beta_slow))), half_dim - 1)
        span = max(high - low, 1e-3)  # avoid division-by-zero
        ramp = ((idx - low) / span).clamp_(0, 1)  # 0 → extrapolation, 1 → interpolation

        # 3. Blend the two spectra according to the ramp weights
        inv_freq = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1.0 - ramp)

        return inv_freq, self.get_attention_rescale_factor()

    def get_attention_rescale_factor(self) -> float:
        """Compute the attention rescale factor based on section 3.4 of the YaRN paper"""
        return 0.1 * math.log(self.factor) + 1.0

    def to_hf_config(self) -> dict:
        """YaRN scaling corresponds to HF's yarn scaling."""
        return {
            "rope_type": "yarn",
            "factor": self.factor,
            "original_max_position_embeddings": self.old_context_len,
            "beta_fast": self.beta_fast,
            "beta_slow": self.beta_slow,
            "attention_factor": self.get_attention_rescale_factor(),
        }


@dataclass
class RoPEConfig(ModuleConfig):
    """
    A config for conveniently building any of the different RoPE classes.

    See the individual :class:`RotaryEmbedding` subclasses for a description of the
    configuration options.
    """

    name: RoPEType = RoPEType.default
    """
    The name of the implementation.
    """
    theta: int = 500_000
    """The base frequency parameter for the RoPE."""

    full_precision: bool = True
    """Whether to always apply RoPE in full precision regardless of the input data type."""

    no_global_rope: bool = False
    """Whether to disable RoPE on global (non-SWA) attention layers."""

    scaling: Optional[RoPEScalingConfig] = None
    """The scaling config to apply to RoPE."""

    partial_rotary_factor: float = 1.0
    """
    Fraction of each head dimension to apply RoPE to. When less than 1.0, only the leading
    ``int(head_size * partial_rotary_factor)`` dimensions are rotated; the rest pass through
    unchanged. Used by Qwen3.5 (default 0.25).
    """

    def build(
        self,
        head_size: int,
        cache: Optional[BufferCache] = None,
    ) -> "RotaryEmbeddingBase":
        """
        Construct the corresponding RoPE class.

        :param head_size: The size of the attention heads.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.pop("no_global_rope")
        kwargs.update(head_size=head_size, cache=cache)

        try:
            if self.name == "default":
                return RotaryEmbedding(**kwargs)
            elif self.name == "fused":
                return FusedRotaryEmbedding(**kwargs)
            elif self.name == "complex":
                return ComplexRotaryEmbedding(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


@dataclass
class RoPEBuffers:
    pos_sin: Optional[torch.Tensor] = None
    """Precomputed sine positional embeddings for RoPE."""

    pos_cos: Optional[torch.Tensor] = None
    """Precomputed cosine positional embeddings for RoPE."""

    freqs_cis: Optional[torch.Tensor] = None
    """Precomputed complex frequency tensor (used by complex RoPE implementations)."""


class RotaryEmbeddingBase(nn.Module):
    """
    Base class for RoPE implementations.
    """

    def __init__(
        self,
        *,
        head_size: int,
        theta: int = 500_000,
        full_precision: bool = True,
        cache: Optional[BufferCache] = None,
        scaling: Optional[RoPEScalingConfig] = None,
        partial_rotary_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = head_size
        self.rotary_dim = int(head_size * partial_rotary_factor)
        self.theta = theta
        self.full_precision = full_precision
        self.scaling = scaling
        self._cache = (cache or BufferCache()).with_namespace(
            f"RoPE_theta={self.theta}_partial={partial_rotary_factor}_scaling={repr(self.scaling)}"
        )

    @abstractmethod
    def warmup_cache(self, max_seq_len: int, device: torch.device):
        """
        Warmup the buffer cache.
        """
        raise NotImplementedError

    @abstractmethod
    def get_buffers(self, max_seq_len: int, device: torch.device) -> RoPEBuffers:
        """
        Get the cached buffers.
        """
        raise NotImplementedError


class RotaryEmbedding(RotaryEmbeddingBase):
    """
    `Rotary positional embeddings (RoPE) <https://arxiv.org/abs/2104.09864>`_.

    .. seealso::
        - :class:`ComplexRotaryEmbedding`
        - :class:`FusedRotaryEmbedding`

    :param head_size: The size of the attention heads.
    :param theta: The theta base value to use.
    :param full_precision: Always apply RoPE in full precision regardless of the input data type.
    :param scaling: The scaling config.
    """

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_rotary_embedding(max_seq_len, device)

    def get_buffers(self, max_seq_len: int, device: torch.device) -> RoPEBuffers:
        pos_sin, pos_cos = self._get_rotary_embedding(max_seq_len, device)
        return RoPEBuffers(pos_sin=pos_sin, pos_cos=pos_cos)

    def _get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :returns: The sine and cosine positional embeddings of shape ``(seq_len, head_size)``.
        """
        if (
            (pos_sin := self._cache.get("rope_pos_sin")) is not None
            and (pos_cos := self._cache.get("rope_pos_cos")) is not None
            # DANGER: possible sharp edge when using variable seq_len and a scaling config
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self._cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self._cache["rope_pos_cos"] = pos_cos
            return pos_sin[:seq_len, :], pos_cos[:seq_len, :]

        with torch.autocast(device.type, enabled=False):
            if self.scaling is None:
                inv_freq = compute_inv_freqs(self.theta, self.rotary_dim, device)
                attention_rescale_factor = 1.0
            else:
                inv_freq, attention_rescale_factor = self.scaling.compute_scaled_inv_freq(
                    theta=self.theta, dim=self.rotary_dim, device=device
                )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin(), positions.cos()

        # https://arxiv.org/pdf/2309.00071 (section 3.4)
        pos_sin = pos_sin * attention_rescale_factor
        pos_cos = pos_cos * attention_rescale_factor

        self._cache["rope_pos_sin"] = pos_sin
        self._cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        rotary_dim = pos_sin.shape[-1]
        if rotary_dim == t.shape[-1]:
            return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)
        t_rot, t_pass = t[..., :rotary_dim], t[..., rotary_dim:]
        t_rot = ((t_rot * pos_cos) + (self._rotate_half(t_rot) * pos_sin)).to(t.dtype)
        return torch.cat([t_rot, t_pass], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        head_first: bool = True,
        start_pos: Optional[int] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query (``q``) and key (``k``) matrices.

        :param q: The query matrix of shape ``(batch_size, num_heads, seq_len, head_size)``
            if ``head_first`` (the default) otherwise ``(batch_size, seq_len, num_heads, head_size)``.
        :param k: The key matrix of shape ``(batch_size, num_kv_heads, seq_len, head_size)``
            if ``head_first`` (the default) otherwise
            ``(batch_size, seq_len, num_kv_heads, head_size)``.
        :param head_first: If the head dim comes before the sequence dim.
        :param start_pos: The absolute position of the first query token (eg for decoding
            where the first query token is just the most recently decoded token).
        :param cu_doc_lens: Cumulative document lengths for intra-document RoPE in packed
            inputs. When supplied, each document's tokens receive positions starting from 0
            (matching per-document forwards). Mutually exclusive with ``start_pos``.

        :returns: The query and key matrices after RoPE has been applied.
        """
        if freqs_cis is not None:
            raise RuntimeError(f"'freqs_cis' is invalid for {self.__class__.__name__}")

        if cu_doc_lens is not None and start_pos is not None:
            raise RuntimeError("'cu_doc_lens' and 'start_pos' are mutually exclusive")

        if position_ids is not None:
            if cu_doc_lens is not None:
                raise RuntimeError("'position_ids' and 'cu_doc_lens' are mutually exclusive")
            if start_pos is not None:
                raise RuntimeError("'position_ids' and 'start_pos' are mutually exclusive")

        if head_first:
            q_len = q.size(2)
            k_len = k.size(2)
        else:
            q_len = q.size(1)
            k_len = k.size(1)

        if self.full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            seq_len_needed = (start_pos + k_len) if start_pos is not None else k_len
            if pos_sin is None or pos_cos is None:
                pos_sin, pos_cos = self._get_rotary_embedding(seq_len_needed, q_.device)
            pos_sin, pos_cos = pos_sin.type_as(q_), pos_cos.type_as(q_)

            if pos_sin.size(-2) < seq_len_needed or pos_cos.size(-2) < seq_len_needed:
                raise RuntimeError(
                    f"RoPE buffers shorter than required: need {seq_len_needed}, "
                    f"have {pos_sin.size(-2)}."
                )

            def _broadcast(t: torch.Tensor) -> torch.Tensor:
                return t[None, None, :, :] if head_first else t[None, :, None, :]

            if cu_doc_lens is not None:
                if q_len != k_len:
                    raise RuntimeError(
                        "'cu_doc_lens' requires q_len == k_len (no kv-cache packed mode)"
                    )
                B = q_.size(0)
                flat_idx = torch.arange(B * k_len, device=q_.device, dtype=cu_doc_lens.dtype)
                doc_id = torch.bucketize(flat_idx, cu_doc_lens[1:], right=True)
                pos_idx = (flat_idx - cu_doc_lens[doc_id]).view(B, k_len)
                sin_sel = pos_sin.index_select(0, pos_idx.reshape(-1)).view(B, k_len, -1)
                cos_sel = pos_cos.index_select(0, pos_idx.reshape(-1)).view(B, k_len, -1)
                if head_first:
                    sin_qk = sin_sel.unsqueeze(1)
                    cos_qk = cos_sel.unsqueeze(1)
                else:
                    sin_qk = sin_sel.unsqueeze(2)
                    cos_qk = cos_sel.unsqueeze(2)
                q_ = self._apply_rotary_pos_emb(sin_qk, cos_qk, q_)
                k_ = self._apply_rotary_pos_emb(sin_qk, cos_qk, k_)
            elif position_ids is not None:
                # Explicit per-token positions (e.g. subsegment / branch packing where
                # parallel branches share an overlapping position range). Gather the
                # precomputed sin/cos for each token's position. ``position_ids`` has
                # shape ``(B, k_len)`` and every entry must be < ``seq_len_needed``.
                if q_len != k_len:
                    raise RuntimeError("'position_ids' requires q_len == k_len")
                pos_idx = position_ids.to(torch.long)
                sin_sel = pos_sin.index_select(0, pos_idx.reshape(-1)).view(*pos_idx.shape, -1)
                cos_sel = pos_cos.index_select(0, pos_idx.reshape(-1)).view(*pos_idx.shape, -1)
                if head_first:
                    sin_qk = sin_sel.unsqueeze(1)
                    cos_qk = cos_sel.unsqueeze(1)
                else:
                    sin_qk = sin_sel.unsqueeze(2)
                    cos_qk = cos_sel.unsqueeze(2)
                q_ = self._apply_rotary_pos_emb(sin_qk, cos_qk, q_)
                k_ = self._apply_rotary_pos_emb(sin_qk, cos_qk, k_)
            else:
                q_abs_start = start_pos if start_pos is not None else (k_len - q_len)
                k_abs_start = start_pos if start_pos is not None else 0

                sin_q = _broadcast(pos_sin[q_abs_start : q_abs_start + q_len, :])
                cos_q = _broadcast(pos_cos[q_abs_start : q_abs_start + q_len, :])
                sin_k = _broadcast(pos_sin[k_abs_start : k_abs_start + k_len, :])
                cos_k = _broadcast(pos_cos[k_abs_start : k_abs_start + k_len, :])

                q_ = self._apply_rotary_pos_emb(sin_q, cos_q, q_)
                k_ = self._apply_rotary_pos_emb(sin_k, cos_k, k_)

        return q_.type_as(q), k_.type_as(k)


class FusedRotaryEmbedding(RotaryEmbeddingBase):
    """
    A "fused" triton-based implementation of :class:`RotaryEmbedding`.

    .. warning::
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.

    :param head_size: The size of the attention heads.
    :param theta: The theta base value to use.
    :param full_precision: Always apply RoPE in full precision regardless of the input data type.
    :param scaling: The scaling config.
    """

    def __init__(
        self,
        *,
        head_size: int,
        theta: int = 500_000,
        full_precision: bool = True,
        cache: Optional[BufferCache] = None,
        scaling: Optional[RoPEScalingConfig] = None,
        partial_rotary_factor: float = 1.0,
    ):
        from flash_attn.layers.rotary import apply_rotary_emb_qkv_  # type: ignore

        if partial_rotary_factor != 1.0:
            raise OLMoConfigurationError(
                "partial_rotary_factor is not yet supported for FusedRotaryEmbedding"
            )

        super().__init__(
            head_size=head_size,
            theta=theta,
            full_precision=full_precision,
            cache=cache,
            scaling=scaling,
            partial_rotary_factor=partial_rotary_factor,
        )
        self._apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_rotary_embedding(max_seq_len, device)

    def get_buffers(self, max_seq_len: int, device: torch.device) -> RoPEBuffers:
        pos_sin, pos_cos = self._get_rotary_embedding(max_seq_len, device)
        return RoPEBuffers(pos_sin=pos_sin, pos_cos=pos_cos)

    def _get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :returns: The sine and cosine positional embeddings of shape ``(seq_len, head_size // 2)``.
        """
        if (
            (pos_sin := self._cache.get("rope_pos_sin")) is not None
            and (pos_cos := self._cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self._cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self._cache["rope_pos_cos"] = pos_cos
            return pos_sin, pos_cos

        with torch.autocast(device.type, enabled=False):
            if self.scaling is None:
                inv_freq = compute_inv_freqs(self.theta, self.dim, device)
                attention_rescale_factor = 1.0
            else:
                inv_freq, attention_rescale_factor = self.scaling.compute_scaled_inv_freq(
                    theta=self.theta, dim=self.dim, device=device
                )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)  # (seq_len, head_size // 2)
            # Note: no concat here, unlike the non-fused implementation
            pos_sin, pos_cos = freqs.sin(), freqs.cos()  # 2x (seq_len, head_size // 2)

        pos_sin = pos_sin * attention_rescale_factor
        pos_cos = pos_cos * attention_rescale_factor

        self._cache["rope_pos_sin"] = pos_sin
        self._cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def forward(
        self,
        qkv: torch.Tensor,
        start_pos: Optional[int] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply RoPE to ``qkv``.

        .. warning::
            This operates on ``qkv`` *in place* unless ``full_precision=True`` and ``qkv``
            is not in full precision.

        :param qkv: The query, key, and value matrix of shape
            ``(batch_size, seq_len, 3, n_heads, head_size)``.
        :param start_pos: The absolute position of the first query token (eg for decoding
            where the first query token is just the most recently decoded token).
        :return: The qkv tensor after applying RoPE, of the same shape and dtype as the input.
        """
        if freqs_cis is not None:
            raise RuntimeError(f"'freqs_cis' is invalid for {self.__class__.__name__}")

        if self.full_precision:
            qkv_ = qkv.float()
        else:
            qkv_ = qkv

        seqlen_offsets = start_pos or 0
        if pos_sin is None or pos_cos is None:
            pos_sin, pos_cos = self._get_rotary_embedding(
                qkv_.size(1) + seqlen_offsets, qkv_.device
            )
        pos_sin, pos_cos = pos_sin.type_as(qkv_), pos_cos.type_as(qkv_)
        qkv_ = self._apply_rotary_emb_qkv_(
            qkv_, pos_cos, pos_sin, interleaved=False, seqlen_offsets=seqlen_offsets
        )
        return qkv_.type_as(qkv)


class ComplexRotaryEmbedding(RotaryEmbeddingBase):
    """
    An implementation of `RoPE <https://arxiv.org/abs/2104.09864>`_ as a rotation in complex space.

    :param head_size: The dimensionality of the attention heads.
    :param theta: The theta base value to use.
    :param full_precision: Always apply RoPE in full precision regardless of the input data type.
    """

    def __init__(
        self,
        *,
        head_size: int,
        theta: int = 500_000,
        full_precision: bool = True,
        cache: Optional[BufferCache] = None,
        scaling: Optional[RoPEScalingConfig] = None,
        partial_rotary_factor: float = 1.0,
    ):
        if scaling is not None:
            raise OLMoConfigurationError("scaling is not yet supported for ComplexRotaryEmbedding")

        super().__init__(
            head_size=head_size,
            theta=theta,
            full_precision=full_precision,
            cache=cache,
            scaling=scaling,
            partial_rotary_factor=partial_rotary_factor,
        )

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_rotary_embedding(max_seq_len, device)

    def get_buffers(self, max_seq_len: int, device: torch.device) -> RoPEBuffers:
        freqs_cis = self._get_rotary_embedding(max_seq_len, device)
        return RoPEBuffers(freqs_cis=freqs_cis)

    def _get_rotary_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        :returns: The complex frequency tensor of shape ``(seq_len, head_size // 2)``.
        """
        if (freqs_cis := self._cache.get("rope_freqs_cis")) is not None and freqs_cis.shape[
            -2
        ] >= seq_len:
            if freqs_cis.device != device:
                freqs_cis = freqs_cis.to(device)
                self._cache["rope_freqs_cis"] = freqs_cis
            return freqs_cis[:seq_len, :]

        with torch.autocast(device.type, enabled=False):
            inv_freq = compute_inv_freqs(self.theta, self.rotary_dim, device)
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self._cache["rope_freqs_cis"] = freqs_cis
        return freqs_cis

    def _apply_rotary_pos_emb(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(x * freqs_cis).flatten(3)

    def _apply_rotary_pos_emb_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        head_first: bool,
        q_abs_start: int,
        k_abs_start: int,
        q_len: int,
        k_len: int,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim < self.dim:
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot = torch.view_as_complex(q_rot.reshape(*q_rot.shape[:-1], -1, 2))
            k_rot = torch.view_as_complex(k_rot.reshape(*k_rot.shape[:-1], -1, 2))
        else:
            q_rot = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2))
            k_rot = torch.view_as_complex(k.reshape(*k.shape[:-1], -1, 2))
            q_pass = k_pass = None

        if head_first:
            q_rot = self._apply_rotary_pos_emb(
                freqs_cis[None, None, q_abs_start : q_abs_start + q_len, :], q_rot
            )
            k_rot = self._apply_rotary_pos_emb(
                freqs_cis[None, None, k_abs_start : k_abs_start + k_len, :], k_rot
            )
        else:
            q_rot = self._apply_rotary_pos_emb(
                freqs_cis[None, q_abs_start : q_abs_start + q_len, None, :], q_rot
            )
            k_rot = self._apply_rotary_pos_emb(
                freqs_cis[None, k_abs_start : k_abs_start + k_len, None, :], k_rot
            )

        if q_pass is not None and k_pass is not None:
            return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)
        return q_rot, k_rot

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        head_first: bool = True,
        start_pos: Optional[int] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query (``q``) and key (``k``) matrices.

        :param q: The query matrix of shape ``(batch_size, num_heads, seq_len, head_size)``
            if ``head_first`` (the default) otherwise ``(batch_size, seq_len, num_heads, head_size)``.
        :param k: The key matrix of shape ``(batch_size, num_kv_heads, seq_len, head_size)``
            if ``head_first`` (the default) otherwise
            ``(batch_size, seq_len, num_kv_heads, head_size)``.
        :param head_first: If the head dim comes before the sequence dim.
        :param start_pos: The absolute position of the first query token (eg for decoding
            where the first query token is just the most recently decoded token).

        :returns: The query and key matrices after RoPE has been applied.
        """
        if pos_sin is not None or pos_cos is not None:
            raise RuntimeError(f"'pos_sin' and 'pos_cos' are invalid for {self.__class__.__name__}")

        if head_first:
            q_len = q.size(2)
            k_len = k.size(2)
        else:
            q_len = q.size(1)
            k_len = k.size(1)

        if self.full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            seq_len_needed = (start_pos + k_len) if start_pos is not None else k_len
            if freqs_cis is None:
                freqs_cis = self._get_rotary_embedding(seq_len_needed, q_.device)
            q_abs_start = start_pos if start_pos is not None else (k_len - q_len)
            k_abs_start = start_pos if start_pos is not None else 0
            q_, k_ = self._apply_rotary_pos_emb_to_qk(
                q_,
                k_,
                head_first=head_first,
                q_abs_start=q_abs_start,
                k_abs_start=k_abs_start,
                q_len=q_len,
                k_len=k_len,
                freqs_cis=freqs_cis,
            )

        return q_.type_as(q), k_.type_as(k)
