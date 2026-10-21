from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DType, StrEnum
from ..exceptions import OLMoConfigurationError
from .config import ModuleConfig
from .functional import l2_normalize

__all__ = [
    "LayerNormType",
    "LayerNormConfig",
    "LayerNorm",
    "RMSNorm",
    "QwenRMSNorm",
    "NemotronRMSNorm",
    "CuTeRMSNorm",
    "FusedRMSNorm",
    "L2Norm",
]


class LayerNormType(StrEnum):
    """
    An enumeration of the different layer norm implementations.
    """

    default = "default"
    """
    ➡️ :class:`LayerNorm`
    """
    rms = "rms"
    """
    ➡️ :class:`RMSNorm`
    """
    qwen_rms = "qwen_rms"
    """
    ➡️ :class:`QwenRMSNorm`
    """
    nemotron_rms = "nemotron_rms"
    """
    ➡️ :class:`NemotronRMSNorm`
    """
    cute_rms = "cute_rms"
    """
    ➡️ :class:`CuTeRMSNorm`
    """
    fused_rms = "fused_rms"
    """
    ➡️ :class:`FusedRMSNorm`
    """
    l2_norm = "l2_norm"
    """
    ➡️ :class:`L2Norm`
    """


@dataclass
class LayerNormConfig(ModuleConfig):
    """
    A config for conveniently building any one of the different layer norm classes.

    See the :class:`LayerNorm` subclasses to learn which fields are valid for each implementation.
    """

    name: LayerNormType = LayerNormType.default
    """
    The name of the implementation.
    """
    eps: Optional[float] = None
    elementwise_affine: Optional[bool] = None
    bias: Optional[bool] = None
    full_precision: Optional[bool] = None
    dtype: Optional[DType] = None

    def num_params(self, size: int) -> int:
        """
        The number of parameters in the module once built.

        :param size: The size of the input along the dimension to be normalized.
        """
        elementwise_affine = (
            self.elementwise_affine
            if self.elementwise_affine is not None
            else self.name != LayerNormType.l2_norm
        )
        bias = self.bias if self.bias is not None else self.name != LayerNormType.l2_norm
        ln_params = 0
        if elementwise_affine:
            ln_params += size
            if bias:
                ln_params += size
        return ln_params

    def build(
        self,
        size: int,
        init_device: str = "cpu",
        weight_shape: Optional[Tuple[int, ...]] = None,
    ) -> "LayerNorm":
        """
        Construct the corresponding LayerNorm class.

        :param size: The size of the input along the dimension to be normalized.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        :param weight_shape: An optional shape for the element-wise weight/bias parameters, if it
            should differ from ``(size,)``. Normalization statistics are always computed over the
            last ``size`` elements of the input regardless of this value; see :class:`LayerNorm`
            for a detailed explanation. Not supported by the kernel-backed implementations
            (:class:`FusedRMSNorm`, :class:`CuTeRMSNorm`), which require the weight shape to
            match ``(size,)``.
        """
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        if (dtype := kwargs.pop("dtype", None)) is not None:
            kwargs.update(dtype=dtype.as_pt())
        if weight_shape is not None:
            kwargs.update(weight_shape=weight_shape)

        try:
            if self.name == LayerNormType.default:
                return LayerNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.rms:
                return RMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.qwen_rms:
                return QwenRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.nemotron_rms:
                return NemotronRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.cute_rms:
                return CuTeRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.fused_rms:
                return FusedRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.l2_norm:
                return L2Norm(size=size, **kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class LayerNorm(nn.Module):
    """
    Layer normalization.

    This module (and its subclasses) distinguishes between two shapes that coincide in a
    conventional layer norm but can be configured independently here:

    - ``normalized_shape``: always ``(size,)``. This determines which elements the normalization
      *statistics* (mean/variance, or RMS) are computed over — namely the last ``size`` elements
      of the input. Every window of ``size`` elements along the last dimension is normalized
      using only its own statistics.
    - ``weight_shape``: the shape of the learnable element-wise weight (gain) and bias
      parameters. By default this is also ``(size,)``, in which case the same gains are applied
      to every normalized window. Passing an explicit ``weight_shape`` decouples the two: the
      statistics are still computed over the last dimension only, but the affine transform is
      applied by broadcasting the weight/bias against the (normalized) input, so different
      windows can receive different gains.

    For example, for per-head QK-norm with per-head gains on an input of shape
    ``(batch, seq_len, n_heads, head_dim)``, use ``size=head_dim`` together with
    ``weight_shape=(n_heads, head_dim)``. Each head is normalized with its own statistics
    (from ``normalized_shape=(head_dim,)``), and the ``(n_heads, head_dim)`` weight broadcasts
    over the batch and sequence dimensions to give each head its own gains. With the default
    ``weight_shape=(head_dim,)`` instead, all heads would share a single gain vector.

    :param size: The size of the input along the dimension to be normalized.
    :param eps: The epsilon used for numerical stability.
    :param elementwise_affine: Whether to include an element-wise affine transform.
    :param bias: Whether the element-wise affine should include an element-wise bias.
        Ignored if ``elementwise_affine=False``.
    :param full_precision: Force the operation to run in full precision regardless of the input
        data type.
    :param dtype: The default data type to use for the weight and bias in the element-wise affine.
        If ``full_precision=False`` it can be useful to set this to the expected input data type.
        Ignored if ``elementwise_affine=False``.
    :param init_device: The device used when initializing the element-wise weight/bias.
    :param weight_shape: An optional shape for the element-wise weight/bias parameters, if it
        should differ from ``(size,)``. This only affects the affine transform, never the
        normalization statistics, which are always computed over the last ``size`` elements of
        the input. The shape must broadcast against the input, e.g. ``(n_heads, size)`` for a
        per-head affine on input of shape ``(batch, seq_len, n_heads, size)``.
        Ignored if ``elementwise_affine=False``.
    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        weight_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.normalized_shape = (size,)
        self.weight_shape = tuple(weight_shape) if weight_shape is not None else (size,)
        self.eps = eps
        self.full_precision = full_precision
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.weight_shape, dtype=dtype, device=init_device)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.weight_shape, dtype=dtype, device=init_device)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)
        self.output_scale = 1.0
        """
        A constant multiplier on the output, folded into the element-wise weight and bias. Only
        set it if :meth:`supports_output_scale` returns ``True``.
        """
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def supports_output_scale(self) -> bool:
        """
        Whether :data:`output_scale` can be folded into this norm's element-wise affine.
        """
        return self.weight is not None and type(self).forward in (
            LayerNorm.forward,
            RMSNorm.forward,
        )

    def _affine(self, x: torch.Tensor, param: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if param is None:
            return None
        if self.output_scale == 1.0:
            return param.type_as(x)
        return param.type_as(x) * self.output_scale

    def extra_repr(self):
        if self.weight is not None and self.bias is not None:
            return f"{tuple(self.weight.shape)}, bias=True, eps={self.eps}"
        elif self.weight is not None:
            return f"{tuple(self.weight.shape)}, eps={self.eps}"
        else:
            return f"eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm.

        :param x: The input.
        """
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype

            if self.full_precision:
                x = x.float()

            weight, bias = self._affine(x, self.weight), self._affine(x, self.bias)
            if weight is not None and self.weight_shape != self.normalized_shape:
                # 'F.layer_norm' requires the weight/bias to match 'normalized_shape', so apply
                # the element-wise affine manually (via broadcasting) after normalizing.
                x = F.layer_norm(x, self.normalized_shape, eps=self.eps)
                x = weight * x
                if bias is not None:
                    x = x + bias
            else:
                x = F.layer_norm(x, self.normalized_shape, weight=weight, bias=bias, eps=self.eps)

            return x.to(og_dtype)


class RMSNorm(LayerNorm):
    """
    RMSNorm, a simplified layer norm implementation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        :param x: The input.
        """
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype

            if self.full_precision:
                x = x.float()

            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)

            weight, bias = self._affine(x, self.weight), self._affine(x, self.bias)
            if weight is not None:
                if bias is not None:
                    x = weight * x + bias
                else:
                    x = weight * x

            return x.to(og_dtype)


class QwenRMSNorm(RMSNorm):
    """
    RMSNorm variant matching HuggingFace's ``Qwen3RMSNorm`` rounding order: the input is
    cast back to its original dtype before being multiplied by the affine weight, so the
    weight multiply happens in the input dtype rather than fp32.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype

            if self.full_precision:
                x = x.float()

            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

            if self.weight is not None:
                x = x * self.weight.type_as(x)
            if self.bias is not None:
                x = x + self.bias.type_as(x)

            return x


class NemotronRMSNorm(RMSNorm):
    """
    RMSNorm variant matching HuggingFace's Nemotron-H norm: the variance is always computed in
    fp32, then the normalized activations are cast back to the input dtype before the affine
    weight (and optional bias) are applied.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            input_dtype = x.dtype
            y = x.to(torch.float32)
            variance = y.pow(2).mean(-1, keepdim=True)
            y = y * torch.rsqrt(variance + self.eps)
            if self.weight is not None:
                y = self.weight.type_as(x) * y.to(input_dtype)
            else:
                y = y.to(input_dtype)
            if self.bias is not None:
                y = y + self.bias.type_as(y)
            return y


class CuTeRMSNorm(RMSNorm):
    """
    A CuTe-based implementation from the QuACK library.

    .. warning::
        This requires `quack <https://github.com/Dao-AILab/quack>`_ to be installed.

    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        weight_shape: Optional[Tuple[int, ...]] = None,
    ):
        from quack import rmsnorm as rms_norm_fn  # type: ignore

        if not full_precision:
            # the CUTE kernel always casts to full precision internally
            raise NotImplementedError(
                f"Currently only 'full_precision=True' is supported with '{self.__class__.__name__}'"
            )
        if weight_shape is not None and tuple(weight_shape) != (size,):
            # the CUTE kernel only supports a 1D weight matching the normalized dimension
            raise NotImplementedError(
                f"Custom 'weight_shape' is not supported with '{self.__class__.__name__}'"
            )

        super().__init__(
            size=size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            full_precision=full_precision,
            dtype=dtype,
            init_device=init_device,
        )
        self._rms_norm_fn = rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._rms_norm_fn(
            x,
            weight=None if self.weight is None else self.weight.type_as(x),
            bias=None if self.bias is None else self.bias.type_as(x),
            eps=self.eps,
        ).to(x.dtype)


class FusedRMSNorm(RMSNorm):
    """
    A "fused" triton-based implementation of :class:`RMSNorm`.

    .. warning::
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.

    .. warning::
        Currently only ``elementwise_affine=True`` is supported.
    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        weight_shape: Optional[Tuple[int, ...]] = None,
    ):
        from flash_attn.ops.triton.layer_norm import rms_norm_fn  # type: ignore

        if not elementwise_affine:
            raise NotImplementedError(
                f"Currently only 'elementwise_affine=True' is supported with '{self.__class__.__name__}'"
            )
        if not full_precision:
            # the triton kernel always casts to full precision internally
            raise NotImplementedError(
                f"Currently only 'full_precision=True' is supported with '{self.__class__.__name__}'"
            )
        if weight_shape is not None and tuple(weight_shape) != (size,):
            # the triton kernel only supports a 1D weight matching the normalized dimension
            raise NotImplementedError(
                f"Custom 'weight_shape' is not supported with '{self.__class__.__name__}'"
            )

        super().__init__(
            size=size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            full_precision=full_precision,
            dtype=dtype,
            init_device=init_device,
        )
        self._rms_norm_fn = rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        og_dtype = x.dtype
        if self.full_precision:
            x = x.float()
        return self._rms_norm_fn(
            x,
            self.weight.type_as(x),
            None if self.bias is None else self.bias.type_as(x),
            eps=self.eps,
        ).to(og_dtype)


class L2Norm(LayerNorm):
    """
    A variant of layer norm that just normalizes the last dimension of the input by its L2 norm,
    as done in nGPT.

    :param size: The size of the input along the dimension to be normalized.
    """

    def __init__(
        self,
        *,
        size: int,
    ):
        super().__init__(size=size, elementwise_affine=False, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(x)
