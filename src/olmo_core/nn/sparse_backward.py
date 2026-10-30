"""
Backward compression with an EXACT forward for frozen-base + LoRA linears (``--st-otr-sbw-mode``, 2026-10-30).

With whole-model LoRA over a frozen base (the recipe ``lora64all32r20to7``), every block's backward is the
input-gradient vector-Jacobian product (VJP) of its frozen linears, ``dx = dy W``, plus the attention / GDN core
VJPs, the norms and the adapters' own gradients. The linear VJPs are most of it. This module replaces each adapted
linear's autograd with a custom function whose forward is bit-for-bit the usual ``y = x W^T + b + s B (A x)`` and
whose backward computes the linear and LoRA VJPs only for a subset of token rows of ``dy`` (rows not selected
contribute nothing through this linear: neither to ``dx`` nor to the adapter gradients). The residual stream, the
norms and the attention / GDN core VJPs stay dense, so every token still receives the gradient that reaches it
through the residual path and through attention.

Modes (selection is per linear call, over the flattened micro-batch's rows whose ``dy`` is nonzero):

  ``topk``    the top ``frac`` of rows by ``||dy_t||`` (deterministic, biased).
  ``energy``  the fewest rows whose ``||dy_t||^2`` covers ``energy`` of the call's total (deterministic, biased;
              adaptive count).
  ``sample``  each row independently with ``p_t = min(1, c ||dy_t||)``, ``c`` set so ``sum p_t = frac * n``; a
              kept row is reweighted by ``1 / p_t`` (unbiased for ``dx`` and for the adapter gradients).
  ``sketch``  every row, but ``dx = dy W`` through a cheaper operator: the exact top-``rank`` SVD part of ``W``
              plus a fresh Gaussian sketch (``sketch`` columns) of the residual ``E = W - W_r``:
              ``dy E ~= (dy E Omega) Omega^T / m`` (unbiased); the adapter gradients stay exact. ``sketch=0`` is the
              biased truncated-SVD backward. The SVD is computed once per linear from the (frozen) weight.

FLOP accounting charges exactly what runs, relative to the dense charge the meter already applies (the meter
prices a frozen linear's backward as its input-gradient VJP, ``2 in out`` per token, and the adapters at
``4 r (in + out)`` per token for their backward): each backward call adds ``actual - dense`` to
``model._flop_extra_actual``, where ``dense`` counts only the rows whose ``dy`` is nonzero (rows the meter charges
but whose ``dy`` is zero -- a handful per row -- are left charged: conservative) and ``actual`` adds the row-norm
(``2 out`` per row) and selection overhead. Sketch mode adds its per-call sketch (``2 out in m``).
"""

from __future__ import annotations

import logging
import math
import weakref
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = ["SparseBackwardConfig", "install_sparse_backward", "sparse_backward_metrics", "select_rows"]

MODES = ("off", "topk", "energy", "sample", "sketch")


@dataclass
class SparseBackwardConfig:
    mode: str = "off"
    frac: float = 0.1
    """topk: kept share of the nonzero rows; sample: EXPECTED kept share."""
    energy: float = 0.95
    """energy: share of sum_t ||dy_t||^2 the kept rows must cover."""
    rank: int = 256
    """sketch: exact top-``rank`` SVD part."""
    sketch: int = 256
    """sketch: Gaussian sketch width of the residual (0 = truncated SVD only, biased)."""
    start_block: int = 0
    """Only adapted linears in blocks >= start_block are compressed."""
    end_block: int = 1 << 30
    """... and in blocks < end_block."""
    min_rows: int = 1
    """Never keep fewer rows than this per call (topk / energy)."""
    dense_every: int = 0
    """If > 0, every ``dense_every``-th optimizer step runs the EXACT backward (the compressed one otherwise), so no
    component of the gradient is dropped on every step. Driven by the online router callback."""
    lora_all_rows: bool = False
    """Adapter weight gradients from EVERY nonzero row (exact given the arriving dy; ~4 r (in + out) per row, a few %
    of the VJP), while the input gradient ``dx`` is still computed for the selected rows only."""

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"sparse backward mode {self.mode!r} not in {MODES}")
        if self.mode in ("topk", "sample") and not 0.0 < self.frac <= 1.0:
            raise ValueError(f"sparse backward frac must be in (0, 1], got {self.frac}")
        if self.mode == "energy" and not 0.0 < self.energy <= 1.0:
            raise ValueError(f"sparse backward energy must be in (0, 1], got {self.energy}")
        if self.mode == "sketch" and self.rank <= 0:
            raise ValueError("sketch mode needs rank > 0")

    @property
    def active(self) -> bool:
        return self.mode != "off"


class _State:
    """Shared (non-Module) holder: the config, the model (weakref) whose FLOP counter is charged, and stats."""

    def __init__(self, model: Any, cfg: SparseBackwardConfig):
        self.cfg = cfg
        self._model = weakref.ref(model)
        self.enabled = True
        self.gen: Optional[torch.Generator] = None
        self.svd_flops = 0.0
        self.assume_no_pad = False
        """True when every row of every call is a real token (the probe's unpadded single rows)."""
        self.real_rows: Dict[int, float] = {}
        """flattened row count N of a call -> real (non-pad) tokens among them, registered by the forward (the online
        router, per stage) and read in the backward: the meter charged the dense VJP for exactly those rows."""
        self.reset_stats()

    def reset_stats(self) -> None:
        self.rows_kept = 0.0
        self.rows_nz = 0.0
        self.flops_dense = 0.0
        self.flops_actual = 0.0
        self.calls = 0

    def dense_rows(self, N: int, nz: int) -> float:
        """Rows the meter charged a dense VJP for in a call over N flattened rows: all of them (no padding), the
        registered real-token count, or (unknown) the nonzero-dy rows -- conservative, since every charged row with
        dy = 0 is one the compressed backward skips but is not credited for."""
        if self.assume_no_pad:
            return float(N)
        r = self.real_rows.get(int(N))
        return float(r) if r is not None else float(nz)

    def charge(self, dense: float, actual: float, kept: float, nz: float) -> None:
        self.rows_kept += kept
        self.rows_nz += nz
        self.flops_dense += dense
        self.flops_actual += actual
        self.calls += 1
        self._to_model(actual - dense)

    def add_svd(self, flops: float) -> None:
        self.svd_flops += flops
        self._to_model(flops)

    def _to_model(self, n: float) -> None:
        m = self._model()
        if m is not None:
            m._flop_extra_actual = float(getattr(m, "_flop_extra_actual", 0.0)) + float(n)


def _water_fill(s: torch.Tensor, budget: float) -> torch.Tensor:
    """p = min(1, c s) with sum p = budget (s >= 0, budget <= count of s > 0). Returns p (float32)."""
    n = s.numel()
    if budget >= float((s > 0).sum()):
        return (s > 0).to(torch.float32)
    sd, order = torch.sort(s, descending=True)
    suf = torch.flip(torch.cumsum(torch.flip(sd, [0]), 0), [0])  # suf[j] = sum_{i >= j} sd_i
    j = torch.arange(n, device=s.device, dtype=torch.float32)
    c = (budget - j) / suf.clamp(min=1e-30)
    ok = (c * sd <= 1.0) & (budget - j > 0)
    first = int(torch.nonzero(ok, as_tuple=True)[0][0]) if bool(ok.any()) else n - 1
    cj = float(c[first])
    return (cj * s).clamp(max=1.0)


def select_rows(norm2: torch.Tensor, cfg: SparseBackwardConfig, gen: Optional[torch.Generator] = None):
    """Row selection over ``norm2`` (N,) = ||dy_t||^2. Returns ``(idx, weight or None, n_nonzero)``."""
    nz = norm2 > 0
    n = int(nz.sum())
    if n == 0:
        return norm2.new_zeros(0, dtype=torch.long), None, 0
    if cfg.mode == "topk":
        k = min(n, max(cfg.min_rows, int(math.ceil(cfg.frac * n))))
        idx = torch.topk(norm2, k, sorted=False).indices
        return idx, None, n
    if cfg.mode == "energy":
        sd, order = torch.sort(norm2, descending=True)
        cs = torch.cumsum(sd.double(), 0)
        k = int(torch.searchsorted(cs, cs[-1] * cfg.energy).item()) + 1
        k = min(n, max(cfg.min_rows, k))
        return order[:k], None, n
    if cfg.mode == "sample":
        s = norm2.float().sqrt()
        p = _water_fill(s, cfg.frac * n)
        u = torch.rand(p.shape, device=p.device, generator=gen)
        keep = (u < p) & nz
        idx = torch.nonzero(keep, as_tuple=True)[0]
        return idx, (1.0 / p[idx]), n
    raise ValueError(cfg.mode)


def _svd_factors(mod: nn.Linear, rank: int):
    """(U S (out, r), V (in, r)) of the frozen weight. The full factorization (fp32 SVD, stored in the weight dtype)
    is computed once per linear and cached on the module; its one-time cost goes to the FLOP counter (not the
    per-call stats)."""
    f = getattr(mod, "_sbw_svd", None)
    if f is None:
        W = mod.weight.detach().to(torch.float64 if mod.weight.dtype == torch.float64 else torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        f = mod._sbw_svd = ((U * S).to(mod.weight.dtype).contiguous(), Vh.T.to(mod.weight.dtype).contiguous())
        o, i = W.shape
        mn = min(o, i)
        mod._sbw_state.add_svd(4.0 * o * i * mn + 22.0 * mn ** 3)  # one-time SVD, conventional estimate
    r = min(rank, f[0].shape[1])
    return f[0][:, :r], f[1][:, :r]


class _SparseLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lora_A, lora_B, scale, mod):  # type: ignore[override]
        y = F.linear(x, weight, bias)
        if lora_A is not None:
            y = y + scale * F.linear(F.linear(x, lora_A), lora_B)
        ctx.save_for_backward(x, weight, lora_A, lora_B)
        ctx.scale = scale
        ctx.mod = mod
        ctx.has_bias = bias is not None and bias.requires_grad
        return y

    @staticmethod
    def backward(ctx, gy):  # type: ignore[override]
        x, W, A, Bm = ctx.saved_tensors
        mod = ctx.mod
        st: _State = mod._sbw_state
        cfg = st.cfg
        s = ctx.scale
        out_f, in_f = W.shape
        r = 0 if A is None else int(A.shape[0])
        g = gy.reshape(-1, out_f)
        xf = x.reshape(-1, in_f)
        N = g.shape[0]
        need_x = ctx.needs_input_grad[0]
        need_A = A is not None and ctx.needs_input_grad[3]
        need_B = Bm is not None and ctx.needs_input_grad[4]
        if ctx.needs_input_grad[1]:
            raise RuntimeError("sparse backward: the base weight must be frozen")
        norm2 = g.float().pow(2).sum(-1)
        row_dense = 2.0 * in_f * out_f + 4.0 * r * (in_f + out_f)
        if cfg.mode == "sketch":
            nz = int((norm2 > 0).sum())
            us, v = _svd_factors(mod, cfg.rank)
            rr = us.shape[1]
            dx = None
            if need_x:
                dx = (g @ us) @ v.T
                m = int(cfg.sketch)
                if m > 0:
                    om = torch.randn(in_f, m, device=g.device, dtype=g.dtype, generator=st.gen)
                    e_om = W @ om - us @ (v.T @ om)  # (out, m): the residual's sketch, fresh every call
                    dx = dx + ((g @ e_om) @ om.T) / m
                if A is not None:
                    dx = dx + s * ((g @ Bm) @ A)
            dA = dB = None
            if need_A or need_B:
                gb = g @ Bm
                if need_B:
                    dB = s * (g.T @ (xf @ A.T))
                if need_A:
                    dA = s * (gb.T @ xf)
            m = int(cfg.sketch)
            act_row = 2.0 * rr * (in_f + out_f) + 2.0 * m * (in_f + out_f) + 4.0 * r * (in_f + out_f) + 2.0 * out_f
            act_call = (2.0 * out_f * in_f * m + 2.0 * rr * m * (in_f + out_f)) if m > 0 else 0.0
            st.charge(st.dense_rows(N, nz) * row_dense, nz * act_row + act_call, float(nz), float(nz))
            return (dx.view_as(x) if dx is not None else None), None, None, dA, dB, None, None
        idx, w, nz = select_rows(norm2, cfg, st.gen)
        k = int(idx.numel())
        gS = g.index_select(0, idx)
        if w is not None:
            gS = gS * w.to(gS.dtype)[:, None]
        dx = dA = dB = None
        gb = None
        if need_x:
            dxS = gS @ W
            if A is not None:
                gb = gS @ Bm
                dxS = dxS + s * (gb @ A)
            dx = torch.zeros(N, in_f, device=g.device, dtype=g.dtype)
            dx.index_copy_(0, idx, dxS.to(dx.dtype))
            dx = dx.view_as(x)
        lora_rows = 0.0
        if (need_A or need_B) and cfg.lora_all_rows and cfg.mode != "sample":
            # adapter weight gradients from every nonzero row; the selected rows' share of them is already priced in
            # row_dense, so only the unselected nonzero rows add 4 r (in + out)
            nzi = torch.nonzero(norm2 > 0, as_tuple=True)[0]
            gN, xN = g.index_select(0, nzi), xf.index_select(0, nzi)
            if need_B:
                dB = s * (gN.T @ (xN @ A.T))
            if need_A:
                dA = s * ((gN @ Bm).T @ xN)
            lora_rows = float(max(0, nz - k))
        elif need_A or need_B:
            xS = xf.index_select(0, idx)
            if need_B:
                dB = s * (gS.T @ (xS @ A.T))
            if need_A:
                if gb is None:
                    gb = gS @ Bm
                dA = s * (gb.T @ xS)
        st.charge(st.dense_rows(N, nz) * row_dense,
                  k * row_dense + lora_rows * 4.0 * r * (in_f + out_f) + 2.0 * out_f * N, float(k), float(nz))
        return dx, None, None, dA, dB, None, None


def _sbw_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    st: _State = self._sbw_state
    if not (st.enabled and torch.is_grad_enabled() and (x.requires_grad or self.lora_A.requires_grad)):
        self._sbw_hook_adds = True
        return F.linear(x, self.weight, self.bias)
    self._sbw_hook_adds = False
    return _SparseLinearFn.apply(x, self.weight, self.bias, self.lora_A, self.lora_B, self._lora_scale, self)


def install_sparse_backward(model: Any, cfg: SparseBackwardConfig) -> List[nn.Linear]:
    """Route the backward of every LoRA-adapted linear in blocks [start_block, end_block) through the compressed VJP.
    Needs :func:`~olmo_core.nn.early_lora.install_early_lora` first (frozen base + adapters)."""
    cfg.validate()
    if not cfg.active:
        return []
    el = getattr(model, "_early_lora", None)
    if not el:
        raise RuntimeError("sparse backward needs LoRA adapters over a frozen base (install early LoRA first)")
    st = _State(model, cfg)
    mods = []
    for m in el["modules"]:
        if not cfg.start_block <= int(m._lora_block) < cfg.end_block:
            continue
        if m.weight.requires_grad:
            raise RuntimeError("sparse backward: adapted linear with a trainable base weight")
        m._sbw_state = st
        m._sbw_hook_adds = True
        m.forward = MethodType(_sbw_forward, m)
        mods.append(m)
    model._sparse_backward = st
    log.info("[sparse-bw] %s on %d adapted linears (blocks [%d, %d)): %s", cfg.mode, len(mods), cfg.start_block,
             min(cfg.end_block, len(model.blocks)), cfg)
    print(f"[sparse-bw] mode={cfg.mode} on {len(mods)} adapted linears: {cfg}", flush=True)
    return mods


def sparse_backward_metrics(model: Any, reset: bool = True) -> Dict[str, float]:
    st: Optional[_State] = getattr(model, "_sparse_backward", None)
    if st is None or st.calls == 0:
        return {}
    out = {
        "sparse_bw/row_frac": st.rows_kept / max(1.0, st.rows_nz),
        "sparse_bw/vjp_frac": st.flops_actual / max(1.0, st.flops_dense),
        "sparse_bw/saved_pflops_step": (st.flops_dense - st.flops_actual) / 1e15,
    }
    if reset:
        st.reset_stats()
    return out
