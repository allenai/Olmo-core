"""Streamed causal attention with per-edge gradient accounting, edge masking (MASK) and backward-only edge
dropping (DETACH). Pure torch, query-chunked, nothing L^2 is ever stored.

For one attention layer, head h, query i, key j <= i (s = pre-softmax logit, a = softmax(s), o_i = sum_j a_ij v_j,
g_i = dL/do_i at the core attention output, i.e. before the output gate and W_o):

    u_ij      = <g_i, v_j>
    D_i       = sum_j a_ij u_ij  (= <g_i, o_i>)
    dL/ds_ij  = a_ij (u_ij - D_i)                    -> (a) logit-gradient mass |dL/ds_ij|
    a_ij u_ij = first-order effect of edge (i, j)    -> (b) value-path mass a_ij |<g_i, v_j>|
    a_ij                                              -> (c) attention mass
(<g_i, W_o v_j> with the gate is the same number: the gate and W_o act linearly on o_i after the mixing.)

Modes (``EdgeState.mode``):
* ``full``   ordinary causal attention; the backward is the exact gradient.
* ``mask``   kept edges only, in forward and backward (the softmax renormalises over kept keys).
* ``detach`` forward = full attention; the backward keeps only the kept edges' terms of the exact gradient:
             dS_ij -> dS_ij * keep_ij (with the full D_i, as a block-sparse flash backward using the saved
             full-forward lse / D would compute) and dV_j = sum_i a_ij keep_ij g_i. With every edge kept this IS
             ``full`` (the same code path multiplies by ones), and summed over edges it is an exact decomposition of
             the q/k/v gradient into per-edge terms.

A pattern decides keep_ij from token segments (documents, preamble, question, answer) and token rules; see
:class:`Pattern`. GQA: query head h uses KV head h // (H / Hkv) (the repeat_interleave convention of the model).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

LOG_BINS = torch.arange(-14.0, 2.0001, 0.1)  # log10 bins for raw-edge concentration histograms
NDIST = 20  # distance bins: floor(log2(i - j + 1)), 0..19 (>= 2^19 clamps)


# ------------------------------------------------------------------------------------------------ patterns
class Pattern:
    """keep_ij = causal & (j == i | keyall_j | rowall_i | K[seg_i, seg_j] | 0 <= i - j < window).

    ``K`` is (nseg, nseg) bool on the device, or a dict {layer: K} for per-layer patterns; ``None`` = no block term.
    """

    def __init__(self, name: str, seg: torch.Tensor, K=None, rowall: Optional[torch.Tensor] = None,
                 keyall: Optional[torch.Tensor] = None, window: int = 0):
        self.name, self.seg, self.K, self.rowall, self.keyall, self.window = name, seg, K, rowall, keyall, int(window)
        self._frac: Dict[int, float] = {}

    def k_for(self, layer: int):
        if isinstance(self.K, dict):
            return self.K[layer]
        return self.K

    def keep(self, i0: int, i1: int, layer: int) -> torch.Tensor:
        dev = self.seg.device
        qi = torch.arange(i0, i1, device=dev)
        kj = torch.arange(i1, device=dev)
        diff = qi[:, None] - kj[None, :]
        causal = diff >= 0
        m = diff == 0
        K = self.k_for(layer)
        if K is not None:
            m = m | K[self.seg[i0:i1]][:, self.seg[:i1]]
        if self.keyall is not None:
            m = m | self.keyall[None, :i1]
        if self.rowall is not None:
            m = m | self.rowall[i0:i1, None]
        if self.window > 0:
            m = m | (diff < self.window)
        return m & causal

    def kept_fraction(self, T: int, layers: List[int], chunk: int = 2048) -> Dict[int, float]:
        """Kept causal edges / all causal edges, per layer (identical across heads)."""
        tot = T * (T + 1) / 2
        out = {}
        for L in layers:
            if not isinstance(self.K, dict) and out:
                out[L] = next(iter(out.values()))
                continue
            n = 0
            for i0 in range(0, T, chunk):
                n += int(self.keep(i0, min(T, i0 + chunk), L).sum())
            out[L] = n / tot
        return out


# ------------------------------------------------------------------------------------------------ stats
@dataclass
class EdgeStats:
    """Per-layer accumulators for one row (filled in the backward of a ``full`` pass)."""

    seg: torch.Tensor  # (T,) long segment id per token
    nseg: int
    per_head: bool = True
    raw_hist: bool = True
    blocks: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)  # layer -> qty -> (H?, nseg, nseg)
    dist: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)  # layer -> qty -> (H, NDIST)
    hist: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)  # layer -> qty -> (2, nbins) cnt/mass
    dist_cnt: Optional[torch.Tensor] = None  # (NDIST,) edges per distance bin (per head)

    QTY = ("logit", "value", "attn")

    def _bounds(self, dev):
        if not hasattr(self, "_st"):
            T = self.seg.numel()
            st = torch.zeros(self.nseg, dtype=torch.long, device=dev)
            en = torch.zeros(self.nseg, dtype=torch.long, device=dev)
            pos = torch.arange(T, device=dev)
            st.scatter_reduce_(0, self.seg, pos, reduce="amin", include_self=False)
            en.scatter_reduce_(0, self.seg, pos + 1, reduce="amax", include_self=False)
            self._st, self._en = st, en
        return self._st, self._en

    def add(self, layer: int, i0: int, i1: int, mats: Dict[str, torch.Tensor], causal: torch.Tensor,
            dbin=None) -> None:
        """Key-side sums by one fp64 cumsum over keys per quantity: segment blocks = differences of the cumsum at
        segment bounds, distance bins = differences at i - 2^b + 1 (segments are contiguous)."""
        H = next(iter(mats.values())).shape[0]
        dev = causal.device
        cq = i1 - i0
        sq = self.seg[i0:i1]
        st, en = self._bounds(dev)
        kst, ken = st.clamp(max=i1), en.clamp(max=i1)  # (nseg,) into the padded cumsum
        qi = torch.arange(i0, i1, device=dev)[:, None]
        b = torch.arange(NDIST, device=dev)[None, :]
        hi = qi - (2 ** b) + 1  # last key (inclusive) of bin b
        lo = qi - (2 ** (b + 1)) + 2  # first key of bin b
        lo = torch.where(b == NDIST - 1, torch.zeros_like(lo), lo)
        hi_i = (hi + 1).clamp(0, i1)
        lo_i = lo.clamp(0, i1)
        lo_i = torch.minimum(lo_i, hi_i)
        if layer not in self.blocks:
            shape = (H, self.nseg, self.nseg) if self.per_head else (self.nseg, self.nseg)
            self.blocks[layer] = {q: torch.zeros(shape, device=dev, dtype=torch.float64) for q in self.QTY}
            self.dist[layer] = {q: torch.zeros(H, NDIST, device=dev, dtype=torch.float64) for q in self.QTY}
            if self.raw_hist:
                nb = LOG_BINS.numel() + 1
                self.hist[layer] = {q: torch.zeros(2, nb, device=dev, dtype=torch.float64) for q in self.QTY}
        if self.dist_cnt is None:
            self.dist_cnt = torch.zeros(NDIST, device=dev, dtype=torch.float64)
        if layer == min(self.blocks):
            self.dist_cnt += (hi_i - lo_i).sum(0).double()
        for q, X in mats.items():  # X: (H, Cq, i1) fp32, zero off the causal part
            Cp = torch.nn.functional.pad(torch.cumsum(X, -1, dtype=torch.float64), (1, 0))  # (H, Cq, i1 + 1)
            Y = Cp[:, :, ken] - Cp[:, :, kst]  # (H, Cq, nseg)
            if self.per_head:
                self.blocks[layer][q].index_add_(1, sq, Y)
            else:
                self.blocks[layer][q].index_add_(0, sq, Y.sum(0))
            idx_hi = hi_i[None].expand(H, -1, -1)
            idx_lo = lo_i[None].expand(H, -1, -1)
            self.dist[layer][q] += (Cp.gather(2, idx_hi) - Cp.gather(2, idx_lo)).sum(1)
            del Cp, Y
            if self.raw_hist:
                v = X[:, causal].flatten()
                bb = torch.bucketize(torch.log10(v.clamp_min(1e-30)), LOG_BINS.to(dev))
                h = self.hist[layer][q]
                h[0] += torch.bincount(bb, minlength=h.shape[1]).double()
                h[1] += torch.bincount(bb, weights=v.double(), minlength=h.shape[1])


# ------------------------------------------------------------------------------------------------ state
class EdgeState:
    """Global switch read by every patched attention layer during one pass."""

    def __init__(self):
        self.mode = "full"
        self.pattern: Optional[Pattern] = None  # used in mask / detach
        self.stats: Optional[EdgeStats] = None  # collected in the backward when set (mode full)
        self.chunk_elems = 2 ** 27  # H * Cq * T fp32 elements per chunk matrix (~512 MB)
        self.calls = 0
        self.bwd_calls = 0

    def chunk_for(self, H: int, T: int) -> int:
        c = max(16, self.chunk_elems // max(1, H * T))
        return int(min(T, 1 << int(math.floor(math.log2(c)))))


STATE = EdgeState()


class EdgeAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, layer: int, scale: float, state: EdgeState):
        # q (1, T, H, D), k/v (1, T, Hk, D)
        B, T, H, D = q.shape
        Hk = k.shape[2]
        rep = H // Hk
        dev = q.device
        mode = state.mode
        pat = state.pattern if mode == "mask" else None
        qf = q[0].permute(1, 0, 2).reshape(Hk, rep, T, D).float()
        kf = k[0].permute(1, 0, 2).float()  # (Hk, T, D)
        vf = v[0].permute(1, 0, 2).float()
        out = torch.empty(Hk, rep, T, D, device=dev, dtype=torch.float32)
        C = state.chunk_for(H, T)
        for i0 in range(0, T, C):
            i1 = min(T, i0 + C)
            cq = i1 - i0
            s = torch.matmul(qf[:, :, i0:i1].reshape(Hk, rep * cq, D), kf[:, :i1].transpose(1, 2)).mul_(scale)
            s = s.view(Hk, rep, cq, i1)
            if pat is not None:
                keep = pat.keep(i0, i1, layer)
            else:
                keep = torch.arange(i1, device=dev)[None, :] <= torch.arange(i0, i1, device=dev)[:, None]
            s.masked_fill_(~keep, float("-inf"))
            p = torch.softmax(s, dim=-1)
            out[:, :, i0:i1] = torch.matmul(p.view(Hk, rep * cq, i1), vf[:, :i1]).view(Hk, rep, cq, D)
            del s, p
        ctx.save_for_backward(q, k, v)
        ctx.layer, ctx.scale, ctx.state = layer, scale, state
        ctx.mode, ctx.pat = mode, pat  # forward-time pattern (mask)
        state.calls += 1
        return out.view(H, T, D).permute(1, 0, 2).unsqueeze(0).to(q.dtype).contiguous()

    @staticmethod
    def backward(ctx, g):
        q, k, v = ctx.saved_tensors
        state: EdgeState = ctx.state
        layer, scale = ctx.layer, ctx.scale
        B, T, H, D = q.shape
        Hk = k.shape[2]
        rep = H // Hk
        dev = q.device
        fwd_pat = ctx.pat
        mode = ctx.mode
        # the backward pattern is read NOW (several backwards over one detach forward use different patterns)
        bwd_pat = state.pattern if mode == "detach" else None
        stats = state.stats if mode == "full" or (mode == "detach" and bwd_pat is None) else None
        qf = q[0].permute(1, 0, 2).reshape(Hk, rep, T, D).float()
        kf = k[0].permute(1, 0, 2).float()
        vf = v[0].permute(1, 0, 2).float()
        gf = g[0].permute(1, 0, 2).reshape(Hk, rep, T, D).float()
        dq = torch.zeros(Hk, rep, T, D, device=dev, dtype=torch.float32)
        dk = torch.zeros(Hk, T, D, device=dev, dtype=torch.float32)
        dv = torch.zeros(Hk, T, D, device=dev, dtype=torch.float32)
        C = state.chunk_for(H, T)
        for i0 in range(0, T, C):
            i1 = min(T, i0 + C)
            cq = i1 - i0
            qc = qf[:, :, i0:i1].reshape(Hk, rep * cq, D)
            gc = gf[:, :, i0:i1].reshape(Hk, rep * cq, D)
            causal = torch.arange(i1, device=dev)[None, :] <= torch.arange(i0, i1, device=dev)[:, None]
            fkeep = fwd_pat.keep(i0, i1, layer) if fwd_pat is not None else causal
            s = torch.matmul(qc, kf[:, :i1].transpose(1, 2)).mul_(scale).view(Hk, rep, cq, i1)
            s.masked_fill_(~fkeep, float("-inf"))
            p = torch.softmax(s, dim=-1)
            del s
            u = torch.matmul(gc, vf[:, :i1].transpose(1, 2)).view(Hk, rep, cq, i1)
            Dl = (p * u).sum(-1, keepdim=True)
            ds = p * (u - Dl)
            if stats is not None:
                mats = {"logit": ds.abs().view(H, cq, i1), "value": (p * u.abs()).view(H, cq, i1),
                        "attn": p.view(H, cq, i1)}
                stats.add(layer, i0, i1, mats, causal)
                del mats
            del u
            if bwd_pat is not None:
                bk = bwd_pat.keep(i0, i1, layer).to(ds.dtype)
                ds = ds * bk
                pk = p * bk
                del bk
            else:
                pk = p
            ds2 = ds.view(Hk, rep * cq, i1)
            dq[:, :, i0:i1] = torch.matmul(ds2, kf[:, :i1]).mul_(scale).view(Hk, rep, cq, D)
            dk[:, :i1] += torch.matmul(ds2.transpose(1, 2), qc).mul_(scale)
            dv[:, :i1] += torch.matmul(pk.view(Hk, rep * cq, i1).transpose(1, 2), gc)
            del ds, ds2, p, pk
        state.bwd_calls += 1
        dq = dq.view(H, T, D).permute(1, 0, 2).unsqueeze(0).to(q.dtype).contiguous()
        dk = dk.permute(1, 0, 2).unsqueeze(0).to(k.dtype).contiguous()
        dv = dv.permute(1, 0, 2).unsqueeze(0).to(v.dtype).contiguous()
        return dq, dk, dv, None, None, None


def edge_attention(q, k, v, layer: int, scale: Optional[float] = None, state: EdgeState = STATE):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return EdgeAttnFn.apply(q, k, v, layer, float(scale), state)


def patch_model(model, state: EdgeState = STATE) -> List[int]:
    """Route every softmax-attention layer's core attention through :func:`edge_attention`. Returns the block
    indices patched (GDN blocks are untouched)."""
    from olmo_core.nn.attention import Attention

    patched = []
    for key, block in model.blocks.items():
        for m in block.modules():
            if isinstance(m, Attention):
                scale = getattr(m.backend, "scale", None) if getattr(m, "backend", None) is not None else None
                L = int(key)

                def sdpa(q, k, v, *a, _L=L, _scale=scale, **kw):
                    for name in ("cu_doc_lens", "cu_doc_lens_q", "cu_doc_lens_k", "local_k_slice"):
                        if kw.get(name) is not None:
                            raise RuntimeError(f"edge_attention: unsupported {name}")
                    return edge_attention(q, k, v, _L, _scale, state)

                m.sdpa = sdpa
                if getattr(m, "sinks", None) is not None:
                    raise RuntimeError("edge_attention: attention sinks not supported")
                patched.append(L)
    return patched
