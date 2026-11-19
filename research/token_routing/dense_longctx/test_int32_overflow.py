"""Localize the >262k-token Gated DeltaNet breakdown to specific fla kernels, and check the int32-safe guard.

Each op runs once on the whole row and once on exact segments (state / window carried across segment boundaries).
Segmenting does not change the math, so wherever the two disagree at long T but agree at short T the full-row call
is wrong. The conv is also checked against torch's own (int64-indexed) depthwise conv1d.

    python test_int32_overflow.py --ops conv,chunk,onorm --lens 131072,262144,300000,524288,600000,1048576

Output: one JSON line per (op, width, T): max |diff| over the head and tail 4096 positions, NaN counts.
Part 2 (``--module``): whole olmo_core GatedDeltaNet (4B / 0.8B dims), forward (+ backward at short T) with the
``enable_int32_safe_long_seq`` guard off vs on, plus a short-T exactness check with the segment bound forced small.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F


def stats(a: torch.Tensor, b: torch.Tensor, w: int = 4096):
    a = a.float().reshape(a.shape[0], a.shape[1], -1)
    b = b.float().reshape(b.shape[0], b.shape[1], -1)
    head = (a[:, :w] - b[:, :w]).abs().max().item()
    tail = (a[:, -w:] - b[:, -w:]).abs().max().item()
    allm = (a - b).abs().amax(dim=(0, 2))  # per position
    bad = (allm > 1e-2 * (b.abs().amax() + 1e-6)).nonzero()
    first_bad = int(bad[0].item()) if bad.numel() else -1
    return dict(head=head, tail=tail, first_bad_pos=first_bad, nan_a=int(torch.isnan(a).sum().item()),
                nan_b=int(torch.isnan(b).sum().item()), scale=b.abs().amax().item())


def seg_conv(x, w, seg, W=4):
    from fla.modules.convolution import causal_conv1d
    outs, T = [], x.shape[1]
    for s0 in range(0, T, seg):
        a = max(0, s0 - (W - 1))
        y = causal_conv1d(x=x[:, a:min(s0 + seg, T)], weight=w, activation="silu")[0]
        outs.append(y[:, s0 - a:])
    return torch.cat(outs, 1)


def torch_conv(x, w, W=4):
    # depthwise causal conv + silu in torch (int64-safe), chunked over channels to bound memory
    outs = []
    for c0 in range(0, x.shape[2], 1024):
        xc = x[..., c0:c0 + 1024].transpose(1, 2).float()
        y = F.conv1d(F.pad(xc, (W - 1, 0)), w[c0:c0 + 1024].float().unsqueeze(1), groups=xc.shape[1])
        outs.append(F.silu(y).transpose(1, 2).to(x.dtype))
    return torch.cat(outs, 2)


def op_conv(T, C, dev):
    from fla.modules.convolution import causal_conv1d
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, T, C, device=dev, dtype=torch.bfloat16, generator=g)
    w = (torch.randn(C, 4, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    full = causal_conv1d(x=x, weight=w, activation="silu")[0]
    seg = seg_conv(x, w, 131072)
    ref = torch_conv(x, w)
    return dict(full_vs_seg=stats(full, seg), full_vs_torch=stats(full, ref), seg_vs_torch=stats(seg, ref))


def op_chunk(T, H, dev, seg=131072):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as cg
    g_ = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16, generator=g_)
    k = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16, generator=g_)
    v = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16, generator=g_)
    gl = -F.softplus(torch.randn(1, T, H, device=dev, generator=g_)) * 0.1
    beta = torch.rand(1, T, H, device=dev, generator=g_).to(torch.bfloat16)
    full, _ = cg(q=q, k=k, v=v, g=gl, beta=beta, use_qk_l2norm_in_kernel=True)
    outs, st = [], None
    for s0 in range(0, T, seg):
        sl = slice(s0, min(s0 + seg, T))
        o, st = cg(q=q[:, sl], k=k[:, sl], v=v[:, sl], g=gl[:, sl], beta=beta[:, sl], initial_state=st,
                   output_final_state=True, use_qk_l2norm_in_kernel=True)
        outs.append(o)
    return dict(full_vs_seg=stats(full, torch.cat(outs, 1)))


def op_chunkbwd(T, H, dev, seg=65536):
    """Backward of the chunk kernel: grads of q, k, v, g, beta for a fixed random upstream gradient."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as cg
    g_ = torch.Generator(device=dev).manual_seed(0)
    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch.bfloat16, generator=g_)
    q, k, v = mk(1, T, H, 128), mk(1, T, H, 128), mk(1, T, H, 128)
    gl = (-F.softplus(torch.randn(1, T, H, device=dev, generator=g_)) * 0.1)
    beta = torch.rand(1, T, H, device=dev, generator=g_).to(torch.bfloat16)
    do = mk(1, T, H, 128)
    leaves = [t.requires_grad_(True) for t in (q, k, v, gl, beta)]

    def grads(segmented):
        for t in leaves:
            t.grad = None
        if segmented:
            outs, st = [], None
            for s0 in range(0, T, seg):
                sl = slice(s0, min(s0 + seg, T))
                o, st = cg(q=q[:, sl], k=k[:, sl], v=v[:, sl], g=gl[:, sl], beta=beta[:, sl], initial_state=st,
                           output_final_state=True, use_qk_l2norm_in_kernel=True)
                outs.append(o)
            o = torch.cat(outs, 1)
        else:
            o, _ = cg(q=q, k=k, v=v, g=gl, beta=beta, use_qk_l2norm_in_kernel=True)
        o.backward(do)
        return [t.grad.detach().clone() for t in leaves]

    gf = grads(False)
    gs = grads(True)
    return {n: stats(a_.unsqueeze(-1) if a_.dim() == 3 else a_, b_.unsqueeze(-1) if b_.dim() == 3 else b_)
            for n, a_, b_ in zip(["dq", "dk", "dv", "dg", "dbeta"], gf, gs)}


def op_onorm(T, H, dev, seg=131072):
    from fla.modules import FusedRMSNormGated
    n = FusedRMSNormGated(128, eps=1e-6).to(dev)
    g_ = torch.Generator(device=dev).manual_seed(0)
    o = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16, generator=g_)
    gt = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16, generator=g_)
    full = n(o, gt)
    seg_ = torch.cat([n(o[:, s:s + seg], gt[:, s:s + seg]) for s in range(0, T, seg)], 1)
    return dict(full_vs_seg=stats(full, seg_))


def module_test(scale, lens, dev):
    import olmo_core.nn.convolution as conv_mod
    from olmo_core.nn.attention.recurrent import GatedDeltaNet
    from olmo_core.nn.ffn_chunk import enable_int32_safe_long_seq
    from olmo_core.nn.transformer.init import InitMethod
    dims = dict(d_model=2560, n_heads=16, n_v_heads=32) if scale == "4b" else dict(d_model=1024, n_heads=16, n_v_heads=16)
    torch.manual_seed(0)
    m = GatedDeltaNet(**dims, head_dim=128, expand_v=1.0, allow_neg_eigval=False, conv_size=4, norm_eps=1e-6,
                      init_device="cuda")
    m.init_weights(init_method=InitMethod.normal, d_model=dims["d_model"], block_idx=0, num_blocks=32)
    m = m.to(torch.bfloat16).train()

    def run(x, safe, grad=False):
        m._int32_safe = safe
        for c in m._convs():
            c._int32_safe = safe
        if grad:
            x = x.detach().requires_grad_(True)
            y = m(x)
            (y.float().square().mean()).backward()
            gr = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
            m.zero_grad(set_to_none=True)
            return y.detach(), x.grad.detach(), gr
        with torch.no_grad():
            return m(x)

    # (a) exactness of the segmented path at short T: force tiny segment bounds
    old = conv_mod._INT32_LIMIT
    T = 16384
    x = torch.randn(1, T, dims["d_model"], device=dev, dtype=torch.bfloat16)
    y0, gx0, g0 = run(x, False, grad=True)
    conv_mod._INT32_LIMIT = 4096 * 4096  # -> 4096-token segments for 4096-wide rows (1024..8192 for the others)
    y1, gx1, g1 = run(x, True, grad=True)
    conv_mod._INT32_LIMIT = old
    gerr = max(((g0[n] - g1[n]).float().norm() / (g0[n].float().norm() + 1e-12)).item() for n in g0)
    print(json.dumps(dict(test="module_exact_short", scale=scale, T=T, out=stats(y1, y0),
                          dx_rel=((gx0 - gx1).float().norm() / gx0.float().norm()).item(), max_param_grad_rel=gerr)),
          flush=True)
    del y0, y1, gx0, gx1, g0, g1
    # (b) long T forward, guard off vs on
    for T in lens:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1, T, dims["d_model"], device=dev, dtype=torch.bfloat16)
        t0 = time.time()
        y_off = run(x, False)
        y_on = run(x, True)
        torch.cuda.synchronize()
        print(json.dumps(dict(test="module_long", scale=scale, T=T, off_vs_on=stats(y_off, y_on),
                              s=round(time.time() - t0, 1),
                              peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True)
        del x, y_off, y_on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="conv,chunk,onorm")
    ap.add_argument("--lens", default="131072,262144,300000,524288,600000")
    ap.add_argument("--conv-widths", default="2048,4096,8192")
    ap.add_argument("--heads", default="16,32")
    ap.add_argument("--module", default="", help="4b,0.8b")
    ap.add_argument("--module-lens", default="300000,600000")
    a = ap.parse_args()
    dev = torch.device("cuda")
    lens = [int(x) for x in a.lens.split(",") if x]
    for op in [o for o in a.ops.split(",") if o and o != "none"]:
        for T in lens:
            widths = [int(c) for c in a.conv_widths.split(",")] if op == "conv" else [int(h) for h in a.heads.split(",")]
            for wd in widths:
                torch.cuda.empty_cache()
                t0 = time.time()
                try:
                    r = {"conv": op_conv, "chunk": op_chunk, "chunkbwd": op_chunkbwd, "onorm": op_onorm}[op](T, wd, dev)
                    err = None
                except Exception as e:  # noqa: BLE001
                    r, err = {}, f"{type(e).__name__}: {str(e)[:200]}"
                row_width = wd if op == "conv" else wd * 128
                print(json.dumps(dict(op=op, width=row_width, T=T, T_x_width_over_2p31=round(T * row_width / 2**31, 3),
                                      s=round(time.time() - t0, 1), err=err, **r)), flush=True)
                if err and "illegal" in err.lower():
                    raise SystemExit("CUDA context lost after an illegal memory access; rerun the remaining cases")
    for sc in [s for s in a.module.split(",") if s]:
        module_test(sc, [int(x) for x in a.module_lens.split(",")], dev)


if __name__ == "__main__":
    main()
