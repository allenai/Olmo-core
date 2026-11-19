"""Numerical check of the packed-row (cu_doc_lens) int32 guard in GatedDeltaNet (1 GPU, seconds).

The bound is lowered (_int32_safe_limit) so a short packed stream crosses it: documents below the bound go through
per-group kernel calls, one document longer than the bound goes through state-carrying segments. Reference = the same
module and input with the guard off (plain varlen kernel; far below the real int32 bound, so it is correct).
Also checks the gradient (backward through both paths)."""
import torch

import olmo_core.nn.attention.recurrent as R
from olmo_core.nn.attention.recurrent import GatedDeltaNet

CALLS = []
_orig = R.dispatch_chunk_gated_delta_rule


def _counted(**kw):
    CALLS.append((kw["q"].shape[1], kw.get("cu_seqlens") is not None, kw.get("initial_state") is not None))
    return _orig(**kw)


R.dispatch_chunk_gated_delta_rule = _counted

torch.manual_seed(0)
dev = "cuda"
m = GatedDeltaNet(d_model=512, n_heads=4, n_v_heads=8, head_dim=64).to(dev).to(torch.bfloat16)
docs = [700, 900, 300, 5000, 1200, 400, 64, 2047]  # one doc (5000) crosses the lowered bound of 2048 tokens
cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(docs), 0).tolist()), dtype=torch.int32, device=dev)
T = int(cu[-1])
x = torch.randn(1, T, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)


def run(guard):
    m._int32_safe = guard
    for c in m._convs():
        c._int32_safe = guard
    width = max(m.n_heads * m.head_k_dim, m.n_v_heads * m.head_v_dim) if hasattr(m, "head_k_dim") else 512
    m._int32_safe_limit = 2048 * width if guard else 2 ** 30
    xi = x.detach().clone().requires_grad_(True)
    CALLS.clear()
    y = m(xi, cu_doc_lens=cu)
    print(f"guard={guard}: kernel calls (tokens, varlen, carried state): {CALLS}")
    y.float().pow(2).mean().backward()
    return y.detach().float(), xi.grad.detach().float()


y0, g0 = run(False)
y1, g1 = run(True)
print(f"T={T} docs={docs} lowered bound=2048 tokens")
print(f"forward  max|diff| {float((y1 - y0).abs().max()):.3e}  (max|y| {float(y0.abs().max()):.3e})")
print(f"grad x   max|diff| {float((g1 - g0).abs().max()):.3e}  (max|g| {float(g0.abs().max()):.3e})")
for i, (a, b) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
    print(f"  doc {i} len {b - a}: max|diff| {float((y1[:, a:b] - y0[:, a:b]).abs().max()):.3e}")
# same documents run one at a time, unpacked, plain kernel
outs = []
m._int32_safe = False
for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
    outs.append(m(x[:, a:b].detach()).float())
yu = torch.cat(outs, 1)
print(f"guard vs per-document unpacked: max|diff| {float((y1 - yu).abs().max()):.3e}")
