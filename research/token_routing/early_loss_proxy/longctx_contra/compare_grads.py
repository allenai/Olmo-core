"""Compare dumped step-1 gradients (dense_probe --probe-grads) between configs: per-block cosine and relative norm
difference of the LoRA gradients, plus the global numbers.   python compare_grads.py REF.pt OTHER.pt [OTHER2.pt ...]"""
import re
import sys

import torch


def load(p):
    return torch.load(p, map_location="cpu")


ref = load(sys.argv[1])
for other_p in sys.argv[2:]:
    o = load(other_p)
    keys = sorted(set(ref) & set(o))
    by = {}
    for k in keys:
        m = re.search(r"blocks\.(\d+)\.", k)
        b = int(m.group(1)) if m else -1
        a, c = ref[k].flatten(), o[k].flatten()
        d = by.setdefault(b, [0.0, 0.0, 0.0, 0.0])
        d[0] += float(a @ c); d[1] += float(a @ a); d[2] += float(c @ c); d[3] += float((a - c) @ (a - c))
    tot = [sum(v[i] for v in by.values()) for i in range(4)]
    cosg = tot[0] / (tot[1] * tot[2]) ** 0.5
    print(f"== {sys.argv[1].split('/')[-1]} vs {other_p.split('/')[-1]}: {len(keys)} tensors (ref {len(ref)}, other {len(o)}); "
          f"global cos {cosg:.5f}, |ref| {tot[1] ** 0.5:.4e}, |other| {tot[2] ** 0.5:.4e}, rel diff {(tot[3] / tot[1]) ** 0.5:.4f}")
    worst = sorted(by.items(), key=lambda kv: kv[1][0] / max(1e-30, (kv[1][1] * kv[1][2]) ** 0.5))[:6]
    print("   per-block cos (block: cos, rel diff): " + "  ".join(
        f"{b}: {v[0] / max(1e-30, (v[1] * v[2]) ** 0.5):.4f}/{(v[3] / max(1e-30, v[1])) ** 0.5:.3f}" for b, v in sorted(by.items())))
    print("   worst blocks: " + ", ".join(f"{b} ({v[0] / max(1e-30, (v[1] * v[2]) ** 0.5):.4f})" for b, v in worst))
