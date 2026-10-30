"""2-rank CPU (gloo) check of merge_and_reinit_lora on FSDP2-style Shard(0) DTensor parameters.

    python -m torch.distributed.run --nproc_per_node=2 research/token_routing/decay_fix/check_relora_dtensor.py

Checks, on adapters whose out dim does not divide evenly across ranks: the merged base equals W + scale * B A computed
densely, B is zero afterwards, each rank re-drew its own rows of A (the two halves differ), the optimizer state of the
adapters is zeroed, and the returned FLOPs sum to 2 out r in per adapter across ranks.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))


def main():
    dist.init_process_group("gloo")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from olmo_core.nn.early_lora import install_early_lora, merge_and_reinit_lora

    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    torch.manual_seed(0)

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(10, 7, bias=False)
            self.b = nn.Linear(7, 5, bias=False)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(3, 10)
            self.blocks = nn.ModuleDict({"0": Blk()})

    m = M()
    mods = install_early_lora(m, 1, 4, scale=2.0, check_calls=False)
    with torch.no_grad():
        for mod in mods:
            mod.lora_B.normal_()
    expect = {i: (mod.weight + 2.0 * mod.lora_B @ mod.lora_A).detach().clone() for i, mod in enumerate(mods)}
    for mod in mods:
        for n in ("weight", "lora_A", "lora_B"):
            t = getattr(mod, n)
            setattr(mod, n, nn.Parameter(distribute_tensor(t.detach(), mesh, [Shard(0)]),
                                         requires_grad=t.requires_grad))
    params = [p for mod in mods for p in (mod.lora_A, mod.lora_B)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    for p in params:
        opt.state[p] = {"step": torch.tensor(5.0), "exp_avg": torch.ones_like(p), "exp_avg_sq": torch.ones_like(p)}
    out = merge_and_reinit_lora(m, opt, seed=3, restart=1)
    fl = torch.tensor([out["flops"]], dtype=torch.float64)
    dist.all_reduce(fl)
    want_fl = sum(2.0 * mod.out_features * 4 * mod.in_features for mod in mods)
    assert abs(float(fl) - want_fl) < 1e-6, (float(fl), want_fl)
    for i, mod in enumerate(mods):
        W = mod.weight.full_tensor()
        assert torch.allclose(W, expect[i], atol=1e-5), (i, (W - expect[i]).abs().max())
        assert float(mod.lora_B.full_tensor().abs().sum()) == 0.0
        A = mod.lora_A.full_tensor()
        assert not torch.allclose(A[:2], A[2:]), "rank shards of A are copies"
        assert float(A.abs().max()) <= 1.0 / mod.in_features ** 0.5 + 1e-6
    for p in params:
        st = opt.state[p]
        assert float(st["step"]) == 0.0
        assert float(st["exp_avg"].full_tensor().abs().sum()) == 0.0
    if rank == 0:
        print(f"RELORA_DTENSOR_OK rel_delta={out['rel_delta']:.3e} flops={float(fl):.0f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
