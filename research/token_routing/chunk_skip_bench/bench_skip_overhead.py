"""Wall-clock overhead of per-layer token skipping, token-level vs chunk-level, on real Qwen3.5-4B blocks.

For one GDN block (block 0) and one full-attention block (block 3) of a Qwen3.5-4B (bf16, 1 GPU, random init: timing
only), times forward + backward of:

  full     the block on all T tokens
  ideal    the block on k*T tokens, no gather / scatter (the best case: what dropping a token for all later layers
           costs per layer, since its single gather is amortized over every later layer)
  tok      per-layer skipping: gather a random k*T tokens, run the block, scatter the outputs back into the full-length
           residual stream (skipped tokens pass through), as mixture-of-depths does at every routed layer
  chunkC   the same with whole chunks of C tokens chosen at random (C = 100, 128)

Reports ms per call and overhead = (gathered - ideal) / full, i.e. the share of the full-layer time lost to doing the
skip per layer instead of once.

    python research/token_routing/chunk_skip_bench/bench_skip_overhead.py --out /results/x.json
"""
from __future__ import annotations

import argparse
import json

import torch

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig

D_VOCAB = 151936


def timeit(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def make_idx(T: int, k: float, chunk: int, g: torch.Generator) -> torch.Tensor:
    if chunk <= 1:
        n = int(round(k * T))
        return torch.randperm(T, generator=g)[:n].sort().values
    n_chunks = T // chunk
    keep = torch.randperm(n_chunks, generator=g)[: max(1, int(round(k * n_chunks)))].sort().values
    return (keep[:, None] * chunk + torch.arange(chunk)[None, :]).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_skip_overhead.json")
    ap.add_argument("--lengths", default="8192,32768")
    ap.add_argument("--keeps", default="0.25,0.5")
    ap.add_argument("--chunks", default="100,128")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    dev = "cuda"
    cfg = TransformerConfig.qwen3_5_4B(vocab_size=D_VOCAB, dtype=DType.bfloat16, n_layers=4)
    model = cfg.build(init_device=dev)
    model.init_weights(max_seq_len=65536, device=torch.device(dev))
    model.to(torch.bfloat16)
    blocks = list(model.blocks.values())
    kinds = {"gdn (block 0)": blocks[0], "full attention (block 3)": blocks[3]}
    g = torch.Generator().manual_seed(0)
    d = cfg.d_model
    rows = []
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    for kname, blk in kinds.items():
        for T in [int(x) for x in a.lengths.split(",")]:
            h = torch.randn(1, T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)

            def run_full():
                y = blk(h)
                y.float().square().mean().backward()

            t_full = timeit(run_full, a.iters)
            for k in [float(x) for x in a.keeps.split(",")]:
                n = int(round(k * T))
                hs = torch.randn(1, n, d, device=dev, dtype=torch.bfloat16, requires_grad=True)

                def run_ideal():
                    y = blk(hs)
                    y.float().square().mean().backward()

                t_ideal = timeit(run_ideal, a.iters)
                res = {"block": kname, "T": T, "keep": k, "full_ms": t_full, "ideal_ms": t_ideal}
                for chunk in [1] + [int(c) for c in a.chunks.split(",")]:
                    idx = make_idx(T, k, chunk, g).to(dev)

                    def run_gather():
                        x = h.index_select(1, idx)
                        y = blk(x)
                        out = h.index_copy(1, idx, y)
                        out.float().square().mean().backward()

                    t = timeit(run_gather, a.iters)
                    name = "tok" if chunk == 1 else f"chunk{chunk}"
                    res[f"{name}_ms"] = t
                    res[f"{name}_overhead"] = (t - t_ideal) / t_full
                rows.append(res)
                print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res.items()}), flush=True)
                del hs
            del h
            torch.cuda.empty_cache()
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
