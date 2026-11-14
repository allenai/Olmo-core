"""
A muP coordinate check.

Trains small Llama-like models at several widths for a few steps on a fixed batch, then reports
how much each tracked activation moved per coordinate (the RMS of its change). Under muP
(:data:`~olmo_core.nn.transformer.init.InitMethod.nanochat` with
:class:`~olmo_core.optim.MuPConfig`) these changes should stay roughly constant as the width
grows. Under standard parameterization (std 0.02 and one learning rate) they grow with width.

The head dimension is fixed, so the number of heads grows with the width. The "width exponent"
column is the slope of log(change) against log(width): about 0 means width-independent.

Run on one GPU::

    python src/examples/mup/coord_check.py
"""

import argparse
import math
from typing import Dict, List

import torch
import torch.nn.functional as F

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import InitMethod, TransformerConfig
from olmo_core.optim import AdamWConfig, MuPConfig
from olmo_core.utils import seed_all

HEAD_DIM = 64
TRACKED = ("embedding", "middle block", "logits")


def activation_changes(width: int, *, mup: bool, args: argparse.Namespace) -> Dict[str, float]:
    """
    Train one model and return the per-coordinate RMS change of each tracked activation.
    """
    seed_all(0)
    device = torch.device(args.device)
    model = TransformerConfig.llama_like(
        d_model=width,
        vocab_size=args.vocab_size,
        n_layers=args.n_layers,
        n_heads=width // HEAD_DIM,
        feed_forward=FeedForwardConfig(hidden_size=4 * width, bias=False),
        attn_backend=AttentionBackendName.torch,
        init_method=InitMethod.nanochat if mup else InitMethod.normal,
    ).build(init_device="meta")
    model.init_weights(device=device)
    optim = AdamWConfig(
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        mup=MuPConfig(base_width=args.base_width) if mup else None,
    ).build(model)

    generator = torch.Generator(device).manual_seed(0)
    input_ids = torch.randint(
        args.vocab_size, (args.batch_size, args.seq_len), device=device, generator=generator
    )

    captured: Dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(module, inputs, output):
            del module, inputs
            captured[name] = output.detach().float()

        return hook

    hooks = [
        model.embeddings.register_forward_hook(capture("embedding")),
        model.blocks[str(args.n_layers // 2)].register_forward_hook(capture("middle block")),
    ]

    def snapshot() -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = model(input_ids).float()
        return {**captured, "logits": logits}

    initial = snapshot()
    for _ in range(args.steps):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1)
        )
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    final = snapshot()
    for hook in hooks:
        hook.remove()

    return {name: (final[name] - initial[name]).pow(2).mean().sqrt().item() for name in TRACKED}


def width_exponent(widths: List[int], values: List[float]) -> float:
    """
    The least-squares slope of log(value) against log(width).
    """
    xs = [math.log(w) for w in widths]
    ys = [math.log(v) for v in values]
    x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum(
        (x - x_mean) ** 2 for x in xs
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--base-width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    header = f"{'':16s}" + "".join(f"{f'w={w}':>11s}" for w in args.widths) + "  width exponent"
    for name, mup in (("standard", False), ("muP", True)):
        results = [activation_changes(w, mup=mup, args=args) for w in args.widths]
        print(f"\n{name} parameterization: per-coordinate RMS change after {args.steps} steps")
        print(header)
        for tensor in TRACKED:
            values = [r[tensor] for r in results]
            row = "".join(f"{v:11.4g}" for v in values)
            print(f"{tensor:16s}{row}  {width_exponent(args.widths, values):+.2f}")


if __name__ == "__main__":
    main()
