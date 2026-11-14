"""
A CompletedP coordinate check across width and depth.

Trains small Llama-like models for a few steps on a fixed batch, then reports how much each
tracked activation moved per coordinate (the RMS of its change). Two sweeps are run: width at a
fixed depth, and depth at a fixed width. Under CompletedP
(:class:`~olmo_core.nn.transformer.CompletedPConfig` on the model and
:class:`~olmo_core.optim.CompletedPOptimConfig` on the optimizer) these changes should stay roughly
constant along both sweeps. Under standard parameterization (std 0.02, one learning rate, no
residual scaling) they grow with width, and the residual stream's change grows with depth.

The head dimension is fixed, so the number of heads grows with the width. Each "exponent" column
is the least-squares slope of log(change) against log(width) or log(depth): about 0 means the
change does not depend on it.

Run on one GPU::

    python src/examples/completedp/coord_check.py
"""

import argparse
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.transformer import (
    CompletedPConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.optim import AdamWConfig, CompletedPOptimConfig
from olmo_core.utils import seed_all

HEAD_DIM = 64
TRACKED = ("embedding", "middle block", "last block", "logits")


def activation_changes(
    width: int, depth: int, *, alpha: Optional[float], args: argparse.Namespace
) -> Dict[str, float]:
    """
    Train one model and return the per-coordinate RMS change of each tracked activation.

    :param alpha: The CompletedP depth exponent, or ``None`` for standard parameterization.
    """
    seed_all(0)
    device = torch.device(args.device)
    completedp = (
        None
        if alpha is None
        else CompletedPConfig(base_width=args.base_width, base_depth=args.base_depth, alpha=alpha)
    )
    model = TransformerConfig.llama_like(
        d_model=width,
        vocab_size=args.vocab_size,
        n_layers=depth,
        n_heads=width // HEAD_DIM,
        feed_forward=FeedForwardConfig(hidden_size=4 * width, bias=False),
        qk_norm=True,
        use_head_qk_norm=True,
        attn_backend=AttentionBackendName.torch,
        block_name=TransformerBlockType(args.block),
        embedding_norm=(
            LayerNormConfig(name=LayerNormType.rms, bias=False) if args.embedding_norm else None
        ),
        completedp=completedp,
    ).build(init_device="meta")
    model.init_weights(device=device)
    optim = AdamWConfig(
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        completedp=None if alpha is None else CompletedPOptimConfig(),
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
        model.blocks[str(depth // 2)].register_forward_hook(capture("middle block")),
        model.blocks[str(depth - 1)].register_forward_hook(capture("last block")),
    ]

    def snapshot() -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = model(input_ids).float()
        return {**captured, "logits": logits}

    initial = snapshot()
    for _ in range(args.steps):
        batch = input_ids
        if args.fresh_batches:
            batch = torch.randint(
                args.vocab_size, input_ids.shape, device=device, generator=generator
            )
        logits = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
        )
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    final = snapshot()
    for hook in hooks:
        hook.remove()

    return {name: (final[name] - initial[name]).pow(2).mean().sqrt().item() for name in TRACKED}


def log_slope(xs: List[int], values: List[float]) -> float:
    """
    The least-squares slope of log(value) against log(x).
    """
    lx = [math.log(x) for x in xs]
    ly = [math.log(v) for v in values]
    x_mean, y_mean = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(lx, ly)) / sum(
        (x - x_mean) ** 2 for x in lx
    )


def print_sweep(title: str, label: str, xs: List[int], results: List[Dict[str, float]], steps: int):
    print(f"\n{title}: per-coordinate RMS change after {steps} steps")
    print(f"{'':16s}" + "".join(f"{f'{label}={x}':>11s}" for x in xs) + "   exponent")
    for tensor in TRACKED:
        values = [r[tensor] for r in results]
        row = "".join(f"{v:11.4g}" for v in values)
        print(f"{tensor:16s}{row}   {log_slope(xs, values):+.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--fixed-depth", type=int, default=4, help="Depth of the width sweep.")
    parser.add_argument("--fixed-width", type=int, default=256, help="Width of the depth sweep.")
    parser.add_argument("--base-width", type=int, default=128)
    parser.add_argument("--base-depth", type=int, default=2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--block",
        default=TransformerBlockType.default.value,
        choices=[t.value for t in (TransformerBlockType.default, TransformerBlockType.peri_norm)],
    )
    parser.add_argument("--embedding-norm", action="store_true", help="Normalize the embeddings.")
    parser.add_argument(
        "--fresh-batches",
        action="store_true",
        help="Train on a new random batch each step instead of one fixed batch (activations are "
        "still measured on the fixed batch).",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    parameterizations: List[Optional[float]] = [None, *args.alphas]
    for alpha in parameterizations:
        name = "standard" if alpha is None else f"CompletedP (alpha={alpha:g})"
        width_results = [
            activation_changes(w, args.fixed_depth, alpha=alpha, args=args) for w in args.widths
        ]
        print_sweep(
            f"{name}, width sweep at depth {args.fixed_depth}",
            "w",
            args.widths,
            width_results,
            args.steps,
        )
        depth_results = [
            activation_changes(args.fixed_width, d, alpha=alpha, args=args) for d in args.depths
        ]
        print_sweep(
            f"{name}, depth sweep at width {args.fixed_width}",
            "L",
            args.depths,
            depth_results,
            args.steps,
        )


if __name__ == "__main__":
    main()
