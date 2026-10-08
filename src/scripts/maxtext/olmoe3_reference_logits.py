"""
Run an OLMoE3 (OLMo 3.5) model in OLMo Core and save a logit summary to check a MaxText
conversion against (see ``olmoe3_maxtext_logits.py``).

The model runs in fp32 with the torch attention backend and in eval mode, so EMO routing uses its
eval expert pool. OLMo Core's KDA needs flash-linear-attention's Triton kernels, so this needs a
CUDA GPU.

Either load a checkpoint::

    python src/scripts/maxtext/olmoe3_reference_logits.py \\
        --checkpoint gs://bucket/run/step1000 --random-tokens 2 4096 --output ref.npz

or initialize a random model from a config and save it as a checkpoint to convert (the norm gains
and other constant-initialized parameters get noise so a swapped mapping can't hide)::

    python src/scripts/maxtext/olmoe3_reference_logits.py \\
        --config small_config.json --save-checkpoint /tmp/olmoe3-random --random-tokens 2 256 \\
        --output ref.npz
"""

import argparse
import copy
import json
import logging
import os
from typing import Any, Dict

# A reference has to be true fp32. On Ampere and newer, Triton's fp32 tl.dot defaults to TF32,
# and FLA's KDA kernels also hard-code TF32 for the triangular solve whenever the GPU supports
# it; together they move logits by ~1e-2 relative. Both must be set before FLA's kernels import.
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
try:
    import fla.utils  # type: ignore

    fla.utils.IS_TF32_SUPPORTED = False
except ImportError:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from olmo_core.nn.hf.convert_checkpoint import _normalize_legacy_latent_moe_config
from olmo_core.nn.maxtext.checkpoint import (
    load_olmo_core_config,
    load_olmo_core_weights,
    save_olmo_core_weights,
)
from olmo_core.nn.maxtext.olmoe3 import OLMoE3Geometry
from olmo_core.nn.maxtext.parity import concat, default_positions, summarize_logits
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def _for_reference(value: Any) -> None:
    """Use the torch attention backend and FLA's KDA kernels: same parameters, fewer deps."""
    if isinstance(value, dict):
        if value.get("type") == "attention" and value.get("backend") not in (None, "torch"):
            value["backend"] = "torch"
            value.pop("use_flash", None)
        if value.get("type") == "kimi_delta_attention":
            value["use_experimental_kernels"] = False
        for child in value.values():
            _for_reference(child)
    elif isinstance(value, list):
        for child in value:
            _for_reference(child)


def build_model(model_config: Dict[str, Any], device: torch.device):
    model_config = copy.deepcopy(model_config)
    _normalize_legacy_latent_moe_config(model_config)
    _for_reference(model_config)
    model = TransformerConfig.from_dict(model_config).build(init_device="meta")
    model.to_empty(device=device)
    return model


@torch.no_grad()
def _randomize(model: torch.nn.Module, seed: int) -> None:
    model.init_weights(device=next(model.parameters()).device)
    # The OLMoDDP model casts itself to its bf16 training dtype while initializing.
    model.float()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in model.named_parameters():
        # Norm gains (all ones), SSMax gains and zero biases would map correctly even if the
        # mapping mixed them up, so make every parameter distinct.
        noise = torch.randn(p.shape, generator=gen).to(p.device, p.dtype)
        if p.numel() > 1 and torch.all(p == p.reshape(-1)[0]):
            p.add_(0.1 * noise)
            log.info(f"Perturbed constant-initialized {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="OLMo Core checkpoint step dir")
    src.add_argument(
        "--config", help="experiment config.json (or bare model config) to init randomly"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-checkpoint", help="with --config: save the random model here")
    toks = parser.add_mutually_exclusive_group(required=True)
    toks.add_argument("--tokens", help=".npy of [batch, seq] token IDs")
    toks.add_argument("--random-tokens", type=int, nargs=2, metavar=("BATCH", "SEQ"))
    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=100257,
        help="kept out of random tokens so each row is one document",
    )
    parser.add_argument(
        "--n-positions", type=int, default=16, help="positions with full logits kept"
    )
    parser.add_argument("--output", required=True, help="where to write the .npz summary")
    args = parser.parse_args()
    prepare_cli_environment()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.checkpoint:
        experiment_config = load_olmo_core_config(args.checkpoint)
    else:
        with open(args.config) as f:
            experiment_config = json.load(f)
        if "model" not in experiment_config:
            experiment_config = {"model": experiment_config}
    model_config = experiment_config["model"]
    geometry = OLMoE3Geometry.from_olmo_core_config(_normalized(model_config))
    model = build_model(model_config, device)

    if args.checkpoint:
        state = load_olmo_core_weights(args.checkpoint, geometry)
        model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        del state
    else:
        _randomize(model, args.seed)
        if args.save_checkpoint:
            state = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
            save_olmo_core_weights(
                args.save_checkpoint, state, experiment_config, save_overwrite=True
            )
            log.info(f"Saved the random model to {args.save_checkpoint}")
    model.eval()

    if args.tokens:
        tokens = np.load(args.tokens).astype(np.int64)
    else:
        rng = np.random.default_rng(args.seed)
        tokens = rng.integers(0, geometry.vocab_size, size=tuple(args.random_tokens))
        tokens[tokens == args.eos_token_id] = 0
    positions = default_positions(tokens.shape[1], args.n_positions)

    summaries = []
    with torch.no_grad():
        for row in tokens:
            ids = torch.from_numpy(row[None]).to(device)
            logits = model(input_ids=ids).float().cpu().numpy()
            summaries.append(summarize_logits(row[None], logits, positions))
    summary = concat(summaries)
    summary.save(args.output)
    log.info(
        f"Wrote {args.output}: {tokens.shape[0]} x {tokens.shape[1]} tokens, "
        f"loss {-summary.next_token_logprob.mean():.4f}"
    )


def _normalized(model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = copy.deepcopy(model_config)
    _normalize_legacy_latent_moe_config(model_config)
    return model_config


if __name__ == "__main__":
    main()
