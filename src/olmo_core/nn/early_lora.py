"""
Low-rank early stack: blocks [0, L) train only LoRA adapters from step 0 (``--st-otr-early-lora r``).

Freezing the early stack from step 0 loses (contradiction -0.24 on the mt4 screen) because those blocks carry
42-51% of the gradient energy early in training; freezing them later (warmup-then-freeze) recovers most of it but
the peak memory is paid during the warmup. LoRA keeps the early stack learning while its base weights carry no
gradient and no Adam state from the first step.

Every ``nn.Linear`` inside blocks [0, L) gets two extra parameters on the module itself, ``lora_A`` (r, in) and
``lora_B`` (out, r), applied by a forward hook: ``y = W x + b + scale * B (A x)``. The base parameter names do not
change, so the base checkpoint loads as before (the adapters are new keys, allowed by the tolerant base load) and
the eval side re-installs them from ``config.json`` (``early_lora`` block) before its strict load, so a LoRA
checkpoint is scored with its adapters and can never silently load as the base.

``B`` starts at 0, so the enabled-but-untrained model is exactly the base. ``scale`` acts as the adapters' learning
rate multiplier under Adam (dB per step is ~LR whatever the scale, so dW = scale x dB x A): at the full-model
fine-tuning LR (5e-6) an unscaled adapter barely moves, hence the default 32.

On a meta-built model (FSDP) ``Transformer.init_weights`` re-runs every module's ``reset_parameters``; each adapted
linear's is patched to also re-init ``A`` (kaiming) and zero ``B``. A forward that bypasses a linear's ``forward``
(direct ``F.linear(x, mod.weight)``) would make its adapter a silent no-op, so the first training forward checks
that every adapter's hook fired and raises otherwise.
"""

from __future__ import annotations

import logging
import math
from types import MethodType
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = ["install_early_lora", "lora_flop_adjust_per_token", "early_lora_flops_per_token", "early_lora_flops_per_block", "enable_from_config_block", "lora_block_from_checkpoint"]


def _lora_hook(mod: nn.Linear, inp, out):
    mod._lora_calls += 1  # type: ignore[attr-defined]
    x = inp[0]
    return out + mod._lora_scale * F.linear(F.linear(x, mod.lora_A), mod.lora_B)  # type: ignore[attr-defined]


def _reset_lora(mod: nn.Linear) -> None:
    with torch.no_grad():
        nn.init.kaiming_uniform_(mod.lora_A, a=math.sqrt(5))  # type: ignore[arg-type]
        mod.lora_B.zero_()  # type: ignore[union-attr]


def _lora_reset_parameters(self: nn.Linear) -> None:
    self._lora_orig_reset()  # type: ignore[attr-defined]
    _reset_lora(self)


def install_early_lora(model: Any, n_blocks: int, rank: int, scale: float = 32.0, freeze_embeddings: bool = True,
                       check_calls: bool = True, start: int = 0, freeze_head: bool = False) -> List[nn.Linear]:
    """Freeze blocks [0, n_blocks) (and the embeddings) and give every ``nn.Linear`` in blocks [start, n_blocks) a
    rank-``rank`` adapter; blocks [0, start) stay frozen with no adapter (the frozen forward-only bottom stack of
    ``--st-otr-freeze-early --st-otr-early-lora-start B``). Returns the adapted modules (``model._early_lora``)."""
    if rank <= 0 or n_blocks <= 0 or not 0 <= start < n_blocks:
        raise ValueError(f"early LoRA needs rank > 0 and 0 <= start < n_blocks, got rank={rank} start={start} "
                         f"n_blocks={n_blocks}")
    if freeze_embeddings:
        for p in model.embeddings.parameters():
            p.requires_grad_(False)
    if freeze_head:
        # frozen LM head (final norm + vocabulary projection): no weight gradient or Adam state; charged at 2/3
        for p in model.lm_head.parameters():
            p.requires_grad_(False)
        model._head_frozen = True
    mods: List[nn.Linear] = []
    for i, block in enumerate(model.blocks.values()):
        if i >= n_blocks:
            break
        for p in block.parameters():
            p.requires_grad_(False)
        if i < start:
            continue
        for m in block.modules():
            if not isinstance(m, nn.Linear):
                continue
            w = m.weight
            m.lora_A = nn.Parameter(torch.empty(rank, m.in_features, device=w.device, dtype=w.dtype))
            m.lora_B = nn.Parameter(torch.empty(m.out_features, rank, device=w.device, dtype=w.dtype))
            m._lora_scale = float(scale)
            m._lora_calls = 0
            m._lora_block = i
            m._lora_orig_reset = m.reset_parameters
            m.reset_parameters = MethodType(_lora_reset_parameters, m)
            if w.device.type != "meta":
                _reset_lora(m)
            m.register_forward_hook(_lora_hook)
            mods.append(m)
    if not mods:
        raise RuntimeError(f"early LoRA: no nn.Linear found in blocks [0, {n_blocks})")
    model._early_lora = {"modules": mods, "n_blocks": int(n_blocks), "rank": int(rank), "scale": float(scale),
                         "start": int(start), "freeze_head": bool(freeze_head), "checked": not check_calls}

    def _check(module, args, output):
        st = module._early_lora
        if st["checked"]:
            return
        # blocks skipped whole by layer drop / grad layer select never run their linears on purpose
        ld = getattr(module, "_layer_drop", None)
        skip = ld.skipped() if (ld is not None and module.training) else frozenset()
        dead = [i for i, m in enumerate(st["modules"]) if m._lora_calls == 0 and m._lora_block not in skip]
        if dead:
            raise RuntimeError(f"early LoRA: {len(dead)} of {len(st['modules'])} adapted linears never ran their "
                               "forward (a fused path reads .weight directly): their adapters would be silent no-ops")
        st["checked"] = True

    model.register_forward_hook(_check)
    n = sum(m.lora_A.numel() + m.lora_B.numel() for m in mods)
    log.info("[early-lora] blocks [%d, %d) (base frozen from 0): %d linears, rank %d, scale %g, %.1fM adapter params",
             start, n_blocks, len(mods), rank, scale, n / 1e6)
    return mods


def early_lora_flops_per_token(model: Any) -> float:
    """Forward + backward FLOPs of the adapters per token: 6 r (in + out) per adapted linear (forward 2 r (in + out),
    input gradient and adapter weight gradients 4 r (in + out))."""
    st = getattr(model, "_early_lora", None)
    if not st:
        return 0.0
    r = st["rank"]
    return float(sum(6 * r * (m.in_features + m.out_features) for m in st["modules"]))


def early_lora_flops_per_block(model: Any) -> List[float]:
    """Per-block forward + backward adapter FLOPs per token (6 r (in + out) per adapted linear), one entry per block."""
    st = getattr(model, "_early_lora", None)
    n = len(model.blocks)
    out = [0.0] * n
    if not st:
        return out
    r = st["rank"]
    for m in st["modules"]:
        out[int(m._lora_block)] += 6.0 * r * (m.in_features + m.out_features)
    return out


def lora_flop_adjust_per_token(model: Any, seq_len: int) -> float:
    """FLOP adjustment per token vs the dense count, for adapters installed WITHOUT the online router (which charges
    them itself): blocks [0, start) run forward-only (nothing below them needs a gradient: -2/3 of the block), blocks
    [start, K) have frozen base weights (no weight gradient: -1/3) plus their adapters' 6 r (in + out)."""
    st = getattr(model, "_early_lora", None)
    if not st:
        return 0.0
    B, K = int(st.get("start", 0)), int(st["n_blocks"])
    ab = early_lora_flops_per_block(model)
    adj = 0.0
    for i, blk in enumerate(model.blocks.values()):
        if i >= K:
            break
        f = float(blk.num_flops_per_token(int(seq_len)))
        adj += -(2.0 / 3.0) * f if i < B else -(1.0 / 3.0) * f + ab[i]
    return adj


def lora_block_from_checkpoint(model_path: str) -> Optional[Dict[str, Any]]:
    """The ``early_lora`` block for a checkpoint whose ``config.json`` lacks one (configs written before 2026-10-24
    dropped it): None if ``model_and_optim/`` holds no adapter keys; else rebuilt from ``provenance.json``'s args.
    Raises rather than return None when adapters exist but the block cannot be rebuilt -- a load without them would
    silently score the frozen base early stack (a DCP load reads only the keys the model asks for)."""
    import json
    import os
    import re

    try:
        from torch.distributed.checkpoint import FileSystemReader

        keys = list(FileSystemReader(os.path.join(model_path, "model_and_optim")).read_metadata().state_dict_metadata)
    except Exception:  # no / unreadable DCP metadata: nothing to say
        return None
    lora = [k for k in keys if k.endswith(".lora_A")]
    if not lora:
        return None
    blocks = {int(m.group(1)) for k in lora if (m := re.search(r"blocks\.(\d+)\.", k))}
    try:
        with open(os.path.join(model_path, "provenance.json")) as f:
            args = json.load(f)["args"]
        rank, scale = int(args["st_otr_early_lora"]), float(args["st_otr_early_lora_scale"])
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"{model_path}: checkpoint holds {len(lora)} LoRA adapters but neither config.json nor "
                           f"provenance.json describes them ({exc}); refusing to load it as a plain model") from exc
    out = {"n_blocks": max(blocks) + 1, "rank": rank, "scale": scale}
    if min(blocks) > 0:
        out["start"] = min(blocks)
    return out


def enable_from_config_block(model: Any, block: Dict[str, Any]) -> None:
    """Eval side: re-install the adapters recorded in ``config.json`` before the strict checkpoint load."""
    # check_calls: the first eval forward (the prefill) must run every adapter's hook -- a fused eval path that reads
    # .weight directly would otherwise score the frozen base silently
    install_early_lora(model, int(block["n_blocks"]), int(block["rank"]), float(block.get("scale", 32.0)),
                       check_calls=True, start=int(block.get("start", 0)))
    print(f"[early-lora] adapters re-installed from config.json: {block}", flush=True)
