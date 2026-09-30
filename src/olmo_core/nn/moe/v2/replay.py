"""Explicit expert routing replay for scoring and activation recomputation."""

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch

from .emo_router import EmoRouterV2
from .router import MoERouterV2


@contextmanager
def replay_routes(model: torch.nn.Module, routes: Mapping[str, torch.Tensor]) -> Iterator[None]:
    """Replay routes by router module name, retaining gradients through routing weights.

    Keep this context open through backward so activation recomputation uses the
    same expert identities. Every routed router must be supplied; shared-only
    routers are excluded by the caller's module-name selection. Nested contexts
    restore their previous routes even when a forward or backward fails. Routers
    with unsupported forward implementations are rejected. EMO explicitly delegates
    replay to the standard forward, bypassing document-pool selection.
    """
    routers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, MoERouterV2) and name.endswith("routed_experts_router")
    }
    if not routers or set(routes) != set(routers):
        raise ValueError("Replay must supply exactly the model's routed expert routers")
    for name, indices in routes.items():
        router = routers[name]
        if type(router).forward not in (MoERouterV2.forward, EmoRouterV2.forward):
            raise ValueError(f"Replay does not support {type(router).__name__} at {name}")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Replay indices for {name} must be integers")
        if indices.ndim != 3 or indices.shape[-1] != router.top_k:
            raise ValueError(f"Replay indices for {name} must have shape [batch, tokens, top_k]")
        if bool(((indices < 0) | (indices >= router.num_experts)).any()):
            raise ValueError(f"Replay indices for {name} are outside the expert range")
        if bool((indices.sort(dim=-1).values.diff(dim=-1) == 0).any()):
            raise ValueError(f"Replay indices for {name} must select distinct experts")
    previous = {
        name: getattr(router, "replay_expert_indices", None) for name, router in routers.items()
    }
    try:
        for name, router in routers.items():
            router.replay_expert_indices = routes[name]
        yield
    finally:
        for name, router in routers.items():
            router.replay_expert_indices = previous[name]
