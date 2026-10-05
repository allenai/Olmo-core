"""Verify the original sampled state after a DCP restore onto a doubled or halved DP size."""

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist


def half_sample_indices(old_numel, half):
    """Map the saved evenly spaced samples into one of two equal contiguous shards."""
    assert old_numel > 0 and old_numel % 2 == 0 and half in (0, 1)
    width = old_numel // 2
    offsets = [
        i * (old_numel - 1) // max(1, min(128, old_numel) - 1) for i in range(min(128, old_numel))
    ]
    return [min(max(i - half * width, 0), width - 1) for i in offsets], [
        half * width <= i < (half + 1) * width for i in offsets
    ]


def verify_doubled_dp(trainer, path, actual):
    """Reassemble only sampled optimizer bytes from adjacent new DP ranks."""
    rank = dist.get_rank()
    assert dist.get_world_size() == 128
    saved = json.loads((Path(path) / "resume_audit" / f"rank{rank // 2}.json").read_text())
    assert saved["gpus"] == 64 and saved["rank"] == rank // 2
    for key in ("step", "tokens", "loss_history", "norm_history"):
        assert saved[key] == actual[key], key
    assert saved["tensors"].keys() == actual["tensors"].keys()
    payloads, masks, records = [], [], []
    for name, expected in sorted(saved["tensors"].items()):
        observed = actual["tensors"][name]
        if expected[0] == observed[0]:
            assert expected == observed, name
            continue
        tensor = trainer.train_module.optim.states[name]
        assert len(tensor.placements) == 1 and tensor.placements[0].is_shard(0), name
        local = tensor.to_local().detach().reshape(-1)
        assert expected[0] == local.numel() * 2 and expected[1] == str(local.dtype), name
        indices, selected = half_sample_indices(expected[0], rank % 2)
        sampled = local[torch.tensor(indices, device=local.device)].contiguous().view(torch.uint8)
        mask = torch.tensor(selected, device=local.device).repeat_interleave(local.element_size())
        payloads.append(sampled)
        masks.append(mask)
        records.append((name, sampled.numel(), expected[2]))
    send = torch.cat(payloads)
    receive = torch.empty_like(send)
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send, rank ^ 1),
            dist.P2POp(dist.irecv, receive, rank ^ 1),
        ]
    )
    for request in requests:
        request.wait()
    merged = torch.where(torch.cat(masks), send, receive).cpu().numpy().tobytes()
    start = 0
    for name, size, digest in records:
        assert hashlib.sha256(merged[start : start + size]).hexdigest() == digest, name
        start += size
    assert start == len(merged)
    return {
        "source_gpus": 64,
        "gpus": 128,
        "source_rank": rank // 2,
        "resharded_optimizer_tensors_verified": len(records),
        "rng_policy": "native reseed when world size changes",
    }


def sample_digest(flat, start, width):
    """Hash the evenly spaced samples that state_sample takes from one saved shard."""
    count = min(128, width)
    offsets = [start + i * (width - 1) // max(1, count - 1) for i in range(count)]
    sampled = flat[torch.tensor(offsets, device=flat.device)].contiguous().view(torch.uint8)
    return hashlib.sha256(sampled.cpu().numpy().tobytes()).hexdigest()


def verify_halved_dp(trainer, path, actual):
    """Find both original 128-rank shards, back to back, inside each 64-rank shard."""
    rank = dist.get_rank()
    assert dist.get_world_size() == 64
    sources = (2 * rank, 2 * rank + 1)
    saved = [
        json.loads((Path(path) / "resume_audit" / f"rank{source}.json").read_text())
        for source in sources
    ]
    for source, row in zip(sources, saved):
        assert row["gpus"] == 128 and row["rank"] == source, source
        for key in ("step", "tokens", "loss_history", "norm_history"):
            assert row[key] == actual[key], key
        assert row["tensors"].keys() == actual["tensors"].keys(), source
    verified = 0
    for name, observed in sorted(actual["tensors"].items()):
        first, second = (row["tensors"][name] for row in saved)
        if first[0] == observed[0]:
            assert first == observed and second == observed, name
            continue
        tensor = trainer.train_module.optim.states[name]
        assert len(tensor.placements) == 1 and tensor.placements[0].is_shard(0), name
        local = tensor.to_local().detach().reshape(-1)
        width = first[0]
        assert second[0] == width and local.numel() == 2 * width, name
        assert first[1] == second[1] == str(local.dtype), name
        assert sample_digest(local, 0, width) == first[2], name
        assert sample_digest(local, width, width) == second[2], name
        verified += 1
    return {
        "source_gpus": 128,
        "gpus": 64,
        "source_ranks": list(sources),
        "resharded_optimizer_tensors_verified": verified,
        "rng_policy": "native reseed when world size changes",
    }
