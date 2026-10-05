import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from olmo_core.testing import run_distributed_test


def fingerprint(tensor):
    flat = tensor.flatten()
    indices = [
        i * (flat.numel() - 1) // max(1, min(128, flat.numel()) - 1)
        for i in range(min(128, flat.numel()))
    ]
    return [
        flat.numel(),
        str(flat.dtype),
        hashlib.sha256(flat[indices].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
    ]


def check_exchange():
    import olmoe3_hero_reshard_audit as audit

    rank = dist.get_rank()
    audit.dist = SimpleNamespace(
        get_rank=lambda: rank,
        get_world_size=lambda: 128,
        batch_isend_irecv=dist.batch_isend_irecv,
        P2POp=dist.P2POp,
        isend=dist.isend,
        irecv=dist.irecv,
    )
    original = torch.arange(600, dtype=torch.float32) - 250
    original[2] = -0.0
    local = original.chunk(2)[rank]
    state = SimpleNamespace(
        placements=[SimpleNamespace(is_shard=lambda dim: dim == 0)], to_local=lambda: local
    )
    saved = dict(
        rank=0,
        gpus=64,
        step=300411,
        tokens=5040060235776,
        loss_history=[1.0],
        norm_history=[2.0],
        tensors={"weight.main": fingerprint(original)},
    )
    actual = {**saved, "gpus": 128, "rank": rank, "tensors": {"weight.main": fingerprint(local)}}
    trainer = SimpleNamespace(
        train_module=SimpleNamespace(optim=SimpleNamespace(states={"weight.main": state}))
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        (path / "resume_audit").mkdir()
        (path / "resume_audit/rank0.json").write_text(json.dumps(saved))
        proof = audit.verify_doubled_dp(trainer, path, actual)
        assert proof["resharded_optimizer_tensors_verified"] == 1


def test_resharded_optimizer_sample_exchange():
    run_distributed_test(check_exchange, backend="gloo", start_method="spawn")


def check_halved(monkeypatch, rank, shards, local, replicated, live_replicated):
    import olmoe3_hero_reshard_audit as audit

    monkeypatch.setattr(
        audit, "dist", SimpleNamespace(get_rank=lambda: rank, get_world_size=lambda: 64)
    )
    history = {
        "step": 357500,
        "tokens": 357500 * 16777216,
        "loss_history": [1.0],
        "norm_history": [2.0],
    }
    state = SimpleNamespace(
        placements=[SimpleNamespace(is_shard=lambda dim: dim == 0)], to_local=lambda: local
    )
    trainer = SimpleNamespace(
        train_module=SimpleNamespace(optim=SimpleNamespace(states={"exp_avg.w": state}))
    )
    actual = dict(
        rank=rank,
        gpus=64,
        **history,
        tensors={"exp_avg.w": fingerprint(local), "model_param/w": fingerprint(live_replicated)},
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        (path / "resume_audit").mkdir()
        for half, shard in enumerate(shards):
            source = 2 * rank + half
            row = dict(
                rank=source,
                gpus=128,
                **history,
                tensors={"exp_avg.w": fingerprint(shard), "model_param/w": fingerprint(replicated)},
            )
            (path / "resume_audit" / f"rank{source}.json").write_text(json.dumps(row))
        return audit.verify_halved_dp(trainer, path, actual)


@pytest.mark.parametrize("width", [300, 100])
@pytest.mark.parametrize("rank", [0, 37, 63])
def test_halved_optimizer_samples(monkeypatch, rank, width):
    original = torch.arange(2 * width, dtype=torch.float32) - width
    original[width + 2] = -0.0
    shards = original.chunk(2)
    replicated = torch.linspace(-1, 1, 257)
    proof = check_halved(monkeypatch, rank, shards, original, replicated, replicated)
    assert proof["source_ranks"] == [2 * rank, 2 * rank + 1]
    assert proof["resharded_optimizer_tensors_verified"] == 1
    assert (proof["source_gpus"], proof["gpus"]) == (128, 64)


def test_halved_requires_64_ranks(monkeypatch):
    import olmoe3_hero_reshard_audit as audit

    monkeypatch.setattr(
        audit, "dist", SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 128)
    )
    with pytest.raises(AssertionError):
        audit.verify_halved_dp(None, Path("/nonexistent"), {})


@pytest.mark.parametrize("case", ["swapped", "uneven", "signed-zero", "replicated-mismatch"])
def test_halved_optimizer_samples_reject(monkeypatch, case):
    original = torch.arange(600, dtype=torch.float32) - 250
    original[302] = -0.0
    shards = original.chunk(2)
    replicated = torch.linspace(-1, 1, 257)
    live, local = replicated, original
    if case == "swapped":
        local = torch.cat([shards[1], shards[0]])
    elif case == "uneven":
        shards = (original[:290], original[290:])
    elif case == "signed-zero":
        local = original.clone()
        local[302] = 0.0
    elif case == "replicated-mismatch":
        live = replicated.clone()
        live[0] = 2.0
    with pytest.raises(AssertionError):
        check_halved(monkeypatch, 0, shards, local, replicated, live)
