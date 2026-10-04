import io

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as distcp
from torch.distributed.tensor import (
    Replicate,
    Shard,
    distribute_tensor,
    init_device_mesh,
)

from olmo_core.distributed.checkpoint.contiguous_planner import ContiguousLoadPlanner
from olmo_core.distributed.checkpoint.filesystem import (
    RemoteFileSystemReader,
    RemoteFileSystemWriter,
    _compact_checkpoint_tensor,
)
from olmo_core.io import dir_is_empty
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.train.train_module.transformer.ddp_train_module import FlatSavePlanner
from olmo_core.utils import get_default_device


def run_save_and_load_with_dtensors(
    dir,
    thread_count: int | None = 2,
    process_count: int | None = None,
    throttle: bool = False,
    compact_storage: bool = False,
    balanced: bool = False,
    constant_memory: bool = False,
):
    mesh = init_device_mesh(get_default_device().type, (dist.get_world_size(),))

    x_full = torch.randn(4, 4, device=get_default_device())
    y_full = torch.randn(4, 8, device=get_default_device())
    # Make sure these tensors are the same across all ranks. We could scatt
    dist.broadcast(x_full, 0)
    dist.broadcast(y_full, 0)

    # Shard the tensors.
    x = distribute_tensor(x_full, mesh, [Shard(dim=0)])
    y = distribute_tensor(y_full, mesh, [Shard(dim=0)])

    replicated = distribute_tensor(
        torch.arange(16, device=get_default_device()), mesh, [Replicate()]
    )
    # Save the sharded tensors.
    distcp.state_dict_saver.save(
        {"x": x, "y": y, "replicated": replicated},
        checkpoint_id=dir,
        storage_writer=RemoteFileSystemWriter(
            dir,
            thread_count=thread_count,
            process_count=process_count,
            throttle_uploads=throttle,
            compact_storage=compact_storage,
            profile=True,
        ),
        planner=FlatSavePlanner(
            dedup_save_to_lowest_rank=not balanced, constant_memory_planning=constant_memory
        ),
    )

    # Both legacy and bounded readers must consume either save policy.
    for load_planner in (None, ContiguousLoadPlanner()):
        # Now create new sharded copies with a different sharding strategy and load the checkpoint.
        x_loaded = distribute_tensor(torch.zeros_like(x_full), mesh, [Shard(dim=1)])
        y_loaded = distribute_tensor(torch.zeros_like(y_full), mesh, [Shard(dim=1)])
        replicated_loaded = distribute_tensor(
            torch.zeros_like(replicated.to_local()), mesh, [Replicate()]
        )
        distcp.state_dict_loader.load(
            {"x": x_loaded, "y": y_loaded, "replicated": replicated_loaded},
            checkpoint_id=dir,
            storage_reader=RemoteFileSystemReader(dir, thread_count=thread_count),
            planner=load_planner,
        )

        # Make sure the loaded tensors match the original tensors.
        x_full_loaded = x_loaded.full_tensor()
        y_full_loaded = y_loaded.full_tensor()
        torch.testing.assert_close(x_full, x_full_loaded)
        torch.testing.assert_close(y_full, y_full_loaded, rtol=0, atol=0)
        torch.testing.assert_close(replicated, replicated_loaded, rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "thread_count, process_count",
    [pytest.param(2, None, id="threads"), pytest.param(None, 2, id="processes")],
)
@pytest.mark.parametrize(
    "compact_storage, balanced, constant_memory",
    [(False, False, False), (True, True, False), (True, True, True)],
)
def test_save_and_load_locally_with_dtensors(
    backend, tmp_path, thread_count, process_count, compact_storage, balanced, constant_memory
):
    run_distributed_test(
        run_save_and_load_with_dtensors,
        backend=backend,
        func_args=(
            tmp_path,
            thread_count,
            process_count,
            False,
            compact_storage,
            balanced,
            constant_memory,
        ),
        start_method="spawn",
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "thread_count, process_count, throttle", [(2, None, True), (2, None, False)]
)
def test_save_and_load_remotely_to_s3_with_dtensors(
    backend, s3_checkpoint_dir, thread_count, process_count, throttle
):
    from botocore.exceptions import NoCredentialsError

    try:
        dir_is_empty(s3_checkpoint_dir)
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")

    run_distributed_test(
        run_save_and_load_with_dtensors,
        backend=backend,
        func_args=(s3_checkpoint_dir, thread_count, process_count, throttle),
        start_method="spawn",  # NOTE: forking causes a crash with boto3
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "thread_count, process_count, throttle", [(2, None, True), (2, None, False)]
)
def test_save_and_load_remotely_to_gcs_with_dtensors(
    backend, gcs_checkpoint_dir, thread_count, process_count, throttle
):
    from google.auth.exceptions import DefaultCredentialsError

    try:
        dir_is_empty(gcs_checkpoint_dir)
    except DefaultCredentialsError:
        pytest.skip("Requires authentication with Google Cloud")

    run_distributed_test(
        run_save_and_load_with_dtensors,
        backend=backend,
        func_args=(gcs_checkpoint_dir, thread_count, process_count, throttle),
        start_method="spawn",  # NOTE: forking causes a crash with boto3
    )


@pytest.mark.parametrize(
    "kind", ["compact", "prefix", "offset", "transpose", "stride", "expanded", "empty"]
)
def test_compact_checkpoint_tensor_preserves_values_and_bounds_storage(kind):
    base = torch.arange(1024 * 1024, dtype=torch.float32)
    values = {
        "compact": base,
        "prefix": base[:16],
        "offset": base[16:32],
        "transpose": base[:16].reshape(4, 4).T,
        "stride": base[::65536],
        "expanded": base[:1].expand(16),
        "empty": base[:0],
    }
    original = values[kind]
    result = _compact_checkpoint_tensor(original)
    torch.testing.assert_close(result, original, rtol=0, atol=0)
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    assert result.untyped_storage().nbytes() == result.nbytes
    if kind == "compact":
        assert result is original
    stream = io.BytesIO()
    torch.save(result, stream)
    assert stream.tell() < result.nbytes + 4096
    stream.seek(0)
    torch.testing.assert_close(torch.load(stream, weights_only=True), original, rtol=0, atol=0)


@pytest.mark.parametrize("profile", [False, True])
def test_checkpoint_writer_and_planner_only_profile_when_enabled(tmp_path, monkeypatch, profile):
    import time
    from types import SimpleNamespace
    from unittest import mock

    from olmo_core.distributed.checkpoint import filesystem
    from olmo_core.train.train_module.transformer import ddp_train_module

    clock = mock.Mock(
        side_effect=time.perf_counter if profile else AssertionError("profiling is off")
    )
    monkeypatch.setattr(filesystem, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(ddp_train_module, "time", SimpleNamespace(perf_counter=clock))
    writer = RemoteFileSystemWriter(tmp_path, thread_count=1, profile=profile)
    planner = FlatSavePlanner(profile=profile)
    expected = {"weights": torch.arange(12).reshape(3, 4), "step": 3}
    distcp.save(expected, storage_writer=writer, planner=planner)
    restored = {"weights": torch.zeros_like(expected["weights"]), "step": 0}
    distcp.load(restored, storage_reader=RemoteFileSystemReader(tmp_path))
    torch.testing.assert_close(restored["weights"], expected["weights"])
    assert restored["step"] == 3
    if profile:
        assert writer.timings["tensor_bytes"] == expected["weights"].nbytes
        assert writer.timings["items"] == 2
        assert "metadata_seconds" in writer.timings
        assert "local_plan_seconds" in planner.timings
        assert "global_plan_seconds" in planner.timings
    else:
        clock.assert_not_called()
        assert writer.timings == planner.timings == {}
