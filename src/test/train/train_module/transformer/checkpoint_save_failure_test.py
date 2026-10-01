"""Native save must restore released optimizer storage even if writing fails."""

from unittest import mock

import pytest
import torch

from olmo_core.train.train_module.transformer import ddp_train_module


@pytest.mark.parametrize("failure_phase", ["buffers", "write"])
def test_failed_save_restores_live_optimizer(tmp_path, monkeypatch, failure_phase):
    states = {"parameter.main": torch.tensor([1.0])}
    optimizer = mock.Mock()
    optimizer.state_dict.return_value = states
    module = mock.Mock(
        spec=ddp_train_module.OLMoDDPTrainModule,
        _require_optimizer=lambda: optimizer,
        _persistent_model_buffer_state_dict=mock.Mock(return_value={}),
    )
    error = RuntimeError("injected save failure")
    monkeypatch.setattr(ddp_train_module, "_prepare_env_for_save", lambda path, **kwargs: path)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    write = mock.Mock(side_effect=error if failure_phase == "write" else None)
    monkeypatch.setattr(ddp_train_module.dist_cp.state_dict_saver, "save", write)
    if failure_phase == "buffers":
        module._persistent_model_buffer_state_dict.side_effect = error
    with pytest.raises(RuntimeError, match="injected save failure"):
        ddp_train_module.OLMoDDPTrainModule.save_state_dict_direct(module, tmp_path)
    optimizer.load_state_dict.assert_called_once_with(states, reset_optimizer_moments_on_load=False)


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
@pytest.mark.parametrize("profile", [False, True])
def test_direct_save_profiles_only_when_enabled(tmp_path, monkeypatch, profile, device):
    import time
    from types import SimpleNamespace

    optimizer = mock.Mock()
    optimizer.state_dict.return_value = {"parameter.main": torch.tensor([1.0])}
    module = mock.Mock(
        spec=ddp_train_module.OLMoDDPTrainModule,
        device=torch.device(device),
        _require_optimizer=lambda: optimizer,
        _persistent_model_buffer_state_dict=mock.Mock(return_value={}),
    )
    clock = mock.Mock(
        side_effect=time.perf_counter if profile else AssertionError("profiling is off")
    )
    sync_cuda = profile and device.startswith("cuda")
    sync = mock.Mock(
        side_effect=None if sync_cuda else AssertionError("CUDA synchronization is not needed")
    )
    monkeypatch.setattr(ddp_train_module, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(ddp_train_module, "_prepare_env_for_save", lambda path, **kwargs: path)

    def save(state, *, storage_writer, process_group, planner):
        assert storage_writer.profile is profile
        assert planner.profile is profile

    monkeypatch.setattr(ddp_train_module.dist_cp.state_dict_saver, "save", save)
    result = ddp_train_module.OLMoDDPTrainModule.save_state_dict_direct(
        module, tmp_path, profile=profile
    )
    optimizer.load_state_dict.assert_called_once()
    if profile:
        assert result is not None and "total_seconds" in result
        assert clock.called
    else:
        assert result is None
        clock.assert_not_called()
    if sync_cuda:
        assert sync.call_count > 0
        assert all(call == mock.call(torch.device(device)) for call in sync.call_args_list)
    else:
        sync.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
@pytest.mark.parametrize("profile", [False, True])
def test_direct_load_profiles_only_when_enabled(tmp_path, monkeypatch, profile, device):
    import time
    from types import SimpleNamespace

    reader = mock.Mock()
    reader.read_metadata.return_value = SimpleNamespace(state_dict_metadata={})

    def load_model(metadata, path, reader, group, *, load, constant_memory_planning):
        load({}, checkpoint_id=path)

    module = mock.Mock(
        spec=ddp_train_module.OLMoDDPTrainModule,
        device=torch.device(device),
        eval_only=True,
        _load_model_state_dict_direct=load_model,
        _persistent_model_buffer_state_dict=mock.Mock(return_value={}),
    )
    clock = mock.Mock(
        side_effect=time.perf_counter if profile else AssertionError("profiling is off")
    )
    sync_cuda = profile and device.startswith("cuda")
    sync = mock.Mock(
        side_effect=None if sync_cuda else AssertionError("CUDA synchronization is not needed")
    )
    monkeypatch.setattr(ddp_train_module, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(ddp_train_module, "RemoteFileSystemReader", lambda *args, **kwargs: reader)
    load = mock.Mock()
    monkeypatch.setattr(ddp_train_module.dist_cp.state_dict_loader, "load", load)
    ddp_train_module.OLMoDDPTrainModule.load_state_dict_direct(module, tmp_path, profile=profile)
    load.assert_called_once()
    if profile:
        assert clock.called
    else:
        clock.assert_not_called()
    if sync_cuda:
        assert sync.call_count > 0
        assert all(call == mock.call(torch.device(device)) for call in sync.call_args_list)
    else:
        sync.assert_not_called()


def test_profiled_cpu_checkpoint_roundtrip(tmp_path, monkeypatch):
    states = {"parameter.main": torch.arange(4, dtype=torch.float32)}
    restored = {"parameter.main": torch.zeros(4)}
    optimizer = mock.Mock()
    optimizer.state_dict.return_value = states

    def load_model(metadata, path, reader, group, *, load, constant_memory_planning):
        load(restored, storage_reader=reader, process_group=group)

    module = mock.Mock(
        spec=ddp_train_module.OLMoDDPTrainModule,
        device=torch.device("cpu"),
        eval_only=True,
        _require_optimizer=lambda: optimizer,
        _persistent_model_buffer_state_dict=mock.Mock(return_value={}),
        _load_model_state_dict_direct=load_model,
    )
    sync = mock.Mock(side_effect=AssertionError("CPU profiling must not synchronize CUDA"))
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    path = tmp_path / "checkpoint"
    timings = ddp_train_module.OLMoDDPTrainModule.save_state_dict_direct(module, path, profile=True)
    assert timings is not None and timings["total_seconds"] >= 0
    ddp_train_module.OLMoDDPTrainModule.load_state_dict_direct(module, path, profile=True)
    torch.testing.assert_close(restored["parameter.main"], states["parameter.main"])
    sync.assert_not_called()
