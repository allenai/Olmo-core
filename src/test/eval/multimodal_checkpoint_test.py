from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.checkpoint.metadata import (
    TensorProperties,
    TensorStorageMetadata,
)

from olmo_core.eval import multimodal_checkpoint as checkpoint
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath


@pytest.mark.parametrize("response_only", [False, True])
def test_response_ce_keeps_per_example_prefixes_and_weights(response_only):
    labels = torch.tensor([[0, 1, 2, -100], [2, 1, -100, -100]])
    weights = torch.tensor([[1.0, 2.0, 0.5, 0.0], [3.0, 1.0, 0.0, 0.0]])
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 7
    selected = weights > 0
    actual = checkpoint.response_ce_by_example(
        {"labels": labels, "loss_masks": weights}, logits[selected] if response_only else logits
    )
    assert [row["response_tokens"] for row in actual] == [3, 2]
    for index, row in enumerate(actual):
        ce = F.cross_entropy(
            logits[index][selected[index]], labels[index][selected[index]], reduction="none"
        )
        selected_weights = weights[index][selected[index]]
        assert row["windows"]["first_1"] == pytest.approx(float(ce[0]))
        expected = float((ce * selected_weights).sum() / selected_weights.sum())
        for window in ("all", "first_8", "first_32"):
            assert row["windows"][window] == pytest.approx(expected)


def _coverage_fixture():
    model = torch.nn.Module()
    model.connector = torch.nn.Parameter(torch.zeros(2, 2))
    model.vision = torch.nn.Parameter(torch.zeros(3, 3), requires_grad=False)
    router = torch.zeros(1)
    eval_state = {"module.connector.main": model.connector.flatten()}
    frozen_state = {"frozen_model.vision": model.vision}
    buffers = {"model_buffer.router": router}
    metadata = SimpleNamespace(
        state_dict_metadata={
            key: TensorStorageMetadata(
                properties=TensorProperties(dtype=tensor.dtype), size=tensor.size(), chunks=[]
            )
            for key, tensor in (eval_state | frozen_state | buffers).items()
        }
    )
    module = SimpleNamespace(
        model_parts=[model],
        _get_model_state_dict_for_eval_load=lambda metadata: eval_state,
        _resolve_model_checkpoint_key=lambda name, keys: (
            "module.connector.main" if name == "connector" else None
        ),
        _frozen_checkpoint_model_param_state_dict_for_load=lambda keys: frozen_state,
        _frozen_checkpoint_param_state_dict_for_load=lambda keys: frozen_state,
        _persistent_model_buffer_state_dict=lambda: buffers,
    )
    return module, metadata


@pytest.mark.parametrize("failure", [None, "router", "vision", "shape", "unused"])
def test_load_coverage_includes_frozen_vision_and_router(monkeypatch, tmp_path, failure):
    module, metadata = _coverage_fixture()
    monkeypatch.setattr(checkpoint, "get_checkpoint_metadata", lambda path: metadata)
    if failure == "router":
        metadata.state_dict_metadata.pop("model_buffer.router")
    elif failure == "vision":
        module._frozen_checkpoint_model_param_state_dict_for_load = lambda keys: {}
        module._frozen_checkpoint_param_state_dict_for_load = lambda keys: {}
    elif failure == "shape":
        metadata.state_dict_metadata["module.connector.main"].size = torch.Size([2, 2])
    elif failure == "unused":
        metadata.state_dict_metadata["module.orphan.main"] = metadata.state_dict_metadata[
            "module.connector.main"
        ]
    if failure:
        with pytest.raises(RuntimeError):
            checkpoint.native_checkpoint_load_coverage(module, tmp_path)
    else:
        report = checkpoint.native_checkpoint_load_coverage(module, tmp_path)
        assert report["complete"]
        assert report["model_parameter_count"] == 2
        assert report["frozen_state_key_count"] == 1
        assert report["persistent_buffer_count"] == 1
        assert report["unused_model_bearing_key_count"] == 0


def test_eval_configuration_preserves_router_coefficients():
    ep = SimpleNamespace(path=ExpertParallelPath.sync_1d)
    block = SimpleNamespace(sequence_mixer=AttentionConfig(n_heads=2), ep=ep)
    router = {"load_balancing_loss_weight": 0.01, "z_loss_weight": 0.001}
    config = SimpleNamespace(
        block=block,
        block_overrides=None,
        router=router,
        recompute_each_block=True,
        recompute_all_blocks_by_chunk=True,
        two_batch_overlap=True,
    )
    checkpoint.configure_lm_for_eval(config)
    assert block.sequence_mixer.backend == AttentionBackendName.flex
    assert ep.path == ExpertParallelPath.rowwise_nvshmem
    assert config.router == {"load_balancing_loss_weight": 0.01, "z_loss_weight": 0.001}
    assert not config.recompute_each_block
    assert not config.recompute_all_blocks_by_chunk
    assert not config.two_batch_overlap


def test_checkpoint_state_directory(tmp_path):
    assert checkpoint.checkpoint_state_dir(tmp_path) == tmp_path
    nested = tmp_path / "model_and_optim"
    nested.mkdir()
    assert checkpoint.checkpoint_state_dir(tmp_path) == nested
    assert checkpoint.checkpoint_state_dir(nested) == nested
    assert checkpoint.checkpoint_state_dir(Path("model_and_optim")) == Path("model_and_optim")
