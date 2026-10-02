"""
Tests for the checkpoint adapters of :class:`MultimodalOLMoDDPTrainModule`, which ride on the
identity hooks of :class:`OLMoDDPTrainModule` (``_optimizer_state_dict_for_load`` /
``_optimizer_state_dict_after_load``): native text checkpoints without the ``lm.`` prefix,
freshly initialized components, frozen parameters and buffers. The distributed load itself is
stubbed; only the key bookkeeping is under test.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.checkpoint.metadata import (
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)

from olmo_core.optim.multimodal_optimizer import MultimodalOLMoDDPOptimizer
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModule,
)


def _module(params, buffers=None, *, ep=False):
    named = {
        name: torch.nn.Parameter(torch.full((2, 2), float(i)), requires_grad=trainable)
        for i, (name, trainable) in enumerate(params.items())
    }
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module.model_parts = [SimpleNamespace(named_parameters=lambda: list(named.items()))]
    module._persistent_model_buffer_state_dict = lambda: dict(buffers or {})
    module._ep_config = object() if ep else None
    module.eval_only = False
    module._load_key_map = {}
    module._load_deferred = {}
    module._load_in_place_keys = set()
    module._load_resync_names = set()
    module._load_expansions = {}
    module._load_metadata = None
    module.expand_shared_qk_norm_on_load = False
    optim = object.__new__(MultimodalOLMoDDPOptimizer)
    optim.states = {}
    optim.param_groups = [
        {"pg": "dp", "named_params": {n: p for n, p in named.items() if p.requires_grad}}
    ]
    synced = []
    optim._copy_model_params_to_main_params = lambda names: synced.append(set(names))
    module.optim = optim
    return module, named, optim, synced


def test_native_text_checkpoint_is_aliased_and_fresh_components_are_deferred():
    buffer = torch.zeros(3)
    module, named, _, synced = _module(
        {
            "module.lm.blocks.0.w": True,
            "module.lm.lm_head.w": False,
            "module.vision.w": True,
            "module.connector.w": True,
        },
        {"model_buffer.lm.blocks.0.router.score_bias": buffer},
    )
    checkpoint_keys = {
        "module.blocks.0.w.main",
        "module.blocks.0.w.exp_avg",
        "module.lm_head.w.main",
        "model_buffer.blocks.0.router.score_bias",
    }
    a, c, d = (torch.zeros(1) for _ in range(3))
    # A phase handoff loads the masters only (load_optim_state=False).
    state = {
        "module.lm.blocks.0.w.main": a,
        "module.vision.w.main": c,
        "module.connector.w.main": d,
        "__moe_skip_step_losses": [],
    }

    to_load = module._optimizer_state_dict_for_load(state, checkpoint_keys)

    assert to_load["module.blocks.0.w.main"] is a
    # The frozen LM-head parameter is filled in place from the text master ...
    assert to_load["module.lm_head.w.main"].data_ptr() == named["module.lm.lm_head.w"].data_ptr()
    # ... and the LM buffer from its un-prefixed key.
    assert to_load["model_buffer.blocks.0.router.score_bias"] is buffer
    assert "module.vision.w.main" not in to_load and "module.connector.w.main" not in to_load
    assert to_load["__moe_skip_step_losses"] == []

    after = module._optimizer_state_dict_after_load(to_load)

    assert after == {
        "module.lm.blocks.0.w.main": a,
        "module.vision.w.main": c,
        "module.connector.w.main": d,
        "__moe_skip_step_losses": [],
    }
    assert synced == []
    assert module._load_key_map == {} and module._load_deferred == {}


def test_parameters_frozen_at_save_time_are_reloaded_and_resynchronized():
    module, named, optim, synced = _module(
        {"module.lm.blocks.0.w": True, "module.lm.blocks.1.w": True, "module.lm.lm_head.w": False}
    )
    optim.states["module.lm.blocks.1.w.main"] = master = torch.zeros(4)
    checkpoint_keys = {
        "module.lm.blocks.0.w.main",
        "frozen_model.lm.blocks.1.w",
        "frozen_model.lm.lm_head.w",
    }
    state = {
        "module.lm.blocks.0.w.main": torch.zeros(1),
        "module.lm.blocks.1.w.main": torch.ones(1),
    }

    to_load = module._optimizer_state_dict_for_load(state, checkpoint_keys)

    assert set(to_load) == {
        "module.lm.blocks.0.w.main",
        "frozen_model.lm.blocks.1.w",
        "frozen_model.lm.lm_head.w",
    }
    assert (
        to_load["frozen_model.lm.lm_head.w"].data_ptr() == named["module.lm.lm_head.w"].data_ptr()
    )
    assert (
        to_load["frozen_model.lm.blocks.1.w"].data_ptr() == named["module.lm.blocks.1.w"].data_ptr()
    )

    after = module._optimizer_state_dict_after_load(to_load)

    assert synced == [{"module.lm.blocks.1.w"}]
    assert after["module.lm.blocks.1.w.main"] is master
    assert set(after) == {"module.lm.blocks.0.w.main", "module.lm.blocks.1.w.main"}


def test_frozen_parameters_are_saved_under_stable_keys_and_ep_experts_are_rejected():
    module, named, _, _ = _module({"module.lm.blocks.0.w": True, "module.lm.lm_head.w": False})
    frozen = module._frozen_model_param_state_dict()
    assert list(frozen) == ["frozen_model.lm.lm_head.w"]
    assert frozen["frozen_model.lm.lm_head.w"].data_ptr() == named["module.lm.lm_head.w"].data_ptr()

    module, _, _, _ = _module({"module.lm.blocks.0.routed_experts.w": False}, ep=True)
    with pytest.raises(NotImplementedError, match="frozen expert-parallel"):
        module._frozen_model_param_state_dict()


def test_model_checkpoint_keys_resolve_with_and_without_the_lm_prefix():
    module, _, _, _ = _module({"module.lm.blocks.0.w": True, "module.vision.w": True})
    assert (
        module._resolve_model_checkpoint_key("module.lm.blocks.0.w", {"module.blocks.0.w.main"})
        == "module.blocks.0.w.main"
    )
    assert (
        module._resolve_model_checkpoint_key("module.lm.blocks.0.w", {"model.lm.blocks.0.w"})
        == "model.lm.blocks.0.w"
    )
    assert (
        module._resolve_model_checkpoint_key("module.vision.w", {"module.blocks.0.w.main"}) is None
    )
    assert (
        module._resolve_optimizer_checkpoint_key(
            "module.lm.blocks.0.w.main", {"module.blocks.0.w.main"}
        )
        == "module.blocks.0.w.main"
    )
    assert module._resolve_optimizer_checkpoint_key("module.vision.w.main", set()) is None
    assert module._allow_missing_optimizer_checkpoint_key("module.vision.w.main")
    assert not module._allow_missing_optimizer_checkpoint_key("module.lm.blocks.0.w.main")


def test_genuinely_missing_lm_state_is_left_for_the_loader_to_reject():
    module, _, _, _ = _module({"module.lm.blocks.0.w": True})
    state = {"module.lm.blocks.0.w.main": torch.zeros(1)}
    to_load = module._optimizer_state_dict_for_load(state, {"something.else"})
    assert to_load == state  # untouched, so the distributed loader reports the missing key


def test_fresh_components_are_deferred_only_on_masters_only_loads():
    module, _, _, _ = _module({"module.lm.blocks.0.w": True, "module.connector.w": True})
    checkpoint_keys = {"module.lm.blocks.0.w.main", "module.lm.blocks.0.w.exp_avg"}
    # A full resume loads the moments too: a missing connector entry stays in the dict, so the
    # distributed loader reports it instead of silently keeping fresh state.
    state = {
        "module.lm.blocks.0.w.main": torch.zeros(1),
        "module.lm.blocks.0.w.exp_avg": torch.zeros(1),
        "module.connector.w.main": torch.zeros(1),
        "module.connector.w.exp_avg": torch.zeros(1),
    }
    to_load = module._optimizer_state_dict_for_load(dict(state), checkpoint_keys)
    assert set(to_load) == set(state)
    assert module._load_deferred == {}
    # A masters-only load (model-only handoff or moment reset) defers them.
    masters = {k: v for k, v in state.items() if k.endswith(".main")}
    masters["__moe_skip_step_losses"] = []
    to_load = module._optimizer_state_dict_for_load(dict(masters), checkpoint_keys)
    assert set(to_load) == {"module.lm.blocks.0.w.main", "__moe_skip_step_losses"}
    assert set(module._load_deferred) == {"module.connector.w.main"}


def test_shared_qk_gains_are_expanded_under_native_text_keys():
    module, named, optim, _ = _module({"module.lm.blocks.0.attention.q_norm.weight": True})
    module.expand_shared_qk_norm_on_load = True
    param = named["module.lm.blocks.0.attention.q_norm.weight"]  # (2, 2): two heads of two
    master = torch.zeros(4)
    checkpoint_key = "module.blocks.0.attention.q_norm.weight.main"
    module._load_metadata = Metadata(
        state_dict_metadata={
            checkpoint_key: TensorStorageMetadata(
                properties=TensorProperties(dtype=torch.float32), size=torch.Size([2]), chunks=[]
            )
        }
    )
    optim.param_groups = [{"pg": "dp", "named_params": dict(named)}]

    to_load = module._optimizer_state_dict_for_load(
        {"module.lm.blocks.0.attention.q_norm.weight.main": master}, {checkpoint_key}
    )

    # The aliased entry became a tiny shared-vector destination keyed by the checkpoint key.
    assert set(to_load) == {checkpoint_key}
    assert to_load[checkpoint_key].shape == (2,)
    assert set(module._load_expansions) == {checkpoint_key}
    to_load[checkpoint_key].copy_(torch.tensor([1.0, 2.0]))  # what the loader would read

    after = module._optimizer_state_dict_after_load(to_load)

    assert set(after) == {"module.lm.blocks.0.attention.q_norm.weight.main"}
    expanded = after["module.lm.blocks.0.attention.q_norm.weight.main"]
    assert expanded is master
    torch.testing.assert_close(expanded, torch.tensor([1.0, 2.0, 1.0, 2.0]))
    assert tuple(param.shape) == (2, 2)
    assert module._load_expansions == {}
