"""The wrapper hooks on :class:`OLMoDDPTrainModule` and its config are inert by default."""

import torch

from olmo_core.train.train_module.transformer import (
    ddp_train_module as train_module_impl,
)
from olmo_core.train.train_module.transformer.config import OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer.ddp_train_module import OLMoDDPTrainModule


def test_compatibility_check_accepts_opt_in_wrappers_only():
    class Plain(torch.nn.Module):
        pass

    class OptedIn(torch.nn.Module):
        _olmo_ddp_compatible = True

    assert not train_module_impl._is_olmo_ddp_compatible(Plain())
    assert train_module_impl._is_olmo_ddp_compatible(OptedIn())


def test_default_hooks_are_identity():
    module = OLMoDDPTrainModule.__new__(OLMoDDPTrainModule)
    assert module._batch_auxiliary_loss_kwargs({"input_ids": torch.zeros(1, 2)}) == {}
    state = {"module.w.main": torch.zeros(2)}
    assert module._optimizer_state_dict_for_load(state, {"module.w.main"}) is state
    assert module._optimizer_state_dict_after_load(state) is state


def test_build_hook_receives_model_device_and_eval_only():
    class Recording(OLMoDDPTrainModuleConfig):
        def _build_train_module(self, **kwargs):
            return kwargs

    from olmo_core.optim import OLMoDDPOptimizerConfig

    config = Recording(
        optim=OLMoDDPOptimizerConfig(),
        rank_microbatch_size=16,
        max_sequence_length=8,
    )
    model = torch.nn.Linear(2, 2)
    kwargs = config.build(model, device=torch.device("cpu"), eval_only=True)
    assert kwargs["model"] is model
    assert kwargs["device"] == torch.device("cpu")
    assert kwargs["eval_only"] is True
    assert kwargs["rank_microbatch_size"] == 16
