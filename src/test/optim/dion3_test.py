# Patch torch.compiler.disable before olmo_core imports execute
import torch
if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    original_disable = torch.compiler.disable

    def patched_disable(*args, **kwargs):
        kwargs.pop("reason", None)
        return original_disable(*args, **kwargs)

    torch.compiler.disable = patched_disable  # type: ignore[assignment]

from typing import Any, cast
import pytest
import tempfile
import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.optim.dion import Dion3Config
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.testing import DEVICES
from olmo_core.testing.utils import requires_dion
from olmo_core.train import Trainer
from olmo_core.utils import seed_all



class _FakeTrainer:
    """Minimal stand-in for Trainer, just enough for scheduler.set_lr."""

    def __init__(self, global_step: int, max_steps: int):
        self.global_step = global_step
        self.max_steps = max_steps
        self.global_train_tokens_seen = None


def _as_trainer(fake: _FakeTrainer) -> Trainer:
    return cast(Trainer, cast(Any, fake))


# =====================================================================
# 1. Config Verification
# =====================================================================
@requires_dion
def test_dion3_config_builds():
    from dion import Dion3  # type: ignore[reportMissingImports]

    config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)
    model = config.build()
    optim_cfg = Dion3Config()
    optim = optim_cfg.build(model)

    assert isinstance(optim, Dion3)
    assert len(optim.param_groups) == 4
    for group in optim.param_groups:
        assert "pristine_lr" in group


# =====================================================================
# 2. LR Continuity Across Checkpoint Save/Load (Cold Start)
# =====================================================================
@requires_dion
@pytest.mark.parametrize("device", DEVICES)
def test_dion3_lr_survives_checkpoint_recovery(device: torch.device):
    seed_all(0)
    model_config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)

    def build():
        model = model_config.build().train().to(device)
        optim = Dion3Config().build(model)
        scheduler = CosWithWarmup(warmup_steps=5)
        return model, optim, scheduler

    # --- Run A: Uninterrupted Baseline ---
    seed_all(0)
    model_a, optim_a, scheduler_a = build()
    baseline_lrs = []
    for step in range(20):
        optim_a.zero_grad(set_to_none=True)
        model_a(torch.randint(0, 1024, (2, 8), device=device)).sum().backward()
        for group in optim_a.param_groups:
            lr = scheduler_a.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
            baseline_lrs.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
        optim_a.step()

    # --- Run B: Train & Save Checkpoint ---
    seed_all(0)
    model_b, optim_b, scheduler_b = build()
    lrs_resumed = []
    for step in range(10):
        optim_b.zero_grad(set_to_none=True)
        model_b(torch.randint(0, 1024, (2, 8), device=device)).sum().backward()
        for group in optim_b.param_groups:
            lr = scheduler_b.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
            lrs_resumed.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
        optim_b.step()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save the decayed mid-run state to disk
        save_model_and_optim_state(tmp_dir, model_b, optim_b)

        # --- Run C: True Cold Start Resumption ---
        # Create fresh objects from the config, exactly like a new torchrun command
        seed_all(0)
        model_c, optim_c, scheduler_c = build()

        # Merge the decayed disk state over the pristine config dictionaries
        load_model_and_optim_state(tmp_dir, model_c, optim_c)

        for step in range(10, 20):
            optim_c.zero_grad(set_to_none=True)
            model_c(torch.randint(0, 1024, (2, 8), device=device)).sum().backward()
            for group in optim_c.param_groups:
                lr = scheduler_c.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
                lrs_resumed.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
            optim_c.step()

    assert lrs_resumed == pytest.approx(baseline_lrs), (
        "Resumed LR trajectory diverged after a cold restart! The config's pristine_lr "
        "was either overwritten by the decayed LR or wiped by the DCP schema merge."
    )