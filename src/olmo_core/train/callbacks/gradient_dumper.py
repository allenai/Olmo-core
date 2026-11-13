import logging
from dataclasses import dataclass

import torch

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GradientDumperCallback(Callback):
    enabled: bool = True

    def pre_optim_step(self):
        if not self.enabled:
            return

        output_dir = self.trainer.work_dir / "grad_dumper"
        output_dir.mkdir(exist_ok=True, parents=True)

        assert hasattr(self.trainer.train_module, "model")
        for name, p in self.trainer.train_module.model.named_parameters():
            if p.grad is None:
                continue

            filename = f"rank{get_rank()}_step{self.step}_{name}.pt"
            filepath = output_dir / filename
            log.info(f"Saving gradient of '{name}' to '{filepath}'")
            torch.save(p.grad.cpu(), filepath)
