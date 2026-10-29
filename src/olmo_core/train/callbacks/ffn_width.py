import logging
from dataclasses import dataclass

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["FFNWidthCallback"]


@dataclass
class FFNWidthCallback(Callback):
    """
    Drives partial-MLP training (:mod:`olmo_core.nn.ffn_width`): before each step it hands the holder the global step,
    so every rank, every micro-batch of the step and the activation-checkpoint recompute use the same neuron subset.
    Records ``ffn_width/frac`` (mean realised fraction over the partial layers).
    """

    def _holder(self):
        return getattr(self.trainer.train_module.model, "_ffn_width", None)  # type: ignore[union-attr]

    def pre_step(self, batch):
        h = self._holder()
        if h is not None:
            h.set_step(int(self.step))

    def post_step(self):
        h = self._holder()
        if h is not None and h.layers:
            self.trainer.record_metric("ffn_width/frac", sum(h.frac_of(i) for i in h.layers) / len(h.layers))
