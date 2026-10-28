import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["GradLayerSelectCallback"]


@dataclass
class GradLayerSelectCallback(Callback):
    """
    Drives :mod:`olmo_core.nn.grad_layer_select`: before each step it selects the blocks (same on every rank) and, in
    update mode, turns ``requires_grad`` off for the unselected blocks' trainable params; after backward it measures
    each block's gradient energy (``sum ||g||^2`` over its trainable params, summed over ranks) into the EMA. Logs
    ``grad_select/*`` every step, one ``[grad-select]`` line per step (the selection), and writes
    ``grad_select.json`` (per-block selection frequency, final EMA, per-step history) at the end.
    """

    log_every: int = 1
    """Steps between ``[grad-select]`` console lines (0 disables)."""

    _trainable: Optional[Dict[int, List[str]]] = None

    def _sel(self):
        return getattr(self.trainer.train_module.model, "_grad_select", None)  # type: ignore[union-attr]

    def _blocks(self):
        return list(self.trainer.train_module.model.blocks.values())  # type: ignore[union-attr]

    def _params(self, i: int):
        names = set(self._trainable[i])  # type: ignore[index]
        return [p for n, p in self._blocks()[i].named_parameters() if n in names]

    def _record_trainable(self):
        if self._trainable is None:
            self._trainable = {
                i: [n for n, p in blk.named_parameters() if p.requires_grad] for i, blk in enumerate(self._blocks())
            }

    def pre_train(self):
        if self._sel() is not None:
            self._record_trainable()

    def pre_step(self, batch):
        sel = self._sel()
        if sel is None:
            return
        self._record_trainable()
        sel.begin_step(int(self.step))
        if sel.mode == "update":
            for i, on in sel.train_mask().items():
                for p in self._params(i):
                    p.requires_grad_(on)

    def pre_optim_step(self):
        sel = self._sel()
        if sel is None:
            return
        import torch
        import torch.distributed as dist

        n = sel.n_layers
        dev = next(self.trainer.train_module.model.parameters()).device  # type: ignore[union-attr]
        v = torch.zeros(2 * n, dtype=torch.float64, device=dev)
        for i in range(n):
            for p in self._params(i):
                g = p.grad
                if g is None:
                    continue
                g = g.to_local() if hasattr(g, "to_local") else g
                v[i] += g.detach().float().pow(2).sum().double()
                v[n + i] = 1.0
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
        energy = {i: float(v[i]) for i in range(n) if float(v[n + i]) > 0}
        sel.update_energies(energy)
        if energy:
            tot = sum(energy.values())
            self.trainer.record_metric("grad_select/energy_measured", tot)

    def post_step(self):
        sel = self._sel()
        if sel is None:
            return
        chosen = sorted(sel.selected())
        self.trainer.record_metric("grad_select/n_selected", float(len(chosen)))
        self.trainer.record_metric("grad_select/cut", float(sel.cut()))
        self.trainer.record_metric("grad_select/mean_selected_index", sum(chosen) / max(1, len(chosen)))
        if self.log_every > 0 and self.step % self.log_every == 0:
            log.info("[grad-select] step %d: selected %s cut %d", self.step, chosen, sel.cut())
        if self.step % 50 == 0:
            s = sel.summary()
            log.info("[grad-select] step %d selected_frac %s", self.step, [round(x, 3) for x in s["selected_frac"]])

    def post_train(self):
        sel = self._sel()
        if sel is None:
            return
        if sel.mode == "update" and self._trainable is not None:
            for i in range(sel.n_layers):
                for p in self._params(i):
                    p.requires_grad_(True)
        s = sel.summary()
        log.info("[grad-select] final selected_frac %s", [round(x, 3) for x in s["selected_frac"]])
        log.info("[grad-select] final ema %s", [None if e is None else float(f"{e:.4g}") for e in s["ema"]])
        if get_rank() == 0:
            path = os.path.join(str(self.trainer.save_folder), "grad_select.json")
            try:
                with open(path, "w") as f:
                    json.dump(s, f)
            except OSError as e:  # remote save folders
                log.warning("[grad-select] could not write %s: %s", path, e)
