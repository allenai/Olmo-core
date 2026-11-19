"""Selective backward (``--selbp-keep p``): run every micro-batch's forward, then skip the backward of the ones the
model already fits.

Granularity: one synchronized micro-step = the micro-batch at the same index on every data-parallel rank (2 rows per
GPU x 2 GPUs = 4 rows in the proxy). It is not per row because (a) the rows of a micro-batch share one autograd
graph, so dropping one row's backward needs a row mask that still backpropagates zeros through the full-size matmuls
(no compute saved), and (b) under FSDP every backward runs collectives (all-gather of the frozen base for the
activation gradient, reduce-scatter of the adapter gradients), so all ranks must take the same backward / skip
decision at each micro-step or the job hangs. The decision is made after the forward, so a skipped micro-step never
builds anything beyond the forward graph (which is freed): its backward compute is really saved.

Score of a micro-step = the answer CE of its routed forward per supervised token, summed over ranks (one all-reduce
of two numbers). Kept micro-steps are not reweighted (the loss divisor stays the step's full token count).

Modes:
- ``quantile``: keep if the score is >= the (1 - p) quantile of the last ``window`` rows' worth of micro-step scores.
- ``sample``: keep with probability min(1, c * score / mean(recent scores)), c solved on the recent window so the
  mean probability is p (softer, lower bias than the hard threshold).
The backward-FLOP-weighted share actually kept is corrected toward p (effective share p + (p - recent realized
share)), so PF/step ~ forward + p x backward. Warm-up: everything is kept for the first ``warmup_steps`` steps (the window fills meanwhile). ``sched=(a, b)``: p
moves linearly from a to b over training progress. If no micro-step of a step was kept by its last micro-step, the
last one is kept (forced) so every optimizer step has a gradient; forced keeps are counted.

FLOPs: the online router prices each training forward twice (as charged, forward + backward, and forward-only) and
accumulates the difference in ``model._selbp_fwd_delta``; a skipped micro-step adds that difference to
``model._flop_extra_actual``, so the FLOP meter charges it forward-only.
"""
import math
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

__all__ = ["SelectiveBackward"]


class SelectiveBackward:
    def __init__(self, keep: float, mode: str = "quantile", sched: Optional[Tuple[float, float]] = None,
                 window_rows: int = 512, warmup_steps: int = 20, key_tokens: int = 22, seed: int = 0,
                 log_every: int = 10):
        if not (0.0 < keep <= 1.0):
            raise ValueError(f"--selbp-keep must be in (0, 1], got {keep}")
        if mode not in ("quantile", "sample"):
            raise ValueError(f"--selbp-mode must be quantile or sample, got {mode}")
        self.keep, self.mode, self.sched = float(keep), mode, sched
        self.window_rows, self.warmup_steps, self.key_tokens = int(window_rows), int(warmup_steps), int(key_tokens)
        self.log_every = int(log_every)
        self.rng = random.Random(1000003 + int(seed))  # same sequence on every rank (decisions are identical)
        self.hist: Optional[deque] = None
        self.dec: Optional[deque] = None  # recent post-warm-up (keep, backward cost), for the correction
        self._p_eff = keep
        self._fsdp_mods: Optional[List[Any]] = None
        self._step = -1
        self._step_kept = 0
        self._step_groups = 0
        self._step_saved = 0.0
        self.tot_groups = self.tot_kept = self.tot_forced = 0
        self.task_seen: Dict[Tuple[int, ...], int] = {}
        self.task_kept: Dict[Tuple[int, ...], int] = {}

    # ------------------------------------------------------------------ schedule
    def p_now(self, step: int, max_steps: Optional[int]) -> float:
        if self.sched is None:
            return self.keep
        a, b = self.sched
        t = min(1.0, max(0.0, step / float(max_steps))) if max_steps else 0.0
        return a + (b - a) * t

    def _threshold(self, p: float) -> float:
        h = sorted(self.hist)
        q = (1.0 - p) * (len(h) - 1)
        lo, hi = int(math.floor(q)), int(math.ceil(q))
        return h[lo] + (h[hi] - h[lo]) * (q - lo)

    def _sample_prob(self, score: float, p: float) -> float:
        h = list(self.hist)
        m = sum(h) / len(h)
        if m <= 0:
            return p
        r = [x / m for x in h]
        lo, hi = 0.0, 1e4
        for _ in range(60):  # mean(min(1, c * r)) = p
            c = 0.5 * (lo + hi)
            if sum(min(1.0, c * x) for x in r) / len(r) < p:
                lo = c
            else:
                hi = c
        return min(1.0, hi * score / m)

    # ------------------------------------------------------------------ per micro-step
    def begin_mb(self, model) -> None:
        model._selbp_fwd_delta = 0.0

    def decide(self, tm, input_ids: torch.Tensor, labels: torch.Tensor, mb_ce: torch.Tensor,
               div_factor: torch.Tensor, ignore_index: int, is_last_mb: bool, dry_run: bool) -> bool:
        """Called on every rank right after a micro-batch's forward; returns whether to run its backward."""
        if dry_run:
            return True
        trainer = tm.trainer
        step = int(trainer.global_step)
        if step != self._step:
            self._step, self._step_kept, self._step_groups, self._step_saved = step, 0, 0, 0.0
        dev = mb_ce.device
        x = torch.stack([(mb_ce.double().reshape(()) * torch.as_tensor(div_factor).to(dev).double().reshape(())),
                         (labels != ignore_index).sum().to(dev).double(),
                         torch.tensor(float(input_ids.shape[0]), device=dev, dtype=torch.float64),
                         torch.tensor(-float(getattr(tm.model, "_selbp_fwd_delta", 0.0) or 0.0), device=dev,
                                      dtype=torch.float64)])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x)
        ce_sum, ntok, nrows, bwd_cost = x.tolist()  # summed over ranks: identical decisions everywhere
        score = ce_sum / max(1.0, ntok)
        if self.hist is None:
            self.hist = deque(maxlen=max(8, int(round(self.window_rows / max(1.0, nrows)))))
            self.dec = deque(maxlen=self.hist.maxlen)
        p = self.p_now(step, trainer.max_steps)
        forced = False
        if step <= self.warmup_steps or p >= 1.0 or len(self.hist) < 8:
            keep = True
        else:
            # keep-rate correction: the loss falls while the window lags, so a fixed quantile keeps fewer than p;
            # move the effective share by the recent shortfall (p + (p - realized)), clipped
            # realized share is cost-weighted (each micro-step weighted by its backward FLOPs), so PF/step tracks
            # forward + p x backward
            pe = p
            if self.dec is not None and len(self.dec) >= 8:
                tot = sum(c for _, c in self.dec)
                real = sum(c for k, c in self.dec if k) / tot if tot > 0 else p
                pe = min(1.0, max(0.02, p + (p - real)))
            self._p_eff = pe
            if pe >= 1.0:
                keep = True
            elif self.mode == "quantile":
                keep = score >= self._threshold(pe)
            else:
                keep = self.rng.random() < self._sample_prob(score, pe)
        if not keep and is_last_mb and self._step_kept == 0:
            keep = forced = True
        if step > self.warmup_steps and p < 1.0 and self.dec is not None:
            self.dec.append((keep, max(0.0, bwd_cost)))  # forced keeps count toward the realized share
        self.hist.append(score)
        self._step_groups += 1
        self._step_kept += int(keep)
        self.tot_groups += 1
        self.tot_kept += int(keep)
        self.tot_forced += int(forced)
        n = min(self.key_tokens, int(input_ids.shape[1]))
        for row in input_ids[:, :n].tolist():
            k = tuple(row)
            self.task_seen[k] = self.task_seen.get(k, 0) + 1
            self.task_kept[k] = self.task_kept.get(k, 0) + int(keep)
        if keep:
            model = tm.model
            from torch.distributed.fsdp import FSDPModule

            if isinstance(model, FSDPModule):
                # a later micro-batch may be skipped, so every backward that runs finalizes (waits on its gradient
                # reduction and clears the backward-prefetch order)
                model.set_is_last_backward(True)
        if is_last_mb:
            self._end_step(tm, step, p)
        return keep

    def skipped(self, model) -> None:
        """A micro-step whose backward is skipped: charge it forward-only and reset FSDP's post-forward state."""
        d = float(getattr(model, "_selbp_fwd_delta", 0.0) or 0.0)
        model._flop_extra_actual = float(getattr(model, "_flop_extra_actual", 0.0)) + d
        self._step_saved += -d
        if self._fsdp_mods is None:
            from torch.distributed.fsdp import FSDPModule

            self._fsdp_mods = [m for m in model.modules() if isinstance(m, FSDPModule)]
        for m in self._fsdp_mods:
            m.reshard()  # modules with reshard_after_forward=False (the LM head) would otherwise stay unsharded
            st = m._get_fsdp_state()
            pg = st._fsdp_param_group
            if pg is not None:
                pg._post_forward_indices.clear()
            st._comm_ctx.post_forward_order.clear()

    # ------------------------------------------------------------------ logging
    def _end_step(self, tm, step: int, p: float) -> None:
        from olmo_core.train.common import ReduceType

        g = max(1, self._step_groups)
        tm.record_metric("selbp/kept_frac", self._step_kept / g, ReduceType.max, namespace="train")
        tm.record_metric("selbp/p_target", p, ReduceType.max, namespace="train")
        saved = torch.tensor([self._step_saved], dtype=torch.float64,
                             device=next(tm.model.parameters()).device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(saved)
        tm.record_metric("selbp/saved_pf", float(saved.item()) / 1e15, ReduceType.max, namespace="train")
        if step % self.log_every and step > 3:
            return
        rows = [(self.task_seen, self.task_kept)]
        if dist.is_available() and dist.is_initialized():
            rows = [None] * dist.get_world_size()  # type: ignore[list-item]
            dist.all_gather_object(rows, (self.task_seen, self.task_kept))
        if not dist.is_initialized() or dist.get_rank() == 0:
            seen: Dict[Tuple[int, ...], int] = {}
            kept: Dict[Tuple[int, ...], int] = {}
            for s, k in rows:  # type: ignore[misc]
                for key, v in s.items():
                    seen[key] = seen.get(key, 0) + v
                for key, v in k.items():
                    kept[key] = kept.get(key, 0) + v
            per = {",".join(map(str, key)): [kept.get(key, 0), v] for key, v in sorted(seen.items(), key=lambda t: -t[1])}
            print(f"[selbp] step {step} p={p:.3f} p_eff={self._p_eff:.3f} step_kept={self._step_kept}/{self._step_groups} "
                  f"saved_pf={float(saved.item()) / 1e15:.4f} run_kept={self.tot_kept}/{self.tot_groups} "
                  f"forced={self.tot_forced} thr_hist={len(self.hist or [])} per_task_kept_seen={per}", flush=True)
