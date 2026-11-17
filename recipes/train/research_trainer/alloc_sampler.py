"""Loss-allocated row sampling and per-row loss normalisation (early_loss_proxy idea ``alloc``).

Opt-in from ``train_ctc_suite.py`` (``--alloc-mode`` / ``--loss-row-pow``); every flag defaults off.

* Cells are (task, rung) from the pool's ``row_info.jsonl`` (global row index = part offset + row; parts
  in the glob order of ``token_ids_part_*.npy``, i.e. t0..t3).
* ``get_instance(idx)`` is replaced: the position ``idx`` in the loader's shuffled order only seeds a
  draw. A cell is drawn with probability ``p_c``; the row is the next unused row of that cell on this
  DP rank (rank r takes rows r, r+W, ... of a seeded per-cell shuffle; wraps if a cell runs out).
* ``p_c = (1 - floor) * q_c + floor * pi_c`` with ``pi_c`` the pool share of cell c and
  ``q_c ∝ pi_c * w_task(c)`` (static) or ``q_c ∝ pi_c * EMA_c^a / L_c^b`` (ema), ``EMA_c`` the EMA of
  the per-row TRAINING CE (routed forward, labels only; no extra pass) all-gathered over ranks each
  step, ``L_c`` the mean row length of the cell. a=b=0 reproduces the pool composition.
* FLOPs: nothing extra is run; the flop meter prices each row at its own length as before.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from olmo_core.train.callbacks.callback import Callback


class CellSampler:
    def __init__(self, data_dir: str, mode: str, a: float, b: float, floor: float, ema: float,
                 task_weights: str, seed: int, rank: int, world: int, warmup_steps: int,
                 rung_weights: str = "", short_share: float = -1.0, short_rungs: str = "1k,2k"):
        rows = [json.loads(l) for l in open(os.path.join(data_dir, "row_info.jsonl"))]
        parts = sorted(glob.glob(os.path.join(data_dir, "token_ids_part_*.npy")))
        # rows per part from row_info (row_info covers every row)
        per_part: Dict[int, int] = defaultdict(int)
        for r in rows:
            per_part[int(r["part"])] = max(per_part[int(r["part"])], int(r["row"]) + 1)
        assert len(per_part) == len(parts), (len(per_part), len(parts))
        off, acc = {}, 0
        for k in sorted(per_part):
            off[k] = acc
            acc += per_part[k]
        self.n_rows = acc
        self.cell_names: List[str] = []
        cid: Dict[str, int] = {}
        self.row_cell = np.full(acc, -1, dtype=np.int64)
        self.row_len = np.zeros(acc, dtype=np.int64)
        for r in rows:
            name = f"{r['task']}@{r['rung']}"
            if name not in cid:
                cid[name] = len(self.cell_names)
                self.cell_names.append(name)
            g = off[int(r["part"])] + int(r["row"])
            self.row_cell[g] = cid[name]
            self.row_len[g] = int(r["len"])
        assert (self.row_cell >= 0).all()
        C = len(self.cell_names)
        self.cell_rows = []
        for c in range(C):
            ids = np.nonzero(self.row_cell == c)[0].tolist()
            random.Random(f"alloc:{seed}:{c}").shuffle(ids)
            self.cell_rows.append(ids)
        self.pi = np.array([len(x) for x in self.cell_rows], dtype=np.float64)
        self.pi /= self.pi.sum()
        self.L = np.array([self.row_len[self.row_cell == c].mean() for c in range(C)])
        self.mode, self.a, self.b, self.floor, self.ema_beta = mode, a, b, floor, ema
        self.warmup_steps = warmup_steps
        tw = dict((kv.split("=")[0], float(kv.split("=")[1])) for kv in task_weights.split(",") if kv)
        rw = dict((kv.split("=")[0], float(kv.split("=")[1])) for kv in rung_weights.split(",") if kv)
        self.tw = np.array([tw.get(n.split("@")[0], 1.0) * rw.get(n.split("@")[1], 1.0)
                            for n in self.cell_names])
        sr = set(short_rungs.split(","))
        self.is_short = np.array([n.split("@")[1] in sr for n in self.cell_names])
        self.short_share = short_share
        self.ema = np.full(C, np.nan)
        self.seed, self.rank, self.world = seed, rank, world
        self.counter = [0] * C
        self.wraps = [0] * C
        self.drawn = [0] * C
        self.lock = threading.Lock()
        self.step = 0
        self.p = self._compute_p()

    def _compute_p(self) -> np.ndarray:
        if self.mode == "static":
            q = self.pi * self.tw
        elif self.mode == "ema" and self.step >= self.warmup_steps and not np.isnan(self.ema).all():
            e = np.where(np.isnan(self.ema), np.nanmax(self.ema), self.ema)
            q = self.pi * np.power(np.maximum(e, 1e-6), self.a) / np.power(self.L, self.b)
        else:
            q = self.pi.copy()
        q = q / q.sum()
        p = (1.0 - self.floor) * q + self.floor * self.pi
        p = p / p.sum()
        if self.short_share >= 0 and self.is_short.any() and (~self.is_short).any():
            # fixed minority of short cells (pool composition among them); the rule above sets the long cells
            ps = np.where(self.is_short, self.pi, 0.0)
            pl = np.where(self.is_short, 0.0, p)
            p = self.short_share * ps / ps.sum() + (1.0 - self.short_share) * pl / pl.sum()
        return p

    def draw(self, idx: int) -> int:
        u = random.Random(f"alloc-draw:{self.seed}:{self.rank}:{idx}").random()
        with self.lock:
            c = int(min(np.searchsorted(np.cumsum(self.p), u, side="right"), len(self.p) - 1))
            rows = self.cell_rows[c]
            k = self.counter[c] * self.world + self.rank
            self.counter[c] += 1
            self.drawn[c] += 1
            if k >= len(rows):
                self.wraps[c] += 1
            return rows[k % len(rows)]

    def update(self, cell_sum: np.ndarray, cell_n: np.ndarray):
        """cell_sum / cell_n: per-cell sum of per-row mean CE and number of rows this step (global)."""
        for c in np.nonzero(cell_n)[0]:
            m = cell_sum[c] / cell_n[c]
            self.ema[c] = m if np.isnan(self.ema[c]) else self.ema_beta * self.ema[c] + (1 - self.ema_beta) * m
        self.step += 1
        with self.lock:
            self.p = self._compute_p()


def install_sampler(data_loader, sampler: CellSampler) -> None:
    orig = data_loader.get_instance

    def get_instance(idx: int) -> Dict[str, Any]:
        return orig(sampler.draw(int(idx)))

    data_loader.get_instance = get_instance  # type: ignore[method-assign]


class AllocCallback(Callback):
    """Reads the per-row training CE the train module captured, all-gathers it, updates the sampler's
    per-cell EMA (when a sampler is given) and logs per-cell stats."""

    def __init__(self, sampler: Optional[CellSampler], row_cell: Optional[np.ndarray], n_cells: int,
                 cell_names: List[str], log_every: int = 10):
        self.sampler = sampler
        self.row_cell = row_cell
        self.n_cells = n_cells
        self.cell_names = cell_names
        self.log_every = log_every
        self.tot_n = np.zeros(max(n_cells, 1))

    def post_step(self):
        tm = self.trainer.train_module
        ra = getattr(tm, "_row_alloc", None)
        if ra is None:
            return
        local = []
        for idx, sums, cnt, n_calls in ra.get("rows", []):
            for i, s, n in zip(idx.tolist(), sums.tolist(), cnt.tolist()):
                local.append((int(i), float(s), int(n), int(n_calls)))
        if dist.is_initialized():
            gathered: List[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            allrows = [r for g in gathered for r in g]
        else:
            allrows = local
        if self.row_cell is None:
            return
        cs = np.zeros(self.n_cells)
        cn = np.zeros(self.n_cells)
        for i, s, n, _ in allrows:
            if n <= 0 or i < 0 or i >= len(self.row_cell):
                continue
            c = self.row_cell[i]
            cs[c] += s / n
            cn[c] += 1
        self.tot_n += cn
        if self.sampler is not None:
            self.sampler.update(cs, cn)
        rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if rank0 and (self.step <= 3 or self.step % self.log_every == 0):
            calls = sorted({r[3] for r in allrows})
            msg = " ".join(
                f"{self.cell_names[c]}:n{int(self.tot_n[c])}"
                + (f",ema{self.sampler.ema[c]:.3f},p{self.sampler.p[c]:.3f}" if self.sampler is not None else "")
                for c in range(self.n_cells)
            )
            extra = ""
            if self.sampler is not None and sum(self.sampler.wraps):
                extra = f" wraps={self.sampler.wraps}"
            print(f"[alloc] step {self.step} rows {len(allrows)} head_calls {calls} {msg}{extra}", flush=True)
