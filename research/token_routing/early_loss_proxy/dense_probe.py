"""Early dense-loss proxy: train a method for a few steps and score the DENSE forward on fixed held-out rows.

Runs the unmodified research trainer (``recipes/train/research_trainer/train_ctc_suite.py``: the method's own routed /
dropped forward, its optimiser, its real LR schedule over the full 488-step run) and attaches one callback,
``DenseProbeCallback``. At step 0 (after the base load, before any update) and after each step in ``--probe-steps``
it puts the model in eval mode -- every routing / compaction / drop path in ``Transformer.forward`` and the online
router is gated on ``self.training``, so this is the full-context dense forward the eval uses, LoRA adapters included
(they are ordinary submodules) -- and computes the teacher-forced answer CE of each held-out row (B = 1, unpadded,
default position ids 0..T-1; the rows carry the training shards' document markers). The trainer hard-stops at the
last probe step; nothing is saved.

    torchrun ... dense_probe.py --probe-steps 1,2,5,10 --probe-sets suite8k:16,suite2k:16,train:16 \\
        --probe-heldout /data/prasann/early_loss_proxy/heldout_v1.npz --probe-out <json> -- <train_ctc_suite args>

Output JSON (rank 0, rewritten after every probe): per probe step the per-row CE (row-mean over answer tokens), the
LR of every param group, cumulative FLOPs (flop meter), wall-clock of each phase, and a hash of each step's rank-0
batch (to check the data order is identical across arms).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List

T_START = time.time()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _pop_args(argv: List[str]):
    """Split our --probe-* flags from the trainer's argv (everything after an optional ``--`` is the trainer's)."""
    ours: Dict[str, str] = {}
    rest: List[str] = []
    i = 0
    while i < len(argv):
        x = argv[i]
        if x == "--":
            rest += argv[i + 1:]
            break
        if x.startswith("--probe-"):
            ours[x[len("--probe-"):].replace("-", "_")] = argv[i + 1]
            i += 2
            continue
        rest.append(x)
        i += 1
    return ours, rest


def build_callback(cfg: Dict[str, str]):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from olmo_core.train.callbacks.callback import Callback

    steps = sorted({int(s) for s in cfg.get("steps", "1,2,5,10").split(",") if s})
    z = np.load(cfg["heldout"])
    offs, ids_all, mask_all = z["offsets"], z["ids"], z["mask"]
    sets_arr, tasks_arr = z["sets"], z["tasks"]
    rows = []  # (set, task, ids, mask)
    for spec in cfg.get("sets", "suite8k:16").split(","):
        name, _, k = spec.partition(":")
        k = int(k or 16)
        for task in ("contradiction", "nq", "outlier", "oolong"):
            idx = [i for i in range(len(sets_arr)) if sets_arr[i] == name and tasks_arr[i] == task][:k]
            if len(idx) < k:
                raise SystemExit(f"held-out {name}/{task}: only {len(idx)} rows < {k}")
            for i in idx:
                rows.append((name, task, ids_all[offs[i]:offs[i + 1]].astype(np.int64),
                             mask_all[offs[i]:offs[i + 1]].astype(bool)))
    out_path = cfg["out"]

    class DenseProbeCallback(Callback):
        """See the module docstring."""

        priority = -100  # after every other post_step (router fit, flop meter)

        def __init__(self):
            self.res: Dict[str, Any] = {
                "probe_steps": [0] + steps, "rows": [{"set": s, "task": t, "len": int(len(r)),
                                                      "n_ans": int(m.sum())} for s, t, r, m in rows],
                "probes": [], "steps": [], "argv": sys.argv, "t_start": T_START}
            self._t_step = None
            self._train_s = 0.0

        def _rank(self):
            return dist.get_rank() if dist.is_initialized() else 0

        def _ws(self):
            return dist.get_world_size() if dist.is_initialized() else 1

        def _lrs(self):
            opt = self.trainer.train_module.optim
            return [float(g["lr"]) for g in opt.param_groups]

        def _flops(self):
            fm = self.trainer.callbacks.get("flop_meter")
            if fm is None:
                return None
            return {"actual_pf": float(getattr(fm, "_actual", 0.0)) / 1e15,
                    "dense_pf": float(getattr(fm, "_dense", 0.0)) / 1e15}

        @torch.no_grad()
        def _probe(self, step: int):
            model = self.trainer.train_module.model
            torch.cuda.synchronize()
            t0 = time.time()
            was = model.training
            model.eval()
            rank, ws = self._rank(), self._ws()
            n = len(rows)
            per = -(-n // ws)
            ce = torch.zeros(n, device="cuda", dtype=torch.float64)
            ce_sum = torch.zeros(n, device="cuda", dtype=torch.float64)
            # every rank runs the same number of forwards (FSDP all-gathers per forward); a rank's spare
            # forwards re-score its last row and are discarded
            for j in range(per):
                i = rank + j * ws
                real = i < n
                i = i if real else (n - 1)
                _, _, r, m = rows[i]
                x = torch.as_tensor(r[None], device="cuda")
                ans = torch.as_tensor(np.nonzero(m)[0], device="cuda")
                lg = model(x, logits_to_keep=(ans - 1)[None])
                lg = lg[0] if not isinstance(lg, torch.Tensor) or lg.dim() == 3 else lg
                l = F.cross_entropy(lg.float(), x[0, ans], reduction="none")
                if real:
                    ce[i] = l.mean().double()
                    ce_sum[i] = l.sum().double()
            if dist.is_initialized():
                dist.all_reduce(ce)
                dist.all_reduce(ce_sum)
            model.train(was)
            torch.cuda.synchronize()
            dt = time.time() - t0
            rec = {"step": step, "ce": ce.cpu().tolist(), "ce_sum": ce_sum.cpu().tolist(), "lr": self._lrs(),
                   "flops": self._flops(), "probe_s": dt, "train_s_cum": self._train_s,
                   "wall_s": time.time() - T_START}
            self.res["probes"].append(rec)
            if rank == 0:
                by: Dict[str, List[float]] = {}
                for (s, t, _, _), c in zip(rows, rec["ce"]):
                    by.setdefault(f"{s}/{t}", []).append(c)
                msg = " ".join(f"{k}={np.mean(v):.4f}" for k, v in by.items())
                print(f"[dense-probe] step {step} lr {rec['lr'][0]:.3g} probe {dt:.1f}s train_cum "
                      f"{self._train_s:.1f}s wall {rec['wall_s']:.0f}s | {msg}", flush=True)
                self._write()

        def _write(self):
            tmp = out_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.res, f)
            os.replace(tmp, out_path)

        def pre_train(self):
            self.res["t_pre_train"] = time.time() - T_START  # process start -> model built + base loaded
            self._probe(0)

        def pre_step(self, batch: Dict[str, Any]):
            torch.cuda.synchronize()
            self._t_step = time.time()
            ii = batch["input_ids"]
            self._hash = hashlib.sha1(ii.detach().cpu().numpy().tobytes()).hexdigest()[:12]
            self._ntok = int(ii.numel())

        def post_step(self):
            torch.cuda.synchronize()
            dt = time.time() - self._t_step
            self._train_s += dt
            self.res["steps"].append({"step": int(self.step), "train_s": dt, "lr": self._lrs(),
                                      "batch_hash_rank0": self._hash, "tokens_rank0": self._ntok,
                                      "flops": self._flops()})
            if self.step in steps:
                self._probe(int(self.step))
            elif self._rank() == 0:
                self._write()

    return DenseProbeCallback()


def main():
    cfg, rest = _pop_args(sys.argv[1:])
    steps = sorted({int(s) for s in cfg.get("steps", "1,2,5,10").split(",") if s})
    if "--max-steps" not in rest:
        rest += ["--max-steps", str(max(steps))]
    sys.argv = [os.path.join(REPO, "recipes/train/research_trainer/train_ctc_suite.py")] + rest
    sys.path.insert(0, os.path.join(REPO, "recipes", "train", "research_trainer"))
    sys.path.insert(0, os.path.join(REPO, "recipes", "train"))
    import train_ctc_suite as T  # noqa: E402
    from olmo_core.train import TrainerConfig  # noqa: E402

    orig = TrainerConfig.build

    def build(self, *a, **k):
        tr = orig(self, *a, **k)
        tr.add_callback("dense_probe", build_callback(cfg))
        return tr

    TrainerConfig.build = build
    T.main()


if __name__ == "__main__":
    main()
