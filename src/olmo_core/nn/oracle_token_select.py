"""Oracle token selection for token-drop TRAINING: the gradient upper bound for the recipe's drop structure.

Each optimizer step runs three passes over the same batch (Prasann, 2026-11-04):

  A. dense forward + backward (every token through every block). Hooks on the LoRA-adapted linears of the blocks
     after the first cut record each token's gradient ENERGY in the trainable (LoRA) parameter space,
     e_t = s^2 (||delta_t||^2 ||A x_t||^2 + ||B^T delta_t||^2 ||x_t||^2), summed separately over the blocks of each
     stage ([7, 16) and [16, 32) for the recipe's ``7:0.2,16:0.07`` stages). The dense gradient is saved, NOT applied.
  B. the same dense forward + backward again, with each token's exact rank-1 LoRA gradient term added into the
     accumulator of its rank bin (bins = quantiles ``KS`` of the energy ranking, per allocation scope: each
     document's / row's / the whole batch's top-k share). Prefix sums of the bins give the EXACT attribution
     gradient G_S(k) of the top-k set, so cos(G_S(k), G) is read off for every grid k. The smallest grid k with
     cos >= ``target`` is chosen per stage (stage 2 from its own ranking; stage 1 = its own top-k UNION stage 2, so
     the routed sets are nested as the drop structure needs).
  C. the ROUTED forward + backward: the online router (``--st-otr-score oracle``) keeps exactly those tokens at
     each cut (non-body tokens -- prompt, markers, query, answer -- always kept). THIS gradient is applied.

Optionally every stage also keeps uniformly random extra body tokens, ``random_mult`` x the oracle set per row
(the "are negatives needed" arm). Per-step metadata goes to ``<save>/oracle_meta/steps.jsonl`` (rank 0) and the
selected token indices + energies to ``<save>/oracle_meta/sel/step<NNNNN>_rank<R>.npz`` (every rank); scalars also
go to the trainer's metrics (wandb ``oracle/*``).

FLOPs: the trainer's FLOP meter sees only pass C (the dense passes' router/LoRA charges are rolled back), i.e.
"what the routed pass alone costs"; the actual 3-pass cost is logged as ``oracle/actual_step_pflops``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = ["install_oracle", "KS"]

#: grid of body-token shares (per allocation group) at which the attribution cosine is measured
KS = (0.0005, 0.001, 0.0015, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.13, 0.17, 0.22, 0.3, 0.4, 0.55,
      0.75, 1.0)
SCOPES = ("doc", "row", "batch")


def _ws() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


# ------------------------------------------------------------------------------------------------------------- tap
class _Tap:
    """Per-token LoRA-gradient hooks on the adapted linears of blocks >= cuts[0]. Linear inputs are taken from the
    activation-checkpoint RECOMPUTE during backward (holding them from the forward would defeat checkpointing) and
    paired with the output gradient captured by a tensor hook registered in the original forward."""

    def __init__(self, mods, cuts: List[int], scopes: List[str]):
        self.cuts = list(cuts)
        self.scopes = list(scopes)
        self.lins = []
        for m in mods:
            b = int(m._lora_block)
            if b < self.cuts[0]:
                continue
            g = max(i for i, c in enumerate(self.cuts) if b >= c)
            self.lins.append((m, g))
        self.mode: Optional[str] = None
        self.in_bwd = False
        self.xs: Dict[int, Any] = {}
        self.gs: Dict[int, torch.Tensor] = {}
        self.E: List[Optional[torch.Tensor]] = [None] * len(self.cuts)  # energy pass: (B*W,) per stage group
        self.bins: Dict[str, List[torch.Tensor]] = {}  # bins pass: scope -> per stage group (B, T) long
        self.acc: Dict[str, List[Dict[str, torch.Tensor]]] = {}  # scope -> per linear {"A": (K+1,r,in), "B": ...}
        self.fired = 0
        self.skipped = 0
        self.BW = (0, 0)
        self.curB = 0
        for i, (m, _) in enumerate(self.lins):
            m.register_forward_hook(self._fwd(i))
        # the hidden width at the cut blocks (the router compacts even the dense pass, to the longest real row)
        self.lins[0][0].register_forward_pre_hook(self._width)

    def _width(self, mod, inp):
        if self.mode is not None:
            x = inp[0]
            self.BW = (self.curB, int(x.numel() // (self.curB * x.shape[-1])))

    def n_params(self) -> int:
        return sum(m.lora_A.numel() + m.lora_B.numel() for m, _ in self.lins)

    def alloc(self, nb: int, dev) -> None:
        self.acc = {}
        for s in self.scopes:
            self.acc[s] = []
            for m, _ in self.lins:
                r, din = m.lora_A.shape[-2], m.in_features
                dout = m.out_features
                self.acc[s].append({"A": torch.zeros(nb, r, din, device=dev, dtype=torch.float32),
                                    "B": torch.zeros(nb, dout, r, device=dev, dtype=torch.float32)})

    def free(self) -> None:
        self.acc = {}

    def _fwd(self, i):
        def hook(mod, inp, out):
            if self.mode is None:
                return
            if self.in_bwd:
                # activation-checkpoint recompute: the input (and this block's unsharded adapter weights)
                x = inp[0].detach()
                A, Bm = mod.lora_A.detach(), mod.lora_B.detach()
                if i in self.gs:
                    self._grad(i, x, A, Bm, self.gs.pop(i))
                else:
                    self.xs[i] = (x, A, Bm)
                return
            if not out.requires_grad:
                return
            out.register_hook(lambda g, i=i: self._on_g(i, g))

        return hook

    def _on_g(self, i, g):
        if i in self.xs:
            x, A, Bm = self.xs.pop(i)
            self._grad(i, x, A, Bm, g)
        else:
            self.gs[i] = g.detach()

    def check_paired(self) -> None:
        if self.xs or self.gs:
            n1, n2 = len(self.xs), len(self.gs)
            self.xs, self.gs = {}, {}
            raise RuntimeError(f"oracle tap: {n1} linear inputs / {n2} output grads left unpaired after backward "
                               "(activation checkpointing must be full and early stop off)")

    def _grad(self, i, x, A, Bm, g):
        m, grp = self.lins[i]
        B, W = self.BW
        x = x.reshape(-1, x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        if x.shape[0] != B * W:
            # a linear applied to a reshaped tensor: no per-token term (counted; must stay 0)
            self.skipped += 1
            return
        self.fired += 1
        s = float(m._lora_scale)
        xf, gf = x.float(), g.float()
        ax = xf @ A.float().t()  # (N, r)
        btd = gf @ Bm.float()  # (N, r)
        if self.mode == "energy":
            e = (s * s) * (gf.square().sum(-1) * ax.square().sum(-1) + btd.square().sum(-1) * xf.square().sum(-1))
            self.E[grp] = e if self.E[grp] is None else self.E[grp] + e
            return
        # bins: exact per-bin LoRA gradient sums, dL/dB = s * delta (A x)^T, dL/dA = s * (B^T delta) x^T
        for sc in self.scopes:
            bb = self.bins[sc][grp]
            if bb.shape[1] < W:
                bb = F.pad(bb, (0, W - bb.shape[1]))
            bins = bb[:, :W].reshape(-1)
            order = torch.argsort(bins, stable=True)
            cnt = torch.bincount(bins, minlength=len(KS) + 1).tolist()
            acc = self.acc[sc][i]
            gs_, axs, btds, xs_ = gf[order], ax[order], btd[order], xf[order]
            c0 = 0
            for k, n in enumerate(cnt):
                if n == 0:
                    continue
                c1 = c0 + n
                acc["B"][k].addmm_(gs_[c0:c1].t(), axs[c0:c1], alpha=s)
                acc["A"][k].addmm_(btds[c0:c1].t(), xs_[c0:c1], alpha=s)
                c0 = c1


# ------------------------------------------------------------------------------------------------------- selection
def _group_bins(score: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """bins (N,) long in 1..len(KS): a token is in the top-k share of its group (k = KS[j]) iff bin <= j + 1, i.e.
    its within-group rank (descending score) < ceil(KS[j] * n_group)."""
    if score.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=score.device)
    o = torch.argsort(-score.double())
    o = o[torch.argsort(group[o], stable=True)]
    gs = group[o]
    _, inv, cnt = torch.unique_consecutive(gs, return_inverse=True, return_counts=True)
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(len(o), device=score.device) - start[inv]
    n = cnt[inv]
    ks = torch.tensor(KS, dtype=torch.float64, device=score.device)
    need = torch.ceil(ks[None, :] * n[:, None].double() - 1e-9)  # (N, K)
    b = 1 + (rank[:, None].double() >= need).sum(dim=1)
    out = torch.empty_like(b)
    out[o] = b
    return out.long()


def _gather_cat(x: torch.Tensor) -> List[torch.Tensor]:
    """all-gather variable-length 1-D tensors."""
    if _ws() == 1:
        return [x]
    n = torch.tensor([x.numel()], device=x.device)
    ns = [torch.zeros_like(n) for _ in range(_ws())]
    dist.all_gather(ns, n)
    mx = int(max(int(v) for v in ns))
    pad = torch.zeros(mx, dtype=x.dtype, device=x.device)
    pad[: x.numel()] = x
    outs = [torch.zeros_like(pad) for _ in range(_ws())]
    dist.all_gather(outs, pad)
    return [o[: int(v)] for o, v in zip(outs, ns)]


# ------------------------------------------------------------------------------------------------------- installer
class _Oracle:
    def __init__(self, tm, scope: str, target: float, random_mult: float, curve_scopes: List[str], out_dir: str,
                 npz_every: int, seed: int, routed_target: float = 0.0, view: str = "body"):
        self.tm = tm
        self.model = tm.model
        self.router = getattr(self.model, "_online_token_router", None)
        if self.router is None or self.router.cfg.score != "oracle":
            raise SystemExit("--oracle-scope needs the online router with --st-otr-score oracle")
        self.chain = self.router.chain()
        self.cuts = [r.cfg.drop_layer for r in self.chain]
        lora = getattr(self.model, "_early_lora", None)
        if not lora:
            raise SystemExit("--oracle-scope: the trainable space must be the whole-model LoRA (--st-otr-early-lora)")
        if scope not in SCOPES:
            raise SystemExit(f"--oracle-scope {scope!r}: one of {SCOPES}")
        self.scope, self.target, self.random_mult = scope, float(target), float(random_mult)
        self.routed_target = float(routed_target)
        if view not in ("body", "all"):
            raise SystemExit(f"--oracle-view {view!r}: body or all")
        self.view = view
        self.curve_scopes = list(dict.fromkeys([scope] + [s for s in curve_scopes if s]))
        self.tap = _Tap(lora["modules"], self.cuts, self.curve_scopes)
        self.out_dir = out_dir
        self.npz_every = int(npz_every)
        self.seed = int(seed)
        self.orig_train_batch = tm.train_batch
        self.step = 0
        self.jsonl = None
        if _rank() == 0:
            os.makedirs(os.path.join(out_dir, "sel"), exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        os.makedirs(os.path.join(out_dir, "sel"), exist_ok=True)
        self.trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.pgroup = [self._pgroup(n) for n, _ in self.trainable]
        self._ce_rows: List[torch.Tensor] = []
        head = self.model.lm_head
        # prepend: runs before FSDP's post-forward reshard, so the head weight is still unsharded
        head.register_forward_hook(self._head_hook, with_kwargs=True, prepend=True)
        log.info("[oracle] scope=%s target=%.3f random_mult=%g cuts=%s curves=%s tapped linears=%d (%.1fM LoRA params, "
                 "%d bins -> %.2f GB of accumulators per scope)", scope, target, random_mult, self.cuts,
                 self.curve_scopes, len(self.tap.lins), self.tap.n_params() / 1e6, len(KS) + 1,
                 self.tap.n_params() * (len(KS) + 1) * 4 / 2**30)

    def _pgroup(self, name: str) -> str:
        n = name.replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.", "")
        if "lora_" in n and "blocks." in n:
            b = int(n.split("blocks.")[1].split(".")[0])
            g = sum(b >= c for c in self.cuts)
            return f"lora_s{g}"
        if "lm_head" in n:
            return "head"
        return "other"

    # --------------------------------------------------------------------------------------------- per-row CE hook
    def _head_hook(self, mod, args, kwargs, out):
        """per-row answer CE of this forward (no grad; the head runs again only at the label positions)."""
        if not self.model.training:
            return
        labels = kwargs.get("labels")
        x = args[0] if args else kwargs.get("x")
        if labels is None or x is None:
            return
        ign = kwargs.get("ignore_index", -100)
        with torch.no_grad():
            B = labels.shape[0]
            lab = labels.reshape(B, -1)
            xx = x.reshape(B, lab.shape[1], -1) if x.numel() == B * lab.shape[1] * x.shape[-1] else None
            if xx is None:
                return
            ce = torch.zeros(B, device=lab.device)
            nt = torch.zeros(B, device=lab.device)
            for b in range(B):
                pos = (lab[b] != ign).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    continue
                h = xx[b, pos]
                h = mod.norm(h) if getattr(mod, "norm", None) is not None else h
                logits = F.linear(h, _full(mod.w_out.weight),
                                  None if mod.w_out.bias is None else _full(mod.w_out.bias)).float()
                ce[b] = F.cross_entropy(logits, lab[b, pos], reduction="sum")
                nt[b] = pos.numel()
            self._ce_rows.append(torch.stack([ce, nt], dim=1))

    # ------------------------------------------------------------------------------------------------------ passes
    def _microbatches(self, batch):
        from olmo_core.data.utils import get_labels, split_batch

        tm = self.tm
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=tm.label_ignore_index)
        if tm.microbatch_sort_pad_id is not None and batch["input_ids"].shape[0] > 1:
            n_rows = batch["input_ids"].shape[0]
            lengths = (batch["input_ids"] != tm.microbatch_sort_pad_id).sum(dim=1)
            order = torch.argsort(lengths, descending=True)
            batch = {k: (v[order.to(v.device)] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n_rows else v)
                     for k, v in batch.items()}
        seq_len = batch["input_ids"].shape[1]
        return batch, split_batch(batch, tm.rank_microbatch_size // seq_len)

    def _pass(self, batch, mbs, mode: Optional[str], masks=None):
        """one forward + backward over every micro-batch; returns (summed CE, per-row [ce, n_tokens] list)."""
        from torch.utils.checkpoint import set_checkpoint_early_stop

        from olmo_core.utils import move_to_device

        tm = self.tm
        n_tok = move_to_device((batch["labels"] != tm.label_ignore_index).sum(), tm.device)
        ce_sum = torch.zeros((), device=tm.device)
        self._ce_rows = []
        seen = []
        self.tap.mode = mode
        for mi, mb in enumerate(mbs):
            for r in self.chain:
                r._force_dense = masks is None
                r._oracle_keep = None if masks is None else masks[mi][r.stage_idx]
            if mode == "bins":
                self.tap.bins = {s: self._bins[s][mi] for s in self.curve_scopes}
            if mode == "energy":
                self.tap.E = [None] * len(self.cuts)
            with tm._train_microbatch_context(mi, len(mbs)):
                input_ids, labels, kw = tm._prepare_batch(dict(mb))  # it pops keys: keep mb reusable
                self.tap.curB = int(input_ids.shape[0])
                with set_checkpoint_early_stop(False):
                    _, loss, ce, _ = tm.model_forward(input_ids, labels=labels, ignore_index=tm.label_ignore_index,
                                                      loss_reduction="sum", z_loss_multiplier=tm.z_loss_multiplier,
                                                      loss_div_factor=n_tok, return_logits=False, **kw)
                ce_sum += _local(ce.detach())
                if masks is None:
                    seen.append(self.router._oracle_seen)
                    self.router._oracle_seen = None
                self.tap.in_bwd = True
                try:
                    loss.backward()
                finally:
                    self.tap.in_bwd = False
                self.tap.check_paired()
            if mode == "energy":
                B, W = self.tap.BW
                self._E.append([None if e is None else e.view(B, W) for e in self.tap.E])
        self.tap.mode = None
        for r in self.chain:
            r._force_dense = False
            r._oracle_keep = None
        return ce_sum, list(self._ce_rows), seen

    # ----------------------------------------------------------------------------------------------------- step
    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        if dry_run:
            return self.orig_train_batch(batch, dry_run=dry_run)
        tm, model = self.tm, self.model
        self.step += 1
        step = int(tm.trainer.global_step) if getattr(tm, "trainer", None) is not None else self.step
        t0 = time.time()
        tm._set_model_mode("train")
        batch, mbs = self._microbatches(batch)
        dev = tm.device
        import copy

        meter0 = {k: copy.deepcopy(getattr(model, k, None)) for k in _METER_ATTRS}
        rstats0 = [dict(r._stats) for r in self.chain]

        # ---- A: dense pass, energies + dense grad
        self._E = []
        tm.zero_grads()
        ceA, rowsA, seen = self._pass(batch, mbs, "energy")
        dense_g = [_local(p.grad).detach().clone() if p.grad is not None else None for _, p in self.trainable]
        tA = time.time()

        # ---- selection rankings per scope -> bins (original columns)
        info = self._rank(mbs, seen)
        # ---- B: dense pass, exact per-bin attribution gradients
        tm.zero_grads()
        self.tap.alloc(len(KS) + 1, dev)
        self._pass(batch, mbs, "bins")
        tm.zero_grads()
        curves = self._curves(dense_g)
        self.tap.free()
        tB = time.time()
        masks, sel = self._select(mbs, seen, curves, info)

        # roll back the dense passes' FLOP / router-stat side effects: the meter sees the routed pass only
        for k, v in meter0.items():
            if v is None:
                if hasattr(model, k):
                    delattr(model, k) if k != "_flop_extra_actual" else setattr(model, k, 0.0)
            else:
                setattr(model, k, v)
        for r, s0 in zip(self.chain, rstats0):
            r._stats = s0
        model.reset_auxiliary_metrics()

        # ---- C: routed pass (applied). With routed_target > 0 the set grows (every stage +3 grid points, ~2x k)
        # until cos(routed LoRA grad, dense LoRA grad) >= routed_target; only the last try's gradient is applied
        tries, extras = [], []
        while True:
            meter1 = {k: copy.deepcopy(getattr(model, k, None)) for k in _METER_ATTRS}
            rstats1 = [dict(r._stats) for r in self.chain]
            tm.zero_grads()
            fx1 = float(getattr(model, "_flop_extra_actual", 0.0))
            ceC, rowsC, _ = self._pass(batch, mbs, None, masks=masks)
            extras.append(float(getattr(model, "_flop_extra_actual", 0.0)) - fx1)
            if self.routed_target <= 0:
                break
            gst = self._grad_cos(dense_g)
            nk = torch.tensor([float(sum(m[g].sum() for m in masks)) for g in range(len(self.cuts))]
                              + [float(sum(b.sum() for b in info["body"]))], device=dev, dtype=torch.float64)
            if _ws() > 1:
                dist.all_reduce(nk)
            tries.append({"j": list(sel["j"]), "k": list(sel["k"]), "routed_cos_lora": gst["lora_all"]["cos"],
                          "routed_cos_all": gst["all"]["cos"], "keep_frac": [float(x / nk[-1]) for x in nk[:-1]]})
            if gst["lora_all"]["cos"] >= self.routed_target or all(j >= len(KS) - 1 for j in sel["j"]):
                break
            for k, v in meter1.items():
                if v is None:
                    if hasattr(model, k):
                        delattr(model, k) if k != "_flop_extra_actual" else setattr(model, k, 0.0)
                else:
                    setattr(model, k, v)
            for r, s1 in zip(self.chain, rstats1):
                r._stats = s1
            model.reset_auxiliary_metrics()
            masks, sel = self._select(mbs, seen, curves, info, js=[min(len(KS) - 1, j + 3) for j in sel["j"]])
        routed_extra = extras[-1]
        sel["routed_tries"] = tries
        sel["routed_extras_all_tries"] = extras
        tC = time.time()
        model.post_batch(dry_run=False)
        from olmo_core.optim import SkipStepOptimizer

        if isinstance(tm.optim, SkipStepOptimizer):
            ce_r = ceC.clone()
            if _ws() > 1:
                ce_r.div_(tm._reduce_divide_factor)
                dist.all_reduce(ce_r)
                ce_r.div_(_ws())
                ce_r.mul_(tm._reduce_divide_factor)
            tm.record_ce_loss(ce_r)
            tm.optim.latest_loss = ce_r
        else:
            tm.record_ce_loss(ceC, _reduce_mean())
        for name, (val, red) in model.compute_auxiliary_metrics(reset=True).items():
            tm.record_metric(name, val, red, namespace="train")
        gstats = self._grad_cos(dense_g)
        del dense_g
        self._log(step, batch, mbs, seen, info, curves, sel, gstats, ceA, ceC, rowsA, rowsC, routed_extra,
                  (tA - t0, tB - tA, tC - tB))

    # ------------------------------------------------------------------------------------------------- rankings
    def _rank(self, mbs, seen):
        """per micro-batch: body (B, T), doc/row group ids, energies per stage in original columns, and bins per
        (scope, stage)."""
        rank, dev = _rank(), self.tm.device
        rows_per_rank = sum(int(sn["body"].shape[0]) for sn in seen)
        info = {"body": [], "cid": [], "gold": [], "E": [], "rowid": []}
        r0 = rank * rows_per_rank
        for mi, (mb, sn) in enumerate(zip(mbs, seen)):
            body, cid = sn["body"], sn["cid"]
            B, Tr = body.shape
            real = cid != -2
            if not bool((real.long().cumsum(1) == torch.arange(1, Tr + 1, device=dev)[None].clamp(
                    max=real.sum(1, keepdim=True))).all()):
                # the energy columns (compacted hidden at the cut) map to original columns only for right-padded rows
                raise RuntimeError("oracle: real tokens are not a prefix of the row (left padding?)")
            E = []
            for g in range(len(self.cuts)):
                e = self._E[mi][g]
                full = torch.zeros(B, Tr, device=dev)
                if e is not None:
                    w = min(Tr, e.shape[1])
                    full[:, :w] = e[:, :w].float()
                E.append(full)
            info["body"].append(body)
            info["cid"].append(cid)
            info["gold"].append(sn["gold"])
            info["E"].append(E)
            info["rowid"].append(torch.arange(r0, r0 + B, device=dev))
            r0 += B
        self._bins = {s: [] for s in self.curve_scopes}
        for s in self.curve_scopes:
            per_stage = []
            for g in range(len(self.cuts)):
                sc = torch.cat([E[g][b] for E, b in zip(info["E"], info["body"])])
                if s == "doc":
                    grp = torch.cat([(rid[:, None] * 100_000 + c.clamp(min=0).long())[b]
                                     for rid, c, b in zip(info["rowid"], info["cid"], info["body"])])
                    bins = _group_bins(sc, grp)
                elif s == "row":
                    grp = torch.cat([rid[:, None].expand_as(b)[b] for rid, b in zip(info["rowid"], info["body"])])
                    bins = _group_bins(sc, grp)
                else:
                    parts = _gather_cat(sc)
                    allb = _group_bins(torch.cat(parts), torch.zeros(sum(p.numel() for p in parts), dtype=torch.long,
                                                                   device=dev))
                    off = sum(p.numel() for p in parts[:rank])
                    bins = allb[off: off + sc.numel()]
                per_stage.append(bins)
            # scatter back to (B, Tr) per micro-batch; non-body = bin 0 (always kept)
            off = 0
            for mi, b in enumerate(info["body"]):
                n = int(b.sum())
                mbins = []
                for g in range(len(self.cuts)):
                    t = torch.zeros(b.shape, dtype=torch.long, device=dev)
                    t[b] = per_stage[g][off: off + n]
                    mbins.append(t)
                self._bins[s].append(mbins)
                off += n
        return info

    def _curves(self, dense_g):
        """all-reduce the bin accumulators; per (scope, stage): cos(G_S(k), G), <G_S, G>/|G|^2, |G_S|/|G| on the grid,
        plus the bins-vs-true-gradient check (sum over all bins == the dense LoRA gradient)."""
        out = {}
        for s in self.curve_scopes:
            accs = self.tap.acc[s]
            if _ws() > 1:
                for a in accs:
                    for t in a.values():
                        dist.all_reduce(t)
                        t.div_(_ws())
            res = []
            for g in range(len(self.cuts)):
                # view "all": G = every token's term, G_S = non-body (bin 0, always kept) + top-k body bins.
                # view "body": the body tokens' part of the gradient only, G_body vs G_S,body (bin 0 left out of both)
                z = lambda: torch.zeros(len(KS), device=self.tm.device, dtype=torch.float64)  # noqa: E731
                dot, nrm, dotb, nrmb = z(), z(), z(), z()
                tot = torch.zeros((), device=self.tm.device, dtype=torch.float64)
                totb, nb0, cross0 = torch.zeros_like(tot), torch.zeros_like(tot), torch.zeros_like(tot)
                for (m, grp), a in zip(self.tap.lins, accs):
                    if grp != g:
                        continue
                    for t in a.values():
                        T = t.sum(0)
                        Tb = T - t[0]
                        csb = t[1:].cumsum(0)  # body bins 1..j+1
                        cs = csb + t[0][None]  # + always-kept non-body
                        dot += (cs * T[None]).flatten(1).sum(1).double()
                        nrm += cs.square().flatten(1).sum(1).double()
                        tot += T.square().sum().double()
                        dotb += (csb * Tb[None]).flatten(1).sum(1).double()
                        nrmb += csb.square().flatten(1).sum(1).double()
                        totb += Tb.square().sum().double()
                        nb0 += t[0].square().sum().double()
                        cross0 += (t[0] * T).sum().double()
                eps = 1e-300
                cos = dot / (nrm.clamp(min=eps) * tot.clamp(min=eps)).sqrt()
                cosb = dotb / (nrmb.clamp(min=eps) * totb.clamp(min=eps)).sqrt()
                res.append({"cos": cos.tolist(), "proj": (dot / tot.clamp(min=eps)).tolist(),
                            "rel_norm": (nrm / tot.clamp(min=eps)).sqrt().tolist(),
                            "cos_body": cosb.tolist(), "proj_body": (dotb / totb.clamp(min=eps)).tolist(),
                            "rel_norm_body": (nrmb / totb.clamp(min=eps)).sqrt().tolist(),
                            # the always-kept non-body tokens alone: cos with G, and the body share of |G|^2
                            "cos_nonbody_only": float(cross0 / (nb0.clamp(min=eps) * tot.clamp(min=eps)).sqrt()),
                            "body_norm_share": float((totb / tot.clamp(min=eps)).sqrt())})
            out[s] = res
        # check (step 1 and every 50): the summed bins equal the dense LoRA gradient of a few tapped linears
        if self.step == 1 or self.step % 50 == 0:
            name_of = {id(p): i for i, (_, p) in enumerate(self.trainable)}
            cs = []
            for (m, _), a in list(zip(self.tap.lins, self.tap.acc[self.scope]))[:: max(1, len(self.tap.lins) // 6)]:
                i = name_of.get(id(m.lora_B))
                if i is None or dense_g[i] is None:
                    continue
                gfull = self.trainable[i][1]
                full = _full_like_grad(gfull, dense_g[i])
                if full is None:
                    continue
                T = a["B"].sum(0)
                cs.append(float(F.cosine_similarity(T.flatten(), full.float().flatten(), dim=0)))
            out["_check_bins_vs_grad_cos"] = cs
        return out

    def _select(self, mbs, seen, curves, info, js=None):
        """stage masks per micro-batch (original columns) from the chosen grid k per stage (``js`` overrides)."""
        dev = self.tm.device
        cur = curves[self.scope]
        if js is None:
            js = []
            for g in range(len(self.cuts)):
                c = cur[g]["cos_body" if self.view == "body" else "cos"]
                j = next((i for i, v in enumerate(c) if v >= self.target), len(KS) - 1)
                js.append(j)
        gen = torch.Generator(device=dev).manual_seed(self.seed * 1_000_003 + self.step * 7919 + _rank() * 104_729
                                                      + 31 * sum(js))
        masks, sel = [], {"j": js, "k": [KS[j] for j in js], "cos_at_j": [cur[g]["cos"][j] for g, j in enumerate(js)],
                          "cos_body_at_j": [cur[g]["cos_body"][j] for g, j in enumerate(js)],
                          "rnd": []}
        for mi in range(len(mbs)):
            body = info["body"][mi]
            stage = [body & (self._bins[self.scope][mi][g] <= js[g] + 1) for g in range(len(self.cuts))]
            # nesting: a later stage's tokens must survive every earlier cut
            for g in range(len(self.cuts) - 2, -1, -1):
                stage[g] = stage[g] | stage[g + 1]
            orc = [m.clone() for m in stage]
            rnd = [torch.zeros_like(body) for _ in stage]
            if self.random_mult > 0:
                prev = body
                for g in range(len(self.cuts)):
                    # extra uniformly random body tokens, random_mult x this row's oracle count, from the tokens still
                    # alive at this cut that the oracle did not pick
                    pool = prev & ~orc[g]
                    n_x = torch.minimum((orc[g].sum(1).float() * self.random_mult).round().long(), pool.sum(1))
                    u = torch.rand(body.shape, generator=gen, device=dev)
                    u = torch.where(pool, u, torch.full_like(u, 2.0))
                    rk = u.argsort(1).argsort(1)
                    rnd[g] = pool & (rk < n_x[:, None])
                    prev = orc[g] | rnd[g]
                    stage[g] = prev
                # nested: later-stage extras are drawn from the previous stage's survivors (prev chain above)
            masks.append(stage)
            sel["rnd"].append(rnd)
            sel.setdefault("orc", []).append(orc)
        return masks, sel

    # ------------------------------------------------------------------------------------------------ grad cosines
    def _grad_cos(self, dense_g):
        """cos(dense, routed) per parameter group, raw and after Adam's second-moment scaling (sum over ranks of
        the local FSDP shards' dot products)."""
        optim = self.tm.optim
        groups = sorted(set(self.pgroup)) + ["all", "lora_all"]
        acc = {g: torch.zeros(6, device=self.tm.device, dtype=torch.float64) for g in groups}
        for (n, p), pg, gd in zip(self.trainable, self.pgroup, dense_g):
            if gd is None or p.grad is None:
                continue
            gr = _local(p.grad).float()
            gdf = gd.float()
            st = optim.state.get(p, {}) if optim is not None else {}
            v = st.get("exp_avg_sq")
            if v is not None:
                v = _local(v).float()
                stp = st.get("step", 1)
                stp = float(stp.item() if torch.is_tensor(stp) else stp) or 1.0
                beta2 = optim.param_groups[0].get("betas", (0.9, 0.95))[1]
                eps = optim.param_groups[0].get("eps", 1e-8)
                pre = 1.0 / ((v / (1 - beta2 ** max(stp, 1.0))).sqrt() + eps)
            else:
                pre = None
            row = [float((gdf * gr).sum()), float(gdf.square().sum()), float(gr.square().sum())]
            if pre is not None:
                a, b = gdf * pre, gr * pre
                row += [float((a * b).sum()), float(a.square().sum()), float(b.square().sum())]
            else:
                row += row[:3]
            t = torch.tensor(row, device=self.tm.device, dtype=torch.float64)
            for g in (pg, "all") + (("lora_all",) if pg.startswith("lora") else ()):
                acc[g] += t
        keys = list(acc)
        stack = torch.stack([acc[k] for k in keys])
        if _ws() > 1:
            dist.all_reduce(stack)
        out = {}
        for k, v in zip(keys, stack.tolist()):
            out[k] = {"cos": v[0] / max(1e-300, math.sqrt(v[1] * v[2])),
                      "cos_adam": v[3] / max(1e-300, math.sqrt(v[4] * v[5])),
                      "norm_dense": math.sqrt(v[1]), "norm_routed": math.sqrt(v[2])}
        return out

    # ------------------------------------------------------------------------------------------------------ logging
    def _log(self, step, batch, mbs, seen, info, curves, sel, gstats, ceA, ceC, rowsA, rowsC, routed_extra, times):
        tm, dev = self.tm, self.tm.device
        model = self.model
        # ---- token counts per type / stage, gold shares (summed over ranks)
        cnt_names = ["real", "body", "nonbody"] + [f"keep_s{g}" for g in range(len(self.cuts))] + \
            [f"orc_s{g}" for g in range(len(self.cuts))] + [f"rnd_s{g}" for g in range(len(self.cuts))] + \
            ["gold_body"] + [f"gold_keep_s{g}" for g in range(len(self.cuts))] + \
            [f"E_s{g}" for g in range(len(self.cuts))] + [f"E_keep_s{g}" for g in range(len(self.cuts))] + \
            [f"E_gold_s{g}" for g in range(len(self.cuts))]
        c = torch.zeros(len(cnt_names), device=dev, dtype=torch.float64)
        per_row = []
        per_doc = []
        for mi in range(len(mbs)):
            body, cid, gold = info["body"][mi], info["cid"][mi], info["gold"][mi]
            real = cid != -2
            gb = (gold > 0.5) & body if gold is not None else torch.zeros_like(body)
            stage = [sel["orc"][mi][g] | sel["rnd"][mi][g] for g in range(len(self.cuts))]
            vals = [real.sum(), body.sum(), (real & ~body).sum()] + [s.sum() for s in stage] + \
                [o.sum() for o in sel["orc"][mi]] + [r.sum() for r in sel["rnd"][mi]] + [gb.sum()] + \
                [(s & gb).sum() for s in stage] + [info["E"][mi][g][body].sum() for g in range(len(self.cuts))] + \
                [info["E"][mi][g][stage[g]].sum() for g in range(len(self.cuts))] + \
                [info["E"][mi][g][gb].sum() for g in range(len(self.cuts))]
            c += torch.stack([v.double() for v in vals])
            per_row += torch.stack([body.sum(1)] + [s.sum(1) for s in stage], 1).float().tolist()
            # per document (vectorised): key = row * ND + chunk id over body tokens
            ND = int(cid.max().item()) + 2
            key = (torch.arange(body.shape[0], device=dev)[:, None] * ND + cid.clamp(min=0).long())[body]
            uk, inv = torch.unique(key, return_inverse=True)
            cols = [torch.ones_like(key, dtype=torch.float32), gb[body].float()] + [s[body].float() for s in stage]
            sums = torch.stack([torch.zeros(len(uk), device=dev).index_add_(0, inv, v) for v in cols], 1)
            sums[:, 1] = (sums[:, 1] > 0).float()
            per_doc += sums.tolist()
        if _ws() > 1:
            dist.all_reduce(c)
        C = dict(zip(cnt_names, c.tolist()))
        # ---- per-row CE (dense pass A vs routed pass C), summed over ranks into one vector in row order
        ra = torch.cat(rowsA) if rowsA else torch.zeros(0, 2, device=dev)
        rc = torch.cat(rowsC) if rowsC else torch.zeros(0, 2, device=dev)
        ra_all = torch.cat(_gather_cat(ra.flatten())).view(-1, 2) if ra.numel() else ra
        rc_all = torch.cat(_gather_cat(rc.flatten())).view(-1, 2) if rc.numel() else rc
        pr = torch.tensor(per_row, device=dev).flatten()
        pr_all = torch.cat(_gather_cat(pr)).view(-1, 1 + len(self.cuts)).tolist() if pr.numel() else []
        pd = torch.tensor(per_doc, device=dev).flatten()
        pd_all = torch.cat(_gather_cat(pd)).view(-1, 2 + len(self.cuts)) if pd.numel() else None
        ceA_t, ceC_t = ceA.clone(), ceC.clone()
        if _ws() > 1:
            dist.all_reduce(ceA_t)
            dist.all_reduce(ceC_t)
            ceA_t /= _ws()
            ceC_t /= _ws()
        # ---- FLOPs: dense per-row charge (as the meter's per-example dense formula) for the two dense passes
        dense_flops = 0.0
        for mi in range(len(mbs)):
            real = info["cid"][mi] != -2
            for L in real.sum(1).tolist():
                if L > 0:
                    dense_flops += float(L) * float(model.num_flops_per_token(int(L)))
        try:
            from olmo_core.nn.early_lora import early_lora_flops_per_token

            lora_tok = float(early_lora_flops_per_token(model))
        except Exception:  # noqa: BLE001
            lora_tok = 0.0
        dense_flops += lora_tok * float(sum(int((ci != -2).sum()) for ci in info["cid"]))
        ex_all = sel.get("routed_extras_all_tries") or [routed_extra]
        fl = torch.tensor([dense_flops, routed_extra, sum(ex_all)], device=dev, dtype=torch.float64)
        if _ws() > 1:
            dist.all_reduce(fl)
        dense_step, routed_adj, routed_adj_all = fl.tolist()
        routed_step = dense_step + routed_adj
        routed_all = len(ex_all) * dense_step + routed_adj_all
        n_cut = len(self.cuts)
        rec: Dict[str, Any] = {
            "step": step, "scope": self.scope, "target": self.target, "random_mult": self.random_mult,
            "time_s": {"dense_energy": times[0], "dense_bins_select": times[1], "routed": times[2]},
            "view": self.view, "ks": list(KS), "chosen_j": sel["j"], "chosen_k": sel["k"],
            "attr_cos_at_k": sel["cos_at_j"], "attr_cos_body_at_k": sel["cos_body_at_j"],
            "curves": {s: v for s, v in curves.items() if not s.startswith("_")},
            "check_bins_vs_grad_cos": curves.get("_check_bins_vs_grad_cos"),
            "counts": C,
            "keep_frac_body": [C[f"keep_s{g}"] / max(1.0, C["body"]) for g in range(n_cut)],
            "orc_frac_body": [C[f"orc_s{g}"] / max(1.0, C["body"]) for g in range(n_cut)],
            "gold_keep_frac": [C[f"gold_keep_s{g}"] / max(1.0, C["gold_body"]) for g in range(n_cut)],
            "distractor_keep_frac": [(C[f"keep_s{g}"] - C[f"gold_keep_s{g}"]) / max(1.0, C["body"] - C["gold_body"])
                                     for g in range(n_cut)],
            "energy_keep_share": [C[f"E_keep_s{g}"] / max(1e-300, C[f"E_s{g}"]) for g in range(n_cut)],
            "energy_gold_share": [C[f"E_gold_s{g}"] / max(1e-300, C[f"E_s{g}"]) for g in range(n_cut)],
            "gold_body_frac": C["gold_body"] / max(1.0, C["body"]),
            "ce_dense": float(ceA_t), "ce_routed": float(ceC_t),
            "row_ce_dense": [float(a) / max(1.0, float(n)) for a, n in ra_all.tolist()],
            "row_ce_routed": [float(a) / max(1.0, float(n)) for a, n in rc_all.tolist()],
            "row_answer_tokens": [int(n) for _, n in ra_all.tolist()],
            "row_body_keep": pr_all,
            "grad": gstats,
            "routed_tries": sel.get("routed_tries"),
            "flops": {"dense_pass": dense_step, "routed_pass": routed_step, "routed_tries": len(ex_all),
                      "actual_3pass": 2 * dense_step + routed_all,
                      "routed_over_dense": routed_step / max(1.0, dense_step)},
        }
        if pd_all is not None and pd_all.numel():
            dl, dg = pd_all[:, 0], pd_all[:, 1] > 0.5
            for g in range(n_cut):
                fr = pd_all[:, 2 + g] / dl.clamp(min=1)
                rec.setdefault("doc_keep_frac", {})[f"s{g}"] = {
                    "mean": float(fr.mean()), "std": float(fr.std()) if fr.numel() > 1 else 0.0,
                    "zero_docs": float((pd_all[:, 2 + g] == 0).float().mean()),
                    "gold_mean": float(fr[dg].mean()) if bool(dg.any()) else None,
                    "distractor_mean": float(fr[~dg].mean()) if bool((~dg).any()) else None}
            rec["n_docs"] = int(pd_all.shape[0])
        # ---- metrics (wandb)
        for g in range(n_cut):
            tm.record_metric(f"keep_frac_s{g}", rec["keep_frac_body"][g], None, namespace="oracle")
            tm.record_metric(f"k_s{g}", rec["chosen_k"][g], None, namespace="oracle")
            tm.record_metric(f"attr_cos_s{g}", rec["attr_cos_at_k"][g], None, namespace="oracle")
            tm.record_metric(f"attr_cos_body_s{g}", rec["attr_cos_body_at_k"][g], None, namespace="oracle")
            tm.record_metric(f"nonbody_only_cos_s{g}", curves[self.scope][g]["cos_nonbody_only"], None,
                             namespace="oracle")
            tm.record_metric(f"gold_keep_frac_s{g}", rec["gold_keep_frac"][g], None, namespace="oracle")
            tm.record_metric(f"distractor_keep_frac_s{g}", rec["distractor_keep_frac"][g], None, namespace="oracle")
        for k, v in gstats.items():
            tm.record_metric(f"gcos_{k}", v["cos"], None, namespace="oracle")
            tm.record_metric(f"gcos_adam_{k}", v["cos_adam"], None, namespace="oracle")
        tm.record_metric("ce_dense", rec["ce_dense"], None, namespace="oracle")
        tm.record_metric("ce_routed", rec["ce_routed"], None, namespace="oracle")
        tm.record_metric("routed_over_dense_flops", rec["flops"]["routed_over_dense"], None, namespace="oracle")
        tm.record_metric("actual_step_pflops", rec["flops"]["actual_3pass"] / 1e15, None, namespace="oracle")
        if _rank() == 0:
            if self.jsonl is None:
                self.jsonl = open(os.path.join(self.out_dir, "steps.jsonl"), "a")
            self.jsonl.write(json.dumps(rec) + "\n")
            self.jsonl.flush()
            g = gstats.get("all", {})
            log.info("[oracle] step %d k=%s attr_cos(%s view)=%s keep=%s gcos(all)=%.3f adam=%.3f ce dense %.4f routed %.4f "
                     "routed/dense flops %.3f t=%.0f/%.0f/%.0fs check=%s", step,
                     [round(x, 4) for x in rec["chosen_k"]], self.view,
                     [round(x, 3) for x in (rec["attr_cos_body_at_k"] if self.view == "body" else rec["attr_cos_at_k"])],
                     [round(x, 4) for x in rec["keep_frac_body"]], g.get("cos", float("nan")),
                     g.get("cos_adam", float("nan")), rec["ce_dense"], rec["ce_routed"],
                     rec["flops"]["routed_over_dense"], *times, rec["check_bins_vs_grad_cos"])
        # ---- selections (every rank, its own rows): replayable
        if self.npz_every > 0 and step % self.npz_every == 0:
            arrs: Dict[str, np.ndarray] = {}
            idx = batch.get("index")
            rows = 0
            for mi, mb in enumerate(mbs):
                body, cid = info["body"][mi], info["cid"][mi]
                real = cid != -2
                B = body.shape[0]
                pre = f"mb{mi}_"
                if "index" in mb and torch.is_tensor(mb["index"]):
                    arrs[pre + "index"] = mb["index"].cpu().numpy()
                arrs[pre + "real_len"] = real.sum(1).cpu().numpy().astype(np.int32)
                arrs[pre + "cid"] = cid.cpu().numpy().astype(np.int16)
                arrs[pre + "body"] = np.packbits(body.cpu().numpy(), axis=1)
                if info["gold"][mi] is not None:
                    arrs[pre + "gold"] = np.packbits((info["gold"][mi] > 0.5).cpu().numpy(), axis=1)
                for g in range(n_cut):
                    arrs[pre + f"E_s{g}"] = info["E"][mi][g].cpu().numpy().astype(np.float16)
                    arrs[pre + f"orc_s{g}"] = np.packbits(sel["orc"][mi][g].cpu().numpy(), axis=1)
                    if self.random_mult > 0:
                        arrs[pre + f"rnd_s{g}"] = np.packbits(sel["rnd"][mi][g].cpu().numpy(), axis=1)
                rows += B
            del idx
            np.savez_compressed(os.path.join(self.out_dir, "sel", f"step{step:05d}_rank{_rank()}.npz"), **arrs)


#: model counters the FLOP meter reads per step; the dense passes' increments are rolled back
_METER_ATTRS = ("_flop_extra_actual", "_soft_token_compaction", "_flop_example_lens")


def _reduce_mean():
    from olmo_core.train.common import ReduceType

    return ReduceType.mean


def _full(w: torch.Tensor) -> torch.Tensor:
    return w.full_tensor() if hasattr(w, "full_tensor") else w


def _full_like_grad(p: torch.Tensor, g_local: torch.Tensor) -> Optional[torch.Tensor]:
    """the full (unsharded) gradient from a saved local FSDP shard (all-gather along dim 0)."""
    if not hasattr(p, "device_mesh"):
        return g_local
    try:
        from torch.distributed.tensor import DTensor

        dt = DTensor.from_local(g_local, p.device_mesh, p.placements, run_check=False, shape=p.shape,
                                stride=p.stride())
        return dt.full_tensor()
    except Exception as exc:  # noqa: BLE001
        log.warning("[oracle] bins check: could not unshard a gradient (%s)", exc)
        return None


def install_oracle(train_module, *, scope: str, target: float = 0.95, random_mult: float = 0.0,
                   curve_scopes: str = "", out_dir: str, npz_every: int = 1, seed: int = 0,
                   routed_target: float = 0.0, view: str = "body") -> _Oracle:
    """Replace ``train_module.train_batch`` with the 3-pass oracle step (module docstring)."""
    o = _Oracle(train_module, scope, target, random_mult, [s for s in curve_scopes.replace("+", ",").split(",") if s], out_dir,
                npz_every, seed, routed_target, view)
    train_module.train_batch = o.train_batch
    train_module._oracle = o
    return o
