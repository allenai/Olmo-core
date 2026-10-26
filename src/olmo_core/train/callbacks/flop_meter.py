import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from olmo_core.distributed.utils import get_rank
from olmo_core.nn.layer_drop import flops_per_token_ran

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["FlopMeterCallback"]


@dataclass
class FlopMeterCallback(Callback):
    """
    Method-aware training-FLOP meter for the compute-scaling study
    (``records/flop-scaling-ffn-kv-plan.md`` §5).

    The trainer's own ``throughput/total petaflops`` charges every token the DENSE per-token cost.
    That is wrong for the two compute-saving arms, so this callback integrates, per step:

    - **dense**: ``tokens x flops_per_token(seq_len)`` (identical to the trainer's number).
    - **nested-FFN routing**: the FFN share of ``flops_per_token`` is scaled by that step's hard
      routing cost averaged over ALL layers (unrouted / not-yet-opened layers count 1.0); read
      from the model's ``_nested_ffn_moe`` holder.
    - **pooled soft tokens**: the model runs on the COMPACTED sequence, so the dense formula is
      applied to the compacted token count (attention on the compacted row length); read from
      the counters the model accumulates in ``_soft_token_compaction``.
    - **method overhead**: FLOPs a method spends outside the trained forward/backward (the online
      token router's no-grad prefix pass, scoring and per-step ridge fit) are accumulated per rank
      in the model's ``_flop_extra_actual`` and added to the actual cost.
    - **layer drop** (whole-block skipping, :mod:`olmo_core.nn.layer_drop`): the blocks skipped
      this step ran neither forward nor backward, so the ACTUAL cost charges only the blocks that
      ran (composes with every mode above); the dense-equivalent still charges every block.
    - **token exit** (:mod:`olmo_core.nn.token_exit`, on top of the soft-token compaction): blocks
      ``[0, L)`` are charged at the FULL row length and blocks ``[L, n)`` plus the LM head at the
      compacted length (:func:`~olmo_core.nn.token_exit.token_exit_example_flops`). Token exit and
      layer drop together are not priced by either formula: the meter refuses the combination.

    Records ``flop_meter/actual_pflops`` (cumulative), ``flop_meter/dense_pflops`` (what the same
    tokens would have cost dense) and their ratio every step, and writes ``flops.json`` into the
    save folder at the end of training for the results collector.
    """

    seq_len: int = 0
    """The padded sequence length the batch is shaped to (per-token FLOPs depend on it)."""

    pad_id: Optional[int] = None
    """Pad token id. When given, tokens are counted as NON-pad tokens of each step's batch (summed
    over ranks), so a padded single-example path (the soft-token arm) and a packed path (dense)
    are charged for the same real tokens; when ``None`` the padded global batch size is used."""

    per_example: bool = False
    """Charge every example at ITS OWN length: ``sum_i L_i * flops_per_token(L_i)`` over the real
    examples of the step (packed documents or padded rows; pad excluded), with ``L_i`` the
    compacted length for the soft-token arm. Needs ``pad_id``. Without it, attention is charged at
    ``seq_len`` for every token, which on a packed 32k window of ~4.5k-token examples overstates a
    dense arm's attention cost ~7x and so flatters any arm whose rows are shorter."""

    _actual: float = 0.0
    _dense: float = 0.0
    _tokens: int = 0
    _real_tokens_this_step: Optional[int] = None
    _fpt_cache: Optional[Dict[int, int]] = None
    _examples: int = 0

    def pre_train(self):
        if self.per_example:
            if self.pad_id is None:
                raise ValueError("FlopMeterCallback(per_example=True) needs pad_id")
            # the model records per-example lengths of every TRAINING forward while this is set
            self._model()._flop_pad_id = int(self.pad_id)

    def _per_example_flops(self, m, skipped=frozenset()):
        """``(dense, actual, tokens, examples)`` summed over ranks from the model's length log;
        ``actual`` charges only the blocks that ran (``skipped``: this step's layer-drop set)."""
        import torch
        import torch.distributed as dist

        st = getattr(m, "_flop_example_lens", None) or {"dense": [], "actual": []}
        if self._fpt_cache is None:
            self._fpt_cache = {}
        cache = self._fpt_cache

        def cost(lens, skip=frozenset()):
            tot = 0.0
            for L in lens:
                f = cache.get((L, skip))
                if f is None:
                    f = cache[(L, skip)] = flops_per_token_ran(m, int(L), skip)
                tot += float(L) * f
            return tot

        exit_layer = self._exit_layer(m)
        if exit_layer > 0 and skipped:
            raise ValueError("FlopMeter: token exit + layer drop in one step is not priced; refusing to guess")
        if exit_layer > 0:
            # token exit: the prefix blocks ran on the full (dense) example, the rest on the kept
            from olmo_core.nn.token_exit import token_exit_example_flops

            actual = sum(
                token_exit_example_flops(m, d, a, exit_layer, cache=cache)  # type: ignore[arg-type]
                for d, a in zip(st["dense"], st["actual"])
            )
        else:
            actual = cost(st["actual"], skipped)
        dev = next(m.parameters()).device
        v = torch.tensor(
            [cost(st["dense"]), actual, float(sum(st["dense"])), float(len(st["dense"]))],
            dtype=torch.float64,
            device=dev,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
        m._flop_example_lens = {"dense": [], "actual": []}
        return float(v[0]), float(v[1]), int(v[2]), int(v[3])

    def pre_step(self, batch):
        if self.pad_id is None:
            return
        import torch
        import torch.distributed as dist

        ids = batch["input_ids"]
        n = torch.tensor([int((ids != self.pad_id).sum().item())], device=ids.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(n, op=dist.ReduceOp.SUM)
        self._real_tokens_this_step = int(n.item())
    _ffn_per_tok: Optional[int] = None
    _dense_per_tok: Optional[int] = None
    _attn_score_per_tok: Optional[int] = None
    _kv_keep_weighted: float = 0.0
    _ffn_cost_weighted: float = 0.0
    _block_flops: Optional[list] = None
    _skip_keep_weighted: float = 0.0
    _overhead: float = 0.0
    _ld_run_frac_sum: float = 0.0
    _ld_steps: int = 0

    def _model(self):
        return self.trainer.train_module.model  # type: ignore[union-attr]

    def _model_or_none(self):
        try:
            return self._model()
        except Exception:
            return None

    @staticmethod
    def _exit_layer(m) -> int:
        """Token-exit layer of the model (0 = none / plain input drop)."""
        tx = getattr(m, "_token_exit", None)
        return int(tx.exit_layer) if tx is not None else 0

    def _per_token(self):
        if self._dense_per_tok is None:
            m = self._model()
            self._dense_per_tok = int(m.num_flops_per_token(self.seq_len))
            self._ffn_per_tok = int(
                sum(b.feed_forward.num_flops_per_token(self.seq_len) for b in m.blocks.values())
            )
        return self._dense_per_tok, self._ffn_per_tok

    def post_step(self):
        m = self._model()
        dense_tok, ffn_tok = self._per_token()
        ld = getattr(m, "_layer_drop", None)
        ld_skip = ld.skipped() if ld is not None else frozenset()
        tokens = int(self.trainer.global_batch_size)  # tokens in the (padded) global batch
        if self._real_tokens_this_step is not None:
            tokens = self._real_tokens_this_step
        dense_flops = tokens * dense_tok
        actual = float(dense_flops)

        nffn = getattr(m, "_nested_ffn_moe", None)
        if nffn is not None:
            holder = nffn["holder"]
            per_layer = holder.per_layer_cost(last_forward=True)
            n_layers = len(m.blocks)
            # layers absent from the snapshot ran dense (below the curriculum's min layer, or
            # below start_layer): cost 1.0
            cost_all = (sum(per_layer.values()) + (n_layers - len(per_layer))) / n_layers
            actual = tokens * (dense_tok - ffn_tok * (1.0 - cost_all))
            self.trainer.record_metric("flop_meter/ffn_cost_all_layers", cost_all)
            self._ffn_cost_weighted += cost_all * tokens

        kvr = getattr(m, "_kv_route", None)
        if kvr is not None:
            # Routed attention layers score only KEPT keys: their length-dependent (QK^T, PV)
            # FLOPs scale with the hard keep fraction of this step's last forward.
            keep = float(kvr["holder"].mean_keep(last_forward=True))
            if self._attn_score_per_tok is None:
                self._attn_score_per_tok = int(
                    sum(
                        m.blocks[str(li)].attention.num_flops_per_token(self.seq_len)
                        - m.blocks[str(li)].attention.num_flops_per_token(0)
                        for li in kvr["routed"]
                    )
                )
            actual -= tokens * self._attn_score_per_tok * (1.0 - keep)
            self._kv_keep_weighted += keep * tokens
            self.trainer.record_metric("flop_meter/kv_keep", keep)

        bsk = getattr(m, "_block_skip", None)
        if bsk is not None:
            # Skipped tokens save the WHOLE block (projections + FFN + scores); the FFN / KV
            # savings above were computed over all tokens, so charge them only on run tokens by
            # re-pricing per block: cost_b = keep_b * (proj_b + ffn_b*c_b + attn_b*kv_b).
            if self._block_flops is None:
                per = []
                for blk in m.blocks.values():
                    ffn = int(blk.feed_forward.num_flops_per_token(self.seq_len)) if hasattr(blk, "feed_forward") else 0
                    a = getattr(blk, "attention", None)
                    aa = int(a.num_flops_per_token(self.seq_len)) if a is not None else 0
                    sc = aa - int(a.num_flops_per_token(0)) if (a is not None and hasattr(a, "w_q")) else 0
                    per.append((aa - sc, ffn, sc))
                self._block_flops = per
            ffn_pl = nffn["holder"].per_layer_cost(last_forward=True) if nffn is not None else {}
            kv_pl = kvr["holder"].last_per_layer_keep if kvr is not None else {}
            keep_pl = bsk["holder"].per_layer_keep(last_forward=True)
            blocks_total = 0.0
            for i, (proj, ffn, sc) in enumerate(self._block_flops):
                blocks_total += keep_pl.get(i, 1.0) * (proj + ffn * ffn_pl.get(i, 1.0) + sc * kv_pl.get(i, 1.0))
            fixed = dense_tok - sum(p + f + c for p, f, c in self._block_flops)
            actual = tokens * (fixed + blocks_total)
            self._skip_keep_weighted += bsk["holder"].mean_keep(last_forward=True) * tokens
            self.trainer.record_metric("flop_meter/block_keep", bsk["holder"].mean_keep(last_forward=True))

        comp = getattr(m, "_soft_token_compaction", None)
        if comp is not None and comp["tokens_out"] > 0:
            rows = max(1, comp["rows"])
            out_len = comp["tokens_out"] / rows
            # the counters are per rank and count PADDED input tokens; the compacted output is
            # what actually ran. Scale per-rank -> world by the padded global batch.
            scale = int(self.trainer.global_batch_size) / max(1, comp["tokens_in"])
            actual = comp["tokens_out"] * scale * flops_per_token_ran(m, int(out_len), ld_skip)
            exit_layer = self._exit_layer(m)
            if exit_layer > 0 and ld_skip:
                raise ValueError("FlopMeter: token exit + layer drop in one step is not priced; refusing to guess")
            if exit_layer > 0:
                # token exit: blocks [0, L) ran on the full rows
                from olmo_core.nn.token_exit import prefix_flops_per_token

                in_len = comp["tokens_in"] / rows
                actual += scale * (
                    comp["tokens_in"] * prefix_flops_per_token(m, int(in_len), exit_layer)
                    - comp["tokens_out"] * prefix_flops_per_token(m, int(out_len), exit_layer)
                )
            self.trainer.record_metric(
                "flop_meter/compaction_ratio", comp["tokens_in"] / comp["tokens_out"]
            )
            if comp.get("labels_in"):
                # train CE is divided by the PRE-compaction label count: divide the logged CE
                # by this to get the mean CE over the labels that were actually trained on.
                self.trainer.record_metric(
                    "flop_meter/kept_label_frac", comp["labels_out"] / comp["labels_in"]
                )
            comp["tokens_in"] = comp["tokens_out"] = comp["rows"] = 0
            comp["labels_in"] = comp["labels_out"] = 0
        elif ld_skip:
            # the router / dense costs above priced every block at seq_len: drop the skipped ones
            actual *= flops_per_token_ran(m, self.seq_len, ld_skip) / max(1, dense_tok)

        if ld is not None:
            run_frac = 1.0 - len(ld_skip) / len(m.blocks)
            self._ld_run_frac_sum += run_frac
            self._ld_steps += 1
            self.trainer.record_metric("flop_meter/layers_run_frac", run_frac)

        if self.per_example:
            dense_flops, actual, tokens, n_ex = self._per_example_flops(m, ld_skip)
            self._examples += n_ex
            if dense_flops > 0:
                self.trainer.record_metric("flop_meter/step_actual_over_dense", actual / dense_flops)

        fw = getattr(m, "_ffn_width", None)
        otr = getattr(m, "_online_token_router", None)
        if fw is not None and not (otr is not None and int(getattr(otr.cfg, "drop_layer", 0)) > 0):
            # partial-MLP without a mid-stack router: every partial FFN ran at its fraction on every real token (the
            # mid-stack router charges this credit itself, per document and at each block's length)
            actual += tokens * sum(fw.credits_per_block(len(m.blocks)))
        if getattr(m, "_early_lora", None) and otr is None:
            # LoRA without the online router (the --lora-rank modifier): frozen-base blocks skip their weight
            # gradient, blocks below the adapters run forward-only, plus the adapters themselves
            from olmo_core.nn.early_lora import lora_flop_adjust_per_token

            actual += tokens * lora_flop_adjust_per_token(m, self.seq_len)
        if hasattr(m, "_flop_extra_actual"):
            # a method's own overhead outside the model forward/backward (the online token router's
            # prefix pass, scoring and ridge fit), accumulated per rank by the model this step
            import torch
            import torch.distributed as dist

            ov = torch.tensor([float(m._flop_extra_actual)], dtype=torch.float64,
                              device=next(m.parameters()).device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(ov, op=dist.ReduceOp.SUM)
            m._flop_extra_actual = 0.0
            actual += float(ov.item())
            self._overhead += float(ov.item())
            self.trainer.record_metric("flop_meter/overhead_pflops", self._overhead / 1e15)

        self._actual += actual
        self._dense += dense_flops
        self._tokens += tokens
        self.trainer.record_metric("flop_meter/actual_pflops", self._actual / 1e15)
        self.trainer.record_metric("flop_meter/dense_pflops", self._dense / 1e15)
        self.trainer.record_metric("flop_meter/actual_over_dense", self._actual / max(1.0, self._dense))
        if self.step % 50 == 0:
            log.info(
                "[flop-meter] step %d: actual %.2f PF, dense-equivalent %.2f PF (ratio %.3f), tokens %.1fM",
                self.step,
                self._actual / 1e15,
                self._dense / 1e15,
                self._actual / max(1.0, self._dense),
                self._tokens / 1e6,
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "actual_pflops": self._actual / 1e15,
            "dense_equivalent_pflops": self._dense / 1e15,
            "actual_over_dense": self._actual / max(1.0, self._dense),
            "tokens_processed": self._tokens,
            "tokens_are_real": self.pad_id is not None,
            "per_example": self.per_example,
            "examples": self._examples if self.per_example else None,
            "ffn_cost_frac": (
                self._ffn_cost_weighted / self._tokens if self._ffn_cost_weighted > 0 and self._tokens else None
            ),
            "block_skip_keep_frac": (
                self._skip_keep_weighted / self._tokens if self._skip_keep_weighted > 0 and self._tokens else None
            ),
            "kv_route_keep_frac": (
                self._kv_keep_weighted / self._tokens if self._kv_keep_weighted > 0 and self._tokens else None
            ),
            "overhead_pflops": self._overhead / 1e15,
            "layer_drop_run_frac": (self._ld_run_frac_sum / self._ld_steps if self._ld_steps else None),
            "steps": self.step,
            "seq_len": self.seq_len,
            "token_exit_layer": (
                self._exit_layer(m) if (m := self._model_or_none()) is not None else None
            ),
        }

    def post_train(self):
        if get_rank() == 0:
            path = os.path.join(str(self.trainer.save_folder), "flops.json")
            try:
                with open(path, "w") as f:
                    json.dump(self.summary(), f, indent=2)
                log.info("[flop-meter] wrote %s: %s", path, self.summary())
            except OSError as e:  # remote save folders
                log.warning("[flop-meter] could not write %s: %s", path, e)
