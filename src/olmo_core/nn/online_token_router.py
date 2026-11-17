"""
Online, train-time token router for soft-token compaction (``keep_token_rule="online"``).

Every training forward, a linear scorer over cheap per-token features ranks the DOCUMENT BODY
tokens of each row; the top ``keep_frac`` of them (a share ``explore_frac`` of that budget drawn
uniformly at random instead) stay real and the rest are dropped by the existing slot-less
compaction (:func:`~olmo_core.nn.pooled_soft_token.compact_pooled_rows`). Markers, the prompt, the
query and the answer are never routed.

The scorer is refit every optimizer step by ridge regression on targets measured in that step's own
backward pass: for a sample of the KEPT tokens, the log norm of the loss gradient at the compacted
row's embedding output (``target="gnorm"``) or ``|<grad, embedding>|`` (``"gxh"``, the first-order
cost of zeroing the token). Targets are centred per row, so only the within-row ranking is fit.
There is no heuristic warm start: until the first fit (``warmup_steps``) every row trains
uncompressed (keep-all fallback), which also gives the first fit unbiased targets over every token.
The exploration tokens keep later fits honest about the tokens the scorer would otherwise never
keep (selection bias), and they are what the logged held-out rank correlation is measured on.

Features (``feature_names``): document structure (offset from the body start / end as one-hots and
log2 buckets, relative position, document index, log length), lexical query match (does the token
id occur in the query; log repeat count in the row), the RMS-normalised hidden state ``h`` at
``feature_layer`` (0 = the input embedding; ``L > 0`` = a no-grad prefix pass through blocks
``[0, L)`` on the FULL row, charged to the FLOP meter), the query interaction ``h * q`` (``q`` = the
mean normalised hidden state of the query tokens; with ``query_position="after"`` document tokens
cannot see the query causally, so this is the only query-aware hidden feature), and optionally the
per-document 0/1 gold flag (``use_gold``, from the shard's gold-fingerprint sidecar).

Scoring never materialises the (T, F) feature matrix: ``score = s.w_s + h.(w_h + q * w_hq)``.

Every FLOP the router spends (prefix pass, scoring, target hook, ridge statistics, solve) is added
to ``model._flop_extra_actual``, which :class:`~olmo_core.train.callbacks.FlopMeterCallback` adds to
``flop_meter/actual_pflops``.

State (ridge statistics, weights) is NOT checkpointed: a resumed run restarts the router (warmup
included).
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

log = logging.getLogger(__name__)

__all__ = [
    "OnlineRouterConfig",
    "OnlineTokenRouter",
    "structural_features",
    "query_mask",
    "N_STRUCT",
]

#: one-hot offsets 0..15 + 11 log2 buckets (16..2^15), from the body start and from the body end
_N_ONEHOT = 16
_N_LOG = 11
#: bias, start one-hot+log, end one-hot+log, rel-in-doc, doc rel index, log doc length
N_STRUCT = 1 + 2 * (_N_ONEHOT + _N_LOG) + 3
#: in-query flag, log repeat count
N_LEX = 2
#: gnorm = log ||dL/dh_t|| (at the embedding output, or at the drop layer in mid-stack mode); gxh = log |<dL/dh_t, h_t>|;
#: dh = log ||h_t^L - h_t^0|| (residual change through blocks [0, L): an activation-strength target that does not
#: vanish once a token is well learned; mid-stack only, no backward needed); mix = within-row average of the gnorm
#: and dh ranks. The gradient norm is recorded under every target (fidelity metrics stay comparable).
TARGETS = ("gnorm", "gxh", "dh", "mix")


@dataclass
class OnlineRouterConfig:
    keep_frac: float = 0.3
    """Body tokens kept per row (fraction of the row's body tokens)."""
    explore_frac: float = 0.1
    """Share of the kept budget drawn uniformly at random among body tokens (exploration). >= 1.0 is the
    random control: every kept token is random, with the same budget and ``doc_share`` allocation."""
    doc_share: float = 1.0
    """Share of the budget spread over documents in proportion to their length (every document keeps
    its own top ``ceil(doc_share * keep_frac * |body|)`` tokens by score); the rest is allocated by
    score over the whole row. 1.0 = the router picks only WHICH tokens of each document stay, so the
    per-document kept fraction is (up to exploration and rounding) the same for every document and
    cannot name the gold one (count-leak AUC 0.5); 0.0 = pure row-level top-k, which a gradient-
    faithful scorer concentrates on gold documents (the "dense = gold" training shortcut)."""
    feature_layer: int = 0
    """0: features from the input embedding (free). L > 0: a no-grad prefix pass through blocks [0, L)."""
    drop_layer: int = 0
    """0: drop at the input (dropped tokens skip every block). L > 0: MID-STACK drop -- every token runs
    blocks [0, L) on the full row (trained, full context), then the router scores the forward's own
    h at layer L (no prefix pass) and the dropped tokens skip blocks [L, n_layers): a two-depth
    per-(token, layer) skip. Targets are then read at layer L."""
    score: str = "router"
    """Who picks the tokens: ``router`` (the learned scorer), or a CONTROL with the same budget, allocation and
    drop point: ``random`` (uniform; also ``explore_frac >= 1``) or ``first_last`` (each document's first and
    last tokens). Controls still fit and log the router, so the fidelity metrics stay comparable.
    ``dh`` (mid-stack only): a fixed, gold-blind, fit-free RULE -- each document's top tokens by the residual
    change ||h_t^L - h_t^0|| through blocks [0, L), read off the forward itself (no exploration)."""
    use_hidden: bool = True
    use_query: bool = True
    use_gold: bool = False
    target: str = "gnorm"
    ridge: float = 0.1
    """Ridge strength, relative to the mean diagonal of the (decayed) Gram matrix."""
    decay: float = 0.9
    """Per-step forgetting factor of the ridge statistics."""
    warmup_steps: int = 1
    """Leading optimizer steps (1-based) that train uncompressed (keep-all) and collect targets."""
    dense_every: int = 0
    """If > 0, every ``dense_every``-th step is also uncompressed (fresh unbiased targets)."""
    early_nograd: bool = False
    """Mid-stack only: on ROUTED steps run blocks [0, drop_layer) forward-only (no_grad; the hidden state is a fresh leaf
    at the split, so the router's target hook still reads dL/dh there). Those blocks then learn only on the dense
    (full-context) steps, and the FLOP meter charges them forward-only (1/3) on routed steps."""
    freeze_early: bool = False
    """Mid-stack only: FREEZE the embeddings and blocks [0, L) (L = the first drop layer): ``requires_grad=False`` before
    the train module is built (:func:`freeze_early_blocks`), so they never get gradients or optimizer state, and every
    forward (dense steps included) runs them forward-only -- no stored activations. Charged forward-only."""
    freeze_after: int = 0
    """Warmup-then-freeze: train every block normally for the first ``freeze_after`` optimizer steps, then (from step
    ``freeze_after + 1``) freeze the embeddings and blocks [0, L) and free their gradients and optimizer state
    (:func:`freeze_and_release`, run by the callback); afterwards it behaves as ``freeze_early``. 0 = off."""
    early_lora: int = 0
    """Low-rank early stack (:mod:`olmo_core.nn.early_lora`, installed by the recipe): blocks [0, drop_layer) train
    only rank-``early_lora`` adapters from step 0. Charged: those blocks at 2/3 (the activation gradient still runs
    down to block 0 for the adapters; no base weight gradient) plus the adapters' own cost. With ``freeze_after`` the
    adapters freeze too at that step, and from then on blocks [0, drop_layer) run forward-only (1/3, adapters forward
    only). Exclusive with freeze_early / freeze_schedule / early_nograd. 0 = off."""
    target_layer: int = 0
    """Read the router's gradient targets at block ``target_layer`` (> ``drop_layer``) instead of at the drop layer,
    e.g. 16: the importance that matters late in training sits deeper. Features still come from the drop layer. With
    the drop, routed steps fit on the EXPLORATION tokens only -- a uniform random sample of the body that is kept, so
    it reaches that block uncensored (the router's own picks would be a censored sample); full-context steps, if any,
    fit on every sampled token. With ``mod_layers`` every token reaches it on every step. Needs ``explore_frac > 0``
    with the drop. Targets gnorm / gxh only. 0 = the drop layer."""
    mass_target: float = 0.0
    """ADAPTIVE keep by predicted gradient mass: per document, keep the fewest top-scored body tokens whose predicted
    ||dL/dh||^2 (exp(2 x score); the score predicts the row-centred log gradient norm) covers this share of the
    document's predicted total, e.g. 0.8 -- easy documents keep few tokens, hard or tiny ones many. Then the
    ``min_per_doc`` floor, then the per-row ``max_keep_frac`` cap (peak memory). ``keep_frac`` only sizes the
    exploration draw (``explore_frac`` x ``keep_frac`` x |body|); ``doc_share`` is unused. Learned scorer only.
    0 = off (fixed ``keep_frac``)."""
    max_keep_frac: float = 1.0
    """With ``mass_target``: hard cap on the kept share of each row's body tokens (bounds the kept row, so peak
    memory); the exploration / forced / floor tokens rank first, the rest by score."""
    freeze_schedule: str = ""
    """PROGRESSIVE freeze, e.g. ``"4:0.25,7:0.5,12:0.75"``: freeze the embeddings + blocks [0, 4) after 25% of the
    optimizer steps, [0, 7) after 50%, [0, 12) after 75% (each releasing those blocks' gradients and optimizer state;
    run by the callback). Frozen blocks below the drop layer run forward-only (charged 1/3); frozen blocks at or above
    it still pass the activation gradient down to the router's target hook, so they skip only their weight gradient
    (charged 2/3, at the kept length). Exclusive with ``freeze_after`` / ``freeze_early``. Empty = off."""
    dense_tail_frac: float = 0.0
    """Train the LAST ``dense_tail_frac`` of the optimizer steps uncompressed (full context, every token, every block),
    e.g. 0.1; combine with ``dense_every=0`` for "routed, then a dense tail". Needs the total step count (set by
    :class:`~olmo_core.train.callbacks.OnlineRouterCallback`)."""
    dense_keep_frac: float = 1.0
    """Keep share on the periodic ``dense_every`` steps: 1.0 = full context (default); < 1 routes those steps at this share
    instead (e.g. 0.5), bounding the peak memory. Such steps are ordinary routed steps (no full-pass fidelity stats).
    Warmup steps stay full."""
    exit_loss_weight: float = 0.0
    """Mid-stack, single stage: an EARLY-EXIT LM loss for the dropped tokens -- their hidden state at the split goes through
    the model's LM head (final norm + unembedding) and ``exit_loss_weight`` x its CE (same reduction / ``loss_div_factor``
    as the main loss, so 1.0 counts every label once) is added to the loss. Only labelled dropped tokens contribute (zero
    in answer-only SFT, every token in CPT). Charged: the head on every dropped token, forward + backward. 0 = off."""
    merge_dropped: bool = False
    """Mid-stack only: instead of discarding each document's dropped tokens at the split, average their hidden states
    into ONE summary token per document, placed at the position of that document's last dropped token (causal: every
    token it averages precedes it), with no LM label. It joins the kept row (and is charged as a kept token); the
    gradient flows back through the mean into the dropped tokens' early blocks."""
    stages: str = ""
    """Pyramid (multi-stage) mid-stack drop, e.g. ``"3:0.5,7:0.1,16:0.03"``: at each ``layer:keep`` the kept tokens are cut
    to ``keep`` of the row's body tokens (absolute, decreasing), each stage scored from ITS layer's hidden state by its
    own router (own ridge fit, own gradient targets at that layer). Overrides ``drop_layer`` / ``keep_frac`` with the
    first stage's. Empty = one stage."""
    min_per_doc: int = 0
    """Per-document token FLOOR: every document keeps max(its share, ``min_per_doc``) body tokens (its top ones by score;
    all of a shorter document), counted against the budget. 0 = no floor."""
    keep_first: int = 0
    """Always keep each document's first ``keep_first`` body tokens (its header / id span); they count against the
    budget (and their document's share), and the router picks the rest. 0 = everything is routable."""
    mod_layers: str = ""
    """MIXTURE-OF-DEPTHS instead of a drop, e.g. ``"8-31:2"`` (blocks 8, 10, ..., 30) or ``"8,12,16"``: at the first
    listed block the router picks ``keep_frac`` of each document's body tokens (same scorer, budget, exploration and
    floors as the drop); every LISTED block then runs on those tokens plus every non-body token only (gathered to a
    shorter row at their original positions, scattered back), and the other tokens pass the residual through unchanged
    there. Unlisted blocks run on every token and every token reaches the LM head, so no label is dropped (CPT keeps
    its full-row loss). ``drop_layer`` is set to the first listed block. Targets are the gradient norms at that block
    over EVERY token -- skipped tokens still get a gradient through the residual, so nothing is censored. Charged:
    each listed block at the kept length. Empty = off."""
    per_task_key_tokens: int = 0
    """A separate router PER TASK in a mixed-task batch: a row's task key is its first N tokens (the prompt template;
    on the CTC mixes N = 11 identifies every task file, except the two retrieval files that share one template), and
    each task keeps its own ridge statistics and weights. The pooled fit scores a task until its own fit exists. The
    keys are synchronised across ranks every step. 0 = one shared router."""
    samples_per_row: int = 2048
    proj_dim: int = 0
    """Random-project the hidden features to this many dims (0: use all d_model dims)."""
    seed: int = 0
    log_every: int = 10

    def validate(self) -> None:
        if not 0.0 < self.keep_frac <= 1.0:
            raise ValueError(f"online router keep_frac must be in (0, 1], got {self.keep_frac}")
        if not 0.0 <= self.doc_share <= 1.0:
            raise ValueError(f"online router doc_share must be in [0, 1], got {self.doc_share}")
        if not 0.0 <= self.explore_frac <= 1.0:
            raise ValueError(f"online router explore_frac must be in [0, 1], got {self.explore_frac}")
        if self.stages:
            st = parse_stages(self.stages)
            if [l for l, _ in st] != sorted({l for l, _ in st}) or st[0][0] <= 0:
                raise ValueError(f"online router stages must have strictly increasing layers > 0, got {self.stages!r}")
            if any(not 0.0 < k <= 1.0 for _, k in st) or [k for _, k in st] != sorted((k for _, k in st), reverse=True):
                raise ValueError(f"online router stage keeps must be in (0, 1] and non-increasing, got {self.stages!r}")
        if not 0.0 < self.dense_keep_frac <= 1.0 or self.dense_keep_frac < self.keep_frac:
            raise ValueError(f"online router dense_keep_frac must be in [keep_frac, 1], got {self.dense_keep_frac}")
        if self.exit_loss_weight < 0 or (self.exit_loss_weight > 0 and (self.drop_layer <= 0 or self.stages)):
            raise ValueError("online router exit_loss_weight must be >= 0 and needs a single-stage mid-stack drop")
        if self.merge_dropped and (self.drop_layer <= 0 or self.stages):
            raise ValueError("online router merge_dropped needs a single-stage mid-stack drop (drop_layer > 0, no stages)")
        if self.freeze_after < 0 or (self.freeze_after and self.freeze_early):
            raise ValueError("online router freeze_after must be >= 0 and is exclusive with freeze_early")
        if self.freeze_early and self.drop_layer <= 0 and not self.stages:
            raise ValueError("online router freeze_early needs the mid-stack drop (drop_layer > 0 or stages)")
        if self.early_nograd and self.drop_layer <= 0 and not self.stages:
            raise ValueError("online router early_nograd needs the mid-stack drop (drop_layer > 0)")
        if not 0.0 <= self.dense_tail_frac < 1.0:
            raise ValueError(f"online router dense_tail_frac must be in [0, 1), got {self.dense_tail_frac}")
        if self.min_per_doc < 0:
            raise ValueError(f"online router min_per_doc must be >= 0, got {self.min_per_doc}")
        if self.keep_first < 0:
            raise ValueError(f"online router keep_first must be >= 0, got {self.keep_first}")
        if self.score not in ("router", "random", "first_last", "dh"):
            raise ValueError(f"online router score must be router / random / first_last / dh, got {self.score!r}")
        if self.score == "dh" and self.drop_layer <= 0:
            raise ValueError("online router score 'dh' needs the mid-stack drop (drop_layer > 0)")
        if self.target not in TARGETS:
            raise ValueError(f"online router target must be one of {TARGETS}, got {self.target!r}")
        if self.target in ("dh", "mix") and self.drop_layer <= 0:
            raise ValueError(f"online router target {self.target!r} needs the mid-stack drop (drop_layer > 0)")
        if self.feature_layer < 0 or self.drop_layer < 0:
            raise ValueError("online router feature_layer / drop_layer must be >= 0")
        if self.drop_layer > 0 and self.feature_layer > 0:
            raise ValueError("online router: with drop_layer > 0 the features come from that layer; "
                             "leave feature_layer at 0")
        if self.warmup_steps < 1:
            raise ValueError("online router warmup_steps must be >= 1 (the first fit needs targets)")
        if self.mod_layers:
            ml = parse_mod_layers(self.mod_layers)
            if self.drop_layer != ml[0]:
                raise ValueError(f"online router mod_layers {self.mod_layers!r}: drop_layer must be its first block "
                                 f"{ml[0]}, got {self.drop_layer}")
            if self.stages or self.merge_dropped or self.exit_loss_weight > 0:
                raise ValueError("online router mod_layers is exclusive with stages / merge_dropped / exit_loss_weight")
        if self.freeze_schedule:
            fs = parse_freeze_schedule(self.freeze_schedule)
            if self.freeze_after or self.freeze_early or self.drop_layer <= 0:
                raise ValueError("online router freeze_schedule needs the mid-stack drop and is exclusive with "
                                 "freeze_after / freeze_early")
            if [n for n, _ in fs] != sorted({n for n, _ in fs}) or [f for _, f in fs] != sorted({f for _, f in fs}) \
                    or not all(0.0 < f < 1.0 and n > 0 for n, f in fs):
                raise ValueError(f"online router freeze_schedule needs increasing blocks > 0 and increasing fractions "
                                 f"in (0, 1), got {self.freeze_schedule!r}")
        if not 0.0 <= self.mass_target <= 1.0 or not 0.0 < self.max_keep_frac <= 1.0:
            raise ValueError("online router mass_target must be in [0, 1] and max_keep_frac in (0, 1]")
        if self.mass_target > 0 and self.score != "router":
            raise ValueError("online router mass_target needs the learned scorer (score='router')")
        if self.early_lora:
            if self.early_lora < 0 or self.drop_layer <= 0:
                raise ValueError("online router early_lora needs rank > 0 and the mid-stack drop (drop_layer > 0)")
            if self.freeze_early or self.freeze_schedule or self.early_nograd:
                raise ValueError("online router early_lora is exclusive with freeze_early / freeze_schedule / "
                                 "early_nograd (freeze_after is allowed: LoRA from step 0, then the adapters freeze too)")
        if self.target_layer:
            if self.drop_layer <= 0 or self.target_layer <= self.drop_layer:
                raise ValueError("online router target_layer must be > drop_layer (> 0)")
            if self.target not in ("gnorm", "gxh") or self.stages:
                raise ValueError("online router target_layer needs target gnorm / gxh and a single stage")
            if not self.mod_layers and self.explore_frac <= 0:
                raise ValueError("online router target_layer with the drop fits on the exploration tokens: "
                                 "set explore_frac > 0")
        if self.per_task_key_tokens < 0:
            raise ValueError(f"online router per_task_key_tokens must be >= 0, got {self.per_task_key_tokens}")


def _log_bucket(j: torch.Tensor) -> torch.Tensor:
    """Bucket of an offset j >= 16: floor(log2 j) - 4, clamped to [0, _N_LOG)."""
    jf = j.clamp(min=16).to(torch.float32)
    return (torch.floor(torch.log2(jf)) - 4).clamp(0, _N_LOG - 1).to(torch.long)


def structural_features(
    input_ids: torch.Tensor,
    chunk_ids: torch.Tensor,
    *,
    doc_start_id: int,
    doc_end_id: int,
    offsets: Optional[Dict[str, torch.Tensor]] = None,
    marker_free: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token document-structure features.

    :param marker_free: positional-block rows (CPT, ``positional_block_size > 0``): a "document" is a
        block with no ``<|doc_start|>`` / ``<|doc_end|>``, so offsets count from the block's own first /
        last token instead of from the markers.
    :param offsets: if given, filled with ``j`` / ``r`` (B, T): body offset from the document's start / end.
    :returns: ``(feats (B, T, N_STRUCT) float32, body (B, T) bool)``; rows of non-body tokens are 0.
    """
    B, T = input_ids.shape
    dev = input_ids.device
    cid = chunk_ids.to(torch.long)
    markers = (input_ids == doc_start_id) | (input_ids == doc_end_id)
    body = (cid >= 0) & ~markers
    n_docs = max(1, int(cid.max().item()) + 1)
    pos = torch.arange(T, device=dev).expand(B, T)
    gid = (torch.arange(B, device=dev)[:, None] * n_docs + cid.clamp(min=0)).reshape(-1)
    inside = (cid >= 0).reshape(-1)
    big = torch.full((B * n_docs,), T, dtype=torch.long, device=dev)
    start = big.scatter_reduce(0, gid[inside], pos.reshape(-1)[inside], reduce="amin")
    end = torch.full_like(big, -1).scatter_reduce(
        0, gid[inside], pos.reshape(-1)[inside], reduce="amax"
    )
    gid2 = gid.reshape(B, T)
    s_t = start[gid2]
    e_t = end[gid2]
    m = 0 if marker_free else 1
    j = (pos - s_t - m).clamp(min=0)  # 0 = first body token (s_t is the doc_start marker, or the token itself)
    r = (e_t - pos - m).clamp(min=0)  # 0 = last body token (e_t is the doc_end marker, or the token itself)
    blen = (e_t - s_t + 1 - 2 * m).clamp(min=1)
    # number of documents present per row, for the relative document index
    n_row = (cid.max(dim=1).values + 1).clamp(min=1).to(torch.float32)
    f = torch.zeros(B, T, N_STRUCT, dtype=torch.float32, device=dev)
    f[..., 0] = 1.0
    o = 1
    for off in (j, r):
        oh = off < _N_ONEHOT
        idx = torch.where(oh, off, _N_ONEHOT + _log_bucket(off))
        f.scatter_(2, (o + idx).unsqueeze(-1), 1.0)
        o += _N_ONEHOT + _N_LOG
    f[..., o] = j.to(torch.float32) / (blen - 1).clamp(min=1).to(torch.float32)
    f[..., o + 1] = cid.clamp(min=0).to(torch.float32) / (n_row - 1).clamp(min=1)[:, None]
    f[..., o + 2] = torch.log(blen.to(torch.float32)) / 10.0
    f = f * body.unsqueeze(-1)
    if offsets is not None:
        offsets["j"], offsets["r"] = j, r
    return f, body


def query_mask(
    chunk_ids: torch.Tensor, labels: Optional[torch.Tensor], ignore_index: int = -100
) -> torch.Tensor:
    """
    The query tokens of each row: FREE tokens after the last document token and up to (including)
    the last prompt position before the answer. ``labels`` are the shifted training labels (label
    at ``i`` = token ``i + 1``), so answer tokens are never part of the query.
    """
    B, T = chunk_ids.shape
    dev = chunk_ids.device
    pos = torch.arange(T, device=dev).expand(B, T)
    cid = chunk_ids.to(torch.long)
    last_doc = torch.where(cid >= 0, pos, torch.full_like(pos, -1)).max(dim=1).values
    if labels is not None:
        lab = labels != ignore_index
        first_lab = torch.where(lab, pos, torch.full_like(pos, T)).min(dim=1).values
    else:
        first_lab = torch.full((B,), T, dtype=torch.long, device=dev)
    return (cid == -1) & (pos > last_doc[:, None]) & (pos <= first_lab[:, None])


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 3:
        return float("nan")
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = (ra.norm() * rb.norm()).item()
    return float((ra @ rb).item() / den) if den > 0 else float("nan")


def freeze_early_blocks(model, n_blocks: int) -> int:
    """``requires_grad_(False)`` on the embeddings and blocks [0, n_blocks); returns the number of frozen parameters."""
    n = 0
    mods = [model.embeddings] + [b for i, b in enumerate(model.blocks.values()) if i < n_blocks]
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(False)
            n += p.numel()
    return n


def freeze_and_release(model, n_blocks: int, optimizer=None) -> int:
    """Freeze the embeddings and blocks [0, n_blocks) mid-training: ``requires_grad_(False)``, drop their gradients and
    (if given) their optimizer state, so the memory is returned. Returns the number of parameters frozen."""
    mods = [model.embeddings] + [b for i, b in enumerate(model.blocks.values()) if i < n_blocks]
    n = 0
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(False)
            p.grad = None
            if optimizer is not None:
                optimizer.state.pop(p, None)
            n += p.numel()
    return n


def parse_stages(spec: str) -> List[Tuple[int, float]]:
    """``"3:0.5,7:0.1"`` -> ``[(3, 0.5), (7, 0.1)]``."""
    out = []
    for part in [p for p in spec.split(",") if p.strip()]:
        layer, keep = part.split(":")
        out.append((int(layer), float(keep)))
    if not out:
        raise ValueError(f"empty stages spec {spec!r}")
    return out


def parse_freeze_schedule(spec: str) -> List[Tuple[int, float]]:
    """``"4:0.25,7:0.5"`` -> ``[(4, 0.25), (7, 0.5)]`` (blocks frozen, share of the optimizer steps)."""
    out = []
    for part in [p for p in spec.split(",") if p.strip()]:
        n, frac = part.split(":")
        out.append((int(n), float(frac)))
    if not out:
        raise ValueError(f"empty freeze_schedule {spec!r}")
    return out


def parse_mod_layers(spec: str) -> List[int]:
    """``"8-31:2"`` -> ``[8, 10, ..., 30]``; ``"8,12,16"`` and mixes (``"8-15:2,20"``) also work. Sorted, unique."""
    out: Set[int] = set()
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        rng, _, step = part.partition(":")
        if "-" in rng:
            lo, hi = (int(x) for x in rng.split("-"))
            out.update(range(lo, hi + 1, int(step or 1)))
        else:
            out.add(int(rng))
    if not out or min(out) <= 0:
        raise ValueError(f"mod_layers must list blocks > 0, got {spec!r}")
    return sorted(out)


def _row_ranks(x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Ranks of ``x`` within each row of ``rows``, scaled to [0, 1]."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for r in torch.unique(rows).tolist():
        m = rows == r
        n = int(m.sum())
        out[m] = x[m].argsort().argsort().to(torch.float32) / max(1, n - 1)
    return out


def _auc(score: torch.Tensor, pos: torch.Tensor) -> float:
    """P(score of a random positive > a random negative), ties half."""
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = score.to(torch.float64)
    gt = (s[pos][:, None] > s[~pos][None, :]).to(torch.float64).sum()
    eq = (s[pos][:, None] == s[~pos][None, :]).to(torch.float64).sum()
    return float((gt + 0.5 * eq).item() / (n1 * n0))


class OnlineTokenRouter:
    """See the module docstring. One instance per model; driven by the model's compaction hook
    (:meth:`select`), the embedding-output hook (:meth:`attach_target_hook`) and a trainer callback
    (:meth:`begin_step`, :meth:`metrics`)."""

    def __init__(
        self,
        cfg: OnlineRouterConfig,
        *,
        doc_start_id: int,
        doc_end_id: int,
        eos_id: int,
        d_model: int,
        gold_table: Optional[Dict[str, Any]] = None,
        rank: int = 0,
    ):
        if cfg.mod_layers and cfg.drop_layer == 0:
            import dataclasses

            cfg = dataclasses.replace(cfg, drop_layer=parse_mod_layers(cfg.mod_layers)[0])
        cfg.validate()
        self.stage_idx = 0
        self._next: Optional["OnlineTokenRouter"] = None
        if cfg.stages:
            import dataclasses

            st = parse_stages(cfg.stages)
            (l0, k0), rest = st[0], st[1:]
            cfg = dataclasses.replace(cfg, stages="", drop_layer=l0, keep_frac=k0)
            prev_k, prev = k0, self
            for i, (l, k) in enumerate(rest, start=1):
                # later stages: keep ``k`` of the ORIGINAL body = k / prev_k of the tokens entering; their own fit
                c = dataclasses.replace(cfg, drop_layer=l, keep_frac=min(1.0, k / prev_k), early_nograd=False,
                                        seed=cfg.seed + 1009 * i, keep_first=0)
                nxt = OnlineTokenRouter(c, doc_start_id=doc_start_id, doc_end_id=doc_end_id, eos_id=eos_id,
                                        d_model=d_model, gold_table=None, rank=rank)
                nxt.stage_idx = i
                prev._next, prev, prev_k = nxt, nxt, k
        if cfg.use_gold and gold_table is None:
            raise ValueError("online router use_gold needs the gold-fingerprint table")
        self.cfg = cfg
        self.doc_start_id, self.doc_end_id, self.eos_id = int(doc_start_id), int(doc_end_id), int(eos_id)
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.gold: Optional[Dict[str, List[int]]] = None
        if gold_table is not None:
            self.gold = {}
            for fp, val in gold_table.items():
                items = list(val)
                if items and isinstance(items[0], (list, tuple)):
                    items = [a for p in items for a in p]
                self.gold[fp] = sorted({int(a) for a in items})
        self.hdim = cfg.proj_dim if (cfg.proj_dim and cfg.use_hidden) else (self.d_model if cfg.use_hidden else 0)
        self.n_scalar = N_STRUCT + N_LEX + (1 if cfg.use_query and cfg.use_hidden else 0) + (1 if cfg.use_gold else 0)
        self.n_features = self.n_scalar + self.hdim * (1 + int(cfg.use_query))
        self.step = 0
        self.total_steps: Optional[int] = None  # set by the callback (dense_tail_frac)
        self._nograd_fwd = False  # this forward runs blocks [0, _nograd_until) forward-only (early_nograd / freeze)
        self._nograd_until = 0
        self.frozen_blocks = 0  # leading blocks frozen so far by freeze_schedule (set by the callback)
        self._exit: Optional[Dict[str, Any]] = None  # this forward's dropped-token hidden states for the exit loss
        self._calls = 0
        self.w: Optional[torch.Tensor] = None
        self.XtX: Optional[torch.Tensor] = None
        self.Xty: Optional[torch.Tensor] = None
        self.n_seen = 0.0
        # per-task routers (per_task_key_tokens > 0): task key -> ridge state / weights, and the cross-rank slot order
        self.task_w: Dict[int, torch.Tensor] = {}
        self.task_XtX: Dict[int, torch.Tensor] = {}
        self.task_Xty: Dict[int, torch.Tensor] = {}
        self.task_seen: Dict[int, float] = {}
        self._row_task: Optional[torch.Tensor] = None  # this forward's (B,) task keys
        self.mod_layers: List[int] = parse_mod_layers(cfg.mod_layers) if cfg.mod_layers else []
        self._mod: Optional[Dict[str, Any]] = None  # this forward's mixture-of-depths gather
        self._tap: Optional[Dict[str, Any]] = None  # target entry waiting for block target_layer
        self._proj: Optional[torch.Tensor] = None
        self._pending: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None
        self._stats: Dict[str, List[float]] = {}
        self.last_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------ naming
    _marker_free: bool = False

    def _doc_ids(self) -> Tuple[int, int]:
        """Marker ids as the model's chunk ids see them: none (-1) on positional-block rows."""
        return (-1, -1) if self._marker_free else (self.doc_start_id, self.doc_end_id)

    def feature_names(self) -> List[str]:
        names = ["bias"]
        for side in ("start", "end"):
            names += [f"{side}_off{i}" for i in range(_N_ONEHOT)]
            names += [f"{side}_log{i}" for i in range(_N_LOG)]
        names += ["rel_in_doc", "doc_rel_idx", "log_doc_len", "in_query", "log_repeat"]
        if self.cfg.use_query and self.cfg.use_hidden:
            names.append("cos_h_q")
        if self.cfg.use_gold:
            names.append("gold")
        names += [f"h{i}" for i in range(self.hdim)]
        if self.cfg.use_query:
            names += [f"hq{i}" for i in range(self.hdim)]
        return names

    # ------------------------------------------------------------------ schedule
    def dense_now(self) -> bool:
        """Whether this step trains uncompressed (warmup or a periodic dense step)."""
        c = self.cfg
        if self.w is None or self.step <= c.warmup_steps:
            return True
        if c.dense_tail_frac > 0:
            if not self.total_steps:
                raise RuntimeError("online router dense_tail_frac needs total_steps (OnlineRouterCallback sets it)")
            if self.step > int(round(self.total_steps * (1.0 - c.dense_tail_frac))):
                return True
        return c.dense_every > 0 and self.step % c.dense_every == 0

    # ------------------------------------------------------------------ features
    def _flop(self, model, n: float) -> None:
        model._flop_extra_actual = float(getattr(model, "_flop_extra_actual", 0.0)) + float(n)

    def _hidden(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """(B, T, D) hidden state at ``feature_layer`` on the full row, no grad."""
        with torch.no_grad():
            h = model.embeddings(input_ids)
            if model.embed_scale is not None:
                h = h * model.embed_scale
            if model.embedding_norm is not None:
                h = model.embedding_norm(h)
            L = self.cfg.feature_layer
            if L > 0:
                _, _, abk, pbk, _ = model._prepare_inputs(input_ids, None)
                blocks = list(model.blocks.values())
                if L > len(blocks):
                    raise ValueError(f"feature_layer {L} > {len(blocks)} blocks")
                for i in range(L):
                    h = blocks[i](h, **abk, **pbk.get(i, {}))
                # forward-only cost of the prefix (num_flops_per_token is forward+backward = 3x fwd)
                T = int(input_ids.shape[1])
                per_tok = sum(int(b.num_flops_per_token(T)) for b in blocks[:L]) / 3.0
                self._flop(model, per_tok * input_ids.numel())
        return h

    def _row_gold(self, input_ids: torch.Tensor, chunk_ids: torch.Tensor) -> torch.Tensor:
        """(B, T) float gold-document flag per token (0 where unknown)."""
        from .attention.gold_grad_mask import content_fingerprint_from_row

        assert self.gold is not None
        B, T = input_ids.shape
        cid = chunk_ids.to(torch.long)
        n_docs = max(1, int(cid.max().item()) + 1)
        g = torch.zeros(B, n_docs + 1, dtype=torch.float32)
        rows = input_ids.detach().to("cpu").tolist()
        hits = 0
        for b in range(B):
            gold = self.gold.get(content_fingerprint_from_row(rows[b], self.eos_id))
            if gold is None:
                continue
            hits += 1
            for d in gold:
                if 0 <= d < n_docs:
                    g[b, d] = 1.0
        self._stat("gold_fp_hit", hits / max(1, B))
        g = g.to(input_ids.device)
        return torch.where(cid >= 0, g.gather(1, cid.clamp(min=0)), torch.zeros_like(cid, dtype=torch.float32))

    def _projection(self, device) -> Optional[torch.Tensor]:
        if not self.cfg.proj_dim:
            return None
        if self._proj is None or self._proj.device != device:
            gen = torch.Generator(device="cpu").manual_seed(1234567 + self.cfg.seed)
            p = torch.randn(self.d_model, self.cfg.proj_dim, generator=gen) / math.sqrt(self.cfg.proj_dim)
            self._proj = p.to(device)
        return self._proj

    def token_features(
        self,
        model,
        input_ids: torch.Tensor,
        chunk_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        ignore_index: int = -100,
        hidden: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Per-token feature PARTS (scalars ``s`` (B,T,n_scalar); hidden ``h`` (B,T,hdim) and query
        ``q`` (B,hdim), both RMS-normalised) plus the ``body`` mask. ``hidden=False`` defers the
        hidden parts to :meth:`add_hidden` (mid-stack drop)."""
        c = self.cfg
        offs: Dict[str, torch.Tensor] = {}
        ds, de = self._doc_ids()
        struct, body = structural_features(
            input_ids, chunk_ids, doc_start_id=ds, doc_end_id=de, offsets=offs, marker_free=self._marker_free
        )
        B, T = input_ids.shape
        qm = query_mask(chunk_ids, labels, ignore_index)
        # lexical: token id occurs in the query; log repeat count of the id among the row's body tokens
        V = int(max(int(input_ids.max().item()) + 1, 1))
        rows = torch.arange(B, device=input_ids.device)[:, None].expand(B, T)
        key = rows * V + input_ids.to(torch.long)
        qset = torch.zeros(B * V, dtype=torch.bool, device=input_ids.device)
        qset[key[qm]] = True
        in_q = qset[key].to(torch.float32)
        cnt = torch.zeros(B * V, dtype=torch.float32, device=input_ids.device).index_add_(
            0, key[body], torch.ones(int(body.sum()), device=input_ids.device)
        )
        rep = torch.log1p(cnt[key]) / 5.0
        out: Dict[str, torch.Tensor] = {"body": body, "query": qm, "_edge": torch.minimum(offs["j"], offs["r"]),
                                        "_j": offs["j"]}
        out["_s0"] = torch.cat([struct, (in_q * body).unsqueeze(-1), (rep * body).unsqueeze(-1)], dim=-1)
        if c.use_gold:
            out["_gold"] = (self._row_gold(input_ids, chunk_ids) * body).unsqueeze(-1)
        if c.use_hidden and hidden:
            self.add_hidden(out, self._hidden(model, input_ids))
        elif not c.use_hidden:
            self._assemble(out)
        return out

    def add_hidden(self, out: Dict[str, torch.Tensor], h: torch.Tensor) -> None:
        """Add the hidden parts from a raw (B, T, D) hidden state and assemble ``s``."""
        c = self.cfg
        body, qm = out["body"], out["query"]
        h = h.detach().float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        P = self._projection(h.device)
        if P is not None:
            h = h @ P
        if c.use_query:
            nq = qm.sum(1).clamp(min=1).to(torch.float32)
            q = (h * qm.unsqueeze(-1)).sum(1) / nq[:, None]
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
            out["q"] = q
            out["_cos"] = ((h * q[:, None, :]).mean(-1) * body).unsqueeze(-1)
        out["h"] = h * body.unsqueeze(-1)
        self._assemble(out)

    @staticmethod
    def _assemble(out: Dict[str, torch.Tensor]) -> None:
        out["s"] = torch.cat([out[k] for k in ("_s0", "_cos", "_gold") if k in out], dim=-1)

    def full_features(self, fp: Dict[str, torch.Tensor], rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """Materialise the (N, F) feature rows of the selected tokens."""
        parts = [fp["s"][rows, cols]]
        if "h" in fp:
            h = fp["h"][rows, cols]
            parts.append(h)
            if "q" in fp:
                parts.append(h * fp["q"][rows])
        return torch.cat(parts, dim=-1)

    def score(self, fp: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, T) linear score without materialising the full feature matrix."""
        assert self.w is not None
        sc = self._score_with(fp, self.w)
        if self.task_w and self._row_task is not None:
            # per-task routers: a row whose task already has its own fit is scored by it (else by the pooled fit)
            for b, k in enumerate(self._row_task.tolist()):
                wk = self.task_w.get(int(k))
                if wk is not None:
                    sc[b] = self._score_with({n: fp[n][b : b + 1] for n in ("s", "h", "q") if n in fp}, wk)[0]
        return sc

    def _score_with(self, fp: Dict[str, torch.Tensor], w: torch.Tensor) -> torch.Tensor:
        w = w.to(torch.float32)
        ns, hd = self.n_scalar, self.hdim
        sc = fp["s"] @ w[:ns]
        if "h" in fp:
            wh = w[ns : ns + hd]
            if "q" in fp:
                wh = wh[None, :] + fp["q"] * w[ns + hd : ns + 2 * hd][None, :]  # (B, hd)
                sc = sc + torch.einsum("btd,bd->bt", fp["h"], wh)
            else:
                sc = sc + fp["h"] @ wh
        return sc

    def task_keys(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B,) int64 task key per row: a polynomial hash of its first ``per_task_key_tokens`` tokens."""
        N = min(int(self.cfg.per_task_key_tokens), int(input_ids.shape[1]))
        pw = torch.tensor([pow(1_000_003, i, 2**31 - 1) for i in range(N)], dtype=torch.int64, device=input_ids.device)
        return (input_ids[:, :N].to(torch.int64) * pw[None, :]).sum(dim=1)

    # ------------------------------------------------------------------ selection
    def select(
        self,
        model,
        input_ids: torch.Tensor,
        chunk_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """The (B, T) keep mask over body tokens for this training forward's INPUT compaction
        (non-body ignored). Stashes the feature rows of a sample of the kept tokens for
        :meth:`attach_target_hook`. With ``drop_layer > 0`` every body token is kept here and the
        choice is made at that layer by :meth:`mid_compact`."""
        self._calls += 1
        self._ignore_index = ignore_index
        self._row_task = self.task_keys(input_ids) if self.cfg.per_task_key_tokens > 0 else None
        # positional-block rows (CPT): documents are blocks with no markers (see structural_features)
        self._marker_free = int((getattr(model, "_pooled_soft_tokens", None) or {}).get("positional_block_size", 0)) > 0
        # trim the right pad: features / prefix pass only over the longest real row
        real = (chunk_ids != -2).sum(dim=1)
        Tr = int(real.max().item())
        ids, cid = input_ids[:, :Tr], chunk_ids[:, :Tr]
        lab = labels[:, :Tr] if labels is not None else None
        mid = self.cfg.drop_layer > 0
        fp = self.token_features(model, ids, cid, lab, ignore_index, hidden=not mid)
        full = torch.zeros_like(input_ids, dtype=torch.bool)
        if mid:
            self._mid = {"fp": fp, "ids": ids, "cid": cid, "Tr": Tr}
            L0 = self.cfg.drop_layer
            if model.training and self.frozen_blocks > 0:
                self._nograd_until = min(self.frozen_blocks, L0)
            elif model.training and (self.cfg.freeze_early or (self.cfg.early_nograd and not self.dense_now())):
                self._nograd_until = L0
            else:
                self._nograd_until = 0
            self._nograd_fwd = self._nograd_until > 0
            full[:, :Tr] = fp["body"]
            return full
        keep, entry = self._choose(model, fp, ids, cid)
        self._current = entry
        full[:, :Tr] = keep
        return full

    def _choose(self, model, fp, ids, cid) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """(keep (B, T) over body tokens, target-sample entry) from assembled feature parts."""
        c = self.cfg
        body = fp["body"]
        dev = ids.device
        gen = torch.Generator(device=dev).manual_seed(
            (self.cfg.seed * 1_000_003 + self.step * 7919 + self.rank * 104_729 + self._calls) % (2**62)
        )
        u = torch.rand(ids.shape, generator=gen, device=dev)
        n_body = body.sum(dim=1)
        dense = self.dense_now()
        kf = c.keep_frac
        if dense and c.dense_keep_frac < 1.0 and self.w is not None and self.step > c.warmup_steps:
            # the periodic "dense" step routes at dense_keep_frac instead of the full context (bounded peak memory)
            dense, kf = False, c.dense_keep_frac
            self._stat("dense_routed", 1.0)
        explore = torch.zeros_like(body)
        score = None
        if self.w is not None:
            score = self.score(fp)
            self._flop(model, 2.0 * self.n_features * float(body.sum()))
        if dense:
            keep = body.clone()
        else:
            assert score is not None
            if c.score == "random" or c.explore_frac >= 1.0:
                # random CONTROL: the same budget and allocation, uniform scores (the router is fit and
                # logged but never chooses a token)
                score, c_explore = u.clone(), 0.0
            elif c.score == "first_last":
                # fixed gold-blind rule (control): each document's first and last tokens, same allocation
                score, c_explore = -fp["_edge"].to(torch.float32) + 1e-3 * u, 0.0
            elif c.score == "dh":
                # fixed fit-free rule: residual change through blocks [0, L) (set by mid_compact)
                score, c_explore = fp["_dh"] + 1e-6 * u, 0.0
            else:
                c_explore = c.explore_frac
            k = torch.ceil(n_body.to(torch.float32) * kf).to(torch.long)
            k_e = torch.round(k.to(torch.float32) * c_explore).to(torch.long)
            # exploration: the k_e body tokens with the smallest uniform draw
            ru = torch.where(body, u, torch.full_like(u, 2.0))
            rank_u = ru.argsort(dim=1).argsort(dim=1)
            explore = body & (rank_u < k_e[:, None])
            # each document's first keep_first body tokens (its header / id span) are always kept
            forced = body & (fp["_j"] < c.keep_first) if c.keep_first > 0 else torch.zeros_like(body)
            if c.keep_first > 0:
                self._stat("forced_frac", float(forced.sum()) / max(1.0, float(body.sum())))
            keep = explore | forced
            kd = 0.0 if c.mass_target > 0 else kf * c.doc_share
            if c.mass_target > 0:
                keep = keep | self._mass_keep(score, body, cid, c.mass_target)
            if kd > 0:
                # per-document share: each document's own top ceil(kd * |body_d|) by score
                # (exploration tokens rank first, so they count against their document's share)
                from .attention.chunked_mask import FREE_CHUNK_ID, mark_doc_topk_tokens_free

                if kd >= 1.0:
                    keep = body.clone()
                else:
                    sc_d = torch.where(explore | forced, torch.full_like(score, 1e30), score)
                    ds, de = self._doc_ids()
                    out = mark_doc_topk_tokens_free(cid, ids, sc_d, doc_start_id=ds, doc_end_id=de, k=kd)
                    keep = keep | (body & (out == FREE_CHUNK_ID))
            if c.min_per_doc > 0:
                # per-document FLOOR: every document keeps at least its top min_per_doc body tokens by score (all of
                # a shorter one), counted against the budget -- tiny documents (strmatch: ~25 tokens) are otherwise
                # cut to their header
                from .attention.chunked_mask import FREE_CHUNK_ID, mark_doc_topk_tokens_free

                sc_f = torch.where(explore | forced, torch.full_like(score, 1e30), score)
                ds, de = self._doc_ids()
                out_f = mark_doc_topk_tokens_free(cid, ids, sc_f, doc_start_id=ds, doc_end_id=de,
                                                  k=float(c.min_per_doc))
                floor = body & (out_f == FREE_CHUNK_ID)
                self._stat("floor_extra_frac", float((floor & ~keep).sum()) / max(1.0, float(body.sum())))
                keep = keep | floor
            if c.mass_target > 0:
                # per-row cap (peak memory): exploration / forced first, then the floor, then by score
                cap = torch.ceil(n_body.to(torch.float32) * c.max_keep_frac).to(torch.long)
                pri = torch.where(explore | forced, torch.full_like(score, 2e30), score)
                if c.min_per_doc > 0:
                    pri = torch.where(floor & ~(explore | forced), torch.full_like(score, 1e30), pri)
                pri = torch.where(keep, pri, torch.full_like(score, -float("inf")))
                rank_c = (-pri).argsort(dim=1).argsort(dim=1)
                capped = keep & ~(rank_c < cap[:, None])
                self._stat("mass_capped_frac", float(capped.sum()) / max(1.0, float(body.sum())))
                keep = keep & ~capped
            else:
                # the rest of the budget by score over the whole row
                left = (k - keep.sum(dim=1)).clamp(min=0)
                sc = torch.where(body & ~keep, score, torch.full_like(score, -float("inf")))
                rank_s = (-sc).argsort(dim=1).argsort(dim=1)
                keep = keep | (body & ~keep & (rank_s < left[:, None]))
        # sample kept tokens for targets: every exploration token first, then random kept ones
        # (mixture-of-depths: every body token keeps a gradient at this layer, so the targets sample all of them)
        pool = body if self.mod_layers else keep
        pri = torch.where(pool, u + 2.0 * (~explore).to(u.dtype), torch.full_like(u, 9.0))
        rank_p = pri.argsort(dim=1).argsort(dim=1)
        samp = pool & (rank_p < c.samples_per_row)
        rows, cols = samp.nonzero(as_tuple=True)
        entry: Dict[str, Any] = {
            "rows": rows,
            "pos": cols,
            "feats": self.full_features(fp, rows, cols),
            "explore": explore[rows, cols],
            "pred": (score[rows, cols] if score is not None else None),
            "target": None,
            "dense": dense,
            "task": (self._row_task[rows] if self._row_task is not None else None),
        }
        # diagnostics: realised keep, gold leak (does per-document kept density name gold?)
        self._stat("keep_frac", float(keep.sum()) / max(1.0, float(body.sum())))
        self._stat("dense", float(dense))
        if self.gold is not None and not dense:
            gflag = fp["_gold"][..., 0] if "_gold" in fp else self._row_gold(ids, cid)
            self._stat("leak_auc", self._leak_auc(keep, body, cid, gflag))
        return keep, entry

    def mid_compact(self, model, h: torch.Tensor, block_kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Mid-stack drop at block ``drop_layer``: score the forward's own hidden state, gather the kept
        tokens (plus every non-body token) into a shorter row, and re-key ``position_ids``. Records the
        gather for :meth:`mid_labels`, the target hook at this layer, and the FLOP adjustment (the meter
        charges the full row through every block; the blocks from here on ran on the kept row only)."""
        st, self._mid = getattr(self, "_mid", None), None
        self._mid_gather = None
        self._mod = None
        self._tap = None
        if st is None:
            return h, block_kwargs
        fp, ids, cid, Tr = st["fp"], st["ids"], st["cid"], st["Tr"]
        if h.shape[1] < Tr:
            raise RuntimeError(f"mid_compact: hidden width {h.shape[1]} < real width {Tr}")
        B, Tw, D = h.shape
        if self.cfg.use_hidden:
            self.add_hidden(fp, h[:, :Tr])
        dh_all = None
        if self.cfg.score == "dh" or self.cfg.target in ("dh", "mix"):
            # residual change of every real token through blocks [0, L), on the full row (no grad)
            with torch.no_grad():
                e0 = model.embeddings(ids)
                if model.embed_scale is not None:
                    e0 = e0 * model.embed_scale
                dh_all = torch.log((h[:, :Tr].float() - e0.float()).norm(dim=-1) + 1e-12)
            fp["_dh"] = dh_all
            self._flop(model, 3.0 * self.d_model * float((cid != -2).sum()))
        keep_body, entry = self._choose(model, fp, ids, cid)
        body = fp["body"]
        real = cid != -2
        keep_all = torch.zeros(B, Tw, dtype=torch.bool, device=h.device)
        keep_all[:, :Tr] = real & (~body | keep_body)
        counts = keep_all.sum(dim=1)
        pm = int((getattr(model, "_pooled_soft_tokens", None) or {}).get("pad_multiple", 1) or 1)
        pid = block_kwargs.get("position_ids")
        if pid is None:
            pid = torch.arange(Tw, device=h.device)[None, :].expand(B, Tw)
        kw = dict(block_kwargs)
        rows, pos = entry["rows"], entry["pos"]
        n_extra = 0
        if self.cfg.merge_dropped and not entry["dense"]:
            # one summary token per document: the mean of its dropped tokens' hidden states, slotted right after the
            # document's last dropped token (sort key column + 0.5); the source row is h extended by the summaries
            dropped = real & body & ~keep_body
            nd = int(cid.max().item()) + 1 if bool((cid >= 0).any()) else 1
            dk = (torch.arange(B, device=h.device)[:, None] * nd + cid.clamp(min=0).long())[dropped]
            hd = h[:, :Tr][dropped].float()
            sums = torch.zeros(B * nd, D, device=h.device, dtype=torch.float32).index_add(0, dk, hd)
            cnt = torch.zeros(B * nd, device=h.device).index_add(0, dk, torch.ones_like(dk, dtype=torch.float32))
            col = torch.arange(Tr, device=h.device)[None, :].expand(B, Tr)
            last = torch.full((B * nd,), -1, device=h.device, dtype=torch.long).scatter_reduce(
                0, dk, col[dropped], reduce="amax")
            has = cnt > 0
            summ = (sums / cnt.clamp(min=1)[:, None]).to(h.dtype).view(B, nd, D)
            key_tok = torch.where(keep_all, torch.arange(Tw, device=h.device, dtype=torch.float32)[None, :].expand(B, Tw),
                                  torch.full((B, Tw), float("inf"), device=h.device))
            key_sum = torch.where(has, last.to(torch.float32) + 0.5, torch.full_like(last, 0, dtype=torch.float32)
                                  + float("inf")).view(B, nd)
            n_sum = has.view(B, nd).sum(dim=1)
            counts = counts + n_sum
            src = torch.cat([h, summ], dim=1)
            pid_src = torch.cat([pid, torch.gather(pid[:, :Tr], 1, last.view(B, nd).clamp(min=0))], dim=1)
            keys = torch.cat([key_tok, key_sum], dim=1)
            n_extra = nd
            self._stat("merge_tokens", float(n_sum.sum()) / B)
            self._flop(model, 2.0 * D * float(dropped.sum()))
        else:
            src, pid_src = h, pid
            keys = torch.where(keep_all, torch.zeros(B, Tw, device=h.device), torch.ones(B, Tw, device=h.device))
        W = int(counts.max().item())
        W = min(src.shape[1], ((W + pm - 1) // pm) * pm)
        order = torch.argsort(keys, dim=1, stable=True)[:, :W]
        valid = torch.arange(W, device=h.device)[None, :] < counts[:, None]
        h_new = torch.gather(src, 1, order[..., None].expand(B, W, D)) * valid[..., None].to(h.dtype)
        pnew = torch.gather(pid_src, 1, order)
        kw["position_ids"] = torch.where(valid, pnew, pid.max(dim=1, keepdim=True).values.expand_as(pnew))
        if self.mod_layers:
            return self._mod_begin(model, h, block_kwargs, entry, real, counts, order, valid, kw["position_ids"])
        self._mid_gather = (order, valid, n_extra)
        # targets at this layer, on the kept (gathered) columns
        newcol = torch.zeros(B, src.shape[1], dtype=torch.long, device=h.device)
        newcol.scatter_(1, order, torch.arange(W, device=h.device)[None, :].expand(B, W))
        entry["ok"] = keep_all[rows, pos]
        entry["cols"] = newcol[rows, pos]
        if dh_all is not None:
            entry["dh"] = dh_all[rows, pos]
        if self.cfg.target_layer:
            # targets from a deeper block: on routed steps only the exploration tokens (uniform, kept) are uncensored
            if not entry["dense"]:
                entry["ok"] = entry["ok"] & entry["explore"]
            self._tap = entry
        else:
            self._register_target(model, entry, h_new)
        self._exit = None
        if self.cfg.exit_loss_weight > 0 and not entry["dense"]:
            dropped = real & body & ~keep_body
            if bool(dropped.any()):
                # their hidden states at the split (with grad: the exit loss trains blocks [0, L)); labels come in mid_labels
                self._exit = {"h": h[:, :Tr][dropped], "mask": dropped, "Tr": Tr}
                head = int(model.lm_head.num_flops_per_token(0)) if model.lm_head is not None else 0
                self._flop(model, float(head) * float(dropped.sum()))
                self._stat("exit_tokens_frac", float(dropped.sum()) / max(1.0, float(real.sum())))
        # FLOP adjustment, per row: blocks before drop_layer at the full length, the rest + head at the
        # kept length, minus the dense charge the meter applies to the (uncompacted) row
        # (blocks skipped whole this step by layer drop ran for no token: the meter already leaves them out of
        # its charge, so they are left out of both sides here)
        ld = getattr(model, "_layer_drop", None)
        skip = ld.skipped() if (ld is not None and model.training) else frozenset()
        if self.cfg.drop_layer in skip:
            raise RuntimeError(f"online router drop_layer {self.cfg.drop_layer} is skipped by layer drop")
        blocks = [b if i not in skip else None for i, b in enumerate(model.blocks.values())]
        L0 = self.cfg.drop_layer
        head = int(model.lm_head.num_flops_per_token(0)) if model.lm_head is not None else 0
        adj = 0.0
        # early_nograd / freeze: blocks below _nograd_until ran forward-only (1/3 of forward + backward); frozen blocks
        # at or above L0 skipped only their weight gradient (2/3)
        fac = self._block_factors(len(blocks))
        self._stat("early_nograd", float(self._nograd_fwd))
        if self.stage_idx == 0 and model.training:
            adj += self._lora_charge(model) * float(real.sum())
        for L, Lk in zip(real.sum(dim=1).tolist(), counts.tolist()):
            fb = [int(b.num_flops_per_token(int(L))) if b is not None else 0 for b in blocks]
            fk = [int(b.num_flops_per_token(int(Lk))) if b is not None else 0 for b in blocks[L0:]]
            if self.stage_idx == 0:
                actual = (L * sum(f * x for f, x in zip(fac[:L0], fb[:L0]))
                          + Lk * (sum(f * x for f, x in zip(fac[L0:], fk)) + head))
                adj += actual - L * (sum(fb) + head)
            else:
                # a later pyramid stage: the earlier stages already charged blocks >= L0 at THIS stage's entry length L
                adj += Lk * (sum(fk) + head) - L * (sum(fb[L0:]) + head)
        self._flop(model, adj)
        self._stat("mid_row_frac", float(counts.sum()) / max(1.0, float(real.sum())))
        if self._next is not None:
            self._next._mid = self._stash_for_next(fp, ids, cid, Tr, Tw, order, valid)
            self._next._nograd_fwd = False
            self._next._nograd_until = 0
            self._next._ignore_index = getattr(self, "_ignore_index", -100)
        return h_new, kw

    def _mod_begin(self, model, h, block_kwargs, entry, real, counts, order, valid, pos_c):
        """Mixture-of-depths at the first listed block: keep the full row, remember the gather the listed blocks run
        on, read the targets at this block over every sampled token, and charge the listed blocks at the kept length."""
        dense = bool(entry["dense"]) or bool((counts == real.sum(dim=1)).all())
        self._mod = None if dense else {"order": order, "valid": valid, "pos": pos_c}
        rows, pos = entry["rows"], entry["pos"]
        entry["ok"] = real[rows, pos]
        entry["cols"] = pos
        if self.cfg.target_layer:
            self._tap = entry  # every token reaches the deeper block under mixture-of-depths
        else:
            self._register_target(model, entry, h)
        self._exit = None
        ld = getattr(model, "_layer_drop", None)
        skip = ld.skipped() if (ld is not None and model.training) else frozenset()
        blocks = [b if i not in skip else None for i, b in enumerate(model.blocks.values())]
        L0, mods = self.cfg.drop_layer, [i for i in self.mod_layers if i < len(blocks)]
        fac = self._block_factors(len(blocks))
        self._stat("early_nograd", float(self._nograd_fwd))
        adj = self._lora_charge(model) * float(real.sum()) if model.training else 0.0
        ms = set(mods)
        for L, Lk in zip(real.sum(dim=1).tolist(), counts.tolist()):
            if dense:
                Lk = L
            fb = [int(b.num_flops_per_token(int(L))) if b is not None else 0 for b in blocks]
            for i, b in enumerate(blocks):
                if b is None:
                    continue
                if i in ms:
                    adj += fac[i] * Lk * int(b.num_flops_per_token(int(Lk))) - L * fb[i]
                else:
                    adj += (fac[i] - 1.0) * L * fb[i]
        self._flop(model, adj)
        self._stat("mod_row_frac", float(counts.sum()) / max(1.0, float(real.sum())) if not dense else 1.0)
        return h, block_kwargs

    @staticmethod
    def _mass_keep(score: torch.Tensor, body: torch.Tensor, cid: torch.Tensor, share: float) -> torch.Tensor:
        """(B, T) per-document minimal top-score set whose predicted mass exp(2 x score) covers ``share``."""
        out = torch.zeros_like(body)
        B, T = body.shape
        for b in range(B):
            idx = body[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            d = cid[b, idx].to(torch.long)
            sc = score[b, idx].to(torch.float64)
            nd = int(d.max()) + 1
            mx = torch.full((nd,), -float("inf"), device=sc.device, dtype=sc.dtype).scatter_reduce(0, d, sc,
                                                                                                    reduce="amax")
            m = torch.exp(2.0 * (sc - mx[d]))
            tot = torch.zeros(nd, device=sc.device, dtype=sc.dtype).index_add(0, d, m)
            # group by document, score descending within it
            r = (-sc).argsort().argsort().to(torch.float64)
            order = (d.to(torch.float64) * (idx.numel() + 1) + r).argsort()
            ms, ds = m[order], d[order]
            cs = ms.cumsum(0)
            start = torch.zeros(nd, device=sc.device, dtype=sc.dtype).index_add(0, ds, ms)  # per-doc totals
            offs = torch.cumsum(start, 0) - start
            before = cs - ms - offs[ds]  # predicted mass of the doc's higher-scored tokens
            sel = before < (share - 1e-9) * tot[ds]  # tolerance: the offset subtraction leaves a rounding residue
            out[b, idx[order[sel]]] = True
        return out

    @staticmethod
    def lora_flops_per_token(model) -> float:
        st = getattr(model, "_early_lora", None)
        if not st:
            return 0.0
        if "fpt" not in st:
            from .early_lora import early_lora_flops_per_token

            st["fpt"] = early_lora_flops_per_token(model)
        return float(st["fpt"])

    def _lora_charge(self, model) -> float:
        """Adapter FLOPs per token for this forward: forward + backward, or forward only (1/3) once frozen."""
        return self.lora_flops_per_token(model) * (1.0 / 3.0 if self._nograd_fwd else 1.0)

    def _block_factors(self, n_blocks: int) -> List[float]:
        """Per-block share of the forward + backward cost this forward pays: 1/3 below ``_nograd_until`` (forward
        only), 2/3 for frozen blocks at or above the drop layer (no weight gradient), else 1."""
        nu, L0, P = self._nograd_until, self.cfg.drop_layer, self.frozen_blocks
        lora = self.cfg.early_lora > 0 and self.stage_idx == 0
        return [1.0 / 3.0 if i < nu else (2.0 / 3.0 if (L0 <= i < P or (lora and i < L0)) else 1.0)
                for i in range(n_blocks)]

    def tap_target(self, model, h: torch.Tensor, block_idx: int) -> torch.Tensor:
        """At the input of block ``target_layer``: register this forward's pending target hook there."""
        e, self._tap = self._tap, None
        if e is not None and block_idx == self.cfg.target_layer and h.requires_grad:
            self._register_target(model, e, h)
        elif e is not None:
            self._tap = e
        return h

    def mod_block(self, block, h: torch.Tensor, kwargs: Dict[str, Any]) -> torch.Tensor:
        """Run one LISTED block under mixture-of-depths: on the kept tokens only (gathered at their original
        positions), scattered back; every other token passes the residual through unchanged."""
        m = self._mod
        if m is None:
            return block(h, **kwargs)
        if kwargs.get("cu_doc_lens") is not None:
            raise RuntimeError("online router mod_layers: packed rows (cu_doc_lens) are not supported")
        order, valid = m["order"], m["valid"]
        B, W = order.shape
        D = h.shape[-1]
        idx = order[..., None].expand(B, W, D)
        hc = torch.gather(h, 1, idx)
        kw = dict(kwargs)
        kw["position_ids"] = m["pos"]
        out = block(hc * valid[..., None].to(hc.dtype), **kw)
        # padded slots of the gather hold dropped tokens' columns: write their own (unchanged) values back
        return h.scatter(1, idx, torch.where(valid[..., None], out.to(h.dtype), hc))

    def chain(self) -> List["OnlineTokenRouter"]:
        """This router and its later pyramid stages, in order."""
        out, r = [], self
        while r is not None:
            out.append(r)
            r = r._next
        return out

    @staticmethod
    def _stash_for_next(fp, ids, cid, Tr, Tw, order, valid) -> Dict[str, Any]:
        """The next pyramid stage's feature stash: every per-token part gathered to this stage's kept row."""
        B, W = order.shape

        def g(x, fill):
            pad = torch.full((B, Tw) + tuple(x.shape[2:]), fill, dtype=x.dtype, device=x.device)
            pad[:, :Tr] = x
            idx = order.to(x.device).view(B, W, *([1] * (x.dim() - 2))).expand(B, W, *x.shape[2:])
            out = torch.gather(pad, 1, idx)
            v = valid.to(x.device).view(B, W, *([1] * (x.dim() - 2)))
            return torch.where(v, out, torch.full_like(out, fill))

        fp2 = {}
        for k, v in fp.items():
            if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == B and v.shape[1] == Tr:
                fp2[k] = g(v, False if v.dtype == torch.bool else 0)
        return {"fp": fp2, "ids": g(ids, 0), "cid": g(cid, -2), "Tr": W}

    def mid_labels(self, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Apply this forward's mid-stack gather(s) -- every pyramid stage in order -- to the (input-compacted) labels."""
        for r in self.chain():
            labels = r._own_mid_labels(labels)
        return labels

    def pop_exit(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``(h (N, D), labels (N,))`` of this forward's dropped tokens for the early-exit loss, or None."""
        e, self._exit = self._exit, None
        if e is None or "labels" not in e:
            return None
        return e["h"], e["labels"]

    def _own_mid_labels(self, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self._exit is not None and labels is not None:
            # the dropped tokens' labels, read before the gather removes them
            self._exit["labels"] = labels[:, : self._exit["Tr"]][self._exit["mask"].to(labels.device)]
        g, self._mid_gather = getattr(self, "_mid_gather", None), None
        if g is None or labels is None:
            return labels
        order, valid, n_extra = g
        ign = int(getattr(self, "_ignore_index", -100))
        if n_extra:  # merge_dropped summary slots carry no label
            labels = torch.cat([labels, torch.full((labels.shape[0], n_extra), ign, dtype=labels.dtype,
                                                   device=labels.device)], dim=1)
        lab = torch.gather(labels, 1, order.to(labels.device))
        return torch.where(valid.to(labels.device), lab, torch.full_like(lab, int(getattr(self, "_ignore_index", -100))))

    def _leak_auc(self, keep, body, cid, gflag) -> float:
        """Mean over rows of AUC(per-document kept fraction -> gold). 0.5 = density names nothing."""
        aucs = []
        c = cid.to(torch.long)
        for b in range(keep.shape[0]):
            bb = body[b]
            if not bb.any():
                continue
            d = c[b][bb]
            nd = int(d.max()) + 1
            kept = torch.zeros(nd, device=d.device).index_add_(0, d, keep[b][bb].to(torch.float32))
            tot = torch.zeros(nd, device=d.device).index_add_(0, d, torch.ones_like(d, dtype=torch.float32))
            g = torch.zeros(nd, device=d.device).index_add_(0, d, gflag[b][bb]) > 0
            present = tot > 0
            a = _auc((kept / tot.clamp(min=1))[present], g[present])
            if not math.isnan(a):
                aucs.append(a)
        return float(sum(aucs) / len(aucs)) if aucs else float("nan")

    # ------------------------------------------------------------------ targets
    def attach_target_hook(self, model, h: torch.Tensor, position_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Register the target hook on the compacted embedding output ``h`` (B, T', D); its original
        positions are ``position_ids`` (B, T'). No-op if this forward did not route."""
        entry, self._current = self._current, None
        if entry is None or not h.requires_grad:
            return h
        rows, pos = entry["rows"], entry["pos"]
        if position_ids is None:
            cols = pos
            ok = pos < h.shape[1]
        else:
            pid = position_ids.to(pos.device)
            cols = torch.searchsorted(pid[rows].contiguous(), pos[:, None].contiguous()).squeeze(-1)
            cols = cols.clamp(max=pid.shape[1] - 1)
            ok = pid[rows, cols] == pos
        entry["ok"] = ok
        entry["cols"] = cols
        self._register_target(model, entry, h)
        return h

    def _register_target(self, model, entry: Dict[str, Any], h: torch.Tensor) -> None:
        rows, cols = entry["rows"], entry["cols"]
        if self.cfg.target == "gxh":
            entry["hsel"] = h.detach()[rows, cols].float()
        self._pending.append(entry)
        n_sel = int(rows.numel())
        self._flop(model, 2.0 * self.d_model * n_sel * (2 if self.cfg.target == "gxh" else 1))

        def hook(g: torch.Tensor) -> None:
            gs = g.detach()[rows, cols].float()
            gn = torch.log(gs.norm(dim=-1) + 1e-12)
            entry["gn"] = gn
            tgt = self.cfg.target
            if tgt == "gnorm":
                entry["target"] = gn
            elif tgt == "gxh":
                entry["target"] = torch.log((gs * entry["hsel"]).sum(-1).abs() + 1e-12)
            elif tgt == "dh":
                entry["target"] = entry["dh"]
            else:  # mix: within-row average of the two ranks
                entry["target"] = 0.5 * (_row_ranks(gn, rows) + _row_ranks(entry["dh"], rows))

        h.register_hook(hook)
        return h

    # ------------------------------------------------------------------ fitting
    def _stat(self, k: str, v: float) -> None:
        if not math.isnan(v):
            self._stats.setdefault(k, []).append(float(v))

    def begin_step(self, step: int, model=None) -> None:
        """Consume the previous step's targets (all ranks: one all-reduce), refit, advance."""
        import torch.distributed as dist

        F = self.n_features
        if self.step == 0:
            self._pending = []  # forwards before the first step = the trainer's dry-run batch
        X, y, rows_all, expl, preds, dense_e, gns, tks = [], [], [], [], [], [], [], []
        for e in self._pending:
            if e["target"] is None:
                continue
            ok = e["ok"]
            if not ok.any():
                continue
            t = e["target"][ok]
            r = e["rows"][ok]
            # per-row centring (only the within-row ranking is fit)
            nb = int(r.max()) + 1
            s = torch.zeros(nb, device=t.device).index_add_(0, r, t)
            n = torch.zeros(nb, device=t.device).index_add_(0, r, torch.ones_like(t))
            t = t - (s / n.clamp(min=1))[r]
            X.append(e["feats"][ok])
            y.append(t)
            if e.get("task") is not None:
                tks.append(e["task"][ok])
            rows_all.append(r + 10_000 * len(rows_all))
            expl.append(e["explore"][ok])
            if e["pred"] is not None:
                preds.append(e["pred"][ok])
            dense_e.append(torch.full_like(t, float(bool(e.get("dense")))).bool())
            g = e.get("gn", e["target"])[ok]
            gns.append(g - (torch.zeros(nb, device=g.device).index_add_(0, r, g) / n.clamp(min=1))[r])
        self._pending = []
        dev = None
        if model is not None:
            dev = next(model.parameters()).device
        elif X:
            dev = X[0].device
        if dev is None:
            return
        if X:
            Xc = torch.cat(X).to(dev, torch.float64)
            yc = torch.cat(y).to(dev, torch.float64)
        else:
            Xc = torch.zeros(0, F, dtype=torch.float64, device=dev)
            yc = torch.zeros(0, dtype=torch.float64, device=dev)
        # online held-out quality of the CURRENT weights (before refitting on these targets)
        if preds and len(preds) == len(X):
            pc = torch.cat(preds).to(dev, torch.float64)
            ec = torch.cat(expl).to(dev)
            rc = torch.cat(rows_all).to(dev)
            for name, m in (("rank_corr_explore", ec), ("rank_corr_kept", ~ec)):
                vals = []
                for rid in torch.unique(rc[m]).tolist():
                    mm = m & (rc == rid)
                    vals.append(_spearman(pc[mm], yc[mm]))
                vals = [v for v in vals if not math.isnan(v)]
                if vals:
                    self._stat(name, sum(vals) / len(vals))
            # gradient-fidelity estimate on FULL-CONTEXT (dense) steps: of the full-pass squared
            # embedding-gradient mass, the share the current scorer's top keep_frac would keep (row-level
            # top-k over a uniform token sample), vs the oracle top-k and random (= keep_frac)
            dc = torch.cat(dense_e).to(dev)
            if bool(dc.any()):
                # held-out rank correlation on FULL-CONTEXT steps: there the sampled tokens are a uniform sample of
                # every body token, so this is unbiased for any explore_frac (including 0)
                vals = []
                for rid in torch.unique(rc[dc]).tolist():
                    mm = dc & (rc == rid)
                    vals.append(_spearman(pc[mm], yc[mm]))
                vals = [v for v in vals if not math.isnan(v)]
                if vals:
                    self._stat("rank_corr_fullctx", sum(vals) / len(vals))
                fr, fo = [], []
                gc = torch.cat(gns).to(dev, torch.float64)  # the gradient norm, whatever the router's target
                for rid in torch.unique(rc[dc]).tolist():
                    mm = dc & (rc == rid)
                    yy, pp = gc[mm], pc[mm]
                    if yy.numel() < 10:
                        continue
                    k = max(1, int(math.ceil(self.cfg.keep_frac * yy.numel())))
                    w = torch.exp(2.0 * (yy - yy.max()))
                    fr.append(float(w[pp.argsort(descending=True)[:k]].sum() / w.sum()))
                    fo.append(float(w[yy.argsort(descending=True)[:k]].sum() / w.sum()))
                if fr:
                    self._stat("fid_mass_router", sum(fr) / len(fr))
                    self._stat("fid_mass_oracle", sum(fo) / len(fo))
                    self._stat("fid_mass_random", self.cfg.keep_frac)
        G = Xc.T @ Xc
        b = Xc.T @ yc
        cnt = torch.tensor([float(Xc.shape[0])], dtype=torch.float64, device=dev)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(G)
            dist.all_reduce(b)
            dist.all_reduce(cnt)
        if model is not None:
            self._flop(model, 2.0 * F * F * Xc.shape[0] + 2.0 * F * Xc.shape[0])
        if self.XtX is None:
            self.XtX, self.Xty = G, b
        else:
            self.XtX = self.cfg.decay * self.XtX + G
            self.Xty = self.cfg.decay * self.Xty + b
        self.n_seen = self.cfg.decay * self.n_seen + float(cnt.item())
        self._stat("fit_samples", float(cnt.item()))
        if self.n_seen > 0:
            A = self.XtX.clone()
            lam = self.cfg.ridge * float(torch.diagonal(A).mean().clamp(min=1e-12))
            A.diagonal().add_(lam)
            try:
                Lc = torch.linalg.cholesky(A)
                self.w = torch.cholesky_solve(self.Xty[:, None], Lc).squeeze(-1).to(torch.float32)
            except RuntimeError as exc:  # pragma: no cover - numerically singular
                log.warning("[online-router] ridge solve failed (%s); keeping previous weights", exc)
            if model is not None:
                self._flop(model, F**3 / 3.0 + 4.0 * F * F)
        if self.cfg.per_task_key_tokens > 0:
            self._fit_per_task(model, Xc, yc, torch.cat(tks).to(dev) if len(tks) == len(X) and tks else None, dev)
        self.step = int(step)

    def _fit_per_task(self, model, Xc: torch.Tensor, yc: torch.Tensor, tk: Optional[torch.Tensor], dev) -> None:
        """One ridge fit per task key (same decay / ridge as the pooled fit; a task's statistics decay only on the
        steps it is present). Every rank walks the same sorted union of keys, so the all-reduces line up."""
        import torch.distributed as dist

        F = self.n_features
        local = sorted({int(k) for k in tk.tolist()}) if tk is not None and tk.numel() else []
        if dist.is_available() and dist.is_initialized():
            gathered: List[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            keys = sorted({k for g in gathered for k in (g or [])})
        else:
            keys = local
        for k in keys:
            m = (tk == k) if tk is not None else torch.zeros(Xc.shape[0], dtype=torch.bool, device=dev)
            Xk, yk = Xc[m], yc[m]
            G, b = Xk.T @ Xk, Xk.T @ yk
            cnt = torch.tensor([float(Xk.shape[0])], dtype=torch.float64, device=dev)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(G)
                dist.all_reduce(b)
                dist.all_reduce(cnt)
            if k in self.task_XtX:
                self.task_XtX[k] = self.cfg.decay * self.task_XtX[k] + G
                self.task_Xty[k] = self.cfg.decay * self.task_Xty[k] + b
                self.task_seen[k] = self.cfg.decay * self.task_seen[k] + float(cnt.item())
            else:
                self.task_XtX[k], self.task_Xty[k], self.task_seen[k] = G, b, float(cnt.item())
            A = self.task_XtX[k].clone()
            lam = self.cfg.ridge * float(torch.diagonal(A).mean().clamp(min=1e-12))
            A.diagonal().add_(lam)
            try:
                Lc = torch.linalg.cholesky(A)
                self.task_w[k] = torch.cholesky_solve(self.task_Xty[k][:, None], Lc).squeeze(-1).to(torch.float32)
            except RuntimeError as exc:  # pragma: no cover - numerically singular
                log.warning("[online-router] task %d ridge solve failed (%s); keeping its previous weights", k, exc)
            if model is not None:
                self._flop(model, 2.0 * F * F * Xk.shape[0] + 2.0 * F * Xk.shape[0] + F**3 / 3.0 + 4.0 * F * F)
        self._stat("task_routers", float(len(self.task_w)))

    def metrics(self) -> Dict[str, float]:
        """Means of the per-forward diagnostics since the last call (then reset)."""
        pre = "online_router/" if self.stage_idx == 0 else f"online_router/s{self.stage_idx}_"
        out = {f"{pre}{k}": sum(v) / len(v) for k, v in self._stats.items() if v}
        self._stats = {}
        if self.w is not None:
            w = self.w
            ns = self.n_scalar
            out["online_router/w_norm_scalar"] = float(w[:ns].norm())
            out["online_router/w_norm_hidden"] = float(w[ns:].norm()) if w.numel() > ns else 0.0
            if self.cfg.use_gold:
                out["online_router/w_gold"] = float(w[ns - 1])
            names = self.feature_names()
            out["online_router/w_in_query"] = float(w[names.index("in_query")])
        self.last_metrics = out
        return out

    def weights_digest(self) -> str:
        """Short hash of the current weights (cross-rank consistency check)."""
        if self.w is None:
            return "none"
        return hashlib.sha1(self.w.detach().cpu().numpy().tobytes()).hexdigest()[:12]
