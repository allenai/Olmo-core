"""
Compact logit summaries for checking that two implementations (e.g. OLMo Core on GPU and MaxText
on TPU) compute the same function from converted weights.

Full logits are ``batch * seq * vocab`` floats (about 1.6 GB per 4096-token sequence with the
100k vocab), so a summary keeps the full distribution only at a few positions and, everywhere,
the next-token log-probability and the argmax. That's enough to compare per-token loss over long
sequences and KL / top-1 agreement where it matters.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = ["LogitSummary", "summarize_logits", "concat", "compare_summaries", "default_positions"]


@dataclass
class LogitSummary:
    tokens: np.ndarray
    """``[B, S]`` input token IDs."""

    positions: np.ndarray
    """``[P]`` sequence positions at which the full logits are kept."""

    logits: np.ndarray
    """``[B, P, V]`` full logits (fp32) at ``positions``."""

    next_token_logprob: np.ndarray
    """``[B, S - 1]`` log p(tokens[t + 1] | tokens[:t + 1])."""

    argmax: np.ndarray
    """``[B, S]`` argmax of the logits at every position."""

    def save(self, path: str) -> None:
        np.savez(path, **self.__dict__)

    @classmethod
    def load(cls, path: str) -> "LogitSummary":
        with np.load(path) as f:
            return cls(**{k: f[k] for k in cls.__dataclass_fields__})


def default_positions(seq_len: int, n: int = 16) -> np.ndarray:
    """``n`` positions spread over the sequence, always including the first and last."""
    return np.unique(np.linspace(0, seq_len - 1, num=min(n, seq_len)).round().astype(np.int64))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def summarize_logits(
    tokens: np.ndarray, logits: np.ndarray, positions: Optional[np.ndarray] = None
) -> LogitSummary:
    """
    Summarize ``[B, S, V]`` logits for ``[B, S]`` tokens. Call it per sequence (``B = 1``) and
    :func:`concat` the results if a whole batch of logits doesn't fit in memory.
    """
    B, S, _ = logits.shape
    if positions is None:
        positions = default_positions(S)
    next_lp = np.empty((B, S - 1), dtype=np.float32)
    for b in range(B):
        lp = _log_softmax(logits[b, :-1])
        next_lp[b] = lp[np.arange(S - 1), tokens[b, 1:]]
    return LogitSummary(
        tokens=np.asarray(tokens, dtype=np.int32),
        positions=np.asarray(positions, dtype=np.int64),
        logits=np.asarray(logits[:, positions], dtype=np.float32),
        next_token_logprob=next_lp,
        argmax=np.asarray(logits.argmax(-1), dtype=np.int32),
    )


def concat(summaries: Sequence[LogitSummary]) -> LogitSummary:
    first = summaries[0]
    for s in summaries[1:]:
        if not np.array_equal(s.positions, first.positions):
            raise ValueError("summaries must keep the same positions")
    return LogitSummary(
        tokens=np.concatenate([s.tokens for s in summaries]),
        positions=first.positions,
        logits=np.concatenate([s.logits for s in summaries]),
        next_token_logprob=np.concatenate([s.next_token_logprob for s in summaries]),
        argmax=np.concatenate([s.argmax for s in summaries]),
    )


def compare_summaries(reference: LogitSummary, actual: LogitSummary) -> Dict[str, float]:
    """
    Metrics for how closely ``actual`` reproduces ``reference``:

    - ``max_rel_logit_err``: max |Δlogit| / max |reference logit| at the kept positions
    - ``mean_kl``, ``max_kl``: KL(reference || actual) per kept position
    - ``top1_agreement``: fraction of all positions with the same argmax
    - ``mean_abs_dloss``, ``max_abs_dloss``: |Δ next-token log-prob| over all positions
    - ``ref_loss``, ``actual_loss``: mean next-token cross entropy of each
    """
    if not np.array_equal(reference.tokens, actual.tokens):
        raise ValueError("the summaries are for different tokens")
    if not np.array_equal(reference.positions, actual.positions):
        raise ValueError("the summaries keep different positions")
    ref_logits = reference.logits.astype(np.float64)
    act_logits = actual.logits.astype(np.float64)
    ref_lp, act_lp = _log_softmax(ref_logits), _log_softmax(act_logits)
    kl = (np.exp(ref_lp) * (ref_lp - act_lp)).sum(-1)
    dloss = np.abs(reference.next_token_logprob.astype(np.float64) - actual.next_token_logprob)
    return {
        "max_rel_logit_err": float(
            np.abs(act_logits - ref_logits).max() / max(np.abs(ref_logits).max(), 1e-9)
        ),
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "top1_agreement": float((reference.argmax == actual.argmax).mean()),
        "mean_abs_dloss": float(dloss.mean()),
        "max_abs_dloss": float(dloss.max()),
        "ref_loss": float(-reference.next_token_logprob.mean()),
        "actual_loss": float(-actual.next_token_logprob.mean()),
    }
