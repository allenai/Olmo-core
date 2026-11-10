"""Beaker minRuntime from a run's expected runtime, and expected runtime from run history (stdlib).

Why (records/gpu-usage-workflow-audit-2026-11.md): Beaker preempts our jobs by *allocation balance*
as soon as they pass minRuntime, at any priority, and the research trainer writes no mid-run
checkpoint, so a preempted run restarts from step 0. 2026-10-20..11-10: 309 Beaker training attempts
(1,083 GPU-h) were lost that way, 304 of them at minRuntime 1h, while 99% of training runs finish
within 4 h. Rule (Prasann, 2026-11-10): minRuntime = min(4 h, 1.3 x expected runtime), 1 h when
nothing is known; runs expected past 4 h also get a mid-run resume point.

Expected runtime comes from, in order: the caller's explicit estimate, else the median
``runtime_minutes`` of finished runs in the registry with the same launcher and the longest
shared run-name prefix that has at least ``MIN_SAMPLES`` runs (``ops status --record`` writes
``runtime_minutes`` when it sees a run finish), else nothing (-> the 1 h fallback).
"""
from __future__ import annotations

import math
import re
import statistics

FALLBACK_MIN = 60
CAP_MIN = 240
FACTOR = 1.3
MIN_SAMPLES = 3

# run-name tokens that do not change runtime: hashes, timestamps, LR, seeds, relaunch/cluster tags
_NOISE = re.compile(
    r"^(?:[0-9a-f]{8}|\d{10,14}|lr[0-9.e+-]+|sd\d+|seed\d+|r\d|retry\d*|rs|lam|lambda|bk|js|jst)$"
)


def min_runtime_minutes(expected_minutes) -> int:
    """minutes = min(240, max(60, ceil(1.3 x expected))); 60 when ``expected_minutes`` is unknown."""
    try:
        exp = float(expected_minutes)
    except (TypeError, ValueError):
        return FALLBACK_MIN
    if not exp > 0:
        return FALLBACK_MIN
    return int(min(CAP_MIN, max(FALLBACK_MIN, math.ceil(FACTOR * exp))))


def beaker_duration(minutes: int) -> str:
    """Beaker / Go duration string: 60 -> '1h', 78 -> '1h18m', 240 -> '4h'."""
    h, m = divmod(int(minutes), 60)
    return (f"{h}h" if h else "") + (f"{m}m" if m or not h else "")


def needs_resume_point(expected_minutes) -> bool:
    """True when the run is expected to outlast the 4 h minRuntime cap (so a preemption can hit it)."""
    try:
        return float(expected_minutes) > CAP_MIN
    except (TypeError, ValueError):
        return False


def name_tokens(run_id: str) -> list:
    """Run name split on '-'/'_' with runtime-irrelevant tokens (hash, LR, seed, cluster tag) dropped."""
    toks = [t for t in re.split(r"[-_]", (run_id or "").lower()) if t]
    return [t for t in toks if not _NOISE.match(t)]


def launcher_of(command: str | None) -> str | None:
    """Basename of the first ``*.py`` / ``*.sbatch`` / ``*.sh`` in a registered command line."""
    m = re.search(r"([\w.-]+\.(?:py|sbatch|sh))\b", command or "")
    return m.group(1) if m else None


def history_minutes(run_id: str, runs: dict, launcher: str | None = None, kind: str = "train",
                    min_samples: int = MIN_SAMPLES):
    """(median runtime minutes, n, matched prefix) over finished runs like ``run_id``, else (None, 0, None).

    ``runs`` is ``registry.load_runs()``. Candidates: ``final_state == done``, a numeric
    ``runtime_minutes``, the same ``kind`` and (when both sides know it) the same launcher. The
    match is the longest token prefix of ``run_id`` shared by at least ``min_samples`` candidates.
    """
    want = name_tokens(run_id)
    pool = []
    for rid, r in runs.items():
        if rid == run_id or r.get("final_state") != "done" or (kind and r.get("kind") != kind):
            continue
        try:
            rt = float(r.get("runtime_minutes"))
        except (TypeError, ValueError):
            continue
        if not rt > 0:
            continue
        lch = launcher_of(r.get("command"))
        if launcher and lch and lch != launcher:
            continue
        pool.append((name_tokens(rid), rt))
    # Never match on the first token alone (e.g. just "f8k"). Leave-one-out on the 2026-10-20..11-10
    # Beaker training runs: 1.3 x this median covers the actual runtime for 70% of runs (1h flat: 67%;
    # p75: 76%, p90: 78% at a ~1.7 h mean minRuntime). An explicit --expected-minutes is better.
    for n in range(len(want), 1, -1):
        hits = [rt for toks, rt in pool if toks[:n] == want[:n]]
        if len(hits) >= min_samples:
            return statistics.median(hits), len(hits), "-".join(want[:n])
    return None, 0, None


def suggest(run_id: str | None = None, expected_minutes=None, runs: dict | None = None,
            launcher: str | None = None, kind: str = "train") -> dict:
    """Everything a launcher needs: ``min_runtime`` (Beaker string), ``expected_minutes`` used,
    ``source`` (explicit / history:<prefix> (n=) / fallback), and ``resume`` (bool)."""
    src = "explicit"
    exp = None
    try:
        exp = float(expected_minutes) if expected_minutes not in (None, "") else None
    except (TypeError, ValueError):
        exp = None
    if exp is None and run_id and runs:
        med, n, prefix = history_minutes(run_id, runs, launcher=launcher, kind=kind)
        if med is not None:
            exp, src = round(med, 1), f"history:{prefix} (median of {n})"
    if exp is None:
        src = "fallback (no expected runtime known)"
    mins = min_runtime_minutes(exp)
    return {"min_runtime": beaker_duration(mins), "min_runtime_minutes": mins, "expected_minutes": exp,
            "source": src, "resume": needs_resume_point(exp)}
