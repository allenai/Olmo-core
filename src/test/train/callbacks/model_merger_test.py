"""Unit tests for ModelMergeCallback."""

from types import SimpleNamespace

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.callbacks import ModelMergeCallback
from olmo_core.train.callbacks.model_merger import (
    compute_merge_steps_from_decay_schedule,
    compute_merge_window_starts,
)

# ============================================================================
# merge_step validation
# ============================================================================


@pytest.mark.parametrize("value", [0, -10])
def test_merge_last_n_steps_non_positive_raises_error(value):
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=100, merge_last_n_steps=value, enabled=True)


@pytest.mark.parametrize("value", [0, -5])
def test_merge_step_non_positive_raises_error(value):
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=value, merge_last_n_steps=10, enabled=True)


def test_merge_step_list_with_invalid_raises_error():
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=[100, -5, 200], merge_last_n_steps=10, enabled=True)


def test_empty_merge_step_raises_error():
    with pytest.raises(OLMoConfigurationError, match="merge_step or merge_interval must be set"):
        ModelMergeCallback(merge_step=[], merge_last_n_steps=10, enabled=True)


def test_disabled_skips_validation():
    # Should not raise even with no merge steps configured
    ModelMergeCallback(enabled=False)


@pytest.mark.parametrize(
    "merge_step, merge_last_n_steps",
    [
        ([100, 250], 100),  # non-overlapping
        ([100, 200], 100),  # exact boundary
        (100, 100),  # single step
        ([100, 150], 100),  # overlapping windows (50 step gap < 100 window)
        ([100, 110, 120], 50),  # multiple overlapping windows
    ],
)
def test_merge_windows_ok(merge_step, merge_last_n_steps):
    ModelMergeCallback(merge_step=merge_step, merge_last_n_steps=merge_last_n_steps, enabled=True)


# ============================================================================
# merge_interval validation
# ============================================================================


@pytest.mark.parametrize("value", [0, -5])
def test_merge_interval_non_positive_raises_error(value):
    with pytest.raises(OLMoConfigurationError, match="merge_interval must be positive"):
        ModelMergeCallback(merge_interval=value, merge_last_n_steps=10, enabled=True)


def test_merge_interval_and_merge_step_both_set_raises_error():
    with pytest.raises(OLMoConfigurationError, match="Cannot set both"):
        ModelMergeCallback(merge_step=100, merge_interval=500, merge_last_n_steps=10, enabled=True)


def test_merge_interval_valid():
    cb = ModelMergeCallback(merge_interval=500, merge_last_n_steps=100, enabled=True)
    assert cb.merge_interval == 500
    assert cb._merge_steps == []  # deferred to pre_train


# ============================================================================
# Window boundary tests (off-by-one checks)
# ============================================================================


def test_window_start():
    cb = ModelMergeCallback(merge_step=100, merge_last_n_steps=10, enabled=True)
    # Window for merge at step 100 with 10 steps: [91, 100]
    assert cb._window_start(100) == 91


def test_window_start_clamps_to_zero():
    cb = ModelMergeCallback(merge_step=5, merge_last_n_steps=100, enabled=True)
    # merge_step=5, window would be [5-100+1, 5] = [-94, 5], clamped to [0, 5]
    assert cb._window_start(5) == 0


def test_window_start_single_step_window():
    cb = ModelMergeCallback(merge_step=100, merge_last_n_steps=1, enabled=True)
    # Window of 1 step: [100, 100]
    assert cb._window_start(100) == 100


def _make_cb_at_step(merge_steps, merge_last_n_steps, current_step, completed=None):
    """Create a callback and simulate being at a given step."""
    cb = ModelMergeCallback(
        merge_step=merge_steps, merge_last_n_steps=merge_last_n_steps, enabled=True
    )
    # Patch the step property to return our test value
    cb.__class__ = type(
        "_TestMergeCallback", (ModelMergeCallback,), {"step": property(lambda self: current_step)}
    )
    if completed:
        cb._completed_merges = set(completed)
    return cb


def test_active_windows_before_window():
    cb = _make_cb_at_step([100], 10, current_step=90)
    # Step 90 is before window [91, 100]
    assert cb._active_windows() == []


def test_active_windows_at_window_start():
    cb = _make_cb_at_step([100], 10, current_step=91)
    # Step 91 is the first step of window [91, 100]
    assert cb._active_windows() == [100]


def test_active_windows_at_merge_step():
    cb = _make_cb_at_step([100], 10, current_step=100)
    # Step 100 is the last step of window [91, 100]
    assert cb._active_windows() == [100]


def test_active_windows_after_merge_step():
    cb = _make_cb_at_step([100], 10, current_step=101)
    # Step 101 is past the window
    assert cb._active_windows() == []


def test_active_windows_overlapping():
    cb = _make_cb_at_step([100, 105], 10, current_step=98)
    # Step 98: in window [91, 100] and in window [96, 105]
    assert cb._active_windows() == [100, 105]


def test_active_windows_skips_completed():
    cb = _make_cb_at_step([100, 105], 10, current_step=98, completed=[100])
    # Step 98: would be in both windows, but 100 is completed
    assert cb._active_windows() == [105]


# ============================================================================
# Resume mid-window skipping
# ============================================================================


def test_resume_mid_window_skips_merge():
    """If we resume inside a merge window, that merge should be skipped."""
    cb = _make_cb_at_step([100], 10, current_step=95)
    # Step 95 is inside window [91, 100], but not at the start
    # Simulate pre_train detecting mid-window resume
    current_step = 95
    for ms in cb._merge_steps:
        if ms not in cb._completed_merges and cb._window_start(ms) < current_step <= ms:
            cb._completed_merges.add(ms)
    assert 100 in cb._completed_merges
    assert cb._active_windows() == []


def test_resume_at_window_start_does_not_skip():
    """If we resume exactly at the window start, the merge should proceed."""
    cb = _make_cb_at_step([100], 10, current_step=91)
    # Step 91 is exactly the window start [91, 100]
    current_step = 91
    for ms in cb._merge_steps:
        if ms not in cb._completed_merges and cb._window_start(ms) < current_step <= ms:
            cb._completed_merges.add(ms)
    assert 100 not in cb._completed_merges
    assert cb._active_windows() == [100]


def test_resume_before_window_does_not_skip():
    """If we resume before the window, the merge should proceed."""
    cb = _make_cb_at_step([100], 10, current_step=85)
    current_step = 85
    for ms in cb._merge_steps:
        if ms not in cb._completed_merges and cb._window_start(ms) < current_step <= ms:
            cb._completed_merges.add(ms)
    assert 100 not in cb._completed_merges


def test_resume_mid_window_overlapping_skips_affected():
    """If two windows overlap and we resume mid-way, only the affected one is skipped."""
    cb = _make_cb_at_step([100, 120], 20, current_step=95)
    # Window for 100: [81, 100], window for 120: [101, 120]
    # Step 95 is inside window for 100 but before window for 120
    current_step = 95
    for ms in cb._merge_steps:
        if ms not in cb._completed_merges and cb._window_start(ms) < current_step <= ms:
            cb._completed_merges.add(ms)
    assert 100 in cb._completed_merges
    assert 120 not in cb._completed_merges


# ============================================================================
# Helper functions
# ============================================================================


def test_compute_merge_window_starts():
    # Non-overlapping: each gets its own checkpoint
    assert compute_merge_window_starts([500, 1000], 100) == [401, 901]
    # Clamp to 0
    assert compute_merge_window_starts([50], 100) == [0]
    # Overlapping: only the earliest start in the group
    # merge at 5000 (window 4501-5000) and 5200 (window 4701-5200)
    # 4701 <= 5000, so only 4501 is needed
    assert compute_merge_window_starts([5000, 5200], 500) == [4501]
    # Three overlapping windows
    assert compute_merge_window_starts([100, 150, 200], 100) == [1]
    # Mixed: first two overlap, third is separate
    assert compute_merge_window_starts([500, 550, 1000], 100) == [401, 901]


def test_compute_merge_steps_from_decay_schedule_with_fixed_decay():
    # 2 periods of 10000 tokens, batch size 100, decay 1000 tokens
    steps = compute_merge_steps_from_decay_schedule(
        period_lengths=[10000, 10000],
        tokens_per_step=100,
        decay=1000,
    )
    # Period 1: 10000 tokens, pre-decay at 10000-1000=9000, step=90
    # Period 2: 20000 tokens, pre-decay at 20000-1000=19000, step=190
    assert steps == [90, 190]


def test_compute_merge_steps_from_decay_schedule_with_decay_fraction():
    steps = compute_merge_steps_from_decay_schedule(
        period_lengths=[10000],
        tokens_per_step=100,
        decay_fraction=0.1,
    )
    # 10000 tokens, decay=10000*0.1=1000, pre-decay at 9000, step=90
    assert steps == [90]


def test_compute_merge_steps_from_decay_schedule_requires_decay():
    with pytest.raises(ValueError, match="Either decay or decay_fraction"):
        compute_merge_steps_from_decay_schedule(period_lengths=[10000], tokens_per_step=100)


# ============================================================================
# save_checkpoint
# ============================================================================


def test_save_checkpoint_defaults_to_true():
    cb = ModelMergeCallback(merge_step=100, merge_last_n_steps=10, enabled=True)
    assert cb.save_checkpoint


def test_eval_only_merge_skips_the_save(monkeypatch, tmp_path):
    """With ``save_checkpoint=False`` the averaged weights are swapped in, evaluated, and
    restored, and nothing is written to disk."""
    import torch

    from olmo_core.train.callbacks import model_merger

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(4.0)

    trainer = SimpleNamespace(
        save_folder=str(tmp_path), callbacks={}, train_module=SimpleNamespace(model=model)
    )
    cb = ModelMergeCallback(
        merge_step=10, merge_last_n_steps=2, enabled=True, save_checkpoint=False
    )
    cb._trainer = trainer  # type: ignore[assignment]
    cb._accumulators[10] = {"weight": torch.full((1, 2), 2.0)}
    cb._accumulator_counts[10] = 2

    seen = []
    monkeypatch.setattr(cb, "_evaluate_merged", lambda: seen.append(model.weight.detach().clone()))
    monkeypatch.setattr(model_merger, "save_state_dict", lambda *a, **k: pytest.fail("saved"))
    monkeypatch.setattr(model_merger, "barrier", lambda: None)

    cb._save_merged_checkpoint(10)

    assert seen and torch.equal(seen[0], torch.full((1, 2), 1.0))
    assert torch.equal(model.weight.detach(), torch.full((1, 2), 4.0))
    assert not list(tmp_path.iterdir())
    assert 10 in cb._completed_merges
