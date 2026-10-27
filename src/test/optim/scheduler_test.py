import math
import warnings
from typing import Any, Dict, List, Tuple, Type

import pytest

import olmo_core.optim.scheduler as scheduler_module
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import (
    WSDS,
    ConstantWithWarmup,
    CosWithWarmup,
    ExponentialScheduler,
    InvSqrtWithWarmup,
    LinearWithWarmup,
    PowerLR,
    SequentialScheduler,
)
from olmo_core.optim.scheduler import (
    ComposableScheduler,
    ComposableSchedulerStage,
    ComposableSchedulerStageType,
    OverrideDecay,
    Scheduler,
)


def test_constant_with_warmup():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    scheduler = ConstantWithWarmup(warmup=warmup_steps, warmup_min_lr=warmup_min_lr)
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 5_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 10.0


def test_linear_with_warmup():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    alpha_f = 0.2
    scheduler = LinearWithWarmup(warmup=warmup_steps, alpha_f=alpha_f, warmup_min_lr=warmup_min_lr)
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 5_000, max_steps) == 10.0 - 8.0 * (2_000 / 7_000)
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 2.0


def test_inv_sqrt_with_warmup():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    alpha_f = 0.2
    scheduler = InvSqrtWithWarmup(warmup=warmup_steps, alpha_f=alpha_f, warmup_min_lr=warmup_min_lr)
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 5_000, max_steps) == 2.0 + 8.0 * math.sqrt(3_000 / 5_000)
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 2.0 + 8.0 * math.sqrt(3_000 / 10_000)


def test_cos_with_warmup_scheduler():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    alpha_f = 0.2
    scheduler = CosWithWarmup(warmup=warmup_steps, alpha_f=alpha_f, warmup_min_lr=warmup_min_lr)
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert (
        scheduler.get_lr(initial_lr, 5_000, max_steps)
        == 2.0 + 8.0 * (1 + math.cos(math.pi * 2_000 / 7_000)) / 2
    )
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 2.0


def test_sequential_scheduler():
    initial_lr = 10.0
    max_steps = 20_000

    first_scheduler = InvSqrtWithWarmup(alpha_f=0.5, warmup=1_000)
    second_scheduler = CosWithWarmup(warmup=0)
    third_scheduler = LinearWithWarmup(t_max=3_000, warmup=0)
    schedulers_max_steps = [2_500, 4_000]
    scheduler = SequentialScheduler(
        schedulers=[first_scheduler, second_scheduler, third_scheduler],
        schedulers_max=schedulers_max_steps,
    )

    first_scheduler_final_lr = first_scheduler.get_lr(initial_lr, 2_500, 2_500)
    second_scheduler_final_lr = second_scheduler.get_lr(first_scheduler_final_lr, 4_000, 4_000)

    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == first_scheduler.get_lr(
        initial_lr, 1_000, 2_500
    )
    assert scheduler.get_lr(initial_lr, 4_000, max_steps) == second_scheduler.get_lr(
        first_scheduler_final_lr, 1_500, 4_000
    )
    assert scheduler.get_lr(initial_lr, 7_500, max_steps) == third_scheduler.get_lr(
        second_scheduler_final_lr, 1_000, max_steps - 6_500
    )


def test_power_lr():
    initial_lr = 10.0
    warmup_steps = 1_000
    decay_steps = 2_000
    t_max = 10_000
    b = -0.5
    scheduler = PowerLR(warmup=warmup_steps, decay=decay_steps, decay_fraction=None, b=b)

    # Phase 1: linear warmup from 0 to initial_lr over warmup_steps.
    assert scheduler.get_lr(initial_lr, 0, t_max) == pytest.approx(0.0)
    assert scheduler.get_lr(initial_lr, 500, t_max) == pytest.approx(5.0)
    assert scheduler.get_lr(initial_lr, warmup_steps, t_max) == pytest.approx(initial_lr)

    # Phase 2: power-law decay lr = initial_lr * (current / warmup) ** b.
    for current in (2_000, 4_000, 6_000):
        expected = initial_lr * (current / warmup_steps) ** b
        assert scheduler.get_lr(initial_lr, current, t_max) == pytest.approx(expected)

    # Phase 3: linear decay tail from power-phase value at t_max - decay -> decay_min_lr=0.
    decay_start = t_max - decay_steps  # 8_000
    lr_at_tail_start = initial_lr * (decay_start / warmup_steps) ** b
    assert scheduler.get_lr(initial_lr, decay_start, t_max) == pytest.approx(lr_at_tail_start)
    # Midpoint of decay tail: halfway down linearly.
    assert scheduler.get_lr(initial_lr, decay_start + decay_steps // 2, t_max) == pytest.approx(
        lr_at_tail_start * 0.5
    )
    assert scheduler.get_lr(initial_lr, t_max, t_max) == pytest.approx(0.0)


def test_power_lr_errors():
    # b must be negative.
    with pytest.raises(OLMoConfigurationError, match="'b' must be negative"):
        PowerLR(warmup=100, decay=100, decay_fraction=None, b=0.5)

    # Need exactly one of warmup / warmup_fraction.
    with pytest.raises(OLMoConfigurationError, match="warmup_fraction.*or.*warmup"):
        PowerLR(decay=100, decay_fraction=None)

    # warmup_fraction out of range.
    with pytest.raises(OLMoConfigurationError, match="warmup_fraction"):
        PowerLR(warmup_fraction=1.5, decay=100, decay_fraction=None)

    # Specifying both 'decay' and 'decay_fraction' is rejected (PowerLR enforces exclusivity).
    with pytest.raises(OLMoConfigurationError, match="decay_fraction.*or.*decay"):
        PowerLR(warmup=100, decay=100, decay_fraction=0.1)


def test_sequential_scheduler_override_decay_linear():
    initial_lr = 10.0
    max_steps = 20_000

    first_scheduler = ConstantWithWarmup(warmup=1_000)
    second_scheduler = CosWithWarmup(warmup=0, alpha_f=0.1)
    base = SequentialScheduler(
        schedulers=[first_scheduler, second_scheduler],
        schedulers_max=[5_000],
    )
    scheduler = SequentialScheduler(
        schedulers=[first_scheduler, second_scheduler],
        schedulers_max=[5_000],
        override_decay=OverrideDecay(
            start=10_000,
            duration=2_000,
            shape=ComposableSchedulerStageType.linear,
            end_lr=1.0,
        ),
    )

    # Before override: same as base sequence.
    assert scheduler.get_lr(initial_lr, 500, max_steps) == base.get_lr(initial_lr, 500, max_steps)
    assert scheduler.get_lr(initial_lr, 4_999, max_steps) == base.get_lr(
        initial_lr, 4_999, max_steps
    )
    assert scheduler.get_lr(initial_lr, 9_999, max_steps) == base.get_lr(
        initial_lr, 9_999, max_steps
    )

    # At override start: same LR as base would produce.
    main_lr_at_start = base.get_lr(initial_lr, 10_000, max_steps)
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == pytest.approx(main_lr_at_start)

    # Midpoint of override: halfway between start LR and end LR.
    expected_mid = main_lr_at_start + (1.0 - main_lr_at_start) * 0.5
    assert scheduler.get_lr(initial_lr, 11_000, max_steps) == pytest.approx(expected_mid)

    # End of override: at end_lr.
    assert scheduler.get_lr(initial_lr, 12_000, max_steps) == pytest.approx(1.0)

    # Past end of override: held at end_lr.
    assert scheduler.get_lr(initial_lr, 15_000, max_steps) == pytest.approx(1.0)


def test_sequential_scheduler_override_decay_cosine():
    initial_lr = 10.0
    max_steps = 20_000

    first_scheduler = ConstantWithWarmup(warmup=1_000)
    second_scheduler = CosWithWarmup(warmup=0, alpha_f=0.1)
    base = SequentialScheduler(
        schedulers=[first_scheduler, second_scheduler],
        schedulers_max=[5_000],
    )
    scheduler = SequentialScheduler(
        schedulers=[first_scheduler, second_scheduler],
        schedulers_max=[5_000],
        override_decay=OverrideDecay(
            start=10_000,
            duration=2_000,
            shape=ComposableSchedulerStageType.cosine,
            end_lr_fraction=0.05,
        ),
    )

    main_lr_at_start = base.get_lr(initial_lr, 10_000, max_steps)
    # Midpoint of cosine override: half-cosine -> start + (end - start) * 0.5.
    end_lr = initial_lr * 0.05
    expected_mid = end_lr + (main_lr_at_start - end_lr) * (1 + math.cos(math.pi * 0.5)) / 2
    assert scheduler.get_lr(initial_lr, 11_000, max_steps) == pytest.approx(expected_mid)
    assert scheduler.get_lr(initial_lr, 12_000, max_steps) == pytest.approx(end_lr)
    assert scheduler.get_lr(initial_lr, 15_000, max_steps) == pytest.approx(end_lr)


def test_sequential_scheduler_override_decay_inside_first_segment():
    """Override starts inside the first sub-scheduler, not the last."""
    initial_lr = 10.0
    max_steps = 20_000

    first_scheduler = ConstantWithWarmup(warmup=1_000)  # holds at 10.0 after step 1000
    second_scheduler = CosWithWarmup(warmup=0, alpha_f=0.1)
    scheduler = SequentialScheduler(
        schedulers=[first_scheduler, second_scheduler],
        schedulers_max=[5_000],
        override_decay=OverrideDecay(
            start=2_000,
            duration=1_000,
            shape=ComposableSchedulerStageType.linear,
            end_lr=2.0,
        ),
    )

    # At override start: LR should be 10.0 (constant phase of first scheduler).
    assert scheduler.get_lr(initial_lr, 2_000, max_steps) == pytest.approx(10.0)
    # Midpoint: linear from 10.0 -> 2.0.
    assert scheduler.get_lr(initial_lr, 2_500, max_steps) == pytest.approx(6.0)
    # End of override and beyond.
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == pytest.approx(2.0)
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == pytest.approx(2.0)


_PARITY_CASES: List[Tuple[ComposableSchedulerStageType, Type[Scheduler], float]] = [
    (ComposableSchedulerStageType.linear, LinearWithWarmup, 0.0),
    (ComposableSchedulerStageType.linear, LinearWithWarmup, 0.1),
    (ComposableSchedulerStageType.cosine, CosWithWarmup, 0.0),
    (ComposableSchedulerStageType.cosine, CosWithWarmup, 0.1),
]


@pytest.mark.parametrize("stage_shape, sub_scheduler_cls, alpha_f", _PARITY_CASES)
def test_composable_and_sequential_match_when_t_max_aligns(stage_shape, sub_scheduler_cls, alpha_f):
    """
    A warmup + decay schedule should produce identical LR curves whether
    expressed as a ComposableScheduler or as a SequentialScheduler, provided
    't_max' passed at evaluation time equals the composable's total stage
    duration (i.e. ComposableScheduler's absolute-time view and
    SequentialScheduler's last-segment-stretches-to-t_max view happen to agree).
    """
    initial_lr = 10.0
    warmup_steps = 1_000
    decay_steps = 9_000
    t_max = warmup_steps + decay_steps

    composable = ComposableScheduler(
        stages=[
            ComposableSchedulerStage(
                duration=warmup_steps,
                shape=ComposableSchedulerStageType.linear,
                start_lr_fraction=0.0,
                end_lr_fraction=1.0,
            ),
            ComposableSchedulerStage(
                duration=decay_steps,
                shape=stage_shape,
                end_lr_fraction=alpha_f,
            ),
        ],
    )

    sequential = SequentialScheduler(
        schedulers=[
            ConstantWithWarmup(warmup=warmup_steps),
            sub_scheduler_cls(warmup=0, alpha_f=alpha_f),
        ],
        schedulers_max=[warmup_steps],
    )

    # Spans warmup interior, exact boundary, decay interior, exact end-of-schedule.
    sample_points = [0, 1, 100, 500, 999, 1_000, 1_001, 2_500, 5_500, 7_500, 9_999, t_max]
    for current in sample_points:
        composable_lr = composable.get_lr(initial_lr, current, t_max)
        sequential_lr = sequential.get_lr(initial_lr, current, t_max)
        assert composable_lr == pytest.approx(
            sequential_lr
        ), f"mismatch at step {current}: composable={composable_lr}, sequential={sequential_lr}"


def test_sequential_scheduler_override_decay_t_max_warning():
    scheduler = SequentialScheduler(
        schedulers=[ConstantWithWarmup(warmup=100), CosWithWarmup(warmup=0)],
        schedulers_max=[1_000],
        override_decay=OverrideDecay(start=500, duration=200, end_lr=0.0),
    )

    # Warning fires once, when override becomes active.
    with pytest.warns(UserWarning, match="ignores 't_max'"):
        scheduler.get_lr(10.0, 600, 10_000)

    # No further warnings on subsequent calls.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        scheduler.get_lr(10.0, 700, 10_000)


def test_composable_scheduler_t_max_warning_rank0_only(monkeypatch):
    monkeypatch.setattr(scheduler_module, "get_rank", lambda: 1)
    scheduler = ComposableScheduler(
        stages=[
            ComposableSchedulerStage(
                duration=100,
                shape=ComposableSchedulerStageType.linear,
                end_lr=0.0,
            )
        ],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        assert scheduler.get_lr(1.0, 50, 1_000) == pytest.approx(0.5)

    assert not [warning for warning in caught if "ignores 't_max'" in str(warning.message)]
    assert scheduler._warned_t_max_ignored


def test_composable_scheduler_override_decay_linear():
    initial_lr = 10.0
    scheduler = ComposableScheduler(
        stages=[
            ComposableSchedulerStage(
                duration=100,
                shape=ComposableSchedulerStageType.linear,
                start_lr_fraction=0.0,
                end_lr_fraction=1.0,
            ),
            ComposableSchedulerStage(
                duration=900,
                shape=ComposableSchedulerStageType.linear,
                end_lr_fraction=0.1,
            ),
        ],
        override_decay=OverrideDecay(
            start=200,
            duration=100,
            shape=ComposableSchedulerStageType.linear,
            end_lr=3.0,
        ),
    )

    assert scheduler.get_lr(initial_lr, 150, 1000) == pytest.approx(9.5)
    assert scheduler.get_lr(initial_lr, 200, 1000) == pytest.approx(9.0)
    assert scheduler.get_lr(initial_lr, 250, 1000) == pytest.approx(6.0)
    assert scheduler.get_lr(initial_lr, 300, 1000) == pytest.approx(3.0)
    assert scheduler.get_lr(initial_lr, 800, 1000) == pytest.approx(3.0)


def test_composable_scheduler_override_decay_cosine():
    initial_lr = 10.0
    scheduler = ComposableScheduler(
        stages=[
            ComposableSchedulerStage(
                duration=1000,
                shape=ComposableSchedulerStageType.linear,
                end_lr_fraction=1.0,
            ),
        ],
        override_decay=OverrideDecay(
            start=100,
            duration=100,
            shape=ComposableSchedulerStageType.cosine,
            end_lr_fraction=0.2,
        ),
    )

    assert scheduler.get_lr(initial_lr, 99, 1000) == pytest.approx(10.0)
    assert scheduler.get_lr(initial_lr, 100, 1000) == pytest.approx(10.0)
    assert scheduler.get_lr(initial_lr, 150, 1000) == pytest.approx(6.0)
    assert scheduler.get_lr(initial_lr, 200, 1000) == pytest.approx(2.0)
    assert scheduler.get_lr(initial_lr, 500, 1000) == pytest.approx(2.0)


def test_composable_scheduler_override_decay_validation():
    with pytest.raises(
        OLMoConfigurationError, match="exactly one of 'end_lr' or 'end_lr_fraction'"
    ):
        OverrideDecay(start=100, duration=100)


class TestWSDSScheduler:
    """Test suite for WSDS (Warmup-Stable-Decay-Simplified) scheduler."""

    def test_basic_single_period_with_warmup(self):
        """Test a single period with warmup, stable, and decay phases."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=100,
            decay=200,
            warmup_min_lr=0.0,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0

        assert scheduler.get_lr(initial_lr, 0, 1000) == 0.0
        lr_mid_warmup = scheduler.get_lr(initial_lr, 50, 1000)
        assert 0.0 < lr_mid_warmup < initial_lr
        assert scheduler.get_lr(initial_lr, 100, 1000) == pytest.approx(initial_lr)

        assert scheduler.get_lr(initial_lr, 200, 1000) == initial_lr
        assert scheduler.get_lr(initial_lr, 500, 1000) == initial_lr
        assert scheduler.get_lr(initial_lr, 799, 1000) == initial_lr

        lr_mid_decay = scheduler.get_lr(initial_lr, 900, 1000)
        assert 0.0 < lr_mid_decay < initial_lr
        assert scheduler.get_lr(initial_lr, 1000, 1000) == 0.0

    def test_multiple_periods_with_reset(self):
        """Test that LR resets to peak at the start of each new period."""
        scheduler = WSDS(
            period_lengths=[1000, 1000],
            warmup=100,
            decay=200,
            warmup_min_lr=0.0,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0

        # end of first period
        assert scheduler.get_lr(initial_lr, 1000, 2000) == 0.0

        # start of second period - should reset to peak
        assert scheduler.get_lr(initial_lr, 1001, 2000) == initial_lr

        # middle of second period stable phase
        assert scheduler.get_lr(initial_lr, 1400, 2000) == initial_lr

        # end of second period
        assert scheduler.get_lr(initial_lr, 2000, 2000) == 0.0

    def test_warmup_fraction(self):
        """Test warmup specified as a fraction."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup_fraction=0.1,
            decay_fraction=0.2,
            warmup_min_lr=0.0,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0

        assert scheduler.get_lr(initial_lr, 0, 1000) == 0.0
        assert scheduler.get_lr(initial_lr, 100, 1000) == pytest.approx(initial_lr)
        assert scheduler.get_lr(initial_lr, 500, 1000) == initial_lr
        assert scheduler.get_lr(initial_lr, 1000, 1000) == 0.0

    def test_decay_fraction(self):
        """Test decay specified as a fraction."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=50,
            decay_fraction=0.3,
            warmup_min_lr=0.0,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0

        # Stable phase should be 1000 - 50 - 300 = 650 steps
        # Phases: warmup [0, 50), stable [50, 700), decay [700, 1000]
        assert scheduler.get_lr(initial_lr, 50, 1000) == initial_lr  # Start of stable
        assert scheduler.get_lr(initial_lr, 699, 1000) == initial_lr  # End stable
        assert scheduler.get_lr(initial_lr, 700, 1000) == initial_lr  # Still stable (boundary)
        lr_decay = scheduler.get_lr(initial_lr, 701, 1000)  # First step of decay
        assert 0.0 < lr_decay < initial_lr  # In decay
        assert scheduler.get_lr(initial_lr, 1000, 1000) == 0.0  # End of period

    def test_no_warmup_in_subsequent_periods(self):
        """Test that only period 0 has warmup."""
        scheduler = WSDS(
            period_lengths=[1000, 1000],
            warmup=100,
            decay=200,
        )
        initial_lr = 1.0

        # period 1 starts immediately at peak LR (no warmup)!
        assert scheduler.get_lr(initial_lr, 1001, 2000) == initial_lr

    def test_beyond_all_periods(self):
        """Test behavior when current step exceeds all periods."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=100,
            decay=200,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0

        assert scheduler.get_lr(initial_lr, 1001, 1000) == 0.0
        assert scheduler.get_lr(initial_lr, 5000, 1000) == 0.0

    def test_non_zero_decay_min_lr(self):
        """Test that decay_min_lr can be non-zero."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=100,
            decay=200,
            decay_min_lr=0.1,
        )
        initial_lr = 1.0

        assert scheduler.get_lr(initial_lr, 1000, 1000) == pytest.approx(0.1)
        assert scheduler.get_lr(initial_lr, 1500, 1000) == 0.1

    def test_non_zero_warmup_min_lr(self):
        """Test that warmup can start from non-zero LR."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=100,
            decay=200,
            warmup_min_lr=0.1,
        )
        initial_lr = 1.0

        assert scheduler.get_lr(initial_lr, 0, 1000) == 0.1

    def test_error_empty_period_lengths(self):
        """Test that empty period_lengths raises error."""
        with pytest.raises(OLMoConfigurationError, match="must be provided and non-empty"):
            WSDS(period_lengths=[])

    def test_error_negative_period_length(self):
        """Test that negative period lengths raise error."""
        with pytest.raises(OLMoConfigurationError, match="must be > 0"):
            WSDS(period_lengths=[1000, -500])

    def test_error_both_warmup_specified(self):
        """Test that specifying both warmup and warmup_fraction raises error."""
        with pytest.raises(OLMoConfigurationError, match="Exactly one"):
            WSDS(
                period_lengths=[1000],
                warmup=100,
                warmup_fraction=0.1,
                decay=200,
            )

    def test_error_neither_warmup_specified(self):
        """Test that specifying neither warmup nor warmup_fraction raises error."""
        with pytest.raises(OLMoConfigurationError, match="Exactly one"):
            WSDS(
                period_lengths=[1000],
                decay=200,
            )

    def test_error_both_decay_specified(self):
        """Test that specifying both decay and decay_fraction raises error."""
        with pytest.raises(OLMoConfigurationError, match="Exactly one"):
            WSDS(
                period_lengths=[1000],
                warmup=100,
                decay=200,
                decay_fraction=0.2,
            )

    def test_error_neither_decay_specified(self):
        """Test that specifying neither decay nor decay_fraction raises error."""
        with pytest.raises(OLMoConfigurationError, match="Exactly one"):
            WSDS(
                period_lengths=[1000],
                warmup=100,
            )

    def test_error_warmup_fraction_out_of_range(self):
        """Test that warmup_fraction must be in [0, 1]."""
        with pytest.raises(OLMoConfigurationError, match="must be in"):
            WSDS(
                period_lengths=[1000],
                warmup_fraction=1.5,
                decay=200,
            )

    def test_error_decay_fraction_out_of_range(self):
        """Test that decay_fraction must be in [0, 1]."""
        with pytest.raises(OLMoConfigurationError, match="must be in"):
            WSDS(
                period_lengths=[1000],
                warmup=100,
                decay_fraction=-0.1,
            )

    def test_error_warmup_plus_decay_exceeds_first_period(self):
        """Test that warmup + decay exceeding first period raises error."""
        with pytest.raises(OLMoConfigurationError, match="exceeds period length"):
            WSDS(
                period_lengths=[1000],
                warmup=600,
                decay=500,  # 600 + 500 = 1100 > 1000
            )

    def test_error_decay_exceeds_subsequent_period(self):
        """Test that decay exceeding any period length raises error."""
        with pytest.raises(OLMoConfigurationError, match="Period 1.*exceeds period length"):
            WSDS(
                period_lengths=[1000, 500],
                warmup=100,
                decay=600,
            )

    @pytest.mark.parametrize("num_periods", [1, 2, 5])
    def test_variable_period_counts(self, num_periods):
        """Test scheduler works with different numbers of periods."""
        scheduler = WSDS(
            period_lengths=[1000] * num_periods,
            warmup=100,
            decay=200,
        )
        initial_lr = 1.0

        for i in range(num_periods):
            period_end = (i + 1) * 1000
            assert scheduler.get_lr(initial_lr, period_end, period_end) == 0.0

    @pytest.mark.parametrize("period_length", [100, 1000, 10000])
    def test_variable_period_lengths(self, period_length):
        """Test scheduler works with different period lengths."""
        warmup = int(0.1 * period_length)
        decay = int(0.2 * period_length)

        scheduler = WSDS(
            period_lengths=[period_length],
            warmup=warmup,
            decay=decay,
        )
        initial_lr = 1.0

        assert scheduler.get_lr(initial_lr, 0, period_length) == 0.0
        assert scheduler.get_lr(initial_lr, period_length // 2, period_length) == initial_lr
        assert scheduler.get_lr(initial_lr, period_length, period_length) == 0.0


def test_exponential_scheduler():
    """Test exponential LR scheduler for LR range testing."""
    initial_lr = 10.0  # max LR
    lr_min = 1e-9
    max_steps = 10_000
    scheduler = ExponentialScheduler(lr_min=lr_min)

    # At step 0, should be at lr_min
    assert scheduler.get_lr(initial_lr, 0, max_steps) == pytest.approx(lr_min)

    # At step t_max, should be at initial_lr (max LR)
    assert scheduler.get_lr(initial_lr, max_steps, max_steps) == pytest.approx(initial_lr)

    # At step t_max/2, should be at geometric mean of lr_min and lr_max
    # lr(t_max/2) = lr_min * (lr_max/lr_min)^0.5 = sqrt(lr_min * lr_max)
    expected_mid = math.sqrt(lr_min * initial_lr)
    assert scheduler.get_lr(initial_lr, max_steps // 2, max_steps) == pytest.approx(expected_mid)

    # Verify exponential growth at various points
    # lr(t) = lr_min * (lr_max / lr_min)^(t / t_max)
    for step in [1_000, 5_000, 8_000]:
        ratio = step / max_steps
        expected_lr = lr_min * (initial_lr / lr_min) ** ratio
        assert scheduler.get_lr(initial_lr, step, max_steps) == pytest.approx(expected_lr)

    # Verify LR is monotonically increasing
    lr_at_1000 = scheduler.get_lr(initial_lr, 1_000, max_steps)
    lr_at_5000 = scheduler.get_lr(initial_lr, 5_000, max_steps)
    lr_at_9000 = scheduler.get_lr(initial_lr, 9_000, max_steps)
    assert lr_min < lr_at_1000 < lr_at_5000 < lr_at_9000 < initial_lr


def test_exponential_scheduler_error():
    """Test that ExponentialScheduler raises error for invalid lr_min."""
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ExponentialScheduler(lr_min=0.0)

    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ExponentialScheduler(lr_min=-1e-9)


def test_exponential_scheduler_integration(tiny_model):
    """Integration test: ExponentialScheduler with actual optimizer and training steps."""
    from unittest.mock import Mock

    import torch.optim as optim

    from olmo_core.train import Trainer

    # Setup
    lr_min = 1e-5
    lr_max = 1.0
    max_steps = 100

    # Create optimizer with max LR
    optimizer = optim.Adam(tiny_model.parameters(), lr=lr_max)

    # Create scheduler
    scheduler = ExponentialScheduler(lr_min=lr_min)

    # Create a mock trainer with necessary attributes
    mock_trainer = Mock(spec=Trainer)
    mock_trainer.max_steps = max_steps
    mock_trainer.max_tokens = None

    # Record LRs at each step
    recorded_lrs = []

    for step in range(max_steps + 1):
        mock_trainer.global_step = step

        # Apply scheduler to optimizer param group
        for group in optimizer.param_groups:
            new_lr = scheduler.set_lr(group, mock_trainer)
            recorded_lrs.append(new_lr)

    # Verify LR progression
    assert recorded_lrs[0] == pytest.approx(lr_min, rel=1e-6), "LR at step 0 should be lr_min"
    assert recorded_lrs[max_steps] == pytest.approx(
        lr_max, rel=1e-6
    ), "LR at max_steps should be lr_max"

    # Verify exponential growth (each LR should be larger than the previous)
    for i in range(1, len(recorded_lrs)):
        assert (
            recorded_lrs[i] > recorded_lrs[i - 1]
        ), f"LR should increase monotonically at step {i}"

    # Verify the exponential formula at midpoint
    mid_step = max_steps // 2
    expected_mid_lr = lr_min * (lr_max / lr_min) ** (mid_step / max_steps)
    assert recorded_lrs[mid_step] == pytest.approx(expected_mid_lr, rel=1e-5)

    # Verify LR is correctly set in optimizer param group
    assert optimizer.param_groups[0]["lr"] == pytest.approx(lr_max, rel=1e-6)


_LEGACY_SCHEDULER_CLASSES = [
    ConstantWithWarmup,
    scheduler_module.WSD,
    PowerLR,
    LinearWithWarmup,
    InvSqrtWithWarmup,
    CosWithWarmup,
    scheduler_module.HalfCosWithWarmup,
    scheduler_module.CosWithWarmupAndLinearDecay,
    SequentialScheduler,
]


def _legacy_scheduler_example(scheduler_cls, units):
    scale = 1_000 if units == scheduler_module.SchedulerUnits.tokens else 1
    if scheduler_cls is SequentialScheduler:
        canonical = SequentialScheduler(
            units=units,
            schedulers=[
                ConstantWithWarmup(units=units, warmup=10 * scale),
                CosWithWarmup(units=units, warmup=0),
            ],
            schedulers_max=[40 * scale],
        )
    else:
        kwargs = {"units": units, "warmup": 10 * scale}
        if scheduler_cls in (
            scheduler_module.WSD,
            PowerLR,
            scheduler_module.CosWithWarmupAndLinearDecay,
        ):
            kwargs.update(decay=20 * scale, decay_fraction=None)
        canonical = scheduler_cls(**kwargs)

    return canonical


@pytest.mark.parametrize("scheduler_cls", _LEGACY_SCHEDULER_CLASSES)
@pytest.mark.parametrize("units", list(scheduler_module.SchedulerUnits))
def test_legacy_scheduler_config_schedule_parity(scheduler_cls, units):
    """Legacy JSON aliases preserve LR curves, units, and nested scheduler boundaries."""
    import json
    from unittest.mock import Mock

    scale = 1_000 if units == scheduler_module.SchedulerUnits.tokens else 1
    canonical = _legacy_scheduler_example(scheduler_cls, units)

    aliases = {
        "warmup": "warmup_steps",
        "decay": "decay_steps",
        "schedulers_max": "schedulers_max_steps",
    }

    def legacy_fields(value):
        if isinstance(value, dict):
            return {
                aliases.get(key, key): legacy_fields(item)
                for key, item in value.items()
                if key not in aliases.values()
            }
        if isinstance(value, list):
            return [legacy_fields(item) for item in value]
        return value

    # Keep explicit nulls: decay_fraction=None disables the non-null default when decay is set.
    serialization_options: Dict[str, Any] = dict(
        exclude_none=False, exclude_private_fields=True, include_class_name=True, json_safe=True
    )
    legacy_json = json.dumps(legacy_fields(canonical.as_dict(**serialization_options)))
    with pytest.warns(DeprecationWarning):
        restored = scheduler_cls.from_dict(json.loads(legacy_json))
    assert restored.as_config_dict() == canonical.as_config_dict()
    assert scheduler_cls.from_dict(restored.as_dict(**serialization_options)) == canonical

    legacy_group = {"lr": 0.01}
    canonical_group = {"lr": 0.01}
    trainer = Mock(max_steps=100, max_tokens=100_000)
    for step in range(101):
        trainer.global_step = step
        trainer.global_train_tokens_seen = step * 1_000
        expected = canonical.get_lr(0.01, step * scale, 100 * scale)
        assert canonical.set_lr(canonical_group, trainer) == expected
        assert restored.set_lr(legacy_group, trainer) == expected
        assert legacy_group == canonical_group


@pytest.mark.parametrize("scheduler_cls", _LEGACY_SCHEDULER_CLASSES)
@pytest.mark.parametrize("units", list(scheduler_module.SchedulerUnits))
def test_legacy_scheduler_cli_overrides(scheduler_cls, units):
    """Legacy CLI fields override canonical recipe values and are cleared after loading."""
    from olmo_core.optim import AdamWConfig
    from olmo_core.train.train_module import TransformerTrainModuleConfig

    scheduler = _legacy_scheduler_example(scheduler_cls, units)
    config = TransformerTrainModuleConfig(
        rank_microbatch_size=1,
        max_sequence_length=1,
        optim=AdamWConfig(),
        scheduler=scheduler,
    )
    scale = 1_000 if units == scheduler_module.SchedulerUnits.tokens else 1
    original = config.as_dict()
    aliases = {"warmup_steps": "warmup", "decay_steps": "decay"}
    changes: List[Tuple[str, str, Any]] = [
        (alias, name, 5 * scale) for alias, name in aliases.items() if hasattr(scheduler, name)
    ]
    if scheduler_cls is SequentialScheduler:
        changes = [
            ("schedulers_max_steps", "schedulers_max", [50 * scale]),
            ("schedulers.0.warmup_steps", "schedulers.0.warmup", 5 * scale),
            ("schedulers.0.warmup_steps", "schedulers.0.warmup", 0),
        ]
    else:
        changes.append(("warmup_steps", "warmup", 0))

    for alias, name, value in changes:
        # Match launch CLI syntax and nested config reconstruction, including prefix stripping.
        with pytest.warns(DeprecationWarning, match=alias.split(".")[-1]):
            restored = config.merge(
                [f"--train_module.scheduler.{alias}={value}"], prefix="train_module"
            )
        expected = config.merge([f"scheduler.{name}={value}"])
        assert restored == expected
        assert config.as_dict() == original
        assert restored.scheduler is not None
        assert expected.scheduler is not None
        assert restored.scheduler.units == units
        # Zero must still migrate, but some schedulers do not define an LR at zero warmup.
        if value != 0:
            for step in range(101):
                assert restored.scheduler.get_lr(0.01, step * scale, 100 * scale) == (
                    expected.scheduler.get_lr(0.01, step * scale, 100 * scale)
                )
        # No stale legacy value can shadow subsequent canonical overrides.
        next_value = [60 * scale] if isinstance(value, list) else 3 * scale
        assert restored.merge([f"scheduler.{name}={next_value}"]) == config.merge(
            [f"scheduler.{name}={next_value}"]
        )
