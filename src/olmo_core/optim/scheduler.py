import logging
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from math import cos, pi, sqrt
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch

from ..config import Config, Registrable, StrEnum
from ..distributed.utils import get_rank
from ..exceptions import OLMoConfigurationError
from .config import INITIAL_LR_FIELD, LR_FIELD

if TYPE_CHECKING:
    from olmo_core.train import Trainer

log = logging.getLogger(__name__)


def _warn_rank0(message: str, category: type[Warning], *, stacklevel: int = 1) -> None:
    if get_rank() == 0:
        warnings.warn(message, category, stacklevel=stacklevel + 1)


class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"


@dataclass
class Scheduler(Config, Registrable, metaclass=ABCMeta):
    """
    Learning rate scheduler base class.
    """

    lr_field: str = LR_FIELD
    initial_lr_field: str = INITIAL_LR_FIELD
    units: SchedulerUnits = SchedulerUnits.steps

    @abstractmethod
    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        """
        Get the learning rate given the initial/max learning rate, current step/token count, and the maximum
        number of steps/tokens.
        """
        raise NotImplementedError

    def set_lr(self, group: Dict[str, Any], trainer: "Trainer") -> Union[float, torch.Tensor]:
        """
        Set the learning rate on an optimizer param group given a trainer's state.
        """
        if (lr_field := self.lr_field) not in group and (
            initial_lr_field := self.initial_lr_field
        ) not in group:
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in group.items() if k != "params"]
            )
            raise RuntimeError(
                f"learning rate field '{lr_field}' and initial learning rate field "
                f"'{initial_lr_field}' not found in optimizer param group "
                f"with {len(group['params'])} parameter(s):\n"
                f" - {group_fields_list}"
            )

        # Ensure 'initial_lr' is set.
        if group.get(self.initial_lr_field) is None:
            group[self.initial_lr_field] = group[self.lr_field]

        # Set new LR.
        if self.units == SchedulerUnits.steps:
            if trainer.max_steps is None:
                raise OLMoConfigurationError(
                    "'max_steps' must be known in the trainer for step-based scheduling."
                )
            new_lr = self.get_lr(
                group[self.initial_lr_field],
                trainer.global_step,
                trainer.max_steps,
            )
        elif self.units == SchedulerUnits.tokens:
            if trainer.max_tokens is None:
                raise OLMoConfigurationError(
                    "'max_tokens' must be known in the trainer for token-based scheduling."
                )
            new_lr = self.get_lr(
                group[self.initial_lr_field],
                trainer.global_train_tokens_seen,
                trainer.max_tokens,
            )
        else:
            raise NotImplementedError(self.units)

        if isinstance(current_lr := group.get(self.lr_field), torch.Tensor):
            current_lr.fill_(new_lr)
        else:
            group[self.lr_field] = new_lr

        return new_lr


@Scheduler.register("constant")
@dataclass
class ConstantScheduler(Scheduler):
    """
    Constant learning rate schedule, basically a no-op.
    """

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        del current, t_max
        return initial_lr


@Scheduler.register("constant_with_warmup")
@dataclass
class ConstantWithWarmup(Scheduler):
    """
    Constant learning rate schedule with a warmup.
    """

    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self, *args):
        del args
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("'warmup_fraction' must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current <= warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)

        return initial_lr


@Scheduler.register("wsd")
@dataclass
class WSD(Scheduler):
    """
    Warmup-stable-decay scheduler
    """

    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    decay: Optional[int] = None
    decay_steps: Optional[int] = None  # deprecated, use 'decay' instead.
    decay_fraction: Optional[float] = 0.1
    warmup_min_lr: float = 0.0
    decay_min_lr: float = 0.0

    def __post_init__(self, *args):
        del args
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

        if self.decay is None and self.decay_steps is not None:
            self.decay = self.decay_steps
            self.decay_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.decay_steps' is deprecated, please use '.decay' instead.",
                DeprecationWarning,
            )

        if (self.decay_fraction is None) == (self.decay is None):
            raise OLMoConfigurationError(
                "Either 'decay_fraction' or 'decay' must be specified. Never both."
            )

        if self.decay_fraction is not None and (self.decay_fraction < 0 or self.decay_fraction > 1):
            raise OLMoConfigurationError("decay_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current <= warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)

        if self.decay is None:
            assert self.decay_fraction is not None
            decay = round(t_max * self.decay_fraction)
        else:
            decay = self.decay

        if current >= t_max - decay:
            return _linear_decay(initial_lr, t_max - current, decay, self.decay_min_lr)

        return initial_lr


@Scheduler.register("power_lr")
@dataclass
class PowerLR(Scheduler):
    """
    Power learning‑rate schedule with

    1. **Linear warm‑up** to a reference peak LR (`initial_lr`) during the first
       `warmup` steps/tokens.
    2. **Power phase** where the LR decays following a power‑law
       `lr = initial_lr * (current / warmup) ** b`.
       This makes the LR independent of the eventual training horizon.
    3. **Optional linear decay tail** during the last `decay` steps/tokens to
       smoothly anneal to `decay_min_lr`.

    Notes
    -----
    * `b` should be *negative* (e.g. ‑0.51); magnitude controls how quickly the
      LR decays in the power phase.
    * If both `warmup` and `warmup_fraction` (or both `decay` and
      `decay_fraction`) are specified, an `OLMoConfigurationError` is raised to
      mirror the behaviour of other schedulers in this file.
    """

    b: float = -0.51  # power‑law exponent (negative)
    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated alias
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    decay: Optional[int] = None
    decay_steps: Optional[int] = None  # deprecated alias
    decay_fraction: Optional[float] = 0.1
    decay_min_lr: float = 0.0

    def __post_init__(self, *args):
        del args
        # --- handle deprecated aliases -------------------------------------------------
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )
        if self.decay is None and self.decay_steps is not None:
            self.decay = self.decay_steps
            self.decay_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.decay_steps' is deprecated, please use '.decay' instead.",
                DeprecationWarning,
            )

        # --- sanity checks -------------------------------------------------------------
        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and not (0 <= self.warmup_fraction <= 1):
            raise OLMoConfigurationError("'warmup_fraction' must be between 0 and 1.")

        if (self.decay_fraction is None) == (self.decay is None):
            raise OLMoConfigurationError(
                "Either 'decay_fraction' or 'decay' must be specified. Never both."
            )

        if self.decay_fraction is not None and not (0 <= self.decay_fraction <= 1):
            raise OLMoConfigurationError("'decay_fraction' must be between 0 and 1.")

        if self.b >= 0:
            raise OLMoConfigurationError("'b' must be negative for a decaying power‑law.")

    # -------------------------------------------------------------------------
    # Core scheduling logic
    # -------------------------------------------------------------------------
    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        """
        Compute the learning rate for the given *current* step/token count.

        *Linear warm‑up*:
            lr = warmup_min_lr + (initial_lr - warmup_min_lr) * current / warmup

        *Power phase*:
            lr = initial_lr * (current / warmup) ** b

        *Linear decay tail* (last ``decay`` steps/tokens):
            lr is linearly annealed from the power‑phase value at the start of
            the tail to ``decay_min_lr``.
        """
        # --- warm‑up and decay extents ------------------------------------------------
        if self.warmup is not None:
            warmup = self.warmup
        else:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        if self.decay is not None:
            decay = self.decay
        else:
            assert self.decay_fraction is not None
            decay = round(t_max * self.decay_fraction)

        # --- phase 1: warm‑up ---------------------------------------------------------
        if current <= warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)

        # --- phase 3: linear decay tail ----------------------------------------------
        if current >= t_max - decay:
            # lr at the beginning of decay‑tail
            lr_start_tail = initial_lr * ((t_max - decay) / warmup) ** self.b
            return _linear_decay(lr_start_tail, t_max - current, decay, self.decay_min_lr)

        # --- phase 2: power‑law region -----------------------------------------------
        lr = initial_lr * (current / warmup) ** self.b
        return lr


@Scheduler.register("linear_with_warmup")
@dataclass
class LinearWithWarmup(Scheduler):
    """
    Linear learning rate schedule with a warmup.
    """

    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self, *args):
        del args
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        t_max = t_max if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return eta_min
        else:
            current = current - warmup
            t_max = t_max - warmup
            return initial_lr - (initial_lr - eta_min) * (current / t_max)


@Scheduler.register("inv_sqrt_with_warmup")
@dataclass
class InvSqrtWithWarmup(Scheduler):
    """
    Inverse square root learning rate (LR) schedule with a warmup.
    """

    alpha_f: float = 0.1
    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self, *args):
        del args
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)

        eta_min = initial_lr * self.alpha_f
        return eta_min + (initial_lr - eta_min) * sqrt(warmup / current)


@Scheduler.register("cos_with_warmup")
@dataclass
class CosWithWarmup(Scheduler):
    """
    Cosine learning rate schedule with a warmup.
    """

    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self, *args):
        del args
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        t_max = t_max if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return eta_min
        else:
            current = current - warmup
            t_max = t_max - warmup
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * current / t_max)) / 2


@Scheduler.register("half_cos_with_warmup")
@dataclass
class HalfCosWithWarmup(Scheduler):
    """
    Second half of a cosine learning rate schedule, with a warmup before that.
    Note: This assumes that the peak LR set is for the full cosine schedule.
    """

    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self, *args):
        del args
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        t_max = t_max if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            max_lr = eta_min + (initial_lr - eta_min) / 2
            return _linear_warmup(max_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return eta_min
        else:
            current = current - warmup
            t_max = t_max - warmup
            current += t_max
            t_max *= 2
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * current / t_max)) / 2


@Scheduler.register("cos_with_warmup_and_linear_decay")
@dataclass
class CosWithWarmupAndLinearDecay(CosWithWarmup):
    """
    Cosine learning rate schedule with a warmup, cut short at the end and followed by a linear decay.
    """

    decay: Optional[int] = None
    decay_steps: Optional[int] = None  # deprecated, use 'decay' instead.
    decay_fraction: Optional[float] = 0.1
    decay_min_lr: float = 0.0

    def __post_init__(self, *args):
        del args
        super().__post_init__()

        if self.decay is None and self.decay_steps is not None:
            self.decay = self.decay_steps
            self.decay_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.decay_steps' is deprecated, please use '.decay' instead.",
                DeprecationWarning,
            )

        if (self.decay_fraction is None) == (self.decay is None):
            raise OLMoConfigurationError("Either 'decay_fraction' or 'decay' must be specified.")

        if self.decay_fraction is not None and (self.decay_fraction < 0 or self.decay_fraction > 1):
            raise OLMoConfigurationError("'decay_fraction' must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if self.decay is None:
            assert self.decay_fraction is not None
            decay = round(t_max * self.decay_fraction)
        else:
            decay = self.decay

        if current >= t_max - decay:
            final_cosine_lr = super().get_lr(initial_lr, t_max - decay, t_max)
            return _linear_decay(final_cosine_lr, t_max - current, decay, self.decay_min_lr)

        return super().get_lr(initial_lr, current, t_max)


class ComposableSchedulerStageType(StrEnum):
    linear = "linear"
    cosine = "cosine"


@dataclass
class ComposableSchedulerStage(Config):
    """
    A single stage in :class:`ComposableScheduler`.
    """

    duration: int
    shape: ComposableSchedulerStageType = ComposableSchedulerStageType.linear

    start_lr: Optional[float] = None
    start_lr_fraction: Optional[float] = None

    end_lr: Optional[float] = None
    end_lr_fraction: Optional[float] = None

    def __post_init__(self):
        if self.duration <= 0:
            raise OLMoConfigurationError("'duration' must be > 0 for every stage.")

        if self.start_lr is not None and self.start_lr_fraction is not None:
            raise OLMoConfigurationError(
                "Specify at most one of 'start_lr' or 'start_lr_fraction' for a stage."
            )

        if self.start_lr is not None and self.start_lr < 0:
            raise OLMoConfigurationError("'start_lr' must be >= 0.")

        if self.start_lr_fraction is not None and self.start_lr_fraction < 0:
            raise OLMoConfigurationError("'start_lr_fraction' must be >= 0.")

        if (self.end_lr is None) == (self.end_lr_fraction is None):
            raise OLMoConfigurationError(
                "Specify exactly one of 'end_lr' or 'end_lr_fraction' for each stage."
            )

        if self.end_lr is not None and self.end_lr < 0:
            raise OLMoConfigurationError("'end_lr' must be >= 0.")

        if self.end_lr_fraction is not None and self.end_lr_fraction < 0:
            raise OLMoConfigurationError("'end_lr_fraction' must be >= 0.")


@dataclass
class OverrideDecay(Config):
    """
    Optional decay override for :class:`ComposableScheduler`.

    When active, the main stage schedule is ignored from ``start`` onward.
    The override starts from the LR that the main schedule would have produced at ``start``,
    and then decays to ``end_lr`` (or ``end_lr_fraction`` of ``initial_lr``) over ``duration``.
    """

    start: int
    duration: int
    shape: ComposableSchedulerStageType = ComposableSchedulerStageType.linear

    end_lr: Optional[float] = None
    end_lr_fraction: Optional[float] = None

    def __post_init__(self):
        if self.start < 0:
            raise OLMoConfigurationError("'start' must be >= 0 for override decay.")

        if self.duration <= 0:
            raise OLMoConfigurationError("'duration' must be > 0 for override decay.")

        if (self.end_lr is None) == (self.end_lr_fraction is None):
            raise OLMoConfigurationError(
                "Specify exactly one of 'end_lr' or 'end_lr_fraction' for override decay."
            )

        if self.end_lr is not None and self.end_lr < 0:
            raise OLMoConfigurationError("'end_lr' must be >= 0 for override decay.")

        if self.end_lr_fraction is not None and self.end_lr_fraction < 0:
            raise OLMoConfigurationError("'end_lr_fraction' must be >= 0 for override decay.")


@Scheduler.register("composable")
@dataclass
class ComposableScheduler(Scheduler):
    """
    Piecewise LR schedule composed of multiple stages.

    - Each stage has a duration and interpolation shape (linear/cosine).
    - Stage start LR defaults to the previous stage's end LR.
    - Stage end LR is required and can be absolute or as a fraction of ``initial_lr``.
    - After all stages are exhausted, LR stays constant at the last stage's end LR.
    - The ``t_max`` argument passed to :meth:`get_lr` is **ignored**: the schedule is
      defined absolutely by the stage durations (and the optional :data:`override_decay`),
      so it does not rescale to fit the trainer's max horizon.
    """

    stages: List[ComposableSchedulerStage] = field(default_factory=list)
    override_decay: Optional[OverrideDecay] = None

    _warned_t_max_ignored: bool = field(default=False, init=False, repr=False)

    def __post_init__(self, *args):
        del args
        if len(self.stages) == 0:
            raise OLMoConfigurationError("'stages' must be specified and non-empty.")

    def _resolve_stage_start(
        self,
        stage: ComposableSchedulerStage,
        initial_lr: Union[float, torch.Tensor],
        previous_end_lr: Union[float, torch.Tensor],
    ) -> Union[float, torch.Tensor]:
        if stage.start_lr is None and stage.start_lr_fraction is None:
            return previous_end_lr
        return _resolve_lr_from_initial(initial_lr, stage.start_lr, stage.start_lr_fraction)

    def _main_schedule_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int
    ) -> Union[float, torch.Tensor]:
        current = max(current, 0)
        stage_start = 0
        previous_end_lr: Union[float, torch.Tensor] = initial_lr
        for stage in self.stages:
            start_lr = self._resolve_stage_start(stage, initial_lr, previous_end_lr)
            end_lr = _resolve_lr_from_initial(initial_lr, stage.end_lr, stage.end_lr_fraction)

            stage_end = stage_start + stage.duration
            if current < stage_end:
                stage_current = current - stage_start
                return _interpolate_lr(
                    shape=stage.shape,
                    start_lr=start_lr,
                    end_lr=end_lr,
                    current=stage_current,
                    duration=stage.duration,
                )

            previous_end_lr = end_lr
            stage_start = stage_end

        return previous_end_lr

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        """
        Compute the LR at ``current``.

        .. note::
            ``t_max`` is ignored. The schedule is defined absolutely by the stage
            durations (and the optional :data:`override_decay`), independent of
            the trainer's max horizon.
        """
        if not self._warned_t_max_ignored:
            total_duration = sum(stage.duration for stage in self.stages)
            _warn_rank0(
                f"'{self.__class__.__name__}' ignores 't_max'; the schedule is defined "
                f"absolutely by stage durations (total={total_duration}, "
                f"t_max={t_max}). The LR will not rescale to fit the trainer's horizon.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_t_max_ignored = True
        del t_max
        current = max(current, 0)

        if self.override_decay is None or current < self.override_decay.start:
            return self._main_schedule_lr(initial_lr, current)

        override_decay = self.override_decay
        override_start_lr = self._main_schedule_lr(initial_lr, override_decay.start)
        override_end_lr = _resolve_override_decay_end(override_decay, initial_lr)
        override_current = current - override_decay.start

        if override_current < override_decay.duration:
            return _interpolate_lr(
                shape=override_decay.shape,
                start_lr=override_start_lr,
                end_lr=override_end_lr,
                current=override_current,
                duration=override_decay.duration,
            )

        return override_end_lr


def _resolve_lr_from_initial(
    initial_lr: Union[float, torch.Tensor],
    value: Optional[float],
    fraction: Optional[float],
) -> Union[float, torch.Tensor]:
    if value is not None:
        return value
    assert fraction is not None
    return initial_lr * fraction


def _resolve_override_decay_end(
    override_decay: OverrideDecay,
    initial_lr: Union[float, torch.Tensor],
) -> Union[float, torch.Tensor]:
    return _resolve_lr_from_initial(
        initial_lr, override_decay.end_lr, override_decay.end_lr_fraction
    )


def _interpolate_lr(
    shape: ComposableSchedulerStageType,
    start_lr: Union[float, torch.Tensor],
    end_lr: Union[float, torch.Tensor],
    current: int,
    duration: int,
) -> Union[float, torch.Tensor]:
    if shape == ComposableSchedulerStageType.linear:
        return start_lr + (end_lr - start_lr) * current / duration
    elif shape == ComposableSchedulerStageType.cosine:
        return end_lr + (start_lr - end_lr) * (1 + cos(pi * current / duration)) / 2
    else:
        raise NotImplementedError(shape)


def _linear_warmup(
    initial_lr: Union[float, torch.Tensor], current: int, warmup: int, warmup_min_lr: float = 0.0
) -> Union[float, torch.Tensor]:
    if isinstance(initial_lr, float):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= warmup_min_lr < initial_lr
    return warmup_min_lr + (initial_lr - warmup_min_lr) * min(current, warmup) / warmup


def _linear_decay(
    initial_lr: Union[float, torch.Tensor],
    step_from_end: int,
    decay: int,
    decay_min_lr: float = 0.0,
) -> Union[float, torch.Tensor]:
    if isinstance(initial_lr, float):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= decay_min_lr < initial_lr

    return decay_min_lr + (initial_lr - decay_min_lr) * min(step_from_end, decay) / decay


@Scheduler.register("sequential")
@dataclass
class SequentialScheduler(Scheduler):
    """
    A scheduler that calls a sequence of schedulers sequentially during the optimization
    process. The initial LR of a scheduler in the sequence is set to the final LR of the
    previous scheduler.
    """

    schedulers: List[Scheduler] = field(default_factory=lambda: [ConstantWithWarmup()])
    schedulers_max: Optional[List[int]] = None
    """
    A list of the steps or token counts for which each scheduler runs.
    The last scheduler is assumed to run until the end of training, so any value provided for it is ignored.
    """
    schedulers_max_steps: Optional[List[int]] = None  # deprecated, use 'schedulers_max' instead.

    override_decay: Optional[OverrideDecay] = None
    """
    Optional late-stage override. When ``current >= override_decay.start``, the
    sub-scheduler sequence is bypassed and the LR decays from "whatever the main
    sequence would have produced at ``start``" to the override's target over
    ``duration`` (linear or cosine). After ``start + duration``, the LR is held
    at the override's end LR.

    .. note::
        While the override is active, ``t_max`` is ignored — the override is
        defined absolutely by ``start`` and ``duration``.
    """

    _warned_t_max_ignored: bool = field(default=False, init=False, repr=False)

    def __post_init__(self, *args):
        del args
        if self.schedulers_max is None and self.schedulers_max_steps is not None:
            self.schedulers_max = self.schedulers_max_steps
            self.schedulers_max_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.schedulers_max_steps' is deprecated, please use '.schedulers_max' instead.",
                DeprecationWarning,
            )

        if self.schedulers_max is None:
            raise OLMoConfigurationError("'schedulers_max' must be specified")

        if len(self.schedulers_max) == len(self.schedulers):
            log.info(
                "Max steps are set for the last scheduler in sequential scheduling. "
                "The last scheduler is assumed to run until the end of training, so this value is ignored."
            )
            self.schedulers_max.pop()

        if len(self.schedulers_max) + 1 != len(self.schedulers):
            raise OLMoConfigurationError(
                f"Max steps must be set for all schedulers except the last when using '{self.__class__.__name__}'"
            )

    def _sequential_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        assert self.schedulers_max is not None

        # Call schedulers sequentially until the current step/token count is within the max steps/token count
        # of the scheduler or the last scheduler is reached.
        for scheduler, scheduler_max in zip(self.schedulers[:-1], self.schedulers_max, strict=True):
            if current <= scheduler_max:
                return scheduler.get_lr(initial_lr, current, min(t_max, scheduler_max))

            # The next scheduler's initial LR should be the final LR of the current schedule
            initial_lr = scheduler.get_lr(initial_lr, scheduler_max, scheduler_max)

            current -= scheduler_max
            t_max -= scheduler_max

        assert t_max > 0
        return self.schedulers[-1].get_lr(initial_lr, current, t_max)

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        assert 0 <= current <= t_max

        if self.override_decay is None or current < self.override_decay.start:
            return self._sequential_lr(initial_lr, current, t_max)

        if not self._warned_t_max_ignored:
            _warn_rank0(
                f"'{self.__class__.__name__}' ignores 't_max' once 'override_decay' is active; "
                f"the override is defined absolutely by 'start' ({self.override_decay.start}) "
                f"and 'duration' ({self.override_decay.duration}) (t_max={t_max}).",
                UserWarning,
                stacklevel=2,
            )
            self._warned_t_max_ignored = True

        override_decay = self.override_decay
        override_start_lr = self._sequential_lr(initial_lr, override_decay.start, t_max)
        override_end_lr = _resolve_override_decay_end(override_decay, initial_lr)
        override_current = current - override_decay.start

        if override_current < override_decay.duration:
            return _interpolate_lr(
                shape=override_decay.shape,
                start_lr=override_start_lr,
                end_lr=override_end_lr,
                current=override_current,
                duration=override_decay.duration,
            )

        return override_end_lr


@Scheduler.register("wsds")
@dataclass
class WSDS(Scheduler):
    """
    Warmup–Stable–Decay—Simplified (WSD‑S) scheduler for continual pretraining.
    Reference: https://arxiv.org/abs/2410.05192
    """

    period_lengths: List[int] = field(default_factory=list)
    period_lr_multipliers: Optional[List[float]] = None

    warmup: Optional[int] = None
    warmup_fraction: Optional[float] = None

    decay: Optional[int] = None
    decay_fraction: Optional[float] = None

    warmup_min_lr: float = 0.0
    decay_min_lr: float = 0.0

    _cum_period_end: List[int] = field(default_factory=list, init=False, repr=False)
    _warmup_steps: int = field(default=0, init=False, repr=False)
    _adjusted_period_lengths: List[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self, *args):
        del args
        if not self.period_lengths:
            raise OLMoConfigurationError("'period_lengths' must be provided and non-empty.")
        if any(p <= 0 for p in self.period_lengths):
            raise OLMoConfigurationError("All entries in 'period_lengths' must be > 0.")
        if self.period_lr_multipliers is not None:
            if len(self.period_lr_multipliers) != len(self.period_lengths):
                raise OLMoConfigurationError(
                    "'period_lr_multipliers' length must match 'period_lengths' length."
                )
            if any(m <= 0.0 for m in self.period_lr_multipliers):
                raise OLMoConfigurationError("All entries in 'period_lr_multipliers' must be > 0.")

        # warmup validation
        if (self.warmup is None) == (self.warmup_fraction is None):
            raise OLMoConfigurationError(
                "Exactly one of 'warmup' or 'warmup_fraction' must be specified."
            )
        if self.warmup_fraction is not None and not (0.0 <= self.warmup_fraction <= 1.0):
            raise OLMoConfigurationError("'warmup_fraction' must be in [0, 1].")

        # decay validation
        if (self.decay is None) == (self.decay_fraction is None):
            raise OLMoConfigurationError(
                "Exactly one of 'decay' or 'decay_fraction' must be specified."
            )
        if self.decay_fraction is not None and not (0.0 <= self.decay_fraction <= 1.0):
            raise OLMoConfigurationError("'decay_fraction' must be in [0, 1].")
        if self.decay_min_lr < 0.0:
            raise OLMoConfigurationError("'decay_min_lr' must be >= 0.")

        # Resolve warmup based on first period length
        L0 = self.period_lengths[0]
        if self.warmup is not None:
            self._warmup_steps = int(self.warmup)
        else:
            assert self.warmup_fraction is not None
            self._warmup_steps = int(round(self.warmup_fraction * L0))

        # Validate first period: warmup + decay <= L0
        D0 = self._resolve_decay(L0)
        if self._warmup_steps + D0 > L0:
            raise OLMoConfigurationError(
                f"First period: warmup ({self._warmup_steps}) + decay ({D0}) = "
                f"{self._warmup_steps + D0} exceeds period length ({L0})."
            )

        # Validate remaining periods: decay <= Li
        for i, Li in enumerate(self.period_lengths[1:], start=1):
            Di = self._resolve_decay(Li)
            if Di > Li:
                raise OLMoConfigurationError(
                    f"Period {i}: decay ({Di}) exceeds period length ({Li})."
                )

        # Adjust period lengths: subtract warmup from first period
        self._adjusted_period_lengths = [L0 - self._warmup_steps] + self.period_lengths[1:]

        # Precompute cumulative ends based on ADJUSTED periods
        self._cum_period_end = np.cumsum(self._adjusted_period_lengths).tolist()

    def _resolve_decay(self, Li: int) -> int:
        if self.decay is not None:
            return int(self.decay)
        else:
            assert self.decay_fraction is not None
            return int(round(self.decay_fraction * Li))

    def _find_period(self, x: int) -> int:
        for idx, end in enumerate(self._cum_period_end):
            if x <= end:
                return idx
        return len(self._cum_period_end) - 1

    def _get_peak_lr(
        self, initial_lr: Union[float, torch.Tensor], pidx: int
    ) -> Union[float, torch.Tensor]:
        if self.period_lr_multipliers is None:
            return initial_lr
        else:
            return initial_lr * self.period_lr_multipliers[pidx]

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        del t_max
        if current < self._warmup_steps:
            return _linear_warmup(
                self._get_peak_lr(initial_lr, 0), current, self._warmup_steps, self.warmup_min_lr
            )

        adjusted_current = current - self._warmup_steps

        if adjusted_current >= self._cum_period_end[-1]:
            return self.decay_min_lr

        # Find current period (using adjusted boundaries)
        pidx = self._find_period(adjusted_current)
        start = 0 if pidx == 0 else self._cum_period_end[pidx - 1]
        Li = self._adjusted_period_lengths[pidx]
        pos = min(max(adjusted_current - start, 0), Li)

        D = self._resolve_decay(self.period_lengths[pidx])
        S = Li - D

        if pos < S:
            return self._get_peak_lr(initial_lr, pidx)
        else:
            t = pos - S
            return _linear_decay(self._get_peak_lr(initial_lr, pidx), D - t, D, self.decay_min_lr)


@Scheduler.register("exponential")
@dataclass
class ExponentialScheduler(Scheduler):
    """
    Exponential learning rate schedule that increases from a minimum LR to a maximum LR. Thus:
        - lr(0) = lr_min
        - lr(t_max) = initial_lr
    """

    lr_min: float = 1e-9

    def __post_init__(self, *args):
        del args
        if self.lr_min <= 0:
            raise OLMoConfigurationError("'lr_min' must be positive.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if current >= t_max:
            return initial_lr

        if current == 0:
            return self.lr_min

        # Exponential growth: lr(t) = lr_min * (lr_max / lr_min)^(t / t_max)
        ratio = current / t_max
        if isinstance(initial_lr, torch.Tensor):
            growth_factor = torch.pow(initial_lr / self.lr_min, ratio)
        else:
            growth_factor = (initial_lr / self.lr_min) ** ratio

        return self.lr_min * growth_factor


@Scheduler.register("per_group")
@dataclass
class PerGroupScheduler(Scheduler):
    """
    Dispatch to a different sub-scheduler per optimizer param group.

    OLMo-core calls :meth:`Scheduler.set_lr` once per param group with the *same*
    scheduler object, so a single scheduler cannot apply different warmups to
    different groups. This wrapper reads a ``scheduler_name`` key from each param
    group (set via :class:`~olmo_core.optim.OptimGroupOverride`'s ``opts``) and
    delegates to the matching sub-scheduler, falling back to :attr:`default`.

    This reproduces mm_olmo's ``multimodal`` scheduler, where the connector and LM
    use different warmups (e.g. 200 vs 2000 steps) but the same cosine decay.

    Example::

        PerGroupScheduler(
            schedulers={"connector": CosWithWarmup(warmup=200, alpha_f=0.1)},
            default=CosWithWarmup(warmup=2000, alpha_f=0.1),
        )

    with an ``OptimGroupOverride(params=["connector.*"], opts={"scheduler_name": "connector"})``.
    """

    schedulers: Dict[str, Scheduler] = field(default_factory=dict)
    """Sub-schedulers keyed by the param group's ``scheduler_name``."""

    default: Optional[Scheduler] = None
    """Scheduler used for groups without a matching ``scheduler_name``."""

    group_name_field: str = "scheduler_name"
    """The param-group key used to select a sub-scheduler."""

    def __post_init__(self, *args):
        if self.default is None:
            raise OLMoConfigurationError("PerGroupScheduler requires a 'default' scheduler")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        raise NotImplementedError(
            "PerGroupScheduler dispatches at the param-group level; use set_lr()."
        )

    def _scheduler_for(self, group: Dict[str, Any]) -> Scheduler:
        name = group.get(self.group_name_field)
        if name is not None and name in self.schedulers:
            return self.schedulers[name]
        assert self.default is not None
        return self.default

    def set_lr(self, group: Dict[str, Any], trainer: "Trainer") -> Union[float, torch.Tensor]:
        return self._scheduler_for(group).set_lr(group, trainer)
