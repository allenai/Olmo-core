"""Three authorized 100B-token MT continuations with identical reference-scaled K8."""

from dataclasses import dataclass
from pathlib import Path

import olmoe3_qkgain_plan as base

CAMPAIGN = "adaptive-small-mt-20261004-32g"
BRANCH = "jacobm/adaptive-compute-2026-10-04-mt"
SCRIPT = "src/examples/olmo_ddp/olmoe3_adaptive_mt.py"
WORKSPACE = "ai2/olmo3p5-training"
GPUS = 32
GPUS_PER_NODE = 8
NODES = GPUS // GPUS_PER_NODE
ROOT = base.MOUNT / "adaptive-compute-redux" / CAMPAIGN
EVAL = ROOT / "evals"
AUTO = Path(
    "/weka/oe-adapt-default/jacobm/adaptive-compute-redux/results/data/olmo_small_mt_2026-10-04"
)
START = 0
SOURCE_STEP = 120000
REQUESTED_TOKENS = 100_000_000_000
BATCH = 16777216
END = (REQUESTED_TOKENS + BATCH - 1) // BATCH
SEED = 1387822106
WARMUP = 2000
SCHEDULES = ("fixed8", "gradual8", "native8")  # Explicit submission order.
DECAY = base.MOUNT / "adaptive-compute-redux/adaptive-small-decay-20261003-32g"
SOURCES = {
    "fixed8": DECAY / "adaptive-small-decay-20261003-32g-fixed8/step120000",
    "gradual8": DECAY / "adaptive-small-decay-20261003-32g-gradual8/step120000",
    "native8": base.MOUNT / "production-dolci-hero/olmo35-dolci-hero-20260920/olmo35-dolci-hero-20260920-hero/step120000",
}


@dataclass(frozen=True)
class Run(base.Run):
    """Immutable identities, checkpoint roots and matched global training recipe."""

    schedule: str = "fixed8"

    @property
    def gpus(self):
        return GPUS

    @property
    def nodes(self):
        return NODES

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{self.schedule}"

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def prefix(self):
        return f"adaptive-compute-redux/{CAMPAIGN}/{self.schedule}"

    @property
    def start(self):
        return START

    @property
    def end(self):
        return END

    @property
    def source(self):
        return SOURCES[self.schedule]

    @property
    def emo(self):
        return False

    @property
    def hf(self):
        return EVAL / self.run_id / f"step{END}" / "hf"

    @property
    def saves(self):
        return sorted({2, 4, 1491, 2981, END, *range(500, END, 500)})

    def as_dict(self):
        return dict(
            super().as_dict(),
            schedule=self.schedule,
            source_world_size=128 if self.schedule == "native8" else 32,
            start=START,
            added_tokens=(END - START) * BATCH,
            reference_top_k=16,
            normalization="native-top16-denominator-times16",
            initial_gpus=128 if self.schedule == "native8" else 32,
            accumulation=self.batch // (self.gpus * self.microbatch),
            transfer="Weights and model buffers; optimizer and data counters reset",
        )


def run(schedule):
    """Resolve one explicitly authorized arm."""
    if schedule not in SCHEDULES:
        raise ValueError(schedule)
    return Run("3to1-shared", "mt", False, schedule)


def runs(smoke=False):
    """Return the three full runs; their startup checks are inside their allocations."""
    if smoke:
        return []
    return [run(s) for s in SCHEDULES]


def find_run(name):
    """Resolve an exact owned run identifier."""
    return next(r for r in runs() if r.run_id == name)


def install():
    """Bind the historical recipe adapter to this campaign before importing it."""
    base.CAMPAIGN, base.BRANCH, base.ROOT, base.EVAL_ROOT = CAMPAIGN, BRANCH, ROOT, EVAL
    base.AUTOMATION, base.WORKSPACE = AUTO, WORKSPACE
    base.runs, base.find_run = runs, find_run


def self_test():
    assert END == 5961 and END * BATCH == 100_008_984_576
    assert tuple(r.schedule for r in runs()) == SCHEDULES
    for r in runs():
        assert r.batch // (r.gpus * r.microbatch) == 16
        assert r.lr == 2.2e-4 and r.sequence == 8192 and r.start == 0
        assert r.source.name == "step120000" and r.end in r.saves
        assert r.source != r.root and r.root.is_relative_to(ROOT)


if __name__ == "__main__":
    self_test()
    print("ADAPTIVE_MT_PLAN_PASSED")
