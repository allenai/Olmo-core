"""beaker_ctc_suite.plan_runtime: minRuntime = min(4h, 1.3 x expected), 1h fallback, resume past 4h.

    python -m pytest -q recipes/train/research_trainer/beaker_runtime_test.py
"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "ops"))

import beaker_ctc_suite as B  # noqa: E402
from opslib import runtime  # noqa: E402


class FakeOps:
    def __init__(self, history=None):
        self.history = history

    def suggest_min_runtime(self, run_id=None, expected_minutes=None, launcher=None, kind="train"):
        runs = None
        if self.history is not None:
            runs = {f"{run_id}-old{i}": {"run_id": f"{run_id}-old{i}", "kind": "train", "final_state": "done",
                                         "runtime_minutes": m} for i, m in enumerate(self.history)}
        return runtime.suggest(run_id=run_id, expected_minutes=expected_minutes, runs=runs, launcher=launcher)


def opts(**kw):
    base = dict(expected_minutes=None, min_runtime="auto", resume_interval=None, extra_args="", max_steps=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_fallback_1h_no_resume():
    o = opts()
    mr, resume, _ = B.plan_runtime(o, "f8k-x-s100", FakeOps())
    assert (mr, resume, o.expected_minutes) == ("1h", None, None)


def test_explicit_expected():
    mr, resume, _ = B.plan_runtime(opts(expected_minutes=100), "f8k-x-s100", FakeOps())
    assert (mr, resume) == ("2h10m", None)


def test_history_fills_expected_minutes():
    o = opts()
    mr, resume, notes = B.plan_runtime(o, "f8k-mt4-arm-s1464", FakeOps(history=[90, 100, 110]))
    assert mr == "2h10m" and o.expected_minutes == 100 and "history" in notes[0]


def test_long_run_gets_resume_point():
    mr, resume, _ = B.plan_runtime(opts(expected_minutes=480, max_steps=1952), "r", FakeOps())
    assert mr == "4h" and resume == round(1952 * 30 / 480)
    _, resume, _ = B.plan_runtime(opts(expected_minutes=480), "r", FakeOps())
    assert resume == 200


def test_overrides():
    mr, resume, _ = B.plan_runtime(opts(expected_minutes=480, min_runtime="2h", resume_interval=0), "r", FakeOps())
    assert (mr, resume) == ("2h", None)
    _, resume, _ = B.plan_runtime(opts(expected_minutes=480, extra_args="--resume-interval 50"), "r", FakeOps())
    assert resume is None  # caller already passes one


def test_no_ops_module():
    mr, resume, _ = B.plan_runtime(opts(expected_minutes=300), "r", None)
    assert (mr, resume) == ("1h", 200)
