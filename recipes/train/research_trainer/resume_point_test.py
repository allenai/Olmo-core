"""--resume-interval: a rolling resume point that survives --no-final-checkpoint (CPU only, no GPU).

Before this flag, the Beaker launcher's default --no-final-checkpoint set CheckpointerCallback
enabled=False, which also silently disabled --save-interval: every preempted run restarted from step 0.

    python -m pytest -q recipes/train/research_trainer/resume_point_test.py
"""
import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import train_ctc_suite as T  # noqa: E402


class FakeTrainer:
    def __init__(self):
        self.global_step = 0
        self.saved = []
        self.block_ephemeral_checkpoints = False
        self.checkpoint_loaded = True

    def save_checkpoint(self, ephemeral=False):
        path = f"/tmp/run/step{self.global_step}"
        self.saved.append((self.global_step, ephemeral))
        return path


def make(skip_final=True, interval=5):
    cb = T.ResumePointCheckpointer(save_interval=None, ephemeral_save_interval=interval,
                                   pre_train_checkpoint=False, skip_final_save=skip_final,
                                   save_async=False, max_checkpoints=None)
    cb.trainer = FakeTrainer()
    return cb


def run_steps(cb, n):
    removed = []
    with mock.patch.object(cb, "_remove_checkpoint", side_effect=removed.append):
        for s in range(1, n + 1):
            cb.trainer.global_step = s
            cb.post_train_batch()
    return removed


def test_rolling_resume_point_keeps_only_newest():
    cb = make()
    removed = run_steps(cb, 12)
    assert cb.trainer.saved == [(5, True), (10, True)]
    assert cb._ephemeral_checkpoints == ["/tmp/run/step10"]
    assert removed == ["/tmp/run/step5"]  # older one deleted on the step after step10 was saved
    assert cb._checkpoints_to_remove == []


def test_no_final_full_save_when_no_final_checkpoint():
    cb = make(skip_final=True)
    run_steps(cb, 7)
    cb.trainer.global_step = 7
    cb.post_train()
    assert cb.trainer.saved == [(5, True)]  # nothing at step 7


def test_final_save_kept_without_no_final_checkpoint():
    cb = make(skip_final=False)
    run_steps(cb, 7)
    cb.trainer.global_step = 7
    cb.post_train()
    assert cb.trainer.saved[-1] == (7, False)


def test_drop_resume_points_after_export():
    cb = make()
    run_steps(cb, 12)
    removed = []
    with mock.patch.object(cb, "_remove_checkpoint", side_effect=removed.append):
        cb.drop_resume_points()
    assert removed == ["/tmp/run/step10"]
    assert cb._ephemeral_checkpoints == [] and cb._checkpoints_to_remove == []


def test_cli_flag_and_wiring_present():
    src = open(T.__file__).read()
    assert '"--resume-interval"' in src
    assert "ResumePointCheckpointer(" in src and "cb.drop_resume_points()" in src
