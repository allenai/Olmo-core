"""CheckpointerCallback: an interval save that lands on the final step is synchronous and post_train reuses it."""
from types import SimpleNamespace

from olmo_core.train.callbacks.checkpointer import CheckpointerCallback


class _Fut:
    def result(self):
        return None

    def done(self):
        return True


def _setup(max_steps, hard_stop=None):
    calls = []
    t = SimpleNamespace(global_step=0, max_steps=max_steps, hard_stop=hard_stop, block_ephemeral_checkpoints=False,
                        checkpointer=SimpleNamespace(dir_is_checkpoint=lambda p: True),
                        _get_max_steps=lambda d: d)

    def save(ephemeral=False):
        calls.append((t.global_step, "sync"))
        return f"step{t.global_step}"

    def save_async(ephemeral=False):
        calls.append((t.global_step, "async"))
        return f"step{t.global_step}", _Fut()

    t.save_checkpoint, t.save_checkpoint_async = save, save_async
    cb = CheckpointerCallback(save_interval=60, save_async=True, max_checkpoints=None)
    cb._trainer = t  # type: ignore[assignment]
    return cb, t, calls


def _run(cb, t, steps):
    for s in range(1, steps + 1):
        t.global_step = s
        cb.post_train_batch()
    cb.post_train()


def test_interval_save_on_the_final_step_is_synchronous_and_reused():
    cb, t, calls = _setup(max_steps=120)
    _run(cb, t, 120)
    assert calls == [(60, "async"), (120, "sync")]  # no second save in post_train


def test_final_step_not_on_the_interval_keeps_async_interval_saves():
    cb, t, calls = _setup(max_steps=130)
    _run(cb, t, 130)
    assert calls == [(60, "async"), (120, "async"), (130, "sync")]


def test_hard_stop_counts_as_the_final_step():
    cb, t, calls = _setup(max_steps=1000, hard_stop=120)
    _run(cb, t, 120)
    assert calls == [(60, "async"), (120, "sync")]
