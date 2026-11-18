import sys

import pytest

from olmo_core.train.callbacks.beaker import BeakerCallback


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        (
            {
                "BEAKER_JOB_ID": "job",
                "BEAKER_NODE_ID": "node",
                "BEAKER_JOB_KIND": "batch",
            },
            True,
        ),
    ],
)
def test_post_attach_does_not_import_optional_beaker_dependencies(monkeypatch, env, expected):
    for name in ("BEAKER_JOB_ID", "BEAKER_NODE_ID", "BEAKER_JOB_KIND"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    # Importing olmo_core.launch.beaker would load the optional beaker and gantry packages.
    monkeypatch.setitem(sys.modules, "olmo_core.launch.beaker", None)

    callback = BeakerCallback()
    callback.post_attach()

    assert callback.enabled is expected
