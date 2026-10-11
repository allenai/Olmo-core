import pytest

from olmo_core.launch.utils import (
    is_running_in_beaker,
    is_running_in_beaker_batch_job,
    parse_git_remote_url,
)


@pytest.mark.parametrize(
    ("env", "in_beaker", "in_batch_job"),
    [
        ({}, False, False),
        ({"BEAKER_JOB_ID": "job"}, False, False),
        ({"BEAKER_JOB_ID": "job", "BEAKER_NODE_ID": "node"}, True, False),
        (
            {
                "BEAKER_JOB_ID": "job",
                "BEAKER_NODE_ID": "node",
                "BEAKER_JOB_KIND": "batch",
            },
            True,
            True,
        ),
    ],
)
def test_beaker_environment_detection(monkeypatch, env, in_beaker, in_batch_job):
    for name in ("BEAKER_JOB_ID", "BEAKER_NODE_ID", "BEAKER_JOB_KIND"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert is_running_in_beaker() is in_beaker
    assert is_running_in_beaker_batch_job() is in_batch_job


def test_parse_git_remote_url():
    # HTTPS format.
    assert parse_git_remote_url("https://github.com/allenai/Olmo-core.git") == (
        "allenai",
        "Olmo-core",
    )
    # SSH format.
    assert parse_git_remote_url("git@github.com:allenai/Olmo-core.git") == (
        "allenai",
        "Olmo-core",
    )
    # Username+password format.
    assert parse_git_remote_url("https://USERNAME:PASSWORD@github.com/allenai/Olmo-core.git") == (
        "allenai",
        "Olmo-core",
    )
