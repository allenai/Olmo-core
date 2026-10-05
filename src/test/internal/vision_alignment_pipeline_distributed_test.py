import json
import socket
import textwrap
from contextlib import ExitStack

import pytest

from olmo_core.internal import vision_alignment_pipeline as pipeline
from olmo_core.internal.experiment import CliContext, SubCmd


@pytest.fixture
def stage_ports():
    for _ in range(32):
        with ExitStack() as stack:
            reservations = [stack.enter_context(socket.socket()) for _ in range(3)]
            try:
                reservations[0].bind(("127.0.0.1", 0))
                first_port = reservations[0].getsockname()[1]
                if first_port > 65533:
                    continue
                for index, reservation in enumerate(reservations[1:], start=1):
                    reservation.bind(("127.0.0.1", first_port + index))
            except OSError:
                continue
            yield first_port, reservations
            return
    pytest.fail("Could not reserve three consecutive localhost ports")


def test_three_fresh_cpu_stages_share_allocation_metadata(tmp_path, monkeypatch, stage_ports):
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "TORCHELASTIC_RUN_ID",
        "BEAKER_LEADER_REPLICA_HOSTNAME",
        pipeline._WORKER_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    first_port, reservations = stage_ports
    for name, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "GLOO_SOCKET_IFNAME": "lo",
        "BEAKER_REPLICA_COUNT": "1",
        "BEAKER_REPLICA_RANK": "0",
        "BEAKER_ASSIGNED_GPU_COUNT": "2",
        "BEAKER_EXPERIMENT_ID": "alignment-pipeline-test",
        "MASTER_PORT": str(first_port),
        "ALIGNMENT_TEST_OUTPUT": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)

    worker = tmp_path / "worker.py"
    worker.write_text(
        textwrap.dedent(
            """\
            import json
            import os
            import sys
            from datetime import timedelta
            from pathlib import Path

            import torch
            import torch.distributed as dist

            stage = int(sys.argv[-1].partition("=")[2])
            dist.init_process_group("gloo", timeout=timedelta(seconds=20))
            rank = dist.get_rank()
            value = torch.tensor(rank + 1)
            dist.all_reduce(value)
            result = {
                "argv": sys.argv[1:], "rank": rank, "sum": value.item(),
                "pid": os.getpid(), "parent_pid": os.getppid(),
                "local_rank": int(os.environ["LOCAL_RANK"]),
                "world_size": int(os.environ["WORLD_SIZE"]),
                "local_world_size": int(os.environ["LOCAL_WORLD_SIZE"]),
                "worker": os.environ["OLMO_ALIGNMENT_PIPELINE_STAGE"],
                "port": int(os.environ["MASTER_PORT"]),
                "run_id": os.environ["TORCHELASTIC_RUN_ID"],
            }
            output = Path(os.environ["ALIGNMENT_TEST_OUTPUT"]) / f"stage{stage}-rank{rank}.json"
            output.write_text(json.dumps(result))
            dist.destroy_process_group()
            """
        )
    )
    worker_pids = set()
    launcher_pids = set()
    for index, phase in enumerate(pipeline._STAGES):
        cli = CliContext(
            str(worker), SubCmd.train, f"alignment-{phase}", "local", [f"--stage-index={index}"]
        )
        command = pipeline._torchrun_command(cli, index)
        # Retain the other reservations until their stages start.
        reservations[index].close()
        pipeline._run_command(command)
        stage_launcher_pids = set()
        for rank in range(2):
            result = json.loads((tmp_path / f"stage{index}-rank{rank}.json").read_text())
            assert result["argv"] == ["train", cli.run_name, "local", f"--stage-index={index}"]
            assert result["rank"] == result["local_rank"] == rank
            assert result["world_size"] == result["local_world_size"] == 2
            assert result["sum"] == 3
            assert result["worker"] == "1"
            assert result["port"] == first_port + index
            assert result["run_id"] == f"alignment-pipeline-test-{index}"
            worker_pids.add(result["pid"])
            stage_launcher_pids.add(result["parent_pid"])
        assert len(stage_launcher_pids) == 1
        launcher_pids.update(stage_launcher_pids)
    assert len(worker_pids) == 6
    assert len(launcher_pids) == 3
