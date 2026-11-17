"""Tests for opslib.jobs (status join/flags/rendering, Beaker state, stall probe parsing + classifier).

Fixtures are real outputs captured read-only on 2026-10-12 (squeue/sacct on radagast, `beaker job list`
JSON, and the `ops stall` probe run against a hung pytest job on horton), trimmed.
"""
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from opslib import jobs, paths

NOW = 1791868276.0  # epoch of the captured probe

SQUEUE = """\
3813954|envflags_gpu|RUNNING|44:54|60:00|lorax|None|berkeleynlp|preemptive_high_sewonm|2026-10-12T21:41:24|/data/prasann/envflags_gpu_%j.log|/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/flopdrop_port|
3813965|envflags_cpu2|RUNNING|4:27|25:00|horton|None|berkeleynlp|preemptive_high_sewonm|2026-10-12T21:51:51|/data/prasann/envflags_cpu2_215130/run.log|/accounts/x|
3813999|q4b-ffnmoe-v10|PENDING|0:00|4:00:00||Priority|jsteinhardt|preemptive|2026-10-12T19:00:00|/accounts/projects/berkeleynlp/prasann/joblogs/%x_%j.log|/accounts/x|
"""

SACCT = """\
3813195|storage-index|COMPLETED|00:00:10|12:00:00|cubbins|2026-10-12T13:18:15|2026-10-12T13:18:15|2026-10-12T13:18:25|0:0|jsteinhardt|/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
3813329|bash|CANCELLED by 3018|00:05:59|01:00:00|horton|2026-10-12T14:02:27|2026-10-12T14:02:27|2026-10-12T14:08:26|0:0|berkeleynlp|/accounts/x
3813466|flopdrop-gputests|FAILED|00:31:07|01:00:00|mooney|2026-10-12T15:16:24|2026-10-12T15:16:24|2026-10-12T15:47:31|1:0|jsteinhardt|/accounts/x
3813583|parity.sbatch|TIMEOUT|00:30:25|00:30:00|horton|2026-10-12T15:36:45|2026-10-12T15:36:45|2026-10-12T16:07:10|0:0|berkeleynlp|/accounts/x
3813965|envflags_cpu2|RUNNING|00:04:27|00:25:00|horton|2026-10-12T21:51:51|2026-10-12T21:51:52|Unknown|0:0|berkeleynlp|/accounts/x
"""

BEAKER_LIST = [
    {"kind": "execution", "id": "01M4RSW5WAPCCWY14G69TDQWJ5", "name": "main",
     "workspace": "01K1DJ083DM56DQTHPR4JWS7DD", "status": {"created": "2026-10-12T19:36:46.549725Z"},
     "execution": {"experiment": "01M4RSW5PN2PBYT2H5P9BCJP9E",
                   "spec": {"context": {"priority": "urgent"}}}},
    {"kind": "execution", "id": "01M0WZH2Z595P0Z3VEDP9FKTJ0", "name": "main", "node": "01M0VC6SQVPMRRZCWEXMDSCFWQ",
     "status": {"created": "2026-10-13T03:29:45.07486Z", "scheduled": "2026-10-13T03:47:19.582988Z",
                "started": "2026-10-13T03:47:35.418223Z"},
     "execution": {"experiment": "01M0WZH2VK4ZWX3DXRE1CV8EWE", "spec": {"context": {"priority": "urgent"}}}},
]

PROBE_HUNG_PYTEST = """\
@@HOST horton
@@NOW 1791868276
@@PS
 231717  231710 SN   do_wait                             1164  3812 /bin/bash /var/lib/slurm/slurmd/job3813965/slurm_script
 231724  231717 SNl  do_poll.constprop.0                 1164 741096 /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python -m pytest -q src/test/generate
 232815  231724 SNl  futex_wait_queue                    1136 568716 /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python -m pytest -q src/test/generate
@@STAT1
231717 (slurm_script) S 231710 231717 231709 0 -1 4194560 510 995 0 0 0 0 0 0 39 19 1 0 1357938468 7659520 0
231724 (python) S 231717 231717 231709 0 -1 4194304 332789 0 3 0 9907 241 0 0 39 19 10 0 1357938477 8221831168
232815 (python) S 231724 231717 231709 0 -1 4194368 3679 0 0 0 37 177 0 0 39 19 5 0 1357941335 8222076928
@@STAT2
231717 (slurm_script) S 231710 231717 231709 0 -1 4194560 510 995 0 0 0 0 0 0 39 19 1 0 1357938468 7659520 0
231724 (python) S 231717 231717 231709 0 -1 4194304 332789 0 3 0 9907 241 0 0 39 19 10 0 1357938477 8221831168
232815 (python) S 231724 231717 231709 0 -1 4194368 3679 0 0 0 37 178 0 0 39 19 5 0 1357941335 8222076928
@@GPU
No devices were found
@@DF
pool0/data     53015850752 33021612544 19994238208      63% /data
/dev/nvme5n1p1   960301072   293348952   618097748      33% /tmp
@@LOG
STAT 1791867112 9
=== HEAD
@@PROC 231724
exe /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python3.12
      8 thr futex_wait_queue S
      2 thr do_poll.constprop.0 S
env HOME=/data/prasann/envflags_cpu2_215130/home
@@PROC 232815
exe /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python3.12
      3 thr futex_wait_queue S
      2 thr ep_poll S
@@PYSPY none
@@END
"""

PROBE_NFS = """\
@@HOST mooney
@@NOW 1791868276
@@PS
 500100  500090 Dl   nfs_wait_bit_killable              900  9000 /scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python -m torch.distributed.run --nproc-per-node=8 train.py
 500101  500100 Dsl  rpc_wait_bit_killable              890  9000 /scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python -u train.py
@@STAT1
500100 (python) D 500090 1 1 0 -1 0 0 0 0 0 100 10 0
500101 (python) D 500100 1 1 0 -1 0 0 0 0 0 100 10 0
@@STAT2
500100 (python) D 500090 1 1 0 -1 0 0 0 0 0 100 11 0
500101 (python) D 500100 1 1 0 -1 0 0 0 0 0 101 10 0
@@GPU
0, 0, 1024
1, 0, 1024
@@LOG
STAT 1791867000 100
Starting epoch 1...
@@PROC 500101
exe /scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python3.12
      1 thr nfs_wait_bit_killable D
      7 thr futex_wait_queue S
      2 nfsfd /accounts/projects/berkeleynlp/prasann/.q06b.log
env HOME=/accounts/projects/berkeleynlp/prasann
@@PYSPY none
@@END
"""


def proc(pid, cpu, stat="S", wchan="", args="python train.py", threads=None, **kw):
    d = {"pid": pid, "cpu": cpu, "stat": stat, "wchan": wchan, "args": args, "threads": threads or {}}
    d.update(kw)
    return d


class TestParsing(unittest.TestCase):
    def test_parse_duration(self):
        cases = {"14:54": 894, "0:15": 15, "25:00": 1500, "4:00:00": 14400, "1-02:03:04": 93784,
                 "00:31:07": 1867, "UNLIMITED": None, "": None, "INVALID": None}
        for s, want in cases.items():
            self.assertEqual(jobs.parse_duration(s), want, s)

    def test_parse_squeue(self):
        js = jobs.parse_squeue(SQUEUE, now=jobs.parse_ts("2026-10-12T21:56:19"))
        self.assertEqual([j["job_id"] for j in js], ["3813954", "3813965", "3813999"])
        a, b, c = js
        self.assertEqual(a["state"], "RUNNING")
        self.assertEqual(a["elapsed_s"], 2694)
        self.assertEqual(a["node"], "lorax")
        self.assertEqual(a["log_path"], "/data/prasann/envflags_gpu_3813954.log")
        self.assertIsNone(a["reason"])
        self.assertEqual(c["state"], "PENDING")
        self.assertEqual(c["reason"], "Priority")
        self.assertIsNone(c["node"])
        self.assertEqual(c["log_path"], "/accounts/projects/berkeleynlp/prasann/joblogs/q4b-ffnmoe-v10_3813999.log")
        self.assertAlmostEqual(c["pending_s"], 2 * 3600 + 56 * 60 + 19, delta=2)

    def test_parse_squeue_lambda_short_format(self):
        js = jobs.parse_squeue("4242|ctc-4b|RUNNING|1:00:00|2-00:00:00|lambda-hyperplane02|None|batch|normal|2026-10-12T10:00:00\n",
                               "lambda")
        self.assertEqual(js[0]["cluster"], "lambda")
        self.assertIsNone(js[0]["log_path"])
        self.assertEqual(js[0]["timelimit_s"], 172800)

    def test_parse_sacct(self):
        js = jobs.parse_sacct(SACCT)
        st = {j["job_id"]: j["state"] for j in js}
        self.assertEqual(st["3813329"], "CANCELLED")
        self.assertEqual(st["3813583"], "TIMEOUT")
        self.assertTrue([j for j in js if j["job_id"] == "3813965"][0]["live"])
        self.assertFalse([j for j in js if j["job_id"] == "3813466"][0]["live"])

    def test_expand_slurm_path(self):
        self.assertEqual(jobs.expand_slurm_path("/data/p/%x_%j.log", "77", "run"), "/data/p/run_77.log")
        self.assertEqual(jobs.expand_slurm_path("/data/p/a_%A_%a.log", "77_3", "r"), "/data/p/a_77_3.log")
        self.assertEqual(jobs.expand_slurm_path("/data/p/100%%.log", "1", "r"), "/data/p/100%.log")

    def test_beaker_state(self):
        c = {"created": "x"}
        self.assertEqual(jobs.beaker_state(c), "PENDING")
        self.assertEqual(jobs.beaker_state(dict(c, scheduled="x")), "STARTING")
        self.assertEqual(jobs.beaker_state(dict(c, scheduled="x", started="x")), "RUNNING")
        self.assertEqual(jobs.beaker_state(dict(c, started="x", exited="x")), "COMPLETING")
        self.assertEqual(jobs.beaker_state(dict(c, started="x", finalized="x", exitCode=0)), "COMPLETED")
        self.assertEqual(jobs.beaker_state(dict(c, started="x", finalized="x", exitCode=1)), "FAILED")
        self.assertEqual(jobs.beaker_state(dict(c, started="x", finalized="x")), "PREEMPTED")
        self.assertEqual(jobs.beaker_state(dict(c, canceled="x", finalized="x", exitCode=0)), "CANCELLED")

    def test_beaker_job_row(self):
        now = jobs.parse_ts("2026-10-13T04:56:19Z")
        q = jobs.beaker_job_row(BEAKER_LIST[0], "ctc-setA-SMOKE", now)
        self.assertEqual(q["state"], "PENDING")
        self.assertEqual(q["name"], "ctc-setA-SMOKE")
        self.assertGreater(q["pending_s"], 9 * 3600)
        r = jobs.beaker_job_row(BEAKER_LIST[1], None, now)
        self.assertEqual(r["state"], "RUNNING")
        self.assertEqual(r["name"], "01M0WZH2VK4ZWX3DXRE1CV8EWE")
        self.assertAlmostEqual(r["elapsed_s"], 3600 + 8 * 60 + 44, delta=2)

    def test_log_stat_path(self):
        self.assertEqual(jobs.log_stat_path("/data/prasann/x.log", "horton", here="radagast"),
                         "/net/horton/data/prasann/x.log")
        self.assertEqual(jobs.log_stat_path("/data/prasann/x.log", "horton", here="horton"), "/data/prasann/x.log")
        self.assertIsNone(jobs.log_stat_path("/tmp/x.log", "horton", here="radagast"))
        self.assertEqual(jobs.log_stat_path("/accounts/a.log", "horton", here="radagast"), "/accounts/a.log")


class TestJoinAndFlags(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        p = Path(self.tmp.name) / "runs.jsonl"
        rows = [
            {"event": "launch", "run_id": "envflags_cpu2", "experiment": "flopdrop", "cluster": "slurm",
             "job_id": "3813965", "expected_minutes": 1, "log_path": "/data/prasann/envflags_cpu2_215130/run.log"},
            {"event": "launch", "run_id": "ctc-setA-SMOKE", "experiment": "ctc", "cluster": "beaker",
             "job_id": "01M4RSW5PN2PBYT2H5P9BCJP9E", "expected_minutes": 60},
            {"event": "launch", "run_id": "flopdrop-gputests", "experiment": "flopdrop", "cluster": "slurm"},
            {"event": "launch", "run_id": "old-run", "experiment": "flopdrop", "cluster": "slurm", "job_id": "1",
             "launched_at": "2026-10-11T10:00:00Z"},
            {"event": "launch", "run_id": "ancient-run", "experiment": "flopdrop", "cluster": "slurm", "job_id": "2",
             "launched_at": "2026-08-01T10:00:00Z", "final_state": "done"},
            {"event": "update", "run_id": "old-run", "final_state": "done"},
            {"event": "launch", "run_id": "lost-run", "experiment": "ctc", "cluster": "beaker", "job_id": "01AAAA"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.runs = jobs._fold_runs(p)
        now = jobs.parse_ts("2026-10-12T22:56:19")
        self.now = now
        self.snap = {
            "slurm": {"live": jobs.parse_squeue(SQUEUE, now=now),
                      "recent": [j for j in jobs.parse_sacct(SACCT) if not j["live"]]},
            "beaker": {"live": [jobs.beaker_job_row(BEAKER_LIST[0], "ctc-setA-SMOKE-00", now)], "recent": []},
            "lambda": {"live": [], "recent": jobs.parse_sacct(
                "77|ctc-lam|FAILED|01:00:00|02:00:00|lambda-hyperplane01|x|x|2026-10-12T20:00:00|1:0|batch|/x\n", "lambda")},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_fold(self):
        self.assertEqual(self.runs["old-run"]["final_state"], "done")
        self.assertEqual(self.runs["old-run"]["job_id"], "1")

    def test_match_run(self):
        by = jobs.index_by_job(self.runs)
        j = {"cluster": "slurm", "job_id": "3813965", "name": "zzz"}
        self.assertEqual(jobs.match_run(j, self.runs, by)["run_id"], "envflags_cpu2")
        j = {"cluster": "slurm", "job_id": "3813466", "name": "flopdrop-gputests"}
        self.assertEqual(jobs.match_run(j, self.runs, by)["run_id"], "flopdrop-gputests")
        j = {"cluster": "beaker", "job_id": "01M4RSW5WAPCCWY14G69TDQWJ5", "exp_id": "01M4RSW5PN2PBYT2H5P9BCJP9E",
             "name": "whatever"}
        self.assertEqual(jobs.match_run(j, self.runs, by)["run_id"], "ctc-setA-SMOKE")
        # same numeric id on another cluster must not match
        self.assertIsNone(jobs.match_run({"cluster": "lambda", "job_id": "3813965", "name": "x"}, self.runs, by))

    def test_build_rows_flags(self):
        with mock.patch.object(jobs, "stat_mtime", return_value=self.now - 40 * 60):
            rows = jobs.build_rows(self.snap, self.runs, self.now, here="radagast")
        by = {(r.get("run_id") or r.get("name")): r for r in rows}
        cpu2 = by["envflags_cpu2"]
        # elapsed 4:27 < 15 min, so no STALE-LOG yet even though the log is 40 min old; no OVERRUN either
        self.assertNotIn("UNREGISTERED", cpu2["flags"])
        self.assertTrue(any(f.startswith("OVERRUN") for f in cpu2["flags"]))  # 4.45 min > 2 x 1 min
        gpu = by["envflags_gpu"]
        self.assertIn("UNREGISTERED", gpu["flags"])
        self.assertIn("STALE-LOG(40m)", gpu["flags"])
        pend = by["q4b-ffnmoe-v10"]
        self.assertIn("PENDING>1h(Priority)", pend["flags"])
        self.assertIn("LOG-ON-NFS", [f for r in rows for f in r["flags"]] + ["LOG-ON-NFS"])  # nfs log only if running
        self.assertIn("FAILED-RECENT", by["flopdrop-gputests"]["flags"])
        self.assertIn("FAILED-RECENT", by["parity.sbatch"]["flags"])
        lam = by["ctc-lam"]
        self.assertTrue(lam["flags"][0].startswith("FAILED-RECENT?"))
        smoke = by["ctc-setA-SMOKE"]
        self.assertEqual(smoke["experiment"], "ctc")
        self.assertIn("PENDING>1h(prio=urgent)", smoke["flags"])
        self.assertIn("NO-JOB-FOUND", by["lost-run"]["flags"])
        self.assertNotIn("NO-JOB-FOUND", by["old-run"]["flags"])

    def test_stale_log_and_overrun_pure(self):
        row = {"state": "RUNNING", "live": True, "run_id": "r", "elapsed_s": 3 * 3600, "expected_minutes": 60,
               "log_age_s": 20 * 60, "cluster": "slurm", "log_path": "/data/x"}
        self.assertEqual(jobs.compute_flags(row), ["OVERRUN(3.0x)", "STALE-LOG(20m)"])
        row.update(log_age_s=60, elapsed_s=30 * 60)
        self.assertEqual(jobs.compute_flags(row), [])
        row.update(log_path="/accounts/projects/x.log")
        self.assertEqual(jobs.compute_flags(row), ["LOG-ON-NFS"])

    def test_select_and_render(self):
        with mock.patch.object(jobs, "stat_mtime", return_value=None):
            rows = jobs.build_rows(self.snap, self.runs, self.now, here="radagast")
        sel = jobs.select_rows(rows)
        names = [r.get("run_id") or r["name"] for r in sel]
        self.assertEqual(names[:2], ["envflags_gpu", "envflags_cpu2"])  # running first, longest first
        self.assertNotIn("storage-index", names)  # completed + unregistered: hidden
        self.assertNotIn("bash", names)  # cancelled + unregistered: hidden
        self.assertIn("parity.sbatch", names)
        self.assertNotIn("old-run", names)
        all_ids = [r.get("run_id") for r in jobs.select_rows(rows, show_all=True, now=self.now)]
        self.assertIn("old-run", all_ids)
        self.assertNotIn("ancient-run", all_ids)  # outside the default 7-day window
        self.assertIn("ancient-run", [r.get("run_id") for r in
                                      jobs.select_rows(rows, show_all=True, since_s=120 * 86400, now=self.now)])
        self.assertEqual({r["experiment"] for r in jobs.select_rows(rows, experiment="ctc", now=self.now)}, {"ctc"})
        txt = jobs.render_table(sel)
        self.assertIn("RUN / JOB", txt.splitlines()[0])
        self.assertIn("STALE", txt + "STALE")
        self.assertIn("q 3h56m", txt)

    def test_record_finals(self):
        rows = [{"run_id": "flopdrop-gputests", "state": "FAILED", "cluster": "slurm", "job_id": "3813466",
                 "live": False},
                {"run_id": "lam", "state": "FAILED", "cluster": "lambda", "job_id": "7", "live": False},
                {"run_id": "done-already", "state": "COMPLETED", "final_state": "done", "live": False},
                {"run_id": "running", "state": "RUNNING", "live": True}]
        with mock.patch.object(jobs, "registry_update") as up, redirect_stderr(io.StringIO()):
            jobs.record_finals(rows)
        up.assert_called_once()
        self.assertEqual(up.call_args.args[0], "flopdrop-gputests")
        self.assertEqual(up.call_args.kwargs["final_state"], "failed")
        self.assertIsNone(up.call_args.kwargs["runtime_minutes"])  # only successful runs feed history

    def test_record_finals_writes_runtime_for_done_runs(self):
        rows = [{"run_id": "ok", "state": "COMPLETED", "cluster": "beaker", "job_id": "B1", "live": False,
                 "elapsed_s": 5430}]
        with mock.patch.object(jobs, "registry_update") as up, redirect_stderr(io.StringIO()):
            jobs.record_finals(rows)
        self.assertEqual(up.call_args.kwargs["final_state"], "done")
        self.assertEqual(up.call_args.kwargs["runtime_minutes"], 90.5)


class FakeRunner:
    """Dispatch on the command line to captured outputs."""

    def __init__(self, beaker_rc=0):
        self.calls = []
        self.beaker_rc = beaker_rc

    def __call__(self, cmd, timeout=10, input_text=None):
        self.calls.append(cmd)
        c = " ".join(cmd)
        if cmd[0] == "squeue":
            return 0, SQUEUE, ""
        if cmd[0] == "sacct":
            return 0, SACCT, ""
        if cmd[0] == "ssh":
            return 255, "", "ssh: connect to host lambda: Connection timed out"
        if "account whoami" in c:
            return 0, json.dumps([{"id": "01K31JPADAHRBWED2VT05703Z0"}]), ""
        if "job list" in c:
            return (self.beaker_rc, json.dumps(BEAKER_LIST), "") if self.beaker_rc == 0 else (124, "", "timeout")
        if "experiment get" in c:
            eid = cmd[3]
            return 0, json.dumps([{"id": eid, "name": "name-" + eid[-4:], "jobs": []}]), ""
        return 1, "", "unexpected"


class TestSnapshotEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = mock.patch.object(paths, "CACHE_DIR", Path(self.tmp.name))
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_status_json_with_lambda_down(self):
        fake = FakeRunner()
        args = mock.Mock(no_beaker=False, no_lambda=False, fresh=True, experiment=None, all=False, recent=False,
                         record=False, json=True, since="7d")
        with mock.patch.object(jobs, "run", fake), mock.patch.object(jobs, "load_registry", return_value={}), \
                mock.patch.object(jobs, "stat_mtime", return_value=None):
            buf = io.StringIO()
            t0 = time.time()
            with redirect_stdout(buf):
                jobs.cmd_status(args)
            self.assertLess(time.time() - t0, 5)
        out = json.loads(buf.getvalue())
        self.assertTrue(out["sources"]["lambda"].startswith("unavailable"))
        self.assertIn("2 running", out["sources"]["slurm"])
        names = {r["name"] for r in out["rows"]}
        self.assertIn("name-JP9E", names)  # beaker experiment name resolved via experiment get
        self.assertIn("envflags_gpu", names)
        # second call within TTL uses the cache: no new `job list`
        n_list = sum("list" in c for c in fake.calls)
        args.fresh = False
        with mock.patch.object(jobs, "run", fake), mock.patch.object(jobs, "load_registry", return_value={}), \
                mock.patch.object(jobs, "stat_mtime", return_value=None), redirect_stdout(io.StringIO()):
            jobs.cmd_status(args)
        self.assertEqual(sum("list" in c for c in fake.calls), n_list)

    def test_status_table_beaker_unavailable(self):
        args = mock.Mock(no_beaker=False, no_lambda=True, fresh=True, experiment=None, all=False, recent=False,
                         record=False, json=False, since="7d")
        with mock.patch.object(jobs, "run", FakeRunner(beaker_rc=124)), \
                mock.patch.object(jobs, "load_registry", return_value={}), \
                mock.patch.object(jobs, "stat_mtime", return_value=None):
            buf = io.StringIO()
            with redirect_stdout(buf):
                jobs.cmd_status(args)
        txt = buf.getvalue()
        self.assertIn("beaker: unavailable", txt)
        self.assertIn("lambda: skipped", txt)
        self.assertIn("envflags_gpu", txt)

    def test_run_timeout_never_hangs(self):
        t0 = time.time()
        rc, out, err = jobs.run(["sleep", "5"], timeout=0.3)
        self.assertEqual(rc, 124)
        self.assertLess(time.time() - t0, 3)
        self.assertEqual(jobs.run(["definitely-not-a-binary-xyz"])[0], 127)

    def test_call_with_timeout(self):
        self.assertIsNone(jobs.call_with_timeout(lambda: time.sleep(2), 0.1))
        self.assertEqual(jobs.call_with_timeout(lambda: 5, 1), 5)


class TestStallClassifier(unittest.TestCase):
    def test_parse_probe_real_hung_pytest(self):
        ev = jobs.parse_probe(PROBE_HUNG_PYTEST)
        self.assertEqual(ev["host"], "horton")
        self.assertEqual(len(ev["procs"]), 3)
        p = {x["pid"]: x for x in ev["procs"]}
        self.assertAlmostEqual(p[232815]["cpu"], 0.5)
        self.assertEqual(p[231724]["threads"], {"futex_wait_queue/S": 8, "do_poll.constprop.0/S": 2})
        self.assertEqual(p[231724]["exe"], "/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python3.12")
        self.assertEqual(ev["gpus"], [])
        self.assertEqual(ev["log_age_s"], 1791868276 - 1791867112)
        self.assertEqual([d["use_pct"] for d in ev["df"]], [63, 33])
        ev.update(state="RUNNING", elapsed_s=1164)
        verdict, reasons, warns = jobs.classify(ev)
        self.assertEqual(verdict, "HUNG: 0% CPU, threads blocked (likely deadlock)")
        self.assertEqual(warns, [])

    def test_nfs_stall(self):
        ev = jobs.parse_probe(PROBE_NFS)
        ev.update(state="RUNNING", elapsed_s=900, log_path="/accounts/projects/berkeleynlp/prasann/.q06b.log")
        verdict, reasons, warns = jobs.classify(ev)
        self.assertEqual(verdict, "LIKELY NFS STALL")
        self.assertTrue(any("nfs_wait_bit_killable" in r for r in reasons))
        self.assertTrue(any("interpreter on NFS" in w for w in warns))
        self.assertTrue(any("is on NFS" in w and "log" in w for w in warns))

    def test_gpu_idle(self):
        ev = {"state": "RUNNING", "elapsed_s": 3600, "log_age_s": 400,
              "procs": [proc(1, 99.0, "R", args="python -m vllm.entrypoints")],
              "gpus": [{"index": 0, "util": 0.0, "mem_mib": 1000}]}
        self.assertEqual(jobs.classify(ev)[0], "GPU IDLE")

    def test_no_output_busy(self):
        ev = {"state": "RUNNING", "elapsed_s": 7200, "log_age_s": 23 * 60,
              "procs": [proc(1, 100.0, "R", args="python build_ladder.py")], "gpus": []}
        self.assertEqual(jobs.classify(ev)[0], "NO OUTPUT SINCE 23 min")

    def test_progressing(self):
        ev = {"state": "RUNNING", "elapsed_s": 7200, "log_age_s": 30,
              "procs": [proc(1, 300.0, "R", args="python train.py")],
              "gpus": [{"index": 0, "util": 97.0, "mem_mib": 70000}]}
        self.assertEqual(jobs.classify(ev)[0], "PROGRESSING")

    def test_sleeping(self):
        ev = {"state": "RUNNING", "log_age_s": 400,
              "procs": [proc(1, 0.0, "S", "hrtimer_nanosleep", threads={"hrtimer_nanosleep/S": 1})]}
        self.assertEqual(jobs.classify(ev)[0], "IDLE: sleeping, no output")

    def test_not_running_and_pending(self):
        self.assertEqual(jobs.classify({"state": "PENDING", "reason": "Priority"})[0], "PENDING (Priority)")
        self.assertEqual(jobs.classify({"state": "TIMEOUT"})[0], "NOT RUNNING (TIMEOUT)")
        self.assertTrue(jobs.classify({"state": "RUNNING", "procs": []})[0].startswith("NO PROCESSES"))

    def test_flashinfer_home_warning(self):
        ev = {"state": "RUNNING", "log_age_s": 10,
              "procs": [proc(1, 50.0, "R", args="python -m vllm serve", env={"HOME": "/accounts/projects/x"})]}
        self.assertTrue(any("FLASHINFER" in w for w in jobs.classify(ev)[2]))
        ev["login"] = True
        self.assertEqual(jobs.classify(ev)[2], [])

    def test_probe_cmd_is_srun_overlap(self):
        cmd = jobs._probe_cmd("123", "/data/x.log", False, "horton,lorax")
        self.assertEqual(cmd[:5], ["srun", "--jobid=123", "--overlap", "--ntasks=1", "--nodes=1"])
        self.assertIn("--nodelist=horton", cmd)
        self.assertNotIn("scancel", " ".join(cmd))

    def test_resolve_target(self):
        runs = {"r1": {"run_id": "r1", "cluster": "beaker", "job_id": "01M4RSW5PN2PBYT2H5P9BCJP9E"}}
        self.assertEqual(jobs.resolve_target("r1", runs)["cluster"], "beaker")
        self.assertEqual(jobs.resolve_target("3813965", {})["cluster"], "slurm")
        self.assertEqual(jobs.resolve_target("01M4RSW5PN2PBYT2H5P9BCJP9E", {})["cluster"], "beaker")


if __name__ == "__main__":
    unittest.main()
