"""Tests for opslib.capacity (fixture text only; no slurm/ssh calls)."""
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from opslib import capacity

BERKELEY = """\
cubbins|jsteinhardt|gpu:H200:8|mixed
mooney|jsteinhardt|gpu:H200:8(S:0-1)|mixed
shadowfax|jsteinhardt|gpu:RTX2080:8|idle
lorax|berkeleynlp|gpu:H200:7|down$
__SQ__
101|prasann|jsteinhardt|preemptive_high|gres/gpu:2|cubbins|1|1:00:00|f8k-arm
102|alice|jsteinhardt|normal|gres/gpu:1|cubbins|1|5:00|young
103|alice|jsteinhardt|normal|gres:gpu:H200:2|cubbins|1|2:00:00|old
104|rhys|jsteinhardt|preemptive_high|N/A|mooney|1|3:00:00|big
105|bob|jsteinhardt|preemptive|gres/gpu:1|mooney|1|10:00|small
106|prasann|jsteinhardt|preemptive_high|gres/gpu:1|mooney|1|20:00|probe
107|carol|jsteinhardt|normal|N/A|cubbins|1|1:00|cpu-only
__TRES__
101 cpu=16,mem=64G,node=1,billing=24,gres/gpu=2
104 cpu=32,mem=200G,node=1,billing=48,gres/gpu=6
107 cpu=4,mem=8G,node=1,billing=4
__QOS__
normal|preemptible
preemptive|normal
preemptive_high|normal,preemptive
"""


class ParseTest(unittest.TestCase):
    def setUp(self):
        self.snap = capacity.parse(BERKELEY)

    def test_nodes_and_types(self):
        n = self.snap["nodes"]
        self.assertEqual((n["mooney"]["gpus"], n["mooney"]["gpu_type"]), (8, "H200"))
        self.assertFalse(n["lorax"]["available"])
        self.assertEqual(n["shadowfax"]["gpu_type"], "RTX2080")

    def test_jobs_gpu_counts_incl_tres_fallback_and_cpu_jobs_dropped(self):
        g = {j["job_id"]: j["gpus_per_node"] for j in self.snap["jobs"]}
        self.assertEqual(g, {"101": 2, "102": 1, "103": 2, "104": 6, "105": 1, "106": 1})

    def test_qos_table(self):
        self.assertEqual(self.snap["preempts"]["preemptive_high"], {"normal", "preemptive"})
        self.assertEqual(capacity.can_preempt({}, "preemptive_high_sewonm"), {"normal", "preemptive"})

    def test_node_view(self):
        v = {r["node"]: r for r in capacity.node_view(self.snap, me="prasann")}
        self.assertEqual((v["cubbins"]["used"], v["cubbins"]["free"]), (5, 3))
        self.assertEqual(dict(v["cubbins"]["ours"]), {"preemptive_high": 2})
        self.assertEqual(dict(v["mooney"]["others"]), {("rhys", "preemptive_high"): 6, ("bob", "preemptive"): 1})
        self.assertEqual(v["mooney"]["free"], 0)
        self.assertEqual(v["lorax"]["free"], 0)

    def test_expand_nodelist(self):
        self.assertEqual(capacity.expand_nodelist("lambda-hyperplane[01-02,05]"),
                         ["lambda-hyperplane01", "lambda-hyperplane02", "lambda-hyperplane05"])
        self.assertEqual(capacity.expand_nodelist("cubbins,mooney"), ["cubbins", "mooney"])

    def test_cap_use_by_experiment(self):
        caps = capacity.our_cap_use(self.snap, "berkeley", {"101": {"experiment": "tokdrop_lr"}}, me="prasann")
        self.assertEqual(caps["preemptive_high"], {"gpus": 3, "cap": 8,
                                                   "by_experiment": {"tokdrop_lr": 2, "unregistered": 1}})

    def test_berkeley_preemptive_shown_uncapped(self):
        # jsteinhardt: 8 at preemptive_high only; preemptive runs beyond it when capacity is free (2026-11-10)
        snap = capacity.parse(BERKELEY.replace("105|bob|", "105|prasann|"))
        caps = capacity.our_cap_use(snap, "berkeley", {}, me="prasann")
        self.assertEqual(caps["preemptive"], {"gpus": 1, "cap": None, "by_experiment": {"unregistered": 1}})
        text = capacity.render("berkeley", snap, caps)
        self.assertIn("preemptive_high 3/8", text)
        self.assertIn("preemptive 1 (no cap", text)
        args = SimpleNamespace(gpus=4, qos="preemptive", partition="jsteinhardt")
        wp = capacity.would_preempt(snap, 4, "preemptive", partition="jsteinhardt", me="prasann")
        self.assertNotIn("OVER CAP", capacity.render_preempt("berkeley", args, caps, wp))


class WouldPreemptTest(unittest.TestCase):
    def setUp(self):
        self.snap = capacity.parse(BERKELEY)

    def test_fits_free_preempts_nobody(self):
        wp = capacity.would_preempt(self.snap, 3, "preemptive_high", partition="jsteinhardt", me="prasann")
        self.assertEqual((wp["fits_free"], wp["node"], wp["victims"]), (True, "cubbins", []))

    def test_rtx_nodes_skipped_unless_asked(self):
        wp = capacity.would_preempt(self.snap, 8, "preemptive_high", me="prasann")
        self.assertTrue(wp.get("pending") or wp["node"] != "shadowfax")
        wp = capacity.would_preempt(self.snap, 8, "preemptive_high", me="prasann", any_gpu=True)
        self.assertEqual((wp["fits_free"], wp["node"]), (True, "shadowfax"))

    def test_victims_youngest_first_and_lower_qos_only(self):
        wp = capacity.would_preempt(self.snap, 4, "preemptive_high", partition="jsteinhardt", me="prasann")
        self.assertFalse(wp["fits_free"])
        self.assertEqual(wp["node"], "cubbins")
        self.assertEqual([v["job_id"] for v in wp["victims"]], ["102"])  # 3 free + the 5-min job
        self.assertGreater(wp["others_gpu_h_lost"], 0)

    def test_same_qos_is_not_preemptable(self):
        wp = capacity.would_preempt(self.snap, 2, "preemptive_high", node="mooney", me="prasann")
        # mooney: 0 free; only bob's preemptive job (1 GPU) is below preemptive_high -> cannot fit 2
        self.assertTrue(wp.get("pending"))


class CliTest(unittest.TestCase):
    def test_exit_3_when_another_user_would_be_preempted(self):
        args = SimpleNamespace(cluster="berkeley", would_preempt=True, gpus=4, qos="preemptive_high",
                               partition="jsteinhardt", node=None, any_gpu=False, json=False)
        with mock.patch.object(capacity, "fetch", return_value=BERKELEY), \
                mock.patch.object(capacity, "ME", "prasann"), \
                mock.patch.object(capacity, "_runs_by_job", return_value={}), redirect_stdout(io.StringIO()) as out:
            rc = capacity.cmd_capacity(args)
        self.assertEqual(rc, 3)
        self.assertIn("WOULD PREEMPT", out.getvalue())
        self.assertIn("alice", out.getvalue())

    def test_exit_0_when_it_fits(self):
        args = SimpleNamespace(cluster="berkeley", would_preempt=True, gpus=1, qos="preemptive_high",
                               partition="jsteinhardt", node=None, any_gpu=False, json=False)
        with mock.patch.object(capacity, "fetch", return_value=BERKELEY), \
                mock.patch.object(capacity, "_runs_by_job", return_value={}), redirect_stdout(io.StringIO()):
            self.assertEqual(capacity.cmd_capacity(args), 0)


if __name__ == "__main__":
    unittest.main()
