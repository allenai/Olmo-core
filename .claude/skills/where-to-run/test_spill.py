"""Spill rule in compute_avail.recommend (stdlib; no cluster calls).   python3 test_spill.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_avail as CA  # noqa: E402


def res(jst_status="preempt", jst_used=2, beaker_eta=3600, gpu_type="H200"):
    jst = dict(partition="jsteinhardt", qos="preemptive_high", gpus=2, gpu_type=gpu_type, status=jst_status,
               eta_s=60, node="cubbins", how="preempt normal-QOS jobs", notes=[], preempt_risk="none")
    return {
        "slurm": {"options": [jst]},
        "beaker": [{"cluster": "ai2/jupiter", "gpu_type": "H100", "options": [
            dict(gpus=2, verdict=(("free" if beaker_eta < 300 else "wait"), beaker_eta), urgent_ahead=40,
                 recent_wait_p50_s=beaker_eta, recent_wait_p90_s=beaker_eta * 2, slots_free=0)]}],
        "cap_use": {"jst": {"gpus": jst_used, "cap": 8, "by_experiment": {}}},
    }


class SpillTest(unittest.TestCase):
    def setUp(self):
        self._g = (CA.REQ_SIZES, CA.RUN_HOURS)
        CA.REQ_SIZES, CA.RUN_HOURS = [2], 1.5

    def tearDown(self):
        CA.REQ_SIZES, CA.RUN_HOURS = self._g

    def beaker_pick(self, eta):
        return dict(gpus=2, status="wait", eta_s=eta, done_s=eta + 6000, where="beaker ai2/jupiter (H100)",
                    why="", runners_up=[])

    def test_spills_when_beaker_waits_over_30_min(self):
        r = CA.spill(self.beaker_pick(3600), res())
        self.assertTrue(r["where"].startswith("jsteinhardt --qos=preemptive_high"))
        self.assertIn("SPILL RULE", r["why"])
        self.assertIn("2/8", r["why"])

    def test_no_spill_when_beaker_is_quick(self):
        self.assertIsNone(CA.spill(self.beaker_pick(600), res()))

    def test_no_spill_when_cap_full(self):
        self.assertIsNone(CA.spill(self.beaker_pick(3600), res(jst_used=7)))

    def test_no_spill_for_big_or_long_jobs(self):
        big = dict(self.beaker_pick(3600), gpus=4)
        self.assertIsNone(CA.spill(big, res()))
        CA.RUN_HOURS = 3
        self.assertIsNone(CA.spill(self.beaker_pick(3600), res()))

    def test_no_spill_to_a100_or_unfit(self):
        self.assertIsNone(CA.spill(self.beaker_pick(3600), res(gpu_type="A100")))
        self.assertIsNone(CA.spill(self.beaker_pick(3600), res(jst_status="wait")))

    def test_cap_line_format(self):
        self.assertEqual(CA.fmt_cap({"gpus": 5, "cap": 8, "by_experiment": {"tokdrop_lr": 4, "x": 1}}),
                         "5/8 [tokdrop_lr 4, x 1]")


if __name__ == "__main__":
    unittest.main()
