"""Tests for opslib.runtime (minRuntime rule + history median) and safe.suggest_min_runtime."""
import os
import tempfile
import unittest
from pathlib import Path

from opslib import registry, runtime, safe


class MinRuntimeRuleTest(unittest.TestCase):
    def test_fallback_when_unknown(self):
        for x in (None, "", "abc", 0, -5):
            self.assertEqual(runtime.min_runtime_minutes(x), 60)

    def test_floor_scale_and_cap(self):
        self.assertEqual(runtime.min_runtime_minutes(20), 60)  # 26 -> floor 1 h
        self.assertEqual(runtime.min_runtime_minutes(60), 78)  # 1.3 x
        self.assertEqual(runtime.min_runtime_minutes(100), 130)
        self.assertEqual(runtime.min_runtime_minutes(185), 240)  # 240.5 -> cap
        self.assertEqual(runtime.min_runtime_minutes(600), 240)

    def test_duration_strings(self):
        self.assertEqual(runtime.beaker_duration(60), "1h")
        self.assertEqual(runtime.beaker_duration(78), "1h18m")
        self.assertEqual(runtime.beaker_duration(240), "4h")
        self.assertEqual(runtime.beaker_duration(45), "45m")

    def test_resume_only_past_cap(self):
        self.assertFalse(runtime.needs_resume_point(240))
        self.assertTrue(runtime.needs_resume_point(241))
        self.assertFalse(runtime.needs_resume_point(None))


class HistoryTest(unittest.TestCase):
    def runs(self):
        def r(rid, rt, kind="train", fs="done", cmd="python beaker_ctc_suite.py --run-name x"):
            return rid, {"run_id": rid, "kind": kind, "final_state": fs, "runtime_minutes": rt, "command": cmd}

        return dict([
            r("f8k-mt4-dense-s1464-tde-lr5e-06", 100), r("f8k-mt4-dense-s1464-tde-lr1e-05", 110),
            r("f8k-mt4-dense-s1464-tde-lr2e-06-6b6ea8e1", 120),
            r("f8k-mt4-dense-s488-tde-lr5e-06", 30), r("f8k-mt4-dense-s488-tde-lr1e-05", 35),
            r("f8k-mt4-dense-s1464-tde-lr3e-06", 999, fs="failed"),  # not done: ignored
            r("f8k-mt4-dense-s1464-tde-lr4e-06", 999, kind="eval"),  # other kind: ignored
            r("f8k-mt4-dense-s1464-tde-lr6e-06", 999, cmd="launch_otr.sh"),  # other launcher: ignored
        ])

    def test_longest_prefix_with_enough_samples(self):
        got = runtime.history_minutes("f8k-mt4-dense-s1464-tde-lr7e-06", self.runs(), launcher="beaker_ctc_suite.py")
        self.assertEqual(got, (110, 3, "f8k-mt4-dense-s1464-tde"))

    def test_falls_back_to_shorter_prefix(self):
        med, n, prefix = runtime.history_minutes("f8k-mt4-dense-s976-tde-lr5e-06", self.runs(),
                                                 launcher="beaker_ctc_suite.py")
        self.assertEqual((med, n, prefix), (100, 5, "f8k-mt4-dense"))

    def test_no_match_on_first_token_alone(self):
        self.assertIsNone(runtime.history_minutes("f8k-other-arm", self.runs())[0])

    def test_suggest_sources(self):
        s = runtime.suggest(run_id="f8k-mt4-dense-s1464-tde-lr7e-06", runs=self.runs(), launcher="beaker_ctc_suite.py")
        self.assertEqual((s["min_runtime"], s["resume"]), ("2h23m", False))  # ceil(1.3 x 110) = 143
        self.assertTrue(s["source"].startswith("history:"))
        s = runtime.suggest(run_id="x", expected_minutes=300)
        self.assertEqual((s["min_runtime"], s["resume"], s["source"]), ("4h", True, "explicit"))
        s = runtime.suggest(run_id="nothing-like-it", runs=self.runs())
        self.assertEqual((s["min_runtime"], s["source"][:8]), ("1h", "fallback"))


class SafeAndRegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._env = os.environ.get("OPS_STATE_DIR")
        os.environ["OPS_STATE_DIR"] = self.tmp.name

    def tearDown(self):
        if self._env is None:
            os.environ.pop("OPS_STATE_DIR", None)
        else:
            os.environ["OPS_STATE_DIR"] = self._env
        self.tmp.cleanup()

    def test_runtime_minutes_is_a_numeric_registry_field(self):
        for i, rt in enumerate((50, 60, 70)):
            rid = f"exp-arm-s100-run{i}"
            registry.register(run_id=rid, experiment="e", cluster="beaker", kind="train")
            registry.update(rid, final_state="done", runtime_minutes=rt)
        with self.assertRaises(ValueError):
            registry.update("exp-arm-s100-run0", runtime_minutes="slow")
        self.assertEqual(registry.coerce("runtime_minutes", "12.5"), 12.5)
        s = safe.suggest_min_runtime(run_id="exp-arm-s100-run9")
        self.assertEqual((s["expected_minutes"], s["min_runtime"]), (60, "1h18m"))

    def test_safe_never_raises(self):
        os.environ["OPS_STATE_DIR"] = str(Path(self.tmp.name) / "missing" / "deeper")
        self.assertEqual(safe.suggest_min_runtime(run_id="anything")["min_runtime"], "1h")


if __name__ == "__main__":
    unittest.main()
