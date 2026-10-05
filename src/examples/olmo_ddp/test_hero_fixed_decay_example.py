"""CPU-only handoff checks: python -m unittest discover -s src/examples/olmo_ddp -p test_hero_fixed_decay_example.py."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import olmoe3_hero_fixed_decay_example as example


class FixedDecayExampleTest(unittest.TestCase):
    """Exercise budget rounding, source validation and interrupted-run selection."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(example, "CHILD_ROOT", self.root / "children")
        patcher.start()
        self.addCleanup(patcher.stop)

    def checkpoint(self, path, step, gpus=64, name="parent", decay=None):
        """Write synthetic structural metadata, without model tensors or pickle."""
        for directory in [path, path / "model_and_optim", path / "train", path / "resume_audit"]:
            directory.mkdir(parents=True, exist_ok=True)
        (path / ".metadata.json").write_text("{}")
        (path / "model_and_optim/.metadata").touch()
        scheduler = {"_CLASS_": "olmo_core.optim.scheduler.ConstantWithWarmup"}
        if decay is not None:
            scheduler = {"_CLASS_": "olmo_core.optim.scheduler.WSD", "decay": decay}
        config = {
            "run_name": name,
            "train_module": {"optim": {"lr": example.LR}, "scheduler": scheduler},
            "data_loader": {"global_batch_size": example.BATCH},
            "model": {"block": {"routed_experts_router": {"emo": {"enabled": True}}}},
        }
        (path / "config.json").write_text(json.dumps(config))
        for rank in range(gpus):
            row = dict(step=step, tokens=step * example.BATCH, gpus=gpus, rank=rank)
            (path / f"resume_audit/rank{rank}.json").write_text(json.dumps(row))
            (path / f"train/rank{rank}.pt").touch()
        return path

    def test_budget_units_and_rejections(self):
        for text in ["200B", "200b", "0.2T", "200000000000"]:
            self.assertEqual(example.token_count(text), 200_000_000_000)
        for text in ["0", "-1", "1.5", "nan", "inf", "200GB", ""]:
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                example.token_count(text)

    def test_actual_totals_and_boundary_beyond_14t(self):
        for step, gpus in [(238500, 64), (357500, 128), (834466, 128)]:
            with self.subTest(step=step):
                source = self.checkpoint(self.root / str(step), step, gpus)
                alias = self.root / f"alias-{step}"
                alias.symlink_to(source, target_is_directory=True)
                p = example.make_plan(alias, 200_000_000_000, "emo", f"child-{step}")
                self.assertEqual(p["source"], str(source))
                self.assertEqual(p["decay_steps"], 11921)
                self.assertEqual(p["end_step"], step + 11921)
                self.assertEqual(p["end_tokens"], (step + 11921) * example.BATCH)
                self.assertGreaterEqual(p["actual_decay_tokens"], p["requested_decay_tokens"])
                self.assertLess(
                    p["actual_decay_tokens"] - p["requested_decay_tokens"], example.BATCH
                )
                self.assertEqual(example.select_checkpoint(p), (step, str(source)))
                self.assertFalse(Path(p["save_folder"]).exists())

    def test_bad_source_and_reused_output(self):
        source = self.checkpoint(self.root / "source", 357500, 128)
        with self.assertRaisesRegex(ValueError, "--arm"):
            example.make_plan(source, 10, "non-emo", "child")
        with self.assertRaises(ValueError):
            example.make_plan(source, 10, "emo", "../parent")
        (example.CHILD_ROOT / "child").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "unused"):
            example.make_plan(source, 10, "emo", "child")
        (source / "train/rank127.pt").unlink()
        with self.assertRaisesRegex(ValueError, "rank 127"):
            example.make_plan(source, 10, "emo", "other-child")

    def test_resume_ignores_partial_save_and_rejects_foreign_child(self):
        source = self.checkpoint(self.root / "source", 357500)
        p = example.make_plan(source, 200_000_000_000, "emo", "child")
        root = Path(p["save_folder"])
        complete = self.checkpoint(root / "step357502", 357502, name="child", decay=11921)
        (root / "step358000-tmp").mkdir()
        (root / "step358000").mkdir()  # No completion marker.
        self.assertEqual(example.select_checkpoint(p), (357502, str(complete)))
        self.checkpoint(root / "step358001", 358001, name="different-child", decay=11921)
        with self.assertRaisesRegex(ValueError, "different child"):
            example.select_checkpoint(p)

    def test_decayed_source_is_rejected(self):
        source = self.checkpoint(self.root / "decayed", 240000, decay=24000)
        with self.assertRaisesRegex(ValueError, "non-decayed"):
            example.make_plan(source, 200_000_000_000, "emo", "child")


if __name__ == "__main__":
    unittest.main()
