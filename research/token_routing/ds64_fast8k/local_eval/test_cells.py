"""cells.py: plan / merge / rows on a fake work dir (stdlib; no GPU, no S3).   python3 test_cells.py"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cells  # noqa: E402


def fake_cell(work, task, rung, score, n=500, parse_rate=None, log_metric=None):
    d = cells.cell_dir(work, task, rung)
    os.makedirs(d, exist_ok=True)
    key = f"{task}_{rung}"
    meta = {"eval_run_id": "R-evslurm-1", "train_run": "R", "eval_size": {key: n}, "cot_label": "no-cot"}
    if parse_rate is not None:
        meta["parse_rate"] = {key: parse_rate}
    json.dump({"model_path": "/data/x", "variant": "dense", "ladder_version": "v2", "_meta": meta, key: score},
              open(cells.cell_json(work, task, rung), "w"))
    stem = f"{cells.TASK_KEY[task]}_multirung_{task}_{rung}_responses.jsonl"
    open(os.path.join(d, stem), "w").write('{"i": 0, "response": "x"}\n')
    metric = log_metric or ("score" if task == "oolong" else "f1")
    open(os.path.join(d, "cell.log"), "w").write(
        f"noise\n[ladder:{cells.TASK_KEY[task] if task == 'contradiction' else task}@{rung}] "
        f"{metric}={score:.3f} (n={n}, skipped_too_long=0)\n")


class CellsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.w = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_is_8k_first_and_skips_finished(self):
        fake_cell(self.w, "nq", "8k", 0.9)
        fake_cell(self.w, "outlier", "8k", 0.8, n=120)  # short cell = not finished
        p = cells.plan(self.w)
        self.assertEqual(p[:3], [("contradiction", "8k"), ("outlier", "8k"), ("oolong", "8k")])
        self.assertNotIn(("nq", "8k"), p)
        self.assertEqual(len(p), 7)
        self.assertTrue(all(r == "2k" for _, r in p[3:]))

    def test_unreadable_or_null_cell_is_not_finished(self):
        os.makedirs(cells.cell_dir(self.w, "nq", "2k"))
        open(cells.cell_json(self.w, "nq", "2k"), "w").write("{trunc")
        self.assertIsNone(cells.cell_result(self.w, "nq", "2k"))
        fake_cell(self.w, "nq", "8k", 0.0)
        d = json.load(open(cells.cell_json(self.w, "nq", "8k")))
        d["nq_8k"] = None
        json.dump(d, open(cells.cell_json(self.w, "nq", "8k"), "w"))
        self.assertIsNone(cells.cell_result(self.w, "nq", "8k"))

    def test_merge_writes_beaker_layout(self):
        fake_cell(self.w, "contradiction", "8k", 0.959)
        fake_cell(self.w, "contradiction", "2k", 0.977)
        fake_cell(self.w, "oolong", "8k", 0.648, parse_rate=0.99)
        done = cells.merge(self.w)
        self.assertEqual(done, {"contradiction": ["8k", "2k"], "oolong": ["8k"]})
        d = json.load(open(os.path.join(self.w, "eval", "contra_multirung.json")))
        self.assertEqual((d["contradiction_8k"], d["contradiction_2k"]), (0.959, 0.977))
        self.assertEqual(d["_meta"]["eval_size"], {"contradiction_8k": 500, "contradiction_2k": 500})
        self.assertEqual(d["_meta"]["train_run"], "R")
        self.assertEqual(d["model_path"], "/data/x")
        o = json.load(open(os.path.join(self.w, "eval", "oolong_multirung.json")))
        self.assertEqual(o["_meta"]["parse_rate"], {"oolong_8k": 0.99})
        for stem in ("contra_multirung_contradiction_8k_responses.jsonl",
                     "contra_multirung_contradiction_2k_responses.jsonl"):
            self.assertTrue(os.path.exists(os.path.join(self.w, "eval", stem)))

    def test_rows_match_record_beaker_evals_fields(self):
        fake_cell(self.w, "contradiction", "8k", 0.959)
        fake_cell(self.w, "oolong", "2k", 0.856)
        rs = cells.rows(self.w, "R", "R-evslurm-7", "7", ckpt="/data/c", node="cubbins")
        by = {(r["task"], r["rung"]): r for r in rs}
        self.assertEqual(set(by), {("contradiction", "8k"), ("oolong", "2k")})
        c = by[("contradiction", "8k")]
        self.assertEqual((c["metric"], c["value"], c["eval_size"], c["train_run"], c["experiment"], c["cot_label"]),
                         ("f1", 0.959, 500, "R", "tokdrop_lr", "no-cot"))
        self.assertAlmostEqual(c["se"], (0.959 * 0.041 / 500) ** 0.5, places=6)
        self.assertEqual((c["eval_job_id"], c["ckpt_path"]), ("slurm:7", "cubbins:/data/c"))
        self.assertEqual(by[("oolong", "2k")]["metric"], "score")

    def test_rows_use_the_job_that_scored_each_cell(self):
        fake_cell(self.w, "nq", "8k", 0.9)
        fake_cell(self.w, "nq", "2k", 0.95)
        open(os.path.join(cells.cell_dir(self.w, "nq", "8k"), "scored_by"), "w").write("R-evslurm-1 1\n")
        rs = {r["rung"]: r for r in cells.rows(self.w, "R", "R-evslurm-2", "2", tasks=["nq"])}
        self.assertEqual((rs["8k"]["run_id"], rs["8k"]["eval_job_id"]), ("R-evslurm-1", "slurm:1"))
        self.assertEqual((rs["2k"]["run_id"], rs["2k"]["eval_job_id"]), ("R-evslurm-2", "slurm:2"))

    def test_rows_ignore_lines_from_other_cells(self):
        fake_cell(self.w, "nq", "8k", 0.9)
        with open(os.path.join(cells.cell_dir(self.w, "nq", "8k"), "cell.log"), "a") as f:
            f.write("[ladder:nq@2k] f1=0.100 (n=500\n[ladder:nq@8k] f1=0.900 (n=500\n")
        rs = cells.rows(self.w, "R", "id", "1")
        self.assertEqual([(r["rung"], r["value"]) for r in rs], [("8k", 0.9)])

    def test_run_cells_with_stub_runner(self):
        """run_cells.sh end to end against a stub runner: the DC_RUNG_FILES key the runner sees is the LADDER task
        (contradiction, not the runner key contra), every cell runs once, and a rerun skips finished cells."""
        import subprocess

        here = os.path.dirname(os.path.abspath(__file__))
        code = os.path.join(self.w, "code")
        rdir = os.path.join(code, "recipes", "eval", "legacy_native")
        os.makedirs(rdir)
        calls = os.path.join(self.w, "calls.txt")
        open(os.path.join(rdir, "run_beaker_multirung_eval.sh"), "w").write(f"""#!/bin/bash
python3 - <<'PY'
import json, os
files = json.loads(os.environ["DC_RUNG_FILES"]); (task, rungs), = files.items(); (rung, path), = rungs.items()
open("{calls}", "a").write(f"{{os.environ['TASK']}} {{task}} {{rung}} {{path}}\\n")
key = f"{{task}}_{{rung}}"
out = os.path.join(os.environ["EVAL_OUT_DIR"], os.environ["TASK"] + "_multirung.json")
json.dump({{"_meta": {{"eval_size": {{key: 500}}}}, key: 0.5}}, open(out, "w"))
print(f"[ladder:{{task}}@{{rung}}] f1=0.500 (n=500, skipped_too_long=0)")
PY
""")
        env = dict(os.environ, PY=sys.executable, TOOLS=here, WORK=os.path.join(self.w, "work"), CODE=code,
                   CKPT="/x", RUN="R", RUN_ID="R-evslurm-1", NGPU="2", ROOT=os.path.join(self.w, "root"),
                   TASKS="contradiction,nq", RUNGS="8k,2k")
        sh = os.path.join(here, "run_cells.sh")
        r = subprocess.run(["bash", sh], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        got = [ln.split() for ln in open(calls).read().splitlines()]
        self.assertEqual([(g[0], g[1], g[2]) for g in got],
                         [("contra", "contradiction", "8k"), ("nq", "nq", "8k"),
                          ("contra", "contradiction", "2k"), ("nq", "nq", "2k")])
        self.assertTrue(got[0][3].endswith(cells.RUNG_FILES["contradiction"]["8k"]))
        self.assertEqual(cells.plan(env["WORK"], ["contradiction", "nq"]), [])
        r = subprocess.run(["bash", sh], env=env, capture_output=True, text=True)  # resume: nothing left to run
        self.assertEqual(r.returncode, 0)
        self.assertEqual(len(open(calls).read().splitlines()), 4)

    def test_rung_files_match_launch_fast8k(self):
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "launch_fast8k.py")).read()
        for task, rungs in cells.RUNG_FILES.items():
            for rel in rungs.values():
                self.assertIn(rel, src, f"{task}: {rel} not in launch_fast8k.RUNGS")
        self.assertIn('TASK_KEY = {"nq": "nq", "outlier": "outlier", "oolong": "oolong", "contradiction": "contra"}',
                      src)


if __name__ == "__main__":
    unittest.main()
