import datetime as dt
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from opslib import experiments as ex
from opslib.cli import main

SAMPLE = """---
experiment: ds64
title: Soft-KV ds64 scaling   # trailing comment
status: active
updated: 2026-10-01
branch: prasann/landmark
launcher:
records:
  - records/ds64-handoff.md
  - records/ds64-overnight-2026-09-14.md
results: results/ds64/summary.json
artifacts:
  - https://claude.ai/artifact/abc123
wandb_group: https://wandb.ai/x/y#frag
next_step: "launch the 64k rung"
---

# ds64

## Question
Does it scale?
"""


def write(root, name, text):
    p = Path(root) / name / "STATUS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def meta_text(name, status, updated, nxt="do x"):
    return (f"---\nexperiment: {name}\ntitle: T {name}\nstatus: {status}\nupdated: {updated}\n"
            f"next_step: {nxt}\n---\nbody\n")


class ParseTest(unittest.TestCase):
    def test_front_matter(self):
        meta, body = ex.parse_front_matter(SAMPLE)
        self.assertEqual(meta["experiment"], "ds64")
        self.assertEqual(meta["title"], "Soft-KV ds64 scaling")
        self.assertEqual(meta["records"], ["records/ds64-handoff.md",
                                           "records/ds64-overnight-2026-09-14.md"])
        self.assertEqual(meta["results"], ["results/ds64/summary.json"])
        self.assertEqual(meta["artifacts"], ["https://claude.ai/artifact/abc123"])
        self.assertEqual(meta["launcher"], "")
        self.assertEqual(meta["wandb_group"], "https://wandb.ai/x/y#frag")
        self.assertEqual(meta["next_step"], "launch the 64k rung")
        self.assertTrue(body.lstrip().startswith("# ds64"))
        self.assertEqual(ex.validate(meta, name="ds64", today=dt.date(2026, 10, 12)), [])

    def test_hash_in_value_is_not_a_comment(self):
        meta, _ = ex.parse_front_matter(
            "---\nnext_step: merge PR #400 into main  # note\n---\nbody\n"
        )
        self.assertEqual(meta["next_step"], "merge PR #400 into main")

    def test_no_front_matter(self):
        self.assertEqual(ex.parse_front_matter("# hi\n"), ({}, "# hi\n"))
        self.assertEqual(ex.parse_front_matter("---\nkey: v\nno end")[0], {})

    def test_template_roundtrip(self):
        t = ex.template("foo", today=dt.date(2026, 10, 12))
        meta, body = ex.parse_front_matter(t)
        self.assertEqual(meta["experiment"], "foo")
        self.assertEqual(meta["updated"], "2026-10-12")
        self.assertEqual(meta["records"], [])
        for sec in ex.BODY_SECTIONS:
            self.assertIn(f"## {sec}", body)

    def test_validate_problems(self):
        probs = ex.validate({"experiment": "a", "title": "t", "status": "running",
                             "updated": "10/1/2026", "next_step": "x"}, name="b")
        joined = " ".join(probs)
        self.assertIn("status", joined)
        self.assertIn("YYYY-MM-DD", joined)
        self.assertIn("directory name", joined)


class IndexStaleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        write(self.root, "a_old", meta_text("a_old", "active", "2026-09-01"))
        write(self.root, "b_new", meta_text("b_new", "active", "2026-10-11", nxt="a | b"))
        write(self.root, "c_done", meta_text("c_done", "done", "2026-08-19"))
        write(self.root, "d_bad", meta_text("d_bad", "weird", "nope"))
        self.today = dt.date(2026, 10, 12)

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_groups_and_sorts(self):
        exps = ex.load_all(self.root, today=self.today)
        self.assertEqual([e["name"] for e in exps], ["a_old", "b_new", "c_done", "d_bad"])
        text = ex.render_index(exps, today=self.today)
        self.assertLess(text.index("## Active"), text.index("## Done"))
        self.assertLess(text.index("b_new"), text.index("a_old"))  # newest first
        self.assertIn("a \\| b", text)  # pipes escaped
        self.assertIn("## Unknown status", text)
        self.assertIn("Front-matter problems", text)

    def test_stale(self):
        exps = ex.load_all(self.root, today=self.today)
        names = [e["name"] for e in ex.stale(exps, days=7, today=self.today)]
        self.assertEqual(names, ["a_old"])
        names = [e["name"] for e in ex.stale(exps, days=60, today=self.today)]
        self.assertEqual(names, [])

    def test_cli(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = main(["exp", "--dir", self.root, "--today", "2026-10-12", "new", "e_new"])
            self.assertEqual(rc, 0)
            rc = main(["exp", "--dir", self.root, "--today", "2026-10-12", "new", "e_new"])
            self.assertEqual(rc, 1)
            self.assertEqual(main(["exp", "--dir", self.root, "--today", "2026-10-12", "index"]), 0)
            self.assertEqual(main(["exp", "--dir", self.root, "show", "e_new"]), 0)
            self.assertEqual(main(["exp", "--dir", self.root, "show", "zzz"]), 1)
            self.assertEqual(main(["exp", "--dir", self.root, "--today", "2026-10-12",
                                   "stale", "--days", "7"]), 0)
            self.assertEqual(main(["exp", "--dir", self.root, "check"]), 1)  # d_bad
        idx = (Path(self.root) / "INDEX.md").read_text()
        self.assertIn("e_new", idx)
        self.assertIn("a_old", out.getvalue())


class RecordsIndexTest(unittest.TestCase):
    def test_records_index(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "README.md").write_text("| Doc | What |\n|---|---|\n| `old.md` | hand-written desc |\n"
                                         "- [listed.md](listed.md) — bullet desc\n")
            (r / "old.md").write_text("# Old doc\n\nFirst para.\n")
            (r / "listed.md").write_text("# Listed\n\nPara.\n")
            (r / "new.md").write_text("---\ntitle: New\nupdated: 2026-10-01\nkind: plan\n"
                                      "area: landmark\nsuperseded_by:\n"
                                      "summary: the summary\n---\n# New\n")
            (r / "sub").mkdir()
            (r / "sub" / "README.md").write_text("# sub\n")
            (r / "bare.md").write_text("# Bare\n\nOnly a paragraph here.\n")
            rows = {x["name"]: x for x in ex.record_rows(r)}
            self.assertEqual(rows["old.md"]["summary"], "hand-written desc")
            self.assertEqual(rows["listed.md"]["summary"], "bullet desc")
            self.assertEqual(rows["new.md"]["summary"], "the summary")
            self.assertEqual(rows["bare.md"]["summary"], "Only a paragraph here.")
            text = ex.render_records_index(list(rows.values()), today=dt.date(2026, 10, 12))
            self.assertLess(text.index("new.md"), text.index("old.md"))
            self.assertIn("3 docs still lack", text)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["exp", "records", "--records-dir", d, "--write"]), 0)
            written = (r / "README.md").read_text()
            self.assertIn("bare.md", written)
            self.assertIn("| doc | area |", written)
            self.assertIn("| landmark |", written)
            self.assertNotIn("[]", written)
            self.assertIn("[`sub/`](sub/README.md)", written)


if __name__ == "__main__":
    unittest.main()
