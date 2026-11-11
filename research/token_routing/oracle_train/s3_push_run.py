"""Upload a node-local oracle run to S3 _transfer/oracle/runs/<run> WITHOUT its stepN resume checkpoints (they stay on the
node): the eval reads only the model-only checkpoint + the json files at the run root.
    python s3_push_run.py <local run dir> <run name>      (run from research/token_routing/olmo3_cptdrop's parent)"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "olmo3_cptdrop"))
import s3_sync  # noqa: E402

_walk = os.walk


def _walk_no_steps(top, *a, **k):
    for dp, ds, fs in _walk(top, *a, **k):
        ds[:] = [d for d in ds if not re.fullmatch(r"step\d+", d) and not d.startswith("work_")]
        yield dp, ds, fs


os.walk = _walk_no_steps
s3_sync.ROOT = "checkpoints/prasanns/_transfer/oracle"
local, run = sys.argv[1], sys.argv[2]
sys.argv = ["s3_sync", "up", local, f"runs/{run}"]
s3_sync.main()
