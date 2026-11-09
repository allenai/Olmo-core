#!/usr/bin/env python3
"""Launch weka_apply.py as a 0-GPU Beaker job (weka is not mounted at Berkeley).

  python3 launch_weka_apply.py MANIFEST.tsv --tag 20261030          # verify only
  python3 launch_weka_apply.py MANIFEST.tsv --tag 20261030 --apply  # delete

Uploads weka_apply.py + the manifest as a Beaker dataset (mounted at /input), runs urgent with
minRuntime 1h from ai2/flex2 / ai2/oe-other (unallocated), and writes the per-path JSON to the
job's result dataset. Prints the experiment id.
"""
import argparse, json, os, re, shutil, subprocess, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
BEAKER = os.path.expanduser("~/.local/bin/beaker")
WORKSPACE, BUDGET = "ai2/flex2", "ai2/oe-other"
IMAGE = "tylerr/olmo-core-tch291cu128-2025-11-25"
CLUSTERS = ["ai2/saturn", "ai2/neptune", "ai2/ceres", "ai2/jupiter"]

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--tag", required=True)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

kind = "apply" if a.apply else "verify"
name = f"weka-cleanup-{kind}-{a.tag}-{time.strftime('%H%M%S')}"
with tempfile.TemporaryDirectory() as td:
    shutil.copy(os.path.join(HERE, "weka_apply.py"), td)
    shutil.copy(a.manifest, os.path.join(td, "manifest.tsv"))
    out = subprocess.run([BEAKER, "dataset", "create", td, "-w", WORKSPACE, "-n", name + "-in"],
                         capture_output=True, text=True, check=True)
# prints "Uploading <dir> to <dataset id>" (no --format json support)
ds = {"id": re.search(r"Uploading .* to (\S+)", out.stdout + out.stderr).group(1)}
flag = "--apply" if a.apply else ""
script = (f"set -e; export PATH=/opt/conda/bin:$PATH; "
          f"python3 /input/weka_apply.py --manifest /input/manifest.tsv "
          f"--out /results/{kind}_{a.tag}.json {flag}; ls -la /results")
spec = {
    "version": "v2", "budget": BUDGET, "description": f"weka cleanup {kind} {a.tag}",
    "tasks": [{
        "name": "main", "image": {"beaker": IMAGE}, "command": ["bash", "-c", script],
        "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}},
                     {"mountPath": "/input", "source": {"beaker": ds["id"]}}],
        "result": {"path": "/results"}, "resources": {"cpuCount": 4},
        "context": {"priority": "urgent", "minRuntime": "1h"},
        "constraints": {"cluster": CLUSTERS},
    }],
}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump(spec, fh)
out = subprocess.run([BEAKER, "experiment", "create", fh.name, "-w", WORKSPACE, "-n", name,
                      "--format", "json"], capture_output=True, text=True, check=True)
exp = json.loads(out.stdout)
exp = exp[0] if isinstance(exp, list) else exp
print(name, exp["id"])
