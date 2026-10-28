"""Run bench_ffn_speed.py on Beaker through the SAME launch path/image as the training jobs
(OLMoCoreBeakerImage.stable, pip install -e ., flash/fla kernels as trained), 1 GPU.
    python research/ffn_compute/flop_scaling/beaker_bench_launch.py --cluster ai2/jupiter-cirrascale-2 [--extra "--models q35-4B"]"""
import argparse, sys
from datetime import datetime
from olmo_core.internal.common import build_launch_config
from olmo_core.launch.beaker import BeakerEnvVar, OLMoCoreBeakerImage

ap = argparse.ArgumentParser()
ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
ap.add_argument("--extra", default="")
ap.add_argument("--priority", default="urgent")
ap.add_argument("--script", default="research/ffn_compute/flop_scaling/bench_ffn_speed.py", help="benchmark script (repo-relative); gets --out /results/<name>.json")
ap.add_argument("--gpus", type=int, default=1)
ap.add_argument("--env", action="append", default=[], help="KEY=VALUE env vars for the job")
ap.add_argument("--name", default="", help="experiment name (default bench-<script>-<MMDDHHMM>)")
ap.add_argument("--min-runtime", default="1h", help="Beaker minRuntime (urgent jobs at 0s effectively never schedule on jupiter)")
a = ap.parse_args()
tag = a.script.split("/")[-1].replace(".py", "").replace("_", "-")
name = a.name or f"bench-{tag}-{datetime.now().strftime('%m%d%H%M')}"
cmd = ["python", a.script, "--out", f"/results/{tag}.json"] + (a.extra.split() if a.extra else [])
lc = build_launch_config(name=name, cmd=cmd, cluster=a.cluster, beaker_image=OLMoCoreBeakerImage.stable,
                         workspace="ai2/flex2", budget="ai2/oe-other", num_nodes=1, num_gpus=a.gpus)
lc.priority = a.priority
lc.min_runtime = None if a.min_runtime in ("", "0", "0s") else a.min_runtime
for kv in a.env:
    k, v = kv.split("=", 1)
    lc.env_vars.append(BeakerEnvVar(name=k, value=v))
lc.allow_dirty = True  # gantry clones the PUSHED commit; local edits never ship
lc.step_timeout = None; lc.step_soft_timeout = None
wl = lc.launch(follow=False)
wl_id = getattr(wl, "id", None) or getattr(getattr(wl, "experiment", None), "id", None)
print(f"SUBMITTED id={wl_id} name={name}")
