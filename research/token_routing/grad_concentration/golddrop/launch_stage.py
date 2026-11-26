"""Launch the checkpoint-staging job (``stage_ckpts.sh``) on Beaker: CPU only, weka mounted, urgent, minRuntime sized to
the run; registered with ``ops`` (experiment tokdrop_lr). Gantry runs the PUSHED commit.

    python3 launch_stage.py --ckpts "gdd256:1450 gdd256:5800 gdd256:11496 dense256b:300 dense256b:615" --minutes 60
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
LOGIN_PY = "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python"
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
try:
    import beaker  # noqa: F401
except ImportError:
    if os.path.exists(LOGIN_PY) and os.path.realpath(sys.executable) != os.path.realpath(LOGIN_PY):
        env = dict(os.environ, PATH=f"{os.path.expanduser('~/.local/bin')}:{os.path.dirname(LOGIN_PY)}:" + os.environ.get("PATH", ""))
        os.execve(LOGIN_PY, [LOGIN_PY, os.path.abspath(__file__)] + sys.argv[1:], env)
    raise
sys.path.insert(0, os.path.join(REPO, "src"))


def sh(cmd):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--exclude-host", action="append", default=["jupiter-cs-aus-152"])
    ap.add_argument("--tag", default="tokgrad-stage")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    head, branch = sh(["git", "rev-parse", "HEAD"]), sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch in ("main", "prasann/flopdrop_clean"):
        raise SystemExit(f"refusing to launch from {branch}")
    if not sh(["git", "branch", "-r", "--contains", head]):
        raise SystemExit(f"HEAD {head[:9]} ({branch}) is not pushed")
    from olmo_core.internal.common import build_launch_config
    from olmo_core.launch.beaker import BeakerEnvVar

    cfg = build_launch_config(name=a.tag, cmd=["bash", "research/token_routing/grad_concentration/golddrop/stage_ckpts.sh"],
                              cluster=a.cluster, workspace="ai2/flex2", budget="ai2/oe-other", num_nodes=1, num_gpus=0)
    cfg.torchrun = False
    cfg.priority = "urgent"
    cfg.min_runtime = f"{min(240, max(30, math.ceil(1.3 * a.minutes)))}m"
    cfg.allow_dirty = True
    cfg.step_timeout = None
    cfg.step_soft_timeout = None
    cfg.env_vars += [BeakerEnvVar(name="CKPTS", value=a.ckpts)]
    if a.exclude_host:
        out = subprocess.run(["beaker", "cluster", "nodes", a.cluster, "--format", "json"], capture_output=True, text=True).stdout
        hosts = [n.get("hostname") for n in json.loads(out) if not n.get("cordoned")]
        hosts = [h for h in hosts if h and not any(x in h for x in a.exclude_host)]
        assert hosts, "no hosts left"
        cfg.hostnames = hosts
        cfg.clusters, cfg.gpu_types, cfg.tags = [], None, None
    print(f"[stage] commit {head[:9]} ({branch}) ckpts {a.ckpts} minRuntime {cfg.min_runtime}")
    if a.dry_run:
        return
    wl = cfg.launch(follow=False)
    ex = getattr(wl, "id", None) or getattr(getattr(wl, "experiment", None), "id", None)
    print(f"[stage] SUBMITTED id={ex} url=https://beaker.org/ex/{ex}", flush=True)
    subprocess.run([OPS, "register", "--run-id", f"{a.tag}-bk{str(ex)[-8:]}", "--experiment", "tokdrop_lr",
                    "--cluster", "beaker", "--kind", "other", "--job-id", str(ex), "--expected-minutes", str(a.minutes),
                    "--git", REPO])


if __name__ == "__main__":
    main()
