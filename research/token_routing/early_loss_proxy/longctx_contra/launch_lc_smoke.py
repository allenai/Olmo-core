"""Launch longctx_contra feasibility smokes on Beaker: ONE job (``--gpus`` H100s) running ``run_lc_smoke.sh`` over a list
of runs back to back. Each ``--run`` is ``name|mode|ngpu|variant|lr|data|seqlen|gb|steps|probe_sets|flags`` (mode probe
= through dense_probe.py, direct = bare trainer, needed for --cp-degree). Gantry runs the PUSHED commit; registered with
``ops`` (experiment tokdrop_lr). Results: weka/S3 ``early_loss_proxy/results/<tag>/`` (``../fetch_beaker_results.sh``).

    python3 launch_lc_smoke.py --tag lcsmoke1 --gpus 1 --minutes 40 --run 'gdd256|probe|1|softtoken|1e-5|lc256k|262144|1|3|lc256k:1|<flags>'
"""
from __future__ import annotations

import argparse
import base64
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ELP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ELP, "..", "..", ".."))
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--minutes", type=int, required=True, help="expected job minutes (minRuntime = min(4h, 1.3x))")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale")
    ap.add_argument("--presync", default="", help="space-separated paths under early_loss_proxy/ to copy S3 -> weka first")
    ap.add_argument("--save-root", default="", help="real arms: weka save root (no wipe; resumes; keeps stepN checkpoints)")
    ap.add_argument("--keep-ckpt", action="store_true", help="drop --no-final-checkpoint and keep every --save-interval checkpoint")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lines = []
    for r in a.run:
        f = r.split("|")
        assert len(f) == 11, f"--run needs 11 |-separated fields, got {len(f)}: {r}"
        assert int(f[2]) <= a.gpus, f"run {f[0]} wants {f[2]} GPUs > job {a.gpus}"
        lines.append("\t".join(f))
    spec = "\n".join(lines) + "\n"
    min_rt = f"{min(240, max(30, math.ceil(1.3 * a.minutes)))}m"
    head, branch = sh(["git", "rev-parse", "HEAD"]), sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch in ("main", "prasann/flopdrop_clean"):
        raise SystemExit(f"refusing to launch from {branch}")
    if not sh(["git", "branch", "-r", "--contains", head]):
        raise SystemExit(f"HEAD {head[:9]} ({branch}) is not pushed: git push -u origin {branch}")
    from olmo_core.internal.common import build_launch_config
    from olmo_core.launch.beaker import BeakerEnvVar

    cfg = build_launch_config(name=f"lc-{a.tag}", cmd=["bash", "research/token_routing/early_loss_proxy/longctx_contra/run_lc_smoke.sh"],
                              cluster=a.cluster, workspace="ai2/flex2", budget="ai2/oe-other", num_nodes=1, num_gpus=a.gpus)
    cfg.torchrun = False
    cfg.priority = "urgent"
    cfg.min_runtime = min_rt
    cfg.allow_dirty = True
    cfg.step_timeout = None
    cfg.step_soft_timeout = None
    cfg.env_vars += [BeakerEnvVar(name="TAG", value=a.tag),
                     BeakerEnvVar(name="SPEC_B64", value=base64.b64encode(spec.encode()).decode())]
    if a.save_root:
        cfg.env_vars.append(BeakerEnvVar(name="SAVE_ROOT", value=a.save_root))
    if a.keep_ckpt:
        cfg.env_vars.append(BeakerEnvVar(name="KEEPCKPT", value="1"))
    if a.presync:
        cfg.env_vars.append(BeakerEnvVar(name="PRESYNC", value=a.presync))
    print(f"[lc-smoke] {len(lines)} runs: {', '.join(l.split(chr(9))[0] for l in lines)}; commit {head[:9]} ({branch}) "
          f"gpus {a.gpus} expected {a.minutes} min minRuntime {min_rt}")
    if a.dry_run:
        print(spec)
        return
    wl = cfg.launch(follow=False)
    ex = getattr(wl, "id", None) or getattr(getattr(wl, "experiment", None), "id", None)
    print(f"[lc-smoke] SUBMITTED id={ex} url=https://beaker.org/ex/{ex}", flush=True)
    subprocess.run([OPS, "register", "--run-id", f"lc-{a.tag}-bk{str(ex)[-8:]}", "--experiment", "tokdrop_lr",
                    "--cluster", "beaker", "--kind", "other", "--job-id", str(ex), "--expected-minutes", str(a.minutes),
                    "--git", REPO])


if __name__ == "__main__":
    main()
