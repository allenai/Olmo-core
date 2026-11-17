"""Launch early dense-loss proxy runs on AI2 Beaker: ONE 2-GPU H100 job per call, arms back to back, exactly the
``run_proxy.sbatch`` loop (``run_proxy_beaker.sh``: same trainer and probe flags, 16 rows/step, micro-batch 2,
sequence 16,384). Same arguments as ``launch_proxy.py`` where they apply (arms vocabulary imported from it).

Gantry runs the PUSHED commit of the current branch: commit and ``git push -u origin <your-branch>`` first (never
``main`` / ``prasann/flopdrop_clean``); the launcher refuses an unpushed HEAD. Results: weka
``/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy/results/<tag>/`` and, per run as it ends,
``s3://ai2-llm/checkpoints/prasanns/early_loss_proxy/results/<tag>/`` -> ``fetch_beaker_results.sh <tag>``.

    python3 research/token_routing/early_loss_proxy/launch_proxy_beaker.py --tag mytag --arms recipe --seeds 0 \\
        --steps 25,50,75,100 --max-train-s 600 [--custom 'myarm=softtoken,1e-5,<flags>'] [--dry-run]

See BEAKER.md beside this file.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOGIN_PY = "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python"  # has beaker-py + gantry
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
ROOT = f"{W}/early_loss_proxy"
STATE = f"{HERE}/state_beaker.json"
PAYLOAD = "research/token_routing/early_loss_proxy/run_proxy_beaker.sh"

try:  # re-exec under the env that has beaker-py (system python3 does not), keeping the arguments
    import beaker  # noqa: F401
except ImportError:
    if os.path.exists(LOGIN_PY) and os.path.realpath(sys.executable) != os.path.realpath(LOGIN_PY):
        env = dict(os.environ, PATH=f"{os.path.expanduser('~/.local/bin')}:{os.path.dirname(LOGIN_PY)}:"
                   + os.environ.get("PATH", ""))
        os.execve(LOGIN_PY, [LOGIN_PY, os.path.abspath(__file__)] + sys.argv[1:], env)
    raise

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from launch_proxy import ARMS  # noqa: E402


def sh(cmd):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()


def estimate_minutes(n_runs, steps, sets, max_train_s, arms):
    """Rough wall-clock per job on 2 H100s (BEAKER.md has the measured numbers): weka base load + setup, steps,
    probes. Errs long: it sizes minRuntime (min 30 min) and the ops OVERRUN flag, not the result."""
    last = max(int(s) for s in steps.split(","))
    rows = sum(int(s.partition(":")[2] or 16) for s in sets.split(","))
    probe_s = 0.25 * rows  # ~0.2 s/row at 4-16k on 2 GPUs (192 rows ~ 45 s)
    total = 0.0
    for arm in arms:
        sps = 25.0 if arm.startswith("dense") else 12.0
        train = min(last * sps, max_train_s) if max_train_s else last * sps
        n_probe = len(steps.split(",")) + 1
        total += 180 + train + n_probe * probe_s
    return max(10, math.ceil(total / 60 * n_runs / max(1, len(arms))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="dense,recipe")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", default="1,2,5,10")
    ap.add_argument("--sets", default="suite4k:16,suite8k:16,suite16k:16")
    ap.add_argument("--heldout", default=f"{ROOT}/heldout_v2.npz", help="held-out rows on weka")
    ap.add_argument("--data", default=f"{ROOT}/shards/mtb16k", help="training pool on weka")
    ap.add_argument("--base", default=f"{W}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim")
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--micro", type=int, default=2, help="micro-batch rows per GPU (run_proxy.sbatch: 2)")
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--stop-ce", type=float, default=0.0)
    ap.add_argument("--max-train-s", type=float, default=0.0)
    ap.add_argument("--custom", action="append", default=[], metavar="NAME=VARIANT,LR,FLAGS")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--lr-mult", type=float, default=1.0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--minutes", type=int, default=0, help="expected job minutes (default: estimated)")
    ap.add_argument("--min-runtime", default="", help="Beaker minRuntime (default max(30m, 1.3 x expected), <= 4h)")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale", help="any-of, H100s")
    ap.add_argument("--force", action="store_true", help="rerun arms whose result JSON is already done")
    ap.add_argument("--dry-run", action="store_true", help="print the job, submit nothing")
    a = ap.parse_args()

    for c in a.custom:
        name, _, rest = c.partition("=")
        variant, lr, flags = rest.split(",", 2)
        ARMS[name] = (variant, float(lr), flags.strip(), None, "custom")
    lines, arm_list = [], []
    for seed in a.seeds.split(","):
        for arm in a.arms.split(","):
            variant, lr, extra, _, _ = ARMS[arm]
            flags = f"{extra} {a.extra}".strip()
            if "/data/" in flags:
                raise SystemExit(f"arm {arm}: flags name a Berkeley /data path ({flags}); stage it to weka and "
                                 f"pass a --custom arm with the {W}/... path instead")
            name = f"elp-{a.tag}-{arm}-sd{seed}"
            lines.append("\t".join([name, variant, f"{lr * a.lr_mult:g}", seed, flags]))
            arm_list.append(arm)
    spec = "\n".join(lines) + "\n"
    minutes = a.minutes or estimate_minutes(len(lines), a.steps, a.sets, a.max_train_s, arm_list)
    min_rt = a.min_runtime or f"{min(240, max(30, math.ceil(1.3 * minutes)))}m"

    # gantry clones the pushed commit: refuse anything else (an unpushed HEAD fails on the node or ships stale code)
    head = sh(["git", "rev-parse", "HEAD"])
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch in ("main", "prasann/flopdrop_clean"):
        raise SystemExit(f"refusing to launch from {branch}: work on your own branch and push that")
    dirty = sh(["git", "status", "--porcelain", "--untracked-files=no"])
    if not sh(["git", "branch", "-r", "--contains", head]):
        raise SystemExit(f"HEAD {head[:9]} ({branch}) is not on any remote branch: git push -u origin {branch}")
    if dirty:
        print(f"[elp-beaker] note: tracked files are modified; Beaker runs the PUSHED commit {head[:9]}, not them")

    from olmo_core.internal.common import build_launch_config
    from olmo_core.launch.beaker import BeakerEnvVar

    env = {"TAG": a.tag, "STEPS": a.steps, "SETS": a.sets, "HELDOUT": a.heldout, "DATA": a.data, "BASE": a.base,
           "SEQLEN": str(a.seq_len), "NGPU": str(a.gpus), "GB": str(a.global_batch), "MICRO": str(a.micro),
           "STOPCE": str(a.stop_ce) if a.stop_ce else "", "MAXTRAIN": str(a.max_train_s) if a.max_train_s else "",
           "FORCE": "1" if a.force else "", "SPEC_B64": base64.b64encode(spec.encode()).decode()}
    cfg = build_launch_config(name=f"elp-{a.tag}", cmd=["bash", PAYLOAD], cluster=a.cluster, workspace="ai2/flex2",
                              budget="ai2/oe-other", num_nodes=1, num_gpus=a.gpus)
    cfg.torchrun = False  # the payload runs one torch.distributed.run per arm itself
    cfg.priority = "urgent"
    cfg.min_runtime = min_rt
    cfg.allow_dirty = True  # untracked files elsewhere in the tree are fine: the pushed commit is what runs
    cfg.step_timeout = None
    cfg.step_soft_timeout = None
    cfg.env_vars += [BeakerEnvVar(name=k, value=v) for k, v in env.items() if v != ""]
    print(f"[elp-beaker] {len(lines)} runs: {', '.join(ln.split(chr(9))[0] for ln in lines)}")
    print(f"[elp-beaker] commit {head[:9]} ({branch}) cluster {a.cluster} gpus {a.gpus} expected {minutes} min "
          f"minRuntime {min_rt}")
    print(f"[elp-beaker] results -> {ROOT}/results/{a.tag}/ (fetch: bash {HERE}/fetch_beaker_results.sh {a.tag})")
    if a.dry_run:
        print(spec)
        return
    wl = cfg.launch(follow=False)
    ex = getattr(wl, "id", None) or getattr(getattr(wl, "experiment", None), "id", None)
    if not ex:
        raise SystemExit(f"launch returned no id: {str(wl)[:400]}")
    print(f"[elp-beaker] SUBMITTED id={ex} url=https://beaker.org/ex/{ex}", flush=True)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"jobs": {}}
    st["jobs"][ex] = {"tag": a.tag, "runs": [ln.split("\t")[0] for ln in lines], "steps": a.steps, "sets": a.sets,
                      "commit": head, "branch": branch, "extra": a.extra, "lr_mult": a.lr_mult, "data": a.data,
                      "heldout": a.heldout, "seq_len": a.seq_len, "gpus": a.gpus, "micro": a.micro,
                      "custom": a.custom, "min_runtime": min_rt, "expected_minutes": minutes}
    json.dump(st, open(STATE + f".{ex}", "w"), indent=1)
    os.replace(STATE + f".{ex}", STATE)
    subprocess.run([OPS, "register", "--run-id", f"elp-{a.tag}-bk{ex[-8:]}", "--experiment", "tokdrop_lr",
                    "--cluster", "beaker", "--job-id", ex, "--expected-minutes", str(minutes), "--git", REPO])


if __name__ == "__main__":
    main()
