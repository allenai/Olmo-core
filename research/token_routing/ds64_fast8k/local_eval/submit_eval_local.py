#!/usr/bin/env python3
"""Submit the fast8k 4-task eval of one checkpoint to Berkeley slurm (run_eval_local.sbatch). Stdlib; login-node safe.

    # a node-local train_ctc_suite export (config.json + model_and_optim/) on mooney
    python3 research/token_routing/ds64_fast8k/local_eval/submit_eval_local.py \\
        --run f8k-mt4-wtsS3-f1952-wts-lr1e-05-loc --node mooney --ckpt /data/prasann/ctc_suite/ckpts/<run>
    # a model-only copy on the S3 hub, any H200 node that has the env
    python3 .../submit_eval_local.py --run <run> \\
        --ckpt-s3 checkpoints/prasanns/_transfer/gradconc/<run> --dense-config

Defaults: jsteinhardt, ``--qos preemptive`` (bumps ``normal``, is bumped by ``preemptive_high``, and does NOT count
toward our 8-GPU ``preemptive_high`` cap), 2 H200s, the commit at HEAD of the checkout this is run from, every task
and rung, experiment ``tokdrop_lr``. Registers the job with ``ops register`` right after ``sbatch`` (the job
re-registers with its node and log when it starts). ``--dry-run`` prints the sbatch line and checks only.
"""
import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SBATCH = os.path.join(HERE, "run_eval_local.sbatch")
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
ENV_PY = "/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python"
LOG_DIR = "/data/prasann/joblogs"
#: jsteinhardt H200 nodes (A100s need Prasann's OK; RTX cards cannot hold the 8k decode)
H200_NODES = ["cubbins", "mooney", "sneetches", "mcfuzz"]
TASKS = ["contradiction", "nq", "outlier", "oolong"]


def net(node, path):
    """The /net view of a node-local path (light inspection only: existence checks)."""
    return f"/net/{node}{path}"


def node_ready(node):
    """[] when ``node`` has the node-local env and the job-log dir, else what is missing."""
    miss = []
    if not os.path.exists(net(node, ENV_PY)):
        miss.append(ENV_PY)
    if not os.path.isdir(net(node, LOG_DIR)):
        miss.append(LOG_DIR)
    return miss


def git(src, *args):
    return subprocess.run(["git", "-C", src, *args], capture_output=True, text=True, check=True).stdout.strip()


def preempt_note(gpus, qos, nodes):
    """Who a start on one of ``nodes`` would bump (ops capacity --would-preempt), best effort."""
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "ops")))
    try:
        from opslib import capacity
    except Exception:  # noqa: BLE001  (a checkout without ops capacity)
        return None
    try:
        snap = capacity.parse(capacity.fetch("berkeley"))
        opts = [capacity.would_preempt(snap, gpus=gpus, qos=qos, partition="jsteinhardt", node=n) for n in nodes]
    except Exception as e:  # noqa: BLE001
        return f"capacity check unavailable ({type(e).__name__}: {e})"
    free = [w for w in opts if w.get("fits_free")]
    bump = sorted((w for w in opts if w.get("node") and not w.get("fits_free")), key=lambda w: w["others_gpu_h_lost"])
    wp = (free or bump or [{"pending": True}])[0]
    if wp.get("fits_free"):
        return f"fits in free GPUs on {wp['node']}: preempts nobody"
    if wp.get("pending"):
        return f"none of {','.join(nodes)} can host {gpus} GPUs at {qos} now, even by preempting: it will pend"
    who = ", ".join(f"{v['user']} {v['gpus_per_node']}x{v['qos']}" for v in wp["victims"])
    return f"would start on {wp['node']} by preempting: {who} (requeued by slurm)"


def build(a):
    """(sbatch argv, env overrides, run_id placeholder) for the parsed args; raises SystemExit on a bad request."""
    src = a.src or git(os.getcwd(), "rev-parse", "--show-toplevel")
    sha = git(src, "rev-parse", "--verify", f"{a.code_ref}^{{commit}}")
    if bool(a.ckpt) == bool(a.ckpt_s3):
        raise SystemExit("pass exactly one of --ckpt (node-local dir, with --node) or --ckpt-s3")
    if a.ckpt:
        if not a.node:
            raise SystemExit("--ckpt is node-local: say which node holds it (--node)")
        meta = net(a.node, os.path.join(a.ckpt, "model_and_optim", ".metadata"))
        if not os.path.exists(meta):
            raise SystemExit(f"{a.node}:{a.ckpt} has no model_and_optim/.metadata ({meta})")
        if not a.dense_config and not os.path.exists(net(a.node, os.path.join(a.ckpt, "config.json"))):
            raise SystemExit(f"{a.node}:{a.ckpt} has no config.json; pass --dense-config only if it is a plain "
                             "(or LoRA-merged) Qwen3.5-4B model-only resave")
    nodes = [a.node] if a.node else H200_NODES
    ready = [n for n in nodes if not node_ready(n)]
    if a.node and not ready:
        raise SystemExit(f"{a.node} is missing {node_ready(a.node)}")
    if not ready:
        raise SystemExit("no H200 node has the env and log dir")
    tasks = [t for t in a.tasks.split(",") if t]
    if [t for t in tasks if t not in TASKS]:
        raise SystemExit(f"--tasks must come from {TASKS}")
    name = f"f8kev-{a.run}"[:120]
    argv = ["sbatch", "--parsable", "-J", name, f"--qos={a.qos}", f"--gres=gpu:H200:{a.gpus}",
            f"--cpus-per-task={8 * a.gpus}", f"--mem={100 * a.gpus}G", f"--time={a.time}"]
    if a.node:
        argv += ["-w", a.node]
    else:
        skip = [n for n in H200_NODES if n not in ready]
        if skip:
            argv += ["--exclude", ",".join(skip)]
    argv.append(SBATCH)
    env = {"RUN": a.run, "SRC": src, "CODE_REF": sha, "TOOLS": HERE, "TASKS": ",".join(tasks), "RUNGS": a.rungs,
           "OPS_EXPERIMENT": a.experiment, "NGPU": str(a.gpus), "EXPECTED_MINUTES": str(a.expected_minutes)}
    if a.ckpt:
        env["CKPT"] = a.ckpt.rstrip("/")
    else:
        env["CKPT_S3"] = a.ckpt_s3.strip("/")
    if a.dense_config:
        env["DENSE_CONFIG"] = "qwen3_5_4B"
    if a.gen_batch != 1:
        env["LADDER_GEN_BATCH"] = str(a.gen_batch)
    if a.fresh:
        env["FRESH"] = "1"
    if not a.record:
        env["RECORD"] = "0"
    return argv, env, src, sha, ready


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run name (results are filed under this train_run)")
    ap.add_argument("--ckpt", help="node-local checkpoint dir (config.json + model_and_optim/)")
    ap.add_argument("--node", help="node holding --ckpt (required with --ckpt); optional pin with --ckpt-s3")
    ap.add_argument("--ckpt-s3", help="key prefix under s3://ai2-llm/ holding model_and_optim/")
    ap.add_argument("--dense-config", action="store_true",
                    help="checkpoint has no config.json: write a plain Qwen3.5-4B one (model-only resaves)")
    ap.add_argument("--qos", default="preemptive", choices=["preemptive", "preemptive_high", "normal"],
                    help="preemptive (default): outside our 8-GPU preemptive_high cap; bumps normal jobs")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--time", default="04:00:00")
    ap.add_argument("--expected-minutes", type=int, default=90)
    ap.add_argument("--src", help="checkout whose commit runs (default: the git toplevel of the cwd)")
    ap.add_argument("--code-ref", default="HEAD", help="commit / branch in --src (default HEAD)")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--rungs", default="8k,2k")
    ap.add_argument("--experiment", default="tokdrop_lr")
    ap.add_argument("--gen-batch", type=int, default=1, help="1 = Beaker's bs=1 decode (default)")
    ap.add_argument("--fresh", action="store_true", help="rescore every cell (drop this run's finished cells)")
    ap.add_argument("--no-record", dest="record", action="store_false", help="do not file ops result rows")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    sb, env, src, sha, nodes = build(a)
    line = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + " ".join(shlex.quote(x) for x in sb)
    print(f"code {sha[:12]} from {src}\n{line}")
    note = preempt_note(a.gpus, a.qos, nodes)
    if note:
        print(f"[capacity] {note}")
    if a.dry_run:
        return 0
    p = subprocess.run(sb, env={**os.environ, **env}, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout + p.stderr, file=sys.stderr)
        return p.returncode
    job = p.stdout.strip().split(";")[0]
    run_id = f"{a.run}-evslurm-{job}"
    reg = [OPS, "register", "--run-id", run_id, "--experiment", a.experiment, "--cluster", "slurm",
           "--job-id", job, "--kind", "eval", "--parent-run", a.run, "--task", "mt4",
           "--expected-minutes", str(a.expected_minutes), "--git", src, "--source", "submit_eval_local.py",
           "--command", line, "--notes", f"fast8k 4-task local eval, code {sha}"]
    r = subprocess.run(reg, capture_output=True, text=True)
    print(f"submitted job {job} (run_id {run_id}); log {LOG_DIR}/f8kev-{a.run}-{job}.out on the node"
          + ("" if r.returncode == 0 else f"\n[ops] WARNING register failed: {(r.stdout + r.stderr).strip()[-300:]}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
