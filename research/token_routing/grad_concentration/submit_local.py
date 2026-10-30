"""Submit routed_probe shards to jsteinhardt via run_gap_local.sbatch (the fallback when Beaker is backed up).

    python3 research/token_routing/grad_concentration/submit_local.py --label f1952 --ckname f8k-mt4-dense-f1952-tde-lr5e-06 \
        --adam results/adam_f1952.pt --num-shards 8 --node cubbins [--qos preemptive_high] [--shards 0,1,2]
"""
import argparse
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--ckname", required=True)
ap.add_argument("--adam", default="", help="factor file (relative to this dir or absolute); '' = no Adam")
ap.add_argument("--adam-publish", default="", help="--only-adam-factors job: publish to this path")
ap.add_argument("--num-shards", type=int, default=8)
ap.add_argument("--shards", default="")
ap.add_argument("--node", default="cubbins")
ap.add_argument("--qos", default="preemptive_high")
ap.add_argument("--args", default="")
ap.add_argument("--minutes", type=int, default=75)
a = ap.parse_args()
only = [int(x) for x in a.shards.split(",")] if a.shards else range(a.num_shards)
for i in only:
    args = f"--num-shards {a.num_shards} --shard {i} {a.args}".strip() if not a.adam_publish else a.args
    env = [f"SRC={REPO}", f"LABEL={a.label}", f"CKNAME={a.ckname}", f"MINUTES={a.minutes}", f"ARGS={args}"]
    if a.adam:
        env.append(f"ADAM_FROM={a.adam if a.adam.startswith('/') else os.path.join(HERE, a.adam)}")
    if a.adam_publish:
        env.append(f"ADAM_PUBLISH={a.adam_publish}")
    name = f"gradgap-{a.label}-s{i}" if not a.adam_publish else f"gradgap-adam-{a.label}"
    out = subprocess.run(["sbatch", "-w", a.node, f"--qos={a.qos}", "-J", name, "--export=ALL," + ",".join(env),
                          os.path.join(HERE, "run_gap_local.sbatch")], capture_output=True, text=True)
    print(name, out.stdout.strip(), out.stderr.strip()[-300:])
    if a.adam_publish:
        break
