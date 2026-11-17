"""Length-generalization evals for mt11_64k: score checkpoints at r64k / r128k / r256k on olmo-eval's CTC launcher.

The 2k-64k numbers of this experiment come from the legacy native docchunk ladder, which OOMs near ~77k tokens on one
H100 and has no YaRN. The long rungs therefore run on olmo-eval ``scripts/ctc_suite/launch_ctc_suite.py`` (branch
``prasann/ctc-suite-mt11lengen`` = ``prasann/ctc-suite-lengen`` + nothing; worktree ``projects/olmo-eval-mt11lengen``):
chat prompt format + doc markers, full attention, bs 1 above 32k, r256k with YaRN x2 + 131,072-token chunked prefill.
r64k is re-scored on this path for every checkpoint, so each checkpoint's 64k -> 128k -> 256k falloff is measured on
one path (and r64k anchors this path against the legacy ladder's 64k numbers).

OLMo-core is installed from ``claude/mt11-lengen-bed337`` (>= 2011b86d5: ``from_checkpoint`` applies
``post_build_hook_from_config``, so the universal recipe's early-stack LoRA adapters are re-installed; olmo-eval passes
no hook and would otherwise score the bare backbone). ``--olmo-core-no-deps`` keeps the image's torch 2.9.1 / fla 0.4.1.

    python research/token_routing/mt11_64k/lengen_evals.py launch --ckpts m64-dense-f2-lr5e-06,... [--dry]
    python research/token_routing/mt11_64k/lengen_evals.py status
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
OE = os.environ.get("LG_OLMO_EVAL", "/accounts/projects/berkeleynlp/prasann/projects/olmo-eval-mt11lengen")
OE_PY = os.environ.get("LG_OE_PY", f"{OE}/.venv-planner/bin/python")
CKPTS = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts"
STATE = os.path.join(HERE, "state_lengen.json")
OLMO_CORE_REF = os.environ.get("LG_OLMO_CORE_REF", "claude/mt11-lengen-bed337")
#: the six tasks with r64k/r128k/r256k suite rows that this experiment trains on and scores at 64k
ROWS = "ctc_nq,ctc_hpqa,ctc_qdmatch_nq,ctc_outlier,ctc_oolong,ctc_contradiction"
#: r256k ships 125 rows per subset (launcher caps there): flag it inline
POLICY = os.environ.get("LG_POLICY", "r64k:500,r128k:500,r256k:500")
ENV_PATH = f"{os.path.expanduser('~')}/.local/bin:/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin"


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def launch(a):
    st = load()
    st.setdefault("lengen", {})
    for name in a.ckpts.split(","):
        run = f"m11lg-{name}{a.suffix}"
        if run in st["lengen"] and not a.force and not a.dry:
            print(f"[skip] {run} already launched")
            continue
        cmd = [OE_PY, "scripts/ctc_suite/launch_ctc_suite.py", "--ckpt", f"{CKPTS}/{name}", "--arm", "full",
               "--doc-markers", "--rows", a.rows, "--policy", a.policy, "--olmo-core-ref", OLMO_CORE_REF,
               "--olmo-core-no-deps", "--min-runtime", a.min_runtime, "--run-name", run]
        if a.cluster:
            cmd += ["--cluster", a.cluster]
        if a.dry:
            cmd.append("--dry-run")
        env = {**os.environ, "PYTHONPATH": "src", "PATH": ENV_PATH + ":" + os.environ["PATH"]}
        out = subprocess.run(cmd, cwd=OE, env=env, capture_output=True, text=True).stdout
        jobs = re.findall(r"job(\d+) https://beaker.org/ex/([A-Z0-9]{26})", out)
        print(f"{run}: {len(jobs)} jobs" + ("" if jobs or a.dry else f"\n{out[-1500:]}"))
        if a.dry:
            print("\n".join(l for l in out.splitlines() if "GPU-h" in l or "jobs" in l))
            continue
        for idx, ex in jobs:
            subprocess.run([OPS, "register", "--run-id", f"{run}-j{idx}", "--experiment", "mt11_64k", "--cluster",
                            "beaker", "--kind", "eval", "--job-id", ex, "--expected-minutes", "90", "--parent-run", name,
                            "--notes", f"lengen olmo-eval {a.rows} {a.policy} (r256k YaRN x2 chunked); ref {OLMO_CORE_REF}",
                            "--git", OE], capture_output=True)
        st["lengen"][run] = {"ckpt": name, "jobs": dict(jobs), "rows": a.rows, "policy": a.policy}
        save(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["launch"])
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--rows", default=ROWS)
    ap.add_argument("--policy", default=POLICY)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--cluster", default="")
    ap.add_argument("--min-runtime", default="2h")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    launch(a)


if __name__ == "__main__":
    sys.exit(main())
