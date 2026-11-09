"""1-GPU Beaker launcher for the sparsity_scale probes (records/sparsity-scale-plan.md).

One job per (size, task): reads the fs35 dense checkpoint and the fs35 eval rows straight from weka
(no transfer), runs PROBE (default marker_check.py), writes JSON to weka and prints a SUMMARY line.
Urgent, minRuntime 1h, ai2/flex2 + ai2/oe-other (unallocated), the olmo-core stable image via
``build_launch_config`` (the path the fs35 Qwen3.5 GPU jobs used). Beaker runs the PUSHED commit.

    python research/token_routing/sparsity_scale/launch_beaker.py --sizes 0.8b --tasks contradiction --rows 16 dry_run
    python research/token_routing/sparsity_scale/launch_beaker.py --sizes 0.8b,2b,9b,27b --tasks contradiction,oolong launch
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))

W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
CKPT = {(s, t): f"{W}/ctc_suite/ckpts/fs35s{s.replace('.', '')}-{t}-dense-{'s56M' if t == 'contradiction' else 's80M'}"
        for s in ("0.8b", "2b", "9b", "27b") for t in ("contradiction", "oolong")}
EVAL = {"contradiction": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
        "oolong": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl"}
OUT_ROOT = f"{W}/sparsity_scale"
EXPECTED_MIN = {"0.8b": 15, "2b": 20, "9b": 35, "27b": 75}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dry_run", "launch"])
    ap.add_argument("--sizes", default="0.8b")
    ap.add_argument("--tasks", default="contradiction")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--probe", default="marker_check.py")
    ap.add_argument("--extra", default="", help="extra args passed to the probe")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--min-runtime", default="1h")
    ap.add_argument("--git-ref", default="", help="pushed commit to run (default: the checkout's HEAD)")
    ap.add_argument("--git-branch", default="")
    a = ap.parse_args()

    from olmo_core.internal.common import build_launch_config
    from olmo_core.launch.beaker import BeakerEnvVar, OLMoCoreBeakerImage

    stamp = time.strftime("%m%d%H%M")
    probe_tag = os.path.splitext(a.probe)[0].replace("_", "")
    for size in a.sizes.split(","):
        for task in a.tasks.split(","):
            name = f"ssc-{probe_tag}-{size.replace('.', 'p')}-{task}-{stamp}"
            out = f"{OUT_ROOT}/{probe_tag}/{name}.json"
            cmd = [f"research/token_routing/sparsity_scale/{a.probe}", "--size", size, "--task", task,
                   "--ckpt", CKPT[(size, task)], "--eval-jsonl", EVAL[task], "--rows", str(a.rows), "--out", out]
            cmd += a.extra.split() if a.extra else []
            lc = build_launch_config(name=name, cmd=cmd, cluster=a.cluster, beaker_image=OLMoCoreBeakerImage.stable,
                                     workspace="ai2/flex2", budget="ai2/oe-other", num_nodes=1, num_gpus=1)
            lc.priority = "urgent"
            if a.git_ref:  # run a pushed commit other than HEAD (the local tree has other sessions' edits)
                lc.git.ref, lc.git.branch, lc.allow_dirty = a.git_ref, a.git_branch or None, True
            lc.min_runtime = a.min_runtime
            lc.step_timeout = None
            lc.step_soft_timeout = None
            lc.env_vars.append(BeakerEnvVar(name="PYTORCH_CUDA_ALLOC_CONF", value="expandable_segments:True"))
            if a.mode == "dry_run":
                print(f"[ssc] DRY RUN {name}\n  cmd: {' '.join(cmd)}\n  ref: {getattr(getattr(lc, 'git', None), 'ref', None)}")
                continue
            wl = lc.launch(follow=False)
            wl_id = getattr(wl, "id", None) or getattr(getattr(wl, "experiment", None), "id", None)
            print(f"[ssc] SUBMITTED id={wl_id} name={name} url=https://beaker.org/ex/{wl_id} out={out}", flush=True)
            os.system(f"/accounts/projects/berkeleynlp/prasann/bin/ops register --run-id {name} --experiment sparsity_scale "
                      f"--cluster beaker --kind eval --job-id {wl_id} --task {task} --expected-minutes {EXPECTED_MIN[size]} "
                      f"--ckpt-path {CKPT[(size, task)]} --git {REPO} >/dev/null")


if __name__ == "__main__":
    main()
