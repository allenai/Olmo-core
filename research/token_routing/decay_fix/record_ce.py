"""Record fetched answer-CE results (results/ce/<job>.json) with `ops result add --check`, one row per (ckpt, cell).

    python3 research/token_routing/decay_fix/record_ce.py [--commit SHA]
Metric ``answer_ce`` = mean over rows of the per-row mean CE on the answer's content tokens (teacher forced, no-cot);
also ``answer_top1`` (token-level teacher-forced argmax accuracy). Idempotence: skips jobs listed in results/ce/.recorded.
"""
import argparse
import glob
import json
import os
import subprocess

D = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="44d88d266")
    a = ap.parse_args()
    st = json.load(open(f"{D}/state.json"))
    done_p = f"{D}/results/ce/.recorded"
    done = set(open(done_p).read().split()) if os.path.exists(done_p) else set()
    for f in sorted(glob.glob(f"{D}/results/ce/*.json")):
        job = os.path.basename(f)[:-5]
        if job in done:
            continue
        ex = st["ce"].get(job, {}).get("ex")
        d = json.load(open(f))
        for ck, res in d.items():
            if ck.startswith("_") or not res.get("_done"):
                continue
            for cell, v in res.items():
                if not isinstance(v, dict):
                    continue
                task, rung = cell.rsplit("_", 1)
                for metric, val in (("answer_ce", v["ans_row_mean"]), ("answer_top1", v["ans_top1"])):
                    cmd = [OPS, "result", "add", "--check", "--run-id", job, "--train-run", ck,
                           "--experiment", "tokdrop_lr", "--task", task, "--rung", rung, "--metric", metric,
                           "--value", f"{val:.6f}", "--eval-size", str(v["eval_size"]), "--cot-label", "no-cot",
                           "--eval-backend", "answer-ce-teacher-forced", "--ckpt-path", f"{W}/{ck}",
                           "--git-commit", a.commit] + (["--eval-job-id", ex] if ex else [])
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    msg = (p.stdout + p.stderr).strip().splitlines()
                    print(f"{ck} {cell} {metric}={val:.4f}: {msg[-1] if msg else p.returncode}")
        open(done_p, "a").write(job + "\n")


if __name__ == "__main__":
    main()
