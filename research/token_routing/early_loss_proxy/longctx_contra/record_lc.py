"""Record checkpoint-then-score results with `ops result add --check` (one row per checkpoint x held-out set).

    python3 record_lc.py [--res /tmp/claude-3018/elp_beaker_results] [--dry-run]
Already-recorded (run_id, rung) pairs are skipped via a local ledger (record_lc_done.txt beside this file).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
TRAIN = {"dense256b": "lc-lcA-dense256-bkM7SXW87S", "gdd256": "lc-lcA-gdd256-bk83V6HYW0",
         "dense1m": "lc-lcA-dense1m-bkV24B68AB", "dense1m_mix": "lc-lcA-dense1mmix-bkD47QXQVS", "dense256_cp8chk": "lc-lcA-cp8check-bkDVXR7RBS", "gdd1m": "lc-lcA-gdd1m-bkTDZEJEEA", "base": "q35-4b-base-markerfix"}
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy/runs_lc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="/tmp/claude-3018/elp_beaker_results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--provenance", action="store_true",
                    help="re-add every row with git commit + eval job id from score_provenance.json (score_provenance.py); "
                    "own ledger record_lc_prov_done.txt; the later row supersedes the provenance-less one")
    a = ap.parse_args()
    ledger = os.path.join(HERE, "record_lc_prov_done.txt" if a.provenance else "record_lc_done.txt")
    prov = json.load(open(os.path.join(HERE, "score_provenance.json"))) if a.provenance else {}
    done = set(open(ledger).read().split()) if os.path.exists(ledger) else set()
    for f in sorted(glob.glob(os.path.join(a.res, "lcS-*", "score*-s*.json"))):
        m = re.search(r"/(score1m8|score1m|score)-([a-z0-9_]+)-s(\d+)\.json$", f)
        if not m or m.group(1) == "score1m" and m.group(2) == "base":
            continue
        kind, arm, step = m.group(1), m.group(2), int(m.group(3))
        d = json.load(open(f))
        if not d.get("probes"):
            continue
        p = d["probes"][0]
        by = {}
        for r, c in zip(d["rows"], p["ce"]):
            by.setdefault(r["set"], []).append(c)
        backend = "dense_probe 1-GPU logits path" if kind == "score" else "dense_probe CP8 labels path (z-loss removed)"
        for S, v in by.items():
            key = f"{os.path.basename(f)[:-5]}:{S}"
            if key in done:
                continue
            cmd = [OPS, "result", "add", "--check", "--run-id", os.path.basename(f)[:-5], "--train-run", TRAIN[arm],
                   "--experiment", "tokdrop_lr", "--task", "contradiction", "--rung", S,
                   "--metric", "heldout_answer_ce", "--value", f"{sum(v) / len(v):.5f}", "--eval-size", str(len(v)),
                   "--cot-label", "no-cot", "--eval-backend", backend,
                   "--ckpt-path", "base" if arm == "base" else f"{W}/{arm}/step{step}"]
            pv = [x for x in prov.get(os.path.basename(f)[:-5], []) if x.get("exit") == 0]
            if a.provenance:
                if not pv:
                    print(key, "no provenance (skipped)")
                    continue
                cmd += ["--git-commit", pv[-1]["commit"], "--eval-job-id", pv[-1]["job"]]
            if a.dry_run:
                print(" ".join(cmd))
                continue
            out = subprocess.run(cmd, capture_output=True, text=True)
            print(key, (out.stdout + out.stderr).strip()[-200:])
            if out.returncode == 0:
                with open(ledger, "a") as g:
                    g.write(key + "\n")


if __name__ == "__main__":
    main()
