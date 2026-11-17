"""Launch support_probe.py on Beaker (jupiter, 1 GPU each, urgent, minRuntime 1h, ai2/flex2 / ai2/oe-other), one job per
(checkpoint, shard), and register each with ops. Sketches go to weka (SUPPORT_DIR/<label>/b<j>.npz); Adam factors are
the gap probe's (GAP_DIR/adam_<label>.pt). Gantry runs the PUSHED commit: commit and push first.

    python research/token_routing/grad_concentration/launch_support.py --labels f1952,recipe --num-shards 16
    python research/token_routing/grad_concentration/launch_support.py --analyze --labels f1952,recipe
"""
from __future__ import annotations

import argparse
import sys

from launch_gap import GAP_DIR, CKPTS as GAP_CKPTS, W, load, submit  # noqa: F401
from launch import CKPTS as BASE_CKPTS  # noqa: E402

SUPPORT_DIR = f"{W}/gradgap/support"
#: ~21 H100-hours per checkpoint at 512 rows (vs ~40 for the full default): splits 7 / 16, fewer k, no token x layer cells
PRESETS = {"trimmed": "--splits 7,16 --gs-ks 0.01,0.02,0.03,0.05,0.1,0.2 --gs-ks-other 0.01,0.03,0.07,0.1 "
                       "--parts q1,gs,joint,hh", "full": ""}
CKPTS = {**BASE_CKPTS, **GAP_CKPTS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="f1952,recipe")
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--shards-only", default="")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--extra", default="")
    ap.add_argument("--preset", default="trimmed", choices=sorted(PRESETS))
    ap.add_argument("--prefix", default="sup-")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--minutes", type=int, default=90)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    st = load()
    if a.analyze:
        key = f"{a.prefix}analyze-{a.labels}"
        extra = f"--dir {SUPPORT_DIR} --labels {a.labels} {a.extra}".strip()
        submit("research/token_routing/grad_concentration/analyze_support.py", extra, a.cluster, 60,
               f"gradsupp-analyze-{a.labels.replace(',', '-')}", key, st, {})
        return
    for lab in a.labels.split(","):
        ck = CKPTS[lab][0]
        only = [int(x) for x in a.shards_only.split(",")] if a.shards_only else range(a.num_shards)
        for sh in only:
            key = f"{a.prefix}{lab}-s{sh}of{a.num_shards}"
            if key in st and not a.force:
                print(f"[skip] {key} {st[key]['ex']}")
                continue
            extra = (f"--ckpt {ck} --label {lab} --rows {a.rows} --num-shards {a.num_shards} --shard {sh} "
                     f"--save-dir {SUPPORT_DIR} --adam-factors {GAP_DIR}/adam_{lab}.pt {PRESETS[a.preset]} {a.extra}").strip()
            submit("research/token_routing/grad_concentration/support_probe.py", extra, a.cluster, a.minutes,
                   f"gradsupp-{lab}-s{sh}of{a.num_shards}", key, st, {"label": lab, "shard": sh, "num_shards": a.num_shards})


if __name__ == "__main__":
    sys.exit(main())
