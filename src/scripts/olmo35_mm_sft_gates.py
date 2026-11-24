"""Read the multimodal SFT pre-run gates from a gate run's metrics and batch hashes.

    python olmo35_mm_sft_gates.py steps RUN_DIR            # gate (c): steps 1-6 vs the text trainer
    python olmo35_mm_sft_gates.py resume RUN_DIR ORIG_HASHES RESUMED_HASHES ORIG_METRICS
                                                            # gate (d): steps 11-20 equal

Gate (c) reference: open-instruct h182-gpu@c7398092c, the text-only GPU SFT of the 64K base, data
seed 34521, two step-0 runs on 2 x 8 B300 (W&B ai2-llm/open_instruct_internal c8wyjcox, R1, and
0yxnp9m4, R2), train CE at full precision.
"""

import glob
import json
import os
import sys

R1 = [1.1583552360534668, 1.1905646324157715, 1.1569013595581055, 1.153390645980835, 0.9810333847999573, 1.1549816131591797]
R2 = [1.1583552360534668, 1.190606951713562, 1.1571483612060547, 1.1544365882873535, 0.9821919202804565, 1.155256748199463]
HF_STEP1 = 1.15827  # open-instruct h161_refce.py on seed 34521's step-1 batch (H182 G2 receipt)


def ce(run_dir, step):
    with open(os.path.join(run_dir, f"metrics_step{step}.json")) as f:
        return json.load(f)["train/CE loss"]


def hashes(d):
    out = {}
    for path in glob.glob(os.path.join(d, "rank*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            out[(r["step"], r["rank"])] = r["hashes"]["all"]
    return out


def steps(run_dir):
    ours = [ce(run_dir, s) for s in range(1, 7)]
    f = max(abs(a - b) for a, b in zip(R1, R2))
    d1 = [abs(a - b) for a, b in zip(ours, R1)]
    d2 = [abs(a - b) for a, b in zip(ours, R2)]
    res = {
        "ours": ours,
        "minus_R1": d1,
        "minus_R2": d2,
        "max_minus_R1": max(d1),
        "R1_R2_spread_F": f,
        "step1_minus_hf": abs(ours[0] - HF_STEP1),
        "pass_1e-3": max(d1) <= 1e-3,
        "pass_max(1e-3,2F)": max(d1) <= max(1e-3, 2 * f),
    }
    print(json.dumps(res, indent=1))


def resume(run_dir, orig_hashes, resumed_hashes, orig_metrics):
    a, b = hashes(orig_hashes), hashes(resumed_hashes)
    keys = sorted(k for k in b if 11 <= k[0] <= 20)
    equal = sum(a.get(k) == b[k] for k in keys)
    ce_gaps = {s: abs(ce(orig_metrics, s) - ce(run_dir, s)) for s in range(11, 21)}
    print(
        json.dumps(
            {
                "hash_keys_compared": len(keys),
                "hash_equal": equal,
                "steps_resumed": sorted({k[0] for k in keys}),
                "ce_abs_gap_by_step": ce_gaps,
                "max_ce_gap": max(ce_gaps.values()),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    {"steps": steps, "resume": resume}[sys.argv[1]](*sys.argv[2:])
