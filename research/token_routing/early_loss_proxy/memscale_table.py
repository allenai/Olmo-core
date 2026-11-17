"""Memory table for the memscale sweep: per run, the forward+backward transient peak (GiB above the pre-step
baseline) from the per-block memory trace (dense_probe --probe-memtrace), its location, the stored-activation
level at the end of the forward, and the baseline.

    python3 memscale_table.py /net/mooney/data/prasann/early_loss_proxy/results/memscale [more dirs]
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys


def summarize(d):
    tr = d.get("memtrace") or {}
    out = []
    for st, rows in sorted(tr.items()):
        if not rows:
            continue
        peak = max(r[4] for r in rows)
        # where the peak was first reached
        at = next(f"{r[0]}{r[1]}" for r in rows if r[4] >= peak - 1e-6)
        prev = 0.0
        for r in rows:  # the peak is reached between the previous event and this one
            if r[4] >= peak - 1e-6:
                break
            prev = r[4]
        fwd = [r for r in rows if r[0] in ("f", "f0")]
        stored = fwd[-1][3] if fwd else None
        head = next((r for r in rows if r[0] == "head"), None)
        widths = sorted({r[2] for r in fwd})
        out.append({"step": int(st), "peak": peak, "peak_at": at, "stored_end_fwd": stored,
                    "after_head": head[3] if head else None, "widths": widths})
    return out


def main():
    rows = []
    for dr in sys.argv[1:]:
        for f in sorted(glob.glob(os.path.join(dr, "*.json"))):
            d = json.load(open(f))
            name = os.path.basename(f).replace(".json", "")
            log = f.replace(".json", ".log")
            base = None
            if os.path.exists(log):
                m = re.findall(r"\[memtrace\] step \d+ base ([\d.]+) GiB", open(log, errors="ignore").read())
                base = float(m[-1]) if m else None
                oom = "OutOfMemoryError" in open(log, errors="ignore").read()
            s = summarize(d)
            last = s[-1] if s else {}
            rows.append((name, base, last, oom))
    print(f"{'run':38s} {'base':>6s} {'peak':>7s} {'at':>6s} {'stored':>7s} {'head':>6s} widths")
    for name, base, s, oom in rows:
        if not s:
            print(f"{name:38s} {'':>6s} {'OOM' if oom else 'n/a':>7s}")
            continue
        print(f"{name:38s} {base or 0:6.2f} {s['peak']:7.2f} {s['peak_at']:>6s} {s['stored_end_fwd']:7.2f} "
              f"{(s['after_head'] or 0):6.2f} {s['widths'][:4]}")


if __name__ == "__main__":
    main()
