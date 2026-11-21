"""Mean logged train/CE over a step window from a trainer log (shortcut indicator: the arm's CE on its own grouped rows
vs gdd256's on the same rows).  python3 train_ce.py <log> <lo> <hi>  (z-loss included, as logged; rows with labels)"""
import re
import sys

step, vals = None, []
lo, hi = int(sys.argv[2]), int(sys.argv[3])
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r"\[step=(\d+)/", line)
    if m:
        step = int(m.group(1))
        continue
    m = re.search(r"train/CE loss=([0-9.eE+-]+)", line)
    if m and step is not None and lo <= step <= hi:
        vals.append(float(m.group(1)))
print(f"{len(vals)} steps, mean train/CE {sum(vals) / max(1, len(vals)):.4f}")
