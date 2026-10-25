"""Peak GPU memory per training run from its Beaker log (GPUMemoryMonitorCallback ``gpu_memory/*`` lines, rank 0):
the max of every ``gpu_memory/<key>=<value>`` key over the run.

    python3 research/token_routing/mt11_64k/peak_mem.py <run> [<run> ...]
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r"gpu_memory/(GPU (?:active|reserved) mem \((?:GiB|%)\))=([0-9.]+)")


def main():
    st = json.load(open(os.path.join(HERE, "state.json")))
    for run in sys.argv[1:]:
        r = st["runs"][run]
        path = os.path.join(HERE, "results", "logs", f"{r['ex']}.log")
        txt = open(path, errors="replace").read() if os.path.exists(path) else subprocess.run(
            [os.path.expanduser("~/.local/bin/beaker"), "experiment", "logs", r["ex"]], capture_output=True,
            text=True, timeout=900).stdout
        peak = {}
        for k, v in LINE.findall(txt):
            peak[k.strip()] = max(peak.get(k.strip(), 0.0), float(v.replace(",", "")))
        print(f"{run} ({r.get('gpus', '?')} GPUs): " + ", ".join(f"{k}={v:g}" for k, v in sorted(peak.items())))


if __name__ == "__main__":
    main()
