"""Compact markdown tables for the record from analyze_context.py JSON.

    python3 research/token_routing/context_fidelity/tables.py results/ctx_f1952.json f1952 [variant,...]
"""
import json
import sys

ORDER = ("drop", "frozen", "refresh15", "mean1", "span64", "lin7", "lin7r", "stale20", "stale50", "frozen_all",
         "kvfresh", "fwdonly")
NAME = {"frozen": "frozen (stale, attention only)", "frozen_all": "stale, all layers", "kvfresh": "kvfresh (diag.)",
        "fwdonly": "fwdonly (diag. bound)"}
SEL = (("rtr", "router"), ("rnd", "random"), ("od", "attribution"))


def table(o, only=None):
    vs = [v for v in ORDER if any(f"{s}_{v}" in o["flops"] for s, _ in SEL) and (not only or v in only)]
    out = ["| variant | step cost (speedup) | " + " | ".join(
        f"{n}: exp cos Adam / raw, CE, err31, CE gap closed" for _, n in SEL) + " |",
        "|---|---|" + "---|" * len(SEL)]
    for v in vs:
        c0 = f"rtr_{v}"
        cost = o["flops"][c0]["frac"]
        cells = []
        for s, _ in SEL:
            c = f"{s}_{v}"
            if c not in o["flops"]:
                cells.append("-")
                continue
            ad, rw = o["adam"][c]["exp_cos"], o["raw"][c]["exp_cos"]
            ce = o["ce"][c]["pooled"]["routed"][0]
            e31 = o["err"][c]["pooled"]["31"]["kept"][0]
            g = o["gap_closed"][c]["ce"]
            cells.append(f"{ad[0]:.2f}±{ad[1]:.2f} / {rw[0]:.2f}, {ce:.3f}, {e31:.2f}, {100 * g:.0f}%")
        out.append(f"| {NAME.get(v, v)} | {cost:.3f} ({1 / cost:.2f}x) | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    o = json.load(open(sys.argv[1]))[sys.argv[2]]
    print(table(o, sys.argv[3].split(",") if len(sys.argv) > 3 else None))
