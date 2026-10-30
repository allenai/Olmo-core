"""KV-only probe summary (numpy only; login-node safe): per (ckpt, setting, rule), each context mode side by side.

    python kv_probe_table.py --merged merged.json [--view trained] [--md kv_probe_table.md]

merged.json = grad_fidelity2_merge.py output over the gfkv job results. Columns per mode: median single-draw cos,
8-batch (128-row) cos, median dCE (truncated minus full answer CE). "recover" = how much of the gap between the
plain split and the plain block-7 split (same rule, same stage structure) the mode closes, on 8-batch cos and on dCE.
"""
import argparse
import json

MODES = ("plain", "kv", "kvdet", "kva", "kvadet")
REF7 = {"1st_1_10": "1st_7_10", "1st_3_10": "1st_7_10", "2st_1_16": "2st_7_16", "2st_3_16": "2st_7_16"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--view", default="trained")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    S = {}
    for s in json.load(open(a.merged))["summary"]:
        base, _, mode = s["setting"].partition("@")
        S[(s["tag"], base, s["rule"], mode or "plain")] = s
    v = a.view
    hdr = "| ckpt | setting | rule | rows | " + " | ".join(f"{m} cos1 / 8b / dCE" for m in MODES) + \
          " | kv recover 8b / dCE | kva recover 8b / dCE |"
    lines = [hdr, "|" + "---|" * (4 + len(MODES) + 2)]
    keys = sorted({k[:3] for k in S})
    for tag, base, rule in keys:
        cells = []
        for m in MODES:
            s = S.get((tag, base, rule, m))
            cells.append("—" if s is None else
                         f"{s.get(f'{v}_cos1', float('nan')):+.3f} / {s[f'{v}_accum_cos']:+.3f} / {s['dce']:+.3f}")
        rec = []
        ref = S.get((tag, REF7.get(base, ""), rule, "plain"))
        p = S.get((tag, base, rule, "plain"))
        for m in ("kv", "kva"):
            s = S.get((tag, base, rule, m))
            if ref is None or p is None or s is None:
                rec.append("—")
                continue
            g8 = ref[f"{v}_accum_cos"] - p[f"{v}_accum_cos"]
            gd = p["dce"] - ref["dce"]
            r8 = (s[f"{v}_accum_cos"] - p[f"{v}_accum_cos"]) / g8 if abs(g8) > 1e-6 else float("nan")
            rd = (p["dce"] - s["dce"]) / gd if abs(gd) > 1e-6 else float("nan")
            rec.append(f"{r8:+.0%} / {rd:+.0%}")
        n = (p or {}).get("eval_size", 0)
        lines.append(f"| {tag} | {base} | {rule} | {n} | " + " | ".join(cells) + " | " + " | ".join(rec) + " |")
    md = "\n".join(lines)
    print(md)
    if a.md:
        open(a.md, "w").write(f"### view: {v}\n\n{md}\n")


if __name__ == "__main__":
    main()
