"""Table + curves from cpt_oracle_budget.py JSONs (small; runs anywhere with numpy + matplotlib).

    python research/token_routing/grad_concentration/cpt_oracle_table.py --in-dir <dir with <label>.json> \
        --labels base_32k,cpt_32k,base_64k,cpt_64k --out-dir research/token_routing/grad_concentration/gradmaps_cpt

The kept share at which each rule's attribution cosine first reaches the threshold is linearly interpolated between
the two grid points that bracket the crossing (the fraction grid is log-spaced, so it is coarse near 1).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

RULES = ("global", "row", "blk512", "random")
RULE_NAMES = {"global": "Global oracle", "row": "Per-row quota (= per document window)",
              "blk512": "Per-512-token-block quota", "random": "Random"}


def crossing(kept, cos, thr):
    kept, cos = np.asarray(kept), np.asarray(cos)
    o = np.argsort(kept, kind="stable")
    kept, cos = kept[o], cos[o]
    ok = np.flatnonzero(cos >= thr)
    if ok.size == 0:
        return None
    j = ok[0]
    if j == 0:
        return float(kept[0])
    k0, k1, c0, c1 = kept[j - 1], kept[j], cos[j - 1], cos[j]
    return float(k0 + (thr - c0) / max(c1 - c0, 1e-12) * (k1 - k0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    labs = a.labels.split(",")
    R = {lab: json.load(open(os.path.join(a.in_dir, lab + ".json"))) for lab in labs}
    out = {}
    for lab, r in R.items():
        e = {"step_rows": r["step_rows"], "T": r["T"],
             "sketch_check_hi": [c["hi_ratio"] for c in r["sketch_check"]],
             "sketch_check_lo": [c["lo_ratio"] for c in r["sketch_check"]], "budget": {}}
        for scope in ("dn", "wm"):
            for rule in RULES:
                c = r["curves"][f"{scope}_{rule}"]
                for thr in (0.9, 0.95, 0.99):
                    e["budget"][f"{scope}_{rule}_{thr}"] = crossing(c["kept"], c["cos"], thr)
        out[lab] = e
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, "oracle_budgets.json"), "w"), indent=1)
    # markdown table: per length, base / cpt at 0.95 and 0.9, change base -> cpt at 0.95
    lines = []
    for scope, sname in (("dn", "blocks 7-31 (split at block 7; SFT table convention)"), ("wm", "whole model (split 0)")):
        lines.append(f"\n### Scope: {sname}\n")
        lens = sorted({lab.split("_")[1] for lab in labs}, key=lambda s: int(s[:-1]))
        hdr = "| selection rule | " + " | ".join(f"{L} base, cos 0.95 | {L} dense CPT, cos 0.95 | {L} dense CPT, cos 0.9 | {L} base -> CPT" for L in lens) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (1 + 4 * len(lens)))
        for rule in RULES:
            cells = [RULE_NAMES[rule]]
            for L in lens:
                b = out.get(f"base_{L}", {}).get("budget", {})
                c = out.get(f"cpt_{L}", {}).get("budget", {})
                vb, vc, vc9 = b.get(f"{scope}_{rule}_0.95"), c.get(f"{scope}_{rule}_0.95"), c.get(f"{scope}_{rule}_0.9")
                f = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"  # noqa: E731
                cells += [f(vb), f(vc), f(vc9), "n/a" if not (vb and vc) else f"{vc / vb:.1f}x more"]
            lines.append("| " + " | ".join(cells) + " |")
    txt = "\n".join(lines)
    open(os.path.join(a.out_dir, "oracle_table.md"), "w").write(txt + "\n")
    print(txt)
    for lab, e in out.items():
        print(lab, "step rows", e["step_rows"], "sketch/exact norm^2 blocks 7-31 range",
              f"{min(e['sketch_check_hi']):.3f}-{max(e['sketch_check_hi']):.3f}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    col = {"global": "#c44e52", "row": "#4c72b0", "blk512": "#55a868", "random": "#8c8c8c"}
    for scope in ("dn", "wm"):
        fig, axs = plt.subplots(1, len(labs), figsize=(4.2 * len(labs), 3.6), sharey=True)
        for ax, lab in zip(np.atleast_1d(axs), labs):
            r = R[lab]
            for rule in RULES:
                c = r["curves"][f"{scope}_{rule}"]
                k = np.asarray(c["kept"])
                o = np.argsort(k)
                ax.plot(k[o], np.asarray(c["cos"])[o], color=col[rule], label=RULE_NAMES[rule], lw=1.3)
            for t in (0.9, 0.95):
                ax.axhline(t, color="k", lw=0.5, ls=":")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.01)
            ax.set_xlabel("share of tokens kept")
            ax.set_title(f"{lab}: one step = {r['step_rows']} rows x {r['T'] // 1024}k", fontsize=8.5)
        np.atleast_1d(axs)[0].set_ylabel("attribution cos(kept, full step gradient)")
        np.atleast_1d(axs)[0].legend(fontsize=7)
        fig.suptitle(f"CPT oracle budgets, scope {'blocks 7-31' if scope == 'dn' else 'whole model'} "
                     f"(attribution: kept tokens' terms under the dense forward, not a routed gradient)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, f"sum_oracle_cos_{scope}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
