"""Tables + curves for the span-loss oracle runs (cpt_oracle_budget.py / cpt_oracle_cells.py --span).

    python research/token_routing/grad_concentration/cpt_span_table.py --in-dir <dir of tok_*/cell_* JSONs> --out-dir <gradmaps_cpt>

Labels are {tok,cell}_{base,cpt}_<len>_s<span>. Budgets = the kept share of CONTEXT tokens (or context cells) at which
the attribution cosine first reaches the threshold, linearly interpolated between grid points:
  context-only  cos(sum of kept context terms, sum of all context terms)
  total         cos(span terms + kept context terms, full span-loss gradient)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from cpt_oracle_table import crossing

TOK_RULES = ("global", "row", "blk512", "recent", "random")
CELL_RULES = ("global", "row", "blockrow", "random")
NAMES = {"global": "Global oracle", "row": "Per-row quota", "blk512": "Per-512-token-block quota",
         "recent": "Most recent context (local window)", "random": "Random", "blockrow": "Per-(row, block) token quota"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scope", default="dn")
    a = ap.parse_args()
    R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.in_dir, "*.json")))}
    settings = sorted({lab.split("_", 2)[2] for lab in R}, key=lambda s: (int(s.split("_")[0][:-1]), int(s.split("_s")[1])))
    out = {}
    for lab, r in R.items():
        e = {"step_rows": r["step_rows"], "T": r["T"], "span": r["span"],
             "ctx_energy_share": r.get(f"{a.scope}_ctx_energy_share"), "budget": {},
             "sketch_check_hi": [c["hi_ratio"] for c in r["sketch_check"]]}
        for key, c in r["curves"].items():
            if not key.startswith(a.scope + "_"):
                continue
            rule = key.split("_", 1)[1]
            for thr in (0.9, 0.95):
                e["budget"][f"ctx_{rule}_{thr}"] = crossing(c["kept"], c["cos_ctx"], thr)
                e["budget"][f"tot_{rule}_{thr}"] = crossing(c["kept"], c["cos"], thr)
        out[lab] = e
    json.dump(out, open(os.path.join(a.out_dir, f"span_oracle_budgets_{a.scope}.json"), "w"), indent=1)
    f = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    lines = [f"Scope {'blocks 7-31' if a.scope == 'dn' else 'whole model'}. Share of CONTEXT tokens / cells (everything "
             f"before the span) needed; the span's own tokens are always kept.\n"]
    for kind, nm in (("ctx", "context-only cosine, 0.95"), ("tot", "total cosine (span + kept context), 0.95")):
        lines.append(f"\n### {nm}\n")
        hdr = ["selection rule"] + [f"{s.replace('_s', ', span ')} {c}" for s in settings for c in ("base", "dense CPT")]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        for unit, rules in (("tok", TOK_RULES), ("cell", CELL_RULES)):
            for rule in rules:
                cells = [f"{NAMES[rule]} ({'tokens' if unit == 'tok' else 'cells'})"]
                for s in settings:
                    for c in ("base", "cpt"):
                        cells.append(f(out.get(f"{unit}_{c}_{s}", {}).get("budget", {}).get(f"{kind}_{rule}_0.95")))
                lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n### Share of the span loss's gradient (sketched norm^2 of the summed terms) on context tokens\n")
    lines.append("| | " + " | ".join(f"{s.replace('_s', ', span ')} {c}" for s in settings for c in ("base", "dense CPT")) + " |")
    lines.append("|" + "---|" * (1 + 2 * len(settings)))
    lines.append("| context share | " + " | ".join(
        "n/a" if out.get(f"tok_{c}_{s}") is None else f"{out[f'tok_{c}_{s}']['ctx_energy_share']:.2f}"
        for s in settings for c in ("base", "cpt")) + " |")
    txt = "\n".join(lines)
    open(os.path.join(a.out_dir, f"span_oracle_table_{a.scope}.md"), "w").write(txt + "\n")
    print(txt)
    for lab, e in out.items():
        print(lab, "sketch/exact", f"{min(e['sketch_check_hi']):.3f}-{max(e['sketch_check_hi']):.3f}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    col = {"global": "#c44e52", "row": "#4c72b0", "blk512": "#55a868", "recent": "#dd8452", "random": "#8c8c8c",
           "blockrow": "#8172b2"}
    for unit in ("tok", "cell"):
        labs = [f"{unit}_{c}_{s}" for s in settings for c in ("base", "cpt") if f"{unit}_{c}_{s}" in R]
        if not labs:
            continue
        fig, axs = plt.subplots(2, len(labs) // 2, figsize=(4.4 * len(labs) // 2, 7.2), sharey=True, squeeze=False)
        for i, lab in enumerate(labs):
            ax = axs[i % 2, i // 2]
            r = R[lab]
            for key, c in r["curves"].items():
                if not key.startswith(a.scope + "_"):
                    continue
                rule = key.split("_", 1)[1]
                k = np.asarray(c["kept"])
                o = np.argsort(k)
                ax.plot(k[o], np.asarray(c["cos_ctx"])[o], color=col[rule], lw=1.3, label=NAMES[rule])
            ax.axhline(0.95, color="k", lw=0.5, ls=":")
            ax.axhline(0.9, color="k", lw=0.5, ls=":")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 1.01)
            ax.set_title(f"{lab.split('_', 1)[1].replace('_', ' ')}: {r['step_rows']} rows x {r['T'] // 1024}k, "
                         f"loss on last {r['span']:,}", fontsize=8)
            ax.set_xlabel(f"share of context {'tokens' if unit == 'tok' else 'cells'} kept", fontsize=8)
        axs[0, 0].set_ylabel("cos(kept context terms, all context terms)", fontsize=8)
        axs[1, 0].set_ylabel("cos(kept context terms, all context terms)", fontsize=8)
        axs[0, 0].legend(fontsize=6.5)
        fig.suptitle(f"Long-range credit: which context {'tokens' if unit == 'tok' else 'cells'} carry a late span's "
                     f"gradient (attribution, one dense pass; scope {'blocks 7-31' if a.scope == 'dn' else 'whole model'})",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, f"sum_span_oracle_{unit}_{a.scope}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
