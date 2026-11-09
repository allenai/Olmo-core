"""Tables + plots for grad_tokens.py outputs (per-layer decomposition of the dropped-context gradient).

    python3 research/token_routing/learned_router/gradprobe/gtok_report.py results/json/gtok/gtok_*.json --out-dir results

Per model (family) and drop rule: per-layer medians over rows and tasks of ||D||/||G|| (share lost with the dropped
tokens), cos(K, G) (kept tokens' share of the full direction), cos(K', K) (how much the kept tokens' own contribution
moved), cos(K', G), and the forward / backward split cos(K_fwd, K), cos(K_bwd, K); category mass shares; the
kept-body forward-drift profile vs distance to the nearest dropped token; the decoded top tokens.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

KEYS = [("D_over_G", "‖D‖/‖G‖  (dropped tokens' share)"), ("cos_K_G", "cos(K, G)  kept part vs full"),
        ("cos_Kd_K", "cos(K', K)  kept tokens, dropped vs full ctx"), ("cos_Kd_G", "cos(K', G)  = param-grad cosine"),
        ("cos_Kfwd_K", "cos(K_fwd, K)  activations changed only"), ("cos_Kbwd_K", "cos(K_bwd, K)  signal changed only")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
FAM = {"qwen3_5": "Qwen3.5-4B (GDN 3:1)", "qwen3": "Qwen3-4B (full attn)"}


def model_name(d):
    ck = d["argv"][d["argv"].index("--ckpt") + 1].rstrip("/")
    return f"{os.path.basename(ck)} [{FAM.get(d.get('family'), d.get('family'))}]"


def pick(points, rule, rho):
    for q in points:
        if q["rule"] == rule and (rule == "bar" or abs(q["rho"] - rho) < 1e-6):
            return q
    return None


def collect(files):
    by = defaultdict(list)  # model -> list of (task, data)
    for f in files:
        d = json.load(open(f))
        by[model_name(d)].append((d["task"], d))
    return by


def per_layer(datas, rule, rho, lin, key):
    """(n_layers,) median over rows (all tasks) of layer metric; also the mixer type per layer."""
    vals = defaultdict(list)
    mix = None
    for _, d in datas:
        mix = d["mixers"]
        for r in d["rows"]:
            q = pick(r["points"], rule, rho)
            if q is None:
                continue
            for li, L in enumerate(q["layers"]):
                if lin in L and key in L[lin]:
                    vals[li].append(L[lin][key])
    Ls = sorted(vals)
    return np.array(Ls), np.array([np.median(vals[l]) for l in Ls]), mix, (len(next(iter(vals.values()))) if vals else 0)


def table(by, rule, rho, lin):
    s = [f"#### {lin}, rule {rule}" + ("" if rule == "bar" else f" ρ={rho}"),
         "| model | depth | " + " | ".join(k for k, _ in KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
    for m, datas in sorted(by.items()):
        cols = {k: per_layer(datas, rule, rho, lin, k) for k, _ in KEYS}
        Ls = cols[KEYS[0][0]][0]
        if not len(Ls):
            continue
        n = len(Ls)
        for nm, sl in (("first quarter", slice(0, n // 4)), ("middle half", slice(n // 4, 3 * n // 4)), ("last quarter", slice(3 * n // 4, n))):
            s.append(f"| {m} (⚠ eval_size={cols[KEYS[0][0]][3]}) | {nm} | " + " | ".join(f"{np.median(cols[k][1][sl]):+.2f}" for k, _ in KEYS) + " |")
    return "\n".join(s)


def mass_table(by, rule, rho, lin, layer_frac=(0.0, 0.5, 1.0)):
    s = [f"#### contribution mass share by token category ({lin}, {rule}" + ("" if rule == "bar" else f" ρ={rho}") + "): full ctx → dropped ctx",
         "| model | layer | prefix | marker | gold body | other body | DROPPED (all / gold) | suffix (question) | answer |", "|---|---|---|---|---|---|---|---|---|"]
    for m, datas in sorted(by.items()):
        nl = len(datas[0][1]["mixers"])
        for fr in layer_frac:
            li = min(nl - 1, int(round(fr * (nl - 1))))
            acc_f, acc_d = defaultdict(list), defaultdict(list)
            for _, d in datas:
                for r in d["rows"]:
                    q = pick(r["points"], rule, rho)
                    if q is None or lin not in q["layers"][li]:
                        continue
                    E = q["layers"][li][lin]
                    for k, v in E["mass_full"].items():
                        acc_f[k].append(v)
                    for k, v in E["mass_drop"].items():
                        acc_d[k].append(v)
            f = {k: np.mean(v) for k, v in acc_f.items()}
            g = {k: np.mean(v) for k, v in acc_d.items()}

            def c(k):
                return f"{f.get(k, 0):.2f}→{g.get(k, 0):.2f}"
            s.append(f"| {m} | L{li} | {c('prefix')} | {c('marker')} | {c('gold_body')} | {c('body')} | "
                     f"{f.get('dropped', 0):.2f} / {f.get('dropped_gold', 0):.2f} | {c('suffix')} | {c('answer')} |")
    return "\n".join(s)


def dist_table(by, rule, rho):
    s = [f"#### kept body tokens vs distance to the nearest dropped token (residual stream, {rule}" + ("" if rule == "bar" else f" ρ={rho}") +
         "): median cos(h_full, h_drop) \\| cos(dCE/dh) at 1/4, 1/2, 3/4 depth", "| model | layer | " +
         " | ".join(["1", "2-4", "5-16", "17-64", "65-256", "257+"]) + " |", "|---|---|" + "---|" * 6]
    bins = ["1-1", "2-4", "5-16", "17-64", "65-256", "257-1000000000"]
    for m, datas in sorted(by.items()):
        nl = len(datas[0][1]["mixers"])
        for li in (nl // 4, nl // 2, 3 * nl // 4):
            acc = defaultdict(lambda: ([], []))
            for _, d in datas:
                for r in d["rows"]:
                    q = pick(r["points"], rule, rho)
                    if q is None:
                        continue
                    for b, v in q["layers"][li].get("resid", {}).get("by_dist", {}).items():
                        acc[b][0].append(v["cos_x_med"])
                        acc[b][1].append(v["cos_d_med"])
            s.append(f"| {m} | L{li} | " + " | ".join(f"{np.median(acc[b][0]):.2f} \\| {np.median(acc[b][1]):.2f}" if acc[b][0] else "–" for b in bins) + " |")
    return "\n".join(s)


def tops(by, rule, rho, n=8):
    s = []
    for m, datas in sorted(by.items()):
        for task, d in datas[:3]:
            r = d["rows"][0]
            q = pick(r["points"], rule, rho)
            if q is None or not q.get("tops"):
                continue
            for li, t in sorted(q["tops"].items(), key=lambda kv: int(kv[0])):
                dt = ", ".join(f"{x['piece']!r}({x['cat'][:4]})" for x in t["dropped_top"][:n] if x["v"] > 0) or "(none: no gradient reaches context tokens here)"
                kt = ", ".join(f"{x['piece']!r}({x['cat'][:4]},d{x['dist']})" for x in t["kept_top_lost"][:n])
                s.append(f"- {m} · {task} row0 · L{li}: top dropped: {dt}\n  kept, most changed: {kt}")
    return "\n".join(s)


def plots(by, out_dir, rule, rho, lin):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(by)
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 3.8), squeeze=False, sharey=True)
    for i, m in enumerate(models):
        ax = axes[0][i]
        mix = by[m][0][1]["mixers"]
        for L, ty in enumerate(mix):
            if "GatedDeltaNet" not in ty and any("GatedDeltaNet" in t for t in mix):
                ax.axvspan(L - 0.5, L + 0.5, color="#e5e4df", lw=0, zorder=0)
        n = 0
        for (k, lab), col in zip(KEYS, COLORS):
            Ls, v, _, n = per_layer(by[m], rule, rho, lin, k)
            if len(Ls):
                ax.plot(Ls, v, "-", lw=2, color=col, label=lab)
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_title(f"{m}\n{lin}, {rule}" + ("" if rule == "bar" else f" ρ={rho}") + f" (eval_size={n}; shaded = full-attn layers)", fontsize=8.5)
        ax.set_xlabel("layer", fontsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("median over rows", fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fn = os.path.join(out_dir, f"gtok_layers_{lin.split('.')[-1]}_{rule}{'' if rule == 'bar' else rho}.png")
    fig.savefig(fn, dpi=130)
    plt.close(fig)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    by = collect(a.paths)
    os.makedirs(a.out_dir, exist_ok=True)
    for rule, rho in (("a0.5_abs", 0.1), ("rand", 0.1), ("bar", None), ("a0.5_abs", 0.5), ("rand", 0.5)):
        for lin in ("feed_forward.w2", "attention.w_out"):
            print(table(by, rule, rho, lin), "\n")
            print(plots(by, a.out_dir, rule, rho, lin))
        print(mass_table(by, rule, rho, "feed_forward.w2"), "\n")
        print(dist_table(by, rule, rho), "\n")
    print(tops(by, "a0.5_abs", 0.1))


if __name__ == "__main__":
    main()
