"""Markdown tables + one plot from analyze_tokgrad.py's summary JSON.

    python table_tokgrad.py summary.json --png tokgrad_golddrop.png > tables.md
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("summary")
ap.add_argument("--png", default="")
a = ap.parse_args()
S = json.load(open(a.summary))


def f(v, pct=True, nd=1):
    if v is None:
        return "-"
    m, se, n = v
    return f"{100 * m:.{nd}f} ± {100 * se:.{nd}f}" if pct else f"{m:.{nd}g} ± {se:.1g}"


def g(d, k, **kw):
    return f(d.get(k), **kw)


print("| condition | rows x draws | CE A / B / C (per answer token) | A: body tok for 50/90/99% | B: of present body | C: of body entering block 7 | C: of original body | FLOPs B/A | FLOPs C/A |")
print("|---|---|---|---|---|---|---|---|---|")
for c, d in S.items():
    m = d["_meta"]
    ce = " / ".join(f"{d[k][0]:.2g}" if k in d else "-" for k in ("ceA", "ceB", "ceC"))
    tri = lambda p: " / ".join(f"{100 * d[f'{p}|n{q}'][0]:.2f}" if f"{p}|n{q}" in d else "-" for q in (0.5, 0.9, 0.99))  # noqa: E731
    print(f"| {c} | {m['rows']} x {m['draws']} | {ce} | {tri('A|body')} | {tri('B|body_present')} | {tri('C|body_present7')} | "
          f"{tri('C|body_of_orig')} | {g(d, 'flopB/A')} | {g(d, 'flopC/A', nd=2)} |")
print()
print("SE (over rows) of the 90% points: " + "; ".join(
    f"{c}: A {g(d, 'A|body|n0.9')}, B {g(d, 'B|body_present|n0.9')}, C {g(d, 'C|body_present7|n0.9')}" for c, d in S.items()))
print()
print("| condition | gold drop keeps: body tokens | A body mass on kept | oracle mass, same #tokens | A body mass on gold docs (gold = % of body tok) | router @7: body tok kept (of post-drop) | A body mass 7-15 on kept7 | B body mass 7-15 on kept7 (oracle) | router @16: kept (of post-drop) | B body mass 16-31 on kept16 (oracle) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c, d in S.items():
    print(f"| {c} | {g(d, 'drop|tok_share_body')} | {g(d, 'drop|A_body_mass_kept')} | {g(d, 'drop|A_body_mass_oracle_same_tokens')} | "
          f"{g(d, 'A|gold_share_of_body')} ({100 * d['A|gold_tok_share_of_body'][0]:.1f}) | "
          f"{g(d, 'C|tok7_share_of_dropped_body')} | {g(d, 'router|A_body_mass7-15_on_kept7')} | "
          f"{g(d, 'router|B_body_mass7-15_on_kept7')} ({g(d, 'router|B_body_mass7-15_oracle_same_tokens')}) | "
          f"{g(d, 'C|tok16_share_of_dropped_body')} | {g(d, 'router|B_body_mass16-31_on_kept16')} ({g(d, 'router|B_body_mass16-31_oracle_same_tokens')}) |")
print()
print("Body share of the whole-row (A) gradient mass: " + "; ".join(f"{c} {g(d, 'A|body_share')}" for c, d in S.items()))
if a.png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cs = list(S)
    x = np.arange(len(cs))
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(cs)), 4))
    for j, (k, lab) in enumerate([("A|body|n0.9", "A whole row (of body)"), ("B|body_of_orig|n0.9", "B after drop (of orig body)"),
                                  ("C|body_of_orig|n0.9", "C drop+router (of orig body)"), ("flopC/A", "FLOPs C/A")]):
        v = [S[c].get(k, [np.nan, np.nan])[:2] for c in cs]
        ax.bar(x + (j - 1.5) * 0.2, [100 * a_[0] for a_ in v], 0.2, yerr=[100 * a_[1] for a_ in v], label=lab)
    ax.set_xticks(x, [c.replace("-", "\n", 1) for c in cs], fontsize=8)
    ax.set_ylabel("% of original body tokens (FLOPs: % of dense)")
    ax.set_title("Tokens holding 90% of body gradient mass vs FLOP ratio")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.png, dpi=110)
