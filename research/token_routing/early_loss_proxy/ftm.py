"""FLOPs-to-match from FINISHED runs only (complete LR schedules): final 16k held-out CE against total training PF.

Every finished heldout_v3 run is one point (total PF, final CE): one-epoch runs on mtb16k (tags v3, w1-w3, ks, gc, sbw),
equal-FLOP runs on the 4x pool (x4), budget-matched runs (bud: 495 / 991 PF) and runs on the 16x pool (x16).
Baseline (rules of 2026-11-18: LoRA and fixed re-mixes belong to the dense baseline) = Pareto frontier of the DENSE points
(dense full fine-tune and dense LoRA, any budget). For each drop arm: the PF the baseline frontier needs for the arm's final
loss (log-linear interpolation in PF between frontier points) divided by the arm's PF. A second ratio uses dense full
fine-tune points only. "> x" = the arm's loss is below every dense point: lower bound from the largest dense budget.

    python3 ftm.py [--json hub_tab/ftm.json]        (16k CE = gen16k + rung16k of heldout_v3, eval_size 320, below 500)
"""
import contextlib, io, json, math, sys

sys.argv, argv = [sys.argv[0]], sys.argv
with contextlib.redirect_stdout(io.StringIO()):
    import ladder
P = []
for k, r in ladder.runs.items():
    if not r["done"] or len(r["curve"]) < 2: continue
    s, pf, ce, _ = r["curve"][-1]
    arm = k.split("-", 1)[1].rsplit("-sd", 1)[0]
    if "mix" in arm or arm == "denselora2e5" or not pf: continue
    kind = "dense_ft" if arm.startswith("dense") and "lora" not in arm else ("dense_lora" if arm.startswith("denselora") else "drop")
    pool = "mtb16kx16" if k.startswith("x16-") else ("mtb16kx4" if k.startswith(("x4-", "bud-", "ds-")) else "mtb16k")
    P.append(dict(run=k, arm=arm, kind=kind, pool=pool, steps=s, pf=pf, ce=ce))

# average seeds of the same (arm, pool, steps) before anything else, so the frontier is not built from the luckier seed
G = {}
for q in P: G.setdefault((q["arm"], q["pool"], q["steps"]), []).append(q)
P = [dict(run=g[0]["run"].rsplit("-sd", 1)[0] + ("-sd" + "+".join(sorted(x["run"].rsplit("-sd", 1)[1] for x in g))), arm=g[0]["arm"], kind=g[0]["kind"], pool=g[0]["pool"],
          steps=g[0]["steps"], pf=sum(x["pf"] for x in g) / len(g), ce=sum(x["ce"] for x in g) / len(g), seeds=len(g), ce_seeds=[round(x["ce"], 4) for x in g]) for g in G.values()]


def frontier(pts):  # Pareto: increasing PF, strictly decreasing CE
    out = []
    for p in sorted(pts, key=lambda p: p["pf"]):
        if not out or p["ce"] < out[-1]["ce"]: out.append(p)
    return out


def pf_for(F, ce):
    """(PF the frontier needs for this loss, exact?)"""
    if not F: return None, False
    if ce >= F[0]["ce"]: return F[0]["pf"] * 1.0, False  # shallower than the cheapest dense point: upper bound on the baseline's PF
    for a, b in zip(F, F[1:]):
        if b["ce"] <= ce <= a["ce"]:
            t = (a["ce"] - ce) / (a["ce"] - b["ce"])
            return math.exp(math.log(a["pf"]) + t * (math.log(b["pf"]) - math.log(a["pf"]))), True
    return F[-1]["pf"], False  # deeper than every dense point: lower bound


FB = frontier([p for p in P if p["kind"] != "drop"]); FD = frontier([p for p in P if p["kind"] == "dense_ft"])
print("dense baseline frontier (dense full FT + dense LoRA): " + "  ".join(f"{p['pf']:.0f} PF -> {p['ce']:.3f} ({p['run']})" for p in FB))
print("dense full fine-tune only:                            " + "  ".join(f"{p['pf']:.0f} PF -> {p['ce']:.3f}" for p in FD))
rows = []
for p in sorted(P, key=lambda p: (p["kind"] != "drop", p["ce"])):
    b, be = pf_for(FB, p["ce"]); d, de = pf_for(FD, p["ce"])
    deep_b = p["ce"] < FB[-1]["ce"] if FB else False; deep_d = p["ce"] < FD[-1]["ce"] if FD else False
    shallow_b = bool(FB) and p["ce"] >= FB[0]["ce"]
    rows.append(dict(**p, base_pf=b, x_base=(b / p["pf"]) if b else None, x_base_exact=be, x_base_lower_bound=deep_b, x_base_upper_bound=shallow_b,
                     dense_ft_pf=d, x_dense_ft=(d / p["pf"]) if d else None, x_dense_ft_exact=de, x_dense_ft_lower_bound=deep_d))
print(f"\n{'run':30s} {'pool':9s} steps    PF    CE (* = mean of 2+ seeds) | vs dense baseline family | vs dense full FT only")
for r in rows:
    f = lambda x, exact, lb: "      -" if x is None else (f"x{x:5.2f} " if exact else (f">x{x:4.2f} " if lb else f"<x{x:4.2f} "))
    print(f"{r['run'][:30]:30s} {r['pool']:9s} {r['steps']:5d} {r['pf']:5.0f} {r['ce']:.3f}{'*' if r['seeds'] > 1 else ' '}| {f(r['x_base'], r['x_base_exact'], r['x_base_lower_bound']):>10s}              | {f(r['x_dense_ft'], r['x_dense_ft_exact'], r['x_dense_ft_lower_bound'])}")
if "--json" in argv:
    json.dump(dict(frontier=FB, frontier_dense_ft=FD, runs=rows, eval_size=320), open(argv[argv.index("--json") + 1], "w"), indent=1)
