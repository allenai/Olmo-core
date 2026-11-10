"""Side-by-side of two analyze_gap.py outputs (e.g. dense f1952 vs the routed-trained recipe checkpoint) at chosen selections.

    python3 research/token_routing/grad_concentration/compare_gap.py results/gap_f1952.json:f1952 results/gap_recipe.json:recipe
"""
import json
import sys

SELS = ("og_recipe", "rnd_recipe", "rtr_recipe", "og_L16_k0.07", "rnd_L16_k0.07", "rtr_L16_k0.07")


def f(x):
    if isinstance(x, list):
        return f"{x[0]:.3f}±{x[1]:.3f}"
    return f"{x:.3f}"


def main():
    runs = []
    for arg in sys.argv[1:]:
        p, lab = arg.split(":")
        runs.append((lab, json.load(open(p))[lab]))
    hdr = ("sel", "ckpt", "routed cos raw", "routed cos adam", "exp cos raw", "exp cos adam", "adam proj E[R].E[D]",
           "pair cos R-R raw (D-D)", "bias share R / D raw", "routed CE contra/nq/outlier/oolong (dense)")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for s in SELS:
        for lab, d in runs:
            r, a = d["raw"][s]["whole"], d["adam"][s]["whole"]
            ce = d["ce"][s]
            ces = "/".join(f"{ce[t]['ce_routed'][0]:.3f}" for t in ("contradiction", "nq", "outlier", "oolong"))
            ced = "/".join(f"{ce[t]['ce_dense'][0]:.3f}" for t in ("contradiction", "nq", "outlier", "oolong"))
            print(f"| {s} | {lab} | {f(r['routed_cos'])} | {f(a['routed_cos'])} | {f(r['exp_cos_routed'])} | "
                  f"{f(a['exp_cos_routed'])} | {f(a['exp_proj_routed'])} | {f(r['pair_cos_routed'])} ({f(r['pair_cos_dense'])})"
                  f" | {f(r['bias_share_routed'])} / {f(r['bias_share_dense'])} | {ces} ({ced}) |")
    print()
    print("attribution cos / dense cos to E (raw, adam):")
    for lab, d in runs:
        print(lab, {s: (f(d["raw"][s]["whole"]["attr_cos"]), f(d["raw"][s]["whole"]["dense_cos_E"]),
                        f(d["adam"][s]["whole"]["dense_cos_E"])) for s in SELS[:3]})
    print("per type (share of ||D-R||^2) at the recipe shape:")
    for lab, d in runs:
        for s in SELS[:3]:
            pt = d["per_type"][s]
            print(lab, s, {k: f(v["share_of_lost_norm2"]) for k, v in pt.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
