"""Summarize probe.py outputs: k (share of tokens) needed for cos 0.7 / 0.9 / 0.95 to the full batch update, per stage,
view, scope and depth; energy view; recipe fidelity; and the first-order FLOP bound on token dropping.

    python research/token_routing/grad_concentration/analyze.py [--results results] [--out summary.json]

FLOP model (per token, per block, linear in tokens): a single split at L keeping a fraction q of tokens past L costs
L/32 + (1 - L/32) q of dense. In the body view q = (non-body share) + (body share) * k_body, since non-body tokens are
always kept. Attention's quadratic term, the embedding and the LM head are ignored.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = (0.7, 0.9, 0.95)
NB = 32


def k_at(ks, ys, t):
    """smallest k where the curve reaches t, linear interpolation in k (ks ascending, ys ~ increasing)."""
    ks, ys = np.asarray(ks, float), np.asarray(ys, float)
    if ys[0] >= t:
        return float(ks[0])
    for i in range(1, len(ks)):
        if ys[i] >= t:
            f = (t - ys[i - 1]) / max(ys[i] - ys[i - 1], 1e-12)
            return float(ks[i - 1] + f * (ks[i] - ks[i - 1]))
    return float("nan")


def ms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0) if len(x) else (float("nan"), 0.0)


def curves(dot, sq, full, groups):
    """cos, relnorm, proj per batch over cumulative bins, summed over ``groups``. dot/sq (B, G, nb), full (B, G)."""
    d, s, f = dot[:, groups].sum(1), sq[:, groups].sum(1), full[:, groups].sum(1)[:, None]
    return d / np.sqrt(np.maximum(s * f, 1e-300)), np.sqrt(s / np.maximum(f, 1e-300)), d / np.maximum(f, 1e-300)


def summarize(lab, d, A, steps, rows_seen):
    KS = list(d["ks"])
    nb = len(KS) + 1
    lins = d["lins"]
    out = {"label": lab, "steps": steps, "rows_seen_per_task": rows_seen, "eval_size_rows": len(d["rows"]),
           "ce_mean": float(np.mean([r["ce"] for r in d["rows"]])), "scopes": {}}
    for s in d["scopes"]:
        S = {}
        g = lambda r, q: A[f"{s}|{r}|{q}"]  # noqa: E731
        d0 = g("down0_all", "full")
        for L in d["splits"]:
            for v in ("all", "body"):
                r = f"down{L}_{v}"
                groups = list(range(2 * L, 2 * NB))
                cos, rel, proj = curves(g(r, "dot"), g(r, "sq"), g(r, "full"), groups)
                kx = [0.0] + KS
                e = {"cos": cos.mean(0).tolist(), "cos_sd": cos.std(0).tolist(), "relnorm": rel.mean(0).tolist(),
                     "proj": proj.mean(0).tolist(), "k": kx}
                for t in TARGETS:
                    e[f"k_cos{t}"] = ms([k_at(kx[1 if v == 'all' else 0:], c[1 if v == 'all' else 0:], t) for c in cos])
                es = g(r, "eshare")
                e["eshare"] = es.mean(0).tolist()
                for t in TARGETS:
                    e[f"k_energy{t}"] = ms([k_at(KS, c, t) for c in es])
                ne = g(r, "nelig")
                e["elig_token_share"] = float((ne[:, 0] / ne[:, 1]).mean())
                if v == "body":
                    e["body_energy_share"] = float(g(r, "elig_energy_share").mean())
                # whole model: blocks < L get the full update, blocks >= L the top-k one
                early = d0[:, :2 * L].sum(1)[:, None]
                dd, ss, ff = (g(r, "dot")[:, groups].sum(1), g(r, "sq")[:, groups].sum(1),
                              g(r, "full")[:, groups].sum(1)[:, None])
                wc = (early + dd) / np.sqrt((early + ss) * (early + ff))
                e["whole_cos"] = wc.mean(0).tolist()
                e["downstream_share_of_update_energy"] = float((ff[:, 0] / (early[:, 0] + ff[:, 0])).mean())
                for t in TARGETS:
                    e[f"k_whole_cos{t}"] = ms([k_at(kx[1 if v == 'all' else 0:], c[1 if v == 'all' else 0:], t)
                                               for c in wc])
                S[r] = e
        for L in (7, 16):
            r = f"rand{L}_body"
            cos, _, _ = curves(g(r, "dot"), g(r, "sq"), g(r, "full"), list(range(2 * L, 2 * NB)))
            S[r] = {"cos": cos.mean(0).tolist(), "k": [0.0] + KS,
                    **{f"k_cos{t}": ms([k_at([0.0] + KS, c, t) for c in cos]) for t in TARGETS}}
        # per linear (own ranking) and per block part (each linear ranked by its own energy, summed)
        own = {}
        for ln in lins:
            if ln["block"] not in d["rep_blocks"]:
                continue
            for v in ("all", "body"):
                r = f"own_{ln['name']}_{v}"
                cos, _, _ = curves(g(r, "dot"), g(r, "sq"), g(r, "full"), [0])
                kx = [0.0] + KS
                own[r] = {"block": ln["block"], "part": ln["part"], "cos": cos.mean(0).tolist(),
                          **{f"k_cos{t}": ms([k_at(kx[1 if v == 'all' else 0:], c[1 if v == 'all' else 0:], t)
                                              for c in cos]) for t in TARGETS},
                          **{f"k_energy{t}": ms([k_at(KS, c, t) for c in g(r, "eshare")]) for t in TARGETS}}
        S["own_linear"] = own
        parts = {}
        for b in d["rep_blocks"]:
            for p, pn in ((0, "mixer"), (1, "mlp")):
                for v in ("all", "body"):
                    names = [ln["name"] for ln in lins if ln["block"] == b and ln["part"] == p]
                    dot = sum(g(f"own_{n}_{v}", "dot")[:, 0] for n in names)
                    sq = sum(g(f"own_{n}_{v}", "sq")[:, 0] for n in names)
                    ff = sum(g(f"own_{n}_{v}", "full")[:, 0] for n in names)[:, None]
                    cos = dot / np.sqrt(sq * ff)
                    kx = [0.0] + KS
                    parts[f"b{b}_{pn}_{v}"] = {"cos": cos.mean(0).tolist(), **{
                        f"k_cos{t}": ms([k_at(kx[1 if v == 'all' else 0:], c[1 if v == 'all' else 0:], t)
                                         for c in cos]) for t in TARGETS}}
        S["own_block_part"] = parts
        # recipe shape (keep 20% of body past 7, 7% past 16), oracle and random, whole-model cos
        for r in ("recipe_oracle", "recipe_random"):
            dot, sq, full = g(r, "dot"), g(r, "sq"), g(r, "full")
            groups = list(range(14, 2 * NB))
            early = d0[:, :14].sum(1)
            dd, ss, ff = dot[:, groups, 1].sum(1), sq[:, groups, 1].sum(1), full[:, groups].sum(1)
            S[r] = {"whole_cos": ms((early + dd) / np.sqrt((early + ss) * (early + ff))),
                    "down7_cos": ms(dd / np.sqrt(ss * ff)),
                    "down16_cos": ms(dot[:, 32:, 1].sum(1) / np.sqrt(sq[:, 32:, 1].sum(1) * full[:, 32:].sum(1))),
                    "kept_token_share": g(r, "kfrac").mean(0).tolist()}
        out["scopes"][str(s)] = S
    # sketch validation (smallest scope)
    val = {}
    for nme in d["exact"]:
        ex_d, ex_s = A[f"exact|{nme}|down7_body|dot"], A[f"exact|{nme}|down7_body|sq"]
        sk_d, sk_s = A[f"sketch|{nme}|down7_body|dot"], A[f"sketch|{nme}|down7_body|sq"]
        ce = ex_d / np.sqrt(ex_s * ex_s[:, -1:])
        cs = sk_d / np.sqrt(sk_s * sk_s[:, -1:])
        val[nme] = {"max_abs_cos_err_down7_body": float(np.nanmax(np.abs(ce - cs))),
                    "mean_abs_cos_err": float(np.nanmean(np.abs(ce - cs)))}
        sm = str(min(d["scopes"]))
        own_s = A[f"{sm}|own_{nme}_all|dot"][:, 0] / np.sqrt(A[f"{sm}|own_{nme}_all|sq"][:, 0] *
                                                              A[f"{sm}|own_{nme}_all|full"][:, :1])
        ex2 = A[f"exact|{nme}|own_all|dot"] / np.sqrt(A[f"exact|{nme}|own_all|sq"] * A[f"exact|{nme}|own_all|sq"][:, -1:])
        val[nme]["max_abs_cos_err_own_all"] = float(np.nanmax(np.abs(own_s - ex2)))
    out["sketch_validation"] = val
    return out


def flop_bound(S, splits, targets=(0.9, 0.95)):
    """min over single splits L of the FLOP ratio reaching whole-model cos >= t (body view, non-body kept)."""
    res = {}
    for t in targets:
        best = None
        per = {}
        for L in splits:
            e = S[f"down{L}_body"]
            k = e[f"k_whole_cos{t}"][0]
            if not np.isfinite(k):
                continue
            fb = e["elig_token_share"]
            q = (1 - fb) + fb * k
            ratio = L / NB + (1 - L / NB) * q
            per[L] = {"k_body": k, "kept_token_share": q, "flop_ratio": ratio, "speedup": 1 / ratio}
            if best is None or ratio < best[1]:
                best = (L, ratio)
        res[str(t)] = {"per_split": per, "best_split": best[0] if best else None,
                       "best_ratio": best[1] if best else None, "best_speedup": 1 / best[1] if best else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--state", default=os.path.join(HERE, "state.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "summary.json"))
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    st = json.load(open(a.state))
    allout = []
    for lab in sorted(os.listdir(a.results), key=lambda x: st.get(x, {}).get("steps", 0)):
        p = os.path.join(a.results, lab, "probe.json")
        if not lab.startswith(a.prefix) or not os.path.exists(p):
            continue
        d = json.load(open(p))
        A = dict(np.load(os.path.join(a.results, lab, "probe_arrays.npz")))
        o = summarize(lab, d, A, st.get(lab, {}).get("steps"), st.get(lab, {}).get("rows_seen"))
        for s in o["scopes"]:
            o["scopes"][s]["flop_bound"] = flop_bound(o["scopes"][s], d["splits"])
        allout.append(o)
        print(f"== {lab} ({o['steps']} steps, CE {o['ce_mean']:.3f}, {o['eval_size_rows']} rows) "
              f"sketch val {json.dumps(o['sketch_validation'])}")
        for s, S in o["scopes"].items():
            for L in d["splits"]:
                for v in ("all", "body"):
                    e = S[f"down{L}_{v}"]
                    print(f"  bs{s} down{L:<2} {v:4}: k@cos .7/.9/.95 = " +
                          " / ".join(f"{e[f'k_cos{t}'][0]:.3f}" for t in TARGETS) +
                          "   energy " + " / ".join(f"{e[f'k_energy{t}'][0]:.3f}" for t in TARGETS) +
                          f"   whole .9/.95 " + " / ".join(f"{e[f'k_whole_cos{t}'][0]:.3f}" for t in (0.9, 0.95)))
            print(f"  bs{s} recipe oracle whole cos {S['recipe_oracle']['whole_cos'][0]:.3f} "
                  f"(down7 {S['recipe_oracle']['down7_cos'][0]:.3f}), random {S['recipe_random']['whole_cos'][0]:.3f}; "
                  f"bound speedup @.9 {S['flop_bound']['0.9']['best_speedup']}, @.95 {S['flop_bound']['0.95']['best_speedup']}")
    json.dump(allout, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
