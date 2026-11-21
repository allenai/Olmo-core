"""Group-attention arm vs gdd256 and dense256b (checkpoint-then-score, same scorer and held-out rows).

    python3 attn_groups/compare_ag.py [--arm gddag4] [--res /tmp/claude-3018/elp_beaker_results] [--out x.json]

Per held-out set (lc256k / lc16k, 160 rows each, eval_size < 500): CE at each arm checkpoint; gdd256 and dense256b
interpolated per row at the arm's PFs (paired diff + SE); FLOPs-to-match (the arm's PF to reach each reference
checkpoint's CE, ratio ref PF / arm PF); matched ROWS = gdd256 at the same optimizer step (identical data order and batch,
so the same rows consumed), interpolated per row in step; and per-step cost from the training JSONs (actual PF, train_s,
peak GiB) over steps both arms ran.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_lc as A  # noqa: E402


def curve(res, arm, S, by="pf"):
    pfs = A.step_pf(res, arm)
    sc = A.scores(res, arm)
    out = []
    for st in sc:
        if S not in sc[st]:
            continue
        x = A.pf_at(pfs, st) if by == "pf" else float(st)
        if not math.isnan(x):
            out.append((x, sc[st][S]))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gddag4")
    ap.add_argument("--res", default="/tmp/claude-3018/elp_beaker_results")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    A.RUNS[a.arm] = f"lcA-{a.arm}/{a.arm}.json"
    rep = {}
    for S in ("lc256k", "lc16k"):
        ca = curve(a.res, a.arm, S)
        n = len(ca[0][1]) if ca else 0
        print(f"\n== {S} (eval_size={n}, below 500) ==")
        print(f"  {a.arm}: " + "  ".join(f"{p:.0f}PF:{A.mean(r):.4f}" for p, r in ca))
        R = {"eval_size": n, "curve": [(p, A.mean(r)) for p, r in ca], "vs": {}}
        for ref in ("gdd256", "dense256b"):
            cr = curve(a.res, ref, S)
            rows = []
            for p, r in ca:
                if p == 0:
                    continue
                gi, nb = A.interp_rows(cr, p)
                if gi is None:
                    continue
                d = A.mean(r) - A.mean(gi)
                rows.append({"pf": p, "arm": A.mean(r), "ref": A.mean(gi), "diff": d, "se": A.se_paired(r, gi)})
                print(f"  @{p:.0f} PF: {a.arm} {A.mean(r):.4f} vs {ref} {A.mean(gi):.4f}  diff {d:+.4f} +- {rows[-1]['se']:.4f}")
            ftm = []
            amax = max(p for p, _ in ca)
            for p, r in cr:
                if p == 0 or p > 1.6 * amax:
                    continue
                pa = A.first_reach(ca, A.mean(r))
                ftm.append({"ref_pf": p, "ref_ce": A.mean(r), "arm_pf": pa, "ratio": (p / pa) if pa else None})
                print(f"  {ref} {p:.0f} PF CE {A.mean(r):.4f} -> {a.arm} reaches at "
                      + (f"{pa:.0f} PF ({p / pa:.2f}x)" if pa else "not reached in its range"))
            R["vs"][ref] = {"matched_pf": rows, "ftm": ftm}
        # matched rows: same optimizer step
        cs, gs = curve(a.res, a.arm, S, by="step"), curve(a.res, "gdd256", S, by="step")
        mr = []
        for st, r in cs:
            if st == 0:
                continue
            gi, _ = A.interp_rows(gs, st)
            if gi is None:
                continue
            mr.append({"step": st, "arm": A.mean(r), "gdd256": A.mean(gi), "diff": A.mean(r) - A.mean(gi),
                       "se": A.se_paired(r, gi)})
            print(f"  matched rows, step {st:.0f}: {a.arm} {A.mean(r):.4f} vs gdd256 {A.mean(gi):.4f} "
                  f"diff {mr[-1]['diff']:+.4f} +- {mr[-1]['se']:.4f}")
        R["matched_rows"] = mr
        rep[S] = R
    # per-step cost
    def steps(arm):
        d = json.load(open(os.path.join(a.res, A.RUNS[arm])))
        out, prev = {}, 0.0
        for s in d["steps"]:
            pf = s["flops"]["actual_pf"]
            out[int(s["step"])] = (pf - prev, s["train_s"], s["peak_gib"])
            prev = pf
        return out
    sa, sg = steps(a.arm), steps("gdd256")
    common = sorted(k for k in sa if k in sg and k > 5)
    if common:
        f = lambda t, i: sum(t[k][i] for k in common)  # noqa: E731
        cost = {"steps": len(common), "pf_ratio": f(sa, 0) / f(sg, 0), "time_ratio": f(sa, 1) / f(sg, 1),
                "arm_s_per_step": f(sa, 1) / len(common), "gdd256_s_per_step": f(sg, 1) / len(common),
                "arm_peak_gib": max(sa[k][2] for k in common), "gdd256_peak_gib": max(sg[k][2] for k in common)}
        print(f"\nper-step cost over {len(common)} common steps: PF ratio {cost['pf_ratio']:.3f}, time ratio "
              f"{cost['time_ratio']:.3f} ({cost['arm_s_per_step']:.2f} vs {cost['gdd256_s_per_step']:.2f} s), peak "
              f"{cost['arm_peak_gib']:.1f} vs {cost['gdd256_peak_gib']:.1f} GiB")
        rep["cost"] = cost
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
