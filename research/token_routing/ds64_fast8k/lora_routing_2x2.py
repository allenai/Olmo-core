"""LoRA x routing 2x2 ablation (2026-10-26): how much of the cheap-method gain is LoRA, how much is token routing?

Cells (Qwen3.5-4B SFT, no full-context steps for the routed arms):
  A  dense, no LoRA                     arm dense
  B  dense + whole-model LoRA r64       arm denselora64all        (--lora-rank 64 --lora-blocks 0 --lora-scale 32)
  C  two-stage routing, no LoRA         arm r20to7M7de0 (mt4) / r20to7M7de0f16 (mt11; router session: r20to7f16)
  D  two-stage routing + LoRA-all       arm lora64all32r20to7 (mt4) / lora64all32r20to7f16 (mt11);
                                        router session spells it lora64K32r20to7[f16]
each at LR 5e-6 and 1e-5. Per (mix, pool, rows, cell) the better LR is used (seed copies averaged first).

Scores: mt4 = 8-cell mean (contradiction / nq / outlier / oolong at 8k + 2k); mt11 = 9-task mean at 8k (the
overview's MT11 list). Training PF = flop_meter/actual_pflops. Matched-FLOP delta = score - dense(PF), dense
interpolated linearly in log PF on the best-of-{5e-6, 1e-5} dense curve (all pools of the mix, incl. the hard-stop
cut points s488st31/st61, s240st32/st64, which are flagged). "x dense" = dense PF needed to reach the cell's score
/ the cell's PF (">" = above dense's best point, a lower bound).

Sources: results/pareto_points.json (collect_pareto_points.py; main checkout by default) plus any cell run listed
in the given state.json files that it does not cover yet (scores from ops results, PF from the Beaker log; cached).

    python3 research/token_routing/ds64_fast8k/lora_routing_2x2.py   # -> lora_routing_2x2.json (tracked)
"""

import argparse
import collections
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_pareto_points import _beaker_train, _ops_scores  # noqa: E402

MAIN = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/research/token_routing/ds64_fast8k"
MT4 = [(t, r) for t in ("contradiction", "nq", "outlier", "oolong") for r in ("8k", "2k")]
MT11 = ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction", "xabsence", "strmatch", "grouping"]
CELL_ARMS = {
    "mt4": {"A": {"dense"}, "B": {"denselora64all"}, "C": {"r20to7M7de0"},
            "D": {"lora64all32r20to7", "lora64K32r20to7"}},
    "mt11": {"A": {"dense"}, "B": {"denselora64all"}, "C": {"r20to7M7de0f16", "r20to7f16"},
             "D": {"lora64all32r20to7f16", "lora64K32r20to7f16"}},
}
LRS = ("5e-06", "1e-05")
NAME_RE = re.compile(r"f8k-(mt\d+)-(.+)-([sf])(\d+)(?:st(\d+))?-(td[a-z]|otr)-lr(\d[\d.]*(?:e-\d+)?)$")
LAM_RE = re.compile(r"f8k-(mt\d+)s(\d*)-(.+?)(?:-s(\d))?-(l[a-z]{2})-lr(\d[\d.]*(?:e-\d+)?)(-.*)?-lam$")


def meta_of(run):
    """(mix, pool, rows, arm, lr, max_steps, seed) or None."""
    m = NAME_RE.match(run)
    if m:
        mix, arm, pool, rows, st, _, lr = m.groups()
        seed = int(re.search(r"sd(\d+)$", arm).group(1)) if re.search(r"sd\d+$", arm) else 0
        return mix, pool, int(rows), re.sub(r"sd\d+$", "", arm), float(lr), int(st or 0), seed
    m = LAM_RE.match(run)
    if m:
        mix, rows, arm, seed, _, lr, tail = m.groups()
        st = re.search(r"-s(\d+)", tail or "")
        if tail and re.search(r"-ep\d", tail):
            return None  # repeated epochs are never on a fresh-data curve
        return mix, "s", int(rows or 488), arm, float(lr), int(st.group(1)) if st else 0, int(seed or 0)
    return None


def score_of(mix, c):
    if mix == "mt4":
        return sum(c[x] for x in MT4) / 8 if all(x in c for x in MT4) else None
    return sum(c[(t, "8k")] for t in MT11) / 9 if all((t, "8k") in c for t in MT11) else None


def cell_of(mix, arm):
    return next((k for k, v in CELL_ARMS.get(mix, {}).items() if arm in v), None)


def load_runs(points_path, states, cache_path):
    d = json.load(open(points_path))
    cells = collections.defaultdict(dict)
    for p in d["points"]:
        if p["eval_size"] >= 500:
            cells[p["run"]][(p["task"], p["rung"])] = p["score"]
    runs = {k: dict(pf=r.get("pf"), cells=cells.get(k, {})) for k, r in d["runs"].items()
            if r.get("pf") and (r.get("epochs") or 1) == 1}
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for sp in states:
        if not os.path.exists(sp):
            continue
        for run, v in json.load(open(sp))["runs"].items():
            meta = meta_of(run)
            if not meta or not v.get("ex") or v.get("state") != "DONE" or cell_of(meta[0], meta[3]) is None:
                continue
            if run in runs and score_of(meta[0], runs[run]["cells"]) is not None:
                continue
            sc = {k: x for k, (x, n) in _ops_scores(run).items() if n >= 500}
            if score_of(meta[0], sc) is None:
                continue
            if run not in cache:
                pf, steps, wall = _beaker_train(v["ex"])
                cache[run] = dict(pf=pf, steps=steps, wall=wall, ex=v["ex"])
                json.dump(cache, open(cache_path, "w"), indent=1)
            runs[run] = dict(pf=cache[run]["pf"], cells=sc)
    return runs


def interp(curve, x):
    """curve: sorted [(pf, y)]; linear in log PF. Returns (y, flag) where flag marks extrapolation."""
    if x <= curve[0][0]:
        (x0, y0), (x1, y1) = curve[0], curve[1]
        flag = "below"
    elif x >= curve[-1][0]:
        (x0, y0), (x1, y1) = curve[-2], curve[-1]
        flag = "above"
    else:
        i = next(i for i in range(len(curve) - 1) if curve[i][0] <= x <= curve[i + 1][0])
        (x0, y0), (x1, y1) = curve[i], curve[i + 1]
        flag = ""
    t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return y0 + t * (y1 - y0), flag


def inverse(curve, y):
    """dense PF reaching score y on the (monotone-ised) curve; None = above dense's best point."""
    best = -1e9
    mono = []
    for x, v in curve:
        best = max(best, v)
        mono.append((x, best))
    if y > mono[-1][1]:
        return None
    if y <= mono[0][1]:
        return mono[0][0]
    for (x0, y0), (x1, y1) in zip(mono, mono[1:]):
        if y0 <= y <= y1 and y1 > y0:
            return math.exp(math.log(x0) + (y - y0) / (y1 - y0) * (math.log(x1) - math.log(x0)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=f"{MAIN}/results/pareto_points.json")
    ap.add_argument("--state", action="append", default=None, help="launch_fast8k state.json files (repeatable)")
    ap.add_argument("--out", default=f"{HERE}/lora_routing_2x2.json")
    ap.add_argument("--cache", default=None, help="PF cache (default: lora_routing_2x2_pf_cache.json next to this file)")
    a = ap.parse_args()
    # state_tdx.json = snapshot of the 2x2's -tdx campaign state (run from its own worktree; tracked so this script
    # can be re-run from any checkout)
    states = a.state or [f"{MAIN}/state.json", f"{HERE}/state.json", f"{HERE}/state_tdx.json"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    runs = load_runs(a.points, states, a.cache or os.path.join(HERE, "lora_routing_2x2_pf_cache.json"))

    # group seed copies: (mix, pool, rows, steps, cell|arm, lr) -> [(run, pf, score, o8, c8, per-task 8k)]
    grp = collections.defaultdict(list)
    for run, r in runs.items():
        meta = meta_of(run)
        if not meta:
            continue
        mix, pool, rows, arm, lr, st, seed = meta
        cell = cell_of(mix, arm)
        y = score_of(mix, r["cells"])
        if cell is None or y is None:
            continue
        c = r["cells"]
        grp[(mix, pool, rows, st, cell, f"{lr:.0e}")].append(dict(
            run=run, pf=r["pf"], y=y, o8=c[("outlier", "8k")], c8=c[("contradiction", "8k")], seed=seed))

    def agg(v):
        n = len(v)
        return dict(n=n, pf=sum(x["pf"] for x in v) / n, y=sum(x["y"] for x in v) / n,
                    o8=sum(x["o8"] for x in v) / n, c8=sum(x["c8"] for x in v) / n,
                    y_sd=statistics.stdev([x["y"] for x in v]) if n > 1 else None,
                    o8_sd=statistics.stdev([x["o8"] for x in v]) if n > 1 else None,
                    runs=[x["run"] for x in v])

    G = {k: agg(v) for k, v in grp.items()}

    # dense curves per mix: per (rows, steps) the best of LRS (and, as a sensitivity, of every LR tried)
    out = {"dense_curve": {}, "tables": [], "per_lr": []}
    curves = {}
    for mix in ("mt4", "mt11"):
        for tag, lrs in (("best_5e6_1e5", LRS), ("best_any_lr", None)):
            pts = collections.defaultdict(list)
            for (m, pool, rows, st, cell, lr), g in G.items():
                if m == mix and cell == "A" and (lrs is None or lr in lrs):
                    pts[(pool, rows, st)].append((g["y"], g["pf"], lr, g["n"]))
            curve = []
            for (pool, rows, st), v in pts.items():
                y, pf, lr, n = max(v)
                curve.append((pf, y, f"{pool}{rows}" + (f"st{st}" if st else ""), lr, n))
            curve.sort()
            # one value per PF (s1952 and f1952 sit at the same PF): average them
            merged = []
            for pf, y, lab, lr, n in curve:
                if merged and pf <= merged[-1][0][0] * 1.02:
                    merged[-1].append((pf, y, lab, lr, n))
                else:
                    merged.append([(pf, y, lab, lr, n)])
            cv = [(sum(p[0] for p in v) / len(v), sum(p[1] for p in v) / len(v),
                   "+".join(p[2] for p in v), "/".join(p[3] for p in v)) for v in merged]
            curves[(mix, tag)] = [(x, y) for x, y, _, _ in cv]
            out["dense_curve"][f"{mix}:{tag}"] = [dict(pf=round(x, 1), y=round(y, 4), pts=l, lr=r) for x, y, l, r in cv]

    for (mix, pool, rows, st, cell, lr), g in sorted(G.items()):
        out["per_lr"].append(dict(mix=mix, pool=pool, rows=rows, steps=st, cell=cell, lr=lr,
                                  **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()}))

    sizes = sorted({(m, p, r) for (m, p, r, st, c, lr) in G if st == 0})
    for mix, pool, rows in sizes:
        row = dict(mix=mix, pool=pool, rows=rows, cells={})
        for cell in "ABCD":
            cand = {lr: G[(mix, pool, rows, 0, cell, lr)] for lr in LRS if (mix, pool, rows, 0, cell, lr) in G}
            if not cand:
                continue
            lr, g = max(cand.items(), key=lambda kv: kv[1]["y"])
            e = dict(lr=lr, lrs_tried=sorted(cand), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                     for k, v in g.items()})
            e["other_lr"] = {l: round(x["y"], 4) for l, x in cand.items() if l != lr}
            for tag in ("best_5e6_1e5", "best_any_lr"):
                cv = curves[(mix, tag)]
                yd, flag = interp(cv, g["pf"])
                need = inverse(cv, g["y"])
                e[f"d_{tag}"] = round(g["y"] - yd, 4)
                e[f"flag_{tag}"] = flag
                e[f"x_{tag}"] = (round(need / g["pf"], 2) if need else None)
                e[f"xgt_{tag}"] = need is None
            row["cells"][cell] = e
        cs = row["cells"]
        if all(k in cs for k in "ABCD"):
            for tag in ("best_5e6_1e5", "best_any_lr"):
                dA, dB, dC, dD = (cs[k][f"d_{tag}"] for k in "ABCD")
                row[f"attr_{tag}"] = dict(lora_no_route=round(dB - dA, 4), lora_with_route=round(dD - dC, 4),
                                          route_no_lora=round(dC - dA, 4), route_with_lora=round(dD - dB, 4),
                                          interaction=round(dD - dC - dB + dA, 4), total=round(dD - dA, 4))
                # the same in log2 of the dense-FLOP multiplier ("x dense"), which composes multiplicatively;
                # None when a cell is above dense's best point (its multiplier is only a lower bound)
                lx = {k: (math.log2(cs[k][f"x_{tag}"]) if cs[k][f"x_{tag}"] else None) for k in "ABCD"}
                if all(v is not None for v in lx.values()):
                    row[f"attr_log2x_{tag}"] = dict(
                        lora_no_route=round(lx["B"] - lx["A"], 3), lora_with_route=round(lx["D"] - lx["C"], 3),
                        route_no_lora=round(lx["C"] - lx["A"], 3), route_with_lora=round(lx["D"] - lx["B"], 3),
                        interaction=round(lx["D"] - lx["C"] - lx["B"] + lx["A"], 3), total=round(lx["D"] - lx["A"], 3))
            # matched DATA (same rows): raw score differences and PF ratios
            row["matched_data"] = dict(
                lora_no_route=round(cs["B"]["y"] - cs["A"]["y"], 4), lora_with_route=round(cs["D"]["y"] - cs["C"]["y"], 4),
                route_no_lora=round(cs["C"]["y"] - cs["A"]["y"], 4), route_with_lora=round(cs["D"]["y"] - cs["B"]["y"], 4),
                pf_ratio={k: round(cs[k]["pf"] / cs["A"]["pf"], 3) for k in "BCD"})
        out["tables"].append(row)

    json.dump(out, open(a.out, "w"), indent=1)
    for row in out["tables"]:
        print(f"\n== {row['mix']} {row['pool']}{row['rows']}")
        for cell, e in row["cells"].items():
            sd = f" sd={e['y_sd']:.3f}" if e.get("y_sd") is not None else ""
            x = (">" + "?" if e["xgt_best_5e6_1e5"] else f"{e['x_best_5e6_1e5']}")
            print(f"  {cell} lr={e['lr']} n={e['n']} pf={e['pf']:.0f} y={e['y']:.4f}{sd} o8={e['o8']:.3f} c8={e['c8']:.3f} "
                  f"d={e['d_best_5e6_1e5']:+.4f}{e['flag_best_5e6_1e5'] and '(' + e['flag_best_5e6_1e5'] + ')'} x={x} "
                  f"other={e['other_lr']}")
        if "attr_best_5e6_1e5" in row:
            print("  attr:", row["attr_best_5e6_1e5"], "\n  log2x:", row.get("attr_log2x_best_5e6_1e5"),
                  "\n  matched-data:", row["matched_data"])
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
