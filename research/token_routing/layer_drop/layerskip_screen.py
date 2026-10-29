"""Layer-skip screen (2026-10-28): each arm vs dense at matched training FLOPs, and the recipe arms vs the recipe.

Dense reference, per cell: the better of the LR 2e-6 / 5e-6 dense curves at the arm's PF, each interpolated linearly
in log PF over the dense runs of the SAME data pool (original s<rows> or fresh f<rows>), as make_pareto_section.py
does. Dense / baseline points come from results/pareto_points.json (collect_pareto_points.py); this screen's runs from
ops results (scores) and their Beaker training logs (PF, steps, wall-clock).

    python research/token_routing/layer_drop/layerskip_screen.py \
        --pareto <main checkout>/research/token_routing/ds64_fast8k/results/pareto_points.json --out screen.json
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
F8K = os.path.join(HERE, "..", "ds64_fast8k")
sys.path.insert(0, F8K)
from collect_pareto_points import _beaker_train, _ops_scores  # noqa: E402
from make_pareto_section import CELLS, interp  # noqa: E402

DENSE_LRS = (2e-6, 5e-6)


def pool(run):
    return "fresh" if re.search(r"-f\d{3,5}-", run) else "orig"


def rows_of(run):
    m = re.search(r"-[sf](\d{3,5})(?:st\d+)?-", run) or re.search(r"mt4s(\d+)-", run)
    return int(m.group(1)) if m else 488


def dense_ref(P, pl, cell, pf):
    best = None
    for lr in DENSE_LRS:
        curve = [(r["pf"], P["scores"][(n, *cell)]) for n, r in P["runs"].items()
                 if r.get("mix") == "mt4" and r.get("arm") == "dense" and abs(r["lr"] - lr) < 1e-12
                 and pool(n) == pl and r.get("pf") and (n, *cell) in P["scores"] and r.get("epochs", 1) == 1]
        if len(curve) < 2:
            continue
        v, ext = interp(curve, pf)
        if best is None or v > best[0]:
            best = (v, ext)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", required=True)
    ap.add_argument("--state", default=os.path.join(F8K, "state.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.pareto))
    P = {"runs": d["runs"], "scores": {(p["run"], p["task"], p["rung"]): p["score"] for p in d["points"]}}
    st = json.load(open(a.state))["runs"]
    out = []
    for run, v in st.items():
        sc = _ops_scores(run)
        if len(sc) < len(CELLS):
            print(f"[wait] {run}: {len(sc)}/{len(CELLS)} cells scored")
            continue
        pf, steps, wall = _beaker_train(v["ex"])
        pl = pool(run)
        cells = {}
        for t, r in CELLS:
            x = sc[(t, r)][0]
            ref = dense_ref(P, pl, (t, r), pf)
            cells[f"{t}@{r}"] = {"score": x, "eval_size": sc[(t, r)][1], "dense": ref[0] if ref else None,
                                 "delta": (x - ref[0]) if ref else None, "extrap": ref[1] if ref else None}
        rec = {"run": run, "pool": pl, "rows": rows_of(run), "pf": pf, "steps": steps,
               "s_per_step": (wall / (steps - 1)) if wall and steps and steps > 1 else None, "cells": cells}
        ds = [c["delta"] for c in cells.values() if c["delta"] is not None]
        rec["mean_delta"] = sum(ds) / len(ds) if len(ds) == len(CELLS) else None
        rec["mean_score"] = sum(c["score"] for c in cells.values()) / len(cells)
        # the recipe arms: also vs the recipe itself on the same data (its own baseline)
        m = re.match(r"(f8k-mt4-lora64all32r20to7)-([a-z0-9]+?)(sd\d+)?-((?:s|f)\d+-.*)$", run)
        if m:
            base = f"{m.group(1)}{m.group(3) or ''}-{m.group(4)}"
            bs = {(p_["task"], p_["rung"]): p_["score"] for p_ in d["points"] if p_["run"] == base} \
                or {k: x for k, (x, _) in _ops_scores(base).items()}
            if len(bs) >= len(CELLS):
                rec["recipe_base"] = base
                rec["recipe_base_pf"] = d["runs"].get(base, {}).get("pf")
                rec["delta_vs_recipe"] = {f"{t}@{r}": cells[f"{t}@{r}"]["score"] - bs[(t, r)] for t, r in CELLS}
                rec["mean_delta_vs_recipe"] = sum(rec["delta_vs_recipe"].values()) / len(CELLS)
        if "-gs" in run:
            # selected-layer histogram over training (GradLayerSelectCallback's final line)
            import subprocess

            log = subprocess.run([os.path.expanduser("~/.local/bin/beaker"), "experiment", "logs", v["ex"]],
                                 capture_output=True, text=True, timeout=300).stdout
            hm = re.findall(r"\[grad-select\] final selected_frac \[([^\]]*)\]", log)
            if hm:
                rec["selected_frac"] = [float(x) for x in hm[-1].split(",")]
        out.append(rec)
        print(f"{run}: PF {pf:.0f}  s/step {rec['s_per_step'] or float('nan'):.2f}  mean {rec['mean_score']:.3f}  "
              f"mean Δ vs dense {rec['mean_delta'] if rec['mean_delta'] is None else round(rec['mean_delta'], 3)}  "
              f"outlier@8k Δ {cells['outlier@8k']['delta']}  contra@8k Δ {cells['contradiction@8k']['delta']}"
              + (f"  mean Δ vs recipe {rec['mean_delta_vs_recipe']:+.3f}" if "mean_delta_vs_recipe" in rec else ""))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
