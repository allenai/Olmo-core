"""Score early-loss-proxy runs against the dense reference: FLOPs to reach a dense-forward held-out loss.

For every finished run JSON (``dense_probe.py``) under ``--results`` (one or more result folders; small JSONs, so
reading them through /net/<node>/ is fine):

- curve: (cumulative training PF from the flop meter, mean held-out dense CE) at step 0 and each probe step;
- ``frac``: the run's loss drop at its last probe / the reference's drop at ITS last probe (same seed);
- ``iso``: iso-loss FLOP speedup = PF the reference needs to reach the run's last-probe loss (read off the
  reference curve, piecewise linear in PF on the running-min envelope) / the run's PF. ``iso_lb`` marks a lower
  bound (the run's loss is below everything the reference reached);
- ``to_target``: reference PF at its last probe / PF at which the run first crosses the reference's last-probe loss
  (None if it never does);
- peak GPU memory (max over training steps and ranks) and its ratio to the reference; wall clock.

The reference is, per loss level, the cheaper of the ``--ref`` arms (default dense full FT and dense LoRA), seed
matched (falls back to the mean over the reference's seeds). Run names: ``elp-<tag>-<arm>-sd<seed>``.

    python3 score.py --results /net/mooney/data/prasann/early_loss_proxy/results/v2 [more folders] --out leaderboard.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def load(folders):
    runs = {}
    for fo in folders:
        for p in sorted(glob.glob(os.path.join(fo, "elp-*.json"))):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            if not d.get("done") or not d.get("probes"):
                continue
            m = re.match(r"elp-(.+)-sd(\d+)$", os.path.basename(p)[:-5])
            if not m:
                continue
            tag_arm, seed = m.group(1), int(m.group(2))
            tag = os.path.basename(os.path.normpath(fo))
            arm = tag_arm[len(tag) + 1:] if tag_arm.startswith(tag + "-") else tag_arm
            rows = d["rows"]
            groups = defaultdict(list)
            for i, r in enumerate(rows):
                groups["all"].append(i)
                groups["task/" + r["task"]].append(i)
                groups["set/" + r["set"]].append(i)
            curve = []
            for pr in d["probes"]:
                ce = np.asarray(pr["ce"])
                fl = pr.get("flops") or {}
                curve.append({"step": pr["step"], "pf": float(fl.get("actual_pf") or 0.0),
                              "dense_pf": float(fl.get("dense_pf") or 0.0),
                              **{g: float(ce[ix].mean()) for g, ix in groups.items()}})
            st = d.get("steps", [])
            runs[(tag, arm, seed)] = {
                "tag": tag, "arm": arm, "seed": seed, "curve": curve, "eval_size": len(rows),
                "ce_rows_last": d["probes"][-1]["ce"], "ce_rows_0": d["probes"][0]["ce"],
                "peak_gib": max([s.get("peak_gib", 0.0) for s in st] or [0.0]),
                "train_s": float(sum(s["train_s"] for s in st)), "probe_s": float(sum(p["probe_s"] for p in d["probes"])),
                "wall_s": d.get("process_wall_s"), "steps": len(st), "node": (d.get("slurm") or {}).get("SLURMD_NODENAME"),
                "lr": d["probes"][-1]["lr"][0] if d["probes"][-1].get("lr") else None,
                "batch_hashes": [s.get("batch_hash_rank0") for s in st[:3]],
                "step0_src": d["probes"][0].get("src"), "path": p}
    return runs


def envelope(curve, key="all"):
    """(pf, loss) points of a curve, loss made non-increasing (running min)."""
    pf = np.asarray([c["pf"] for c in curve])
    ls = np.minimum.accumulate(np.asarray([c[key] for c in curve]))
    return pf, ls


def pf_to_reach(curve, loss, key="all"):
    """PF at which the curve's running-min loss first reaches ``loss`` (linear between probes); None if never."""
    pf, ls = envelope(curve, key)
    if loss >= ls[0]:
        return 0.0
    for i in range(1, len(ls)):
        if ls[i] <= loss:
            if ls[i - 1] == ls[i]:
                return float(pf[i])
            w = (ls[i - 1] - loss) / (ls[i - 1] - ls[i])
            return float(pf[i - 1] + w * (pf[i] - pf[i - 1]))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--ref", default="dense,denselora", help="reference arms (cheapest per loss level)")
    ap.add_argument("--ref-tag", default="", help="tag holding the reference runs (default: any)")
    ap.add_argument("--min-frac", type=float, default=0.7, help="iso speedups are ranked only above this drop fraction")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    runs = load(a.results)
    refs = [r for r in runs.values() if r["arm"] in a.ref.split(",") and (not a.ref_tag or r["tag"] == a.ref_tag)]
    if not refs:
        raise SystemExit("no reference runs found")
    ref_seeds = sorted({r["seed"] for r in refs})

    def ref_for(seed):
        return [r for r in refs if r["seed"] == seed] or refs

    out = []
    for (tag, arm, seed), r in sorted(runs.items()):
        rs = ref_for(seed)
        last = r["curve"][-1]
        l0, la, fa = r["curve"][0]["all"], last["all"], last["pf"]
        # reference: per loss level the cheapest reference arm; its last-probe loss = the best any reference reached
        ref_last = min(x["curve"][-1]["all"] for x in rs)
        ref_last_pf = min((x["curve"][-1]["pf"] for x in rs if x["curve"][-1]["all"] <= ref_last + 1e-12))
        ref_peak = max(x["peak_gib"] for x in rs)
        need = [pf_to_reach(x["curve"], la) for x in rs]
        need = [n for n in need if n is not None]
        lb = not need
        ref_pf = min(need) if need else max(x["curve"][-1]["pf"] for x in rs)
        cross = pf_to_reach(r["curve"], ref_last)
        rec = {k: r[k] for k in ("tag", "arm", "seed", "steps", "eval_size", "peak_gib", "train_s", "probe_s",
                                 "wall_s", "node", "lr", "step0_src")}
        rec.update({
            "ce0": l0, "ce_last": la, "drop": l0 - la, "pf": fa, "dense_pf": last["dense_pf"],
            "pf_per_step_vs_dense": (fa / last["dense_pf"]) if last["dense_pf"] else None,
            "frac": (l0 - la) / max(1e-9, l0 - ref_last),
            "iso": (ref_pf / fa) if fa > 0 else None, "iso_lb": lb,
            "to_target": (ref_last_pf / cross) if cross else None,
            "peak_vs_ref": (r["peak_gib"] / ref_peak) if ref_peak else None,
            "by": {k: [c[k] for c in r["curve"]] for k in last if k.startswith(("task/", "set/"))},
            "curve": [[c["step"], c["pf"], c["all"]] for c in r["curve"]],
            "is_ref": r["arm"] in a.ref.split(",")})
        # paired SE of the loss difference vs the seed-matched reference rows (same held-out rows)
        best = min(rs, key=lambda x: x["curve"][-1]["all"])
        if len(best["ce_rows_last"]) == len(r["ce_rows_last"]):
            dlt = np.asarray(r["ce_rows_last"]) - np.asarray(best["ce_rows_last"])
            rec["d_vs_ref"] = float(dlt.mean())
            rec["d_vs_ref_se"] = float(dlt.std(ddof=1) / np.sqrt(len(dlt)))
        out.append(rec)

    # arm-level means over seeds
    arms = defaultdict(list)
    for r in out:
        arms[(r["tag"], r["arm"])].append(r)
    board = []
    for (tag, arm), rs in arms.items():
        def mean(k):
            v = [x[k] for x in rs if x.get(k) is not None]
            return float(np.mean(v)) if v else None
        board.append({"tag": tag, "arm": arm, "seeds": sorted(x["seed"] for x in rs), "steps": rs[0]["steps"],
                      "eval_size": rs[0]["eval_size"], "is_ref": rs[0]["is_ref"],
                      **{k: mean(k) for k in ("ce_last", "drop", "frac", "pf", "pf_per_step_vs_dense", "iso",
                                              "to_target", "peak_gib", "peak_vs_ref", "train_s", "wall_s",
                                              "d_vs_ref", "d_vs_ref_se")},
                      "iso_lb": any(x["iso_lb"] for x in rs)})
    board.sort(key=lambda b: (-(b["iso"] or 0) if (b["frac"] or 0) >= a.min_frac else 1e9 - (b["frac"] or 0)))
    print(f"reference seeds {ref_seeds}; ranked by iso-loss FLOP speedup among arms with frac >= {a.min_frac}")
    print(f"{'tag/arm':34s} {'sd':>4s} {'steps':>5s} {'CE':>6s} {'frac':>5s} {'PF':>7s} {'step/dense':>10s} "
          f"{'iso':>6s} {'toTgt':>6s} {'peakGiB':>7s} {'mem/ref':>7s} {'train_s':>7s}")
    f = lambda v, w, p: (f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-")  # noqa: E731
    for b in board:
        print(f"{(b['tag'] + '/' + b['arm'])[:34]:34s} {len(b['seeds']):>4d} {b['steps']:>5d} {f(b['ce_last'], 6, 3)} "
              f"{f(b['frac'], 5, 2)} {f(b['pf'], 7, 2)} {f(b['pf_per_step_vs_dense'], 10, 3)} "
              f"{f(b['iso'], 6, 2)}{'+' if b['iso_lb'] else ' '}{f(b['to_target'], 5, 2)} {f(b['peak_gib'], 7, 1)} "
              f"{f(b['peak_vs_ref'], 7, 2)} {f(b['train_s'], 7, 0)}")
    if a.out:
        json.dump({"board": board, "runs": out, "ref": a.ref, "min_frac": a.min_frac}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
