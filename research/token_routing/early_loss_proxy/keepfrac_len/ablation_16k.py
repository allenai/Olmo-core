"""16k gold-drop ablation: 1x-pool arms (kl16kabl) and more-data arms (kl16kx) vs the kl16k references (dft, recipe k20,
k40) on the kl16k held-out set (48 rows/task). Per arm: steps, FLOPs/step vs dense, total PF, final CE (mean + per
task), CE at matched training FLOPs (running-min curve, linear interpolation; '-' if the arm spent over 2% less), and FLOP
speedup vs dft and vs the recipe at the arm's own lowest mean CE, or at the reference's lowest CE when the arm gets
below it (PF_ref / PF_arm to reach that level; 'not reached' if the reference never gets that low).

    python3 ablation_16k.py --root /tmp/claude-3018/keepfrac_len_results [--pf 59,138,333] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

T = ["contradiction", "outlier", "qdmatch_nq"]


def load(p):
    d = json.load(open(p))
    pr = [q for q in d["probes"] if q.get("ce")]
    rows = d["rows"]
    by = {t: np.array([np.mean([c for r, c in zip(rows, q["ce"]) if r["task"] == t]) for q in pr]) for t in T}
    last = pr[-1]["flops"]
    return {"steps": [q["step"] for q in pr], "pf": np.array([q["flops"]["actual_pf"] for q in pr]),
            "ce": np.array([np.mean(q["ce"]) for q in pr]), "by": by, "x_dense": last["actual_pf"] / last["dense_pf"],
            "eval_size": len(rows), "done": d.get("done"), "path": p}


def at_pf(r, pf, c=None):
    c = np.minimum.accumulate(r["ce"] if c is None else c)
    return None if pf > r["pf"][-1] * 1.02 else float(np.interp(min(pf, r["pf"][-1]), r["pf"], c))  # 2% slack for the last probe


def reach(r, lvl, c=None):
    c = np.minimum.accumulate(r["ce"] if c is None else c)
    if c[0] <= lvl:
        return float(r["pf"][0])
    for i in range(1, len(c)):
        if c[i] <= lvl:
            return float(r["pf"][i - 1] + (c[i - 1] - lvl) / (c[i - 1] - c[i]) * (r["pf"][i] - r["pf"][i - 1]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pf", default="59,138,333")
    ap.add_argument("--json", default="")
    ap.add_argument("--rows", default="", help="ops result rows (jsonl) for the kl16kabl / kl16kx arms")
    a = ap.parse_args()
    runs = {}
    for name, p in [("dft", "kl16k/elp-kl16k-dft-sd0.json"), ("k20 (recipe) sd0", "kl16k/elp-kl16k-k20-sd0.json"),
                    ("k20 (recipe) sd1", "kl16k/elp-kl16k-k20-sd1.json"), ("k40 sd0", "kl16k/elp-kl16k-k40-sd0.json"),
                    ("k40 sd1", "kl16k/elp-kl16k-k40-sd1.json")]:
        if os.path.exists(f"{a.root}/{p}"):
            runs[name] = load(f"{a.root}/{p}")
    for p in sorted(glob.glob(f"{a.root}/kl16kabl/elp-kl16kabl-*-sd0.json")) + sorted(glob.glob(f"{a.root}/kl16kx/elp-kl16kx-*-sd0.json")):
        n = os.path.basename(p)[4:-5]
        runs[n] = load(p)
    dft, rec = runs["dft"], runs["k20 (recipe) sd0"]
    pfs = [float(x) for x in a.pf.split(",")]
    out = {}
    hdr = f"{'arm':38s} {'steps':>5s} {'x/dense':>7s} {'PF':>6s} {'final':>6s} " + " ".join(f"{t[:6]:>6s}" for t in T) + " | " + \
        " ".join(f"CE@{p:g}PF" for p in pfs) + " | x_dft x_rec at max(own min, ref min) (min CE) node"
    print(hdr)
    for n, r in runs.items():
        mn = float(np.min(r["ce"]))
        # level = the arm's lowest CE, or the reference's own lowest if the arm gets below it (then: FLOPs to match
        # the reference's best)
        ld, lr_ = max(mn, float(np.min(dft["ce"]))), max(mn, float(np.min(rec["ce"])))
        xd, pd_ = reach(dft, ld), reach(r, ld)
        xr, pr_ = reach(rec, lr_), reach(r, lr_)
        e = {"steps": r["steps"][-1], "x_dense_step": r["x_dense"], "pf": float(r["pf"][-1]), "final_ce": float(r["ce"][-1]),
             "final_by_task": {t: float(r["by"][t][-1]) for t in T}, "min_ce": mn,
             "ce_at_pf": {f"{p:g}": at_pf(r, p) for p in pfs},
             "task_ce_at_pf": {f"{p:g}": {t: at_pf(r, p, r["by"][t]) for t in T} for p in pfs},
             "x_vs_dft_at_min": (xd / pd_) if xd and pd_ else None, "x_vs_rec_at_min": (xr / pr_) if xr and pr_ else None,
             "level_dft": ld, "level_rec": lr_,
             "node": json.load(open(r["path"])).get("slurm", {}).get("SLURMD_NODENAME"),
             "eval_size": r["eval_size"], "done": r["done"], "path": r["path"]}
        out[n] = e
        f = lambda v, d=4: "   -  " if v is None else f"{v:.{d}f}"  # noqa: E731
        print(f"{n:38s} {e['steps']:5d} {e['x_dense_step']:7.3f} {e['pf']:6.0f} {e['final_ce']:.4f} "
              + " ".join(f"{e['final_by_task'][t]:.3f}" for t in T) + " | "
              + " ".join(f"{f(e['ce_at_pf'][f'{p:g}']):>9s}" for p in pfs)
              + f" | {f(e['x_vs_dft_at_min'], 2):>6s} {f(e['x_vs_rec_at_min'], 2):>6s} ({mn:.4f}) {e['node']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    if a.rows:
        with open(a.rows, "w") as g:
            for n, e in out.items():
                if not n.startswith(("kl16kabl", "kl16kx")):
                    continue
                d = json.load(open(e["path"]))
                sl = d.get("slurm", {})
                lam = str(sl.get("SLURMD_NODENAME", "")).startswith("lambda")
                base = {"experiment": "tokdrop_lr", "run_id": f"elp-{n}-probe",
                        "train_run": f"elp-{n.split('-')[0]}-{'lambda' if lam else 'job'}{sl.get('SLURM_JOB_ID')}",
                        "rung": "16k", "cot_label": "no-cot", "eval_job_id": sl.get("SLURM_JOB_ID"),
                        "eval_backend": "elp-dense-probe(teacher-forced answer CE, dense forward)",
                        "ckpt_path": "none (early-loss proxy: no checkpoint is saved)",
                        "git_commit": os.path.basename(sl.get("CODE", "")).replace("code_", "") or None}
                for t in T:
                    g.write(json.dumps({**base, "task": t, "metric": "heldout_ce_dense_final", "value": round(e["final_by_task"][t], 5), "eval_size": 48}) + "\n")
                g.write(json.dumps({**base, "task": "mean3", "metric": "heldout_ce_dense_final", "value": round(e["final_ce"], 5), "eval_size": e["eval_size"]}) + "\n")
                for pf, v in e["ce_at_pf"].items():
                    if v is not None:
                        g.write(json.dumps({**base, "task": "mean3", "metric": f"heldout_ce_dense_at_{pf}pf", "value": round(v, 5), "eval_size": e["eval_size"]}) + "\n")
                for k, mt in (("x_vs_dft_at_min", "flop_x_vs_dft_at_own_min_ce"), ("x_vs_rec_at_min", "flop_x_vs_recipe_at_own_min_ce")):
                    if e[k] is not None:
                        g.write(json.dumps({**base, "task": "mean3", "metric": mt, "value": round(e[k], 3), "eval_size": e["eval_size"]}) + "\n")


if __name__ == "__main__":
    main()
