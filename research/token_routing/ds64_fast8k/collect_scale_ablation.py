"""Model-scale ablation of the token-drop recipe (tokdrop_lr, 2026-10-29): FLOPs-to-match against each scale's
OWN dense curve, metered PF per step and peak memory, Qwen3.5 0.8B / 2B (campaigns ``-tds08b`` / ``-tds2b``, this
launcher's state.json) beside the 4B ``-tde`` runs (the hub's pareto_points.json).

Score = the 8-cell mean (contradiction / nq / outlier / oolong x 2k / 8k, eval_size 500, no-cot), read from ops
results. PF = ``flop_meter/actual_pflops`` of the finished run; peak memory = the max ``gpu_memory/GPU reserved mem
(GiB)`` (rank 0) over the training log. FLOPs-to-match = dense PF at the recipe's score (piecewise log-linear along
the dense curve, first segment extrapolated below it, as make_overview_page.ftm_rows) / recipe PF.

    python research/token_routing/ds64_fast8k/collect_scale_ablation.py --out research/token_routing/ds64_fast8k/results/scale_ablation.json
"""

import argparse
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
PTS_4B = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/research/token_routing/ds64_fast8k/results/pareto_points.json"
CELLS = [(t, r) for t in ("contradiction", "nq", "outlier", "oolong") for r in ("2k", "8k")]
CAMPAIGNS = {"-tds08b": "0.8b", "-tds2b": "2b", "-tds9b": "9b"}
RECIPE = {"0.8b": "lora64all24r20to7s12", "2b": "lora64all24r20to7s12", "4b": "lora64all32r20to7", "9b": "lora64all32r20to7"}
NAME = re.compile(r"f8k-mt4-(.+)-f(\d+)(?:st(\d+))?-(tds\w+|tde)-lr([\d.e-]+)$")
LOGDIR = os.path.join(HERE, "results", "scale_logs")
#: the generation metric of each task's ladder eval (f1; oolong reports `score`)
GEN_METRICS = ("f1", "score")


def ops_scores(run):
    out = subprocess.run([OPS, "result", "show", "--train-run", run, "--json"], capture_output=True, text=True,
                         timeout=120).stdout
    best = {}
    for r in json.loads(out) if out.strip().startswith("[") else []:
        if r.get("metric") not in GEN_METRICS:  # other sessions add teacher-forced answer_ce / answer_top1 rows
            continue
        k = (r["task"], r["rung"])
        if (r.get("eval_size") or 0) >= 500 and (k not in best or r["eval_size"] > best[k][1]):
            best[k] = (r["value"], r["eval_size"])
    return {k: v[0] for k, v in best.items()}


def beaker_train(ex):
    """(actual PF, last step, max reserved GiB, train seconds) from the (cached) Beaker training log."""
    import datetime as dt
    os.makedirs(LOGDIR, exist_ok=True)
    path = os.path.join(LOGDIR, f"{ex}.log")
    if not os.path.exists(path):
        log = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=600).stdout
        if "flop_meter/actual_pflops" in log:  # cache finished logs only
            open(path, "w").write(log)
    else:
        log = open(path).read()
    pf = re.findall(r"flop_meter/actual_pflops[ =]+([0-9.]+)", log)
    mem = [float(x) for x in re.findall(r"gpu_memory/GPU reserved mem \(GiB\)[ =]+([0-9.]+)", log)]
    steps = re.findall(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z.*\[step=(\d+)/", log, re.M)
    wall = None
    if len(steps) > 1:
        t = [dt.datetime.fromisoformat(a[:26]) for a, _ in (steps[0], steps[-1])]
        wall = (t[1] - t[0]).total_seconds()
    return (float(pf[-1]) if pf else None, max(int(b) for _, b in steps) if steps else None,
            max(mem) if mem else None, wall)


LOCAL = "/net/sneetches/data/prasann/scale_abl"  # scale_run_local.sbatch (jsteinhardt) runs; /net read = small files only


def local_train(run):
    """(actual PF, last step, max reserved GiB, train seconds) of a finished LOCAL run (flops.json + train log)."""
    import datetime as dt
    fj, log = f"{LOCAL}/runs/{run}/flops.json", f"{LOCAL}/logs/train_{run}.log"
    if not (os.path.exists(fj) and os.path.exists(f"{LOCAL}/runs/{run}/config.json") and os.path.exists(log)):
        return None, None, None, None
    txt = open(log, errors="replace").read()
    mem = [float(x) for x in re.findall(r"gpu_memory/GPU reserved mem \(GiB\)[ =]+([0-9.]+)", txt)]
    steps = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+.*\[step=(\d+)/", txt, re.M)
    wall = None
    if len(steps) > 1:
        t = [dt.datetime.fromisoformat(a) for a, _ in (steps[0], steps[-1])]
        wall = (t[1] - t[0]).total_seconds()
    return (json.load(open(fj))["actual_pflops"], max(int(b) for _, b in steps) if steps else None,
            max(mem) if mem else None, wall)


def need_pf(dpts, y):
    for (xa, ya), (xb, yb) in zip(dpts, dpts[1:]):
        if min(ya, yb) <= y <= max(ya, yb) and yb != ya:
            return xa * (xb / xa) ** ((y - ya) / (yb - ya))
    (xa, ya), (xb, yb) = dpts[0], dpts[1]
    if y > max(v for _, v in dpts):
        return None  # above the dense curve's best point: lower bound only
    return xa * (xb / xa) ** ((y - ya) / (yb - ya))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    st = json.load(open(os.path.join(HERE, "state.json")))
    for run, v in st["runs"].items():
        m = NAME.match(run)
        if not m or v.get("state") != "DONE" or (m.group(3) and (m.group(3) == "4" or m.group(1) != "dense")):  # skip smoke; keep dense step-cuts
            continue
        arm, n, _, camp, lr = m.groups()
        scale = CAMPAIGNS.get("-" + camp)
        if scale is None:
            continue
        pf, steps, mem, wall = beaker_train(v["ex"])
        rows.append(dict(scale=scale, run=run, arm=arm, rows=int(n), lr=float(lr), pf=pf, steps=steps, mem=mem,
                         wall=wall, ex=v["ex"], cells=ops_scores(run)))
    for d in sorted(os.listdir(f"{LOCAL}/runs")) if os.path.isdir(f"{LOCAL}/runs") else []:
        m = NAME.match(d)
        if not m or CAMPAIGNS.get("-" + m.group(4)) is None or (m.group(3) and m.group(1) != "dense"):
            continue
        pf, steps, mem, wall = local_train(d)
        if pf is None:
            continue
        arm, n, _, camp, lr = m.groups()
        rows.append(dict(scale=CAMPAIGNS["-" + camp], run=d, arm=arm, rows=int(n), lr=float(lr), pf=pf, steps=steps,
                         mem=mem, wall=wall, ex=f"local:{d}", cells=ops_scores(d)))
    seen = {}
    for r in rows:
        seen.setdefault(r["run"], r)  # Beaker (state.json) rows come first; a same-name local twin is dropped
    rows[:] = list(seen.values())
    P = json.load(open(PTS_4B))
    cells = {}
    for p in P["points"]:
        if p["eval_size"] >= 500:
            cells.setdefault(p["run"], {})[(p["task"], p["rung"])] = p["score"]
    for run, r in P["runs"].items():
        m = NAME.match(run)
        if not m or m.group(4) != "tde" or m.group(3) or m.group(1) not in ("dense", RECIPE["4b"]):
            continue
        pf, steps, mem, wall = beaker_train(r["beaker_ex"]) if r.get("beaker_ex") else (r["pf"], r["steps"], None, None)
        rows.append(dict(scale="4b", run=run, arm=m.group(1), rows=int(m.group(2)), lr=r["lr"], pf=pf or r["pf"],
                         steps=steps or r["steps"], mem=mem, wall=wall or r.get("train_seconds"), ex=r.get("beaker_ex"),
                         cells=ops_scores(run)))  # ops, not pareto_points.json (which can carry non-task metrics)
    for r in rows:
        r["seed"] = int(re.search(r"sd(\d+)$", r["arm"]).group(1)) if re.search(r"sd\d+$", r["arm"]) else 0
        r["arm"] = re.sub(r"sd\d+$", "", r["arm"])  # seed replicates count as the same arm
        r["y"] = round(sum(r["cells"][c] for c in CELLS) / 8, 4) if all(c in r["cells"] for c in CELLS) else None
        r["pf_step"] = round(r["pf"] / r["steps"], 4) if r["pf"] and r["steps"] else None
        r["cells"] = {f"{t}@{g}": s for (t, g), s in r["cells"].items()}
    out = {"rows": rows, "ftm": []}
    # seed replicates of one (scale, arm, lr, pf) point are averaged before any curve / FTM is computed
    agg = {}
    for r in rows:
        if r["y"] is not None and r["pf"]:
            agg.setdefault((r["scale"], r["arm"], r["lr"], round(r["pf"], -1), r["rows"]), []).append(r)
    pts_all = [dict(scale=k[0], arm=k[1], lr=k[2], rows=k[4], pf=sum(x["pf"] for x in v) / len(v),
                    y=round(sum(x["y"] for x in v) / len(v), 4), ys=[x["y"] for x in v], n_seeds=len(v),
                    runs=[x["run"] for x in v]) for k, v in agg.items()]
    out["points"] = pts_all
    for scale in sorted({p["scale"] for p in pts_all}):
        dense = {}
        for p in pts_all:
            if p["scale"] == scale and p["arm"] == "dense":
                dense.setdefault(p["lr"], []).append((p["pf"], p["y"]))
        for p in sorted(pts_all, key=lambda p: (p["lr"], p["pf"])):
            if p["scale"] != scale or p["arm"] != RECIPE[scale]:
                continue
            for lr, pts in sorted(dense.items()):
                if len(pts) < 2:
                    continue
                pts = sorted(pts)
                need = need_pf(pts, p["y"])
                out["ftm"].append(dict(scale=scale, rows=p["rows"], recipe_lr=p["lr"], y=p["y"], ys=p["ys"], pf=p["pf"],
                                       dense_lr=lr, dense_pts=pts, extrapolated=p["y"] < pts[0][1],
                                       ftm=round(need / p["pf"], 2) if need else None,
                                       ftm_lower_bound=None if need else round(pts[-1][0] / p["pf"], 2)))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{'scale':5} {'run':58} {'y':>6} {'PF':>7} {'PF/step':>8} {'memGiB':>7}")
    for r in sorted(rows, key=lambda r: (r["scale"], r["arm"], r["lr"], r["rows"])):
        print(f"{r['scale']:5} {r['run']:58} {r['y'] if r['y'] is not None else '-':>6} "
              f"{r['pf'] or 0:7.0f} {r['pf_step'] or 0:8.3f} {r['mem'] or 0:7.1f}")
    for f in out["ftm"]:
        txt = f"{f['ftm']}x" if f["ftm"] else f">{f['ftm_lower_bound']}x"
        print(f"FTM {f['scale']:5} f{f['rows']} recipe lr{f['recipe_lr']:g} y={f['y']} (seeds {f['ys']}) vs dense "
              f"lr{f['dense_lr']:g} ({len(f['dense_pts'])} pts): {txt}" + ("  [extrapolated below dense range]" if f["extrapolated"] else ""))


if __name__ == "__main__":
    main()
