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


def ops_scores(run):
    out = subprocess.run([OPS, "result", "show", "--train-run", run, "--json"], capture_output=True, text=True,
                         timeout=120).stdout
    best = {}
    for r in json.loads(out) if out.strip().startswith("[") else []:
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
        if not m or v.get("state") != "DONE" or m.group(3):  # skip smoke / step-cut runs
            continue
        arm, n, _, camp, lr = m.groups()
        scale = CAMPAIGNS.get("-" + camp)
        if scale is None:
            continue
        pf, steps, mem, wall = beaker_train(v["ex"])
        rows.append(dict(scale=scale, run=run, arm=arm, rows=int(n), lr=float(lr), pf=pf, steps=steps, mem=mem,
                         wall=wall, ex=v["ex"], cells=ops_scores(run)))
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
                         cells=cells.get(run, {})))
    for r in rows:
        r["y"] = round(sum(r["cells"][c] for c in CELLS) / 8, 4) if all(c in r["cells"] for c in CELLS) else None
        r["pf_step"] = round(r["pf"] / r["steps"], 4) if r["pf"] and r["steps"] else None
        r["cells"] = {f"{t}@{g}": s for (t, g), s in r["cells"].items()}
    out = {"rows": rows, "ftm": []}
    for scale in sorted({r["scale"] for r in rows}):
        dense = {}
        for r in rows:
            if r["scale"] == scale and r["arm"] == "dense" and r["y"] is not None and r["pf"]:
                dense.setdefault(r["lr"], []).append((r["pf"], r["y"]))
        for r in rows:
            if r["scale"] != scale or r["arm"] != RECIPE[scale] or r["y"] is None or not r["pf"]:
                continue
            for lr, pts in sorted(dense.items()):
                if len(pts) < 2:
                    continue
                pts = sorted(pts)
                need = need_pf(pts, r["y"])
                out["ftm"].append(dict(scale=scale, run=r["run"], rows=r["rows"], y=r["y"], pf=r["pf"], dense_lr=lr,
                                       dense_pts=pts, ftm=round(need / r["pf"], 2) if need else None,
                                       ftm_lower_bound=None if need else round(pts[-1][0] / r["pf"], 2)))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{'scale':5} {'run':58} {'y':>6} {'PF':>7} {'PF/step':>8} {'memGiB':>7}")
    for r in sorted(rows, key=lambda r: (r["scale"], r["arm"], r["lr"], r["rows"])):
        print(f"{r['scale']:5} {r['run']:58} {r['y'] if r['y'] is not None else '-':>6} "
              f"{r['pf'] or 0:7.0f} {r['pf_step'] or 0:8.3f} {r['mem'] or 0:7.1f}")
    for f in out["ftm"]:
        txt = f"{f['ftm']}x" if f["ftm"] else f">{f['ftm_lower_bound']}x"
        print(f"FTM {f['scale']:5} f{f['rows']} recipe y={f['y']} vs dense lr{f['dense_lr']:g}: {txt}")


if __name__ == "__main__":
    main()
