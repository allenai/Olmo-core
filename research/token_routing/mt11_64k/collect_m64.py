"""Collect the mt11_64k runs: training FLOPs / wall-clock and the per-task, per-rung ladder scores, then the
matched-FLOP comparison of every non-dense arm against the dense curve.

* Training: ``flop_meter/actual_pflops`` (last logged value), steps, and wall-clock (Beaker job started -> exited,
  plus the trainer's own step timing) from each FINISHED run's Beaker log (a mid-run cache freezes a partial FLOP
  meter, so unfinished runs are skipped).
* Evals: every ``[ladder:<task>@<rung>] <metric>=<v> (n=<N>`` line of each eval job's log.
* Dense reference: per (task, rung) and budget, the better of the two LRs (the brief: "the dense reference must be
  the per-cell better of the two LRs"); the curve is interpolated LINEARLY IN log PF. A router point outside the
  dense PF range is extrapolated from the end segment and flagged ``extrapolated``.
* ``five_rung_mean``: the mean over the rungs a task has (2k/8k/16k/32k/64k; fewer for reorder, xabsence, strmatch,
  grouping), for the arm and for the interpolated dense at the arm's PF.

Writes ``results/points.json`` (raw), ``results/matched.json`` (deltas) and, with ``--record``, appends every eval
row not yet in ops results via ``ops result add --json ... --check``.

    python3 research/token_routing/mt11_64k/collect_m64.py [--record]
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
CKPTS = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts"
CACHE = os.path.join(HERE, "results", "logs")
SCORE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ([a-z0-9_]+)=(-?[0-9.]+) \(n=(\d+)")
PF = re.compile(r"flop_meter/actual_pflops[\"'=: ]+([0-9.]+(?:e[+-]?\d+)?)")
STEP = re.compile(r"\[step=(\d+)/(\d+)")
KEY_TASK = {"contra": "contradiction"}
RUNG_ORDER = ["2k", "8k", "16k", "32k", "64k"]


def beaker_job(ex):
    out = subprocess.run([BEAKER, "experiment", "get", ex, "--format", "json"], capture_output=True, text=True,
                         timeout=120).stdout
    try:
        return json.loads(out)[0]["jobs"][-1]["status"]
    except Exception:  # noqa: BLE001
        return {}


def log_of(ex, final):
    """Beaker log text; cached on disk only once the job is finalized."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{ex}.log")
    if os.path.exists(path):
        return open(path, errors="replace").read()
    txt = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=900).stdout
    if final:
        open(path, "w").write(txt)
    return txt


def _ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "")[:26]) if s else None


def train_point(name, r):
    st = beaker_job(r["ex"])
    if not st.get("finalized") or st.get("exitCode") != 0:
        return None
    log = log_of(r["ex"], True)
    pf = [float(x) for x in PF.findall(log)]
    steps = [int(a) for a, _ in STEP.findall(log)]
    wall = (_ts(st.get("exited")) - _ts(st.get("started"))).total_seconds() if st.get("started") else None
    return dict(run=name, arm=r["arm"], budget=r["budget"], lr=r["lr"], pf=pf[-1] if pf else None,
                steps=max(steps) if steps else None, wall_s=wall, job=r["ex"])


def eval_rows(ename, e):
    st = beaker_job(e["ex"])
    final = bool(st.get("finalized"))
    out = []
    for t, rung, metric, v, n in SCORE.findall(log_of(e["ex"], final)):
        out.append(dict(eval_run=ename, train_run=e["train_run"], task=KEY_TASK.get(t, t), rung=rung, metric=metric,
                        value=float(v), eval_size=int(n), eval_job=e["ex"], final=final))
    return out


def interp_log(xs_ys, x):
    """Linear interpolation in log x over sorted (x, y); returns (y, extrapolated)."""
    pts = sorted(p for p in xs_ys if p[0])
    if len(pts) < 2:
        return None, True
    lx = math.log(x)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (lx - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + w * (y1 - y0), False
    (x0, y0), (x1, y1) = (pts[0], pts[1]) if x < pts[0][0] else (pts[-2], pts[-1])
    w = (lx - math.log(x0)) / (math.log(x1) - math.log(x0))
    return y0 + w * (y1 - y0), True


def matched(train, scores):
    pf = {t["run"]: t["pf"] for t in train if t["pf"]}
    by = {}
    for s in scores:
        by.setdefault((s["task"], s["rung"]), []).append(s)
    out = []
    for (task, rung), rows in sorted(by.items()):
        # dense reference: average the seeds of each (budget, LR), THEN take the better LR per budget. (Taking the max
        # over seeds instead -- the first version -- picks the luckier of two noisy draws and biases dense upward.)
        by_cfg = {}
        for s in rows:
            t = next((t for t in train if t["run"] == s["train_run"]), None)
            if not t or t["arm"] != "dense" or not t["pf"]:
                continue
            by_cfg.setdefault((t["budget"], t["lr"]), []).append((t["pf"], s["value"]))
        dense = {}  # budget -> (pf, seed-mean score of the better LR, lr)
        for (budget, lr), v in by_cfg.items():
            pf_, val = sum(p for p, _ in v) / len(v), sum(x for _, x in v) / len(v)
            if budget not in dense or val > dense[budget][1]:
                dense[budget] = (pf_, val, lr)
        curve = [(p, v) for p, v, _ in dense.values()]
        for s in rows:
            t = next((t for t in train if t["run"] == s["train_run"]), None)
            if not t or t["arm"] == "dense" or not pf.get(t["run"]):
                continue
            ref, extrap = interp_log(curve, pf[t["run"]])
            out.append(dict(task=task, rung=rung, run=t["run"], arm=t["arm"], budget=t["budget"], lr=t["lr"],
                            pf=pf[t["run"]], score=s["value"], dense_at_pf=ref, eval_size=s["eval_size"],
                            delta=None if ref is None else s["value"] - ref, extrapolated=extrap,
                            dense_points={b: dict(pf=p, score=v, lr=lr) for b, (p, v, lr) in dense.items()}))
    # five-rung mean per (run, task)
    means = {}
    for m in out:
        means.setdefault((m["run"], m["task"]), []).append(m)
    five = []
    for (run, task), ms in sorted(means.items()):
        ok = [m for m in ms if m["dense_at_pf"] is not None]
        if ok:
            five.append(dict(run=run, task=task, rungs=[m["rung"] for m in ok],
                             score=sum(m["score"] for m in ok) / len(ok),
                             dense_at_pf=sum(m["dense_at_pf"] for m in ok) / len(ok),
                             delta=sum(m["delta"] for m in ok) / len(ok),
                             extrapolated=any(m["extrapolated"] for m in ok)))
    return out, five


def record(scores, train):
    have_txt = subprocess.run([OPS, "result", "show", "--json", "--experiment", "mt11_64k"], capture_output=True,
                              text=True, timeout=120).stdout
    have = {(r.get("train_run"), r["task"], r["rung"]) for r in
            (json.loads(have_txt) if have_txt.strip().startswith("[") else [])}
    reg_txt = subprocess.run([OPS, "runs", "--experiment", "mt11_64k", "--json"], capture_output=True, text=True,
                             timeout=120).stdout
    commit = {r["run_id"]: r.get("git_commit") for r in (json.loads(reg_txt) if reg_txt.strip().startswith("[") else [])}
    rows = []
    for s in scores:
        if not s["final"] or (s["train_run"], s["task"], s["rung"]) in have:
            continue
        v, n = s["value"], s["eval_size"]
        se = math.sqrt(max(v * (1 - v), 1e-9) / n) if 0 <= v <= 1 else None
        rows.append(dict(run_id=s["eval_run"], train_run=s["train_run"], experiment="mt11_64k", task=s["task"],
                         rung=s["rung"], metric=s["metric"], value=v, eval_size=n, se=se and round(se, 6),
                         cot_label="no-cot", eval_backend="olmo-core-native-docchunk-ladder suite-rows (beaker)",
                         ckpt_path=f"{CKPTS}/{s['train_run']}", eval_job_id=s["eval_job"],
                         git_commit=commit.get(s["train_run"])))
    if not rows:
        print("nothing new to record")
        return
    path = os.path.join(HERE, "results", "new_rows.jsonl")
    open(path, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    r = subprocess.run([OPS, "result", "add", "--json", path, "--check"], capture_output=True, text=True)
    print(f"recorded {len(rows)} rows: " + "\n".join((r.stdout + r.stderr).strip().splitlines()[-5:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    st = json.load(open(os.path.join(HERE, "state.json")))
    train = [p for p in (train_point(n, r) for n, r in st["runs"].items() if r.get("ex")) if p]
    scores = [row for n, e in st["evals"].items() if e.get("ex") for row in eval_rows(n, e)]
    pts, five = matched(train, scores)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(train=train, scores=scores), open(os.path.join(HERE, "results", "points.json"), "w"), indent=1)
    json.dump(dict(per_rung=pts, five_rung=five), open(os.path.join(HERE, "results", "matched.json"), "w"), indent=1)
    for t in sorted(train, key=lambda t: (t["arm"], t["pf"] or 0)):
        sps = t["wall_s"] / t["steps"] if t["wall_s"] and t["steps"] else None
        print(f"{t['run']:28} PF {t['pf'] or float('nan'):9.1f} steps {t['steps']} wall {t['wall_s'] or 0:7.0f}s "
              f"({sps or 0:.1f} s/step incl. load)")
    for f in five:
        print(f"{f['run']:28} {f['task']:14} mean[{','.join(f['rungs'])}] {f['score']:.3f} vs dense {f['dense_at_pf']:.3f}"
              f" -> {f['delta']:+.3f}{' (extrap)' if f['extrapolated'] else ''}")
    if a.record:
        record(scores, train)


if __name__ == "__main__":
    main()
