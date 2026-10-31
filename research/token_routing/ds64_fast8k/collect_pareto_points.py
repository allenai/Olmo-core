"""Every fast8k token-drop training run (Lambda + Beaker) as one table of Pareto points, for the cross-task
plots on the tokdrop_lr artifact.

One row per (run, task, rung): score (the task's primary metric from the ladder log), eval_size, training PF
(``flop_meter/actual_pflops``; a dense run without the per-example meter gets steps x the per-step PF of the
longest per-example dense run on the same cluster), steps, wall-clock train seconds, cluster, and the parsed
arm / data / LR. Lambda runs come from the run registry (``ops runs``) and the Lambda logs; Beaker runs from
``launch_fast8k.py``'s state.json and ``beaker experiment logs``.

    python research/token_routing/ds64_fast8k/collect_pareto_points.py --out research/token_routing/ds64_fast8k/results/pareto_points.json
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_rule_screen import LOGS, scores, ssh, train_pf  # noqa: E402

OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
SCORE_RE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] [a-z0-9_]+=(-?[0-9.]+) \(n=(\d+)")


def parse_name(run):
    """f8k-mt4s[<rows>]-<arm>[-s<seed>][-lr<c>]-lr<lr>[-ep<E>][-s<steps>][-r<k>]-lam (Lambda) or
    f8k-mt4-<arm>-s<rows>[st<steps>]-tdb-lr<lr> (Beaker). Returns None for names it does not know."""
    m = re.match(r"f8k-(mt\d+)s(\d*)-(.+)$", run)
    if m:
        mix, rows, rest = m.groups()
        lr = re.search(r"-lr(\d[\d.]*(?:e-\d+)?)(?=-|$)", rest)
        if not lr:
            return None
        before, after = rest[: lr.start()].split("-"), rest[lr.end():].split("-")
        camp = before.pop() if re.fullmatch(r"lr[a-z]", before[-1]) else ""
        seed = int(before.pop()[1:]) if len(before) > 1 and re.fullmatch(r"s\d", before[-1]) else 0
        tag = lambda pre: next((int(t[len(pre):]) for t in after if re.fullmatch(pre + r"\d+", t)), 0)
        return dict(mix=mix, rows=int(rows or 488), arm="-".join(before), lr=float(lr.group(1)), campaign=camp,
                    seed=seed, epochs=tag("ep") or 1, max_steps=tag("s"))
    m = re.match(r"f8k-(mt\d+)-(.+)-[sf](\d+)(?:x(\d+))?(?:st(\d+))?-(tdb|tdc|tde|otr)-lr(\d[\d.]*(?:e-\d+)?)$", run)
    if m:
        mix, arm, rows, ep, steps, camp, lr = m.groups()
        seed = int(re.search(r"sd(\d+)$", arm).group(1)) if re.search(r"sd\d+$", arm) else 0
        return dict(mix=mix, rows=int(rows), arm=arm, lr=float(lr), campaign=camp, seed=seed, epochs=int(ep or 1),
                    max_steps=int(steps or 0))
    return None


def lambda_runs():
    """{run: registered log_path} for Lambda training runs of tokdrop_lr and learned_router (the online-router
    session runs from its own staged checkout, so its logs are not under LOGS)."""
    out = {}
    for exp in ("tokdrop_lr", "learned_router"):
        txt = subprocess.run([OPS, "runs", "--experiment", exp, "--cluster", "lambda", "--json"],
                             capture_output=True, text=True, timeout=60).stdout
        for r in (json.loads(txt) if txt.strip().startswith("[") else []):
            if r.get("kind") == "train" and r["run_id"].startswith("f8k-mt"):
                out[r["run_id"]] = r.get("log_path") or ""
    return out


def train_pf_from(path, run=""):
    """(PF, last step) from a Lambda training log outside LOGS. The registry may prefix the path with ``lambda:`` or
    leave it empty; then the online-router session's staged checkouts (OLMo-core-otr*) are searched by run name."""
    path = (path or "").split(":", 1)[-1] if (path or "").startswith("lambda:") else (path or "")
    if not path:
        path = f"$(ls -t /accounts/projects/sewonm/prasann/ctc_suite/OLMo-core-otr*/logs/{run}_*.log 2>/dev/null | head -1)"
    out = ssh(f"f={path}; [ -n \"$f\" ] && [ -f \"$f\" ] && grep -E 'flop_meter/actual_pflops|\\[step=' \"$f\" | tail -40")
    pf = [float(x) for x in re.findall(r"flop_meter/actual_pflops[ =]+([0-9.]+)", out)]
    steps = [int(x) for x in re.findall(r"\[step=(\d+)/", out)]
    return (pf[-1] if pf else None), (max(steps) if steps else None)


def lambda_wall(run):
    out = ssh(f"cd {LOGS}; f=$(ls -t ctc_suite_{run}_*.log 2>/dev/null | head -1); j=${{f##*_}}; j=${{j%.log}}; "
              f"sacct -X -n -P -o ElapsedRaw,State -j $j 2>/dev/null | head -1")
    try:
        sec, state = out.strip().split("|")[:2]
        return int(sec), state
    except ValueError:
        return None, None


ROUTER_STATE = ("origin/claude/online-router-1bc01a", "research/token_routing/online_router/state.json")
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z.*\[step=(\d+)/", re.M)


def _ops_scores(run):
    out = subprocess.run([OPS, "result", "show", "--train-run", run, "--json"], capture_output=True, text=True,
                         timeout=60).stdout
    rows = json.loads(out) if out.strip().startswith("[") else []
    # other sessions record probes against the same train run (grad_composition: 64-row held-out evals at 8k), so
    # keep the largest eval per (task, rung) and never let a sub-500 probe shadow the 500-row suite eval
    best = {}
    for r in rows:
        m = str(r.get("metric", ""))
        if m.startswith("gradcomp_") or r.get("experiment") == "grad_composition":
            continue  # gradient statistics, not task scores
        if r.get("task") in ("mt4", "mt11") or r.get("experiment") == "gradgap" or m.startswith(("gap", "gradgap")):
            continue  # gradient-gap probe metrics recorded against the train run (task = mix name), not task scores
        if m.startswith("answer_") or "_ce" in m or m.endswith("top1"):
            continue  # teacher-forced answer CE / top-1 probes (decay diagnosis) share train runs; not task scores
        k = (r["task"], r["rung"])
        if k not in best or (r["eval_size"] or 0) > best[k][1]:
            best[k] = (r["value"], r["eval_size"] or 0)
    return best


def _beaker_train(ex):
    """(actual PF, last step, train seconds first->last step line) from a Beaker training log."""
    import datetime as dt
    log = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=300).stdout
    pf = re.findall(r"flop_meter/actual_pflops[ =]+([0-9.]+)", log)
    steps = TS_RE.findall(log)
    wall = None
    if len(steps) > 1:
        t = [dt.datetime.fromisoformat(a[:26]) for a, _ in (steps[0], steps[-1])]
        wall = (t[1] - t[0]).total_seconds()
    return (float(pf[-1]) if pf else None), (max(int(b) for _, b in steps) if steps else None), wall


def beaker_runs():
    """Beaker training runs scored in ops results: this launcher's -tdc runs (state.json) and the online-router
    session's runs (its state.json on its branch). Scores come from ops results (any experiment), PF / steps /
    wall-clock from the training log."""
    srcs = []
    st_path = os.path.join(HERE, "state.json")
    if os.path.exists(st_path):
        srcs += [(k, v) for k, v in json.load(open(st_path))["runs"].items() if "-tdc" in k or "-tde" in k]
    try:
        rs = subprocess.run(["git", "-C", HERE, "show", ":".join(ROUTER_STATE)], capture_output=True, text=True,
                            timeout=60).stdout
        srcs += list(json.loads(rs)["runs"].items())
    except Exception as e:  # noqa: BLE001
        print("router state unavailable:", e)
    runs, rows = {}, []
    for run, v in srcs:
        meta, ex = parse_name(run), v.get("ex")
        if meta is None or not ex:
            continue
        sc = _ops_scores(run)
        if not sc:
            continue
        pf, steps, wall = _beaker_train(ex)
        runs[run] = dict(meta, run=run, cluster="beaker-h100", pf=pf, steps=steps, per_example=True,
                         train_seconds=wall, gpus=2, dense=meta["arm"] == "dense", beaker_ex=ex)
        rows += [dict(run=run, task=t, rung=r, score=x, eval_size=n) for (t, r), (x, n) in sc.items()]
    return runs, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-beaker", action="store_true")
    a = ap.parse_args()
    rows, runs = [], {}
    for run, log_path in lambda_runs().items():
        meta = parse_name(run)
        if meta is None:
            continue
        # where the TRAINING log lives decides how PF is read; the shared LOGS dir only for runs registered there
        # (the online-router session's eval logs can land in LOGS while its training logs do not)
        outside = bool(log_path) and not log_path.split(":", 1)[-1].startswith(LOGS)
        sc = (_ops_scores(run) or scores(run)) if outside else (scores(run) or _ops_scores(run))
        if not sc:
            continue
        pf = steps = None
        if not outside:  # registered under LOGS, or unregistered: try the shared logs dir first
            pf, steps, per_example = train_pf(run)
        if pf is None:
            pf, steps = train_pf_from(log_path, run)
            per_example = True
        wall, state = lambda_wall(run)
        # 2026-10-22 option-field bug: lrb runs named tier* kept the keep set WHOLE (--st-kept-doc-k dropped)
        # and lrb *m<N>p8 runs lost their --st-keep-token-min floor. Never plot them under their names.
        if meta["campaign"] == "lrb" and (meta["arm"].startswith("tier") or re.search(r"m\d+p8$", meta["arm"])):
            meta["label_warning"] = "lrb flag-drop bug: ran without --st-kept-doc-k / --st-keep-token-min"
            meta["arm"] = meta["arm"] + "~lrbbug"
        runs[run] = dict(meta, run=run, cluster="lambda-a100", pf=pf, steps=steps, per_example=per_example,
                         train_seconds=wall, gpus=2, dense=meta["arm"] == "dense")  # dense-sk3 etc. are rules, not the reference
        for (task, rung), (v, n) in sc.items():
            rows.append(dict(run=run, task=task, rung=rung, score=v, eval_size=n))
    # old dense runs without the per-example meter: scale per-step PF from the longest per-example dense run
    for r in runs.values():
        if r["dense"] and not r["per_example"] and r["steps"]:
            # per-step PF depends on row lengths: borrow it only from a per-example dense run on the SAME shard
            # (an mt11 run's shorter rows once halved the mt4 122-step dense run's PF)
            same = [q for q in runs.values() if q["dense"] and q["per_example"] and q["pf"] and q["steps"]
                    and q.get("mix") == r.get("mix") and q.get("rows") == r.get("rows")]
            if same:
                ref = max(same, key=lambda q: q["steps"])
                r["pf"] = ref["pf"] / ref["steps"] * r["steps"]
                r["pf_scaled"] = True
    if not a.no_beaker:
        b_runs, b_rows = beaker_runs()
        runs.update(b_runs)
        rows += b_rows
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(runs=runs, points=rows), open(a.out, "w"), indent=1)
    print(f"{len(runs)} runs, {len(rows)} points -> {a.out}")


if __name__ == "__main__":
    main()
