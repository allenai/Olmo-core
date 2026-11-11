"""Oracle threshold sweep (records/oracle-threshold-sweep.md): one table row per arm + the Pareto figure data.

Sources: scores from `ops result show --train-run` (8 cells per run); training PF and steps from the Beaker training log
(`flop_meter/actual_pflops`, the hub's meter: routed pass only for oracle arms); oracle training metadata from
`sweep_meta.jsonl` (METAJSON lines of dump_meta_beaker.sh; orc1doc from its mooney run folder); random arms' realised
keep and routed train CE from the Beaker log (`[online-router] ... keep_frac=`, `train/CE loss`). Logs are cached in
`sweep_cache.json` (only finished runs). Local copies (Lambda `-lam`, cubbins `-cub`) contribute METAJSON / FLOPJSON /
LOGJSON lines to sweep_meta.jsonl via local_meta.py; each arm uses whichever copy has the scores. FTM: dense PF needed for the arm's 8k mean, log-linear along the fresh dense
curve (LR 5e-6, the hub's rule), divided by the arm's PF; below the lowest dense point it is an upper bound (<=).

    python research/token_routing/oracle_train/collect_sweep.py [--json sweep_table.json] [--md]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_meta import parse_log  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
CACHE = os.path.join(HERE, "sweep_cache.json")
META = os.path.join(HERE, "sweep_meta.jsonl")
TASKS = ["contradiction", "nq", "outlier", "oolong"]
ORC1DOC_LOCAL = "/net/mooney/data/prasann/oracle/runs/f8k-mt4-orc1doc-f1952-orc-lr1e-05"

#: (label, train run, kind, attribution-cos target)
ARMS = [
    ("orc1doc (0.95)", "f8k-mt4-orc1doc-f1952-orc-lr1e-05", "oracle", 0.95),
    ("orc1doc90", "f8k-mt4-orc1doc90-f1952-orc-lr1e-05", "oracle", 0.90),
    ("orc1doc85", "f8k-mt4-orc1doc85-f1952-orc-lr1e-05", "oracle", 0.85),
    ("orc1doc80", "f8k-mt4-orc1doc80-f1952-orc-lr1e-05", "oracle", 0.80),
    ("orc1doc70", "f8k-mt4-orc1doc70-f1952-orc-lr1e-05", "oracle", 0.70),
    ("orc1doc60", "f8k-mt4-orc1doc60-f1952-orc-lr1e-05", "oracle", 0.60),
    ("per-row 0.90 + random x1.15 (row scope)", "f8k-mt4-orc1row90r115-f1952-orc-lr1e-05", "oracle_row", 0.90),
]
#: an arm may have trained as a Beaker run or as a local copy (Lambda `-lam`, cubbins `-cub`); the copy with scores wins
COPIES = ("", "-lam", "-cub")
RECIPE = ("recipe lora64all32r20to7", "f8k-mt4-lora64all32r20to7-f1952-tde-lr1e-05", "recipe", None)
DENSE = [f"f8k-mt4-dense-f{r}-tde-lr5e-06" for r in (488, 976, 1952, 3904, 5856, 7808)]


def sh(cmd, timeout=240):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


def scores(run):
    try:
        rows = json.loads(sh([OPS, "result", "show", "--train-run", run, "--json"]))
    except (json.JSONDecodeError, subprocess.TimeoutExpired):
        return {}
    out = {}
    for r in rows:
        if r.get("task") in TASKS and r.get("rung") in ("8k", "2k") and r.get("metric") in ("f1", "score", "accuracy"):
            out[(r["task"], r["rung"])] = (float(r["value"]), int(r.get("eval_size") or 0), r.get("cot_label"))
    return out


def ex_of(run):
    txt = sh([OPS, "runs", "--experiment", "tokdrop_lr", "--json"])
    try:
        rows = json.loads(txt)
    except json.JSONDecodeError:
        return None
    exs = [r.get("job_id") for r in rows if r.get("run_id") == run and r.get("kind", "train") == "train"]
    return exs[-1] if exs else None


def meta_line(kind, run):
    """the LAST `<kind> <run> {json}` line of sweep_meta.jsonl (a re-dump after more steps supersedes), or None"""
    hit = None
    if os.path.exists(META):
        for line in open(META):
            if line.startswith(f"{kind} {run} "):
                hit = json.loads(line.split(" ", 2)[2])
    return hit


def resolve(base, cache):
    """the copy of an arm (Beaker / -lam / -cub) that was evaluated: most score cells, then one with training data
    (a local LOGJSON line, or a cached finished Beaker log), then the Beaker name"""
    best = None
    for c in COPIES:
        run = base + c
        sc = scores(run)
        key = (len(sc), bool(meta_line("LOGJSON", run)) or run in cache)
        if best is None or key > best[0]:
            best = (key, run, sc)
    return best[1], best[2]


def train_log_stats(run, cache):
    """PF, steps, routed keep (router log) and train CE: a local copy's LOGJSON line (local_meta.py), else the Beaker
    training log; Beaker logs are cached once finished"""
    lj = meta_line("LOGJSON", run)
    if lj:
        return lj
    if run in cache:
        return cache[run]
    ex = ex_of(run)
    if not ex:
        return None
    log = sh([BEAKER, "experiment", "logs", ex], timeout=600)
    out = {"ex": ex, **parse_log(log)}
    if out["done"] and out["pf"]:
        cache[run] = out
    return out


def quarters(vals):
    n = len(vals)
    if n < 8:
        return [st.fmean(vals)] if vals else []
    return [st.fmean(vals[i * n // 4:(i + 1) * n // 4]) for i in range(4)]


def meta_of(run):
    """oracle windows (summarize.py --oneline) for a run: sweep_meta.jsonl, or orc1doc's local folder"""
    hit = meta_line("METAJSON", run)
    if hit:
        return hit
    if run.endswith("orc1doc-f1952-orc-lr1e-05") and os.path.isdir(ORC1DOC_LOCAL):
        out = sh(["python3", os.path.join(HERE, "summarize.py"), ORC1DOC_LOCAL, "--every", "100000", "--oneline", run])
        for line in out.splitlines():
            if line.startswith("METAJSON"):
                return json.loads(line.split(" ", 2)[2])
    return None


def flops_json_of(run):
    hit = meta_line("FLOPJSON", run)
    if hit:
        return hit
    if run.endswith("orc1doc-f1952-orc-lr1e-05"):
        p = os.path.join(ORC1DOC_LOCAL, "flops.json")
        return json.load(open(p)) if os.path.exists(p) else None
    return None


def mean_rung(sc, rung):
    v = [sc[(t, rung)][0] for t in TASKS if (t, rung) in sc]
    return sum(v) / len(v) if len(v) == 4 else None


def ftm(y, pf, dense):
    """(text, value) FLOPs-to-match: dense PF at score y (log-linear along the dense curve) / pf"""
    pts = sorted((d["pf"], d["y8"]) for d in dense if d["pf"] and d["y8"] is not None)
    if len(pts) < 2 or not pf or y is None:
        return "-", None
    if y >= max(p[1] for p in pts):
        return f">={pts[-1][0] / pf:.2f}x", pts[-1][0] / pf
    if y < pts[0][1]:
        return f"<={pts[0][0] / pf:.2f}x", pts[0][0] / pf
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if min(ya, yb) <= y <= max(ya, yb) and ya != yb:
            need = xa * (xb / xa) ** ((y - ya) / (yb - ya))
            return f"{need / pf:.2f}x", need / pf
    return "-", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "sweep_table.json"))
    ap.add_argument("--random", default="", help="comma list of random-control arms (rnd1doc<pct>) to include")
    a = ap.parse_args()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    arms = list(ARMS) + [(f"random {r[7:].replace('p', '.')}% ({r})", f"f8k-mt4-{r}-f1952-orc-lr1e-05", "random", None)
                         for r in a.random.split(",") if r] + [RECIPE]
    dense = []
    for base in DENSE:
        run, sc = resolve(base, cache)
        lg = train_log_stats(run, cache) or {}
        dense.append({"run": run, "rows": int(re.search(r"-f(\d+)-", run).group(1)), "pf": lg.get("pf"),
                      "steps": lg.get("steps"), "y8": mean_rung(sc, "8k"), "y2": mean_rung(sc, "2k"),
                      "out8": sc.get(("outlier", "8k"), (None,))[0], "cells": len(sc)})
    d1952 = next(d for d in dense if d["rows"] == 1952)
    rows = []
    for label, base, kind, target in arms:
        run, sc = resolve(base, cache)
        lg = train_log_stats(run, cache) or {}
        fj = flops_json_of(run)
        pf = lg.get("pf") or (fj or {}).get("actual_pflops")
        over = (pf / d1952["pf"]) if (pf and d1952["pf"]) else (fj or {}).get("actual_over_dense")
        m = meta_of(run) if kind.startswith("oracle") else None
        r = {"label": label, "run": run, "kind": kind, "target": target, "cells": len(sc), "pf": pf,
             "step_cost": over, "speedup": (1 / over) if over else None, "steps": lg.get("steps"),
             "resumes": lg.get("resumes"), "starts": lg.get("starts"),
             "y8": mean_rung(sc, "8k"), "y2": mean_rung(sc, "2k"),
             "t8": {t: sc.get((t, "8k"), (None,))[0] for t in TASKS},
             "t2": {t: sc.get((t, "2k"), (None,))[0] for t in TASKS},
             "eval_sizes": sorted({v[1] for v in sc.values()}), "cot": sorted({str(v[2]) for v in sc.values()})}
        if m:
            w = m["windows"]
            full = next(v for k, v in w.items() if k.startswith(f"steps{m['first_step']}-{m['last_step']}"))
            qs = [v for k, v in w.items() if v is not full][:4]
            r.update({"keep": full["keep_s0"], "keep_q": [q["keep_s0"] for q in qs], "attr": full["attr_s0"],
                      "gcos": full["gcos_lora"], "gcos_adam": full["gcos_adam_all"], "gnorm": full["gnorm_ratio"],
                      "ce_dense": full["ce_dense"], "ce_routed": full["ce_routed"],
                      "ce_q_dense": [q["ce_dense"] for q in qs], "ce_q_routed": [q["ce_routed"] for q in qs],
                      "oracle_routed_over_dense": full["routed_over_dense"],
                      "gold_keep": full.get("gold_keep_frac"), "distr_keep": full.get("distractor_keep_frac"),
                      "gold_orc_keep": full.get("gold_orc_keep_frac"),
                      "distr_orc_keep": full.get("distractor_orc_keep_frac"), "orc_share": full.get("orc_frac_body"),
                      "zero_docs": full.get("doc_zero_docs"), "zero_docs_orc": full.get("doc_zero_docs_orc"),
                      "orc_keep_q": [q.get("orc_frac_body") for q in qs],
                      "meta_steps": m["steps"]})
        elif lg.get("keep"):
            ks = [k for _, k in sorted(lg["keep"])]
            r.update({"keep": st.fmean(ks), "keep_q": quarters(ks)})
        if not kind.startswith("oracle") and lg.get("ce"):
            r["ce_routed"] = st.fmean(lg["ce"])
        r["ftm_txt"], r["ftm"] = ftm(r["y8"], pf, dense)
        rows.append(r)
    json.dump(cache, open(CACHE, "w"), indent=1)
    out = {"dense": dense, "arms": rows}
    json.dump(out, open(a.json, "w"), indent=1)

    def f(x, p=3):
        return "-" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))
    print("| arm | target | keep (q1..q4) | step cost | speedup | gcos raw/Adam | CE dense->routed | "
          "contra/nq/outlier/oolong 8k | 8k | 2k | PF | FTM |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in dense:
        print(f"| dense f{d['rows']} | - | 100% | {f(d['pf'] / d1952['pf'] if d['pf'] and d1952['pf'] else None, 2)} | - | - | - | - | "
              f"{f(d['y8'])} | {f(d['y2'])} | {f(d['pf'], 0)} | 1.0x |")
    for r in rows:
        kq = "/".join(f"{100 * q:.0f}" for q in r.get("keep_q", []))
        keep = f"{100 * r['keep']:.1f}% ({kq})" if r.get("keep") is not None else "-"
        print(f"| {r['label']} | {f(r['target'], 2)} | {keep} | {f(r['step_cost'], 3)} | {f(r['speedup'], 2)}x | "
              f"{f(r.get('gcos'), 2)}/{f(r.get('gcos_adam'), 2)} | {f(r.get('ce_dense'), 3)}->{f(r.get('ce_routed'), 3)} | "
              + "/".join(f(r["t8"][t]) for t in TASKS) + f" | {f(r['y8'])} | {f(r['y2'])} | {f(r['pf'], 0)} | {r['ftm_txt']} |"
              + (f" cells={r['cells']}" if r["cells"] != 8 else ""))


if __name__ == "__main__":
    main()
