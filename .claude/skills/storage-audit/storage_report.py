#!/usr/bin/env python3
"""
What storage am I using, where, and what could be cleaned up? — in seconds.

Live (instant):  quota on /accounts + /scratch, df of every node's /data (via /net, 3 s timeout).
From indexes:    per-directory size / file count / last-touched for node-local /data/prasann,
                 /accounts home, /scratch, and weka — written by `storage_index.py` running where
                 the disk is local. Walking these live takes minutes (weka) to hours (/net), so
                 the report never walks; it shows each index's age and `--refresh` rebuilds them
                 in the background.

    python3 storage_report.py                     # report from current indexes (~1-3 s)
    python3 storage_report.py --refresh           # rebuild ALL indexes in background, then report
    python3 storage_report.py --refresh weka      # just weka (or: nodes, home)
    python3 storage_report.py --idle-days 60 --min-gb 50
    python3 storage_report.py --json              # machine-readable, for agents
    python3 storage_report.py --html              # also write an HTML view

Read-only. It never deletes anything; it only names candidates.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import html
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
INDEXER = os.path.join(HERE, "storage_index.py")
CACHE = os.path.expanduser("~/.cache/storage-audit")
ME = os.environ.get("USER", "prasann")
HOME = os.path.realpath(os.path.expanduser("~"))
SCRATCH = f"/scratch/users/{ME}"
BEAKER = os.path.expanduser("~/.local/bin/beaker")

# Berkeley compute nodes (each exports its local /data at /net/<node>/data).
NODES = ["balrog", "cubbins", "feanor", "horton", "lorax", "mcfuzz", "mooney", "rainbowquartz",
         "saruman", "shadowfax", "smaug", "smokyquartz", "sneetches", "sunstone", "thidwick"]
MY_PARTITIONS = ["jsteinhardt", "berkeleynlp"]
NODE_ROOT = f"/data/{ME}"
NODE_INDEX = f"{NODE_ROOT}/.storage_index.json"

WEKA_ROOTS = ["/weka/oe-training-default/ai2-llm/checkpoints/prasanns",
              "/weka/oe-training-default/prasanns",
              "/weka/oe-training-default/ai2-llm/checkpoints/prasanntestsingle"]
BEAKER_WORKSPACE, BEAKER_BUDGET = "ai2/flex2", "ai2/oe-other"
BEAKER_IMAGE = "tylerr/olmo-core-tch291cu128-2025-11-25"
BEAKER_CPU_CLUSTERS = ["ai2/saturn", "ai2/neptune", "ai2/ceres", "ai2/jupiter"]

# Never suggest these without a loud "PROTECTED" tag (from the weka/node cleanup audits).
PROTECT = [
    (r"/ctc_suite/bases(/|$)", "base checkpoints"),
    (r"/ctc_olmo3/bases/.*-fixmark", "marker-repaired bases"),
    (r"xlong5_2k256k_qwen35", "weka is the ONLY copy"),
    (r"contra_mix_.*_10k_128k", "training shards"),
    (r"_eval_bundle", "eval bundles"),
    (r"ctc-olmo3-7b-hpqaret-", "retrain owed"),
    (r"/conda(/|$)|/envs/|_venv(/|$)|/\.venv(/|$)", "python env in use"),
    (r"vllm_serving", "vLLM serving copies"),
    (r"/\.claude(-work)?(/|$)", "Claude config"),
]
STALE_DAYS = 7
NOW = time.time()


# ----------------------------------------------------------------------------- helpers

def quick(cmd: list[str], t: float = 3.0) -> str | None:
    """Run a command that may touch NFS; give up after t seconds without waiting on a D-state."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             start_new_session=True)
    except OSError:
        return None
    try:
        out, _ = p.communicate(timeout=t)
        return out if p.returncode == 0 else None
    except subprocess.TimeoutExpired:
        p.kill()  # don't wait(): a process stuck in NFS D-state may never reap
        return None


def gb(b: float | None) -> str:
    if b is None:
        return "?"
    if b >= 2**40:
        return f"{b / 2**40:.2f}T"
    if b >= 2**30:
        return f"{b / 2**30:.0f}G" if b >= 10 * 2**30 else f"{b / 2**30:.1f}G"
    return f"{b / 2**20:.0f}M"


def age(t: float | None) -> str:
    if not t:
        return "?"
    d = (NOW - t) / 86400
    return "<1h" if d < 1 / 24 else f"{d * 24:.0f}h" if d < 1 else f"{d:.0f}d"


def protected(path: str) -> str:
    for rx, why in PROTECT:
        if re.search(rx, path):
            return why
    return ""


def load_json(path: str, t: float = 5.0) -> dict | None:
    if path.startswith("/net/"):
        out = quick(["cat", path], t)
        return json.loads(out) if out else None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- live sources

def live_quota() -> dict:
    out = quick(["quota", "-s"], 5) or ""
    q = {}
    for line in out.splitlines():  # tab-separated; values contain spaces ("11 G")
        f = [x.strip().replace(" ", "") for x in line.split("\t")]
        if len(f) >= 3 and f[0] in ("accounts", "scratch"):
            try:
                pct = float(f[3])
            except (IndexError, ValueError):
                pct = None
            q[f[0]] = dict(used=f[1], quota=f[2], pct=pct)
    return q


def parse_df(out: str | None) -> dict | None:
    if not out:
        return None
    p = out.strip().splitlines()[-1].split()
    return dict(size=int(p[1]), used=int(p[2]), avail=int(p[3]))


def node_state(node: str) -> dict:
    d = dict(node=node, df=parse_df(quick(["df", "-P", "-B1", f"/net/{node}/data"], 3)))
    if d["df"] is None:
        d["reachable"] = False
        return d
    d["reachable"] = True
    d["has_dir"] = quick(["test", "-d", f"/net/{node}{NODE_ROOT}"], 3) is not None
    d["index"] = load_json(f"/net/{node}{NODE_INDEX}") if d["has_dir"] else None
    prog = quick(["tail", "-1", f"/net/{node}{NODE_INDEX}.progress"], 3) if d["has_dir"] else None
    d["progress"] = (prog or "").strip()
    return d


# ----------------------------------------------------------------------------- weka (Beaker)

def weka_sync() -> dict:
    """Fetch a finished weka index job's result into the cache. Returns job status info."""
    job = load_json(f"{CACHE}/weka_job.json")
    if not job:
        return dict(state="never run")
    idx = load_json(f"{CACHE}/weka.json")
    if idx and idx.get("generated", 0) >= job.get("created", 0):
        return dict(state="done", experiment=job["experiment"])
    out = quick([BEAKER, "experiment", "get", job["experiment"], "--format", "json"], 8)
    if not out:
        return dict(state="unknown (beaker unreachable)", experiment=job["experiment"])
    exp = json.loads(out)
    exp = exp[0] if isinstance(exp, list) else exp
    j = (exp.get("jobs") or [{}])[-1]
    st = j.get("status", {})
    if not st.get("finalized"):
        return dict(state="running" if st.get("started") else "queued",
                    experiment=job["experiment"])
    if st.get("exitCode", 1) != 0:
        return dict(state=f"FAILED (exit {st.get('exitCode')})", experiment=job["experiment"])
    ds = j.get("result", {}).get("beaker") or j.get("assignmentDetails", {}).get("resultDatasetId")
    if not ds:
        return dict(state="finished, no result dataset", experiment=job["experiment"])
    tmp = f"{CACHE}/weka_fetch"
    quick([BEAKER, "dataset", "fetch", ds, "--output", tmp, "--prefix", "weka.json"], 20)
    if os.path.exists(f"{tmp}/weka.json"):
        os.replace(f"{tmp}/weka.json", f"{CACHE}/weka.json")
        return dict(state="done (just fetched)", experiment=job["experiment"])
    return dict(state="finished, fetch failed", experiment=job["experiment"])


def launch_weka() -> str:
    src = base64.b64encode(open(INDEXER, "rb").read()).decode()
    roots = " ".join(f"--root {r}" for r in WEKA_ROOTS)
    script = (f"set -e; export PATH=/opt/conda/bin:$PATH; echo {src} | base64 -d > /tmp/idx.py; "
              f"python3 /tmp/idx.py --label weka {roots} --out /results/weka.json "
              f"--max-depth 4; ls -la /results")
    spec = {
        "version": "v2", "budget": BEAKER_BUDGET,
        "description": "storage-audit weka index (read-only)",
        "tasks": [{
            "name": "main", "image": {"beaker": BEAKER_IMAGE},
            "command": ["bash", "-c", script],
            "datasets": [{"mountPath": "/weka/oe-training-default",
                          "source": {"weka": "oe-training-default"}}],
            "result": {"path": "/results"},
            "resources": {"cpuCount": 4},
            # minRuntime 1h: urgent jobs at 0s sit pending for hours/days (memory: beaker-min-runtime).
            "context": {"priority": "urgent", "minRuntime": "1h"},
            "constraints": {"cluster": BEAKER_CPU_CLUSTERS},
        }],
    }
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/weka_spec.json"
    with open(path, "w") as fh:
        json.dump(spec, fh)
    name = f"storage-index-weka-{time.strftime('%Y%m%d-%H%M%S')}"
    out = subprocess.run([BEAKER, "experiment", "create", path, "-w", BEAKER_WORKSPACE,
                          "-n", name, "--format", "json"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return f"weka: launch FAILED: {out.stderr.strip()[:300]}"
    exp = json.loads(out.stdout)
    exp = exp[0] if isinstance(exp, list) else exp
    with open(f"{CACHE}/weka_job.json", "w") as fh:
        json.dump(dict(experiment=exp["id"], created=NOW, name=name), fh)
    return f"weka: launched beaker experiment {exp['id']} (urgent, 0 GPU) — ~5-30 min"


# ----------------------------------------------------------------------------- refresh

def launch_home() -> list[str]:
    os.makedirs(CACHE, exist_ok=True)
    msgs = []
    for label, root in (("accounts", HOME), ("scratch", SCRATCH)):
        if not os.path.isdir(root):
            continue
        log = open(f"{CACHE}/{label}.log", "w")
        subprocess.Popen(["nice", "-n", "19", "ionice", "-c3", sys.executable, INDEXER,
                          "--label", label, "--root", root, "--out", f"{CACHE}/{label}.json",
                          "--max-depth", "6", "--min-dir-gb", "0.25"],
                         stdout=log, stderr=log, start_new_session=True)
        msgs.append(f"{label}: background walk of {root} started (niced; NFS, a few minutes)")
    return msgs


def launch_nodes(states: list[dict], only: list[str] | None) -> list[str]:
    parts = {}
    for line in (quick(["sinfo", "-h", "-N", "-o", "%N %P"], 5) or "").splitlines():
        n, p = line.split()[:2]
        if p.rstrip("*") in MY_PARTITIONS:
            parts[n] = p.rstrip("*")
    busy = set((quick(["squeue", "-h", "-u", ME, "-n", "storage-index", "-o", "%N %R"], 5)
                or "").split())
    msgs = []
    for s in states:
        n = s["node"]
        if only and n not in only:
            continue
        if not s.get("has_dir") or n not in parts:
            continue
        if n in busy:
            msgs.append(f"{n}: index job already queued/running")
            continue
        cmd = (f"nice -n 19 ionice -c3 python3 {INDEXER} --label {n} --root {NODE_ROOT} "
               f"--out {NODE_INDEX}")
        out = subprocess.run(
            ["sbatch", "--parsable", "-p", parts[n], "--qos=preemptive", "-w", n, "-c", "1",
             "--mem=4G", "--time=12:00:00", "-J", "storage-index",
             f"--output={NODE_ROOT}/.storage_index.log", "--wrap", cmd],
            capture_output=True, text=True, timeout=30)
        msgs.append(f"{n}: sbatch job {out.stdout.strip()}" if out.returncode == 0 else
                    f"{n}: sbatch FAILED {out.stderr.strip()[:200]}")
    return msgs


# ----------------------------------------------------------------------------- analysis

def candidates(locs: list[dict], idle_days: float, min_gb: float,
               under: str | None = None) -> tuple[list, list, list]:
    """Maximal idle dirs, redundant stepN dirs, and idle large files outside listed dirs.

    With `under`, only paths strictly below that prefix count, so the picks are its children
    (drill-down into a big candidate)."""
    pre = under.rstrip("/") + "/" if under else ""
    idle_s, min_b = idle_days * 86400, min_gb * 2**30
    cands, redundant, files = [], [], []
    for loc in locs:
        idx = loc.get("index")
        if not idx:
            continue
        merged: dict[str, dict] = {}
        for p, b, f, t, depth, flag in idx["dirs"]:
            m = merged.setdefault(p, dict(path=p, bytes=b, files=f, newest=t, depth=depth,
                                          flags=set(), loc=loc["name"]))
            if flag:
                m["flags"].add(flag)
        for m in merged.values():
            m["flags"] = sorted(m["flags"])
            m["protected"] = protected(m["path"])
            if (any(f.startswith("redundant_stepN") for f in m["flags"])
                    and (not pre or m["path"].startswith(pre))):
                redundant.append(m)
        picked: list[str] = []
        for m in sorted(merged.values(), key=lambda m: m["depth"]):
            if m["depth"] == 0 or m["bytes"] < min_b or NOW - m["newest"] < idle_s:
                continue
            if pre and not m["path"].startswith(pre):
                continue
            if any(m["path"].startswith(p + "/") for p in picked):
                continue
            picked.append(m["path"])
            cands.append(m)
        for p, b, t in idx.get("files", []):
            if (b >= min_b and NOW - t >= idle_s and (not pre or p.startswith(pre))
                    and not any(p.startswith(q + "/") for q in picked)):
                files.append(dict(path=p, bytes=b, newest=t, loc=loc["name"],
                                  protected=protected(p)))
    key = lambda m: -m["bytes"]  # noqa: E731
    return sorted(cands, key=key), sorted(redundant, key=key), sorted(files, key=key)


# ----------------------------------------------------------------------------- render

USE_COLOR = sys.stdout.isatty()


def c(t: str, k: str) -> str:
    code = dict(h="1", dim="2", red="31", yel="33", grn="32", cyan="36").get(k, "0")
    return f"\033[{code}m{t}\033[0m" if USE_COLOR else t


def bar(used: int, size: int, width: int = 14) -> str:
    if not size:
        return " " * width
    u = min(width, round(width * used / size))
    k = "red" if used / size > 0.9 else "yel" if used / size > 0.75 else "grn"
    return c("█" * u, k) + c("░" * (width - u), "dim")


def render(res: dict, a: argparse.Namespace) -> str:
    L = [c(f"STORAGE AUDIT  {time.strftime('%Y-%m-%d %H:%M')}   candidates = idle ≥ "
           f"{a.idle_days:g}d and ≥ {a.min_gb:g}G (newest mtime anywhere below)"
           + (f", under {a.under}" if a.under else ""), "h")]
    if res.get("refresh"):
        L += [c("refresh launched:", "cyan")] + [f"  {m}" for m in res["refresh"]] + [""]
    L.append(c(f"  {'location':<22}{'mine':>8}  {'filesystem':<32}{'index':<12}candidates", "h"))
    for loc in res["locations"]:
        fs = loc.get("fs")
        fs_s = (bar(fs["used"], fs["size"]) + f" {gb(fs['avail']) + ' free/' + gb(fs['size']):<18} "
                if fs else f"{loc.get('fs_note', ''):<32}")
        idx = loc.get("index")
        ia = ("none" if not idx else age(idx["generated"]))
        stale = (not idx) or (NOW - idx["generated"] > STALE_DAYS * 86400)
        ia_s = c(f"{ia:<12}", "yel" if stale else "dim")
        nc = [m for m in res["candidates"] if m["loc"] == loc["name"]]
        cs = f"{len(nc)} · {gb(sum(m['bytes'] for m in nc))}" if nc else ""
        L.append(f"  {loc['name']:<22}{loc.get('mine', '?'):>8}  {fs_s}{ia_s}{cs:<14}"
                 + (c("  " + loc["note"], "dim") if loc.get("note") else ""))

    def row(m: dict, extra: str = "") -> str:
        tag = c(f"  PROTECTED: {m['protected']}", "red") if m.get("protected") else ""
        return (f"  {gb(m['bytes']):>7} {age(m['newest']):>5}  {m['loc']:<10} {m['path']}"
                + (c(f"  {extra}", "dim") if extra else "") + tag)

    L.append("")
    tot = sum(m["bytes"] for m in res["candidates"] if not m["protected"])
    L.append(c(f"CLEANUP CANDIDATES — largest idle directories (unprotected total {gb(tot)})", "h"))
    L.append(c(f"  {'size':>7} {'idle':>5}  {'where':<10} path", "dim"))
    for m in res["candidates"][:a.top]:
        L.append(row(m, f"{m['files']:,} files"))
    if len(res["candidates"]) > a.top:
        L.append(c(f"  … {len(res['candidates']) - a.top} more (--top N)", "dim"))

    if res["redundant"]:
        tot = sum(m["bytes"] for m in res["redundant"] if not m["protected"])
        L.append("")
        L.append(c(f"REDUNDANT CHECKPOINTS — stepN dirs eval never reads (any age; total "
                   f"{gb(tot)} in {len(res['redundant'])} dirs)", "h"))
        for m in res["redundant"][:min(a.top, 15)]:
            L.append(row(m, m["flags"][0].split(": ", 1)[-1]))
    if res["files"]:
        L.append("")
        L.append(c("LARGE IDLE FILES outside the directories above", "h"))
        for m in res["files"][:10]:
            L.append(row(m))

    L.append("")
    for loc in res["locations"]:
        if loc.get("warn"):
            L.append(c(f"  ⚠ {loc['name']}: {loc['warn']}", "yel"))
    L.append(c("  Node /data is ZFS with ~4 weekly snapshots: deletes free `du` now but `df` only "
               "after snapshots roll (≤4 wks).", "dim"))
    L.append(c("  Deleting: use the two-stage plan/apply tooling in OLMo-core "
               "debug/storage_cleanup/ (explicit manifests, no globs) — never rm from this list.",
               "dim"))
    L.append(c(f"  ({res['elapsed_s']}s)", "dim"))
    return "\n".join(L)


def render_html(text: str) -> str:
    css = {"1": "font-weight:700", "2": "color:#8a8a8a", "31": "color:#d23c3c",
           "33": "color:#c48a00", "32": "color:#2e9e4f", "36": "color:#1f8fb3"}
    body, pos = [], 0
    for m in re.finditer(r"\033\[(\d+)m(.*?)\033\[0m", text, re.S):
        body.append(html.escape(text[pos:m.start()]))
        body.append(f'<span style="{css.get(m.group(1), "")}">{html.escape(m.group(2))}</span>')
        pos = m.end()
    body.append(html.escape(text[pos:]))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Storage Audit</title>
<style>:root{{--bg:#fbfbfa;--fg:#222}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#16171a;--fg:#e4e4e4}}}}
:root[data-theme="dark"]{{--bg:#16171a;--fg:#e4e4e4}}
body{{background:var(--bg);color:var(--fg);margin:0;padding:16px}}
pre{{font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;overflow-x:auto}}
</style></head><body><pre>{''.join(body)}</pre></body></html>"""


# ----------------------------------------------------------------------------- main

def main() -> None:
    global USE_COLOR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", nargs="?", const="all", choices=["all", "nodes", "weka", "home"],
                    help="rebuild indexes in the background (default: all)")
    ap.add_argument("--nodes", help="comma list of nodes to report on / refresh")
    ap.add_argument("--idle-days", type=float, default=30)
    ap.add_argument("--min-gb", type=float, default=10)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--under", help="drill down: only candidates below this path prefix "
                    "(e.g. /data/prasann/ctc_suite); combine with --nodes to pick a node")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--html", nargs="?", const="__default__")
    ap.add_argument("--plain", action="store_true")
    a = ap.parse_args()
    USE_COLOR = (USE_COLOR or bool(a.html)) and not a.plain and not a.json

    with cf.ThreadPoolExecutor(len(NODES) + 2) as ex:
        f_q = ex.submit(live_quota)
        f_w = ex.submit(weka_sync)
        f_n = {n: ex.submit(node_state, n) for n in NODES}
        f_jobs = ex.submit(quick, ["squeue", "-h", "-u", ME, "-n", "storage-index",
                                   "-o", "%T|%j|%r|%M|%n|%N"], 5)
        quota, weka_job = f_q.result(), f_w.result()
        states = [f.result() for f in f_n.values()]
        node_jobs = {}
        for line in (f_jobs.result() or "").splitlines():
            st, _, reason, el, req, run_on = (line.split("|") + [""] * 6)[:6]
            node_jobs[run_on or req] = (f"index job RUNNING {el}" if st == "RUNNING"
                                        else f"index job {st.lower()} ({reason})")

    res: dict = dict(time=NOW, refresh=[])
    if a.refresh in ("all", "home"):
        res["refresh"] += launch_home()
    if a.refresh in ("all", "nodes"):
        res["refresh"] += launch_nodes(states, a.nodes.split(",") if a.nodes else None)
    if a.refresh in ("all", "weka"):
        res["refresh"].append(launch_weka())

    locs = []
    for label, root, key in (("accounts (home)", HOME, "accounts"), ("scratch", SCRATCH, "scratch")):
        q = quota.get(key, {})
        df = parse_df(quick(["df", "-P", "-B1", root], 3))
        idx = load_json(f"{CACHE}/{key}.json")
        loc = dict(name=key, mine=q.get("used", "?"), index=idx,
                   fs_note=f"quota {q.get('quota', '?')} ({q.get('pct', '?')}% used)",
                   note=f"shared fs {gb(df['used'])}/{gb(df['size'])} used" if df else "")
        if q.get("pct") and q["pct"] > 85:
            loc["warn"] = f"quota at {q['pct']}% — writes will start failing with EDQUOT"
        locs.append(loc)
    only = set(a.nodes.split(",")) if a.nodes else None
    for s in states:
        if not s["reachable"] or (only and s["node"] not in only):
            continue
        idx = s.get("index")
        if not s.get("has_dir") and not idx:
            continue
        mine = None
        if idx:
            mine = (idx.get("zfs", {}).get("userspace", {}).get(ME)
                    or sum(t.get("bytes", 0) for t in idx["totals"].values()))
        loc = dict(name=s["node"], mine=gb(mine) if mine else "?", fs=s["df"], index=idx)
        if idx and idx.get("zfs", {}).get("datasets"):
            snap = max(int(d.get("usedbysnap", 0) or 0) for d in idx["zfs"]["datasets"])
            if snap > 100 * 2**30:
                loc["note"] = f"node-wide: {gb(snap)} pinned by ZFS snapshots"
        if s["df"] and s["df"]["used"] / max(1, s["df"]["size"]) > 0.95:
            loc["warn"] = f"/data {100 * s['df']['used'] / s['df']['size']:.0f}% full"
        if s["node"] in node_jobs:
            loc["note"] = node_jobs[s["node"]] + (
                f" — {s['progress'].split('  at ')[0][9:].strip()}" if s.get("progress") else "")
        elif not idx:
            loc["note"] = "no index yet (--refresh nodes)"
        locs.append(loc)
    widx = load_json(f"{CACHE}/weka.json")
    wdf = next(iter((widx or {}).get("df", {}).values()), None)
    locs.append(dict(name="weka", index=widx,
                     mine=gb(sum(t.get("bytes", 0) for t in widx["totals"].values()))
                     if widx else "?",
                     fs=wdf, note=f"refresh job: {weka_job['state']}"
                     + (f" ({weka_job['experiment']})" if weka_job.get("experiment") else "")))
    for loc in locs:
        idx = loc.get("index")
        if not idx and loc["name"] in ("accounts", "scratch", "weka"):
            loc.setdefault("warn", "no index — run with --refresh")
        elif idx and NOW - idx["generated"] > STALE_DAYS * 86400:
            loc.setdefault("warn", f"index is {age(idx['generated'])} old — --refresh")

    res["locations"] = locs
    res["candidates"], res["redundant"], res["files"] = candidates(
        locs, a.idle_days, a.min_gb, a.under)
    res["elapsed_s"] = round(time.time() - NOW, 2)

    if a.json:
        out = dict(res)
        out["locations"] = [{k: v for k, v in loc.items() if k != "index"}
                            | {"index_generated": (loc.get("index") or {}).get("generated")}
                            for loc in locs]
        print(json.dumps(out, indent=1, default=str))
        return
    text = render(res, a)
    if a.html:
        path = a.html
        if path == "__default__":
            path = ("visualizations/storage_audit.html" if os.path.isdir("visualizations")
                    else f"{CACHE}/storage_audit.html")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write(render_html(text))
        if not sys.stdout.isatty():
            text = re.sub(r"\033\[\d+m", "", text)
        print(text + f"\n\nwrote {path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
