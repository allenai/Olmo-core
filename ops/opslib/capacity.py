"""``ops capacity``: everyone's GPUs per node on Berkeley slurm (jsteinhardt, berkeleynlp) and Lambda.

``ops status`` shows only our jobs, so a busy node can look idle. 2026-10-20..11-10 other users held
20.5 of Lambda's 24 working GPUs on average and 29 of their jobs ended within minutes of one of our
preemptive starts (records/gpu-usage-workflow-audit-2026-11.md). This view shows, per node: total
GPUs, our GPUs by QOS, other users' GPUs by user and QOS, and what is free; our use of the per-user
caps (8 GPUs at preemptive_high on jsteinhardt; 8 + 8 at preemptive_high / preemptive on Lambda),
split by experiment via the run registry; and ``--would-preempt``: whether a new job of N GPUs at a
QOS fits in free GPUs or would preempt someone (and whom).

One sinfo + two squeue + one sacctmgr call per cluster (Lambda: one ssh). Read-only; never submits.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

from . import jobs

ME = jobs.USER
BERKELEY_PARTITIONS = ("jsteinhardt", "berkeleynlp")
SQ_FMT = "%i|%u|%P|%q|%b|%N|%D|%M|%j"
# per-user GPU caps we promised to keep (CLAUDE.md); slurm also enforces MaxTRESPU on preemptive_high.
# None = shown, not capped: on jsteinhardt the cap is 8 GPUs at preemptive_high ONLY, and more may run at
# preemptive when capacity is available (it bumps normal and is bumped by preemptive_high; Prasann, 2026-11-10).
CAPS = {"berkeley": {"preemptive_high": 8, "preemptive": None}, "lambda": {"preemptive_high": 8, "preemptive": 8}}
# fallback when sacctmgr is unreadable: which QOS each QOS may preempt
DEFAULT_PREEMPTS = {"preemptive_high": {"normal", "preemptive"}, "preemptive": {"normal"},
                    "normal": {"preemptible"}}
# "nothing below A100" (memory: a100-ask-first): placement checks skip other GPU types unless --any-gpu
GOOD_GPU_TYPES = {"A100", "H100", "H200", "B200", "GB200"}
UNAVAILABLE = re.compile(r"down|drain|drng|fail|maint|inval|unk|\*", re.I)


def script(partitions=None) -> str:
    p = f"-p {','.join(partitions)}" if partitions else ""
    return (f"sinfo -N -h {p} -o '%N|%P|%G|%T'; echo __SQ__; "
            f"squeue -a -h -t R {p} -o '{SQ_FMT}'; echo __TRES__; "
            f"squeue -a -h -t R {p} -O 'JobID:24,tres-alloc:160'; echo __QOS__; "
            "sacctmgr -P -n show qos format=Name,Preempt 2>/dev/null")


def fetch(cluster: str, runner=None) -> str:
    """Raw snapshot text for ``berkeley`` (local slurm) or ``lambda`` (one ssh call)."""
    runner = runner or jobs.run
    if cluster == "lambda":
        rc, out, err = runner(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "lambda", script()], timeout=15)
    else:
        rc, out, err = runner(["bash", "-c", script(BERKELEY_PARTITIONS)], timeout=15)
    if rc != 0 and "__SQ__" not in (out or ""):
        raise RuntimeError(f"{cluster} capacity snapshot rc={rc}: {(err or '').strip()[:160]}")
    return out


def expand_nodelist(spec: str) -> list:
    """'a,b' / 'lambda-hyperplane[01-02,05]' / 'mooney' -> node names (one bracket group per item)."""
    out = []
    for item in re.findall(r"[^,\[]+(?:\[[^\]]*\])?", spec or ""):
        item = item.strip(",")
        m = re.fullmatch(r"(.*)\[([^\]]+)\](.*)", item)
        if not m:
            if item and item not in ("(null)", "None"):
                out.append(item)
            continue
        pre, body, post = m.groups()
        for part in body.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                out += [f"{pre}{i:0{len(a)}d}{post}" for i in range(int(a), int(b) + 1)]
            else:
                out.append(f"{pre}{part}{post}")
    return out


def _gpu_count(text: str):
    m = re.search(r"gpu(?::[A-Za-z0-9_.-]+)?[:=](\d+)", text or "")
    return int(m.group(1)) if m else None


def parse(text: str) -> dict:
    """-> {"nodes": {node: {partition, gpus, state, available}}, "jobs": [...], "preempts": {qos: set}}"""
    sections = {"SINFO": [], "SQ": [], "TRES": [], "QOS": []}
    cur = "SINFO"
    for line in (text or "").splitlines():
        s = line.strip()
        if s in ("__SQ__", "__TRES__", "__QOS__"):
            cur = s.strip("_")
            continue
        if s:
            sections[cur].append(s)
    nodes = {}
    for line in sections["SINFO"]:
        f = line.split("|")
        if len(f) < 4:
            continue
        name, part, gres, state = f[:4]
        g = _gpu_count(gres) or 0
        t = re.search(r"gpu:([A-Za-z][A-Za-z0-9]*):\d", gres or "")
        prev = nodes.get(name)
        nodes[name] = {"node": name, "partition": part.rstrip("*") if not prev else prev["partition"],
                       "gpus": max(g, prev["gpus"] if prev else 0), "gpu_type": t.group(1) if t else "?",
                       "state": state, "available": not UNAVAILABLE.search(state)}
    tres = {}
    for line in sections["TRES"]:
        f = line.split(None, 1)
        if len(f) == 2:
            tres[f[0]] = _gpu_count(f[1].replace("gres/gpu=", "gpu=")) or 0
    out_jobs = []
    for line in sections["SQ"]:
        f = line.split("|")
        if len(f) < 9:
            continue
        jid, user, part, qos, per_node, nodelist, nnodes, elapsed, name = f[:8] + ["|".join(f[8:])]
        names = expand_nodelist(nodelist)
        g = _gpu_count(per_node)
        if g is None:  # --gpus per job, not per node: spread the job total over its nodes
            total = tres.get(jid) or 0
            g = total // max(1, len(names) or int(nnodes or 1)) if total else 0
        if not g:
            continue
        out_jobs.append({"job_id": jid, "user": user, "partition": part, "qos": qos, "gpus_per_node": g,
                         "nodes": names, "elapsed_s": jobs.parse_duration(elapsed) or 0, "name": name})
    preempts = {}
    for line in sections["QOS"]:
        f = line.split("|")
        if len(f) >= 2 and f[0]:
            preempts[f[0]] = {q for q in f[1].split(",") if q}
    return {"nodes": nodes, "jobs": out_jobs, "preempts": preempts or {k: set(v) for k, v in DEFAULT_PREEMPTS.items()}}


def can_preempt(preempts: dict, qos: str) -> set:
    if qos in preempts:
        return preempts[qos]
    if qos.startswith("preemptive_high"):
        return DEFAULT_PREEMPTS["preemptive_high"]
    return DEFAULT_PREEMPTS.get(qos, set())


def node_view(snap: dict, me: str = ME) -> list:
    """Per-node rows: total, ours {qos: gpus}, others [(user, qos, gpus)], used, free."""
    rows = {n: {**v, "ours": defaultdict(int), "others": defaultdict(int), "used": 0} for n, v in snap["nodes"].items()}
    for j in snap["jobs"]:
        for n in j["nodes"]:
            r = rows.setdefault(n, {"node": n, "partition": j["partition"], "gpus": 0, "gpu_type": "?", "state": "?",
                                    "available": True, "ours": defaultdict(int), "others": defaultdict(int), "used": 0})
            r["used"] += j["gpus_per_node"]
            if j["user"] == me:
                r["ours"][j["qos"]] += j["gpus_per_node"]
            else:
                r["others"][(j["user"], j["qos"])] += j["gpus_per_node"]
    for r in rows.values():
        r["free"] = max(0, r["gpus"] - r["used"]) if r["available"] else 0
    return sorted(rows.values(), key=lambda r: (r["partition"], r["node"]))


def our_cap_use(snap: dict, cluster: str, runs_by_job: dict | None = None, me: str = ME,
                partition: str = "jsteinhardt") -> dict:
    """{qos: {"gpus": n, "cap": c, "by_experiment": {exp: n}}} for our capped QOS."""
    out = {}
    for qos, cap in CAPS[cluster].items():
        by_exp = defaultdict(int)
        for j in snap["jobs"]:
            if j["user"] != me or j["qos"] != qos or (cluster == "berkeley" and j["partition"] != partition):
                continue
            exp = ((runs_by_job or {}).get(j["job_id"]) or {}).get("experiment") or "unregistered"
            by_exp[exp] += j["gpus_per_node"] * max(1, len(j["nodes"]))
        out[qos] = {"gpus": sum(by_exp.values()), "cap": cap, "by_experiment": dict(by_exp)}
    return out


def would_preempt(snap: dict, gpus: int, qos: str, partition: str | None = None, node: str | None = None,
                  me: str = ME, any_gpu: bool = False) -> dict:
    """Where a new single-node job of ``gpus`` at ``qos`` would land and whom it would preempt.

    Prefers a node with enough free GPUs (no victims). Otherwise picks the node whose preemptable
    jobs (lower QOS, per the QOS table) cover the shortfall with the fewest GPU-hours of others' work
    lost, taking the youngest jobs first. Slurm's own victim choice may differ; this is a warning.
    """
    victims_ok = can_preempt(snap["preempts"], qos)
    best = None
    for r in node_view(snap, me):
        if not r["available"] or r["gpus"] < gpus or (partition and r["partition"] != partition) or (node and r["node"] != node):
            continue
        if not any_gpu and not node and r.get("gpu_type") not in GOOD_GPU_TYPES:
            continue
        if r["free"] >= gpus:
            return {"fits_free": True, "node": r["node"], "free": r["free"], "victims": []}
        need = gpus - r["free"]
        cand = sorted((j for j in snap["jobs"] if r["node"] in j["nodes"] and j["qos"] in victims_ok),
                      key=lambda j: j["elapsed_s"])
        picked, got = [], 0
        for j in cand:
            if got >= need:
                break
            picked.append(j)
            got += j["gpus_per_node"]
        if got < need:
            continue
        lost = sum(j["gpus_per_node"] * j["elapsed_s"] / 3600 for j in picked if j["user"] != me)
        if best is None or lost < best["others_gpu_h_lost"]:
            best = {"fits_free": False, "node": r["node"], "free": r["free"], "others_gpu_h_lost": round(lost, 1),
                    "victims": [{k: j[k] for k in ("job_id", "user", "qos", "gpus_per_node", "elapsed_s", "name")}
                                for j in picked]}
    return best or {"fits_free": False, "node": None, "victims": [], "pending": True}


# ------------------------------------------------------------------------------------------- CLI


def _runs_by_job():
    try:
        out = {}
        for r in jobs.load_registry().values():
            if r.get("job_id"):
                out[str(r["job_id"])] = r
        return out
    except Exception:  # noqa: BLE001
        return {}


def _abbr(qos):
    return {"preemptive_high": "ph", "preemptive": "p", "normal": "n"}.get(qos, qos.replace("preemptive_high_", "ph_"))


def render(cluster: str, snap: dict, caps: dict) -> str:
    L = []
    def _one(q, v):
        lim = f"/{v['cap']}" if v["cap"] is not None else " (no cap: runs when GPUs are free or held at normal)"
        exp = (" [" + ", ".join(f"{e} {n}" for e, n in sorted(v["by_experiment"].items())) + "]") if v["by_experiment"] else ""
        return f"{q} {v['gpus']}{lim}{exp}"

    L.append(f"{cluster.upper()}  our use: " + ", ".join(_one(q, v) for q, v in caps.items()))
    L.append(f"  {'node':<20} {'part':<12} {'type':<6} {'gpus':>4} {'free':>4} {'ours':<14} others (user gpus qos)")
    for r in node_view(snap):
        if not r["gpus"] and not r["used"]:
            continue
        ours = " ".join(f"{n}{_abbr(q)}" for q, n in sorted(r["ours"].items())) or "-"
        others = ", ".join(f"{u} {n}{_abbr(q)}" for (u, q), n in sorted(r["others"].items(), key=lambda x: -x[1])) or "-"
        st = "" if r["available"] else f"  [{r['state']}]"
        L.append(f"  {r['node']:<20} {r['partition']:<12} {r.get('gpu_type', '?'):<6} {r['gpus']:>4} {r['free']:>4} {ours:<14} {others}{st}")
    return "\n".join(L)


def cmd_capacity(args) -> int:
    clusters = ["berkeley", "lambda"] if args.cluster == "all" else [args.cluster]
    runs_by_job = _runs_by_job()
    result, rc = {}, 0
    for cl in clusters:
        try:
            snap = parse(fetch(cl))
        except Exception as e:  # noqa: BLE001
            print(f"{cl.upper()}: unavailable ({e})")
            rc = 1
            continue
        caps = our_cap_use(snap, cl, runs_by_job)
        result[cl] = {"nodes": node_view(snap), "caps": caps}
        if args.would_preempt:
            wp = would_preempt(snap, args.gpus, args.qos, partition=args.partition, node=args.node, any_gpu=args.any_gpu)
            result[cl]["would_preempt"] = wp
        if not args.json:
            print(render(cl, snap, caps))
            if args.would_preempt:
                print(render_preempt(cl, args, caps, result[cl]["would_preempt"]))
            print()
    if args.json:
        print(json.dumps(result, default=lambda o: dict(o) if isinstance(o, defaultdict) else str(o), indent=1))
    if args.would_preempt and any(not v.get("would_preempt", {}).get("fits_free", True)
                                  and any(x["user"] != ME for x in v["would_preempt"].get("victims", []))
                                  for v in result.values()):
        return 3  # a start would preempt another user's job
    return rc


def render_preempt(cluster, args, caps, wp) -> str:
    L = [f"  NEW JOB {args.gpus} GPU @ {args.qos}" + (f" on {args.partition}" if args.partition else "") + ":"]
    cap = caps.get(args.qos)
    if cap and cap["cap"] is not None and cap["gpus"] + args.gpus > cap["cap"]:
        L.append(f"    OVER CAP: we already hold {cap['gpus']}/{cap['cap']} at {args.qos}; it will pend (or use a lower QOS)")
    if wp.get("fits_free"):
        L.append(f"    fits in free GPUs on {wp['node']} ({wp['free']} free): preempts nobody")
    elif wp.get("pending"):
        L.append("    no node can host it even after preempting lower-QOS jobs: it will pend")
    else:
        others = [v for v in wp["victims"] if v["user"] != ME]
        L.append(f"    {'WOULD PREEMPT' if others else 'would preempt only our own jobs'} on {wp['node']}"
                 + (f" (~{wp['others_gpu_h_lost']} GPU-h of others' work, if they do not checkpoint):" if others else ":"))
        for v in wp["victims"]:
            L.append(f"      {v['user']:<16} job {v['job_id']} {v['gpus_per_node']} GPU @ {v['qos']}, "
                     f"running {jobs.fmt_dur(v['elapsed_s'])}  {v['name'][:40]}")
    return "\n".join(L)


def add_parser(sub):
    p = sub.add_parser("capacity", help="everyone's GPUs per node (jsteinhardt/berkeleynlp, Lambda), our cap use, "
                                        "and whether a new job would preempt someone")
    p.add_argument("--cluster", choices=["all", "berkeley", "lambda"], default="all")
    p.add_argument("--would-preempt", action="store_true",
                   help="check a NEW single-node job (--gpus, --qos): exit 3 if it would preempt another user")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--qos", default="preemptive_high")
    p.add_argument("--partition", help="Berkeley partition to place it in (e.g. jsteinhardt)")
    p.add_argument("--node", help="only this node (e.g. a --nodelist you plan to pass)")
    p.add_argument("--any-gpu", action="store_true", help="also place on GPUs below A100 (RTX etc.)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_capacity)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    add_parser(ap.add_subparsers(dest="cmd"))
    a = ap.parse_args()
    sys.exit(a.func(a))
