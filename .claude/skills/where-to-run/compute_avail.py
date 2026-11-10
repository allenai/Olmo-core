#!/usr/bin/env python3
"""
Where can I run this right now? — one-shot GPU availability across Beaker and local slurm.

Answers, for a request of G GPUs:
  * local slurm (jsteinhardt, berkeleynlp) at every QOS I hold: free now / free via preemption /
    blocked by a QOS cap / earliest start from the time-left of the jobs in the way, plus the
    risk that something higher-QOS preempts *me*.
  * Beaker (urgent + unallocated, our standing policy): free slots, slots held by lower-priority
    unallocated jobs (preemptable once their min-runtime elapses), urgent jobs queued ahead, and
    the *measured* queue wait of urgent jobs that recently scheduled on that cluster.

Stdlib only, so it runs under any python3 (no NFS conda env import tax) in ~1-3 s.

    python3 ~/.claude/skills/where-to-run/compute_avail.py                # 1,2,4,8 GPUs, colored terminal view
    python3 ~/.claude/skills/where-to-run/compute_avail.py -g 8 -g 16     # specific request sizes (16 = 2 full nodes)
    python3 ~/.claude/skills/where-to-run/compute_avail.py --hours 12     # rank by start + 12h-on-H200 runtime
    python3 ~/.claude/skills/where-to-run/compute_avail.py --json         # machine-readable, for agents
    python3 ~/.claude/skills/where-to-run/compute_avail.py --html         # also write an HTML view
    python3 ~/.claude/skills/where-to-run/compute_avail.py --no-beaker    # local only (or --no-local)

All ETAs are *upper bounds* from scheduled end times (slurm jobs usually finish early) or
empirical medians (Beaker) — read them as "order of magnitude", not promises.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ME = os.environ.get("USER", "prasann")
BEAKER_ME = "prasanns"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
PARTITIONS = ["jsteinhardt", "berkeleynlp"]
# berkeleynlp is for small-scale debugging only, at --qos=normal only (Prasann, 2026-10-16; memory:
# berkeleynlp-debug-only). It is never recommended unless --debug. Enforced by ops/hooks/bash_guard.py.
RESTRICTED_QOS = {"berkeleynlp": {"normal"}}
DEBUG_ONLY_PARTITIONS = {"berkeleynlp"}
DEBUG_ONLY_NOTE = "berkeleynlp: small-scale debugging only, --qos=normal only (preemptible by everyone)"
# Users whose jobs we never preempt without asking Prasann first (memory: do-not-preempt-rhys).
ASK_FIRST = {"rhys_gould"}
# Beaker GPU clusters worth considering (name -> note). Order = display order.
BEAKER_CLUSTERS = ["jupiter", "ceres", "titan", "saturn", "prometheus"]
# Standing directive: Beaker GPU work goes to jupiter. Others are shown for info and only
# recommended with --beaker-any (titan/B200 is untested with our image + flash-attn).
RECOMMEND_BEAKER = {"ai2/jupiter"}
OUR_WORKSPACE = "ai2/flex2"
OUR_BUDGET_ID = "01KP6SET80STGTCC9EFRPAE2R6"  # ai2/oe-other
# Rough step-time multiplier vs one H200 (bf16 LM training; memory-bound work favors H200 more).
SPEED = {"H200": 1.0, "B200": 0.6, "H100": 1.2, "A100": 2.5}
# Only A100-or-better GPUs are considered anywhere (local nodes and Beaker clusters).
GOOD_GPUS = set(SPEED)
# Local QOS below this are never considered (normal is preemptible by everything).
SKIP_QOS = {"normal"}
# Preempting `normal` jobs on the local clusters is always fine / expected — treat as free.
FREE_TO_PREEMPT_QOS = {"normal"}
# A100s are a last resort: never recommended over a faster GPU, and flagged "check with Prasann
# first" whenever they are the pick.
ASK_GPUS = {"A100"}
ASK_GPU_NOTE = "A100: check with Prasann before using"
# Spill rule (Prasann, 2026-11-10): small, short jobs go wherever they start soonest. A job of
# <= SPILL_MAX_GPUS GPUs and <= SPILL_MAX_HOURS goes to jsteinhardt (preemptive_high, under our 8-GPU
# cap) whenever Beaker's expected wait is over SPILL_BEAKER_WAIT_S. In the 3-week gpu_audit,
# jsteinhardt sat at 23% of the cap while Beaker jobs spent 60% of their lifetime pending.
SPILL_MAX_GPUS, SPILL_MAX_HOURS, SPILL_BEAKER_WAIT_S = 2, 2.0, 1800
JST_CAP_QOS, JST_CAP = "preemptive_high", 8
PRIO = {"JOB_PRIORITY_LOW": 0, "JOB_PRIORITY_NORMAL": 1, "JOB_PRIORITY_HIGH": 2,
        "JOB_PRIORITY_URGENT": 3, "JOB_PRIORITY_IMMEDIATE": 4}
URGENT = PRIO["JOB_PRIORITY_URGENT"]
NOW = time.time()


# ----------------------------------------------------------------------------- helpers

def experiments_by_job() -> dict:
    """{job_id: experiment} from the ops run registry (repo ops/ next to this skill); {} if unavailable."""
    try:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        sys.path.insert(0, os.path.join(repo, "ops"))
        from opslib import registry  # noqa: WPS433

        return {str(r["job_id"]): r.get("experiment") or "?" for r in registry.load_runs().values() if r.get("job_id")}
    except Exception:  # noqa: BLE001 - never break the availability view
        return {}


def our_cap_use(res: dict) -> dict:
    """Our GPUs under the per-user caps, split by experiment: {"jst": {...}, "lambda": {qos: {...}}}."""
    exps = experiments_by_job()
    out: dict = {}
    sl = res.get("slurm")
    if isinstance(sl, dict) and "running" in sl:
        by: dict = {}
        for j in sl["running"]:
            if j.get("user") == ME and j.get("qos") == JST_CAP_QOS and j.get("partition") == "jsteinhardt":
                e = exps.get(str(j.get("id")), "unregistered")
                by[e] = by.get(e, 0) + (j.get("gpus") or 0)
        out["jst"] = dict(gpus=sum(by.values()), cap=JST_CAP, by_experiment=by)
    lm = res.get("lambda")
    if isinstance(lm, dict) and "mine_jobs" in lm:
        out["lambda"] = {}
        for q, cap in LAMBDA_CAPS.items():
            by = {}
            for jid, qos, g in lm["mine_jobs"]:
                if qos == q:
                    e = exps.get(str(jid), "unregistered")
                    by[e] = by.get(e, 0) + g
            out["lambda"][q] = dict(gpus=sum(by.values()), cap=cap, by_experiment=by)
    return out


def fmt_cap(v: dict) -> str:
    parts = ", ".join(f"{e} {n}" for e, n in sorted(v["by_experiment"].items(), key=lambda x: -x[1]))
    return f"{v['gpus']}/{v['cap']}" + (f" [{parts}]" if parts else "")


def run(cmd: list[str], timeout: float = 25) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout


def ts(s: str | None) -> float | None:
    """Parse a Beaker ISO timestamp (nanosecond fractions, trailing Z) to epoch seconds."""
    if not s:
        return None
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?", s)
    if not m:
        return None
    t = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc).timestamp()
    return t + (float("0." + m.group(2)) if m.group(2) else 0.0)


def dur(sec: float | None) -> str:
    if sec is None:
        return "?"
    sec = max(0, sec)
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.0f}m"
    if sec < 2 * 86400:
        return f"{sec / 3600:.1f}h"
    return f"{sec / 86400:.1f}d"


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


# ----------------------------------------------------------------------------- slurm

def expand_nodes(nodelist: str) -> list[str]:
    if not nodelist:
        return []
    if "[" not in nodelist and "," not in nodelist:
        return [nodelist]
    try:
        return run(["scontrol", "show", "hostnames", nodelist]).split()
    except Exception:
        return [nodelist]


def gpus_from_tres(s: str) -> int:
    m = re.search(r"gres/gpu(?::[^=,]+)?[=:](\d+)", s or "")
    return int(m.group(1)) if m else 0


def slurm_snapshot() -> dict:
    sinfo = run(["sinfo", "-h", "-N", "-p", ",".join(PARTITIONS),
                 "-O", "NodeList:30,Partition:20,StateLong:20,Gres:40,GresUsed:60"])
    nodes: dict[str, dict] = {}
    for line in sinfo.splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        name, part, state, gres, used = f[:5]
        m = re.search(r"gpu:([^:]+):(\d+)", gres)
        mu = re.search(r"gpu:[^:]+:(\d+)", used)
        if not m or m.group(1).upper() not in GOOD_GPUS:
            continue
        usable = not re.search(r"drain|down|maint|fail|reserved|unknown|not_responding", state)
        nodes[name] = dict(name=name, partition=part, state=state.rstrip("*$~#"),
                           gpu_type=m.group(1), total=int(m.group(2)),
                           used=int(mu.group(1)) if mu else 0, usable=usable, jobs=[])

    qos = {}
    for line in run(["sacctmgr", "-n", "-P", "show", "qos",
                     "format=Name,Preempt,MaxTRESPU,GrpTRES"]).splitlines():
        name, preempt, peru, grp = (line.split("|") + ["", "", ""])[:4]
        qos[name] = dict(preempts=[p for p in preempt.split(",") if p],
                         per_user_cap=gpus_from_tres(peru) or None,
                         group_cap=gpus_from_tres(grp) or None)

    assoc: dict[str, list[str]] = {}
    for line in run(["sacctmgr", "-n", "-P", "show", "assoc", f"user={ME}",
                     "format=Partition,QOS"]).splitlines():
        part, qs = (line.split("|") + [""])[:2]
        if part in PARTITIONS:
            assoc[part] = [q for q in qs.split(",") if q and (
                q in RESTRICTED_QOS[part] if part in RESTRICTED_QOS else q not in SKIP_QOS)]

    sq = json.loads(run(["squeue", "-p", ",".join(PARTITIONS), "--json"]))["jobs"]
    running, pending = [], []
    for j in sq:
        state = (j.get("job_state") or ["?"])[0]
        g = gpus_from_tres(j.get("tres_alloc_str") or "") or gpus_from_tres(
            (j.get("tres_per_job") or "") + "," + (j.get("tres_per_node") or ""))
        end = (j.get("end_time") or {}).get("number") or None
        rec = dict(id=j.get("job_id"), user=j.get("user_name"), qos=j.get("qos"),
                   partition=j.get("partition"), gpus=g, end=end,
                   reason=j.get("state_reason"))
        if state == "RUNNING":
            ns = expand_nodes(j.get("nodes") or "")
            for n in ns:
                if n in nodes and g:
                    nodes[n]["jobs"].append(dict(rec, gpus=math.ceil(g / len(ns))))
            running.append(rec)
        elif state == "PENDING":
            pending.append(rec)
    return dict(nodes=nodes, qos=qos, assoc=assoc, running=running, pending=pending)


def preemptors_of(q: str, qos: dict) -> list[str]:
    return [name for name, d in qos.items() if q in d["preempts"]]


def slurm_option(snap: dict, part: str, q: str, G: int, gpu_type: str | None) -> dict:
    """Earliest start for G GPUs on one node of `part` at QOS `q` (multi-node ignored locally)."""
    qos, nodes = snap["qos"], snap["nodes"]
    can_preempt = set(qos.get(q, {}).get("preempts", []))
    cand = [n for n in nodes.values() if n["partition"] == part and n["usable"]
            and (gpu_type is None or n["gpu_type"].upper() == gpu_type.upper())
            and n["total"] >= G]
    out = dict(partition=part, qos=q, gpus=G, gpu_type=None, status="none", eta_s=None,
               node=None, how="", notes=[])
    if not cand:
        out.update(status="impossible", how=f"no usable node with >= {G} GPUs")
        return out

    # QOS caps: per-user (MaxTRESPU) and per-group (GrpTRES).
    qd = qos.get(q, {})
    mine = sum(r["gpus"] for r in snap["running"] if r["qos"] == q and r["user"] == ME)
    group = sum(r["gpus"] for r in snap["running"] if r["qos"] == q)
    cap_wait = 0.0
    for cap, used, label, owners in (
        (qd.get("per_user_cap"), mine, "per-user", lambda r: r["user"] == ME),
        (qd.get("group_cap"), group, "group", lambda r: True),
    ):
        if cap and used + G > cap:
            if G > cap:
                out.update(status="impossible", how=f"{G} GPUs > {label} cap {cap}")
                return out
            ends = sorted((r["end"] or NOW + 86400 * 28, r["gpus"]) for r in snap["running"]
                          if r["qos"] == q and owners(r))
            need, t = used + G - cap, NOW
            for e, g in ends:
                need -= g
                t = e
                if need <= 0:
                    break
            cap_wait = max(cap_wait, t - NOW)
            out["notes"].append(f"{label} cap {cap}: {used} in use")

    best = a100_best = None
    for n in cand:
        free = n["total"] - n["used"]
        pre = [j for j in n["jobs"] if j["qos"] in can_preempt and j["user"] not in ASK_FIRST]
        ask = [j for j in n["jobs"] if j["qos"] in can_preempt and j["user"] in ASK_FIRST]
        pre_g = sum(j["gpus"] for j in pre)
        normal = [j for j in pre if j["qos"] in FREE_TO_PREEMPT_QOS]
        if free >= G:
            cand_r = (0, 0.0, n["name"], f"{free} free on {n['name']}")
        elif free + sum(j["gpus"] for j in normal) >= G:
            users = sorted({j["user"] for j in normal})
            cand_r = (0, 30.0, n["name"],
                      f"preempt normal-QOS {','.join(users)} on {n['name']} (always fine)")
        elif free + pre_g >= G:
            users = sorted({j["user"] for j in pre})
            cand_r = (1, 120.0, n["name"],
                      f"preempt {G - free} GPU(s) of {','.join(users)} on {n['name']}")
        else:
            # Wait for non-preemptable jobs to end (scheduled end time = upper bound).
            have, t = free + pre_g, None
            for j in sorted((j for j in n["jobs"] if j not in pre),
                            key=lambda j: j["end"] or 1e18):
                have += j["gpus"]
                if have >= G:
                    t = (j["end"] or NOW + 86400 * 28) - NOW
                    break
            if t is None:
                continue
            cand_r = (2, t, n["name"], f"wait for jobs on {n['name']} to end (≤{dur(t)})")
        if ask and cand_r[0] >= 1:
            out["notes"].append(f"{n['name']}: {sum(j['gpus'] for j in ask)} GPU(s) of "
                                f"{','.join(sorted({j['user'] for j in ask}))} — ask first")
        cand_r += (n["gpu_type"],)
        # A100 nodes only when no faster node can fit at all; otherwise faster GPU breaks ties.
        def key(r):
            return (r[4].upper() in ASK_GPUS, r[0], r[1], SPEED[r[4]])
        if best is None or key(cand_r) < key(best):
            best = cand_r
        if cand_r[4].upper() in ASK_GPUS:
            a100_best = cand_r if a100_best is None or cand_r[:2] < a100_best[:2] else a100_best
    if best is None:
        out.update(status="impossible", how="no node can fit it")
        return out
    tier, eta, node, how, out["gpu_type"] = best
    if out["gpu_type"].upper() in ASK_GPUS:
        how = f"⚠ {ASK_GPU_NOTE} — {how}"
    elif a100_best is not None and a100_best[:2] < (tier, eta):
        out["notes"].append(f"{a100_best[2]} (A100) could start sooner — {ASK_GPU_NOTE}")
    if cap_wait > eta:
        tier, eta, how = 3, cap_wait, f"QOS cap — wait ≤{dur(cap_wait)} for your/group jobs"
    out.update(status=["free", "preempt", "wait", "capped"][tier], eta_s=eta, node=node, how=how)

    # Risk to ME: pending/running jobs at a QOS that preempts q.
    hi = set(preemptors_of(q, qos))
    pend_hi = sum(p["gpus"] for p in snap["pending"] if p["partition"] == part and p["qos"] in hi
                  and "Held" not in (p["reason"] or "") and "Dependency" not in (p["reason"] or ""))
    out["preempt_risk"] = "none" if not hi else (
        f"high ({pend_hi} GPU(s) of higher-QOS work pending)" if pend_hi else
        f"possible (preemptable by {', '.join(sorted(hi))[:40]})")
    return out


# ----------------------------------------------------------------------------- beaker

def bk_json(*args: str) -> object:
    return json.loads(run([BEAKER, *args, "--format", "json"]))


def bk_list_jobs(opts: dict, max_pages: int = 6) -> list[dict]:
    jobs, token = [], None
    for _ in range(max_pages):
        req = {"options": opts}
        if token:
            req["nextPageToken"] = token
        d = json.loads(run([BEAKER, "rpc", "call", "ListJobs", json.dumps(req), "--format", "json"]))
        jobs += d.get("jobs", [])
        token = d.get("nextPageToken")
        if not token:
            break
    return jobs


def beaker_cluster(c: dict) -> dict:
    name, cid, org = c["name"], c["id"], c["organizationId"]
    shape = c.get("nodeShape", {})
    gtype = shape.get("gpuType", "").replace("GPU_TYPE_NVIDIA_", "").replace("_80GB", "")
    gtype = {"A100": "A100", "H100": "H100", "B200": "B200", "L40S": "L40S"}.get(
        gtype.split("_")[0], gtype.split("_")[0] or "?")
    occ = c.get("clusterOccupancy", {}).get("slotCounts", {})
    policy = c.get("schedulingPolicy", "")
    base = dict(cluster=f"ai2/{name}", gpu_type=gtype, gpus_per_node=shape.get("gpuCount", 0),
                policy="eager" if "EAGER" in policy else "strict+backfill",
                total=int(occ.get("total", 0)), free=int(occ.get("available", 0)),
                cordoned=int(occ.get("cordoned", 0)), timeout=c.get("maxSessionTimeout"),
                restricted=bool(c.get("budgetRestrictions"))
                and OUR_BUDGET_ID not in c.get("budgetRestrictions", []))

    with cf.ThreadPoolExecutor(4) as ex:
        f_usage = ex.submit(bk_json, "cluster", "usage", f"ai2/{name}")
        f_free = ex.submit(bk_json, "cluster", "free-slots", f"ai2/{name}")
        common = {"organizationId": org, "pageSize": 1000}
        f_pend = ex.submit(bk_list_jobs, dict(common, eligibleForClusterId=cid,
                                              scheduled=False, finalized=False))
        f_run = ex.submit(bk_list_jobs, dict(common, scheduledOnClusterId=cid, finalized=False))
        f_recent = ex.submit(bk_list_jobs, dict(
            common, scheduledOnClusterId=cid, pageSize=500,
            sortClause={"sortOrder": "SORT_ORDER_DESCENDING", "created": {}}), 1)
        usage, free_nodes = f_usage.result(), f_free.result()
        pend, running, recent = f_pend.result(), f_run.result(), f_recent.result()

    # A workspace in an allocated group can run BOTH allocated and unallocated jobs (e.g.
    # robots-rl), and jobs carry no per-job flag. So: any allocated slot => treat the whole
    # workspace as allocated (its jobs aren't preemptable by us, its waits aren't ours).
    alloc_ws: dict[str, bool] = {}
    for u in usage:
        a = u.get("assigned")
        if a:
            ws = a.get("workspaceReference")
            alloc_ws[ws] = alloc_ws.get(ws, False) or bool(a.get("allocated"))
    # Per-node capacity for an urgent+unallocated job: free slots + slots of lower-priority
    # unallocated jobs (each preemptable only once its minRuntime has elapsed).
    node_free = {n["nodeId"]: int(n["availableSlots"]) for n in free_nodes}
    node_pre: dict[str, list[tuple[float, int]]] = {}
    mine_running = 0
    for j in running:
        ad, sd = j.get("assignmentDetails", {}), j.get("systemDetails", {})
        node = ad.get("nodeId")
        slots = int(ad.get("occupiedSlotsCount") or ad.get("resourceAssignment", {})
                    .get("assignedSlotsCount") or 0)
        if j.get("authorReference") == BEAKER_ME:
            mine_running += slots
        if not node or not slots:
            continue
        node_free.setdefault(node, 0)
        prio = PRIO.get(sd.get("priority"), 1)
        allocated = alloc_ws.get(j.get("workspaceReference"), False)
        if prio < URGENT and not allocated:
            started = ts(j.get("status", {}).get("started")) or ts(j["status"].get("scheduled"))
            minrt = float((sd.get("minRuntime") or "0s").rstrip("s") or 0)
            ok_at = max(0.0, (started or NOW) + minrt - NOW)
            node_pre.setdefault(node, []).append((ok_at, slots))
    base["preemptable_now"] = sum(s for v in node_pre.values() for t, s in v if t == 0)
    base["preemptable_later"] = sum(s for v in node_pre.values() for t, s in v if t > 0)
    base["mine_running"] = mine_running
    base["nodes_fully_free"] = sum(1 for v in node_free.values() if v >= base["gpus_per_node"])

    def capacity_eta(G: int) -> tuple[float | None, int]:
        """Earliest t at which nodes can host G GPUs (G>per-node => that many whole nodes)."""
        per = base["gpus_per_node"] or 8
        need_nodes, need_each = (math.ceil(G / per), per) if G > per else (1, G)
        times = []
        for node, free in node_free.items():
            if free >= need_each:
                times.append(0.0)
                continue
            have = free
            for t, s in sorted(node_pre.get(node, [])):
                have += s
                if have >= need_each:
                    times.append(t)
                    break
        times.sort()
        n_free = sum(1 for v in node_free.values() if v >= need_each)
        return (times[need_nodes - 1] if len(times) >= need_nodes else None), n_free >= need_nodes

    def gcount(j: dict) -> int:
        return int(j.get("containerSpec", {}).get("resourceRequest", {}).get("gpuCount") or 0)

    waits: dict[str, list[float]] = {"small": [], "node": []}
    for j in recent:
        st = j.get("status", {})
        a, b = ts(st.get("created")), ts(st.get("scheduled"))
        if (a and b and b >= a and not alloc_ws.get(j.get("workspaceReference"), False)
                and PRIO.get(j.get("systemDetails", {}).get("priority")) == URGENT):
            waits["node" if gcount(j) >= (base["gpus_per_node"] or 8) else "small"].append(b - a)
    base["recent_window_h"] = round((NOW - min((ts(j["status"].get("created")) or NOW)
                                               for j in recent)) / 3600, 1) if recent else None

    ahead = [j for j in pend if PRIO.get(j.get("systemDetails", {}).get("priority"), 0) >= URGENT]
    base["queued_total"] = len(pend)
    base["options"] = []
    for G in REQ_SIZES:
        big = G >= (base["gpus_per_node"] or 8)
        same = [j for j in ahead if (gcount(j) >= (base["gpus_per_node"] or 8)) == big]
        oldest = max((NOW - (ts(j["status"].get("created")) or NOW) for j in same), default=None)
        eta, fits_free = capacity_eta(G)
        w = waits["node" if big else "small"]
        base["options"].append(dict(
            gpus=G, capacity_eta_s=eta, fits_in_free_slots=fits_free,
            urgent_ahead=len(same), urgent_ahead_gpus=sum(gcount(j) for j in same),
            oldest_urgent_wait_s=oldest,
            recent_wait_p50_s=pct(w, 0.5), recent_wait_p90_s=pct(w, 0.9), recent_n=len(w),
            verdict=("impossible", None) if base["restricted"] else
            beaker_verdict(eta, fits_free, len(same), pct(w, 0.5), len(w))))
    return base


def beaker_verdict(cap_eta, fits_free, n_ahead, p50, n_samples) -> tuple[str, float | None]:
    """Combine structural capacity with what the queue is actually doing. -> (status, eta_s).

    Structure gates everything: if no node can host the job even after preempting every
    lower-priority unallocated job, it's "nofit" no matter how fast other jobs scheduled.
    Otherwise the ETA is the later of when capacity opens (min-runtime expiries) and the
    measured p50 wait. Pending urgent jobs that have sat for days are usually unsatisfiable,
    so the queue is context only.
    """
    if cap_eta is None:
        return "nofit", None
    measured = p50 if (n_samples >= 5 and p50 is not None) else 0.0
    if fits_free and n_ahead == 0:
        return "free", max(60.0, measured)
    eta = max(cap_eta, measured, 120.0 if not fits_free else 60.0)
    return ("preempt" if eta < 900 else "wait"), eta


def beaker_snapshot() -> list[dict]:
    clusters = [c for c in bk_json("cluster", "list", "ai2")
                if c.get("name") in BEAKER_CLUSTERS and c.get("nodeShape", {}).get("gpuCount")]
    clusters.sort(key=lambda c: BEAKER_CLUSTERS.index(c["name"]))
    # Full analysis (5 API calls) only for clusters we'd recommend; the rest get a one-line
    # summary from the listing we already have. Keeps the default run to ~1-2 s.
    deep = [c for c in clusters if BEAKER_ANY or f"ai2/{c['name']}" in RECOMMEND_BEAKER]
    out = []
    with cf.ThreadPoolExecutor(len(deep) or 1) as ex:
        futs = {c["name"]: ex.submit(beaker_cluster, c) for c in deep}
        for c in clusters:
            if c["name"] not in futs:
                out.append(beaker_summary(c))
                continue
            try:
                out.append(futs[c["name"]].result())
            except Exception as e:  # noqa: BLE001
                out.append(dict(cluster=f"ai2/{c['name']}", error=str(e)[:200]))
    return out


def beaker_summary(c: dict) -> dict:
    shape = c.get("nodeShape", {})
    g = shape.get("gpuType", "").replace("GPU_TYPE_NVIDIA_", "").split("_")[0] or "?"
    occ = c.get("clusterOccupancy", {}).get("slotCounts", {})
    return dict(cluster=f"ai2/{c['name']}", gpu_type=g, summary_only=True,
                policy="eager" if "EAGER" in c.get("schedulingPolicy", "") else "strict+backfill",
                total=int(occ.get("total", 0)), free=int(occ.get("available", 0)),
                cordoned=int(occ.get("cordoned", 0)),
                restricted=bool(c.get("budgetRestrictions"))
                and OUR_BUDGET_ID not in c.get("budgetRestrictions", []))


# ----------------------------------------------------------------------------- rendering

USE_COLOR = sys.stdout.isatty()
COL = dict(free="32", preempt="36", wait="33", capped="35", impossible="90", unknown="90",
           nofit="33",
           none="90", h="1", dim="2", red="31")
ICON = dict(free="●", preempt="◐", wait="○", capped="⊘", impossible="·", unknown="?", none="·",
            nofit="○")


def c(text: str, kind: str) -> str:
    return f"\033[{COL.get(kind, '0')}m{text}\033[0m" if USE_COLOR else text


def unverified(o: dict) -> bool:
    """Structural-only Beaker estimate: competing urgent queue, no recent measured waits."""
    return o["recent_n"] < 5 and o["urgent_ahead"] > 0 and o["verdict"][0] in ("preempt", "wait")


def cell(status: str, eta: float | None, unsure: bool = False) -> str:
    label = {"free": "NOW", "preempt": f"~{dur(eta)}", "wait": f"≤{dur(eta)}",
             "capped": f"cap {dur(eta)}", "impossible": "—", "unknown": "?",
             "nofit": "no fit"}.get(status, "?") + ("?" if unsure else "")
    return c(f"{ICON.get(status, '?')} {label:<8}", status)


def bar(used: int, pre: int, total: int, width: int = 16) -> str:
    if not total:
        return ""
    u = round(width * (used - pre) / total)
    p = round(width * pre / total)
    f = max(0, width - u - p)
    return c("█" * u, "dim") + c("▒" * p, "preempt") + c("░" * f, "free")


# ----------------------------------------------------------------------------- Lambda (ssh lambda)
#: Lambda fair-share policy (OLMo-core lambda_cluster.md): at most 8 GPUs on preemptive_high and 8 on
#: preemptive at once; anything beyond goes at normal. A100s: ask Prasann before using (memory a100-ask-first).
LAMBDA_CAPS = {"preemptive_high": 8, "preemptive": 8}
LAMBDA_ME = os.environ.get("LAMBDA_USER", "prasann")


def lambda_snapshot() -> dict:
    """Per-node GPU allocation + running/pending jobs on the Lambda A100 cluster (one ssh call)."""
    out = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "lambda",
               "scontrol show node -o; echo ===JOBS; scontrol show job -o"], timeout=30)
    nodes_txt, _, jobs_txt = out.partition("===JOBS")

    def kv(line: str) -> dict:
        return dict(t.split("=", 1) for t in line.split() if "=" in t)

    def gpus(tres: str) -> int:
        m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", tres or "")
        return int(m.group(1)) if m else 0

    nodes = {}
    for line in nodes_txt.strip().splitlines():
        d = kv(line)
        tot = re.search(r"gpu(?::[^:,]+)?:(\d+)", d.get("Gres", ""))
        nodes[d.get("NodeName")] = dict(name=d.get("NodeName"), state=d.get("State", "?").lower(),
                                        total=int(tot.group(1)) if tot else 0,
                                        alloc=gpus(d.get("AllocTRES", "")), holders={})
    mine: dict[str, int] = {}
    mine_jobs: list = []
    pending: dict[str, int] = {}
    for line in jobs_txt.strip().splitlines():
        d = kv(line)
        user = d.get("UserId", "?").split("(")[0]
        state, qos = d.get("JobState"), d.get("QOS", "?")
        if state == "PENDING":
            pending[user] = pending.get(user, 0) + 1
            continue
        if state != "RUNNING":
            continue
        g = gpus(d.get("AllocTRES", ""))
        if user == LAMBDA_ME:
            mine[qos] = mine.get(qos, 0) + g
            mine_jobs.append((d.get("JobId"), qos, g))
        for n in expand_nodes(d.get("NodeList", "")):
            if n in nodes and g:
                h = nodes[n]["holders"]
                h[f"{user}({qos})"] = h.get(f"{user}({qos})", 0) + g
    return dict(nodes=list(nodes.values()), mine=mine, mine_jobs=mine_jobs, pending=pending)


def render_text(res: dict) -> str:
    L = []
    t = datetime.fromtimestamp(NOW).strftime("%Y-%m-%d %H:%M")
    L.append(c(f"COMPUTE AVAILABILITY  {t}   (request sizes: {', '.join(map(str, REQ_SIZES))} GPUs)", "h"))
    L.append(c("legend: ● free now  ◐ needs preemption (~minutes)  ○ wait (upper bound)  "
               "○ no fit = no node fits even after preempting  ⊘ QOS cap  — can't fit  "
               "? = Beaker estimate not backed by recent measured waits   bars: █ busy ▒ preemptable-by-you ░ free", "dim"))

    if res.get("recommend"):
        L.append("")
        L.append(c(f"BEST PLACE PER REQUEST SIZE  (ranked by start + {RUN_HOURS:g}h-on-H200 "
                   "runtime scaled by GPU speed)", "h"))
        L.append(c(f"  {DEBUG_ONLY_NOTE}; "
                   + ("included (--debug)" if DEBUG else "left out (pass --debug for a small debug job)"),
                   "dim"))
        for r in res["recommend"]:
            L.append(f"  {r['gpus']:>3} GPU  {cell(r['status'], r['eta_s'])} {r['where']:<46} "
                     f"done ~{dur(r['done_s']):<6} " + c(r["why"], "dim"))
            if r["runners_up"]:
                L.append(c(f"           next: {' | '.join(r['runners_up'])}", "dim"))

    sl = res.get("slurm")
    if isinstance(sl, dict) and "error" in sl:
        L.append(c(f"\nslurm: ERROR {sl['error']}", "red"))
    elif sl:
        L.append("")
        L.append(c("LOCAL SLURM  (single node, A100+, QOS ≥ preemptive; ETA ≤ = scheduled end of blocking jobs)", "h"))
        cu = (res.get("cap_use") or {}).get("jst")
        if cu:
            L.append(c(f"  our jsteinhardt {JST_CAP_QOS} GPUs (all sessions): {fmt_cap(cu)}"
                       f"  · spill rule: <= {SPILL_MAX_GPUS} GPU, <= {SPILL_MAX_HOURS:g}h jobs come here when Beaker waits > 30 min", "dim"))
        hdr = f"  {'partition / qos':<38}" + "".join(f"{str(g) + ' GPU':<11}" for g in REQ_SIZES)
        L.append(hdr + "  preempt-risk to you")
        for part in PARTITIONS:
            for q in sl["assoc"].get(part, []):
                opts = [o for o in sl["options"] if o["partition"] == part and o["qos"] == q]
                risk = opts[0].get("preempt_risk", "") if opts else ""
                L.append(f"  {part + ' / ' + q:<38}"
                         + "".join(cell(o["status"], o["eta_s"]) + " " for o in opts)
                         + " " + c(risk, "red" if risk.startswith("high") else "dim"))
        notes = sorted({n for o in sl["options"] for n in o["notes"]})
        for n in notes:
            L.append(c(f"    note: {n}", "dim"))
        L.append("")
        L.append(c("  nodes", "h"))
        for n in sorted(sl["nodes"].values(), key=lambda n: (n["partition"], n["name"])):
            if not n["usable"]:
                L.append(c(f"  {n['name']:<11}{n['partition']:<12}{n['gpu_type']:<7} "
                           f"{n['state']} — unavailable", "dim"))
                continue
            best_q = max(sl["assoc"].get(n["partition"], [""]),
                         key=lambda q: len(sl["qos"].get(q, {}).get("preempts", [])))
            pre_set = set(sl["qos"].get(best_q, {}).get("preempts", []))
            pre = sum(j["gpus"] for j in n["jobs"] if j["qos"] in pre_set
                      and j["user"] not in ASK_FIRST)
            who = {}
            for j in n["jobs"]:
                who.setdefault(f"{j['user']}({j['qos'].replace('preemptive', 'pre')})", 0)
                who[f"{j['user']}({j['qos'].replace('preemptive', 'pre')})"] += j["gpus"]
            free = n["total"] - n["used"]
            L.append(f"  {n['name']:<11}{n['partition']:<12}{n['gpu_type']:<7}"
                     f"{bar(n['used'], pre, n['total'])} {free}/{n['total']} free, "
                     f"{pre} preemptable  " + c(" ".join(f"{k}:{v}" for k, v in
                                                         sorted(who.items(), key=lambda x: -x[1])), "dim"))

    bk = res.get("beaker")
    if isinstance(bk, dict) and "error" in bk:
        L.append(c(f"\nbeaker: ERROR {bk['error']}", "red"))
    elif bk:
        L.append("")
        L.append(c("BEAKER  (A100+ clusters, urgent + unallocated from ai2/flex2; ETA = time to schedule)", "h"))
        L.append(c("  ETAs assume the job sets minRuntime (default 1h): on jupiter an urgent minRuntime=0 job "
                   "effectively never schedules (memory: beaker-min-runtime)", "dim"))
        L.append(f"  {'cluster':<16}{'gpu':<6}{'x/H200':<7}" + "".join(
            f"{str(g) + ' GPU':<11}" for g in REQ_SIZES) + "  slots  (free/preemptable/total)")
        for b in bk:
            if "error" in b:
                L.append(c(f"  {b['cluster']:<16} error: {b['error']}", "red"))
                continue
            if b.get("summary_only"):
                tot = b["total"] - b["cordoned"]
                L.append(c(f"  {b['cluster']:<16}{b['gpu_type']:<6}{SPEED.get(b['gpu_type'], 0):<7.1f}"
                           f"{b['free']}/{tot} slots free, {b['policy']}"
                           + (", oe-other NOT allowed" if b["restricted"] else "")
                           + "  (info only; --beaker-any for ETAs)", "dim"))
                continue
            info = "" if BEAKER_ANY or b["cluster"] in RECOMMEND_BEAKER else c(" (info only)", "dim")
            L.append(f"  {b['cluster']:<16}{b['gpu_type']:<6}{SPEED.get(b['gpu_type'], 0):<7.1f}"
                     + "".join(cell(*o["verdict"], unverified(o)) + " " for o in b["options"])
                     + f" {bar(b['total'] - b['free'] - b['cordoned'], b['preemptable_now'], b['total'] - b['cordoned'])}"
                     f" {b['free']}/{b['preemptable_now']}/{b['total'] - b['cordoned']}" + info)
            o_small = next((o for o in b["options"] if o["gpus"] < b["gpus_per_node"]), None)
            o_node = next((o for o in b["options"] if o["gpus"] >= b["gpus_per_node"]), None)
            bits = [f"{b['policy']}", f"{b['nodes_fully_free']} whole nodes free",
                    f"{b['queued_total']} queued"]
            for tag, o in (("<node", o_small), ("≥node", o_node)):
                if o:
                    bits.append(f"{tag}: {o['urgent_ahead']} urgent ahead"
                                + (f" (oldest {dur(o['oldest_urgent_wait_s'])})" if o["urgent_ahead"] else "")
                                + f", recent urgent wait p50 {dur(o['recent_wait_p50_s'])}"
                                f"/p90 {dur(o['recent_wait_p90_s'])} (jobs={o['recent_n']})")
            if b["mine_running"]:
                bits.append(f"you hold {b['mine_running']} slots")
            if b["restricted"]:
                bits.append("RESTRICTED: oe-other budget not allowed")
            L.append(c("      " + " · ".join(bits), "dim"))
    lm = res.get("lambda")
    if isinstance(lm, dict) and "error" in lm:
        L.append(c(f"\nlambda: ERROR {lm['error']}", "red"))
    elif lm:
        L.append("")
        L.append(c("LAMBDA  (ssh lambda; A100-80GB x8/node; info only -- A100: check with Prasann before using)", "h"))
        lcu = (res.get("cap_use") or {}).get("lambda") or {}
        caps = " · ".join(f"{q} {fmt_cap(lcu[q])}" if q in lcu else f"{q} {lm['mine'].get(q, 0)}/{cap}"
                          for q, cap in LAMBDA_CAPS.items())
        L.append(c(f"  your GPUs vs fair-share caps: {caps} · normal {lm['mine'].get('normal', 0)} (uncapped, "
                   f"preemptible)   pending jobs by user: {lm['pending'] or 'none'}", "dim"))
        for n in lm["nodes"]:
            if any(x in n["state"] for x in ("down", "drain", "maint")):
                L.append(c(f"  {n['name']:<22} {n['state'].split('+')[0]:<10} unavailable ({n['state']})", "dim"))
                continue
            free = max(0, n["total"] - n["alloc"])
            who = " ".join(f"{k}:{v}" for k, v in n["holders"].items())
            L.append(f"  {n['name']:<22} {n['state']:<10} {bar(n['alloc'], 0, n['total'])} "
                     f"{free}/{n['total']} free  {who}")
    return "\n".join(L)


def recommend(res: dict) -> list[dict]:
    """Best option per request size: minimize start ETA + runtime scaled by GPU speed.

    Ties within 15 min fall back to our routing policy (jsteinhardt > berkeleynlp > Beaker),
    then to GPU speed. Runtime comes from --hours (expected wall-clock on H200s).
    Debug-only partitions (berkeleynlp) are left out unless --debug is passed.
    """
    out = []
    order = {"free": 0, "preempt": 1, "wait": 2, "capped": 3}
    run_s = RUN_HOURS * 3600
    for G in REQ_SIZES:
        cands = []
        sl = res.get("slurm")
        if isinstance(sl, dict) and "options" in sl:
            for o in sl["options"]:
                if o["partition"] in DEBUG_ONLY_PARTITIONS and not DEBUG:
                    continue
                if o["gpus"] == G and o["status"] in order:
                    sp = SPEED.get((o.get("gpu_type") or "H200").upper(), 1.0)
                    risk = 0 if o.get("preempt_risk") == "none" else 1
                    cands.append(((o["eta_s"] or 0) + sp * run_s,
                                  ((o.get("gpu_type") or "").upper() in ASK_GPUS,
                                   risk, order[o["status"]], PARTITIONS.index(o["partition"])),
                                  sp, o["status"], o["eta_s"] or 0,
                                  f"{o['partition']} --qos={o['qos']} -w {o['node']}", o["how"]))
        for b in res.get("beaker") or []:
            if not (BEAKER_ANY or b.get("cluster") in RECOMMEND_BEAKER):
                continue
            for o in b.get("options", []):
                st, eta = o["verdict"]
                if o["gpus"] == G and st in order:
                    sp = SPEED.get(b["gpu_type"], 9)
                    ask = b["gpu_type"] in ASK_GPUS
                    cands.append(((eta or 0) + sp * run_s, (ask, 0, order[st], 5), sp, st,
                                  eta or 0, f"beaker {b['cluster']} ({b['gpu_type']})",
                                  (f"⚠ {ASK_GPU_NOTE} — " if ask else "")
                                  + ("UNVERIFIED (preemptable capacity exists, but no recent "
                                     "unallocated jobs this size to confirm) — "
                                     if unverified(o) else "")
                                  + f"{o['urgent_ahead']} urgent queued; recent wait p50 "
                                  f"{dur(o['recent_wait_p50_s'])}/p90 {dur(o['recent_wait_p90_s'])}"))
        if not cands:
            out.append(dict(gpus=G, status="impossible", eta_s=None, done_s=None, where="nowhere",
                            why="", runners_up=[]))
            continue
        cands.sort(key=lambda x: (x[1][0], math.ceil((x[0] + 1) / 900), x[1][1:], x[2]))
        done, _, _, st, eta, where, why = cands[0]
        seen, runners = {where}, []
        for x in cands[1:]:
            if x[5] not in seen and len(runners) < 2:
                seen.add(x[5])
                runners.append(f"{x[5]} (start {dur(x[4])}, done {dur(x[0])})")
        rec = dict(gpus=G, status=st, eta_s=eta, done_s=done, where=where, why=why, runners_up=runners)
        out.append(spill(rec, res) or rec)
    return out


def spill(rec: dict, res: dict) -> dict | None:
    """The spill rule: a small, short job headed for a Beaker queue over 30 min goes to jsteinhardt
    instead when our preemptive_high cap has room (``None`` = keep the original pick)."""
    G = rec["gpus"]
    if G > SPILL_MAX_GPUS or RUN_HOURS > SPILL_MAX_HOURS or not rec["where"].startswith("beaker"):
        return None
    if (rec.get("eta_s") or 0) <= SPILL_BEAKER_WAIT_S:
        return None
    sl = res.get("slurm")
    if not isinstance(sl, dict) or "options" not in sl:
        return None
    used = (res.get("cap_use") or {}).get("jst", {}).get("gpus", 0)
    if used + G > JST_CAP:
        return None
    opts = [o for o in sl["options"] if o["partition"] == "jsteinhardt" and o["qos"] == JST_CAP_QOS
            and o["gpus"] == G and o["status"] in ("free", "preempt")
            and (o.get("gpu_type") or "").upper() not in ASK_GPUS]
    if not opts:
        return None
    o = min(opts, key=lambda o: o["eta_s"] or 0)
    sp = SPEED.get((o.get("gpu_type") or "H200").upper(), 1.0)
    return dict(gpus=G, status=o["status"], eta_s=o["eta_s"] or 0,
                done_s=(o["eta_s"] or 0) + sp * RUN_HOURS * 3600,
                where=f"jsteinhardt --qos={JST_CAP_QOS} -w {o['node']}",
                why=f"SPILL RULE: Beaker wait ~{dur(rec['eta_s'])} > 30 min for a {G}-GPU <= {SPILL_MAX_HOURS:g}h job; "
                    f"our jsteinhardt {JST_CAP_QOS} use {used}/{JST_CAP}. {o['how']}",
                runners_up=[f"{rec['where']} (start {dur(rec['eta_s'])}, done {dur(rec['done_s'])})"] + rec["runners_up"][:1])


def render_html(res: dict, text: str) -> str:
    ansi = re.compile(r"\033\[(\d+)m(.*?)\033\[0m", re.S)
    css = {"32": "#2e9e4f", "36": "#1f8fb3", "33": "#c48a00", "35": "#a347b8", "90": "#888",
           "2": "#8a8a8a", "1": "font-weight:700", "31": "#d23c3c"}

    def span(m: re.Match) -> str:
        k, body = m.group(1), html.escape(m.group(2))
        style = css.get(k, "")
        style = style if style.startswith("font") else f"color:{style}"
        return f'<span style="{style}">{body}</span>'

    parts, pos = [], 0
    for m in ansi.finditer(text):
        parts.append(html.escape(text[pos:m.start()]))
        parts.append(span(m))
        pos = m.end()
    parts.append(html.escape(text[pos:]))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Compute Availability</title>
<style>
:root{{--bg:#fbfbfa;--fg:#222}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#16171a;--fg:#e4e4e4}}}}
:root[data-theme="dark"]{{--bg:#16171a;--fg:#e4e4e4}}
body{{background:var(--bg);color:var(--fg);margin:0;padding:16px}}
pre{{font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;overflow-x:auto}}
</style></head><body><pre>{''.join(parts)}</pre></body></html>"""


# ----------------------------------------------------------------------------- main

REQ_SIZES = [1, 2, 4, 8]
DEFAULT_HTML = ("visualizations/compute_avail.html" if os.path.isdir("visualizations")
                else os.path.expanduser("~/.cache/where-to-run/compute_avail.html"))
RUN_HOURS = 4.0
BEAKER_ANY = False
DEBUG = False


def main() -> None:
    global REQ_SIZES, USE_COLOR, RUN_HOURS, BEAKER_ANY, DEBUG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-g", "--gpus", type=int, action="append", help="request size(s); repeatable")
    ap.add_argument("--hours", type=float, default=RUN_HOURS,
                    help="expected runtime on H200s; used to rank slow-but-free vs fast-but-queued")
    ap.add_argument("--gpu-type", default="any",
                    help="restrict local nodes to one GPU type, e.g. H200 (default: any A100+)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the text view")
    ap.add_argument("--html", nargs="?", const=DEFAULT_HTML,
                    help="also write an HTML view (default: ./visualizations/compute_avail.html "
                    "if that dir exists, else ~/.cache/where-to-run/compute_avail.html)")
    ap.add_argument("--plain", action="store_true", help="no ANSI colors")
    ap.add_argument("--color", action="store_true", help="force ANSI colors")
    ap.add_argument("--beaker-any", action="store_true",
                    help="recommend any Beaker cluster, not just jupiter")
    ap.add_argument("--debug", action="store_true",
                    help="small-scale debugging job: also recommend berkeleynlp (debug-only partition)")
    ap.add_argument("--no-beaker", action="store_true")
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--no-lambda", action="store_true", help="skip the Lambda section (one ssh call)")
    a = ap.parse_args()
    REQ_SIZES = sorted(set(a.gpus)) if a.gpus else REQ_SIZES
    RUN_HOURS = a.hours
    BEAKER_ANY = a.beaker_any
    DEBUG = a.debug
    USE_COLOR = (USE_COLOR or a.color or bool(a.html)) and not a.plain and not a.json
    gtype = None if a.gpu_type.lower() == "any" else a.gpu_type

    res: dict = dict(time=datetime.fromtimestamp(NOW).isoformat(timespec="seconds"),
                     request_sizes=REQ_SIZES, user=ME)
    with cf.ThreadPoolExecutor(3) as ex:
        f_lm = None if a.no_lambda else ex.submit(lambda_snapshot)
        f_sl = None if a.no_local else ex.submit(slurm_snapshot)
        f_bk = None if a.no_beaker else ex.submit(beaker_snapshot)
        if f_sl:
            try:
                snap = f_sl.result()
                snap["options"] = [slurm_option(snap, p, q, G, gtype)
                                   for p in PARTITIONS for q in snap["assoc"].get(p, [])
                                   for G in REQ_SIZES]
                res["slurm"] = snap
            except Exception as e:  # noqa: BLE001
                res["slurm"] = dict(error=str(e)[:300])
        if f_bk:
            try:
                res["beaker"] = f_bk.result()
            except Exception as e:  # noqa: BLE001
                res["beaker"] = dict(error=str(e)[:300])
        if f_lm:
            try:
                res["lambda"] = f_lm.result()
            except Exception as e:  # noqa: BLE001
                res["lambda"] = dict(error=str(e)[:300])
    res["cap_use"] = our_cap_use(res)
    res["recommend"] = recommend(res)
    res["policy_notes"] = [DEBUG_ONLY_NOTE]
    res["elapsed_s"] = round(time.time() - NOW, 2)

    if a.json:
        sl = res.get("slurm")
        if isinstance(sl, dict) and "nodes" in sl:  # trim bulky raw job lists for agents
            sl = dict(sl)
            sl.pop("running", None)
            sl.pop("pending", None)
            res["slurm"] = sl
        print(json.dumps(res, indent=1, default=str))
        return
    text = render_text(res) + c(f"\n({res['elapsed_s']}s)", "dim")
    if a.html:
        os.makedirs(os.path.dirname(a.html) or ".", exist_ok=True)
        with open(a.html, "w") as fh:
            fh.write(render_html(res, text))
        if not sys.stdout.isatty() and not a.color:
            text = re.sub(r"\033\[\d+m", "", text)
        print(text)
        print(f"\nwrote {a.html}")
    else:
        print(text)


if __name__ == "__main__":
    main()
