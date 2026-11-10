#!/usr/bin/env python3
"""GPU usage + workflow audit, 2026-10-20 .. 2026-11-10 (stdlib only; ~10 s on the login node).

Inputs (pulled by the commands in README.md, kept in raw/, which is gitignored):
  raw/sacct_berkeley_prasann.psv  raw/sacct_lambda_prasann.psv   our slurm jobs (-X -D: one row per attempt)
  raw/sacct_berkeley_all.psv      raw/sacct_lambda_all.psv       everyone's jobs (for shared-capacity context)
  raw/beaker_jobs.jsonl           fetch_beaker.py: one row per Beaker job attempt
  raw/bk_report_*.csv             `beaker report gpu-usage` pulls (Beaker's own GPU-hour accounting)
  /accounts/projects/berkeleynlp/prasann/ops_state/{runs,results}.jsonl   registry + results log
Outputs: data/*.csv, data/summary.json (everything the artifact page draws).
"""
import collections
import csv
import datetime as dt
import json
import re
import statistics
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
RAW, OUT = HERE / "raw", HERE / "data"
STATE = Path("/accounts/projects/berkeleynlp/prasann/ops_state")
UTC, PT = dt.timezone.utc, ZoneInfo("America/Los_Angeles")
W0 = dt.datetime(2026, 10, 20, 7, 0, tzinfo=UTC)  # 2026-10-20 00:00 PT
NOW = dt.datetime(2026, 11, 10, 22, 45, tzinfo=UTC)  # pull time
W1 = NOW
DAYS = [(W0 + dt.timedelta(days=i)).astimezone(PT).date().isoformat() for i in range((W1 - W0).days + 1)]
JST_H200 = {"cubbins", "mcfuzz", "mooney", "sneetches"}
LAMBDA_GPUS = 24  # hyperplane01/02/05 were the only nodes that ran jobs in the window (03 down, 04 drained)
JST_CAP, LAM_PH_CAP, LAM_P_CAP = 8, 8, 8


def pt(s):
    if not s or s in ("Unknown", "None"):
        return None
    return dt.datetime.fromisoformat(s).replace(tzinfo=PT).astimezone(UTC)


def utc(s):
    if not s:
        return None
    return dt.datetime.fromisoformat(re.sub(r"(\.\d{6})\d*", r"\1", s.replace("Z", "+00:00")))


def gpus_of(tres):
    m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", tres or "")
    return int(m.group(1)) if m else 0


def hours(a, b):
    return max(0.0, (b - a).total_seconds() / 3600)


def clip(a, b):
    a, b = max(a, W0), min(b, W1)
    return (a, b) if b > a else None


def q(vals, p):
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))]


# ---------------------------------------------------------------- registry + results
launch_by_job, launch_by_run, finals = {}, {}, collections.defaultdict(list)
all_launches = []
for line in open(STATE / "runs.jsonl"):
    d = json.loads(line)
    if d["event"] == "launch":
        all_launches.append(d)
        if d.get("job_id"):
            launch_by_job[str(d["job_id"])] = d
        launch_by_run[d["run_id"]] = d
    else:
        finals[d["run_id"]].append(d)
results = [json.loads(l) for l in open(STATE / "results.jsonl")]
res_by_job = collections.defaultdict(list)
res_by_run = collections.defaultdict(list)
for r in results:
    if r.get("eval_job_id"):
        res_by_job[str(r["eval_job_id"])].append(r)
    res_by_run[r.get("run_id")].append(r)
    if r.get("train_run"):
        res_by_run[r["train_run"]].append(r)


def kind_of(reg, name):
    k = (reg or {}).get("kind") or "unregistered"
    n = ((reg or {}).get("run_id") or name or "").lower()
    if "smoke" in n:
        return "smoke"
    if k in ("other",) or re.search(r"probe|gradmap|gradsim|gradcomp|grad_|bench|cptor", n):
        return "probe/bench" if k != "train" else "train"
    return k


# ---------------------------------------------------------------- unified attempt table
jobs = []


def add_slurm(path, pool_override=None):
    for r in csv.DictReader(open(path), delimiter="|"):
        g = gpus_of(r["AllocTRES"])
        sub, start, end = pt(r["Submit"]), pt(r["Start"]), pt(r["End"])
        elig = pt(r["Eligible"]) or sub
        state = r["State"].split()[0]
        if state in ("RUNNING",) and not end:
            end = NOW
        if state == "PENDING":
            end = None
        jid = r["JobID"].split(".")[0]
        reg = launch_by_job.get(jid)
        jobs.append({
            "pool": pool_override or r["Partition"], "job_id": jid, "attempt_key": f'{jid}@{r["Start"]}',
            "name": r["JobName"], "gpus": g, "qos": r["QOS"], "min_runtime_s": "",
            "submit": sub, "eligible": max(elig, sub) if elig and sub else sub, "start": start if start and (not end or start <= end) else None,
            "end": end, "state": state, "node": r["NodeList"],
            "experiment": (reg or {}).get("experiment") or "UNREGISTERED", "kind": kind_of(reg, r["JobName"]),
            "run_id": (reg or {}).get("run_id", ""), "expected_min": (reg or {}).get("expected_minutes"),
            "registered": bool(reg), "attempt": 0,
        })


add_slurm(RAW / "sacct_berkeley_prasann.psv")
add_slurm(RAW / "sacct_lambda_prasann.psv", "lambda")

for line in open(RAW / "beaker_jobs.jsonl"):
    d = json.loads(line)
    cf = d.get("canceled_for") or ""
    created, start = utc(d["created"]), utc(d["started"])
    end = utc(d["exited"]) or utc(d.get("canceled")) or utc(d["finalized"])
    if "preempted" in cf:
        state = "PREEMPTED"
    elif "cordoned" in cf:
        state = "NODE_FAIL"
    elif d["exit_code"] == 0:
        state = "COMPLETED"
    elif d["exit_code"] is not None:
        state = "FAILED"
    elif end is None:
        state = "RUNNING" if start else "PENDING"
    elif start is None:
        state = "CANCELLED_PENDING"
    else:
        state = "CANCELLED"
    if state == "RUNNING":
        end = NOW
    reg = launch_by_job.get(d["exp_id"])
    cl = (d.get("cluster") or ["?"])
    cl = cl[0].replace("ai2/", "") if len(cl) == 1 else "multi:" + "+".join(c.replace("ai2/", "") for c in cl)
    jobs.append({
        "pool": "beaker", "job_id": d["exp_id"], "attempt_key": d["job_id"], "name": d["exp_name"], "gpus": d["gpus"],
        "qos": d.get("priority") or "", "min_runtime_s": d.get("min_runtime_s") or 0, "submit": created,
        "eligible": created, "start": start, "end": end, "state": state, "node": cl,
        "experiment": (reg or {}).get("experiment") or "UNREGISTERED", "kind": kind_of(reg, d["exp_name"]),
        "run_id": (reg or {}).get("run_id", ""), "expected_min": (reg or {}).get("expected_minutes"),
        "registered": bool(reg), "attempt": d["attempt"],
    })

# keep attempts that touch the window
def in_window(j):
    a = j["submit"] or j["start"]
    b = j["end"] or NOW
    return a is not None and a < W1 and b > W0


jobs = [j for j in jobs if in_window(j)]
gjobs = [j for j in jobs if j["gpus"] > 0]


def run_h(j):
    if not j["start"]:
        return 0.0
    c = clip(j["start"], j["end"] or NOW)
    return hours(*c) if c else 0.0


for j in gjobs:
    j["gpu_h"] = j["gpus"] * run_h(j)
    j["wait_h"] = hours(j["eligible"], j["start"]) if j["start"] and j["eligible"] else (hours(j["eligible"], j["end"] or NOW) if j["eligible"] else 0)
    j["waited_to_start"] = bool(j["start"])
    j["run_full_h"] = hours(j["start"], j["end"] or NOW) if j["start"] else 0.0

OUT.mkdir(exist_ok=True)
with open(OUT / "jobs.csv", "w", newline="") as f:
    cols = ["pool", "node", "job_id", "attempt", "name", "experiment", "kind", "run_id", "registered", "gpus", "qos",
            "min_runtime_s", "state", "submit", "start", "end", "wait_h", "run_full_h", "gpu_h", "expected_min"]
    w = csv.writer(f)
    w.writerow(cols)
    for j in sorted(gjobs, key=lambda j: j["submit"] or j["start"]):
        w.writerow([j[c].astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ") if isinstance(j[c], dt.datetime) else
                    (round(j[c], 3) if isinstance(j[c], float) else j[c]) for c in cols])

S = {"window": [W0.isoformat(), W1.isoformat()]}


def pool_label(j):
    if j["pool"] == "beaker":
        return "beaker:" + j["node"]
    if j["pool"] == "jsteinhardt":
        return "jsteinhardt:" + j["qos"]
    if j["pool"] == "lambda":
        return "lambda:" + j["qos"]
    return j["pool"]


# ---------------------------------------------------------------- 1. GPU-hours by day / pool / experiment / kind
def daily_split(j, key, acc):
    if not j["start"]:
        return
    c = clip(j["start"], j["end"] or NOW)
    if not c:
        return
    a, b = c
    while a < b:
        d_local = a.astimezone(PT)
        nxt = (d_local.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)).astimezone(UTC)
        e = min(b, nxt)
        acc[(d_local.date().isoformat(), key)] += j["gpus"] * hours(a, e)
        a = e


by_pool, by_exp, by_kind = collections.Counter(), collections.Counter(), collections.Counter()
daily_pool, daily_exp, daily_kind = collections.Counter(), collections.Counter(), collections.Counter()
for j in gjobs:
    pl = pool_label(j)
    by_pool[pl] += j["gpu_h"]
    by_exp[j["experiment"]] += j["gpu_h"]
    by_kind[j["kind"]] += j["gpu_h"]
    daily_split(j, pl.split(":")[0] if not pl.startswith("beaker") else "beaker", daily_pool)
    daily_split(j, j["experiment"], daily_exp)
    daily_split(j, j["kind"], daily_kind)

top_exps = [e for e, _ in by_exp.most_common(9)]


def write_daily(name, acc, keys):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day"] + keys)
        for d in DAYS:
            row = [d]
            for k in keys:
                if k == "other":
                    row.append(round(sum(v for (dd, kk), v in acc.items() if dd == d and kk not in keys), 1))
                else:
                    row.append(round(acc.get((d, k), 0.0), 1))
            w.writerow(row)


write_daily("daily_gpuh_pool.csv", daily_pool, ["beaker", "jsteinhardt", "lambda", "berkeleynlp"])
write_daily("daily_gpuh_experiment.csv", daily_exp, top_exps + ["other"])
write_daily("daily_gpuh_kind.csv", daily_kind, [k for k, _ in by_kind.most_common()])
# Beaker's own accounting (cross-check)
bk_report = collections.Counter()
for r in csv.DictReader(open(RAW / "bk_report_daily_cluster_alloc.csv")):
    if r["day"] >= "2026-10-20":
        bk_report[r["allocated"]] += float(r["gpu_hours"])
S["gpuh_by_pool"] = {k: round(v) for k, v in by_pool.most_common()}
S["gpuh_total"] = round(sum(by_pool.values()))
S["gpuh_by_experiment"] = {k: round(v) for k, v in by_exp.most_common()}
S["gpuh_by_kind"] = {k: round(v) for k, v in by_kind.most_common()}
S["beaker_report_gpuh"] = {"allocated": round(bk_report["true"]), "unallocated": round(bk_report["false"])}
S["n_attempts_gpu"] = len(gjobs)
S["n_attempts_by_pool"] = dict(collections.Counter(j["pool"] for j in gjobs))

# ---------------------------------------------------------------- hourly occupancy + headroom
hours_axis = [W0 + dt.timedelta(hours=i) for i in range(int(hours(W0, W1)) + 1)]
H = len(hours_axis)


def occ(intervals):
    arr = [0.0] * H
    for a, b, g in intervals:
        c = clip(a, b)
        if not c:
            continue
        a, b = c
        i = int(hours(W0, a))
        while i < H and hours_axis[i] < b:
            h0, h1 = hours_axis[i], hours_axis[i] + dt.timedelta(hours=1)
            arr[i] += g * hours(max(a, h0), min(b, h1))
            i += 1
    return arr


series = {}
series["jst_ours_ph"] = occ([(j["start"], j["end"] or NOW, j["gpus"]) for j in gjobs if j["pool"] == "jsteinhardt" and j["qos"] == "preemptive_high" and j["start"]])
series["jst_ours_other_qos"] = occ([(j["start"], j["end"] or NOW, j["gpus"]) for j in gjobs if j["pool"] == "jsteinhardt" and j["qos"] != "preemptive_high" and j["start"]])
series["lam_ours_ph"] = occ([(j["start"], j["end"] or NOW, j["gpus"]) for j in gjobs if j["pool"] == "lambda" and j["qos"] == "preemptive_high" and j["start"]])
series["lam_ours_p"] = occ([(j["start"], j["end"] or NOW, j["gpus"]) for j in gjobs if j["pool"] == "lambda" and j["qos"] != "preemptive_high" and j["start"]])
series["bk_running"] = occ([(j["start"], j["end"] or NOW, j["gpus"]) for j in gjobs if j["pool"] == "beaker" and j["start"]])
series["bk_pending"] = occ([(j["submit"], j["start"] or j["end"] or NOW, j["gpus"]) for j in gjobs if j["pool"] == "beaker" and j["submit"]])
series["slurm_pending_ours"] = occ([(j["eligible"], j["start"] or j["end"] or NOW, j["gpus"]) for j in gjobs if j["pool"] in ("jsteinhardt", "lambda") and j["eligible"]])


def others(path, me):
    ph, any_ = [], []
    lam_users = collections.Counter()
    for r in csv.DictReader(open(path), delimiter="|"):
        if r["User"] == me:
            continue
        g = gpus_of(r["AllocTRES"])
        s, e = pt(r["Start"]), pt(r["End"]) or NOW
        if not g or not s:
            continue
        if "jsteinhardt" in path or r.get("Partition") == "jsteinhardt":
            if r["Partition"] != "jsteinhardt":
                continue
            nodes = r["NodeList"]
            if not any(n in nodes for n in JST_H200 | {"balrog", "saruman"}):
                continue
        any_.append((s, e, g))
        if r["QOS"] == "preemptive_high":
            ph.append((s, e, g))
        c = clip(s, e)
        if c:
            lam_users[r["User"]] += g * hours(*c)
    return occ(ph), occ(any_), lam_users


series["jst_others_ph"], series["jst_others_any"], jst_users = others(str(RAW / "sacct_berkeley_all.psv"), "prasann")
series["lam_others_ph"], series["lam_others_any"], lam_users = others(str(RAW / "sacct_lambda_all.psv"), "prasann")
JST_TOTAL = 32 + 8 + 10
with open(OUT / "hourly_occupancy.csv", "w", newline="") as f:
    w = csv.writer(f)
    keys = list(series)
    w.writerow(["hour_utc"] + keys)
    for i, h in enumerate(hours_axis):
        w.writerow([h.strftime("%Y-%m-%dT%H")] + [round(series[k][i], 2) for k in keys])

# headroom: GPUs we could have started at our top QOS in that hour, given caps and others' top-QOS load
jst_head = [max(0.0, min(JST_CAP - series["jst_ours_ph"][i], JST_TOTAL - series["jst_others_ph"][i] - series["jst_ours_ph"][i])) for i in range(H)]
lam_free = [max(0.0, LAMBDA_GPUS - series["lam_others_any"][i] - series["lam_ours_ph"][i] - series["lam_ours_p"][i]) for i in range(H)]
lam_head = [max(0.0, min(LAM_PH_CAP + LAM_P_CAP - series["lam_ours_ph"][i] - series["lam_ours_p"][i],
                         LAMBDA_GPUS - series["lam_others_ph"][i] - series["lam_ours_ph"][i] - series["lam_ours_p"][i])) for i in range(H)]
lam_head_nopreempt = [min(lam_head[i], lam_free[i]) for i in range(H)]
bk_wait = series["bk_pending"]
S["cap_use"] = {
    "jst_ph_mean_gpus": round(statistics.mean(series["jst_ours_ph"]), 2),
    "jst_ph_share_of_cap_hours": round(sum(series["jst_ours_ph"]) / (JST_CAP * H), 3),
    "jst_hours_at_cap": sum(1 for x in series["jst_ours_ph"] if x >= JST_CAP - 0.5),
    "jst_hours_zero": sum(1 for x in series["jst_ours_ph"] if x < 0.5),
    "lam_mean_gpus": round(statistics.mean([a + b for a, b in zip(series["lam_ours_ph"], series["lam_ours_p"])]), 2),
    "lam_share_of_16cap_hours": round((sum(series["lam_ours_ph"]) + sum(series["lam_ours_p"])) / (16 * H), 3),
    "lam_hours_zero": sum(1 for a, b in zip(series["lam_ours_ph"], series["lam_ours_p"]) if a + b < 0.5),
    "lam_others_mean_gpus": round(statistics.mean(series["lam_others_any"]), 2),
    "lam_free_mean_gpus": round(statistics.mean(lam_free), 2),
    "hours": H,
    "bk_pending_mean_gpus": round(statistics.mean(bk_wait), 1),
    "bk_running_mean_gpus": round(statistics.mean(series["bk_running"]), 1),
    "bk_running_peak": round(max(series["bk_running"])),
}
# idle elsewhere while Beaker work waited (hours with >=4 GPUs pending on Beaker)
idle_while_wait = {"hours_bk_pending_ge4": 0, "jst_headroom_gpuh": 0.0, "lam_free_gpuh": 0.0, "lam_headroom_nopreempt_gpuh": 0.0}
for i in range(H):
    if bk_wait[i] >= 4:
        idle_while_wait["hours_bk_pending_ge4"] += 1
        idle_while_wait["jst_headroom_gpuh"] += jst_head[i]
        idle_while_wait["lam_free_gpuh"] += lam_free[i]
        idle_while_wait["lam_headroom_nopreempt_gpuh"] += lam_head_nopreempt[i]
S["idle_while_beaker_waited"] = {k: round(v) for k, v in idle_while_wait.items()}
S["headroom_total_gpuh"] = {"jst": round(sum(jst_head)), "lam_nopreempt": round(sum(lam_head_nopreempt))}
S["lambda_other_users_gpuh"] = {k: round(v) for k, v in lam_users.most_common(6)}
S["jst_other_users_gpuh_top"] = {k: round(v) for k, v in jst_users.most_common(8)}
with open(OUT / "daily_headroom.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["day", "jst_ours_ph_mean", "jst_headroom_mean", "lam_ours_mean", "lam_others_mean", "lam_free_mean", "bk_running_mean", "bk_pending_mean"])
    for d in DAYS:
        idx = [i for i, h in enumerate(hours_axis) if h.astimezone(PT).date().isoformat() == d]
        m = lambda arr: round(sum(arr[i] for i in idx) / max(1, len(idx)), 2)
        lam_ours = [a + b for a, b in zip(series["lam_ours_ph"], series["lam_ours_p"])]
        w.writerow([d, m(series["jst_ours_ph"]), m(jst_head), m(lam_ours), m(series["lam_others_any"]), m(lam_free), m(series["bk_running"]), m(bk_wait)])

# ---------------------------------------------------------------- 2. queue wait
started = [j for j in gjobs if j["start"] and j["eligible"] and j["start"] >= W0]


def gbucket(g):
    return "1" if g == 1 else "2" if g == 2 else "3-4" if g <= 4 else "5-8" if g <= 8 else ">8"


wait_rows = []
groups = collections.defaultdict(list)
for j in started:
    pl = j["pool"] if j["pool"] != "beaker" else "beaker:" + j["node"]
    groups[(pl, gbucket(j["gpus"]))].append(j["wait_h"])
    groups[(pl, "all")].append(j["wait_h"])
    if j["pool"] == "beaker":
        groups[("beaker:minRuntime=" + ("0" if not j["min_runtime_s"] else f'{int(j["min_runtime_s"] / 3600)}h'), "all")].append(j["wait_h"])
        groups[("beaker:attempt=" + ("first" if j["attempt"] == 0 else "retry"), "all")].append(j["wait_h"])
for (pl, gb), v in sorted(groups.items()):
    if len(v) < 3:
        continue
    wait_rows.append({"pool": pl, "gpus": gb, "count": len(v), "p50_h": round(q(v, .5), 2), "p90_h": round(q(v, .9), 2),
                      "max_h": round(max(v), 1), "mean_h": round(statistics.mean(v), 2),
                      "share_wait_gt_1h": round(sum(1 for x in v if x > 1) / len(v), 3)})
with open(OUT / "waits.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(wait_rows[0]))
    w.writeheader()
    w.writerows(wait_rows)
# histogram per pool (log bins)
bins = [0, 1 / 60, 5 / 60, 0.25, 0.5, 1, 2, 4, 8, 24, 1e9]
labels = ["<1m", "1-5m", "5-15m", "15-30m", "30-60m", "1-2h", "2-4h", "4-8h", "8-24h", ">24h"]
hist = {}
for pool in ("beaker", "jsteinhardt", "lambda"):
    v = [j["wait_h"] for j in started if j["pool"] == pool]
    hist[pool] = [sum(1 for x in v if bins[i] <= x < bins[i + 1]) for i in range(len(labels))]
S["wait_hist"] = {"labels": labels, **hist}
# worst cases
worst = sorted(started, key=lambda j: -j["wait_h"])[:15]
S["worst_waits"] = [{"pool": j["pool"] if j["pool"] != "beaker" else "beaker:" + j["node"], "name": j["name"][:60], "experiment": j["experiment"],
                     "gpus": j["gpus"], "wait_h": round(j["wait_h"], 1), "submit": j["submit"].strftime("%m-%d %H:%MZ"), "min_runtime_s": j["min_runtime_s"]} for j in worst]
# wait share of turnaround per experiment-run (Beaker: per experiment over attempts; slurm: per job)
turn = collections.defaultdict(lambda: [None, None, 0.0, 0.0])
for j in gjobs:
    if not j["submit"]:
        continue
    t = turn[(j["pool"], j["job_id"])]
    t[0] = min(t[0], j["submit"]) if t[0] else j["submit"]
    e = j["end"] or NOW
    t[1] = max(t[1], e) if t[1] else e
    t[2] += j["wait_h"]
    t[3] += j["run_full_h"]
share = collections.defaultdict(lambda: [0.0, 0.0])
for (pool, _), (a, b, wsum, rsum) in turn.items():
    share[pool][0] += wsum
    share[pool][1] += wsum + rsum
S["wait_share_of_turnaround"] = {p: round(a / b, 3) for p, (a, b) in share.items() if b}
S["wait_total_job_hours"] = {p: round(a) for p, (a, b) in share.items()}

# ---------------------------------------------------------------- 3. waste
waste = collections.Counter()
waste_n = collections.Counter()
for j in gjobs:
    st = j["state"]
    if st in ("PREEMPTED", "NODE_FAIL", "FAILED", "CANCELLED", "OUT_OF_MEMORY", "TIMEOUT", "REQUEUED"):
        key = {"OUT_OF_MEMORY": "FAILED", "TIMEOUT": "FAILED", "REQUEUED": "PREEMPTED"}.get(st, st)
        waste[(key, j["kind"])] += j["gpu_h"]
        waste_n[(key, j["kind"])] += 1
S["ended_badly_gpuh"] = {f"{a}|{b}": round(v, 1) for (a, b), v in waste.most_common()}
S["ended_badly_n"] = {f"{a}|{b}": v for (a, b), v in waste_n.most_common()}
# Beaker preemptions in window
bk_pre = [j for j in gjobs if j["pool"] == "beaker" and j["state"] == "PREEMPTED"]
S["beaker_preemptions"] = {"n": len(bk_pre), "gpuh": round(sum(j["gpu_h"] for j in bk_pre)),
                           "train_n": sum(1 for j in bk_pre if j["kind"] == "train"),
                           "train_gpuh": round(sum(j["gpu_h"] for j in bk_pre if j["kind"] == "train")),
                           "after_min_runtime": sum(1 for j in bk_pre if j["min_runtime_s"] and j["run_full_h"] * 3600 >= j["min_runtime_s"]),
                           "minrt0": sum(1 for j in bk_pre if not j["min_runtime_s"])}
pre_by_day = collections.Counter()
for j in bk_pre:
    pre_by_day[(j["end"] or NOW).astimezone(PT).date().isoformat()] += j["gpu_h"]
S["beaker_preempt_gpuh_by_day"] = {d: round(pre_by_day.get(d, 0)) for d in DAYS}
S["beaker_minruntime_share"] = {
    "first_attempts": sum(1 for j in gjobs if j["pool"] == "beaker" and j["attempt"] == 0),
    "with_minrt": sum(1 for j in gjobs if j["pool"] == "beaker" and j["attempt"] == 0 and j["min_runtime_s"]),
    "gpus": dict(collections.Counter(j["gpus"] for j in gjobs if j["pool"] == "beaker" and j["attempt"] == 0)),
    "clusters": dict(collections.Counter(j["node"] for j in gjobs if j["pool"] == "beaker" and j["attempt"] == 0)),
}

# duplicates ("insurance" copies): same run name on two pools, overlapping in time
def norm(n):
    n = (n or "").lower()
    n = re.sub(r"-[0-9a-f]{8}$", "", n)
    n = re.sub(r"[-_](lam|lambda|bk|beaker|js|jst|jsteinhardt|local|slurm|cubbins|mooney|sneetches|mcfuzz|horton)(?=$|[-_])", "", n)
    n = re.sub(r"[-_](r\d|retry\d*|v\d|copy|dup|insurance|ins)$", "", n)
    return n


runs = collections.defaultdict(list)
for j in gjobs:
    if j["kind"] in ("train",) and j["start"]:
        runs[norm(j["run_id"] or j["name"])].append(j)
dups = []
for key, js in runs.items():
    pools = {j["pool"] for j in js}
    if len(pools) < 2:
        continue
    js.sort(key=lambda j: j["start"])
    for a in js:
        for b in js:
            if a is b or a["pool"] >= b["pool"]:
                continue
            if a["start"] < (b["end"] or NOW) and b["start"] < (a["end"] or NOW):
                dups.append((key, a, b))
seen = set()
dup_rows = []
for key, a, b in dups:
    if key in seen:
        continue
    seen.add(key)
    loser = min((a, b), key=lambda j: (j["state"] == "COMPLETED", j["gpu_h"]))
    dup_rows.append({"run": key, "experiment": a["experiment"], "pools": f'{a["pool"]}+{b["pool"]}',
                     "states": f'{a["state"]}/{b["state"]}', "gpuh_a": round(a["gpu_h"], 1), "gpuh_b": round(b["gpu_h"], 1),
                     "smaller_copy_gpuh": round(min(a["gpu_h"], b["gpu_h"]), 1)})
with open(OUT / "duplicates.csv", "w", newline="") as f:
    if dup_rows:
        w = csv.DictWriter(f, fieldnames=list(dup_rows[0]))
        w.writeheader()
        w.writerows(dup_rows)
S["duplicates"] = {"n_runs": len(dup_rows), "smaller_copy_gpuh": round(sum(r["smaller_copy_gpuh"] for r in dup_rows)),
                   "by_experiment": dict(collections.Counter(r["experiment"] for r in dup_rows))}

# expected vs actual
ratios = []
over, under = [], []
for j in gjobs:
    em = j["expected_min"]
    if not em or not j["start"] or j["state"] not in ("COMPLETED",) or j["attempt"]:
        continue
    try:
        em = float(em)
    except (TypeError, ValueError):
        continue
    if em <= 0:
        continue
    r = j["run_full_h"] * 60 / em
    ratios.append(r)
    if r > 2:
        over.append(j)
    if r < 0.25:
        under.append(j)
S["expected_vs_actual"] = {"n": len(ratios), "p10": round(q(ratios, .1), 2), "p50": round(q(ratios, .5), 2), "p90": round(q(ratios, .9), 2),
                           "n_over_2x": len(over), "n_under_quarter": len(under),
                           "over_2x_gpuh": round(sum(j["gpu_h"] for j in over))}
# smokes
smk = [j for j in gjobs if j["kind"] == "smoke"]
smk_groups = collections.Counter(norm(re.sub(r"\d+$", "", j["run_id"] or j["name"])) for j in smk)
S["smokes"] = {"n": len(smk), "gpuh": round(sum(j["gpu_h"] for j in smk)), "failed_or_cancelled": sum(1 for j in smk if j["state"] not in ("COMPLETED", "RUNNING")),
               "names_repeated_3plus": sum(1 for v in smk_groups.values() if v >= 3)}

# ---------------------------------------------------------------- 4. latency: job end -> result recorded, train end -> eval launch
job_end = {}
for j in gjobs:
    if j["end"] and j["state"] == "COMPLETED":
        job_end[j["job_id"]] = max(job_end.get(j["job_id"], j["end"]), j["end"])
lat = []
for jid, rs in res_by_job.items():
    if jid in job_end:
        rec = min(utc(r["recorded_at"]) for r in rs if r.get("recorded_at"))
        if rec >= W0:
            lat.append(hours(job_end[jid], rec) if rec > job_end[jid] else 0.0)
S["end_to_result_h"] = {"n": len(lat), "p50": round(q(lat, .5), 1) if lat else None, "p90": round(q(lat, .9), 1) if lat else None,
                        "max": round(max(lat), 1) if lat else None, "share_gt_24h": round(sum(1 for x in lat if x > 24) / len(lat), 3) if lat else None}
# completed eval jobs with no result row (by job id or run id)
evals_done = [j for j in gjobs if j["kind"] == "eval" and j["state"] == "COMPLETED" and j["registered"] and j["end"] and j["end"] < NOW - dt.timedelta(hours=24)]
ev_ids = {j["job_id"]: j for j in evals_done}
def has_result(j):
    if j["job_id"] in res_by_job or res_by_run.get(j["run_id"]):
        return True
    parent = (launch_by_run.get(j["run_id"]) or {}).get("parent_run")
    return any(utc(r["recorded_at"]) >= j["start"] for r in res_by_run.get(parent, []) if r.get("recorded_at")) if parent else False


unrec = [j for jid, j in ev_ids.items() if not has_result(j)]
S["evals_without_results"] = {"completed_eval_jobs": len(ev_ids), "no_result_row": len(unrec),
                              "gpuh": round(sum(j["gpu_h"] for j in unrec)),
                              "by_experiment": dict(collections.Counter(j["experiment"] for j in unrec).most_common(8))}
# train runs (completed, >24h ago) with no eval child and no result row
children = collections.defaultdict(list)
for d in all_launches:
    if d.get("parent_run"):
        children[d["parent_run"]].append(d)
trains = {}
for j in gjobs:
    if j["kind"] == "train" and j["state"] == "COMPLETED" and j["run_id"] and j["end"] and j["end"] < NOW - dt.timedelta(hours=24):
        trains[j["run_id"]] = j
orphan = [j for r, j in trains.items() if not children.get(r) and not res_by_run.get(r)]
S["train_without_eval_or_result"] = {"completed_train_runs": len(trains), "orphans": len(orphan),
                                     "gpuh": round(sum(j["gpu_h"] for j in orphan)),
                                     "by_experiment": dict(collections.Counter(j["experiment"] for j in orphan).most_common(8))}
# train end -> first eval launch
gap = []
for r, j in trains.items():
    ch = [utc(c["launched_at"]) for c in children.get(r, []) if c.get("launched_at") and c.get("kind") == "eval"]
    ch = [c for c in ch if c]
    if ch:
        first = min(ch)
        gap.append(hours(j["end"], first) if first > j["end"] else 0.0)
S["train_end_to_eval_launch_h"] = {"n": len(gap), "p50": round(q(gap, .5), 2) if gap else None, "p90": round(q(gap, .9), 1) if gap else None,
                                   "share_chained_lt_10min": round(sum(1 for x in gap if x < 1 / 6) / len(gap), 3) if gap else None,
                                   "share_gt_12h": round(sum(1 for x in gap if x > 12) / len(gap), 3) if gap else None}

# ---------------------------------------------------------------- 5. process gaps
status_pages = {p.name for p in Path("/accounts/projects/berkeleynlp/prasann/ops/experiments").iterdir() if (p / "STATUS.md").exists()}
nopage = collections.Counter()
for j in gjobs:
    if j["experiment"] not in status_pages:
        nopage[j["experiment"]] += j["gpu_h"]
S["gpuh_experiment_without_status_page"] = {"total": round(sum(nopage.values())), "by_name": {k: round(v) for k, v in nopage.most_common(12)}}
pools_by_run = collections.defaultdict(collections.Counter)
for j in gjobs:
    if j["kind"] == "train":
        pools_by_run[norm(j["run_id"] or j["name"])][j["pool"]] += j["gpu_h"]
multi = {k: v for k, v in pools_by_run.items() if len(v) > 1}
S["same_run_on_two_pools_any_time"] = {"n_runs": len(multi), "smaller_copy_gpuh": round(sum(min(v.values()) for v in multi.values())),
                                       "examples": list(multi)[:8]}
unreg = [j for j in gjobs if not j["registered"]]
S["unregistered"] = {"n": len(unreg), "gpuh": round(sum(j["gpu_h"] for j in unreg)),
                     "by_pool": dict(collections.Counter(j["pool"] for j in unreg)),
                     "gpuh_by_pool": {p: round(sum(j["gpu_h"] for j in unreg if j["pool"] == p)) for p in {j["pool"] for j in unreg}},
                     "top_names": collections.Counter(re.sub(r"[-_][0-9a-f]{8}$|\d+$", "", j["name"])[:40] for j in unreg).most_common(12)}
# registered runs whose final state was never recorded
no_final = [j for j in gjobs if j["registered"] and j["run_id"] and not finals.get(j["run_id"]) and j["end"] and j["end"] < NOW - dt.timedelta(hours=24)]
S["registered_no_final_state"] = {"n": len({j['run_id'] for j in no_final}), "of": len({j['run_id'] for j in gjobs if j['registered'] and j['run_id']})}
# concurrent sessions on the jsteinhardt cap: distinct experiments holding preemptive_high at once
conc = collections.Counter()
for i, h in enumerate(hours_axis):
    exps = {j["experiment"] for j in gjobs if j["pool"] == "jsteinhardt" and j["qos"] == "preemptive_high" and j["start"] and j["start"] <= h < (j["end"] or NOW)}
    conc[len(exps)] += 1
S["jst_concurrent_experiments_hours"] = dict(sorted(conc.items()))
pend_cap = sum(1 for j in gjobs if j["pool"] == "jsteinhardt" and j["qos"] == "preemptive_high" and j["wait_h"] > 0.25)
S["jst_ph_jobs_waited_gt15m"] = pend_cap
# Lambda: did our starts coincide with preempting another user?
lam_pre_others = []
for r in csv.DictReader(open(RAW / "sacct_lambda_all.psv"), delimiter="|"):
    if r["User"] != "prasann" and r["State"].startswith("PREEMPTED") and pt(r["End"]) and pt(r["End"]) >= W0:
        lam_pre_others.append({"user": r["User"], "job": r["JobID"], "end": r["End"], "gpus": gpus_of(r["AllocTRES"]),
                               "ran_h": round(int(r["ElapsedRaw"] or 0) / 3600, 2), "qos": r["QOS"]})
lam_all = list(csv.DictReader(open(RAW / "sacct_lambda_all.psv"), delimiter="|"))
for x in lam_pre_others:
    e = pt(x["end"])
    node = next((r["NodeList"] for r in lam_all if r["JobID"] == x["job"]), "")
    x["node"] = node
    starters = [r for r in lam_all if r["NodeList"] == node and pt(r["Start"]) and abs((pt(r["Start"]) - e).total_seconds()) <= 180
                and r["QOS"] in ("preemptive_high", "preemptive") and r["JobID"] != x["job"]]
    x["preempted_by_user"] = ",".join(sorted({r["User"] for r in starters})) or "?"
    x["preempted_by_qos"] = ",".join(sorted({r["QOS"] for r in starters}))
S["lambda_other_users_preempted"] = {"n": len(lam_pre_others), "gpuh_lost": round(sum(x["gpus"] * x["ran_h"] for x in lam_pre_others)),
                                     "by_user": dict(collections.Counter(x["user"] for x in lam_pre_others)),
                                     "by_preemptor": dict(collections.Counter(x["preempted_by_user"] for x in lam_pre_others)),
                                     "gpuh_by_preemptor": {u: round(sum(x["gpus"] * x["ran_h"] for x in lam_pre_others if x["preempted_by_user"] == u))
                                                           for u in {x["preempted_by_user"] for x in lam_pre_others}},
                                     "by_day_prasann": dict(collections.Counter(x["end"][:10] for x in lam_pre_others if "prasann" in x["preempted_by_user"]))}
jst_pre_others = collections.Counter()
for r in csv.DictReader(open(RAW / "sacct_berkeley_all.psv"), delimiter="|"):
    if r["User"] != "prasann" and r["Partition"] == "jsteinhardt" and r["State"].startswith("PREEMPTED") and pt(r["End"]) and pt(r["End"]) >= W0:
        jst_pre_others[r["User"]] += 1
S["jst_other_users_preempted_jobs"] = dict(jst_pre_others.most_common(8))

# other users' names stay out of git (the repo remote is public): pseudonymize in data/, map in raw/
umap = {}


def anon(u):
    if u in ("prasann", "?") or not u:
        return u
    return ",".join(umap.setdefault(x, f"user{len(umap) + 1}") if x not in ("prasann", "?") else x for x in u.split(","))


for key in ("lambda_other_users_gpuh", "jst_other_users_gpuh_top", "jst_other_users_preempted_jobs"):
    S[key] = {anon(k): v for k, v in S[key].items()}
lp = S["lambda_other_users_preempted"]
for key in ("by_user", "by_preemptor", "gpuh_by_preemptor"):
    lp[key] = {anon(k): v for k, v in lp[key].items()}
for x in lam_pre_others:
    x["user"], x["preempted_by_user"] = anon(x["user"]), anon(x["preempted_by_user"])
json.dump(umap, open(RAW / "user_map.json", "w"), indent=1)
json.dump(S, open(OUT / "summary.json", "w"), indent=1, default=str)
json.dump(lam_pre_others, open(OUT / "lambda_others_preempted.json", "w"), indent=1)
print(json.dumps(S, indent=1, default=str)[:12000])
