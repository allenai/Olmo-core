"""Live job status (``ops status``) and single-job stall diagnosis (``ops stall``).

Design rules: one snapshot, never a polling loop; every external call has a timeout and degrades to
"unavailable"; nothing here cancels, submits or modifies a job. Beaker results are cached for 60 s in
``state/cache/`` because the CLI is the slow part (``--fresh`` bypasses the cache).

Sources:
  * slurm (Berkeley): ``squeue`` for live jobs, ``sacct`` for jobs that ended in the last 24 h.
  * Beaker: ``beaker job list --author <id> --cluster <c>`` (~1 s; only non-finalized jobs) plus
    ``beaker experiment get --format json`` for names and for recently-ended jobs. State comes from
    ``jobs[-1].status`` keys (the table view lies about canceled jobs).
  * Lambda: one ``ssh lambda`` call running squeue + sacct (skipped with ``--no-lambda`` or on failure).
    Lambda's sbatch often reports FAILED for runs that fully succeeded; such rows are flagged with
    ``?`` and are never recorded as failed.
"""
import argparse
import base64
import datetime as dt
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time

from . import paths

USER = os.environ.get("OPS_USER", "prasann")
LOGIN_HOSTS = {"radagast"}
BEAKER_AUTHOR_ID = "01K31JPADAHRBWED2VT05703Z0"  # prasanns; refreshed from `beaker account whoami`
BEAKER_CLUSTERS = [c for c in os.environ.get("OPS_BEAKER_CLUSTERS", "ai2/jupiter").split(",") if c]
BEAKER_TTL = 60
STALE_LOG_S = 15 * 60
PENDING_FLAG_S = 3600
NFS_PREFIXES = ("/accounts/", "/scratch/", "/net/", "/usr/local/linux/", "/system/linux/")
FAILED_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}
LIVE_STATES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING",
               "STARTING", "STAGE_OUT", "SIGNALING"}
FINAL_MAP = {"COMPLETED": "done", "FAILED": "failed", "TIMEOUT": "failed", "OUT_OF_MEMORY": "failed",
             "NODE_FAIL": "failed", "BOOT_FAIL": "failed", "DEADLINE": "failed", "CANCELLED": "cancelled",
             "PREEMPTED": "cancelled"}

SQUEUE_FIELDS = "JobID:|,Name:|,State:|,TimeUsed:|,TimeLimit:|,NodeList:|,Reason:|,Partition:|,QOS:|,SubmitTime:|,StdOut:|,WorkDir:|"
LAMBDA_SQUEUE_FMT = "%i|%j|%T|%M|%l|%N|%r|%P|%q|%V"
SACCT_FIELDS = "JobID,JobName,State,Elapsed,Timelimit,NodeList,Submit,Start,End,ExitCode,Partition,WorkDir"


# ================================================================================================
# subprocess / threading helpers (all bounded)


def run(cmd, timeout=10, input_text=None):
    """Run ``cmd`` (list) with a hard timeout. Returns ``(rc, stdout, stderr)``; never raises.

    rc 124 = timed out, 127 = binary not found.
    """
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                             errors="replace", start_new_session=True)
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except Exception as e:  # noqa: BLE001
        return 126, "", str(e)
    try:
        out, err = p.communicate(input_text, timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, 9)
        except Exception:  # noqa: BLE001
            pass
        try:
            out, err = p.communicate(timeout=2)
        except Exception:  # noqa: BLE001
            out, err = "", ""
        return 124, out or "", (err or "") + f"\n[timeout after {timeout}s]"


def call_with_timeout(fn, timeout, default=None):
    """Run ``fn()`` in a daemon thread; return ``default`` if it does not finish in ``timeout`` s.

    Daemon threads matter: an ``os.stat`` stuck on a wedged NFS mount must not block interpreter exit.
    """
    box = {}

    def target():
        try:
            box["v"] = fn()
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if "v" in box:
        return box["v"]
    return default


def parallel(fns, timeout):
    """Run ``{name: fn}`` concurrently in daemon threads. Returns ``{name: result | TimeoutError}``."""
    results, threads = {}, {}
    for name, fn in fns.items():
        def target(name=name, fn=fn):
            try:
                results[name] = fn()
            except Exception as e:  # noqa: BLE001
                results[name] = e
        t = threading.Thread(target=target, daemon=True)
        t.start()
        threads[name] = t
    deadline = time.time() + timeout
    for name, t in threads.items():
        t.join(max(0.0, deadline - time.time()))
        if name not in results:
            results[name] = TimeoutError(f"{name}: no answer within {timeout}s")
    return results


# ================================================================================================
# parsing helpers (pure)


def parse_duration(s):
    """slurm ``[D-]HH:MM:SS`` / ``MM:SS`` / ``M:SS`` -> seconds (None for UNLIMITED/INVALID/blank)."""
    s = (s or "").strip()
    if not s or not re.match(r"^[0-9:\-.]+$", s):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [float(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts[-3:]
    return int(days * 86400 + h * 3600 + m * 60 + sec)


def parse_ts(s):
    """ISO-ish timestamp -> epoch seconds. Naive (slurm) timestamps are local time. None if unparsable."""
    s = (s or "").strip()
    if not s or s in ("Unknown", "None", "N/A"):
        return None
    try:
        s2 = s.replace("Z", "+00:00")
        m = re.match(r"^(.*\.\d{6})\d*(.*)$", s2)  # beaker has nanoseconds sometimes
        if m:
            s2 = m.group(1) + m.group(2)
        d = dt.datetime.fromisoformat(s2)
        return d.timestamp()
    except ValueError:
        return None


def fmt_dur(s):
    if s is None:
        return "-"
    s = int(s)
    if s < 0:
        s = 0
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s" if s < 600 else f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def expand_slurm_path(pattern, job_id, name, user=USER, node=None):
    """Expand slurm filename patterns (%j %x %u %N %A %a %%) in an --output path."""
    if not pattern:
        return None
    rep = {"j": str(job_id), "J": str(job_id), "x": name or "", "u": user, "A": str(job_id).split("_")[0],
           "a": str(job_id).split("_")[1] if "_" in str(job_id) else "", "N": (node or "").split(",")[0],
           "n": "0", "t": "0", "s": "batch", "%": "%"}
    return re.sub(r"%(\d*)([jJxuANantNs%])", lambda m: rep.get(m.group(2), m.group(0)), pattern)


def parse_squeue(text, cluster="slurm", now=None):
    """Parse ``squeue -h -O <SQUEUE_FIELDS>`` (or the shorter Lambda ``-o`` format) output."""
    now = now or time.time()
    jobs = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        f = line.split("|")
        if len(f) < 10:
            continue
        jid, name, state, used, limit, node, reason, part, qos, submit = [x.strip() for x in f[:10]]
        stdout = f[10].strip() if len(f) > 10 else ""
        workdir = f[11].strip() if len(f) > 11 else ""
        state = state.split()[0] if state else state
        submit_ts = parse_ts(submit)
        j = {
            "cluster": cluster, "job_id": jid, "name": name, "state": state,
            "elapsed_s": parse_duration(used) if state != "PENDING" else None,
            "timelimit_s": parse_duration(limit), "node": node if node and node != "(null)" else None,
            "reason": reason if reason and reason != "None" else None, "partition": part, "qos": qos,
            "submit_ts": submit_ts, "workdir": workdir or None,
            "pending_s": (now - submit_ts) if state == "PENDING" and submit_ts else None,
            "log_path": expand_slurm_path(stdout, jid, name, node=node) if stdout else None,
            "live": True,
        }
        jobs.append(j)
    return jobs


def parse_sacct(text, cluster="slurm"):
    """Parse ``sacct -X -n -P --format=<SACCT_FIELDS>`` output."""
    jobs = []
    for line in (text or "").splitlines():
        f = line.split("|")
        if len(f) < 10:
            continue
        jid, name, state, elapsed, limit, node, submit, start, end, exitcode = [x.strip() for x in f[:10]]
        part = f[10].strip() if len(f) > 10 else ""
        workdir = f[11].strip() if len(f) > 11 else ""
        st = state.split()[0] if state else ""
        jobs.append({
            "cluster": cluster, "job_id": jid, "name": name, "state": st, "state_detail": state,
            "elapsed_s": parse_duration(elapsed), "timelimit_s": parse_duration(limit),
            "node": node if node and node not in ("None assigned", "(null)") else None,
            "submit_ts": parse_ts(submit), "start_ts": parse_ts(start), "end_ts": parse_ts(end),
            "exit_code": exitcode, "partition": part, "workdir": workdir or None,
            "live": st in LIVE_STATES, "reason": None, "pending_s": None, "log_path": None,
        })
    return jobs


def beaker_state(status):
    """Map a Beaker ``jobs[-1].status`` dict to a slurm-like state word."""
    status = status or {}
    if "canceled" in status:
        return "CANCELLED"
    if "finalized" in status:
        if "exitCode" not in status:
            return "PREEMPTED"  # canceled/preempted jobs finalize with no exit code
        return "COMPLETED" if status.get("exitCode") == 0 else "FAILED"
    if "exited" in status:
        return "COMPLETING"
    if "started" in status:
        return "RUNNING"
    if "scheduled" in status:
        return "STARTING"
    return "PENDING"


def beaker_job_row(job, exp_name=None, now=None):
    """Convert one Beaker job JSON object to a status row."""
    now = now or time.time()
    st = job.get("status") or {}
    state = beaker_state(st)
    created, started = parse_ts(st.get("created")), parse_ts(st.get("started"))
    ended = parse_ts(st.get("exited") or st.get("finalized") or st.get("canceled"))
    exp = (job.get("execution") or {}).get("experiment")
    elapsed = None
    if started:
        elapsed = (ended or now) - started
    node = job.get("node")
    return {
        "cluster": "beaker", "job_id": job.get("id"), "exp_id": exp, "name": exp_name or exp or job.get("id"),
        "state": state, "elapsed_s": elapsed, "timelimit_s": None, "node": node[-6:] if node else None,
        "reason": "prio=" + str((job.get("execution") or {}).get("spec", {}).get("context", {}).get("priority")),
        "pending_s": (now - created) if state in ("PENDING", "STARTING") and created else None,
        "submit_ts": created, "end_ts": ended, "exit_code": st.get("exitCode"),
        "live": state in LIVE_STATES, "log_path": None, "workspace": job.get("workspace"),
    }


# ================================================================================================
# registry access (agent A owns registry.py; degrade gracefully until it exists)


def _fold_runs(path):
    runs = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rid = row.get("run_id")
                if not rid:
                    continue
                cur = runs.setdefault(rid, {})
                for k, v in row.items():
                    if v is not None:
                        cur[k] = v
    except FileNotFoundError:
        pass
    return runs


def load_registry(path=None):
    """``registry.load_runs()`` if available, else a local fold of ``state/runs.jsonl``."""
    try:
        from . import registry  # noqa: WPS433 (lazy on purpose)

        return registry.load_runs(path) if path else registry.load_runs()
    except Exception:  # noqa: BLE001 - missing module or a bug there must not break status
        return _fold_runs(path or paths.RUNS_PATH)


def registry_update(run_id, **fields):
    from . import registry  # noqa: WPS433

    return registry.update(run_id, **fields)


def match_run(job, runs, by_job=None):
    """Find the registry row for a live/recent job: by job_id (or Beaker exp id), else by name."""
    if by_job is None:
        by_job = index_by_job(runs)
    for key in (job.get("job_id"), job.get("exp_id")):
        if key and str(key) in by_job:
            for r in by_job[str(key)]:
                rc = r.get("cluster")
                if not rc or rc == job["cluster"] or (rc in ("slurm", "local") and job["cluster"] == "slurm"):
                    return r
    name = job.get("name")
    if name and name in runs:
        return runs[name]
    return None


def index_by_job(runs):
    by = {}
    for r in runs.values():
        if r.get("job_id"):
            by.setdefault(str(r["job_id"]), []).append(r)
    return by


# ================================================================================================
# collectors


def collect_slurm(now=None):
    rc, out, err = run(["squeue", "-u", USER, "-h", "-O", SQUEUE_FIELDS], timeout=10)
    if rc != 0:
        raise RuntimeError(f"squeue rc={rc}: {err.strip()[:120]}")
    live = parse_squeue(out, "slurm", now)
    rc, out2, err2 = run(["sacct", "-u", USER, "-X", "-n", "-P", "-S", "now-24hours",
                          f"--format={SACCT_FIELDS}"], timeout=10)
    recent = parse_sacct(out2, "slurm") if rc == 0 else []
    live_ids = {j["job_id"] for j in live}
    recent = [j for j in recent if j["job_id"] not in live_ids and not j["live"]]
    return {"live": live, "recent": recent, "note": None if rc == 0 else "sacct unavailable"}


def collect_lambda(now=None):
    remote = (f"squeue -u $USER -h -o '{LAMBDA_SQUEUE_FMT}'; echo __OPS_SACCT__; "
              f"sacct -u $USER -X -n -P -S now-24hours --format={SACCT_FIELDS}")
    rc, out, err = run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "lambda", remote], timeout=12)
    if rc != 0 or "__OPS_SACCT__" not in out:
        raise RuntimeError(f"ssh lambda rc={rc}: {err.strip()[:120]}")
    sq, sa = out.split("__OPS_SACCT__", 1)
    live = parse_squeue(sq, "lambda", now)
    live_ids = {j["job_id"] for j in live}
    recent = [j for j in parse_sacct(sa, "lambda") if j["job_id"] not in live_ids and not j["live"]]
    return {"live": live, "recent": recent, "note": None}


def beaker_bin():
    b = shutil.which("beaker")
    if b:
        return b
    cand = os.path.expanduser("~/.local/bin/beaker")
    return cand if os.path.exists(cand) else "beaker"


def _cache_read(name, ttl=None):
    p = paths.CACHE_DIR / name
    try:
        with open(p) as fh:
            d = json.load(fh)
        if ttl is not None and time.time() - d.get("_ts", 0) > ttl:
            return None
        return d
    except Exception:  # noqa: BLE001
        return None


def _cache_write(name, data):
    try:
        paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = dict(data, _ts=time.time())
        tmp = paths.CACHE_DIR / (name + f".{os.getpid()}.tmp")
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, paths.CACHE_DIR / name)
    except Exception:  # noqa: BLE001
        pass


def beaker_exp_get(exp_or_job_id, timeout=10):
    """``beaker experiment get --format json`` -> (exp_name, last_job_json) or raise."""
    b = beaker_bin()
    rc, out, err = run([b, "experiment", "get", exp_or_job_id, "--format", "json"], timeout=timeout)
    if rc == 0 and out.strip():
        d = json.loads(out)[0]
        jobs = d.get("jobs") or []
        return d.get("name"), (jobs[-1] if jobs else {"id": None, "status": {}, "execution": {"experiment": d.get("id")}})
    rc, out, err = run([b, "job", "get", exp_or_job_id, "--format", "json"], timeout=timeout)
    if rc == 0 and out.strip():
        j = json.loads(out)[0]
        return None, j
    raise RuntimeError(f"beaker lookup {exp_or_job_id} failed: {err.strip()[:100]}")


def collect_beaker(runs, fresh=False, now=None):
    """Live Beaker jobs (cached BEAKER_TTL s) + recently-ended ones we know about."""
    now = now or time.time()
    cached = None if fresh else _cache_read("beaker_snapshot.json", BEAKER_TTL)
    if cached:
        return {"live": cached["live"], "recent": cached["recent"], "known_final": cached.get("known_final", {}),
                "note": f"cached {int(now - cached['_ts'])}s ago"}
    b = beaker_bin()
    who = _cache_read("beaker_whoami.json") or {}
    author = who.get("id")
    if not author:
        rc, out, _ = run([b, "account", "whoami", "--format", "json"], timeout=8)
        try:
            author = json.loads(out)[0]["id"]
            _cache_write("beaker_whoami.json", {"id": author})
        except Exception:  # noqa: BLE001
            author = BEAKER_AUTHOR_ID
    lists = parallel({c: (lambda c=c: run([b, "job", "list", "--author", author, "--cluster", c,
                                           "--format", "json"], timeout=12)) for c in BEAKER_CLUSTERS}, 14)
    raw, errors = {}, []
    for c, res in lists.items():
        if isinstance(res, Exception) or res[0] != 0:
            errors.append(c)
            continue
        try:
            for j in json.loads(res[1] or "[]"):
                raw[j["id"]] = j
        except ValueError:
            errors.append(c)
    if errors and not raw:
        raise RuntimeError(f"beaker job list failed for {','.join(errors)}")

    names = _cache_read("beaker_names.json") or {}
    names.pop("_ts", None)
    final = _cache_read("beaker_final.json") or {}
    final.pop("_ts", None)
    seen = _cache_read("beaker_seen.json") or {}
    seen.pop("_ts", None)

    need = {(j.get("execution") or {}).get("experiment") for j in raw.values()}
    need = [e for e in need if e and e not in names][:30]
    # recently-ended candidates: jobs seen live before but gone now, + registered beaker runs w/o final_state
    gone = [v["exp_id"] for k, v in seen.items() if k not in raw and v.get("exp_id") and now - v.get("t", 0) < 86400]
    for r in runs.values():
        if r.get("cluster") == "beaker" and r.get("job_id") and not r.get("final_state"):
            launched = parse_ts(r.get("launched_at") or r.get("ts"))
            if launched is None or now - launched < 14 * 86400:
                gone.append(str(r["job_id"]))
    live_exps = {(j.get("execution") or {}).get("experiment") for j in raw.values()} | set(raw)
    gone = [g for g in dict.fromkeys(gone) if g not in live_exps and g not in final][:8]
    lookups = parallel({e: (lambda e=e: beaker_exp_get(e)) for e in dict.fromkeys(need + gone)}, 15)
    recent = []
    for e, res in lookups.items():
        if isinstance(res, Exception):
            continue
        name, job = res
        if name:
            names[e] = name
        if e in gone:
            row = beaker_job_row(job, name, now)
            row["lookup_id"] = e
            if not row["live"]:
                final[e] = row
    for e, row in final.items():
        if row.get("end_ts") and now - row["end_ts"] < 86400:
            recent.append(row)
    live = []
    for j in raw.values():
        exp = (j.get("execution") or {}).get("experiment")
        row = beaker_job_row(j, names.get(exp), now)
        live.append(row)
        seen[j["id"]] = {"exp_id": exp, "t": now}
    seen = {k: v for k, v in seen.items() if now - v.get("t", 0) < 2 * 86400}
    final = {k: v for k, v in final.items() if now - (v.get("end_ts") or now) < 14 * 86400}
    _cache_write("beaker_names.json", names)
    _cache_write("beaker_final.json", final)
    _cache_write("beaker_seen.json", seen)
    note = f"partial: {','.join(errors)} failed" if errors else None
    _cache_write("beaker_snapshot.json", {"live": live, "recent": recent, "known_final": final})
    return {"live": live, "recent": recent, "note": note, "known_final": final}


# ================================================================================================
# flags + rows


def is_nfs_path(p):
    return bool(p) and str(p).startswith(NFS_PREFIXES)


def log_stat_path(log_path, node, here=None):
    """Where to stat a log from this host: node-local /data or /tmp go through /net/<node> (inspection)."""
    here = here or socket.gethostname().split(".")[0]
    if not log_path:
        return None
    if log_path.startswith(("/data/", "/tmp/")) and node and node.split(",")[0] != here:
        n = re.sub(r"\[.*$", "", node.split(",")[0])
        if log_path.startswith("/tmp/"):
            return None  # /tmp is not exported under /net
        return f"/net/{n}{log_path}"
    return log_path


def stat_mtime(path, timeout=3.0):
    """Single bounded ``os.stat``: returns mtime, ``"missing"`` or None (unavailable/timeout)."""
    def f():
        try:
            return os.stat(path).st_mtime
        except FileNotFoundError:
            return "missing"
        except OSError:
            return None
    return call_with_timeout(f, timeout)


def compute_flags(row, now=None):
    """Pure: flags for one status row (keys as built by ``build_rows``)."""
    now = now or time.time()
    flags = []
    state = row.get("state") or ""
    live = row.get("live")
    if live and not row.get("run_id"):
        flags.append("UNREGISTERED")
    exp_min = row.get("expected_minutes")
    el = row.get("elapsed_s")
    if state == "RUNNING" and exp_min and el and el > 2 * float(exp_min) * 60:
        flags.append(f"OVERRUN({el / (float(exp_min) * 60):.1f}x)")
    age = row.get("log_age_s")
    if state == "RUNNING" and age is not None and age > STALE_LOG_S and (el or 0) > STALE_LOG_S:
        flags.append(f"STALE-LOG({int(age // 60)}m)")
    if state == "RUNNING" and row.get("log_missing") and (el or 0) > 600:
        flags.append("NO-LOG")
    if state == "RUNNING" and row.get("cluster") == "slurm" and is_nfs_path(row.get("log_path")) \
            and not row.get("registered_log"):
        flags.append("LOG-ON-NFS")
    if state in ("PENDING", "STARTING") and (row.get("pending_s") or 0) > PENDING_FLAG_S:
        flags.append(f"PENDING>1h({row.get('reason') or '?'})")
    if not live and state in FAILED_STATES:
        flags.append("FAILED-RECENT?(lambda exit codes lie: check loss/ckpt)" if row.get("cluster") == "lambda"
                     else "FAILED-RECENT")
    if not live and state == "PREEMPTED":
        flags.append("PREEMPTED")
    if row.get("no_job"):
        flags.append("NO-JOB-FOUND")
    return flags


def build_rows(snap, runs, now=None, stat_logs=True, here=None):
    """Join job snapshots with the registry. Returns a list of rows (dicts)."""
    now = now or time.time()
    by_job = index_by_job(runs)
    rows, matched = [], set()
    to_stat = []
    for src in ("slurm", "beaker", "lambda"):
        s = snap.get(src)
        if not isinstance(s, dict):
            continue
        for j in s.get("live", []) + s.get("recent", []):
            r = match_run(j, runs, by_job)
            row = dict(j)
            row["run_id"] = r.get("run_id") if r else None
            row["experiment"] = r.get("experiment") if r else None
            row["expected_minutes"] = r.get("expected_minutes") if r else None
            row["final_state"] = r.get("final_state") if r else None
            if r and r.get("log_path"):
                row["log_path"] = r["log_path"]
                row["registered_log"] = True
            if r:
                matched.add(r["run_id"])
            if stat_logs and row.get("state") == "RUNNING" and row.get("log_path") and src != "lambda":
                p = log_stat_path(row["log_path"], row.get("node"), here)
                if p:
                    to_stat.append((row, p))
            rows.append(row)
    if to_stat:
        res = parallel({i: (lambda p=p: stat_mtime(p)) for i, (_, p) in enumerate(to_stat)}, 4)
        for i, (row, p) in enumerate(to_stat):
            m = res.get(i)
            if m == "missing":
                row["log_missing"] = True
            elif isinstance(m, float):
                row["log_age_s"] = max(0.0, now - m)
    known = (snap.get("beaker") or {}).get("known_final") or {} if isinstance(snap.get("beaker"), dict) else {}
    for r in runs.values():
        if r["run_id"] in matched:
            continue
        obs = known.get(str(r.get("job_id"))) if r.get("cluster") == "beaker" else None
        if obs and not r.get("final_state"):
            row = dict(obs, run_id=r["run_id"], name=r["run_id"], experiment=r.get("experiment"),
                       expected_minutes=r.get("expected_minutes"), final_state=None, job_id=r.get("job_id"),
                       registry_only=True, observed=True, launched_ts=parse_ts(r.get("launched_at")))
            rows.append(row)
            continue
        rows.append({
            "cluster": r.get("cluster"), "job_id": r.get("job_id"), "name": r["run_id"], "run_id": r["run_id"],
            "experiment": r.get("experiment"), "state": (r.get("final_state") or "?").upper(),
            "final_state": r.get("final_state"), "expected_minutes": r.get("expected_minutes"),
            "node": r.get("node"), "live": False, "registry_only": True,
            "no_job": not r.get("final_state") and r.get("cluster") in ("slurm", "beaker", "lambda"),
            "launched_ts": parse_ts(r.get("launched_at")),
        })
    for row in rows:
        row["flags"] = compute_flags(row, now)
    return rows


def select_rows(rows, experiment=None, show_all=False, show_recent=False, since_s=7 * 86400, now=None):
    """Pick rows to display. Registry-only rows (no live/24h job) need ``--all`` or ``--experiment``
    and are limited to runs launched within ``since_s``."""
    now = now or time.time()
    out = []
    for r in rows:
        if experiment and r.get("experiment") != experiment:
            continue
        if r.get("registry_only"):
            recent_launch = r.get("launched_ts") is not None and now - r["launched_ts"] <= since_s
            if (show_all or experiment) and recent_launch:
                out.append(r)
            continue
        if r.get("live") or show_recent or show_all or r.get("run_id") or \
                any(f.startswith(("FAILED-RECENT", "PREEMPTED")) for f in r["flags"]):
            out.append(r)
    order = {"RUNNING": 0, "COMPLETING": 1, "STARTING": 2, "PENDING": 3}
    out.sort(key=lambda r: (order.get(r.get("state"), 5) if not r.get("registry_only") else 6,
                            r.get("cluster") or "" if r.get("live") else "",
                            -(r.get("elapsed_s") or 0) if r.get("live") else -(r.get("end_ts") or 0)))
    return out


def render_table(rows, now=None):
    now = now or time.time()
    head = ["RUN / JOB", "EXP", "CL", "JOBID", "STATE", "ELAPSED", "EXPECT", "NODE", "FLAGS"]
    lines = []
    for r in rows:
        name = r.get("run_id") or r.get("name") or "?"
        if len(name) > 44:
            name = name[:20] + "…" + name[-23:]
        st = r.get("state") or "?"
        el = r.get("elapsed_s")
        if st in ("PENDING", "STARTING"):
            elapsed = "q " + fmt_dur(r.get("pending_s"))
        elif r.get("registry_only") and not r.get("observed"):
            elapsed = "-"
        else:
            elapsed = fmt_dur(el)
        exp_m = r.get("expected_minutes")
        jid = str(r.get("job_id") or "-")
        if r.get("cluster") == "beaker" and len(jid) > 12:
            jid = "…" + jid[-10:]
        lines.append([name, (r.get("experiment") or "-")[:14], {"slurm": "slurm", "beaker": "bk", "lambda": "lam",
                      "local": "local"}.get(r.get("cluster"), r.get("cluster") or "-"), jid, st[:10], elapsed,
                      fmt_dur(exp_m * 60) if exp_m else "-", (r.get("node") or "-")[:12], " ".join(r["flags"])])
    widths = [max(len(h), *(len(l[i]) for l in lines)) if lines else len(h) for i, h in enumerate(head)]
    widths[-1] = 0
    fmt = lambda cols: "  ".join(c.ljust(w) if w else c for c, w in zip(cols, widths)).rstrip()  # noqa: E731
    return "\n".join([fmt(head)] + [fmt(l) for l in lines])


def snapshot(no_beaker=False, no_lambda=False, fresh=False, runs=None, now=None):
    now = now or time.time()
    runs = runs if runs is not None else load_registry()
    fns = {"slurm": lambda: collect_slurm(now)}
    if not no_beaker:
        fns["beaker"] = lambda: collect_beaker(runs, fresh=fresh, now=now)
    if not no_lambda:
        fns["lambda"] = lambda: collect_lambda(now)
    res = parallel(fns, 14)
    snap, sources = {}, {}
    for k in ("slurm", "beaker", "lambda"):
        if k not in fns:
            sources[k] = "skipped"
            continue
        v = res.get(k)
        if isinstance(v, Exception):
            sources[k] = f"unavailable ({str(v)[:90]})"
        else:
            snap[k] = v
            nl = sum(1 for j in v["live"] if j["state"] == "RUNNING")
            npd = len(v["live"]) - nl
            sources[k] = f"{nl} running, {npd} queued, {len(v['recent'])} ended/24h" + \
                (f" [{v['note']}]" if v.get("note") else "")
    return snap, sources, runs


def parse_since_s(spec):
    m = re.match(r"^(\d+(?:\.\d+)?)([mhdw])$", (spec or "7d").strip())
    if not m:
        return 7 * 86400
    return float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[m.group(2)]


def cmd_status(args):
    t0 = time.time()
    snap, sources, runs = snapshot(args.no_beaker, args.no_lambda, args.fresh)
    rows = build_rows(snap, runs, t0)
    sel = select_rows(rows, args.experiment, args.all, args.recent, parse_since_s(args.since), t0)
    if args.record:
        record_finals(rows)
    if args.json:
        print(json.dumps({"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                          "sources": sources, "rows": sel}, default=str, indent=1))
        return 0
    hidden = sum(1 for r in rows if not r.get("registry_only") and r not in sel and not r.get("live"))
    print(f"ops status  {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}  ({time.time() - t0:.1f}s)  "
          + " | ".join(f"{k}: {v}" for k, v in sources.items()))
    if sel:
        print(render_table(sel))
    else:
        print("(no live jobs)")
    if hidden and not args.recent and not args.experiment:
        print(f"+ {hidden} other jobs ended in the last 24h (unregistered, not failed) -- `--recent` to list")
    if not args.json:
        print("other users' GPUs per node + our cap use (jsteinhardt, Lambda): `ops capacity`")
    return 0


def record_finals(rows):
    """Write ``final_state`` update events for registered runs observed finished (``--record``)."""
    n = 0
    for r in rows:
        if r.get("live") or (r.get("registry_only") and not r.get("observed")) or not r.get("run_id") \
                or r.get("final_state"):
            continue
        fs = FINAL_MAP.get(r.get("state"))
        if not fs or (r.get("cluster") == "lambda" and fs == "failed"):
            continue
        el = r.get("elapsed_s")
        runtime = round(el / 60, 1) if fs == "done" and isinstance(el, (int, float)) and el > 0 else None
        try:
            registry_update(r["run_id"], final_state=fs, runtime_minutes=runtime,
                            state_source=f"ops status:{r.get('cluster')}:{r.get('state')}:{r.get('job_id')}")
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"[record] {r['run_id']}: registry.update unavailable ({e})", file=sys.stderr)
            break
    if n:
        print(f"[record] wrote final_state for {n} run(s)", file=sys.stderr)


# ================================================================================================
# stall diagnosis

NFS_WCHAN = re.compile(r"(nfs|rpc)_?wait|nfs4?_|rpc_|__rpc|wait_on_commit|nfs_", re.I)
FUTEX_WCHAN = re.compile(r"futex", re.I)
SLEEP_WCHAN = re.compile(r"nanosleep|hrtimer|clock_nanosleep|do_nanosleep", re.I)

PROBE = r"""
JOB="$1"; LOG="$2"; PYSPY="$3"; U=$(id -un)
own=$(grep -o 'step_[^/]*' /proc/$$/cgroup 2>/dev/null | head -1)
injob() { c=$(cat /proc/$1/cgroup 2>/dev/null); case "$c" in *"job_${JOB}/"*) case "$c" in *"/${own}/"*|*"/${own}") [ -n "$own" ] && return 1;; esac; return 0;; esac; return 1; }
ALL=$(ps -u "$U" -o pid= | tr -d ' ')
PIDS=""; for p in $ALL; do injob $p && PIDS="$PIDS $p"; done
if [ -z "$PIDS" ]; then echo "@@NOTE no processes found in job cgroup; showing all of $U's processes"; for p in $ALL; do [ "$p" != "$$" ] && PIDS="$PIDS $p"; done; fi
echo "@@HOST $(hostname -s)"
echo "@@NOW $(date +%s)"
echo "@@PS"
for p in $PIDS; do ps -o pid=,ppid=,stat=,wchan:32=,etimes=,rss=,args= -p $p 2>/dev/null | cut -c1-400; done
echo "@@STAT1"; for p in $PIDS; do cat /proc/$p/stat 2>/dev/null; done
sleep 2
echo "@@STAT2"; for p in $PIDS; do cat /proc/$p/stat 2>/dev/null; done
echo "@@GPU"; timeout 10 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -16
echo "@@DF"; df -P /data /tmp 2>/dev/null | tail -n +2
echo "@@LOG"; if [ -n "$LOG" ] && [ -e "$LOG" ]; then echo "STAT $(stat -c '%Y %s' "$LOG")"; tail -n 25 "$LOG" | cut -c1-300; else echo "MISSING $LOG"; fi
PY=$(for p in $PIDS; do a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); case "$a" in *python*|*torchrun*|*vllm*) echo $p;; esac; done | head -6)
for p in $PY; do
  echo "@@PROC $p"
  echo "exe $(readlink /proc/$p/exe 2>/dev/null)"
  for t in /proc/$p/task/*; do echo "thr $(cat $t/wchan 2>/dev/null) $(cut -d' ' -f3 $t/stat 2>/dev/null | head -c1)"; done | sort | uniq -c | sort -rn | head -8
  for fd in /proc/$p/fd/*; do readlink $fd 2>/dev/null; done | grep -E '^/(accounts|scratch|net)/' | sort | uniq -c | sort -rn | head -6 | sed 's/^/nfsfd /'
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -E '^(HOME|TMPDIR|TRITON_CACHE_DIR|FLASHINFER_CACHE_DIR|HF_HOME|XDG_CACHE_HOME|PYTHONUNBUFFERED)=' | sed 's/^/env /'
done
SPY=$(command -v py-spy 2>/dev/null || ls /data/prasann/conda/envs/*/bin/py-spy /data/prasann/*/bin/py-spy 2>/dev/null | head -1)
echo "@@PYSPY ${SPY:-none}"
if [ "$PYSPY" = 1 ] && [ -n "$SPY" ]; then T=$(echo $PY | awk '{print $1}'); [ -n "$T" ] && timeout 20 $SPY dump --pid $T 2>&1 | head -60; fi
echo "@@END"
"""


def parse_proc_stat(line):
    """``/proc/<pid>/stat`` line -> dict(pid, comm, state, ppid, ticks). None if unparsable."""
    try:
        l, r = line.index("("), line.rindex(")")
        pid = int(line[:l].strip())
        rest = line[r + 2:].split()
        return {"pid": pid, "comm": line[l + 1:r], "state": rest[0], "ppid": int(rest[1]),
                "ticks": int(rest[11]) + int(rest[12])}
    except (ValueError, IndexError):
        return None


def parse_probe(text, clk_tck=100, interval=2.0):
    """Parse PROBE output into an evidence dict (pure)."""
    ev = {"procs": [], "gpus": [], "df": [], "log_tail": [], "log_mtime": None, "log_missing": False,
          "now": None, "host": None, "notes": [], "pyspy": None, "pyspy_out": []}
    sec, s1, s2, procs, detail = None, {}, {}, {}, {}
    cur = None
    for line in (text or "").splitlines():
        if line.startswith("@@"):
            tag, _, rest = line[2:].partition(" ")
            sec = tag
            if tag == "HOST":
                ev["host"] = rest.strip()
            elif tag == "NOW":
                ev["now"] = float(rest) if rest.strip().isdigit() else None
            elif tag == "NOTE":
                ev["notes"].append(rest)
            elif tag == "PROC":
                cur = int(rest)
                detail[cur] = {"exe": None, "threads": {}, "nfs_fds": [], "env": {}}
            elif tag == "PYSPY":
                ev["pyspy"] = None if rest.strip() == "none" else rest.strip()
            continue
        if sec == "PS":
            f = line.split(None, 6)
            if len(f) >= 6 and f[0].isdigit():
                wchan = f[3] if f[3] != "-" else ""
                procs[int(f[0])] = {"pid": int(f[0]), "ppid": int(f[1]), "stat": f[2], "wchan": wchan,
                                    "etimes": int(f[4]) if f[4].isdigit() else None,
                                    "args": f[6] if len(f) > 6 else ""}
        elif sec in ("STAT1", "STAT2"):
            p = parse_proc_stat(line)
            if p:
                (s1 if sec == "STAT1" else s2)[p["pid"]] = p["ticks"]
        elif sec == "GPU":
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 3 and f[0].isdigit():
                try:
                    ev["gpus"].append({"index": int(f[0]), "util": float(f[1]), "mem_mib": float(f[2])})
                except ValueError:
                    pass
        elif sec == "DF":
            f = line.split()
            if len(f) >= 6:
                ev["df"].append({"mount": f[5], "use_pct": int(f[4].rstrip("%")) if f[4].rstrip("%").isdigit() else None})
        elif sec == "LOG":
            if line.startswith("STAT "):
                f = line.split()
                ev["log_mtime"] = float(f[1]) if len(f) > 1 else None
            elif line.startswith("MISSING"):
                ev["log_missing"] = True
            else:
                ev["log_tail"].append(line)
        elif sec == "PROC" and cur is not None:
            s = line.strip()
            if s.startswith("exe "):
                detail[cur]["exe"] = s[4:]
            elif s.startswith("nfsfd "):
                parts = s.split(None, 2)
                if len(parts) == 3:
                    detail[cur]["nfs_fds"].append(parts[2])
            elif s.startswith("env "):
                k, _, v = s[4:].partition("=")
                detail[cur]["env"][k] = v
            else:
                m = re.match(r"^(\d+)\s+thr\s*(\S*)\s*(\S?)$", s)
                if m:
                    key = (m.group(2) or "-") + ("/" + m.group(3) if m.group(3) else "")
                    detail[cur]["threads"][key] = detail[cur]["threads"].get(key, 0) + int(m.group(1))
        elif sec == "PYSPY":
            ev["pyspy_out"].append(line)
    for pid, p in procs.items():
        if pid in s1 and pid in s2:
            p["cpu"] = 100.0 * (s2[pid] - s1[pid]) / clk_tck / interval
        else:
            p["cpu"] = None
        p.update(detail.get(pid, {}))
        ev["procs"].append(p)
    if ev["log_mtime"] and ev["now"]:
        ev["log_age_s"] = max(0.0, ev["now"] - ev["log_mtime"])
    return ev


def _is_py(p):
    a = (p.get("args") or "") + " " + (p.get("exe") or "")
    return bool(re.search(r"python|torchrun|vllm", a))


def classify(ev):
    """Pure heuristic stall classifier. Returns ``(verdict, reasons, warnings)``.

    ``ev`` keys used: state, reason, procs[{pid,stat,wchan,cpu,args,exe,threads,nfs_fds,env}],
    gpus[{util}], log_age_s, log_missing, log_path, elapsed_s, df[{mount,use_pct}].
    """
    reasons, warns = [], []
    state = ev.get("state") or "RUNNING"
    if state in ("PENDING", "STARTING"):
        return f"PENDING ({ev.get('reason') or 'no reason given'})", reasons, warns
    if state != "RUNNING":
        return f"NOT RUNNING ({state})", reasons, warns
    procs = ev.get("procs") or []
    pys = [p for p in procs if _is_py(p)] or procs

    # static warnings (do not decide the verdict alone)
    for p in pys if not ev.get("login") else []:  # on the login node NFS interpreters are normal
        exe = p.get("exe") or ""
        if is_nfs_path(exe):
            warns.append(f"pid {p['pid']} interpreter on NFS: {exe} (use /data/prasann/... env)")
        env = p.get("env") or {}
        for k, v in env.items():
            if k in ("TRITON_CACHE_DIR", "FLASHINFER_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR") and is_nfs_path(v + "/"):
                warns.append(f"pid {p['pid']} {k}={v} is on NFS (JIT caches wedge there; point it at /data)")
        if is_nfs_path(env.get("HOME", "") + "/") and "FLASHINFER_CACHE_DIR" not in env and \
                re.search(r"vllm|olmo-eval|flashinfer", p.get("args") or ""):
            warns.append(f"pid {p['pid']} vLLM with HOME={env['HOME']} on NFS and no FLASHINFER_CACHE_DIR: "
                         "the JIT cache lands on NFS and concurrent evals wedge")
        if re.search(r"(?<![\w.-])/net/[a-z]", p.get("args") or ""):
            warns.append(f"pid {p['pid']} command line uses a /net/<node>/ path (NFS) for job I/O")
    if is_nfs_path(ev.get("log_path")):
        warns.append(f"log {ev.get('log_path')} is on NFS (multi-rank logs on NFS deadlock at step 0)")
    for d in ev.get("df") or []:
        if d.get("use_pct") is not None and d["use_pct"] >= 95:
            warns.append(f"{d['mount']} is {d['use_pct']}% full")
    warns = list(dict.fromkeys(warns))

    if not procs:
        return "NO PROCESSES FOUND (probe failed, or the job shell already exited)", reasons, warns

    def nfs_waiting(p):
        if p.get("stat", "").startswith("D") and NFS_WCHAN.search(p.get("wchan") or ""):
            return True
        return any(NFS_WCHAN.search(k) for k in (p.get("threads") or {}))

    cpus = [p["cpu"] for p in pys if p.get("cpu") is not None]
    max_cpu = max(cpus) if cpus else None
    age = ev.get("log_age_s")
    age_txt = (f"last output {int(age // 60)} min ago" if age is not None and age >= 60 else
               f"last output {int(age)} s ago" if age is not None else "log age unknown")
    nfs = [p for p in procs if nfs_waiting(p)]
    if nfs and (max_cpu is None or max_cpu < 15):
        for p in nfs[:4]:
            reasons.append(f"pid {p['pid']} state {p.get('stat')} wchan {p.get('wchan') or '-'} "
                           f"cpu {p.get('cpu') if p.get('cpu') is not None else '?'}% "
                           f"nfs-fds {', '.join(p.get('nfs_fds') or []) or '-'}: {(p.get('args') or '')[:100]}")
        reasons.append(age_txt)
        reasons.append("fix: node-local interpreter + logs/caches/data on /data (see CLAUDE.md 'Local cluster')")
        return "LIKELY NFS STALL", reasons, warns
    gpus = ev.get("gpus") or []
    elapsed = ev.get("elapsed_s") or 0
    silent_thr = ev.get("silent_threshold")  # ops guard: the child already crossed its silence threshold
    hung_after = silent_thr if silent_thr is not None else 300
    stale_after = silent_thr if silent_thr is not None else STALE_LOG_S
    if max_cpu is not None and max_cpu < 2 and (age is None or age >= hung_after):
        thr = {}
        for p in pys:
            for k, v in (p.get("threads") or {}).items():
                thr[k] = thr.get(k, 0) + v
        tot = sum(thr.values())
        fut = sum(v for k, v in thr.items() if FUTEX_WCHAN.search(k))
        slp = sum(v for k, v in thr.items() if SLEEP_WCHAN.search(k))
        reasons.append(f"all {len(pys)} python/main procs at <2% CPU over the sample; {age_txt}")
        if gpus:
            reasons.append(f"GPU util {', '.join(str(int(g['util'])) for g in gpus)}%")
        if tot and slp / tot > 0.5:
            reasons.append(f"{slp}/{tot} threads in a timer sleep (an explicit sleep/poll loop, not compute)")
            return "IDLE: sleeping, no output", reasons, warns
        if tot and fut / tot > 0.5:
            reasons.append(f"{fut}/{tot} threads in futex wait (collective/lock deadlock or waiting on a child)")
            return "HUNG: 0% CPU, threads blocked (likely deadlock)", reasons, warns
        return "HUNG: 0% CPU and no recent output", reasons, warns
    if gpus and all(g["util"] == 0 for g in gpus) and (elapsed > 600 or (age or 0) > hung_after):
        reasons.append(f"{len(gpus)} GPU(s) at 0% util; busiest python proc {max_cpu:.0f}% CPU" if max_cpu is not None
                       else f"{len(gpus)} GPU(s) at 0% util")
        reasons.append(age_txt + " (CPU-bound phase: JIT compile, dataloader, tokenization, or a livelock)")
        return "GPU IDLE", reasons, warns
    if age is not None and age >= stale_after:
        reasons.append(f"busiest python proc {max_cpu:.0f}% CPU" if max_cpu is not None else "cpu unknown")
        if gpus:
            reasons.append(f"GPU util {', '.join(str(int(g['util'])) for g in gpus)}%")
        reasons.append("busy but silent: livelock or missing progress logging (log at items 1,2,5,10)")
        return (f"NO OUTPUT SINCE {int(age // 60)} min" if age >= 60 else f"NO OUTPUT FOR {int(age)} s"), reasons, warns
    if ev.get("log_missing"):
        reasons.append(f"log file {ev.get('log_path')} does not exist on the node")
    reasons.append(age_txt)
    if max_cpu is not None:
        reasons.append(f"busiest python proc {max_cpu:.0f}% CPU")
    if gpus:
        reasons.append(f"GPU util {', '.join(str(int(g['util'])) for g in gpus)}%")
    return "PROGRESSING", reasons, warns


def _probe_cmd(job_id, log_path, pyspy, node=None):
    b64 = base64.b64encode(PROBE.encode()).decode()
    inner = f"echo {b64} | base64 -d | bash -s -- {job_id} '{log_path or ''}' {1 if pyspy else 0}"
    cmd = ["srun", f"--jobid={job_id}", "--overlap", "--ntasks=1", "--nodes=1"]
    if node:
        cmd.append(f"--nodelist={node.split(',')[0]}")
    return cmd + ["bash", "-c", inner]


def resolve_target(target, runs):
    """-> dict(cluster, job_id, run) for a run_id, slurm id, Beaker id, or slurm job name."""
    run_row = runs.get(target)
    if run_row and run_row.get("job_id"):
        return {"cluster": run_row.get("cluster") or "slurm", "job_id": str(run_row["job_id"]), "run": run_row}
    if re.match(r"^01[0-9A-Z]{24}$", target):
        return {"cluster": "beaker", "job_id": target, "run": run_row}
    if re.match(r"^\d+(_\d+)?$", target):
        return {"cluster": "slurm", "job_id": target, "run": run_row}
    rc, out, _ = run(["squeue", "-u", USER, "-h", "-n", target, "-o", "%i"], timeout=8)
    ids = out.split()
    if rc == 0 and ids:
        return {"cluster": "slurm", "job_id": ids[0], "run": run_row}
    return {"cluster": None, "job_id": None, "run": run_row}


def stall_slurm(job_id, run_row, pyspy=True, cluster="slurm"):
    """Diagnose one slurm (or Lambda) job. Returns an evidence dict with verdict fields."""
    ssh = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "lambda"] if cluster == "lambda" else []
    q = ssh + [f"squeue -j {job_id} -h -o '{LAMBDA_SQUEUE_FMT}'"] if ssh else \
        ["squeue", "-j", job_id, "-h", "-O", SQUEUE_FIELDS]
    rc, out, err = run(q, timeout=12)
    jobs = parse_squeue(out, cluster)
    ev = {"cluster": cluster, "job_id": job_id, "run_id": (run_row or {}).get("run_id")}
    if not jobs:
        ev["state"] = "NOT IN QUEUE"
        sacct = ["sacct", "-j", job_id, "-X", "-n", "-P", f"--format={SACCT_FIELDS}"]
        rc, out, _ = run(ssh + [" ".join(sacct)] if ssh else sacct, timeout=12)
        fin = parse_sacct(out, cluster)
        if fin:
            ev["state"] = fin[0]["state"]
            ev["elapsed_s"] = fin[0]["elapsed_s"]
            ev["node"] = fin[0]["node"]
        ev["verdict"], ev["reasons"], ev["warnings"] = classify(ev)
        return ev
    j = jobs[0]
    ev.update({k: j.get(k) for k in ("state", "reason", "elapsed_s", "node", "name", "timelimit_s")})
    ev["log_path"] = (run_row or {}).get("log_path") or j.get("log_path")
    if j["state"] != "RUNNING":
        ev["verdict"], ev["reasons"], ev["warnings"] = classify(ev)
        return ev
    pcmd = _probe_cmd(job_id, ev["log_path"], pyspy, j.get("node"))
    if ssh:
        pcmd = ssh + [" ".join(shlex.quote(x) for x in pcmd)]
    rc, out, err = run(pcmd, timeout=45)
    ev["probe_rc"] = rc
    if rc != 0 and "@@END" not in out:
        ev["probe_error"] = (err or "").strip()[-400:]
    pe = parse_probe(out)
    ev.update({k: v for k, v in pe.items() if k not in ("now",)})
    ev["verdict"], ev["reasons"], ev["warnings"] = classify(ev)
    return ev


def stall_beaker(target, run_row):
    ev = {"cluster": "beaker", "job_id": target, "run_id": (run_row or {}).get("run_id")}
    try:
        name, job = beaker_exp_get(target, timeout=20)
    except Exception as e:  # noqa: BLE001
        ev.update(state="UNKNOWN", verdict="UNAVAILABLE (beaker lookup failed)", reasons=[str(e)], warnings=[])
        return ev
    row = beaker_job_row(job, name)
    ev.update({k: row.get(k) for k in ("state", "elapsed_s", "pending_s", "name", "exp_id", "exit_code")})
    ev["beaker_job_id"] = job.get("id")
    if row["state"] != "RUNNING":
        verdict, reasons, warns = classify(dict(ev, reason=f"queued {fmt_dur(row.get('pending_s'))}"
                                                if row.get("pending_s") else None))
        if row["state"] in ("FAILED", "COMPLETED"):
            reasons.append(f"exitCode={row.get('exit_code')} (exit 0 != success for evals: grep logs for MISSING)")
        if row["state"] == "PREEMPTED":
            reasons.append("finalized with no exitCode = canceled/preempted")
        ev.update(verdict=verdict, reasons=reasons, warnings=warns)
        if row["state"] in ("RUNNING", "COMPLETED", "FAILED", "PREEMPTED") and job.get("id"):
            _beaker_log_tail(ev, job["id"])
        return ev
    _beaker_log_tail(ev, job["id"])
    age = ev.get("log_age_s")
    ev["warnings"] = []
    if age is None:
        ev["verdict"], ev["reasons"] = "UNKNOWN (no log lines fetched)", [ev.get("log_error") or ""]
    elif age > STALE_LOG_S:
        ev["verdict"], ev["reasons"] = f"NO OUTPUT SINCE {int(age // 60)} min", [
            "Beaker gives no process view; if this persists check the job's node/GPU in the Beaker UI"]
    else:
        ev["verdict"], ev["reasons"] = "PROGRESSING", [f"last log line {int(age)} s ago"]
    return ev


def _beaker_log_tail(ev, job_id):
    rc, out, err = run([beaker_bin(), "job", "logs", job_id, "--tail", "25"], timeout=40)
    if rc != 0:
        ev["log_error"] = (err or "").strip()[:200]
        return
    lines = [l for l in out.splitlines() if l.strip()]
    ev["log_tail"] = [re.sub(r"^\S+Z ", "", l)[:300] for l in lines]
    for l in reversed(lines):
        ts = parse_ts(l.split(" ", 1)[0])
        if ts:
            ev["log_age_s"] = max(0.0, time.time() - ts)
            break


def print_stall(ev):
    print(f"VERDICT: {ev.get('verdict')}")
    who = ev.get("run_id") or ev.get("name") or ""
    print(f"  job {ev.get('cluster')}:{ev.get('job_id')} {who}  state={ev.get('state')} "
          f"elapsed={fmt_dur(ev.get('elapsed_s'))} node={ev.get('node') or ev.get('host') or '-'}")
    for r in ev.get("reasons") or []:
        if r:
            print(f"  - {r}")
    for w in ev.get("warnings") or []:
        print(f"  ! {w}")
    if ev.get("probe_error"):
        print(f"  probe error (rc={ev.get('probe_rc')}): {ev['probe_error']}")
    for n in ev.get("notes") or []:
        print(f"  note: {n}")
    procs = ev.get("procs") or []
    if procs:
        print("  processes (cpu% = 2 s sample):")
        for p in sorted(procs, key=lambda p: -(p.get("cpu") or 0))[:12]:
            cpu = f"{p['cpu']:.0f}" if p.get("cpu") is not None else "?"
            print(f"    {p['pid']:>8} {p.get('stat', ''):<5} {(p.get('wchan') or '-')[:24]:<24} {cpu:>4}% "
                  f"{fmt_dur(p.get('etimes')):>7}  {(p.get('args') or '')[:110]}")
            if p.get("threads"):
                top = ", ".join(f"{k}x{v}" for k, v in sorted(p["threads"].items(), key=lambda kv: -kv[1])[:4])
                print(f"             threads: {top}")
            if p.get("exe"):
                print(f"             exe: {p['exe']}")
    if ev.get("gpus"):
        print("  gpus: " + "  ".join(f"#{g['index']} {g['util']:.0f}% {g['mem_mib'] / 1024:.0f}G" for g in ev["gpus"]))
    if ev.get("df"):
        print("  disk: " + "  ".join(f"{d['mount']} {d['use_pct']}%" for d in ev["df"]))
    if ev.get("log_path") or ev.get("log_tail"):
        age = ev.get("log_age_s")
        print(f"  log: {ev.get('log_path') or '(beaker)'}  "
              f"{'last write ' + fmt_dur(age) + ' ago' if age is not None else ('MISSING' if ev.get('log_missing') else '')}")
        for l in (ev.get("log_tail") or [])[-12:]:
            print(f"    | {l}")
    if "pyspy" in ev:
        if ev.get("pyspy_out"):
            print(f"  py-spy dump ({ev['pyspy']}):")
            for l in ev["pyspy_out"][:50]:
                print(f"    {l}")
        elif not ev.get("pyspy"):
            print("  py-spy: not installed on the node (pip install py-spy into the /data env for stack dumps)")


def cmd_stall(args):
    runs = load_registry()
    t = resolve_target(args.target, runs)
    if not t["cluster"]:
        print(f"VERDICT: UNKNOWN TARGET ({args.target}: not a run_id with a job_id, slurm id, Beaker id or job name)")
        return 2
    if t["cluster"] == "beaker":
        ev = stall_beaker(t["job_id"], t["run"])
    else:
        cl = "lambda" if t["cluster"] == "lambda" or args.lambda_ else "slurm"
        ev = stall_slurm(t["job_id"], t["run"], pyspy=not args.no_py_spy, cluster=cl)
    if args.json:
        print(json.dumps(ev, default=str, indent=1))
    else:
        print_stall(ev)
    return 0


# ================================================================================================
# CLI


def add_parser(sub):
    p = sub.add_parser("status", help="one snapshot of slurm/Beaker/Lambda jobs joined to the registry")
    p.add_argument("--experiment", "-e")
    p.add_argument("--all", action="store_true", help="include registry runs with no live job (see --since)")
    p.add_argument("--since", default="7d", help="window for registry-only rows with --all/--experiment (e.g. 24h, 7d)")
    p.add_argument("--recent", action="store_true", help="list every job that ended in the last 24 h")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-beaker", action="store_true")
    p.add_argument("--no-lambda", action="store_true")
    p.add_argument("--fresh", action="store_true", help="bypass the 60 s Beaker cache")
    p.add_argument("--record", action="store_true", help="write final_state for registered runs seen finished")
    p.set_defaults(func=cmd_status)

    s = sub.add_parser("stall", help="deep diagnosis of one job (verdict first, then evidence)")
    s.add_argument("target", help="run_id, slurm job id, Beaker experiment/job id, or slurm job name")
    s.add_argument("--lambda", dest="lambda_", action="store_true", help="numeric id is a Lambda job")
    s.add_argument("--no-py-spy", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stall)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    add_parser(ap.add_subparsers(dest="cmd"))
    a = ap.parse_args()
    sys.exit(a.func(a))
