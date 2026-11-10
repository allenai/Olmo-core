"""Run registry: append-only ``state/runs.jsonl`` of launch/update events, folded by ``run_id``.

Python API: :func:`register`, :func:`update`, :func:`load_runs`, :func:`find`.
CLI: ``ops register``, ``ops update RUN_ID k=v ...``, ``ops runs``.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from opslib import paths

KINDS = ("train", "eval", "data", "other")
# Contract enum + ``vessl`` (a handful of 2026-08 ctc_modelscale runs ran there).
CLUSTERS = ("beaker", "slurm", "lambda", "local", "vessl")
FINAL_STATES = ("done", "failed", "cancelled")
FIELDS = (
    "event", "run_id", "experiment", "kind", "parent_run", "arm", "task", "cluster", "job_id",
    "node", "ckpt_path", "log_path", "wandb_url", "git_commit", "git_branch", "dirty", "command",
    "expected_minutes", "final_state", "state_source", "launched_at", "ts", "launched_by",
    "source", "notes", "runtime_minutes",
)
# runtime_minutes: actual run time of the finished job (written by `ops status --record`); its
# history median is the default expected runtime, hence minRuntime (opslib/runtime.py).
NUMERIC_FIELDS = {"expected_minutes", "runtime_minutes"}
BOOL_FIELDS = {"dirty"}
LOCAL_TZ = "America/Los_Angeles"  # naive ledger timestamps were written on the Berkeley login node

# run-name prefix -> experiment, used when a run_id arrives without an experiment (results backfill).
EXPERIMENT_PREFIXES = (
    ("ds64-", "ds64"), ("f2k-", "ds64_fast2k"), ("fast2k-", "ds64_fast2k"), ("f8k-", "ds64_fast8k"),
    ("fs35", "flop_scaling"), ("fs-", "flop_scaling"), ("prior-dense-", "flop_scaling"),
    ("sdcpt-", "softdetach"), ("fdcpt-", "ffndrop"), ("q4b-ffnmoe", "ffnmoe"),
    ("ctcms-", "ctc_modelscale"), ("ctcfam-", "ctc_modelscale"), ("ctc-olmohyb", "ctc_olmo_hybrid"),
    ("ctc-olmo3", "ctc_olmo_hybrid"), ("ctc-llama8b", "ctc_llama8b"), ("xfm-", "xformmatch_querypos"),
    ("rml-", "xformmatch_querypos"), ("flexcompute-", "flexcompute"),
)


def guess_experiment(run_id: str | None) -> str | None:
    """Best-effort experiment key from a run-name prefix (``None`` if unknown)."""
    for prefix, exp in EXPERIMENT_PREFIXES:
        if run_id and run_id.startswith(prefix):
            return exp
    return None


def runs_path(path=None) -> Path:
    """Resolve runs.jsonl at call time so ``OPS_STATE_DIR`` set after import (tests) is honoured."""
    if path:
        return Path(path)
    return Path(os.environ.get("OPS_STATE_DIR", paths.STATE_DIR)) / "runs.jsonl"


def now_iso() -> str:
    """Current time as ISO-8601 UTC with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_iso_utc(value) -> str | None:
    """Normalise a ledger timestamp (many formats; naive = Pacific) to ISO-8601 UTC, else ``None``."""
    if not value or not isinstance(value, str):
        return None
    s = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        m = re.match(r"\d{4}-\d{2}-\d{2}([ T]\d{1,2}:\d{2}(:\d{2})?)?", s)
        if not m:
            return None
        try:
            dt = datetime.fromisoformat(m.group(0))
        except ValueError:
            return None
    if dt.tzinfo is None:
        try:
            from zoneinfo import ZoneInfo

            dt = dt.replace(tzinfo=ZoneInfo(LOCAL_TZ))
        except Exception:  # pragma: no cover - zoneinfo/tzdata missing
            dt = dt.replace(tzinfo=timezone(timedelta(hours=-7)))
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_since(spec: str) -> datetime:
    """``7d`` / ``12h`` / ``30m`` / an ISO date -> an aware UTC datetime lower bound."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([dhm])", spec.strip())
    if m:
        unit = {"d": "days", "h": "hours", "m": "minutes"}[m.group(2)]
        return datetime.now(timezone.utc) - timedelta(**{unit: float(m.group(1))})
    iso = to_iso_utc(spec)
    if not iso:
        raise ValueError(f"cannot parse --since {spec!r} (use 7d, 12h or an ISO date)")
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


@contextmanager
def locked(path):
    """Exclusive ``fcntl.flock`` on ``<path>.lock`` (creates the parent directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def append_rows(path, rows) -> int:
    """Append ``rows`` (dicts) to a JSONL file under the lock, as one ``O_APPEND`` write."""
    rows = list(rows)
    if not rows:
        return 0
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    with locked(path):
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    return len(rows)


def read_jsonl(path) -> list:
    """All parseable JSON objects in a JSONL file (missing file -> ``[]``; bad lines warned)."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"ops: skipping malformed line {i} of {path}", file=sys.stderr)
    return out


def fold(rows) -> dict:
    """Fold event rows by ``run_id`` in file order; later non-null fields win."""
    runs = {}
    for row in rows:
        rid = row.get("run_id")
        if not rid:
            continue
        cur = runs.setdefault(rid, {"run_id": rid, "n_events": 0})
        cur["n_events"] += 1
        for k, v in row.items():
            if v is not None and k != "event":
                cur[k] = v
    return runs


def _validate(row: dict, launch: bool) -> None:
    """Raise ``ValueError`` on a malformed event row."""
    if not row.get("run_id"):
        raise ValueError("run_id is required")
    if launch:
        missing = [k for k in ("experiment", "cluster") if not row.get(k)]
        if missing:
            raise ValueError(f"launch event missing required fields: {missing}")
    unknown = sorted(set(row) - set(FIELDS))
    if unknown:
        raise ValueError(f"unknown fields {unknown}; allowed: {', '.join(FIELDS)}")
    for key, allowed in (("kind", KINDS), ("cluster", CLUSTERS), ("final_state", FINAL_STATES)):
        if row.get(key) is not None and row[key] not in allowed:
            raise ValueError(f"{key}={row[key]!r} not in {allowed}")
    for key in sorted(NUMERIC_FIELDS):
        if row.get(key) is not None and not isinstance(row[key], (int, float)):
            raise ValueError(f"{key} must be a number")


def default_launched_by() -> str:
    """Claude session id when running under Claude Code, else ``manual``."""
    return os.environ.get("CLAUDE_SESSION_ID") or "manual"


def make_launch(**fields) -> dict:
    """Build (and validate) a launch event row with defaults filled in; does not write."""
    row = {k: v for k, v in fields.items() if v is not None}
    row["event"] = "launch"
    row.setdefault("kind", "train")
    row.setdefault("ts", now_iso())
    row.setdefault("launched_at", row["ts"])
    row.setdefault("launched_by", default_launched_by())
    row.setdefault("source", "cli")
    _validate(row, launch=True)
    return row


def make_update(run_id: str, **fields) -> dict:
    """Build (and validate) an update event row; does not write."""
    row = {k: v for k, v in fields.items() if v is not None}
    row.update(event="update", run_id=run_id)
    row.setdefault("ts", now_iso())
    row.setdefault("source", "cli")
    _validate(row, launch=False)
    return row


def register(_path=None, **fields) -> dict:
    """Append a launch event. Requires ``run_id``, ``experiment``, ``cluster``. Returns the row."""
    row = make_launch(**fields)
    append_rows(runs_path(_path), [row])
    return row


def update(run_id: str, _path=None, _allow_unknown=False, **fields) -> dict:
    """Append an update event for an already-registered ``run_id``. Returns the row."""
    path = runs_path(_path)
    if not _allow_unknown and run_id not in load_runs(path):
        raise KeyError(f"run_id {run_id!r} is not registered (register it first)")
    row = make_update(run_id, **fields)
    append_rows(path, [row])
    return row


def load_runs(path=None) -> dict:
    """``{run_id: folded_row}`` for every run in runs.jsonl."""
    return fold(read_jsonl(runs_path(path)))


def find(experiment=None, since=None, cluster=None, path=None) -> list:
    """Folded runs filtered by experiment / cluster / launched_at >= ``since``; oldest first."""
    if isinstance(since, str):
        since = parse_since(since)
    out = []
    for run in load_runs(path).values():
        if experiment and run.get("experiment") != experiment:
            continue
        if cluster and run.get("cluster") != cluster:
            continue
        if since:
            ts = run.get("launched_at")  # backfilled runs without a launch time are excluded
            if not ts or datetime.fromisoformat(ts.replace("Z", "+00:00")) < since:
                continue
        out.append(run)
    return sorted(out, key=lambda r: r.get("launched_at") or "")


def git_info(repo_dir) -> dict:
    """``{git_commit, git_branch, dirty}`` for ``repo_dir`` via read-only git calls (5 s timeout)."""

    def git(*args):
        try:
            res = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(repo_dir), *args],
                capture_output=True, text=True, timeout=5,
            )
            return res.stdout.strip() if res.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def coerce(key: str, value: str):
    """Turn a CLI ``key=value`` string into the field's type (``null`` -> None)."""
    if value.lower() in ("null", "none", ""):
        return None
    if key in NUMERIC_FIELDS:
        return float(value) if "." in value else int(value)
    if key in BOOL_FIELDS:
        return value.lower() in ("1", "true", "yes", "y")
    return value


def format_table(rows, cols) -> str:
    """Plain fixed-width table (``-`` for missing values)."""
    cells = [[("-" if r.get(c) is None else str(r.get(c))) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip() for row in cells]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- CLI

RUNS_COLS = ("run_id", "experiment", "kind", "cluster", "job_id", "launched_at", "final_state")


def _cmd_register(args):
    fields = {k: getattr(args, k) for k in FIELDS if getattr(args, k, None) is not None}
    if args.git:
        fields.update({k: v for k, v in git_info(args.git).items() if v is not None})
    row = register(**fields)
    print(json.dumps(row, sort_keys=True) if args.json else f"registered {row['run_id']}")


def _cmd_update(args):
    fields = {}
    for kv in args.pairs:
        if "=" not in kv:
            raise SystemExit(f"ops update: expected key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        fields[k] = coerce(k, v)
    try:
        row = update(args.run_id, _allow_unknown=args.force, **fields)
    except (KeyError, ValueError) as e:
        raise SystemExit(f"ops update: {e}")
    print(json.dumps(row, sort_keys=True) if args.json else f"updated {args.run_id}")


def _cmd_runs(args):
    rows = find(experiment=args.experiment, since=args.since, cluster=args.cluster)
    if args.json:
        print(json.dumps(rows, indent=1, sort_keys=True))
    else:
        print(format_table(rows, RUNS_COLS))
        print(f"({len(rows)} runs)")


def add_parser(sub):
    """Register ``register``, ``update`` and ``runs`` subcommands."""
    p = sub.add_parser("register", help="record a launch in state/runs.jsonl")
    for f in FIELDS:
        if f in ("event", "ts", "dirty"):
            continue
        p.add_argument("--" + f.replace("_", "-"), dest=f,
                       type=float if f in NUMERIC_FIELDS else str,
                       required=f in ("run_id", "experiment", "cluster"))
    p.add_argument("--dirty", action="store_true", default=None)
    p.add_argument("--git", metavar="REPO_DIR", help="fill git_commit/git_branch/dirty from this repo")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_register)

    p = sub.add_parser("update", help="append an update event (e.g. final_state=done)")
    p.add_argument("run_id")
    p.add_argument("pairs", nargs="+", metavar="key=value")
    p.add_argument("--force", action="store_true", help="allow an unregistered run_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_update)

    p = sub.add_parser("runs", help="list registered runs")
    p.add_argument("--experiment")
    p.add_argument("--cluster")
    p.add_argument("--since", help="e.g. 7d, 12h, 2026-09-01")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_runs)
