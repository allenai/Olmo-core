"""Never-raising entry points for repo launchers and eval drivers (stdlib only).

A launcher puts ``<repo>/ops`` on ``sys.path`` and imports this module (wrap the import in
``try/except`` too)::

    sys.path.insert(0, f"{REPO}/ops")
    from opslib import safe as ops_safe

    ops_safe.register_launch(git_dir=REPO, run_id=..., experiment=..., cluster="beaker", job_id=...)

Every function catches every exception, prints one ``[ops] WARNING`` line to stderr and returns
``None``: a registry problem must never fail or block a launch. Rows go to the **canonical**
registry, i.e. the checkout the stable ``~prasann/bin/ops`` symlink points at, not to whichever
worktree the launcher happens to run from (``OPS_STATE_DIR`` overrides, e.g. in tests).
"""
from __future__ import annotations

import os
import re
import socket
import sys
from datetime import datetime
from pathlib import Path

from opslib import paths, registry, results, runtime

STABLE_OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"


def _warn(what: str, err: BaseException) -> None:
    print(f"[ops] WARNING: {what} failed ({type(err).__name__}: {err}); continuing", file=sys.stderr, flush=True)


def state_dir() -> Path:
    """``$OPS_STATE_DIR``, else the shared live state dir (:data:`paths.DEFAULT_STATE_DIR`), else the
    state dir of the checkout behind :data:`STABLE_OPS`, else this checkout's."""
    env = os.environ.get("OPS_STATE_DIR")
    if env:
        return Path(env)
    if paths.DEFAULT_STATE_DIR.is_dir():
        return paths.DEFAULT_STATE_DIR
    try:
        real = Path(os.path.realpath(STABLE_OPS))
        if real.is_file():
            return real.parent.parent / "state"
    except OSError:
        pass
    return Path(paths.STATE_DIR)


def derive_experiment(run_id: str | None = None, wandb_group: str | None = None, default: str | None = None):
    """Experiment key: ``$OPS_EXPERIMENT`` > known run-name prefix > slug of the wandb group > ``default``."""
    if os.environ.get("OPS_EXPERIMENT"):
        return os.environ["OPS_EXPERIMENT"]
    guessed = registry.guess_experiment(run_id)
    if guessed:
        return guessed
    if wandb_group:
        return re.sub(r"[^a-z0-9]+", "_", wandb_group.lower()).strip("_") or default
    return default


def eval_job_id() -> str | None:
    """The id of the job this process runs in (Beaker workload/experiment/job, else slurm)."""
    for var in ("BEAKER_WORKLOAD_ID", "BEAKER_EXPERIMENT_ID", "BEAKER_JOB_ID", "SLURM_JOB_ID"):
        if os.environ.get(var):
            return os.environ[var]
    return None


def make_eval_run_id(train_run: str, suite: str, variant: str = "default", when: datetime | None = None) -> str:
    """Eval-pass run id for a launcher without a natural one: ``<train>__ev-<suite>-<variant>-<YYYYmmddHHMM>``."""
    slug = lambda s: re.sub(r"[^A-Za-z0-9.+]+", "-", str(s)).strip("-") or "x"  # noqa: E731
    return f"{train_run}__ev-{slug(suite)}-{slug(variant)}-{(when or datetime.now()).strftime('%Y%m%d%H%M')}"


def unique_run_id(base: str) -> str:
    """``base`` if no run has that id yet, else ``base-<YYYYmmddHHMM>`` (each eval pass = its own run)."""
    try:
        if base not in registry.load_runs(state_dir() / "runs.jsonl"):
            return base
    except Exception as e:  # noqa: BLE001
        _warn("registry read", e)
        return base
    return f"{base}-{datetime.now().strftime('%Y%m%d%H%M')}"


def register_eval_pass(run_id: str, train_run: str, **fields) -> dict | None:
    """Register an eval pass (``kind=eval``, ``parent_run=train_run``) unless ``run_id`` already is.

    ``experiment`` defaults to the training run's. Returns the (existing or new) row; never raises.
    """
    try:
        runs = registry.load_runs(state_dir() / "runs.jsonl")
        if run_id in runs:
            return runs[run_id]
        if not fields.get("experiment"):
            fields["experiment"] = (runs.get(train_run) or {}).get("experiment") \
                or derive_experiment(train_run, default="unassigned")
        if not fields.get("cluster"):
            fields["cluster"] = "beaker" if os.environ.get("BEAKER_JOB_ID") else (
                "slurm" if os.environ.get("SLURM_JOB_ID") else "local")
        fields.setdefault("job_id", eval_job_id())
        fields.setdefault("node", socket.gethostname().split(".")[0])
        return register_launch(run_id=run_id, kind="eval", parent_run=train_run, **fields)
    except Exception as e:  # noqa: BLE001
        _warn(f"registering eval pass {run_id}", e)
        return None


def register_launch(git_dir=None, **fields) -> dict | None:
    """Append a launch row (see ``registry.FIELDS``); ``git_dir`` fills any missing git fields.

    Returns the row, or ``None`` (after a one-line warning) on any failure.
    """
    try:
        if git_dir:
            for key, value in registry.git_info(git_dir).items():
                if fields.get(key) is None:
                    fields[key] = value
        fields.setdefault("source", "launch")
        path = state_dir() / "runs.jsonl"
        row = registry.register(_path=path, **fields)
        print(f"[ops] registered run_id={row['run_id']} experiment={row['experiment']} "
              f"job_id={row.get('job_id')} -> {path}", flush=True)
        return row
    except Exception as e:  # noqa: BLE001 - never fail a launch
        _warn("registry launch row", e)
        return None


def update_run(run_id: str, **fields) -> dict | None:
    """Append an update row (e.g. ``final_state='done'``) for a registered run; ``None`` on failure."""
    try:
        fields.setdefault("source", "launch")
        row = registry.update(run_id, _path=state_dir() / "runs.jsonl", **fields)
        print(f"[ops] updated run_id={run_id}: " + ", ".join(f"{k}={v}" for k, v in fields.items()), flush=True)
        return row
    except Exception as e:  # noqa: BLE001
        _warn(f"registry update of {run_id}", e)
        return None


def record_results(json_path, check: bool = True, **defaults) -> list | None:
    """``ops result add --json json_path [--check]`` as a library call; returns the stored rows.

    Rows that fail validation are reported and skipped; warnings from :func:`results.check` are
    printed, never fatal. Returns ``None`` if the file cannot be read at all.
    """
    try:
        path = state_dir() / "results.jsonl"
        runs = registry.load_runs(state_dir() / "runs.jsonl")
        prior = results.load(path, runs=runs) if check else []
        stored = []
        for fields in results.load_json_file(json_path, defaults=defaults):
            try:
                row = results.normalize(fields, runs=runs)
            except ValueError as e:
                print(f"[ops] result REJECTED {fields.get('task')}/{fields.get('rung')}: {e}", file=sys.stderr, flush=True)
                continue
            if check:
                for w in results.check(row, prior=prior + stored):
                    print(f"[ops] result WARN {row['task']} {row.get('rung')}: {w}", file=sys.stderr, flush=True)
            stored.append(row)
        registry.append_rows(path, stored)
        print(f"[ops] recorded {len(stored)} result row(s) from {json_path} -> {path}", flush=True)
        return stored
    except Exception as e:  # noqa: BLE001
        _warn(f"recording results from {json_path}", e)
        return None


def suggest_min_runtime(run_id: str | None = None, expected_minutes=None, launcher: str | None = None,
                        kind: str = "train") -> dict:
    """Beaker minRuntime for a launch: min(4 h, 1.3 x expected), expected from the caller or the
    registry's history median (``opslib.runtime.suggest``); ``{"min_runtime": "1h", ...}`` on any error."""
    try:
        runs = registry.load_runs(state_dir() / "runs.jsonl") if expected_minutes in (None, "") and run_id else None
        return runtime.suggest(run_id=run_id, expected_minutes=expected_minutes, runs=runs, launcher=launcher, kind=kind)
    except Exception as e:  # noqa: BLE001 - never fail a launch
        _warn("minRuntime suggestion", e)
        return {"min_runtime": "1h", "min_runtime_minutes": 60, "expected_minutes": None,
                "source": "fallback (ops error)", "resume": False}
