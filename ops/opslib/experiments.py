"""Experiment status pages: ``ops/experiments/<name>/STATUS.md`` + a generated ``INDEX.md``.

Each STATUS.md starts with a tiny front-matter block (stdlib-parsed, deliberately simple)::

    ---
    experiment: ds64
    title: Soft-KV ds64 scaling campaign
    status: active            # active | paused | done | abandoned
    updated: 2026-10-12       # YYYY-MM-DD, bump on every edit
    branch: prasann/landmark
    launcher: debug/ds64/launch.sh
    records:
      - records/ds64-handoff.md
    results:
      - results/ds64/summary.json
    artifacts:
      - https://claude.ai/artifact/...   # the experiment's summary page (one per experiment)
    wandb_group: https://wandb.ai/...
    next_step: one line saying what happens next
    ---

Only ``key: value`` lines and ``- item`` list lines are understood; ``#`` starts a comment only
when preceded by whitespace (so URLs with fragments survive). The body is free markdown with the
sections listed in :data:`BODY_SECTIONS`.

CLI: ``ops exp new NAME | index | show NAME | stale [--days 7] | check | records``.
"""
import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

from . import paths

STATUSES = ("active", "paused", "done", "abandoned")
LIST_FIELDS = ("records", "results", "artifacts")
FIELDS = (
    "experiment",
    "title",
    "status",
    "updated",
    "branch",
    "launcher",
    "records",
    "results",
    "artifacts",
    "wandb_group",
    "next_step",
)
BODY_SECTIONS = (
    "Question",
    "Decisions",
    "Current best numbers",
    "Next step",
    "Open issues / traps",
    "Verified vs unverified",
)
_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def experiments_dir():
    """Directory holding ``<name>/STATUS.md`` (env ``OPS_EXPERIMENTS_DIR`` overrides)."""
    return Path(os.environ.get("OPS_EXPERIMENTS_DIR", paths.EXPERIMENTS_DIR))


# --------------------------------------------------------------------------- parsing


def _strip_comment(value):
    # Only " # " (whitespace on both sides) starts a comment, so "PR #400" survives.
    m = re.search(r"\s#(\s|$)", value)
    if m:
        value = value[: m.start()]
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def parse_front_matter(text):
    """Split STATUS.md text into ``(meta, body)``.

    :param text: Full file contents.

    :returns: ``(dict, str)``. List fields always come back as lists; a file without a front
        matter block returns ``({}, text)``.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta = {}
    current_list = None
    end = None
    for i, raw in enumerate(lines[1:], start=1):
        line = raw.rstrip()
        if line.strip() == "---":
            end = i
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- ") or stripped == "-":
            if current_list is None:
                continue  # stray item; ignore rather than crash
            item = _strip_comment(stripped[1:].strip())
            if item:
                meta[current_list].append(item)
            continue
        m = _KEY_RE.match(stripped)
        if not m:
            continue
        key, value = m.group(1), _strip_comment(m.group(2))
        if key in LIST_FIELDS or value == "":
            meta[key] = [value] if value else []
            current_list = key
        else:
            meta[key] = value
            current_list = None
    if end is None:
        return {}, text
    for key in LIST_FIELDS:
        v = meta.get(key)
        if v is None:
            meta[key] = []
        elif isinstance(v, str):
            meta[key] = [v]
    # Scalars that happened to be written as "key:" with nothing after come back as "".
    for key in FIELDS:
        if key not in LIST_FIELDS and isinstance(meta.get(key), list):
            meta[key] = meta[key][0] if meta[key] else ""
    body = "\n".join(lines[end + 1 :])
    return meta, body


def render_front_matter(meta):
    """Inverse of :func:`parse_front_matter` for the known fields (used by ``ops exp new``)."""
    out = ["---"]
    for key in FIELDS:
        val = meta.get(key, [] if key in LIST_FIELDS else "")
        if key in LIST_FIELDS:
            out.append(f"{key}:")
            out.extend(f"  - {item}" for item in val)
        else:
            out.append(f"{key}: {val}".rstrip())
    out.append("---")
    return "\n".join(out)


def validate(meta, name=None, today=None):
    """Return a list of human-readable problems with one STATUS.md's front matter."""
    problems = []
    for key in ("experiment", "title", "status", "updated", "next_step"):
        if not meta.get(key):
            problems.append(f"missing {key}")
    status = meta.get("status")
    if status and status not in STATUSES:
        problems.append(f"status {status!r} not in {'|'.join(STATUSES)}")
    updated = meta.get("updated")
    if updated:
        d = parse_date(updated)
        if d is None:
            problems.append(f"updated {updated!r} is not YYYY-MM-DD")
        elif today is not None and d > today:
            problems.append(f"updated {updated} is in the future")
    if name and meta.get("experiment") and meta["experiment"] != name:
        problems.append(f"experiment {meta['experiment']!r} != directory name {name!r}")
    return problems


def parse_date(s):
    if not s or not _DATE_RE.match(str(s)):
        return None
    try:
        return _dt.date.fromisoformat(str(s))
    except ValueError:
        return None


# --------------------------------------------------------------------------- loading


def load_one(path, today=None):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)
    name = path.parent.name
    return {
        "name": name,
        "path": str(path),
        "meta": meta,
        "body": body,
        "problems": validate(meta, name=name, today=today),
    }


def load_all(root=None, today=None):
    """Load every ``<root>/<name>/STATUS.md`` (sorted by name)."""
    root = Path(root) if root else experiments_dir()
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        f = d / "STATUS.md"
        if f.is_file():
            out.append(load_one(f, today=today))
    return out


def stale(exps, days=7, today=None):
    """Active experiments whose ``updated`` is older than ``days`` (or missing/unparseable)."""
    today = today or _dt.date.today()
    out = []
    for e in exps:
        if e["meta"].get("status") != "active":
            continue
        d = parse_date(e["meta"].get("updated"))
        age = None if d is None else (today - d).days
        if age is None or age > days:
            out.append(dict(e, age_days=age))
    return out


# --------------------------------------------------------------------------- rendering


def _cell(s):
    return str(s or "").replace("|", "\\|").replace("\n", " ").strip()


def render_index(exps, today=None):
    """Markdown index: one table per status (active, paused, done, abandoned, other)."""
    today = today or _dt.date.today()
    lines = [
        "# Experiments index",
        "",
        f"_Generated by `ops exp index` on {today.isoformat()} — do not edit by hand; edit each "
        "`<name>/STATUS.md` and re-run._",
        "",
        "One STATUS.md per experiment is the canonical current state (memory files and records/ "
        "are history). Columns: updated date, status page, title, next step.",
        "",
    ]
    groups = {s: [] for s in STATUSES}
    other = []
    for e in exps:
        groups.get(e["meta"].get("status"), other).append(e)
    order = [(s, groups[s]) for s in STATUSES] + [("unknown status", other)]
    for label, items in order:
        if not items:
            continue
        items = sorted(items, key=lambda e: (e["meta"].get("updated") or "", e["name"]), reverse=True)
        lines += [f"## {label.capitalize()} ({len(items)})", ""]
        lines += ["| updated | experiment | title | next step |", "|---|---|---|---|"]
        for e in items:
            m = e["meta"]
            flag = " ⚠" if e["problems"] else ""
            lines.append(
                f"| {_cell(m.get('updated'))} | [{e['name']}]({e['name']}/STATUS.md){flag} | "
                f"{_cell(m.get('title'))} | {_cell(m.get('next_step'))} |"
            )
        lines.append("")
    problems = [e for e in exps if e["problems"]]
    if problems:
        lines += ["## Front-matter problems", ""]
        for e in problems:
            lines.append(f"- `{e['name']}`: {'; '.join(e['problems'])}")
        lines.append("")
    return "\n".join(lines)


def template(name, today=None):
    """Return a fresh STATUS.md for experiment ``name``."""
    today = today or _dt.date.today()
    meta = {
        "experiment": name,
        "title": "TODO one-line title",
        "status": "active",
        "updated": today.isoformat(),
        "branch": "",
        "launcher": "",
        "records": [],
        "results": [],
        "artifacts": [],
        "wandb_group": "",
        "next_step": "TODO",
    }
    body = f"""
# {name}

## Question
TODO: what this experiment is trying to find out, in one or two sentences.

## Decisions
- {today.isoformat()} (who): decision and why.

## Current best numbers
<!-- One line per number: task, rung, metric=value, eval_size, ±SE, cot label, source path.
     Flag eval_size<500 inline. Prefer `ops result show --experiment {name} --best`. -->
| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|

## Next step
TODO (keep in sync with `next_step` above).

## Open issues / traps
-

## Verified vs unverified
- Verified (how, when):
- UNVERIFIED — confirm with Prasann:
"""
    return render_front_matter(meta) + "\n" + body



# --------------------------------------------------------------------------- records/ index

RECORD_TEMPLATE = """---
title: <one line>
date: YYYY-MM-DD            # date written; bump `updated` on substantive edits
updated: YYYY-MM-DD
experiment: <ops/experiments key, or blank>
area: pipeline | token_routing | ffn_compute | kv_route | landmark
kind: diagnosis | plan | handoff | howto | result | task-brief | audit
status: current | superseded | load-bearing
superseded_by: <records/x.md or ops/experiments/x/STATUS.md, or blank>
artifact: <the experiment's claude.ai artifact URL, or blank>
summary: <one or two sentences: the finding / what the doc is for — this is the README row>
---
# <title>
"""

_ROW_NAME_RE = re.compile(r"[`(\[]([A-Za-z0-9_.-]+\.md)[`)\]]")


def _existing_descriptions(readme_text):
    """Map record filename -> description from a hand-written records/README.md."""
    out = {}
    for line in readme_text.splitlines():
        m = _ROW_NAME_RE.search(line)
        if not m or m.group(1) == "README.md":
            continue
        name = m.group(1)
        rest = line[m.end():]
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            desc = cells[-1] if len(cells) > 1 else ""
        else:
            rest = re.sub(r"^\([^)]*\)", "", rest)  # the (target) half of [name](target)
            desc = re.sub(r"^[^A-Za-z*_`]*(?:—|-|:)\s*", "", rest).strip()
        if desc and name not in out:
            out[name] = desc
    return out


def _first_paragraph(body):
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "<!--", "|", "---", "```")):
            continue
        return s
    return ""


def _git_last_dates(records_dir):
    """filename -> last commit date (YYYY-MM-DD) touching it; {} if git is unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(records_dir), "log", "--format=@%ad",
             "--date=short", "--name-only", "--", "."],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    dates, cur = {}, None
    for line in out.splitlines():
        if line.startswith("@"):
            cur = line[1:].strip()
        elif line.strip() and cur:
            dates.setdefault(Path(line.strip()).name, cur)
    return dates


def record_rows(records_dir, use_git=True):
    """One dict per ``records/*.md`` (README excluded) with name/title/date/kind/status/summary.

    ``date`` comes from the header (``updated``/``date``), else the last git commit touching the file.
    """
    records_dir = Path(records_dir)
    git_dates = _git_last_dates(records_dir) if use_git else {}
    readme = records_dir / "README.md"
    prior = _existing_descriptions(readme.read_text(encoding="utf-8")) if readme.is_file() else {}
    rows = []
    for f in sorted(records_dir.glob("*.md")):
        if f.name == "README.md":
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_front_matter(text)
        title = meta.get("title") or next(
            (l[2:].strip() for l in body.splitlines() if l.startswith("# ")), f.stem)
        summary = meta.get("summary") or prior.get(f.name) or _first_paragraph(body)
        rows.append({
            "name": f.name,
            "title": _scalar(title),
            "date": _scalar(meta.get("updated") or meta.get("date") or git_dates.get(f.name, "")),
            "kind": _scalar(meta.get("kind")),
            "status": _scalar(meta.get("status")),
            "experiment": _scalar(meta.get("experiment")),
            "area": _scalar(meta.get("area")),
            "superseded_by": _scalar(meta.get("superseded_by")),
            "summary": _scalar(summary),
            "has_header": bool(meta),
        })
    return rows


def _scalar(v):
    """Header values written as ``key:`` with nothing after parse as ``[]``; render them as ''."""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return v or ""


def render_records_index(rows, today=None):
    today = today or _dt.date.today()
    lines = [
        "# Records",
        "",
        "Standalone writeups: experiment diagnoses, task briefs, handoffs, setup notes. Reference",
        "documents, not live state — the current state of each experiment is its",
        "`ops/experiments/<name>/STATUS.md` (index: `ops/experiments/INDEX.md`).",
        "",
        f"_Table generated by `ops exp records` on {today.isoformat()}. Edit a doc's front-matter "
        "`summary:` (template below), not this table._",
        "",
    ]
    # Optional `area:` header (e.g. pipeline / token_routing): shown as a column when any doc has it.
    with_area = any(r.get("area") for r in rows)
    if with_area:
        lines += ["| doc | area | updated | kind | status | summary |", "|---|---|---|---|---|---|"]
    else:
        lines += ["| doc | updated | kind | status | summary |", "|---|---|---|---|---|"]
    key = lambda r: (r["date"] or "0000", r["name"])  # noqa: E731
    for r in sorted(rows, key=key, reverse=True):
        summ = _cell(r["summary"])
        if len(summ) > 400:
            summ = summ[:397] + "..."
        status = _cell(r["status"])
        if r.get("superseded_by"):
            status += f" → `{_cell(r['superseded_by'])}`"
        area = f" {_cell(r.get('area'))} |" if with_area else ""
        lines.append(f"| [{r['name']}]({r['name']}) |{area} {_cell(r['date'])} | {_cell(r['kind'])} | "
                     f"{status} | {summ} |")
    missing = [r["name"] for r in rows if not r["has_header"]]
    lines += ["", "## New record header (copy into the top of every new doc)", "", "```yaml",
              RECORD_TEMPLATE.strip(), "```", ""]
    if missing:
        lines += [f"_{len(missing)} docs still lack a front-matter header._", ""]
    return "\n".join(lines)


def cmd_records(args):
    from . import paths as _paths
    if args.records_dir:
        rdir = Path(args.records_dir)
    elif os.environ.get("OPS_REPO_ROOT"):
        rdir = Path(os.environ["OPS_REPO_ROOT"]) / "records"
    else:
        # ops/ lives inside the repo whose records/ it indexes (the checkout ops/ resolves to).
        rdir = _paths.OPS_ROOT.parent / "records"
    if not rdir.is_dir():
        print(f"no records dir {rdir}", file=sys.stderr)
        return 1
    text = render_records_index(record_rows(rdir), today=_today(args))
    subdirs = sorted(d.name for d in rdir.iterdir() if d.is_dir() and (d / "README.md").is_file())
    if subdirs:
        links = ", ".join(f"[`{s}/`]({s}/README.md)" for s in subdirs)
        text = text.rstrip("\n") + f"\n\nSubdirectories (each with its own README): {links}.\n"
    if args.write:
        (rdir / "README.md").write_text(text + "\n", encoding="utf-8")
        print(f"wrote {rdir / 'README.md'}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------- CLI


def _today(args):
    t = getattr(args, "today", None)
    return _dt.date.fromisoformat(t) if t else _dt.date.today()


def cmd_new(args):
    root = Path(args.dir) if args.dir else experiments_dir()
    if not re.match(r"^[A-Za-z0-9_.-]+$", args.name):
        print(f"bad experiment name {args.name!r} (use [A-Za-z0-9_.-])", file=sys.stderr)
        return 2
    path = root / args.name / "STATUS.md"
    if path.exists():
        print(f"{path} already exists", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template(args.name, today=_today(args)), encoding="utf-8")
    print(path)
    return 0


def cmd_index(args):
    root = Path(args.dir) if args.dir else experiments_dir()
    today = _today(args)
    exps = load_all(root, today=today)
    text = render_index(exps, today=today)
    if args.stdout:
        print(text)
    else:
        root.mkdir(parents=True, exist_ok=True)
        (root / "INDEX.md").write_text(text + "\n", encoding="utf-8")
        print(f"wrote {root / 'INDEX.md'} ({len(exps)} experiments)")
    for e in exps:
        for p in e["problems"]:
            print(f"warning: {e['name']}: {p}", file=sys.stderr)
    return 0


def cmd_show(args):
    root = Path(args.dir) if args.dir else experiments_dir()
    path = root / args.name / "STATUS.md"
    if not path.is_file():
        names = [e["name"] for e in load_all(root)]
        close = [n for n in names if args.name.lower() in n.lower()]
        print(f"no {path}. " + (f"Did you mean: {', '.join(close)}" if close else
                                f"Known: {', '.join(names)}"), file=sys.stderr)
        return 1
    e = load_one(path, today=_today(args))
    if args.json:
        print(json.dumps({k: e[k] for k in ("name", "path", "meta", "problems")}, indent=2))
    else:
        print(path.read_text(encoding="utf-8"))
        for p in e["problems"]:
            print(f"warning: {p}", file=sys.stderr)
    return 0


def cmd_stale(args):
    root = Path(args.dir) if args.dir else experiments_dir()
    today = _today(args)
    rows = stale(load_all(root, today=today), days=args.days, today=today)
    if args.json:
        print(json.dumps([{"name": r["name"], "updated": r["meta"].get("updated"),
                           "age_days": r["age_days"]} for r in rows], indent=2))
        return 0
    if not rows:
        print(f"no active experiment older than {args.days} days")
        return 0
    for r in rows:
        age = "?" if r["age_days"] is None else f"{r['age_days']}d"
        print(f"{r['name']:<28} updated {r['meta'].get('updated') or '?':<10} ({age})  "
              f"next: {r['meta'].get('next_step', '')}")
    return 0


def cmd_check(args):
    root = Path(args.dir) if args.dir else experiments_dir()
    exps = load_all(root, today=_today(args))
    bad = [e for e in exps if e["problems"]]
    for e in bad:
        print(f"{e['name']}: {'; '.join(e['problems'])}")
    if not bad:
        print(f"{len(exps)} STATUS.md files OK")
    return 1 if bad else 0


def add_parser(subparsers):
    p = subparsers.add_parser("exp", help="experiment STATUS.md pages (new/index/show/stale/check)")
    p.add_argument("--dir", default=None, help="experiments dir (default ops/experiments)")
    p.add_argument("--today", default=None, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="exp_cmd")

    s = sub.add_parser("new", help="create <name>/STATUS.md from the template")
    s.add_argument("name")
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("index", help="regenerate experiments/INDEX.md")
    s.add_argument("--stdout", action="store_true", help="print instead of writing INDEX.md")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("show", help="print one STATUS.md")
    s.add_argument("name")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("stale", help="active experiments not updated in N days")
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stale)

    s = sub.add_parser("check", help="validate every STATUS.md front matter (exit 1 on problems)")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("records", help="regenerate records/README.md table from each doc's header "
                       "(prints unless --write)")
    s.add_argument("--records-dir", default=None, help="default: $OPS_REPO_ROOT/records, else the records/ next to this ops/ checkout")
    s.add_argument("--write", action="store_true", help="overwrite records/README.md")
    s.set_defaults(func=cmd_records)

    p.set_defaults(func=lambda a: (p.print_help(), 2)[1])
