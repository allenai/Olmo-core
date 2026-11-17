#!/usr/bin/env python3
"""How Claude Code sessions spent wall-clock time, 2026-10-20 .. 2026-11-10 (stdlib, one pass).

For each transcript (~/.claude*/projects/*{OLMo-core,PAIL}*/*.jsonl) we take the timestamps of
user/assistant entries in the window and look at every gap of >= GAP_MIN minutes between two
consecutive entries. A gap is "the session was idle, waiting for something"; what ended it says
what it was waiting for:
  human       a message typed by Prasann (origin.kind == human, plain prompt)
  job/bg      a task-notification (background command, Monitor, subagent) or ci/side-session event
  session     a message from another Claude session (SendMessage)
  wakeup      a self-scheduled /loop or cron prompt
  other       anything else (mostly the app's "linked sessions now" note, i.e. a side session changed state)
Active time = sum of gaps shorter than GAP_MIN. This measures session idling, not GPU idling.

  python3 sessions.py --out data/sessions.csv
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os

W0, W1 = "2026-10-20", "2026-11-11"
GAP_MIN = 10


def classify(d):
    m = d.get("message") if isinstance(d.get("message"), dict) else {}
    c = m.get("content")
    text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else ""
    if isinstance(c, list) and any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
        return "tool"
    o = d.get("origin") or {}
    kind = o.get("kind") if isinstance(o, dict) else None
    t = text.lstrip()[:200]
    if t.startswith("<task-notification") or "side-sessions-event" in t[:40] or t.startswith("<ci-monitor"):
        return "job/bg"
    if t.startswith("Another Claude") or kind == "session":
        return "session"
    if kind == "human" and not d.get("isMeta"):
        return "human"
    if "loop" in t[:60].lower() or "autonomous" in t[:80].lower() or kind in ("scheduled", "cron", "loop"):
        return "wakeup"
    return "other:" + str(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sessions.csv")
    a = ap.parse_args()
    files = []
    for root in (os.path.expanduser("~/.claude/projects"), os.path.expanduser("~/.claude-work/projects")):
        for p in glob.glob(root + "/*/*.jsonl"):
            if ("OLMo-core" in p or "PAIL" in p) and dt.datetime.fromtimestamp(os.path.getmtime(p)).isoformat() >= W0:
                files.append(p)
    rows = []
    for p in files:
        ev = []
        for line in open(p):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") not in ("user", "assistant") or d.get("isSidechain"):
                continue
            ts = d.get("timestamp", "")
            if not (W0 <= ts < W1):
                continue
            ev.append((ts, "assistant" if d["type"] == "assistant" else classify(d)))
        if len(ev) < 5:
            continue
        ev.sort()
        tt = [dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t, _ in ev]
        waits = {}
        nlong = {}
        active = 0.0
        n_human = sum(1 for _, k in ev if k == "human")
        for i in range(1, len(ev)):
            g = (tt[i] - tt[i - 1]).total_seconds() / 3600
            if g * 60 < GAP_MIN:
                active += g
            else:
                k = ev[i][1].split(":")[0]
                waits[k] = waits.get(k, 0.0) + g
                if g >= 1:
                    nlong[k] = nlong.get(k, 0) + 1
        proj = "PAIL" if "PAIL" in p else "OLMo-core"
        rows.append({
            "session": os.path.basename(p)[:8], "project": proj, "dir": os.path.basename(os.path.dirname(p))[-28:],
            "first": ev[0][0][:16], "last": ev[-1][0][:16],
            "span_h": round((tt[-1] - tt[0]).total_seconds() / 3600, 2), "active_h": round(active, 2),
            "n_human_msgs": n_human,
            **{f"wait_{k}_h": round(waits.get(k, 0.0), 2) for k in ("human", "job/bg", "session", "wakeup", "other", "tool", "assistant")},
            **{f"n_waits_gt1h_{k}": nlong.get(k, 0) for k in ("human", "job/bg", "session", "other")},
        })
    rows.sort(key=lambda r: r["first"])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tot = {k: round(sum(r[k] for r in rows), 1) for k in rows[0] if k.endswith("_h") or k.startswith("n_")}
    print(len(rows), "sessions", tot)


if __name__ == "__main__":
    main()
