#!/usr/bin/env python3
"""Inline data/*.csv + summary.json into page_template.html -> page/gpu-audit.html (the artifact).

Other users' real names are restored from raw/user_map.json for the private page only; the
committed data/ files carry pseudonyms (user1, user2, ...).
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "data"


def rows(name):
    return list(csv.DictReader(open(D / name)))


S = json.load(open(D / "summary.json"))
umap = {}
try:
    umap = {v: k for k, v in json.load(open(HERE / "raw" / "user_map.json")).items()}
except OSError:
    pass


def real(d):
    return {",".join(umap.get(x, x) for x in k.split(",")): v for k, v in d.items()}


for key in ("lambda_other_users_gpuh", "jst_other_users_gpuh_top", "jst_other_users_preempted_jobs"):
    S[key] = real(S[key])
for key in ("by_user", "by_preemptor", "gpuh_by_preemptor"):
    S["lambda_other_users_preempted"][key] = real(S["lambda_other_users_preempted"][key])

hourly = rows("hourly_occupancy.csv")
data = {
    "S": S,
    "daily_pool": rows("daily_gpuh_pool.csv"),
    "daily_exp": rows("daily_gpuh_experiment.csv"),
    "daily_kind": rows("daily_gpuh_kind.csv"),
    "hourly": {k: [float(r[k]) if k != "hour_utc" else r[k] for r in hourly] for k in hourly[0]},
    "waits": rows("waits.csv"),
    "sessions": rows("sessions.csv"),
}
inc = D / "workflow_incidents.csv"
data["incidents"] = rows("workflow_incidents.csv") if inc.exists() else []
tpl = (HERE / "page_template.html").read_text()
out = HERE / "page" / "gpu-audit.html"
out.parent.mkdir(exist_ok=True)
out.write_text(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(out, out.stat().st_size)
