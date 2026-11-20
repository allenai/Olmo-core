"""Map every scoring run name -> (git commit, Beaker job id) from the lcS-* experiments registered in ops (their
SPEC_B64 env var lists the runs, GIT_REF the commit). Writes score_provenance.json beside this file."""
import base64
import json
import os
import subprocess

OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
txt = subprocess.run([OPS, "runs", "--experiment", "tokdrop_lr", "--json"], capture_output=True, text=True).stdout
try:
    runs = json.loads(txt)
except Exception:
    runs = [json.loads(l) for l in txt.splitlines() if l.strip().startswith("{")]
runs = runs if isinstance(runs, list) else list(runs.values())
for r in runs:
    rid = r.get("run_id", "")
    if not rid.startswith("lc-lcS-") or not r.get("job_id"):
        continue
    ex = r["job_id"]
    try:
        d = json.loads(subprocess.run(["beaker", "experiment", "get", ex, "--format", "json"], capture_output=True, text=True).stdout)[0]
    except Exception:
        continue
    j = d["jobs"][-1]
    ev = {e.get("name"): e.get("value") for e in j.get("execution", {}).get("spec", {}).get("envVars", [])}
    if "SPEC_B64" not in ev:
        continue
    for line in base64.b64decode(ev["SPEC_B64"]).decode().splitlines():
        name = line.split("\t")[0]
        if name and j.get("status", {}).get("exited") is not None or name:
            out.setdefault(name, []).append({"commit": ev.get("GIT_REF"), "job": j["id"], "experiment": ex,
                                             "exit": j.get("status", {}).get("exitCode")})
json.dump(out, open(os.path.join(HERE, "score_provenance.json"), "w"), indent=1)
print(len(out), "runs mapped")
