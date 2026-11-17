"""Splice ../experiment_log.json into the hub's `const D_LOG` (the experiment log tab). Idempotent.

    python3 update_log.py hub.html [../experiment_log.json]
"""
import json
import os
import sys

H = sys.argv[1]
J = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment_log.json")
h = open(H, encoding="utf-8").read()
d = json.load(open(J, encoding="utf-8"))
a = h.index("const D_LOG = ") + len("const D_LOG = ")
b = h.index("; const D_COMPUTE", a)
h = h[:a] + json.dumps(d).replace("</", "<\\/") + h[b:]
open(H, "w", encoding="utf-8").write(h)
print("spliced", len(d), "log entries into", H)
