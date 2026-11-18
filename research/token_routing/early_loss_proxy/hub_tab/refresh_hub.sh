#!/bin/bash
# Rebuild the hub page's auto-research tab from hub_tab/loop_state.json + hub_tab/ladder.json (2026-11-18: done from the loop
# session because the hub-reorg session was unreachable). Steps around this script are tool calls, not shell:
#   1. Artifact read https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt  -> saved wrapped page; pass its path as $1
#   2. bash hub_tab/refresh_hub.sh <saved live page>                     -> $OUT/new.html (unwrapped, built, verified)
#   3. Artifact publish file_path=$OUT/new.html url=<same URL>  (never force; if a newer version exists, redo from step 1)
# The builder used is the hub's own (main checkout hub_live/, with idea_impl.json and decisions.json), never hub_tab/build_autoresearch_tab.py.
set -eo pipefail
LIVE=${1:?saved live page from Artifact read}
HERE=$(cd "$(dirname "$0")" && pwd)
HUB=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/research/token_routing/ds64_fast8k/hub_live
OUT=${OUT:-/tmp/claude-3018/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core--claude-worktrees-early-loss-proxy-d6e886/0e77bb9a-326f-416e-bedd-8eeb58f8a792/scratchpad/hub}
mkdir -p "$OUT"; cd "$OUT"
python3 - "$LIVE" <<'P'
import sys
live = open(sys.argv[1], encoding="utf-8").read()
if "<body>" in live[:3000]:  # strip the read wrapper
    live = live[live.index("<body>") + 6:]; live = live[:live.rfind("</body>")]
open("base.html", "w", encoding="utf-8").write(live.strip() + "\n")
P
command cp -f base.html new.html
python3 "$HUB/build_autoresearch_tab.py" new.html --state "$HERE/loop_state.json" | tail -n 1
python3 "$HERE/add_ladder_table.py" new.html
python3 - <<'P'
import re, subprocess
h = open("new.html", encoding="utf-8").read(); bad = 0
for k, s in enumerate(re.findall(r'<script(?![^>]*application/json)[^>]*>(.*?)</script>', h, flags=re.S)):
    if s.strip():
        open("chk.js", "w").write(s); r = subprocess.run(["node", "--check", "chk.js"], capture_output=True, text=True); bad += r.returncode != 0
        if r.returncode: print("SCRIPT FAIL", k, r.stderr[:300])
print("node --check:", "OK" if not bad else f"{bad} FAILED")
P
rm -f chk.js
python3 "$HUB/verify_reorg.py" base.html new.html | tail -n 1
command cp -f new.html "$HUB/hub.html"; command cp -f "$HERE/loop_state.json" "$HUB/autoresearch_state.json"
echo "publish: $OUT/new.html"
