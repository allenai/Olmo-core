#!/bin/bash
# Live curves of unfinished Beaker proxy jobs: dump the [dense-probe] lines of every job of a tag (state_beaker.json) into
# /tmp/claude-3018/elp_beaker_results/<tag>/<run>.log so `ladder.py --live` can read them.   bash peek_beaker.sh <tag> [<tag>...]
HERE=$(cd "$(dirname "$0")" && pwd); export PATH=$HOME/.local/bin:$PATH
for TAG in "$@"; do
  O=/tmp/claude-3018/elp_beaker_results/$TAG; mkdir -p "$O"
  python3 - "$HERE/state_beaker.json" "$TAG" <<'P' | while read -r ID RUN; do
import json, sys
s = json.load(open(sys.argv[1]))
if len(s) == 1 and isinstance(next(iter(s.values())), dict) and "tag" not in next(iter(s.values())): s = next(iter(s.values()))
for k, v in s.items():
    if isinstance(v, dict) and v.get("tag") == sys.argv[2]:
        for r in v["runs"]: print(k, r)
P
    [ -f "$O/$RUN.json" ] && grep -q '"done": true' "$O/$RUN.json" && continue
    timeout 60 beaker experiment logs "$ID" 2>/dev/null | grep -a '^\[dense-probe\] step\|dense-probe\] step' | sed 's/^.*\[dense-probe\]/[dense-probe]/' > "$O/$RUN.log.tmp" && [ -s "$O/$RUN.log.tmp" ] && mv "$O/$RUN.log.tmp" "$O/$RUN.log"
  done
done
