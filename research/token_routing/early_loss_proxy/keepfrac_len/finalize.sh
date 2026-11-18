#!/bin/bash
# Rescore everything and refresh the committed outputs: kl_results.json, kl_summary.json, q*_scored.json, the chart,
# the per-length tables appended to the record, and hub_keeplen_block.html.   bash finalize.sh   (login node, ~2 min)
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../.." && pwd)
D=/tmp/claude-3018/keepfrac_len_results
bash "$HERE/score_all.sh" "$D" >/dev/null
cp "$D/kl_results.json" "$D/kl_summary.json" "$D/q32k_scored.json" "$D/q64k_scored.json" "$D/keepfrac_vs_length.png" "$HERE/"
python3 "$HERE/make_tables.py" --summary "$D/kl_summary.json" --md "$D/tables.md" --html "$D/block_table.html"
REC=$REPO/records/keep-fraction-vs-length.md
python3 - "$REC" "$D/tables.md" <<'PY'
import sys
rec, tab = sys.argv[1], open(sys.argv[2]).read()
s = open(rec).read()
a, b = "<!-- tables:begin (finalize.sh) -->", "<!-- tables:end -->"
s = s[:s.index(a) + len(a)] + "\n" + tab + "\n" + s[s.index(b):]
open(rec, "w").write(s)
PY
python3 - "$HERE" "$D/block_table.html" <<'PY'
import base64, sys
here, tab = sys.argv[1], open(sys.argv[2]).read()
head = open(f"{here}/hub_keeplen_head.html").read()
png = base64.b64encode(open(f"{here}/keepfrac_vs_length.png", "rb").read()).decode()
open(f"{here}/hub_keeplen_block.html", "w").write(
    head + tab + f'<img alt="FLOP speedup vs keep per length; best keep vs length" style="max-width:100%" '
    f'src="data:image/png;base64,{png}">\n')
PY
echo "refreshed: $REC, $HERE/{kl_results.json,kl_summary.json,keepfrac_vs_length.png,hub_keeplen_block.html}"
