#!/bin/bash
# Rebuild hub.html = v95 base + November tabs + pager fix + current experiment log.
set -e
S=/tmp/claude-3018/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/c664b673-5989-46a0-b9dc-b1019ba743c0/scratchpad
R=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd $S
cp hub_v95_backup.html hub.html
python3 patch_log_render.py hub.html >/dev/null || true
python3 build_nov_tabs.py hub.html
python3 - <<'PY'
import json
S=/tmp/claude-3018/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/c664b673-5989-46a0-b9dc-b1019ba743c0/scratchpad
R="/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
h=open(S+"/hub.html").read()
old='const title = id => { const a = links.find(a => a.getAttribute("href") === "#" + id); return a ? a.textContent.replace(/^\\S+/, "").trim() || a.textContent : id; };'
if old in h: h=h.replace(old,'const title = id => { const a = links.find(a => a.getAttribute("href") === "#" + id); return a ? (a.lastChild && a.lastChild.nodeType === 3 ? a.lastChild.textContent.trim() : a.textContent) : id; };')
d=json.load(open(R+"/research/token_routing/ds64_fast8k/experiment_log.json"))
i=h.index("const D_LOG = ")+len("const D_LOG = "); j=h.index("; const D_COMPUTE",i)
h=h[:i]+json.dumps(d).replace("</","<\\/")+h[j:]
open(S+"/hub.html","w").write(h)
import re; open(S+"/allscripts.js","w").write("\n;\n".join(re.findall(r'<script>(.*?)</script>',h,re.S)))
PY
node --check allscripts.js && echo REBUILT_OK
