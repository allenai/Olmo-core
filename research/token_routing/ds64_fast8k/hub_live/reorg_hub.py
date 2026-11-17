"""One-off transform used on 2026-11-17: old flat 16-tab hub -> grouped hub (overview + 6 groups).

    python3 reorg_hub.py OLD_HUB.html NEW_HUB.html

Kept for provenance. After this ran, `hub.html` is the source: to add a section, see README.md. The auto-research part
is not handled here (the old marker blocks are stripped); run build_autoresearch_tab.py on the output.
"""
import json
import re
import sys

SRC, DST = sys.argv[1], sys.argv[2]
h = open(SRC, encoding="utf-8").read()

# ---- 0. strip the old auto-research marker blocks (nav link, part, script); the builder re-adds them --------------
h = re.sub(r"<!-- AUTORESEARCH:START -->.*?<!-- AUTORESEARCH:END -->\n?", "", h, flags=re.S)

# ---- 1. split: head | parts | tail -----------------------------------------------------------------------------------
first = h.index('<div class="part" id="')
tail_i = h.index("</main>")
head, body, tail = h[:first], h[first:tail_i], h[tail_i:]
starts = [m.start() for m in re.finditer(r'<div class="part" id="', body)]
chunks = {}
order_old = []
for a, b in zip(starts, starts[1:] + [len(body)]):
    c = body[a:b]
    pid = re.match(r'<div class="part" id="([^"]+)"', c).group(1)
    chunks[pid] = c
    order_old.append(pid)

# ---- 2. metadata -----------------------------------------------------------------------------------------------------
GROUPS = [
    ("g-best", "Best method and how it is measured", "Best method", "The recipe and the metric: verdicts first, then the 4-task study that defines FLOPs-to-match."),
    ("g-scale", "Scale and length", "Scale, length", "Does it hold with more tasks, longer rows, other models and continued pretraining?"),
    ("g-tried", "What was tried and what is next", "Tried, next", "Token-selection rules, ruled-out ideas, prior work and the live auto-research search."),
    ("g-mech", "Mechanism and analysis", "Mechanism", "Why it works: where the gradient lives and how well the router ranks tokens."),
    ("g-cost", "Cost and infrastructure", "Cost", "Memory, wall-clock time and the GPU-hours spent."),
    ("g-log", "Log and provenance", "Log, provenance", "Every experiment in time order, and where the data, code and caveats live."),
]
# id: (group, order, nav label, eyebrow, new title or None)
META = {
    "summary": ("g-best", 1, "Verdicts", "Best method · was tab 1", "Verdicts, at a glance, open decisions"),
    "sft4": ("g-best", 2, "SFT 4-task", "Best method · was tab 2", "SFT 4-task: FLOPs-to-match and the LoRA × routing split"),
    "sft11": ("g-scale", 1, "11 tasks, 64k, length", "Scale and length · was tab 3", "SFT 11-task, 64k and length generalization"),
    "models": ("g-scale", 2, "Other models and sizes", "Scale and length · was tab 3b", "Other models and model sizes"),
    "cpt": ("g-scale", 3, "Continued pretraining", "Scale and length · was tab 4", "Continued pretraining on OLMo 3 7B"),
    "autoresearch": ("g-tried", 1, "Auto-research loop", "Tried · was tab 8d", None),
    "selection": ("g-tried", 2, "Token selection rules", "Tried · was tab 6c", "Which tokens to keep: selection rules"),
    "ruledout": ("g-tried", 3, "Ruled out", "Tried · was tab 7", "What did not work"),
    "next": ("g-tried", 4, "Next ideas", "Tried · was tab 8", "Next ideas with a ≥5× ceiling"),
    "priorwork": ("g-tried", 5, "Prior work", "Tried · was tab 8b", "Prior work: how the recipe compares"),
    "mechanism": ("g-mech", 1, "Gradient and router", "Mechanism · was tab 6", "Where the gradient lives and what the router gets right"),
    "gradmaps": ("g-mech", 2, "Gradient maps", "Mechanism · was tab 6b", "Gradient maps: per token, per layer"),
    "memory": ("g-cost", 1, "Memory and wall-clock", "Cost · was tab 5", "Memory and wall-clock"),
    "gpuaudit": ("g-cost", 2, "GPU audit", "Cost · was tab 8c", "GPU audit, 10-20 to 11-10"),
    "log": ("g-log", 1, "Experiment log", "Log · was the Experiment log tab", "Experiment log: what is running, what was tried"),
    "provenance": ("g-log", 2, "Provenance", "Provenance · was tab 9", "Data, code, caveats and detail pages"),
}
assert set(META) - {"autoresearch"} == set(chunks), set(chunks) ^ set(META)

report = []


def text_of(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


def short(s, n):
    s = text_of(s)
    if len(s) <= n:
        return s
    s = s[:n].rsplit(" ", 1)[0]
    return s.rstrip(",;:") + " …"


# ---- 3. per-part rewrites --------------------------------------------------------------------------------------------
HEAD_RE = re.compile(r'<div class="parthead"><span class="eyebrow">(.*?)</span><h2 class="ptitle">(.*?)</h2>(<p class="verdict">.*?</p>)</div>', re.S)


def rehead(pid, c):
    grp, order, label, eyebrow, title = META[pid]
    m = HEAD_RE.search(c)
    assert m, pid
    new = '<div class="parthead"><span class="eyebrow">%s</span><h2 class="ptitle">%s</h2>%s</div>' % (eyebrow, title, m.group(3))
    c = c[: m.start()] + new + c[m.end():]
    attrs = 'data-group="%s" data-order="%d" data-short="%s"' % (grp, order, label)
    return c.replace('<div class="part" id="%s"' % pid, '<div class="part" id="%s" %s' % (pid, attrs), 1)


SECTION_RE = re.compile(r"<section\b[^>]*>.*?</section>", re.S)
VNOTE_RE = re.compile(r'<p class="note"([^>]*)>(\s*(?:<b>)?\s*Verdict)(.*?)</p>', re.S)
NOTE_RE = re.compile(r'<p class="note"([^>]*)>(.*?)</p>', re.S)
TBL_RE = re.compile(r'<div class="tbl"[^>]*>\s*<table>.*?</table>\s*</div>', re.S)


def lead_verdict_notes(pid, c):
    def fix(sec):
        ms = list(VNOTE_RE.finditer(sec.group(0)))
        s = sec.group(0)
        if len(ms) != 1:
            return s
        m = ms[0]
        h2 = s.find("</h2>")
        if h2 < 0 or m.start() < h2:
            return s
        note = '<p class="note sv"%s>%s%s</p>' % (m.group(1), m.group(2), m.group(3))
        s = s[: m.start()] + s[m.end():]
        h2 = s.find("</h2>") + len("</h2>")
        report.append("verdict note moved to the top of its section in %s: %s" % (pid, short(m.group(3), 60)))
        return s[:h2] + "\n  " + note + s[h2:]
    return SECTION_RE.sub(fix, c)


def collapse_notes(pid, c):
    def fix(m):
        attrs, inner = m.group(1), m.group(2)
        t = text_of(inner)
        if len(t) <= 450 or "id=" in attrs or "⚠" in t:
            return m.group(0)
        pre = re.match(r"\s*<b>(.*?)</b>", inner, re.S)
        label = text_of(pre.group(1)) if pre and len(text_of(pre.group(1))) > 20 else short(inner, 110)
        body_note = m.group(0)
        report.append("long note folded in %s (%d chars): %s" % (pid, len(t), label[:60]))
        return '<details class="more"><summary>%s <span class="mh">full note</span></summary>%s</details>' % (label, body_note)
    return NOTE_RE.sub(fix, c)


def collapse_tables(pid, c):
    out, pos = [], 0
    for m in TBL_RE.finditer(c):
        n = m.group(0).count("<tr") - 1
        if n < 12:
            continue
        pre = c[: m.start()]
        hs = list(re.finditer(r"<h([23])[^>]*>(.*?)</h\1>", pre, re.S))
        heading = text_of(hs[-1].group(2)) if hs else pid
        if heading.startswith("What failed"):
            continue
        out.append(c[pos: m.start()])
        between = re.sub(r"\s+", "", c[hs[-1].end(): m.start()]) if hs else "x"
        what = "Show the table" if not between else "Show table: " + short(heading, 70)
        out.append('<details class="more tblmore"><summary>%s <span class="mh">%d rows</span></summary>%s</details>' % (what, n, m.group(0)))
        pos = m.end()
        report.append("table folded in %s (%d rows): %s" % (pid, n, short(heading, 50)))
    out.append(c[pos:])
    return "".join(out)


for pid in list(chunks):
    c = chunks[pid]
    c = rehead(pid, c)
    c = lead_verdict_notes(pid, c)
    c = collapse_notes(pid, c)
    c = collapse_tables(pid, c)
    chunks[pid] = c

# summary: "At a glance" first, then the long verdict list, then open decisions
c = chunks["summary"]
secs = list(SECTION_RE.finditer(c))
assert [text_of(re.search(r"<h2>(.*?)</h2>", s.group(0)).group(1)) for s in secs] == ["Verdicts", "At a glance", "Open decisions"]
v, g, o = (s.group(0) for s in secs)
c = c[: secs[0].start()] + g + "\n" + v + "\n" + o + c[secs[-1].end():]
chunks["summary"] = c

# ---- 4. overview part (replaces the old <header> and holds the page map) ------------------------------------------------
OVERVIEW = '''<div class="part" id="overview" data-group="home" data-order="0" data-short="Overview">
<div class="parthead"><span class="eyebrow">Train-time token / layer drop · all results in one place · reorganized 2026-11-17</span>
<h1>Cheaper training that matches dense at matched FLOPs</h1>
<p class="verdict">Routing plus LoRA trains ≈2–4× cheaper than dense on SFT at matched quality; in CPT, LoRA alone wins and every token drop loses.</p></div>
<section class="lead">
  <h2>Bottom line</h2>
  <p>The recipe: rank-64 LoRA adapters on blocks 0–6 (frozen base), an online gradient-norm router that keeps 20% of tokens from block 7 and rescores to 7% at block 16, no full-context steps, LR 5e-6. Whole-model LoRA (all 32 blocks, LR 1e-5) is cheaper still at the same quality. Routing does most of the matched-FLOP gain on SFT; LoRA alone is the CPT recipe.</p>
  <p class="note">Most points are one to three seeds, and seed-to-seed spread at 7,808 rows/task is as large as every effect measured, so claims need at least three seeds per arm. Each number below carries its seed count and caveat on its own section. Qwen3.5-4B SFT on CTC long-context tasks (4-task and 11-task mixes, 64k) and OLMo 3 7B long-document continued pretraining. Every plot puts training FLOPs (measured by the FLOP meter; PF = petaFLOPs) on the x-axis; up-and-left of the dense line is better (down-and-left for CPT loss). Private page.</p>
</section>
<section>
  <h2>Five numbers</h2>
  <div class="tiles">
    <a class="tile" href="#sft4"><b>≈3×</b><span>dense FLOPs needed to match the cheap methods, SFT 4-task at 3,904 fresh rows/task (≈2× at 7,808); the recipe costs 0.22× dense per step. 1–2 seeds.</span></a>
    <a class="tile" href="#sft11"><b>+0.037</b><span>task-average over dense at 0.365× dense FLOPs, SFT 11-task at 64k, ¼ epoch; ahead on all 10 graded tasks, 24% less peak active memory. One seed.</span></a>
    <a class="tile" href="#cpt"><b>4.8× → 1.4×</b><span>dense FLOPs to match, CPT on OLMo 3 7B from 48 to 960 steps; whole-model LoRA with a frozen output layer, 0.685× FLOPs per step.</span></a>
    <a class="tile" href="#memory"><b>≈2.4×</b><span>max train length at 80 GB for whole-model LoRA plus the router (≈1.8× for the recipe), extrapolated from short probes.</span></a>
    <a class="tile" href="#memory"><b>0.54–0.65×</b><span>step time for 0.22–0.30× of the FLOPs: wall-clock savings lag the FLOP savings.</span></a>
  </div>
</section>
<section>
  <h2>Running now</h2>
  <div id="ovLive"><p class="note">The auto-research loop reports here once its tab is built. Older sessions: see the <a href="#log">experiment log</a>.</p></div>
</section>
<section>
  <h2>Map of the page</h2>
  <p class="note">Six groups, each with its sections and their one-line verdicts. Every old tab id still opens its section.</p>
  <div id="ovMap" class="map"></div>
</section>
</div>
'''

# ---- 5. assemble ---------------------------------------------------------------------------------------------------------
# old header + flat nav out, new nav shell in
head = re.sub(r"<nav class=\"toc\".*?</nav>\n?", '<nav class="toc" id="toc" aria-label="Sections"><div class="tocin"></div><div class="secbar"></div></nav>\n', head, flags=re.S)
assert 'id="toc"' in head
head = re.sub(r"<header>.*?</header>\n?", "", head, flags=re.S)
head = head.replace('<main class="wrap">', '<main class="wrap">', 1)
GJSON = json.dumps([{"id": g[0], "title": g[1], "short": g[2], "blurb": g[3]} for g in GROUPS], ensure_ascii=False)
head = head.replace('<div class="layout">', '<script type="application/json" id="hubgroups">%s</script>\n<div class="layout">' % GJSON, 1)

# keep the DOM order of the old page (the nav sorts by data-group / data-order); overview first
parts_html = OVERVIEW + "".join(chunks[p] for p in order_old)
new = head + parts_html + tail

# ---- 6. css ---------------------------------------------------------------------------------------------------------------
CSS = open(__file__.replace("reorg_hub.py", "hub_v2.css"), encoding="utf-8").read()
i = new.index('</style>\n<link rel="preconnect"')  # end of the page's own style block (the first </style> is the skeleton reset)
new = new[:i] + CSS + new[i:]

# ---- 7. scripts: replace the nav/pager/router script, add map + details handling ------------------------------------------
NAVJS = open(__file__.replace("reorg_hub.py", "hub_v2_nav.js"), encoding="utf-8").read()
m = re.search(r"<script>\n\(\(\) => \{ const links = \[\.\.\.document\.querySelectorAll\(\"nav\.toc a\"\)\];.*?</script>", new, re.S)
assert m, "old nav script not found"
new = new[: m.start()] + "<script>\n" + NAVJS + "\n</script>" + new[m.end():]
MAPJS = open(__file__.replace("reorg_hub.py", "hub_v2_map.js"), encoding="utf-8").read()
j = new.rindex("</script>") + len("</script>")
new = new[:j] + "\n<script>\n" + MAPJS + "\n</script>" + new[j:]

open(DST, "w", encoding="utf-8").write(new)
print("wrote", DST, len(new))
print("\n".join(report))
