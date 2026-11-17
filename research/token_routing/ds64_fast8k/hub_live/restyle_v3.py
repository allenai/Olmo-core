"""2026-11-17, second pass: restructure the grouped hub like the PAIL Workbench (two rows of tabs, hero, collapsed section bars).

    python3 restyle_v3.py hub.html          # in place; refuses to run twice

What it does: swaps the v2 CSS block for hub_v3.css, the sidebar for a site title + two tab rows, the nav script for
hub_v3_nav.js, and turns every `<section><h2>` of the content parts into `<details class="sec">` bars (heading, counts of
charts / tables, verdict snippet; the first bar of a part, or all of a short part, is open). Parts left alone: overview, log,
autoresearch (its builder writes the same markup), summary. Nothing inside a section is changed.
"""
import re
import sys

P = sys.argv[1]
h = open(P, encoding="utf-8").read()
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
assert "/* ---- v3 (" not in h, "already restyled"

# css: replace the v2 block (from its marker comment to the end of the page's own style block)
a = h.index("/* ---- v2 (2026-11-17)")
b = h.index('</style>\n<link rel="preconnect"')
h = h[:a] + open(HERE + "/hub_v3.css", encoding="utf-8").read() + h[b:]

# nav shell
old = '<nav class="toc" id="toc" aria-label="Sections"><div class="tocin"></div><div class="secbar"></div></nav>'
assert old in h
h = h.replace(old, '<div class="sitehead" id="sitehead">Token Drop Results<span class="cur"></span></div>\n<nav class="toc" id="toc" aria-label="Sections"><div class="tabs1"></div><div class="tabs2"></div></nav>')
# nav script
m = re.search(r"<script>\n/\* two-level navigation \+ pager.*?</script>", h, re.S)
assert m
h = h[: m.start()] + "<script>\n" + open(HERE + "/hub_v3_nav.js", encoding="utf-8").read() + "</script>" + h[m.end():]


def text_of(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


SKIP = {"overview", "log", "autoresearch", "summary"}
SEC = re.compile(r"<section\b([^>]*)>(.*?)</section>", re.S)
done = []


def bar(m, pid, opened):
    attrs, inner = m.group(1), m.group(2)
    hm = re.search(r"<h2\b[^>]*>.*?</h2>", inner, re.S)
    if not hm:
        return m.group(0)
    body = (inner[: hm.start()] + inner[hm.end():]).strip("\n")
    n = []
    c, t, im = body.count("<canvas"), body.count("<table"), body.count("<img")
    if c:
        n.append("%d chart%s" % (c, "" if c == 1 else "s"))
    if im:
        n.append("%d image%s" % (im, "" if im == 1 else "s"))
    if t:
        n.append("%d table%s" % (t, "" if t == 1 else "s"))
    vm = re.search(r'<p class="note sv"[^>]*>(.*?)</p>', body, re.S)
    vl = ""
    if vm:
        s = text_of(vm.group(1))
        s = s if len(s) <= 220 else s[:220].rsplit(" ", 1)[0] + " …"
        vl = '<span class="vl">%s</span>' % s.replace("&", "&amp;").replace("<", "&lt;") if "&" not in s else '<span class="vl">%s</span>' % s
    cls = re.search(r'class="([^"]*)"', attrs)
    rest = re.sub(r'\s*class="[^"]*"', "", attrs)
    klass = "sec" + (" " + cls.group(1) if cls else "")
    done.append((pid, text_of(hm.group(0))[:50]))
    return '<details class="%s"%s%s>\n<summary>%s%s%s</summary>\n<div class="secbody">\n%s\n</div>\n</details>' % (
        klass, rest, " open" if opened else "", hm.group(0), ('<span class="n">%s</span>' % " · ".join(n)) if n else "", vl, body)


starts = [(mm.start(), mm.group(1)) for mm in re.finditer(r'<div class="part" id="([^"]+)"', h)]
out, last = [], 0
for i, (s0, pid) in enumerate(starts):
    e0 = starts[i + 1][0] if i + 1 < len(starts) else h.index("</main>")
    chunk = h[s0:e0]
    if pid not in SKIP:
        secs = list(SEC.finditer(chunk))
        k = [0]

        def sub(m, pid=pid, secs=secs, k=k):
            opened = len(secs) <= 3 or k[0] == 0
            k[0] += 1
            return bar(m, pid, opened)
        chunk = SEC.sub(sub, chunk)
    out.append(h[last:s0])
    out.append(chunk)
    last = e0
out.append(h[last:])
h = "".join(out)
open(P, "w", encoding="utf-8").write(h)
print("restyled", P, len(h), "bars:", len(done))
