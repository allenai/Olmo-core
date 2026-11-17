"""Check that the reorganized hub lost nothing relative to the old one.

    python3 verify_reorg.py OLD.html NEW.html [NEW_DOM_DUMP.html STATE.json]

Compares (excluding the auto-research part, the navigation, the overview, pagers and jump rows, which are rebuilt):
element counts, every old id and canvas id, every static text block (td th li p h2 h3 figcaption summary) as a multiset,
and the hash of each data <script>. With a rendered DOM dump (chrome --dump-dom of NEW#autoresearch) and the state JSON it
also checks every board row, idea, log entry and validation row appears in the auto-research tab.
"""
import collections
import hashlib
import json
import re
import sys
from html.parser import HTMLParser

old = open(sys.argv[1], encoding="utf-8").read()
new = open(sys.argv[2], encoding="utf-8").read()
ok = True


def strip_ar(h):
    h = re.sub(r"<!-- AUTORESEARCH:START -->.*?<!-- AUTORESEARCH:END -->", "", h, flags=re.S)
    h = re.sub(r"<!-- AR:(PART|SCRIPT):START -->.*?<!-- AR:\1:END -->", "", h, flags=re.S)
    return h


def strip_chrome(h):
    h = re.sub(r"<nav class=\"toc\".*?</nav>", "", h, flags=re.S)
    h = re.sub(r"<header>.*?</header>", "", h, flags=re.S)
    h = re.sub(r'<div class="part" id="overview".*?\n</div>\n(?=<div class="part")', "", h, flags=re.S)
    return h


class Blocks(HTMLParser):
    TAGS = {"td", "th", "li", "p", "h2", "h3", "figcaption", "summary"}

    def __init__(self):
        super().__init__()
        self.stack, self.out, self.skip = [], [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
        if tag == "nav" and ("pager" in (a.get("class") or "") or "jump" in (a.get("class") or "")):
            self.skip += 1
            self.stack.append(("skipnav", []))
            return
        if tag in self.TAGS:
            self.stack.append((tag, []))

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        if tag == "nav" and self.stack and self.stack[-1][0] == "skipnav":
            self.stack.pop()
            self.skip -= 1
            return
        if tag in self.TAGS and self.stack and self.stack[-1][0] == tag:
            t, buf = self.stack.pop()
            s = re.sub(r"\s+", " ", "".join(buf)).strip()
            if s:
                self.out.append((t, s))
            if self.stack and self.stack[-1][0] != "skipnav":
                self.stack[-1][1].append(" ".join([s]))

    def handle_data(self, d):
        if self.skip:
            return
        if self.stack:
            self.stack[-1][1].append(d)


def blocks(h):
    p = Blocks()
    p.feed(h)
    return collections.Counter(p.out)


o, n = strip_chrome(strip_ar(old)), strip_chrome(strip_ar(new))
print("== counts (old -> new, auto-research part and chrome excluded)")
for k in ("<table", "<canvas", "<figure", "<img", "<section", "<details", 'class="part"', "data-logfig"):
    print("  %-14s %5d -> %5d" % (k, o.count(k), n.count(k)))
for k in ("<table", "<canvas", "<figure", "<img"):
    if o.count(k) != n.count(k):
        ok = False
        print("  MISMATCH", k)
ids_o = set(re.findall(r'\sid="([^"]+)"', o))
ids_n = set(re.findall(r'\sid="([^"]+)"', n))
miss = sorted(ids_o - ids_n)
print("== ids: %d old, %d new, missing %s" % (len(ids_o), len(ids_n), miss))
ok &= not miss
bo, bn = blocks(o), blocks(n)
lost = bo - bn
print("== text blocks old %d, new %d, lost %d (expected: retitled headings only)" % (sum(bo.values()), sum(bn.values()), sum(lost.values())))
for (t, s), c in sorted(lost.items()):
    print("   lost %s x%d: %s" % (t, c, s[:110]))
    if t not in ("h2",):
        ok = False
def data_scripts(h):
    out = []
    for m in re.finditer(r"<script>(.*?)</script>", h, re.S):
        b = m.group(1)
        if "nav.toc a" in b or "hubgroups" in b or "ovMap" in b or "__ARDATA__" in b or b.lstrip().startswith("/* two-level") or b.lstrip().startswith("/* overview"):
            continue
        out.append(hashlib.sha1(b.encode()).hexdigest()[:10])
    return out
so, sn = data_scripts(strip_ar(old)), data_scripts(strip_ar(new))
print("== data/render scripts: old %d, new %d, identical sets: %s" % (len(so), len(sn), sorted(so) == sorted(sn)))
ok &= sorted(so) == sorted(sn)
if len(sys.argv) > 4:
    dom = open(sys.argv[3], encoding="utf-8").read()
    st = json.load(open(sys.argv[4], encoding="utf-8"))
    s0 = dom.index('id="autoresearch"')
    seg = dom[s0: dom.index('id="provenance"', s0)]
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", seg))
    txt = txt.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    nrows = len(re.findall(r"<tr", seg[seg.index('id="arFam"'): seg.index('id="ar-depth-sec"')])) - len(re.findall(r"<table", seg[seg.index('id="arFam"'): seg.index('id="ar-depth-sec"')]))
    hidden = sum(1 for r in st["board"] if r["tag"] + "/" + r["arm"] == "hp/denselora-lr2e5")
    print("== auto-research: board rows %d (hidden by the loop's own rule: %d), rendered arm rows %d" % (len(st["board"]), hidden, nrows))
    ok &= nrows == len(st["board"]) - hidden
    for k in ("ideas", "log", "validation", "setup"):
        items = st[k]
        miss = 0
        for it in items:
            for field in (it.values() if isinstance(it, dict) else [it]):
                if isinstance(field, str) and len(field) > 12 and re.sub(r"\s+", " ", field).strip().lower() not in txt.lower():
                    miss += 1
                    print("   not found in %s: %s" % (k, field[:80]))
        print("   %s: %d items, %d text fields missing" % (k, len(items), miss))
        ok &= miss == 0
    for r in st["pilot"]:
        if r["arm"] not in txt:
            ok = False
            print("   pilot arm missing", r["arm"])
print("RESULT:", "OK" if ok else "CHECK ABOVE")
