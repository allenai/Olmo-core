#!/usr/bin/env python3
"""MegaSlides helper: turn a short markdown spec into one Slides batchUpdate.

  locate READ.json          parse a saved masked read of the deck -> state (rev, week, template)
  build SPEC.md [--at N]    print `REV <id>` then the compact requests JSON for update_presentation
  thumbs URL [URL ...]      download slide thumbnails, stack them into one PNG, print its path

Stdlib only, except `thumbs` (PIL). Deck constants live in deck.json next to this file.
State goes to $MEGASLIDES_STATE or ~/.cache/megaslides/state.json.
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "deck.json")))
STATE = os.environ.get("MEGASLIDES_STATE", os.path.expanduser("~/.cache/megaslides/state.json"))
EMU = 914400
PT = 12700


# ---------------------------------------------------------------- locate
def _text(shape):
    return "".join(t.get("textRun", {}).get("content", "") for t in shape.get("text", {}).get("textElements", []))


def _paragraphs(shape):
    out = []
    for t in shape.get("text", {}).get("textElements", []):
        pm = t.get("paragraphMarker")
        if pm is not None:
            out.append((t.get("startIndex", 0), t["endIndex"], pm.get("bullet", {}).get("nestingLevel", 0)))
    return out


def locate(path):
    d = json.load(open(path))
    d = d.get("content", d)
    slides = d["slides"]
    rows = []
    for i, s in enumerate(slides):
        title, tmpl = "", None
        for e in s.get("pageElements", []):
            sh = e.get("shape", {})
            ph = sh.get("placeholder", {}).get("type")
            if ph in ("TITLE", "CENTERED_TITLE") and not title:
                title = _text(sh).strip()
            if e["objectId"] == CFG["template"]["body"]:
                tmpl = e
        rows.append((i, s["objectId"], title, tmpl))
    # newest week = first "Week #<digits>" divider (slide 0 is the "Week #N" example template)
    week = next(r for r in rows if re.fullmatch(r"Week #\d+(/\d+)?", r[2]))
    end = next(r for r in rows[week[0]:] if r[2].startswith("End of Week"))
    tb = next((r[3] for r in rows if r[3] is not None), None)
    if tb is None:
        sys.exit("template body %s not found; update deck.json" % CFG["template"]["body"])
    paras = _paragraphs(tb["shape"])
    groups = []
    for p in paras:
        if groups and groups[-1]["level"] == p[2]:
            groups[-1]["paras"].append(p[:2])
        else:
            groups.append({"level": p[2], "paras": [p[:2]]})
    tr = tb["transform"]
    st = {"rev": d["revisionId"], "week": week[2], "week_idx": week[0], "end_idx": end[0],
          "ids": [r[1] for r in rows], "groups": groups, "text_len": paras[-1][1] - 1,
          "body_base": [tb["size"]["width"]["magnitude"], tb["size"]["height"]["magnitude"]],
          "body_xy": [tr.get("translateX", 0), tr.get("translateY", 0)]}
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(st, open(STATE, "w"))
    print(f"rev {st['rev']}  newest: {week[2]} (slide {week[0]})  insert-at {end[0]} (before '{end[2]}')")
    for r in rows[week[0] + 1:end[0]]:
        print(f"  [{r[0]}] {r[1]}  {r[2][:70]}")
    print("template groups:", "".join(str(g["level"]) for g in groups), f"(state: {STATE})")


# ---------------------------------------------------------------- spec parsing
def parse_spec(path):
    """## Title / - bullet / '  - ' sub-bullet / |table| rows / chart: {json} / > caption."""
    slides, cur = [], None
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        s = line.strip()
        if line.startswith("## "):
            cur = {"title": line[3:].strip(), "bullets": [], "table": [], "chart": None, "caption": "", "id": None}
            slides.append(cur)
        elif cur is None or not s or s.startswith("<!--"):
            continue
        elif re.match(r"^\s*[-*] ", line):
            indent = len(line) - len(line.lstrip())
            cur["bullets"].append((0 if indent < 2 else 1, s[2:].strip()))
        elif s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                cur["table"].append(cells)
        elif s.startswith("chart:"):
            cur["chart"] = json.loads(s[6:])
        elif s.startswith(">"):
            cur["caption"] = (cur["caption"] + " " + s[1:].strip()).strip()
        elif s.startswith("id:"):
            cur["id"] = s[3:].strip()
        else:
            sys.exit(f"unparsed spec line: {line!r}")
    return slides


# ---------------------------------------------------------------- request helpers
def emu(inches):
    return int(round(inches * EMU))


def rgb(hexstr):
    h = hexstr.lstrip("#")
    return {c: round(int(h[i:i + 2], 16) / 255, 3) for c, i in (("red", 0), ("green", 2), ("blue", 4))}


def props(page, x, y, w, h, sy=1):
    return {"pageObjectId": page, "size": {"width": {"magnitude": max(emu(w), 1), "unit": "EMU"},
                                           "height": {"magnitude": max(emu(h), 1), "unit": "EMU"}},
            "transform": {"scaleX": 1, "scaleY": sy, "translateX": emu(x), "translateY": emu(y), "unit": "EMU"}}


class Batch:
    """Collects requests. Repeated chart elements are cheap: the first element of each style on a page
    is created and styled (the prototype); later ones are duplicateObject + one RELATIVE transform,
    ~40% fewer request bytes. RELATIVE, not ABSOLUTE: Slides re-normalizes a new shape's stored size
    (e.g. to 3000000 EMU with a compensating scale), so only rendered ratios are reliable."""

    def __init__(self):
        self.r, self.protos = [], {}

    def add(self, **kw):
        self.r.append(kw)

    def _clone(self, key, oid, page, x, y, w, h, flip=False):
        w, h, x, y = max(emu(w), 1), max(emu(h), 1), emu(x), emu(y)
        if key not in self.protos:
            self.protos[key] = (oid, w, h, x, y, -1 if flip else 1)
            return False
        src, w0, h0, x0, y0, f0 = self.protos[key]
        sx, sy = w / w0, (-1 if flip else 1) * f0 * h / h0
        self.add(duplicateObject={"objectId": src, "objectIds": {src: oid}})
        # new = M * proto: scale multiplies, translate = M.scale * proto.translate + M.translate
        self.add(updatePageElementTransform={"objectId": oid, "applyMode": "RELATIVE", "transform": {
            "scaleX": round(sx, 6), "scaleY": round(sy, 6), "translateX": round(x - sx * x0),
            "translateY": round(y - sy * y0), "unit": "EMU"}})  # omitted shear = 0
        return True

    def textbox(self, oid, page, x, y, w, h, text, pt, color=None, align=None, bold=False):
        if self._clone(("tb", page, pt, color, align, bold), oid, page, x, y, w, h):
            self.add(deleteText={"objectId": oid, "textRange": {"type": "ALL"}})
            self.add(insertText={"objectId": oid, "text": text})
            return
        self.add(createShape={"objectId": oid, "shapeType": "TEXT_BOX", "elementProperties": props(page, x, y, w, h)})
        self.add(insertText={"objectId": oid, "text": text})
        style, fields = {"fontSize": {"magnitude": pt, "unit": "PT"}}, ["fontSize"]
        if color:
            style["foregroundColor"] = {"opaqueColor": {"rgbColor": rgb(color)}}
            fields.append("foregroundColor")
        if bold:
            style["bold"] = True
            fields.append("bold")
        self.add(updateTextStyle={"objectId": oid, "textRange": {"type": "ALL"}, "style": style, "fields": ",".join(fields)})
        if align:
            self.add(updateParagraphStyle={"objectId": oid, "textRange": {"type": "ALL"},
                                           "style": {"alignment": align}, "fields": "alignment"})

    def rect(self, oid, page, x, y, w, h, color, shape="RECTANGLE"):
        if self._clone(("rect", page, shape, color), oid, page, x, y, w, h):
            return
        self.add(createShape={"objectId": oid, "shapeType": shape, "elementProperties": props(page, x, y, w, h)})
        self.add(updateShapeProperties={"objectId": oid, "shapeProperties": {
            "shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": rgb(color)}}},
            "outline": {"propertyState": "NOT_RENDERED"}}, "fields": "shapeBackgroundFill.solidFill.color,outline.propertyState"})

    def line(self, oid, page, x1, y1, x2, y2, color, pt=1.5):
        # a line runs top-left -> bottom-right of its box; a negative scaleY flips it to run upward
        up = y2 < y1
        if self._clone(("line", page, color, pt), oid, page, min(x1, x2), y1, abs(x2 - x1), abs(y2 - y1), flip=up):
            return
        self.add(createLine={"objectId": oid, "category": "STRAIGHT", "elementProperties": props(
            page, min(x1, x2), y1, abs(x2 - x1), abs(y2 - y1), sy=-1 if up else 1)})
        self.add(updateLineProperties={"objectId": oid, "lineProperties": {
            "lineFill": {"solidFill": {"color": {"rgbColor": rgb(color)}}}, "weight": {"magnitude": pt, "unit": "PT"}},
            "fields": "lineFill.solidFill.color,weight"})


def fmt(v, spec):
    f = spec.get("fmt")
    if f:
        return f.format(v)
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.3g}"


# ---------------------------------------------------------------- body bullets onto the template groups
def body_requests(b, bid, bullets, st):
    content = []
    for lvl, t in bullets:
        if content and content[-1][0] == lvl:
            content[-1][1].append(t)
        else:
            content.append((lvl, [t]))
    groups, mapping, j = st["groups"], {}, 0
    for ci, (lvl, _) in enumerate(content):
        while j < len(groups) and groups[j]["level"] != lvl:
            j += 1
        if j == len(groups):
            sys.exit("bullet structure %s does not fit template groups %s: use at most 3 top-level bullets "
                     "that have sub-bullets" % ("".join(str(c[0]) for c in content),
                                                 "".join(str(g["level"]) for g in groups)))
        mapping[j] = content[ci][1]
        j += 1
    last_used = max(mapping)
    if last_used < len(groups) - 1:  # trailing unused groups: cut from the previous newline, keep the final one
        b.add(deleteText={"objectId": bid, "textRange": {"type": "FIXED_RANGE",
                          "startIndex": groups[last_used + 1]["paras"][0][0] - 1, "endIndex": st["text_len"]}})
    for gi in range(last_used, -1, -1):
        P = groups[gi]["paras"]
        if gi not in mapping:
            b.add(deleteText={"objectId": bid, "textRange": {"type": "FIXED_RANGE",
                              "startIndex": P[0][0], "endIndex": P[-1][1]}})
            continue
        if len(P) > 1:
            b.add(deleteText={"objectId": bid, "textRange": {"type": "FIXED_RANGE",
                              "startIndex": P[0][1] - 1, "endIndex": P[-1][1] - 1}})
        if P[0][1] - 1 > P[0][0]:
            b.add(deleteText={"objectId": bid, "textRange": {"type": "FIXED_RANGE",
                              "startIndex": P[0][0], "endIndex": P[0][1] - 1}})
        b.add(insertText={"objectId": bid, "insertionIndex": P[0][0], "text": "\n".join(mapping[gi])})


# ---------------------------------------------------------------- right panel: table / charts
def table_requests(b, k, page, rows):
    T, P = CFG["table"], CFG["panel"]
    nc = max(len(r) for r in rows)
    rows = [r + [""] * (nc - len(r)) for r in rows]
    nat = [max(len(r[c]) for r in rows) * T["char_w"] + T["pad"] for c in range(nc)]
    scale = min(1.0, P["w"] / sum(nat)) if sum(nat) > P["w"] else 1.0
    widths = [w * scale for w in nat]
    if sum(widths) < P["w"]:  # spread spare width over the columns
        widths = [w + (P["w"] - sum(widths)) / nc for w in widths]
    x, tid = P["x"] + P["w"] - sum(widths), k + "_tb"
    b.add(createTable={"objectId": tid, "rows": len(rows), "columns": nc,
                       "elementProperties": props(page, x, P["y"], sum(widths), T["row_h"] * len(rows))})
    for c, w in enumerate(widths):
        b.add(updateTableColumnProperties={"objectId": tid, "columnIndices": [c], "fields": "columnWidth",
                                           "tableColumnProperties": {"columnWidth": {"magnitude": emu(w), "unit": "EMU"}}})
    numeric = [all(re.fullmatch(r"[-+~≈]?[\d.,]+%?[kMBx×]?", r[c] or "0") for r in rows[1:]) for c in range(nc)]
    for ri, r in enumerate(rows):
        for c, v in enumerate(r):
            if not v:
                continue
            loc = {"rowIndex": ri, "columnIndex": c}
            b.add(insertText={"objectId": tid, "cellLocation": loc, "text": v})
            b.add(updateTextStyle={"objectId": tid, "cellLocation": loc, "textRange": {"type": "ALL"},
                                   "style": {"fontSize": {"magnitude": 14, "unit": "PT"}, "bold": ri == 0 or c == 0},
                                   "fields": "fontSize,bold"})
            if numeric[c] and c > 0:
                b.add(updateParagraphStyle={"objectId": tid, "cellLocation": loc, "textRange": {"type": "ALL"},
                                            "style": {"alignment": "END"}, "fields": "alignment"})
    return x, P["y"] + T["row_h"] * len(rows)


def legend(b, k, page, names, x, y, w):
    C, step = CFG, w / max(len(names), 1)
    for i, n in enumerate(names):
        col = C["palette"][i % len(C["palette"])]
        b.rect(f"{k}_lg{i}s", page, x + i * step, y + 0.08, 0.13, 0.13, col)
        b.textbox(f"{k}_lg{i}", page, x + i * step + 0.13, y - 0.02, step - 0.15, 0.3, n, C["chart"]["label_pt"])


def bar_chart(b, k, page, spec):
    C, P = CFG, CFG["panel"]
    cats, series = spec["categories"], spec["series"]
    names, h = list(series), spec.get("h", C["chart"]["h"])
    x0, y0, w = P["x"], P["y"], P["w"]
    top = y0 + (0.35 if len(names) > 1 else 0) + 0.3
    base = y0 + h - 0.4
    vmax = spec.get("ymax") or max(max(v) for v in series.values()) * 1.1
    if len(names) > 1:
        legend(b, k, page, names, x0, y0, w)
    gw = w / len(cats)
    bw = gw * 0.72 / len(names)
    for ci, cat in enumerate(cats):
        gx = x0 + ci * gw + gw * 0.14
        for si, n in enumerate(names):
            v = series[n][ci]
            bh = max((base - top) * v / vmax, 0.01)
            bx = gx + si * bw
            b.rect(f"{k}_b{ci}_{si}", page, bx + 0.02, base - bh, bw - 0.04, bh, C["palette"][si % len(C["palette"])])
            lw = max(bw, 0.6)
            b.textbox(f"{k}_v{ci}_{si}", page, bx + bw / 2 - lw / 2, base - bh - 0.3, lw, 0.28, fmt(v, spec),
                      C["chart"]["label_pt"], align="CENTER")
        b.textbox(f"{k}_c{ci}", page, x0 + ci * gw, base + 0.03, gw, 0.35, str(cat), C["chart"]["label_pt"], align="CENTER")
    b.line(f"{k}_ax", page, x0, base, x0 + w, base, C["axis"], 1)
    return x0, y0 + h


def line_chart(b, k, page, spec):
    import math
    C, P = CFG, CFG["panel"]
    xs, series = spec["x"], spec["series"]
    names, h = list(series), spec.get("h", C["chart"]["h"])
    x0, y0, w = P["x"], P["y"], P["w"]
    left, right = x0 + 0.5, x0 + w - spec.get("label_w", 1.1)
    top, base = y0 + 0.15, y0 + h - 0.4
    allv = [v for s in series.values() for v in s if v is not None]
    lo, hi = spec.get("ymin", min(0, min(allv))), spec.get("ymax", max(allv) * 1.1)
    tx = (lambda v: math.log10(v)) if spec.get("xlog") else (lambda v: v)
    if all(isinstance(v, (int, float)) for v in xs):
        a, z = tx(min(xs)), tx(max(xs))
        px = [left + (right - left) * ((tx(v) - a) / (z - a) if z != a else 0.5) for v in xs]
    else:  # categorical x
        px = [left + (right - left) * (i / max(len(xs) - 1, 1)) for i in range(len(xs))]
    py = lambda v: base - (base - top) * (v - lo) / (hi - lo)
    b.line(f"{k}_ax", page, left, base, right, base, C["axis"], 1)
    for ti, tv in enumerate([lo, (lo + hi) / 2, hi]):
        b.textbox(f"{k}_y{ti}", page, x0, py(tv) - 0.14, 0.48, 0.28, fmt(tv, spec), C["chart"]["tick_pt"], C["muted"], "END")
    for xi, xv in enumerate(xs):
        b.textbox(f"{k}_x{xi}", page, px[xi] - 0.35, base + 0.03, 0.7, 0.28, str(xv), C["chart"]["tick_pt"], C["muted"], "CENTER")
    for si, n in enumerate(names):
        col, ys = C["palette"][si % len(C["palette"])], series[n]
        pts = [(px[i], py(v)) for i, v in enumerate(ys) if v is not None]
        for pi in range(len(pts) - 1):
            b.line(f"{k}_l{si}_{pi}", page, *pts[pi], *pts[pi + 1], col, 2)
        for pi, (xx, yy) in enumerate(pts):
            b.rect(f"{k}_m{si}_{pi}", page, xx - 0.045, yy - 0.045, 0.09, 0.09, col, "ELLIPSE")
        b.textbox(f"{k}_n{si}", page, pts[-1][0] + 0.06, pts[-1][1] - 0.14, x0 + w - pts[-1][0] - 0.06, 0.28, n,
                  C["chart"]["label_pt"], col, bold=True)
    if spec.get("xlabel"):
        b.textbox(f"{k}_xl", page, left, base + 0.27, right - left, 0.26, spec["xlabel"], C["chart"]["tick_pt"], C["muted"], "CENTER")
    return x0, y0 + h + (0.25 if spec.get("xlabel") else 0)


# ---------------------------------------------------------------- build
def slide_key(s):
    if s["id"]:
        return s["id"]
    return "cl" + hashlib.sha1(s["title"].encode()).hexdigest()[:8]


def build(spec_path, at=None, replace=False):
    st = json.load(open(STATE))
    T = CFG["template"]
    slides = parse_spec(spec_path)
    at = st["end_idx"] if at is None else (len(st["ids"]) if at == "end" else int(at))
    b = Batch()
    for i, s in enumerate(slides):
        k = slide_key(s)
        if k in st["ids"]:
            if not replace:
                sys.exit(f"slide {k} ({s['title']}) already exists; pass --replace to rebuild it")
            b.add(deleteObject={"objectId": k})
            at -= st["ids"].index(k) < at
        tid, bid = k + "_t", k + "_bd"
        b.add(duplicateObject={"objectId": T["slide"], "objectIds": {T["slide"]: k, T["title"]: tid, T["body"]: bid}})
        b.add(updateSlidesPosition={"slideObjectIds": [k], "insertionIndex": at + i})
        b.add(deleteText={"objectId": tid, "textRange": {"type": "ALL"}})
        b.add(insertText={"objectId": tid, "text": CFG["title_prefix"] + s["title"]})
        body_requests(b, bid, s["bullets"], st)
        panel = bool(s["table"] or s["chart"])
        bw = CFG["body"]["w_half" if panel else "w_full"]
        b.add(updatePageElementTransform={"objectId": bid, "applyMode": "ABSOLUTE", "transform": {
            "scaleX": emu(bw) / st["body_base"][0], "scaleY": emu(CFG["body"]["h"]) / st["body_base"][1],
            "shearX": 0, "shearY": 0, "translateX": st["body_xy"][0], "translateY": st["body_xy"][1], "unit": "EMU"}})
        if s["table"]:
            cx, cy = table_requests(b, k, k, s["table"])
        elif s["chart"]:
            kind = s["chart"].get("type", "bar")
            cx, cy = (bar_chart if kind == "bar" else line_chart)(b, k + "c", k, s["chart"])
        if panel and s["caption"]:
            P = CFG["panel"]
            b.textbox(k + "_cap", k, P["x"], cy + 0.1, P["w"], max(0.3, P["bottom"] - cy - 0.1), s["caption"], 11, CFG["muted"])
    print("REV", st["rev"])
    print(json.dumps(b.r, ensure_ascii=False, separators=(",", ":")))
    print(f"# {len(slides)} slide(s), {len(b.r)} requests, insert at {at}", file=sys.stderr)


# ---------------------------------------------------------------- thumbs
def thumbs(urls, out=None, width=640):
    import io
    import urllib.request
    from PIL import Image
    ims = []
    for u in urls:
        im = Image.open(io.BytesIO(urllib.request.urlopen(u, timeout=30).read())).convert("RGB")
        ims.append(im.resize((width, int(im.height * width / im.width))))
    sheet = Image.new("RGB", (width, sum(i.height for i in ims) + 6 * (len(ims) - 1)), "black")
    yy = 0
    for im in ims:
        sheet.paste(im, (0, yy))
        yy += im.height + 6
    out = out or os.path.join(os.path.dirname(STATE), "thumbs.png")
    sheet.save(out)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("locate"); p.add_argument("read_json")
    p = sub.add_parser("build"); p.add_argument("spec"); p.add_argument("--at"); p.add_argument("--replace", action="store_true")
    p = sub.add_parser("thumbs"); p.add_argument("urls", nargs="+"); p.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "locate":
        locate(a.read_json)
    elif a.cmd == "build":
        build(a.spec, a.at, a.replace)
    else:
        thumbs(a.urls, a.out)


if __name__ == "__main__":
    main()
