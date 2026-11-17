"""Add / replace the "Auto-research loop" part of the FLOP-drop hub page (idempotent).

    python3 build_autoresearch_tab.py hub.html [--state loop_state.json]

v3 (2026-11-17, laid out like the PAIL Workbench "Autoresearch" tab: stat strip, one hero chart, then section bars with
counts, ideas as one-line verdicts grouped by outcome). The hub is a grouped page: each `<div class="part" id=...>` declares
`data-group`, `data-order`, `data-short` (and optionally `data-badge`) and the page builds its tabs from them at run time, so
this builder never touches the navigation. It owns two marker-delimited blocks:

    <!-- AR:PART:START -->   <div class="part" id="autoresearch" ...> ... </div>   <!-- AR:PART:END -->
    <!-- AR:SCRIPT:START -->  <script> ...state JSON + renderer... </script>        <!-- AR:SCRIPT:END -->

Re-running replaces both blocks in place. First run on a page without them: the part goes before the `provenance` part, the
script after the last `</script>`. Legacy v1 blocks (<!-- AUTORESEARCH:START --> ... <!-- AUTORESEARCH:END -->) are stripped.
The overview part's `#ovLive` box is filled by the same script.

Reads the same `loop_state.json` keys as v1/v2: status, updated, setup, log, ideas, best_by_hour, depth, board, curves,
validation, pilot. Board rows map to ideas through `board[i].idea` (the idea family, set from hub_tab/arms.json by
make_loop_state.py), then by the row's `tag` (tag == idea id, or a name in ALIAS below), then by `arm` == idea id, then, when a
family has exactly one idea, to that idea. A row may carry `idea_id` to pin it. Rows that match no idea are listed under
their family as "no queue entry".
"""
import json
import os
import re
import sys

args = sys.argv[1:]
state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loop_state.json")
if "--state" in args:
    i = args.index("--state")
    state_path = args[i + 1]
    del args[i:i + 2]
H = args[0]
h = open(H, encoding="utf-8").read()
STATE = json.load(open(state_path, encoding="utf-8"))
# plain-language implementation notes per idea (make_idea_impl.py writes idea_impl.json); a `how` the loop wrote itself wins
_impl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idea_impl.json")
if os.path.exists(_impl):
    _H = json.load(open(_impl, encoding="utf-8"))
    for _i in STATE.get("ideas", []):
        if not _i.get("how") and _i["id"] in _H:
            _i["how"] = _H[_i["id"]]
    _miss = [_i["id"] for _i in STATE.get("ideas", []) if not _i.get("how")]
    if _miss:
        print("note: no implementation notes for ideas:", ", ".join(_miss), "(add them to make_idea_impl.py)")

# ---- per-context-length speedups (computed here from the per-length held-out CE curves in `curves[i].by`) -------------------
# The loop publishes one speedup per arm and level, on the mean CE over all lengths (`board[i].x`). This adds `board[i].xm[m]` for
# m in s4k / s8k / s16k (CE on the 4k / 8k / 16k held-out sets) and w (weighted CE, WEIGHTS), by the same method as score.py: the
# FLOPs the cheapest dense reference needs to reach its own step-k loss, divided by the arm's FLOPs to reach that loss, for
# k = the reference's probe steps. Reference pool = the `v2` tag's reference curves (this reproduces `board[i].x` exactly).
LENSETS = [("s4k", "set/suite4k", "4k", 0.2), ("s8k", "set/suite8k", "8k", 0.3), ("s16k", "set/suite16k", "16k", 0.5)]


def _reach(pf, ls, loss):
    run, m = [], float("inf")
    for v in ls:
        m = min(m, v)
        run.append(m)
    if loss >= run[0]:
        return 0.0
    for i in range(1, len(run)):
        if run[i] <= loss:
            if run[i - 1] == run[i]:
                return pf[i]
            w = (run[i - 1] - loss) / (run[i - 1] - run[i])
            return pf[i - 1] + w * (pf[i] - pf[i - 1])
    return None


def _series(c, m):
    if m == "all":
        return [p[2] for p in c["pts"]]
    by = c.get("by") or {}
    if m == "w":
        if not all(k in by and len(by[k]) == len(c["pts"]) for _, k, _, _ in LENSETS):
            return None
        return [sum(w * by[k][i] for _, k, _, w in LENSETS) for i in range(len(c["pts"]))]
    k = {a: b for a, b, _, _ in LENSETS}[m]
    return by.get(k) if by.get(k) and len(by[k]) == len(c["pts"]) else None


def _xlevels(c, refs, m):
    sr = [x for x in refs if x["seed"] == c["seed"]] or refs
    sr = [x for x in sr if _series(x, m) is not None]
    mine = _series(c, m)
    if not sr or mine is None:
        return None
    best = min(sr, key=lambda x: _series(x, m)[-1])
    sb, out, mpf = _series(best, m), {}, [p[1] for p in c["pts"]]
    for i, p in enumerate(best["pts"][1:], 1):
        lvl = min(sb[: i + 1])
        need = [n for n in (_reach([q[1] for q in x["pts"]], _series(x, m), lvl) for x in sr) if n is not None]
        mn = _reach(mpf, mine, lvl) if need else None
        out[str(p[0])] = (min(need) / mn) if need and mn else None
    return out


def _bestx(d):
    v = [q for q in (d or {}).values() if q is not None]
    return max(v) if v else None


def add_length_metrics(st):
    curves = st.get("curves") or []
    refs = [c for c in curves if c.get("is_ref") and c.get("tag") == "v2"] or [c for c in curves if c.get("is_ref")]
    ok = bad = 0
    for b in st.get("board") or []:
        if b.get("is_ref"):
            continue
        cs = [c for c in curves if c["tag"] == b["tag"] and c["arm"] == b["arm"]]
        b["xm"] = {}
        for m in ["all", "w"] + [a for a, _, _, _ in LENSETS]:
            per = [_xlevels(c, refs, m) for c in cs] if cs and refs else []
            if not per or any(q is None for q in per):
                continue
            lv = sorted({k for q in per for k in q}, key=int)
            b["xm"][m] = {k: (sum(q[k] for q in per) / len(per) if all(q.get(k) for q in per) else None) for k in lv}
        for k, v in (b.get("x") or {}).items():          # self-check against the loop's own speedups
            mv = (b["xm"].get("all") or {}).get(k)
            if (v is None) == (mv is None) and (v is None or abs(v - mv) <= 0.02 * max(1.0, v)):
                ok += 1
            else:
                bad += 1
        b["xm"].pop("all", None)
    for row in st.get("best_by_hour") or []:             # resolve each hourly entry to its board arm, when it can be told
        arm, row["xm"] = row.get("arm", ""), {}
        cand = [b for b in st["board"] if b["arm"] == arm and not b.get("is_ref")]
        if not cand:
            mm = re.search(r"\(([A-Za-z0-9_.-]+)\)\s*$", arm)
            cand = [b for b in st["board"] if mm and b["arm"] == mm.group(1) and not b.get("is_ref")]
        cand = [b for b in cand if _bestx(b.get("x")) is not None and row.get("x") and "full epoch" not in arm]   # full-epoch results come from depth runs, not the board
        if cand:
            b = min(cand, key=lambda b: abs(_bestx(b["x"]) - row["x"]))
            if abs(_bestx(b["x"]) - row["x"]) <= 0.15 * row["x"]:
                row["xm"] = {m: _bestx(v) for m, v in b["xm"].items()}
    st["meta"] = {"default": "w", "check": [ok, bad], "weights": {a: w for a, _, _, w in LENSETS},
                  "metrics": [{"id": "w", "label": "Weighted, 16k-heavy"}, {"id": "s16k", "label": "16k"}, {"id": "s8k", "label": "8k"},
                              {"id": "s4k", "label": "4k"}, {"id": "all", "label": "All lengths, mean CE (as scored)"}]}
    if ok + bad and bad / (ok + bad) > 0.05:
        print("WARNING: recomputed speedups match the loop's on only %d of %d levels" % (ok, ok + bad))


add_length_metrics(STATE)

# ---- where the gain comes from: like-for-like ratios between arms (decisions.json) -----------------------------------------
# Two views, both read off the same per-length held-out CE curves as the speedups above, but against another ARM instead of the
# dense-LoRA reference: ratio at a level = (PF the base arm needs to reach its own loss at probe step k) / (PF this arm needs).
#   ladder:  base = dense full fine-tune; each rung adds one decision on top of the previous one (LoRA, random drop, router, data).
#   groups:  base = the plain recipe; each item changes ONE decision (random tokens, keep fraction, LR, short rows, ...).
DEC_METRICS = ["w", "s16k", "s8k", "s4k", "all"]


def _ratios(curves, arm, base, m):
    per = []
    for c in [c for c in curves if c["tag"] == arm[0] and c["arm"] == arm[1]]:
        x = _xlevels(c, [base], m)
        if x is not None:
            per.append(x)
    if not per:
        return None
    lv = sorted({k for q in per for k in q}, key=int)
    return {k: (sum(q[k] for q in per) / len(per) if all(q.get(k) for q in per) else None) for k in lv}


def add_decisions(st):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decisions.json")
    if not os.path.exists(path):
        return
    D, curves = json.load(open(path, encoding="utf-8")), st.get("curves") or []
    pick = lambda t: [c for c in curves if c["tag"] == t[0] and c["arm"] == t[1]]
    lb, rb = pick(D["ladder_base"]), pick(D["base"])
    if not lb or not rb:
        print("note: decisions.json base arms missing from the state; skipping the breakdown")
        return
    lb, rb = lb[0], rb[0]
    out = {"base": D["base"], "ladder": [], "groups": []}
    boardce = {(b["tag"], b["arm"]): b for b in st.get("board") or []}
    for i, rung in enumerate(D["ladder"]):
        r = {}
        for m in DEC_METRICS:
            alts = [_ratios(curves, a, lb, m) for a in rung["alts"]]
            lv = sorted({k for a in alts if a for k in a}, key=int)
            r[m] = {k: max([a[k] for a in alts if a and a.get(k)] or [None]) for k in lv}
            if i == 0:
                r[m] = {k: 1.0 for k in lv}
        out["ladder"].append({"label": rung["label"], "what": rung["what"], "arms": ["/".join(a) for a in rung["alts"]], "r": r})
    for g in D["groups"]:
        items = []
        for it in g["items"]:
            b = boardce.get(tuple(it["arm"])) or {}
            items.append({"label": it["label"], "arm": "/".join(it["arm"]), "steps": b.get("steps"), "ce": b.get("ce_last"),
                          "seeds": len(b.get("seeds") or []) or None, "r": {m: _ratios(curves, it["arm"], rb, m) for m in DEC_METRICS}})
        out["groups"].append({"name": g["name"], "items": items})
    st["dec"] = out


add_decisions(STATE)

PS, PE = "<!-- AR:PART:START -->", "<!-- AR:PART:END -->"
SS, SE = "<!-- AR:SCRIPT:START -->", "<!-- AR:SCRIPT:END -->"


def bar(id_, title, body, opened=False, n_id=None):
    return ('<details class="sec"%s id="%s-sec"><summary><h2 id="%s">%s</h2><span class="n"%s></span></summary><div class="secbody">%s</div></details>'
            % (" open" if opened else "", id_, id_, title, (' id="%s"' % n_id) if n_id else "", body))


PART = PS + '''
<div class="part" id="autoresearch" data-group="g-tried" data-order="1" data-short="Auto-research loop" data-badge="live">
<style>
.pill { display: inline-block; font: 600 .66rem/1 var(--mono); letter-spacing: .05em; text-transform: uppercase; border: 1px solid var(--muted); color: var(--muted); border-radius: 999px; padding: 3px 7px; white-space: nowrap; vertical-align: 1px; }
.pill.s-running { border-color: var(--c1); color: var(--c1); } .pill.s-done { border-color: var(--c3); color: var(--c3); } .pill.s-failed { border-color: var(--c2); color: var(--c2); }
.tile { display: grid; gap: 4px; align-content: start; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 14px; min-width: 0; }
.tile b { font: 600 1.5rem/1.1 var(--display); font-variant-numeric: tabular-nums; } .tile span { font-size: .8rem; line-height: 1.4; color: var(--muted); }
.aropt { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; } .aropt > span:first-child { font: 500 .68rem/1 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-right: 4px; }
.aropt button { font: 500 .78rem/1 var(--body); padding: 6px 10px; border-radius: 999px; border: 1px solid var(--rule); background: var(--surface); color: var(--fg); cursor: pointer; }
.aropt button[aria-pressed="true"] { background: var(--fg); color: var(--bg); border-color: var(--fg); }
.aropt select { font: .85rem var(--body); padding: 6px 8px; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); color: var(--fg); max-width: 100%; }
/* every idea at a glance: one bar per idea, each folds open to a plain-language implementation note */
.glance { display: grid; gap: 0; font-size: .82rem; max-width: 980px; }
.glance details.gi { border-top: 1px solid var(--rule); }
.glance details.gi > summary { list-style: none; cursor: pointer; display: grid; grid-template-columns: minmax(180px, 42%) minmax(0, 1fr); gap: 4px 14px; align-items: center; padding: 7px 4px; }
.glance details.gi > summary::-webkit-details-marker { display: none; }
.glance details.gi > summary:hover { background: color-mix(in srgb, var(--rule) 30%, transparent); }
.glance details.gi > summary:focus-visible { outline: 2px solid var(--c2); outline-offset: -2px; }
.glance .gl { line-height: 1.3; min-width: 0; } .glance .gl b { font-size: .88rem; display: block; }
.glance .gl b::before { content: "▸"; display: inline-block; width: 1em; color: var(--muted); transition: transform .12s; } .glance details[open] .gl b::before { transform: rotate(90deg); }
.glance .gl span.gd { display: block; color: var(--muted); font-size: .76rem; line-height: 1.35; padding-left: 1em; }
.glance .gt { position: relative; height: 18px; background: color-mix(in srgb, var(--rule) 45%, transparent); }
.glance .gb { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 0 2px 2px 0; background: var(--muted); }
.glance .gb.s-running { background: var(--c1); } .glance .gb.s-done { background: var(--c3); } .glance .gb.s-dropped { background: var(--muted); opacity: .75; }
.glance .gv { position: absolute; top: 0; line-height: 18px; font: .72rem var(--mono); white-space: nowrap; padding-left: 5px; color: var(--fg); }
.glance .gr { position: absolute; top: -2px; bottom: -2px; border-left: 2px dashed var(--dense); opacity: .75; }
.glance .gax { display: grid; grid-template-columns: minmax(180px, 42%) minmax(0, 1fr); gap: 0 14px; padding: 4px 4px 0; } .glance .gax > div { position: relative; height: 14px; font: .68rem var(--mono); color: var(--muted); } .glance .gax span { position: absolute; transform: translateX(-50%); }
@media (max-width: 640px) { .glance details.gi > summary, .glance .gax { grid-template-columns: minmax(0, 1fr); } .glance .gax > div:first-child { display: none; } }
/* where the gain comes from */
.dec { display: grid; grid-template-columns: minmax(190px, 40%) minmax(0, 1fr); gap: 2px 14px; align-items: center; font-size: .82rem; max-width: 980px; }
.dec .dl { line-height: 1.3; min-width: 0; text-align: right; } .dec .dl small { display: block; color: var(--muted); font-size: .72rem; }
.dec .dh { grid-column: 1 / -1; margin: 12px 0 2px; font: 600 .72rem/1 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--rule); padding-bottom: 4px; }
.dec .dt { position: relative; height: 22px; background: color-mix(in srgb, var(--rule) 35%, transparent); }
.dec .db { position: absolute; top: 3px; bottom: 3px; background: var(--c3); } .dec .db.dn { background: var(--c1); } .dec .db.sh { opacity: .45; }
.dec .dw { position: absolute; top: 10px; height: 2px; background: var(--fg); opacity: .55; } .dec .dw::before, .dec .dw::after { content: ""; position: absolute; top: -3px; height: 8px; width: 1px; background: var(--fg); } .dec .dw::after { right: 0; }
.dec .d1 { position: absolute; top: -2px; bottom: -2px; border-left: 2px dashed var(--dense); opacity: .8; }
.dec .dv.nr { background: var(--bg); }
.dec .dv { position: absolute; top: 0; line-height: 22px; font: .72rem var(--mono); white-space: nowrap; padding: 0 5px; }
.dec .dax { grid-column: 2; position: relative; height: 14px; font: .68rem var(--mono); color: var(--muted); } .dec .dax span { position: absolute; transform: translateX(-50%); }
@media (max-width: 640px) { .dec { grid-template-columns: minmax(0, 1fr); } .dec .dl { text-align: left; } .dec .dax { grid-column: 1; } }
/* implementation note: steps, flags, diagrams */
.impl { display: grid; gap: 8px; padding: 4px 6px 12px 22px; min-width: 0; } .impl p { margin: 0; max-width: 100ch; font-size: .86rem; }
.impl ol { margin: 0; padding-left: 1.25em; max-width: 100ch; font-size: .84rem; line-height: 1.45; } .impl ol li { margin: 2px 0; }
.impl .iflags { font: .74rem/1.4 var(--mono); color: var(--muted); overflow-wrap: anywhere; } .impl .iflags b { font-weight: 600; color: var(--fg); }
.impl .iresult { font-size: .84rem; border-left: 3px solid var(--rule); padding: 2px 0 2px 10px; max-width: 100ch; }
.impl .idiag { display: grid; gap: 6px; min-width: 0; } .impl .idiag h5 { margin: 0; font: 600 .74rem var(--body); color: var(--muted); }
.impl .idiag svg { width: 100%; max-width: 560px; height: auto; display: block; }
.dg-g { fill: var(--c2); } .dg-n { fill: var(--c6); opacity: .5; } .dg-f { fill: var(--c4); opacity: .55; } .dg-c { fill: var(--c5); opacity: .75; }
.dg-ghost { fill: none; stroke: var(--rule); stroke-dasharray: 2 2; } .dg-t { font: 9px var(--mono); fill: var(--muted); } .dg-m { font: 600 9px var(--body); fill: var(--fg); }
.dg-att { fill: var(--dense); } .dg-line { stroke: var(--muted); stroke-width: 1; }
.dlegend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: .72rem; color: var(--muted); } .sw-g { background: var(--c2); } .sw-n { background: var(--c6); opacity: .5; } .sw-f { background: var(--c4); opacity: .55; } .sw-c { background: var(--c5); opacity: .75; }
.dlegend i { display: inline-block; width: 10px; height: 10px; vertical-align: -1px; margin-right: 4px; border-radius: 2px; }
.rowd { display: flex; gap: 2px; max-width: 640px; min-width: 0; } .rowd > div { min-width: 0; font: .66rem/1.2 var(--mono); padding: 5px 3px; text-align: center; border-radius: 3px; overflow: hidden; color: var(--fg); background: color-mix(in srgb, var(--c6) 25%, transparent); white-space: nowrap; text-overflow: ellipsis; }
.rowd .k-p { background: color-mix(in srgb, var(--c6) 40%, transparent); } .rowd .k-G { background: color-mix(in srgb, var(--c3) 55%, transparent); } .rowd .k-d { background: color-mix(in srgb, var(--c2) 28%, transparent); }
.rowd .k-q { background: color-mix(in srgb, var(--c1) 50%, transparent); } .rowd .k-a { background: color-mix(in srgb, var(--c7) 40%, transparent); } .rowd .k-s { background: color-mix(in srgb, var(--c4) 40%, transparent); }
.rowd .k-x { background: none; border: 1px dashed var(--rule); color: var(--muted); } .rowd .k-gap { background: none; border: 1px dotted var(--muted); color: var(--muted); }
.rowd .k-t { background: color-mix(in srgb, var(--c2) 35%, transparent); } .rowd .k-u { background: color-mix(in srgb, var(--c3) 40%, transparent); } .rowd .k-v { background: color-mix(in srgb, var(--c5) 45%, transparent); }
/* idea lists, PAIL style: one line per idea, details folded */
details.qg { border: 0; background: transparent; } details.qg > summary { padding: 6px 0; font-weight: 600; } details.qg[open] > summary { border-bottom: 0; }
ul.ql { list-style: none; margin: 0; padding: 0; display: grid; gap: 0; }
ul.ql > li { max-width: none; margin: 0; padding: 9px 0; border-top: 1px solid var(--rule); display: grid; gap: 3px; }
ul.ql .t { display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: baseline; } ul.ql .t b { font-size: .95rem; }
ul.ql .tag { font: 500 .68rem/1 var(--mono); color: var(--muted); border: 1px solid var(--rule); border-radius: 4px; padding: 2px 6px; }
ul.ql .rd { font: .78rem var(--mono); color: var(--muted); margin-left: auto; }
ul.ql .res { font-size: .88rem; max-width: 110ch; }
ul.ql details.more { border: 0; background: transparent; margin: 2px 0 0; } ul.ql details.more > summary { font-size: .8rem; color: var(--muted); padding: 2px 0; }
ul.ql details.more[open] > summary { border-bottom: 0; }
.ideabody { display: grid; gap: 8px; padding: 6px 0 4px; min-width: 0; } .ideabody p { margin: 0; max-width: 100ch; font-size: .86rem; }
.arlog td:first-child { white-space: nowrap; } .arwide td:last-child { min-width: 22rem; } .arwide td:nth-child(2) { min-width: 11rem; }
</style>
<div class="parthead"><span class="eyebrow">Tried · was tab 8d · live</span><h2 class="ptitle">Auto-research loop: FLOPs to reach the dense loss at 4k-16k</h2><p class="verdict" id="arVerdict"></p><p class="note" id="arUpdated"></p></div>
<div class="aropt" id="arMetric" role="group" aria-label="Context length the gains are measured at"></div>
<p class="note" id="arMetricNote"></p>
<div class="tiles" id="arStrip"></div>
''' + bar("ar-progress", "Best so far, over time", '''
<div id="arBestNote" class="note"></div>
<div class="figs"><figure style="grid-column: 1 / -1"><div class="chart" style="height:230px"><canvas id="arBestChart" aria-label="best FLOP speedup over time"></canvas></div><figcaption>Best FLOP speedup over the dense reference found so far, by time of the loop (hover a point for the method).</figcaption></figure></div>
<details class="more" id="arBestBox"><summary>Every new best, as a table <span class="mh" id="arBestN"></span></summary><div id="arBest"></div></details>
<details class="more" id="arStatusBox"><summary>Full status text <span class="mh">from the loop</span></summary><p class="note" id="arStatus" style="margin:10px 12px 12px"></p></details>''', True) + bar("ar-glance", "Every idea at a glance", '''
<p class="note">Best FLOP speedup any arm of the idea reached, against dense LoRA at the same held-out loss (at a level the arm actually reached; the dashed line is the plain recipe). Ideas with no bar reached no comparison level. Colour is the idea's status. ⚠ eval_size 184, below 500: differences under about two paired SEs are noise. The bar is the best over every arm of the idea and every reference level, so it favours a lucky shallow level; the next section compares like for like. <b>Click an idea</b> to see what it does, how it is implemented (with a diagram of which tokens or rows are used) and its result.</p>
<div class="aropt"><span>Notes</span><button type="button" id="arGOpenAll">Open all</button><button type="button" id="arGCloseAll">Close all</button></div>
<div id="arGlance" class="glance"></div>''', True) + bar("ar-decisions", "Where the gain comes from", '''
<p class="note" id="arDecNote"></p>
<h3 style="margin-top:6px">1 · Build-up: what each decision adds, stacked</h3>
<p class="note" id="arLadNote"></p>
<div id="arLadder" class="dec"></div>
<h3 style="margin-top:22px">2 · One change at a time, against the plain recipe</h3>
<p class="note" id="arAblNote"></p>
<div id="arAbl" class="dec"></div>''', True) + bar("ar-ideas", "Ideas, with their verdicts", '''
<div class="aropt"><span>Group by</span><button type="button" id="arGO" aria-pressed="true">Outcome</button><button type="button" id="arGF" aria-pressed="false">Family</button></div>
<p class="note" id="arFamNote">One line per idea: its verdict first. "Details and arms" opens what was tried and the arms (scored runs) of that idea, with all their columns.</p>
<div id="arFam"></div>''', True, "arIdeasN") + bar("ar-depth", "FLOPs to reach a held-out loss", '''
<p class="note">Against the full-epoch dense-LoRA curve. Ratio above 1 means fewer training FLOPs than dense LoRA needs for the same held-out loss. ⚠ eval_size 184 held-out rows (below 500: differences under about two paired SEs are noise); one seed per row unless the name says otherwise.</p><div id="arDepth"></div>''', False, "arDepthN") + bar("ar-curves", "Loss vs training FLOPs", '''
<div class="aropt"><label for="arSel" style="font-size:.85rem">Also draw the arms of</label><select id="arSel"></select><span class="note" id="arSelNote"></span></div>
<div class="figs"><figure style="grid-column: 1 / -1"><div class="chart" style="height:380px"><canvas id="arCurves" aria-label="held-out loss vs training PFLOPs"></canvas></div><figcaption id="arCurvesCap">Held-out dense CE against cumulative training PFLOPs (log axis); step 0 is omitted. Reference runs are thick and dashed.</figcaption></figure></div>''') + bar("ar-validation", "Larger FLOP-matched runs", '<div id="arVal"></div>', False, "arValN") + bar("ar-scoring", "How it is scored", '<ul id="arSetup" style="margin:0"></ul>', False, "arSetupN") + bar("ar-log", "Log, newest first", '<div id="arLog"></div>', False, "arLogN") + bar("ar-pilot", "Earlier 8k pilot (old task set)", '<p class="note">Contradiction, nq, outlier, oolong at 8k, 50 steps, eval_size 256; ⚠ below 500. Kept for context.</p><div id="arPilot"></div>', False, "arPilotN") + '''
</div>
''' + PE + "\n"

JS = SS + r'''
<script>
(() => { const ST = __ARDATA__;
  const $ = id => document.getElementById(id); if (!$("arFam")) return;
  const el = (tag, txt, cls) => { const e = document.createElement(tag); if (txt != null) e.textContent = txt; if (cls) e.className = cls; return e; };
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const f = (v, d) => (v == null || v === "" || isNaN(v)) ? "" : Number(v).toFixed(d);
  const xs = v => v == null ? "not reached" : f(v, 1) + "×";
  const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
  const sentences = t => String(t || "").split(/(?<=[.!?])\s+(?=[A-Z])/);
  const lead = (t, n) => { const ss = sentences(t); let v = ss[0] || ""; for (let k = 1; k < ss.length && v.length < 70; k++) v += " " + ss[k]; return v.length > n ? v.slice(0, n).replace(/\s+\S*$/, "") + " …" : v; };
  const table = (head, rows, numCols, cls) => { const w = el("div", null, "tbl" + (cls ? " " + cls : "")), t = el("table"), tr = el("tr");
    head.forEach((x, i) => tr.appendChild(el("th", x, numCols.includes(i) ? "n" : ""))); t.appendChild(tr);
    rows.forEach(r => { const q = el("tr"); r.forEach((c, i) => { const td = el("td", null, numCols.includes(i) ? "n" : "");
      if (c && c.nodeType) td.appendChild(c); else td.textContent = c == null ? "" : c; q.appendChild(td); }); t.appendChild(q); });
    w.appendChild(t); return w; };
  const empty = (box, msg) => box.appendChild(el("p", msg, "note"));
  const two = (a, b) => { const s = el("span"); s.appendChild(document.createTextNode(a || "")); if (b) { s.appendChild(el("br")); s.appendChild(el("span", b, "note")); } return s; };
  const pill = s => el("span", s, "pill s-" + s);
  const fold = (summ, open, cls) => { const d = el("details", null, "more" + (cls ? " " + cls : "")); if (open) d.open = true; const sm = el("summary"); (Array.isArray(summ) ? summ : [summ]).forEach(n => sm.appendChild(n.nodeType ? n : document.createTextNode(n))); d.appendChild(sm); return d; };
  const mh = t => el("span", t, "mh");
  const setN = (id, t) => { const e = $(id); if (e) e.textContent = t; };

  /* header and verdict */
  const sents = sentences(ST.status);
  $("arVerdict").textContent = sents[0] || ""; $("arUpdated").textContent = ST.updated ? "updated " + ST.updated : "";
  if (sents.length > 1) $("arStatus").textContent = ST.status; else $("arStatusBox").style.display = "none";

  /* best by hour */
  const BH = ST.best_by_hour || [];
  if (!BH.length) { empty($("arBest"), "No 16k result yet"); $("arBestChart").closest("figure").style.display = "none"; } else {
    $("arBestN").textContent = BH.length + " rows"; }
  const MT = ST.meta || { default: "all", metrics: [{ id: "all", label: "All lengths, mean CE (as scored)" }], weights: {}, check: [0, 0] };
  let M = MT.default || "all"; const mlab = () => (MT.metrics.find(m => m.id === M) || {}).label || M;
  const xOf = r => M === "all" ? (r.x || {}) : ((r.xm || {})[M] || {});
  const bestOfRow = r => { const v = Object.values(xOf(r)).filter(q => q != null); return v.length ? Math.max(...v) : null; };
  const bhVal = r => M === "all" ? r.x : (r.xm || {})[M];
  const bhSeries = () => { let cur = null; return BH.map(r => { const v = bhVal(r); if (v != null && (cur == null || v > cur)) cur = v; return cur; }); };
  const last = BH[BH.length - 1];

  /* board rows -> ideas -> families */
  function HIDE(r) { return r.tag + "/" + r.arm === "hp/denselora-lr2e5" || r.tag === "hp/denselora-lr2e5" || r.arm === "hp/denselora-lr2e5"; }
  const ALIAS = { rowsbk: "rows", "goldrand-lam": "goldrand", grmix50: "goldrand", prefixl: "prefix", "ffnskip-lam": "ffnskip", "earlyloss-lam": "earlyloss", hp: "hparams", mq: "multiq" };
  const ideasAll = ST.ideas || [], byId = {}; ideasAll.forEach(i => { byId[i.id] = i; });
  const famIdeas = {}; ideasAll.forEach(i => { (famIdeas[i.family] = famIdeas[i.family] || []).push(i); });
  const FAMALIAS = { "short rows": "subproblem", "keep ladder": "recipe" };
  Object.keys(FAMALIAS).forEach(k => { if (byId[FAMALIAS[k]] && !famIdeas[k]) famIdeas[k] = [byId[FAMALIAS[k]]]; });
  const ideaOf = r => { if (r.idea_id && byId[r.idea_id]) return byId[r.idea_id];
    const t = ALIAS[r.tag] || r.tag, c = byId[t]; if (c && (!r.idea || c.family === r.idea)) return c;
    if (byId[r.arm] && (!r.idea || byId[r.arm].family === r.idea)) return byId[r.arm];
    const fi = famIdeas[r.idea]; if (fi && fi.length === 1) return fi[0]; return null; };
  const famOf = r => r.idea || (ideaOf(r) || {}).family || "unassigned";
  const rk = r => r.arm === "recipe" ? 0 : r.is_ref ? 1 : 2, ir = r => r.iso_rec == null ? -Infinity : r.iso_rec;
  const B = (ST.board || []).filter(r => !HIDE(r)).map((r, i) => [r, i]).sort((a, b) => rk(a[0]) - rk(b[0]) || ir(b[0]) - ir(a[0]) || a[1] - b[1]).map(x => x[0]);
  const lv = new Set(); B.forEach(r => Object.keys(r.x || {}).forEach(k => lv.add(k)));
  const levels = [...lv].sort((a, b) => a - b), deep = levels[levels.length - 1];
  const iso = r => r.iso == null ? "n/a" : (r.iso_lb ? ">= " : "") + f(r.iso, 1) + "×", isr = r => r.iso_rec == null ? "n/a" : f(r.iso_rec, 2) + "×";
  const bestx = bestOfRow;
  const armsTable = rows => { const head = ["method", "seeds", "steps", "held-out CE", "cost / step vs dense"].concat(levels.map(l => "× @ step " + l + (M === "all" ? "" : " (" + mlab() + ")")), ["× vs dense LoRA at own loss", "× vs recipe at own loss", "worst task @ step " + deep, "train s"]);
    const body = rows.map(r => { const ce = r.ce_last == null ? "" : f(r.ce_last, 3) + (r.d_vs_ref_se != null && !r.is_ref ? "  (Δ " + f(r.d_vs_ref, 3) + " ±" + f(r.d_vs_ref_se, 3) + ")" : "");
      return [two((r.is_ref ? "[ref] " : "") + r.arm, r.desc), (r.seeds || []).length, r.steps, ce, r.pf_per_step_vs_dense == null ? "" : f(r.pf_per_step_vs_dense, 2) + "×"]
        .concat(levels.map(l => xs(xOf(r)[l])), [iso(r), isr(r), xs((r.xw || {})[deep]), f(r.train_s, 0)]); });
    const nc = [1, 2, 3, 4].concat(levels.map((_, i) => 5 + i), [5 + levels.length, 6 + levels.length, 7 + levels.length, 8 + levels.length]);
    const t = table(head, body, nc); [...t.querySelectorAll("tr")].slice(1).forEach((tr, i) => { if (rows[i].is_ref) { tr.style.background = "var(--surface)"; tr.style.fontStyle = "italic"; } }); return t; };

  /* units = ideas, plus a pseudo-unit per family that has arms but no idea ("no queue entry") */
  const units = ideasAll.map(i => ({ key: i.id, idea: i, family: i.family, status: i.status, title: i.title, arms: [] }));
  const unitBy = {}; units.forEach(u => { unitBy[u.key] = u; });
  B.forEach(r => { const i = ideaOf(r); let u = i ? unitBy[i.id] : null;
    if (!u) { const k = "fam:" + famOf(r); u = unitBy[k] = unitBy[k] || (units.push({ key: k, idea: null, family: famOf(r), status: "done", title: famOf(r) === "unassigned" ? "Arms not tied to an idea" : cap(famOf(r)) + " (arms with no queue entry)", arms: [] }), units[units.length - 1]); }
    u.arms.push(r); });
  const calcBest = () => units.forEach(u => { const non = u.arms.filter(r => !r.is_ref).map(bestx).filter(v => v != null); u.best = non.length ? Math.max(...non) : null; });
  calcBest();
  const SR = { running: 0, done: 1, queued: 2, dropped: 3, failed: 4 }, OUT = { running: "Running", done: "Done: measured or kept", queued: "Queued", dropped: "Dropped: no gain", failed: "Failed" };
  const refUnit = unitBy.recipe; let refX = refUnit ? refUnit.best : null;

  /* stat strip */
  const cnt = {}; ideasAll.forEach(i => { cnt[i.status] = (cnt[i.status] || 0) + 1; });
  const cntTxt = Object.keys(cnt).sort((a, b) => (SR[a] ?? 9) - (SR[b] ?? 9)).map(s => cnt[s] + " " + s).join(" · ");
  const es = [...new Set(B.map(r => r.eval_size).filter(v => v != null))];
  const tile = (b, s) => { const t = el("div", null, "tile"); t.appendChild(el("b", b)); t.appendChild(el("span", s)); return t; };
  const strip = $("arStrip");
  const drawStrip = () => { strip.textContent = ""; const sr = bhSeries(), lv = sr.length ? sr[sr.length - 1] : null;
    if (last) strip.appendChild(tile(lv == null ? "n/a" : xs(lv), "best FLOP speedup so far vs dense LoRA, " + mlab() + (M === "all" ? " (" + last.arm + ")" : "")));
    strip.appendChild(tile(String(ideasAll.length), "ideas: " + cntTxt));
    strip.appendChild(tile(String(B.length), "arms scored" + (es.length ? " · ⚠ eval_size " + es.join(" / ") + " (below 500)" : "")));
    if (ST.updated) strip.appendChild(tile(ST.updated.replace(/^\d{4}-/, ""), "last update (Pacific time)")); };

  /* implementation notes: plain text, flags and a small diagram per idea (idea.how, from idea_impl.json) */
  const NS = "http://www.w3.org/2000/svg", ATTN = [3, 7, 11, 15, 19, 23, 27, 31];
  const sv = (tag, at, txt) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); if (txt != null) e.textContent = txt; return e; };
  const MODE = { g: "trained (gradient)", n: "forward-only, no gradient", f: "attention / GDN only, FFN skipped", c: "run in chunks" };
  const depthSvg = d => { const W = 520, L = 4, cw = 16, H = 64, top = (d.marks && d.marks.length) ? 18 : 12, base = top + H;
    const svg = sv("svg", { viewBox: "0 0 " + W + " " + (base + 26), role: "img", "aria-label": d.name });
    for (let b = 0; b < 32; b++) svg.appendChild(sv("rect", { x: L + b * cw, y: top, width: cw - 2, height: H, class: "dg-ghost" }));
    const bars = (segs, cls) => (segs || []).forEach(sg => { const h = Math.max(1.5, H * sg[2]);
      for (let b = sg[0]; b < sg[1]; b++) svg.appendChild(sv("rect", { x: L + b * cw, y: base - h, width: cw - 2, height: h, class: cls || "dg-" + sg[3] })); });
    bars(d.segs); bars(d.overlay, "dg-att");
    (d.segs || []).forEach(sg => { if (sg[1] - sg[0] >= 3) svg.appendChild(sv("text", { x: L + (sg[0] + sg[1]) * cw / 2 - 1, y: Math.max(9, base - H * sg[2] - 3), "text-anchor": "middle", class: "dg-t" }, Math.round(sg[2] * 100) + "%")); });
    ATTN.forEach(b => svg.appendChild(sv("path", { d: "M" + (L + b * cw + 1) + "," + (base + 9) + " l5,-7 l5,7 z", class: "dg-att" })));
    [0, 8, 16, 24, 31].forEach(b => svg.appendChild(sv("text", { x: b === 0 ? L : L + b * cw + 6, y: base + 23, "text-anchor": b === 0 ? "start" : "middle", class: "dg-t" }, b === 0 ? "block 0" : String(b))));
    (d.marks || []).forEach(m => { const x = L + m[0] * cw + 6, late = x + m[1].length * 5.2 > W - 4;
      svg.appendChild(sv("line", { x1: x, x2: x, y1: 12, y2: top + 2, class: "dg-line" }));
      svg.appendChild(sv("text", { x: x + (late ? 3 : -3), y: 9, "text-anchor": late ? "end" : "start", class: "dg-m" }, m[1])); });
    return svg; };
  const rowDiv = d => { const r = el("div", null, "rowd"); r.setAttribute("role", "img"); r.setAttribute("aria-label", d.name);
    d.segs.forEach(sg => { const c = el("div", sg[2] || (sg[1] === "x" ? "·" : ""), "k-" + sg[1]); c.style.flex = sg[0] + " 1 0"; if (sg[2]) c.title = sg[2]; r.appendChild(c); }); return r; };
  const diagrams = how => { const ds = how.diagrams || [], host = el("div", null, "idiag"), modes = new Set(); let depth = false;
    ds.forEach(d => { const h = el("div"); h.appendChild(el("h5", d.name));
      if (d.kind === "depth") { depth = true; (d.segs || []).forEach(sg => modes.add(sg[3])); h.appendChild(depthSvg(d)); } else h.appendChild(rowDiv(d));
      if (d.note) h.appendChild(el("p", d.note, "note")); host.appendChild(h); });
    if (depth) { const lg = el("div", null, "dlegend"); [...modes].forEach(m => { const s = el("span"), i = el("i"); i.className = "sw-" + m; s.appendChild(i); s.appendChild(document.createTextNode(MODE[m] || m)); lg.appendChild(s); });
      lg.appendChild(el("span", "▲ full-attention block (the rest are Gated DeltaNet) · dashed box = dense: every token")); host.appendChild(lg); }
    return host; };
  const implPanel = (u, withResult) => { const i = u.idea || {}, how = i.how, box = el("div", null, "impl");
    if (how) { if (how.does) { const p = el("p"); p.appendChild(el("b", "What it does: ")); p.appendChild(document.createTextNode(how.does)); box.appendChild(p); }
      if ((how.steps || []).length) { box.appendChild(el("p", "How it is implemented", "note")); const ol = el("ol"); how.steps.forEach(t => ol.appendChild(el("li", t))); box.appendChild(ol); }
      if ((how.diagrams || []).length) box.appendChild(diagrams(how));
      if (how.flags || how.branch) { const p = el("p", null, "iflags"); if (how.branch) { p.appendChild(el("b", "branch ")); p.appendChild(document.createTextNode(how.branch + (how.flags ? " · " : ""))); }
        if (how.flags) { p.appendChild(el("b", "flags ")); p.appendChild(document.createTextNode(how.flags)); } box.appendChild(p); } }
    else if (i.what) { const p = el("p"); p.appendChild(el("b", "What it does: ")); p.appendChild(document.createTextNode(i.what)); box.appendChild(p); }
    else box.appendChild(el("p", u.arms.length + " arm" + (u.arms.length === 1 ? "" : "s") + " of family " + u.family + " with no queue entry, so no description.", "note"));
    const bi = withResult && u.best != null ? bestInfo(u) : null;
    if (bi) { const r = el("div", null, "iresult"); r.appendChild(el("b", "What the bar is: "));
      r.appendChild(document.createTextNode("arm " + bi.r.arm + (bi.r.desc ? " (" + bi.r.desc + ")" : "") + " at the loss dense LoRA reached at step " + bi.k + ": " + f(bi.v, 2) + "×" +
        (bi.kd !== bi.k ? ". At the deepest level it reached (step " + bi.kd + ") the same arm is " + f(bi.vd, 2) + "×." : ". That is also its deepest level.") +
        " The bar takes the best over all " + u.arms.filter(q => !q.is_ref).length + " arms and levels, and the arm includes every decision of the recipe, not only this idea; see the next section for what each decision adds.")); box.appendChild(r); }
    if (withResult && i.result) { const r = el("div", null, "iresult"); r.appendChild(el("b", "Result so far: ")); r.appendChild(document.createTextNode(lead(i.result, 320)));
      const a = el("a", " full verdict and arms ↓"); a.href = "#ar-ideas"; r.appendChild(a); box.appendChild(r); }
    return box; };
  const bestInfo = u => { let best = null;
    u.arms.filter(r => !r.is_ref).forEach(r => { const o = xOf(r), ks = Object.keys(o).filter(k => o[k] != null).sort((a, b) => a - b); if (!ks.length) return;
      const kb = ks.reduce((a, k) => o[k] > o[a] ? k : a, ks[0]); if (!best || o[kb] > best.v) best = { r, k: kb, v: o[kb], kd: ks[ks.length - 1], vd: o[ks[ks.length - 1]] }; });
    return best; };
  const oneOf = u => { const i = u.idea || {}; return (i.how && i.how.one) || (i.what ? lead(i.what, 130) : ""); };

  /* every idea at a glance */
  const G = $("arGlance"), gnote = G.previousElementSibling.previousElementSibling, gbase = gnote.textContent;
  const drawGlance = () => { const wasOpen = new Set([...G.querySelectorAll("details.gi[open]")].map(d => d.dataset.k)); G.textContent = ""; const gu = units.filter(u => !(u.arms.length && u.arms.every(r => r.is_ref)));
  gu.sort((a, b) => (b.best == null ? -1 : b.best) - (a.best == null ? -1 : a.best));
  const mx = Math.max(4, ...gu.map(u => u.best || 0), refX || 0) * 1.06;
  gnote.textContent = gbase.replace(/ Measured at: .*$/, "") + " Measured at: " + mlab() + "." + (refX != null ? " Plain recipe: " + f(refX, 2) + "×." : "");
  gu.forEach(u => { const d = el("details", null, "gi"); d.dataset.k = u.key; if (wasOpen.has(u.key)) d.open = true; const sm = el("summary");
    const lab = el("div", null, "gl"); lab.appendChild(el("b", u.title)); lab.appendChild(el("span", oneOf(u), "gd")); sm.appendChild(lab);
    const t = el("div", null, "gt"); if (u.best != null) { const b = el("div", null, "gb s-" + u.status); b.style.width = (u.best / mx * 100) + "%"; t.appendChild(b);
      const na = u.arms.filter(r => !r.is_ref).length, pc = u.best / mx * 100, v = el("span", f(u.best, 2) + "× · " + na + (na === 1 ? " arm" : " arms"), "gv");
      if (pc > 55) { v.style.right = (100 - pc) + "%"; v.style.paddingRight = "5px"; v.style.color = "var(--bg)"; } else v.style.left = pc + "%"; t.appendChild(v); }
    else { const v = el("span", u.arms.length ? "no comparison level reached (" + u.arms.length + " arms)" : "no arms scored yet", "gv"); v.style.left = "0"; t.appendChild(v); }
    if (refX != null) { const r = el("div", null, "gr"); r.style.left = (refX / mx * 100) + "%"; t.appendChild(r); }
    sm.appendChild(t); d.appendChild(sm); d.appendChild(implPanel(u, true)); G.appendChild(d); });
  const ax = el("div", null, "gax"), axi = el("div"); ax.appendChild(el("div")); ax.appendChild(axi);
  [1, 2, 3, 4, 6, 8].filter(v => v < mx).forEach(v => { const s = el("span", v + "×"); s.style.left = (v / mx * 100) + "%"; axi.appendChild(s); }); G.insertBefore(ax, G.firstChild); };
  $("arGOpenAll").onclick = () => G.querySelectorAll("details.gi").forEach(d => { d.open = true; }); $("arGCloseAll").onclick = () => G.querySelectorAll("details.gi").forEach(d => { d.open = false; });

  /* where the gain comes from: ladder (dense -> best) and one-change-at-a-time against the recipe */
  const DEC = ST.dec, LAD = $("arLadder"), ABL = $("arAbl");
  const lpos = (v, lo, hi) => Math.max(0, Math.min(100, (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * 100));
  const keysOf = o => Object.keys(o || {}).filter(k => o[k] != null).sort((a, b) => a - b);
  const fx = v => (v >= 10 ? f(v, 0) : f(v, 2)) + "×";
  const axisRow = (host, lo, hi, ticks) => { host.appendChild(el("div")); const a = el("div", null, "dax");
    ticks.filter(v => v >= lo && v <= hi).forEach(v => { const sp = el("span", v + "×"); sp.style.left = lpos(v, lo, hi) + "%"; a.appendChild(sp); }); host.appendChild(a); };
  const track = (lo, hi) => { const t = el("div", null, "dt"), one = el("div", null, "d1"); one.style.left = lpos(1, lo, hi) + "%"; t.appendChild(one); return t; };
  const decLabel = (host, text, small) => { const l = el("div", null, "dl"); l.appendChild(document.createTextNode(text)); if (small) l.appendChild(el("small", small)); host.appendChild(l); };
  const decText = (t, txt, a, b, lo, hi) => { const v = el("span", txt, "dv"), right = lpos(b, lo, hi);
    if (right > 72) { v.style.right = (100 - lpos(a, lo, hi)) + "%"; } else v.style.left = right + "%"; t.appendChild(v); };
  const drawDecisions = () => { const box = $("arLadder").closest("details"); if (!DEC) { if (box) box.style.display = "none"; return; }
    $("arDecNote").textContent = "The bars above give each idea's best speedup over dense LoRA, and every arm there contains the whole recipe (LoRA, router, two cuts), so a bar mixes the decisions. These two charts take it apart with the same per-length held-out curves, measured at: " + mlab() + ". The dashed line is 1×.";
    LAD.textContent = ""; ABL.textContent = "";
    /* 1 ladder */
    const R = DEC.ladder.map(r => (r.r || {})[M] || {}), common = keysOf(R[0]).filter(k => +k >= 20 && R.every(o => o[k] != null)), nt = $("arLadNote");
    if (!common.length) nt.textContent = "Not enough per-length curves to stack these arms at " + mlab() + ".";
    else { const L = common[common.length - 1], cum = R.map(o => o[L]), mn = R.map(o => Math.min(...common.map(k => o[k]))), mx2 = R.map(o => Math.max(...common.map(k => o[k])));
      const hi = Math.max(...mx2) * 1.14, lo = Math.min(0.85, ...mn) * 0.9;
      nt.textContent = "Each row adds one decision to the row above. The bar spans from the previous row's speedup to this row's, over a dense full fine-tune at the same held-out loss, taken at the loss dense reached at its step " + L + " (the deepest level every row reaches). Whiskers: the range over the levels from step " + common[0] + " to " + L + "; a step that fits inside the whiskers is not a clear effect. One seed per row. In short: " + DEC.ladder.slice(1).map((rg, i) => rg.label.replace(/^\+ /, "") + " ×" + f(cum[i + 1] / cum[i], 2)).join(", ") + ", for " + fx(cum[cum.length - 1]) + " in all.";
      axisRow(LAD, lo, hi, [1, 2, 3, 4, 6, 8]);
      DEC.ladder.forEach((rg, i) => { decLabel(LAD, rg.label, rg.what); const t = track(lo, hi), prev = i ? cum[i - 1] : 1, cur = cum[i], a = Math.min(prev, cur), b = Math.max(prev, cur);
        const br = el("div", null, "db" + (cur < prev ? " dn" : "")); br.style.left = lpos(a, lo, hi) + "%"; br.style.width = (i ? lpos(b, lo, hi) - lpos(a, lo, hi) : 0.6) + "%"; t.appendChild(br);
        const w = el("div", null, "dw"); w.style.left = lpos(mn[i], lo, hi) + "%"; w.style.width = (lpos(mx2[i], lo, hi) - lpos(mn[i], lo, hi)) + "%"; t.appendChild(w);
        decText(t, i ? "×" + f(cur / prev, 2) + " → " + fx(cur) + " total" : "1× (reference)", Math.min(a, mn[i]), Math.max(b, mx2[i]), lo, hi); LAD.appendChild(t); }); }
    /* 2 one change at a time */
    const items = []; DEC.groups.forEach(g => g.items.forEach(it => { if (keysOf((it.r || {})[M]).length) items.push(it); }));
    let lo2 = 0.25, hi2 = 2.6; items.forEach(it => keysOf(it.r[M]).forEach(k => { hi2 = Math.max(hi2, it.r[M][k] * 1.2); lo2 = Math.min(lo2, it.r[M][k] * 0.9); }));
    $("arAblNote").textContent = "Each row changes one decision in the recipe and shows its efficiency relative to the plain recipe at the same held-out loss: right of 1× needs fewer FLOPs than the recipe, left needs more. The bar is the deepest loss level the run reached (the recipe's loss at step N; deeper is the harder comparison; faded bars only reached step 25-50). The whisker is the range over the levels it reached. One seed each, eval_size 184.";
    axisRow(ABL, lo2, hi2, [0.25, 0.5, 1, 2, 3]);
    DEC.groups.forEach(g => { ABL.appendChild(el("div", g.name, "dh"));
      g.items.forEach(it => { const o = (it.r || {})[M] || {}, ks = keysOf(o);
        decLabel(ABL, it.label, ks.length ? "vs the recipe's loss at step " + ks[ks.length - 1] + " · " + (it.steps || "?") + " steps run" : "");
        const t = track(lo2, hi2);
        if (!ks.length) { const v = el("span", it.ce != null ? "never reached the recipe's step-25 loss (final CE " + f(it.ce, 3) + ")" : "n/a at " + mlab(), "dv nr"); v.style.left = "0"; t.appendChild(v); ABL.appendChild(t); return; }
        const k = ks[ks.length - 1], v = o[k], a = Math.min(1, v), b = Math.max(1, v), vals = ks.map(q => o[q]), mnv = Math.min(...vals), mxv = Math.max(...vals);
        const br = el("div", null, "db" + (v < 1 ? " dn" : "") + (+k < 50 ? " sh" : "")); br.style.left = lpos(a, lo2, hi2) + "%"; br.style.width = (lpos(b, lo2, hi2) - lpos(a, lo2, hi2)) + "%"; t.appendChild(br);
        if (ks.length > 1) { const w = el("div", null, "dw"); w.style.left = lpos(mnv, lo2, hi2) + "%"; w.style.width = (lpos(mxv, lo2, hi2) - lpos(mnv, lo2, hi2)) + "%"; t.appendChild(w); }
        decText(t, fx(v), Math.min(a, mnv), Math.max(b, mxv), lo2, hi2); ABL.appendChild(t); }); }); };

  /* ideas list, PAIL style */
  const famNote = $("arFamNote");
  const itemOf = u => { const li = el("li"), t = el("div", null, "t");
    t.appendChild(el("span", u.family, "tag")); t.appendChild(pill(u.status)); t.appendChild(el("b", u.title));
    t.appendChild(el("span", (u.best != null ? "best " + f(u.best, 2) + "× · " : "") + u.arms.length + " arm" + (u.arms.length === 1 ? "" : "s"), "rd")); li.appendChild(t);
    if (u.idea && u.idea.result) li.appendChild(el("div", lead(u.idea.result, 230), "res"));
    const open = u.status === "running", d = fold(["Details and arms"], open);
    const body = el("div", null, "ideabody");
    body.appendChild(implPanel(u, false));
    if (u.idea && u.idea.what) { const p = el("p"); p.appendChild(el("b", "What (as queued by the loop): ")); p.appendChild(document.createTextNode(u.idea.what)); body.appendChild(p); }
    if (u.idea && u.idea.result) { const p = el("p"); p.appendChild(el("b", "Result: ")); p.appendChild(document.createTextNode(u.idea.result)); body.appendChild(p); }
    if (u.arms.length) { const a = fold(["Arms of this idea", mh(u.arms.length + " rows")], false); a.appendChild(armsTable(u.arms)); body.appendChild(a); }
    d.appendChild(body); li.appendChild(d); return li; };
  const drawIdeas = by => { const host = $("arFam"); host.textContent = "";
    const keyOf = by === "family" ? (u => cap(u.family)) : (u => u.status);
    const groups = {}; units.forEach(u => { (groups[keyOf(u)] = groups[keyOf(u)] || []).push(u); });
    const names = Object.keys(groups).sort((a, b) => by === "family" ? (a === "Reference" ? -1 : b === "Reference" ? 1 : a === "Baseline" ? -1 : b === "Baseline" ? 1 : 0) : (SR[a] ?? 9) - (SR[b] ?? 9));
    names.forEach(n => { const us = groups[n].slice().sort((a, b) => (SR[a.status] ?? 9) - (SR[b.status] ?? 9) || (b.best || 0) - (a.best || 0));
      const d = fold([by === "family" ? n : (OUT[n] || cap(n)), mh(us.length + (us.length === 1 ? " idea" : " ideas"))], true, "qg"); const ul = el("ul", null, "ql"); us.forEach(u => ul.appendChild(itemOf(u))); d.appendChild(ul); host.appendChild(d); }); };
  let BYV = "outcome";
  const sw = by => { BYV = by; $("arGO").setAttribute("aria-pressed", by === "outcome"); $("arGF").setAttribute("aria-pressed", by === "family"); drawIdeas(by); };
  $("arGO").onclick = () => sw("outcome"); $("arGF").onclick = () => sw("family");
  setN("arIdeasN", ideasAll.length + " ideas · " + cntTxt + (units.length > ideasAll.length ? " · " + (units.length - ideasAll.length) + " arm groups with no queue entry" : "") + " · " + B.length + " arms listed");
  if (es.length) famNote.textContent += " ⚠ eval_size " + es.join(" / ") + " held-out rows (16 per task and length; below 500, so differences under about two paired SEs are noise). Arm tables: italic rows are references, sorted by speedup over the recipe at the same loss. Memory columns were removed on 11-17: the logged values were wrong; valid memory measurements are in the log and the idea results.";

  /* the context-length tabs drive the strip, the hero chart, the glance bars, the idea list and the arm tables */
  let bestChart = null;
  const drawBestTable = () => { const box = $("arBest"); if (!BH.length) return; box.textContent = "";
    box.appendChild(table(["time", "method", "idea", "FLOP speedup (as logged)", "× at " + mlab(), "worst task", "transient memory vs dense LoRA", "note"],
      BH.map(r => [r.t, r.arm, r.idea, xs(r.x) + (r.level == null || r.level === "" ? "" : (isNaN(r.level) ? " @ " : " @ step ") + r.level), bhVal(r) == null ? "n/a" : xs(bhVal(r)), r.xw == null ? "n/a" : xs(r.xw), r.mem == null ? "n/a" : f(r.mem, 2) + "×", r.note]), [5, 6], "arwide")); };
  const renderMetric = () => { calcBest(); refX = refUnit ? refUnit.best : null;
    const nok = BH.filter(r => bhVal(r) != null).length, wts = MT.weights || {};
    const base = M === "w" ? "Held-out CE averaged with weights " + Object.keys(wts).map(k => k.replace("s", "") + " " + Math.round(wts[k] * 100) + "%").join(" · ") + ", then FLOPs to reach the dense-LoRA loss, as in the published speedups."
      : M === "all" ? "The loop's own metric: held-out CE averaged over all lengths and tasks." : "Held-out CE on the " + mlab() + " rows only (eval_size below 500, so noisy), then FLOPs to reach the dense-LoRA loss.";
    $("arMetricNote").textContent = base + (M === "all" ? "" : " Recomputed here from the per-length CE curves; it reproduces the loop's published speedups on " + MT.check[0] + " of " + (MT.check[0] + MT.check[1]) + " levels. " + nok + " of " + BH.length + " hourly entries can be traced to an arm on the board; the others are left out of the chart.");
    if (last) { const sr = bhSeries(), lv = sr[sr.length - 1]; $("arBestNote").textContent = lv == null ? "No hourly entry can be traced to an arm with this measurement." : "Latest best at " + mlab() + ": " + xs(lv) + (M === "all" ? " (" + last.arm + ", " + last.idea + "), found " + last.t + "." : "."); }
    drawStrip(); drawGlance(); drawDecisions(); drawIdeas(BYV); drawBestTable();
    if (bestChart) { bestChart.data.datasets[0].data = bhSeries(); bestChart.options.scales.y.title.text = "FLOP speedup (×), " + mlab(); bestChart.update(); } };
  const mbar = $("arMetric"); mbar.appendChild(el("span", "Gains at context length"));
  MT.metrics.forEach(m => { const b = el("button", m.label); b.type = "button"; b.dataset.m = m.id; b.setAttribute("aria-pressed", m.id === M); b.onclick = () => { M = m.id; mbar.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x.dataset.m === M)); renderMetric(); }; mbar.appendChild(b); });
  if (MT.metrics.length < 2) mbar.style.display = "none";
  renderMetric();

  /* FLOPs to reach a held-out loss */
  const DP = ST.depth;
  if (!DP || !(DP.runs || []).length) { const s = $("ar-depth-sec"); if (s) s.style.display = "none"; } else {
    const lvls = DP.levels || [], key = l => Number(l).toFixed(3);
    const cell = (x, pf) => { const s = el("span"); s.appendChild(document.createTextNode(x == null ? "not reached" : f(x, 1) + "×")); if (pf != null) { s.appendChild(el("br")); s.appendChild(el("small", f(pf, 0) + " PF", "note")); } return s; };
    let mn = Infinity; const rm = (DP.ref || []).map(q => [q[0], mn = Math.min(mn, q[1])]);
    const refPF = l => { for (let i = 0; i < rm.length; i++) if (rm[i][1] <= l) { if (i === 0) return rm[0][0]; const [p0, c0] = rm[i - 1], [p1, c1] = rm[i]; return c0 === c1 ? p1 : p0 + (p1 - p0) * (c0 - l) / (c0 - c1); } return null; };
    const rows = [[two("dense LoRA, full epoch (reference)"), "", "1.00×", ""].concat(lvls.map(l => { const p = refPF(l); return p == null ? cell(null) : cell(1, p); }))];
    DP.runs.forEach(r => rows.push([r.run, r.steps, r.x_dense == null ? "" : f(r.x_dense, 2) + "×", f(r.last_ce, 3)].concat(lvls.map(l => { const q = (r.levels || {})[key(l)]; return q ? cell(q.x_vs_dense_lora, q.pf) : cell(null); }))));
    $("arDepth").appendChild(table(["run", "steps", "cost per step vs dense", "last CE"].concat(lvls.map(l => "CE " + key(l))), rows, [1, 2, 3].concat(lvls.map((_, i) => 4 + i))));
    const bx = DP.runs.flatMap(r => Object.values(r.levels || {}).map(q => q.x_vs_dense_lora).filter(v => v != null));
    setN("arDepthN", DP.runs.length + " runs" + (bx.length ? " · best " + xs(Math.max(...bx)) : "")); }

  /* validation, scoring, log, pilot */
  const V = ST.validation || [];
  if (!V.length) empty($("arVal"), "None launched yet (planned every 2-3 hours for the most promising methods)."); else
    $("arVal").appendChild(table(["run", "method", "rows", "status", "predicted", "actual", "note"], V.map(r => [r.run, r.arm, r.rows, r.status, r.predicted, r.actual, r.note]), [], "arwide"));
  setN("arValN", V.length ? V.length + " runs · " + [...new Set(V.map(r => r.status))].join(", ") : "none yet");
  const SU = ST.setup || []; SU.forEach(s => $("arSetup").appendChild(el("li", s))); setN("arSetupN", SU.length + " points");
  const P = ST.pilot || [];
  if (P.length) { const pl = [...new Set(P.flatMap(r => Object.keys(r.x || {})))].sort((a, b) => a - b);
    $("arPilot").appendChild(table(["method", "seeds", "steps", "held-out CE", "cost / step vs dense"].concat(pl.map(l => "× @ step " + l)),
      P.map(r => [(r.is_ref ? "[ref] " : "") + r.arm, (r.seeds || []).length, r.steps, f(r.ce_last, 3), r.pf_per_step_vs_dense == null ? "" : f(r.pf_per_step_vs_dense, 2) + "×"].concat(pl.map(l => xs((r.x || {})[l])))), [1, 2, 3, 4].concat(pl.map((_, i) => 5 + i)))); }
  setN("arPilotN", P.length + " rows");
  const L = (ST.log || []).slice().reverse(), LH = ["time", "entry"], lrows = a => a.map(r => [r.t, r.text]);
  if (!L.length) empty($("arLog"), "No entries."); else {
    $("arLog").appendChild(table(LH, lrows(L.slice(0, 5)), [], "arlog arwide"));
    if (L.length > 5) { const d = fold(["Older entries", mh((L.length - 5) + " entries")], false); d.appendChild(table(LH, lrows(L.slice(5)), [], "arlog arwide")); $("arLog").appendChild(d); } }
  setN("arLogN", L.length + " entries");

  /* overview box: what is running now */
  const ov = $("ovLive"); if (ov) { ov.textContent = "";
    ov.appendChild(el("p", sents.slice(0, 2).join(" "), "lr"));
    const run = ideasAll.filter(i => i.status === "running");
    if (run.length) { ov.appendChild(el("p", "Running ideas (" + run.length + "):", "note")); const ul = el("ul"); run.forEach(i => { const li = el("li"); li.appendChild(el("b", i.title)); li.appendChild(document.createTextNode(" (" + i.family + ")")); ul.appendChild(li); }); ov.appendChild(ul); }
    const n = el("p", null, "note"); n.appendChild(document.createTextNode((ST.updated ? "Auto-research loop, updated " + ST.updated + ". " : "") + (es.length ? "⚠ Held-out eval_size " + es.join(" / ") + " (below 500), so small differences are noise. " : "")));
    const a = el("a", "Open the auto-research tab"); a.href = "#autoresearch"; n.appendChild(a); n.appendChild(document.createTextNode(". Other sessions: ")); const b = el("a", "experiment log"); b.href = "#log"; n.appendChild(b); n.appendChild(document.createTextNode(".")); ov.appendChild(n); }

  /* charts */
  if (!window.Chart) return;
  const fg = css("--fg"), mu = css("--muted"), ru = css("--rule");
  const leg = { position: "bottom", labels: { color: fg, boxWidth: 10, font: { size: 10 } } };
  const base = { responsive: true, maintainAspectRatio: false, animation: false };
  const axis = (lab, extra) => Object.assign({ title: { display: true, text: lab, color: mu }, ticks: { color: mu }, grid: { color: ru } }, extra || {});
  if (BH.length && $("arBestChart")) {
    bestChart = new Chart($("arBestChart"), { type: "line", data: { labels: BH.map(r => r.t), datasets: [{ label: "best FLOP speedup", data: bhSeries(), borderColor: css("--c1"), backgroundColor: css("--c1"), tension: 0, stepped: "before", pointRadius: 4 }] },
      options: { ...base, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => (M === "all" ? BH[c.dataIndex].arm + ": " : "best so far: ") + f(c.raw, 1) + "×", afterLabel: c => lead(BH[c.dataIndex].note, 160) } } }, scales: { x: { ticks: { color: mu, maxRotation: 0, autoSkip: true }, grid: { display: false } }, y: axis("FLOP speedup (×), " + mlab(), { beginAtZero: true }) } } });
  }
  const C = (ST.curves || []).filter(c => !HIDE(c));
  if ($("arCurves")) {
    if (!C.length) { $("arCurves").closest("figure").style.display = "none"; $("arSel").parentNode.style.display = "none"; empty($("arCurves").closest(".secbody"), "No curves yet."); }
    else {
      const rowBy = {}; B.forEach(r => { rowBy[r.tag + "|" + r.arm] = r; });
      const order = []; B.forEach(r => { if (!order.includes(r.arm)) order.push(r.arm); });
      const rank = a => { const i = order.indexOf(a); return i < 0 ? 999 : i; };
      const keepArms = new Set(order.slice(0, 12)); C.forEach(c => { if (c.is_ref) keepArms.add(c.arm); });
      const sel = $("arSel"); sel.appendChild(new Option("no idea (references and top arms only)", ""));
      units.forEach(u => { if (u.arms.length) sel.appendChild(new Option(u.title + " (" + u.arms.length + " arms)", u.key)); });
      const inSel = (c, v) => { const r = rowBy[c.tag + "|" + c.arm]; if (!r || !v) return false; const u = unitBy[v]; return !!u && u.arms.includes(r); };
      const pal = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7"];
      let chart = null;
      const draw = () => { const v = sel.value, shown = C.filter(c => keepArms.has(c.arm) || inSel(c, v)).sort((a, b) => rank(a.arm) - rank(b.arm));
        const col = {}, dash = {}; let k = 0; shown.forEach(c => { if (!(c.arm in col)) { col[c.arm] = css(pal[k % pal.length]); dash[c.arm] = k >= pal.length ? [2, 3] : []; k++; } });
        const ds = shown.map(c => { const p = (c.pts || []).filter(q => q[1] > 0);
          return { label: c.arm + " s" + c.seed, arm: c.arm, tag: c.tag, seed: c.seed, raw: p, data: p.map(q => ({ x: q[1], y: q[2] })), borderColor: col[c.arm], backgroundColor: col[c.arm],
            borderWidth: c.is_ref ? 3 : 1.5, borderDash: c.is_ref ? [6, 4] : dash[c.arm], pointRadius: 2, tension: 0 }; });
        if (!chart) chart = new Chart($("arCurves"), { type: "line", data: { datasets: ds }, options: { ...base, parsing: false, interaction: { mode: "nearest", intersect: true },
          plugins: { legend: leg, tooltip: { callbacks: { label: c => { const d = c.dataset, q = d.raw[c.dataIndex]; return d.arm + " (" + d.tag + ") seed " + d.seed + ": step " + q[0] + ", " + f(q[1], 2) + " PF, CE " + f(q[2], 4); } } } },
          scales: { x: axis("cumulative training PFLOPs", { type: "logarithmic" }), y: axis("held-out dense CE") } } });
        else { chart.data.datasets = ds; chart.update(); }
        const hidden = new Set(C.filter(c => !shown.includes(c)).map(c => c.arm)).size;
        $("arSelNote").textContent = shown.length + " curves drawn: references, the top 12 methods by leaderboard order" + (v ? " and the arms of the chosen idea" : "") + "." + (hidden ? " " + hidden + " further method(s) not drawn." : ""); };
      sel.addEventListener("change", draw); draw();
      const sec = $("ar-curves-sec"); if (sec) sec.addEventListener("toggle", () => { if (chart) chart.resize(); });
    }
  }
})();
</script>
''' + SE + "\n"
JS = JS.replace("__ARDATA__", json.dumps(STATE, ensure_ascii=False).replace("</", "<\\/"))

# ---- splice ---------------------------------------------------------------------------------------------------------
LEG = re.compile(r"<!-- AUTORESEARCH:START -->(.*?)<!-- AUTORESEARCH:END -->\n?", re.S)
V2P = re.compile(re.escape(PS) + r".*?" + re.escape(PE) + r"\n?", re.S)
V2S = re.compile(re.escape(SS) + r".*?" + re.escape(SE) + r"\n?", re.S)


def put(h, rx, block, fallback):
    m = rx.search(h)
    if m:
        return h[: m.start()] + block + h[m.end():]
    return fallback(h, block)


def strip_legacy(h):
    """Remove legacy v1 blocks; return (h, offset of the legacy part block or None)."""
    pos, out, last = None, [], 0
    for m in LEG.finditer(h):
        out.append(h[last: m.start()])
        if 'class="part"' in m.group(1) and pos is None:
            pos = sum(len(x) for x in out)
        last = m.end()
    out.append(h[last:])
    return "".join(out), pos


if LEG.search(h):
    h, pos = strip_legacy(h)
    if pos is not None and not V2P.search(h):
        h = h[:pos] + PART + h[pos:]


def ins_part(h, block):
    key = '<div class="part" id="provenance"'
    assert h.count(key) == 1, "provenance part not found"
    return h.replace(key, block + key, 1)


def ins_script(h, block):
    i = h.rindex("</script>") + len("</script>")
    return h[:i] + "\n" + block + h[i:]


h = put(h, V2P, PART, ins_part)
h = put(h, V2S, JS, ins_script)
open(H, "w", encoding="utf-8").write(h)
print("built", len(h), "bytes;", len(STATE.get("board", [])), "board rows,", len(STATE.get("ideas", [])), "ideas")
