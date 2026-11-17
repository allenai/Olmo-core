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
/* every idea at a glance: one bar per idea */
.glance { display: grid; grid-template-columns: minmax(120px, 30%) minmax(0, 1fr); gap: 3px 10px; align-items: center; font-size: .82rem; max-width: 900px; }
.glance .gl { text-align: right; line-height: 1.25; } .glance .gl a { color: var(--fg); text-decoration: none; } .glance .gl a:hover { color: var(--c2); }
.glance .gt { position: relative; height: 18px; background: color-mix(in srgb, var(--rule) 45%, transparent); }
.glance .gb { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 0 2px 2px 0; background: var(--muted); }
.glance .gb.s-running { background: var(--c1); } .glance .gb.s-done { background: var(--c3); } .glance .gb.s-dropped { background: var(--muted); opacity: .75; }
.glance .gv { position: absolute; top: 0; line-height: 18px; font: .72rem var(--mono); white-space: nowrap; padding-left: 5px; color: var(--fg); }
.glance .gr { position: absolute; top: -2px; bottom: -2px; border-left: 2px dashed var(--dense); opacity: .75; }
.glance .gax { grid-column: 2; position: relative; height: 14px; font: .68rem var(--mono); color: var(--muted); } .glance .gax span { position: absolute; transform: translateX(-50%); }
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
<div class="tiles" id="arStrip"></div>
''' + bar("ar-progress", "Best so far, over time", '''
<div id="arBestNote" class="note"></div>
<div class="figs"><figure style="grid-column: 1 / -1"><div class="chart" style="height:230px"><canvas id="arBestChart" aria-label="best FLOP speedup over time"></canvas></div><figcaption>Best FLOP speedup over the dense reference found so far, by time of the loop (hover a point for the method).</figcaption></figure></div>
<details class="more" id="arBestBox"><summary>Every new best, as a table <span class="mh" id="arBestN"></span></summary><div id="arBest"></div></details>
<details class="more" id="arStatusBox"><summary>Full status text <span class="mh">from the loop</span></summary><p class="note" id="arStatus" style="margin:10px 12px 12px"></p></details>''', True) + bar("ar-glance", "Every idea at a glance", '''
<p class="note">Best FLOP speedup any arm of the idea reached, against dense LoRA at the same held-out loss (at a level the arm actually reached; the dashed line is the plain recipe). Ideas with no bar reached no comparison level. Colour is the idea's status. ⚠ eval_size 184, below 500: differences under about two paired SEs are noise.</p>
<div id="arGlance" class="glance"></div>''', True) + bar("ar-ideas", "Ideas, with their verdicts", '''
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
    $("arBest").appendChild(table(["time", "method", "idea", "FLOP speedup", "worst task", "transient memory vs dense LoRA", "note"],
      BH.map(r => [r.t, r.arm, r.idea, xs(r.x) + (r.level == null || r.level === "" ? "" : (isNaN(r.level) ? " @ " : " @ step ") + r.level), r.xw == null ? "n/a" : xs(r.xw), r.mem == null ? "n/a" : f(r.mem, 2) + "×", r.note]), [4, 5], "arwide"));
    $("arBestN").textContent = BH.length + " rows"; }
  const last = BH[BH.length - 1];
  if (last) $("arBestNote").textContent = "Latest best: " + xs(last.x) + " (" + last.arm + ", " + last.idea + "), found " + last.t + ".";

  /* board rows -> ideas -> families */
  function HIDE(r) { return r.tag + "/" + r.arm === "hp/denselora-lr2e5" || r.tag === "hp/denselora-lr2e5" || r.arm === "hp/denselora-lr2e5"; }
  const ALIAS = { rowsbk: "rows", "goldrand-lam": "goldrand", grmix50: "goldrand", prefixl: "prefix", "ffnskip-lam": "ffnskip", "earlyloss-lam": "earlyloss", hp: "hparams", mq: "multiq" };
  const ideasAll = ST.ideas || [], byId = {}; ideasAll.forEach(i => { byId[i.id] = i; });
  const famIdeas = {}; ideasAll.forEach(i => { (famIdeas[i.family] = famIdeas[i.family] || []).push(i); });
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
  const bestx = r => { const v = Object.values(r.x || {}).filter(q => q != null); return v.length ? Math.max(...v) : null; };
  const armsTable = rows => { const head = ["method", "seeds", "steps", "held-out CE", "cost / step vs dense"].concat(levels.map(l => "× @ step " + l), ["× vs dense LoRA at own loss", "× vs recipe at own loss", "worst task @ step " + deep, "train s"]);
    const body = rows.map(r => { const ce = r.ce_last == null ? "" : f(r.ce_last, 3) + (r.d_vs_ref_se != null && !r.is_ref ? "  (Δ " + f(r.d_vs_ref, 3) + " ±" + f(r.d_vs_ref_se, 3) + ")" : "");
      return [two((r.is_ref ? "[ref] " : "") + r.arm, r.desc), (r.seeds || []).length, r.steps, ce, r.pf_per_step_vs_dense == null ? "" : f(r.pf_per_step_vs_dense, 2) + "×"]
        .concat(levels.map(l => xs((r.x || {})[l])), [iso(r), isr(r), xs((r.xw || {})[deep]), f(r.train_s, 0)]); });
    const nc = [1, 2, 3, 4].concat(levels.map((_, i) => 5 + i), [5 + levels.length, 6 + levels.length, 7 + levels.length, 8 + levels.length]);
    const t = table(head, body, nc); [...t.querySelectorAll("tr")].slice(1).forEach((tr, i) => { if (rows[i].is_ref) { tr.style.background = "var(--surface)"; tr.style.fontStyle = "italic"; } }); return t; };

  /* units = ideas, plus a pseudo-unit per family that has arms but no idea ("no queue entry") */
  const units = ideasAll.map(i => ({ key: i.id, idea: i, family: i.family, status: i.status, title: i.title, arms: [] }));
  const unitBy = {}; units.forEach(u => { unitBy[u.key] = u; });
  B.forEach(r => { const i = ideaOf(r); let u = i ? unitBy[i.id] : null;
    if (!u) { const k = "fam:" + famOf(r); u = unitBy[k] = unitBy[k] || (units.push({ key: k, idea: null, family: famOf(r), status: "done", title: famOf(r) === "unassigned" ? "Arms not tied to an idea" : cap(famOf(r)) + " (arms with no queue entry)", arms: [] }), units[units.length - 1]); }
    u.arms.push(r); });
  units.forEach(u => { const non = u.arms.filter(r => !r.is_ref).map(bestx).filter(v => v != null); u.best = non.length ? Math.max(...non) : null; });
  const SR = { running: 0, done: 1, queued: 2, dropped: 3, failed: 4 }, OUT = { running: "Running", done: "Done: measured or kept", queued: "Queued", dropped: "Dropped: no gain", failed: "Failed" };
  const refUnit = unitBy.recipe, refX = refUnit ? refUnit.best : null;

  /* stat strip */
  const cnt = {}; ideasAll.forEach(i => { cnt[i.status] = (cnt[i.status] || 0) + 1; });
  const cntTxt = Object.keys(cnt).sort((a, b) => (SR[a] ?? 9) - (SR[b] ?? 9)).map(s => cnt[s] + " " + s).join(" · ");
  const es = [...new Set(B.map(r => r.eval_size).filter(v => v != null))];
  const tile = (b, s) => { const t = el("div", null, "tile"); t.appendChild(el("b", b)); t.appendChild(el("span", s)); return t; };
  const strip = $("arStrip");
  if (last) strip.appendChild(tile(xs(last.x), "best FLOP speedup so far vs dense LoRA (" + last.arm + ")"));
  strip.appendChild(tile(String(ideasAll.length), "ideas: " + cntTxt));
  strip.appendChild(tile(String(B.length), "arms scored" + (es.length ? " · ⚠ eval_size " + es.join(" / ") + " (below 500)" : "")));
  if (ST.updated) strip.appendChild(tile(ST.updated.replace(/^\d{4}-/, ""), "last update (Pacific time)"));

  /* every idea at a glance */
  const G = $("arGlance"), gu = units.filter(u => !(u.arms.length && u.arms.every(r => r.is_ref)));
  gu.sort((a, b) => (b.best == null ? -1 : b.best) - (a.best == null ? -1 : a.best));
  const mx = Math.max(4, ...gu.map(u => u.best || 0), refX || 0) * 1.06;
  gu.forEach(u => { const lab = el("div", null, "gl"); const a = el("a", u.title); a.href = "#ar-ideas"; lab.appendChild(a); G.appendChild(lab);
    const t = el("div", null, "gt"); if (u.best != null) { const b = el("div", null, "gb s-" + u.status); b.style.width = (u.best / mx * 100) + "%"; t.appendChild(b);
      const na = u.arms.filter(r => !r.is_ref).length, pc = u.best / mx * 100, v = el("span", f(u.best, 2) + "× · " + na + (na === 1 ? " arm" : " arms"), "gv");
      if (pc > 55) { v.style.right = (100 - pc) + "%"; v.style.paddingRight = "5px"; v.style.color = "var(--bg)"; } else v.style.left = pc + "%"; t.appendChild(v); }
    else { const v = el("span", u.arms.length ? "no comparison level reached (" + u.arms.length + " arms)" : "no arms scored yet", "gv"); v.style.left = "0"; t.appendChild(v); }
    if (refX != null) { const r = el("div", null, "gr"); r.style.left = (refX / mx * 100) + "%"; t.appendChild(r); }
    G.appendChild(t); });
  if (refX != null) { const nn = G.previousElementSibling; if (nn) nn.textContent += " Plain recipe: " + f(refX, 2) + "×."; }
  const ax = el("div", null, "gax"); [1, 2, 3, 4, 6, 8].filter(v => v < mx).forEach(v => { const s = el("span", v + "×"); s.style.left = (v / mx * 100) + "%"; ax.appendChild(s); }); G.appendChild(ax);

  /* ideas list, PAIL style */
  const famNote = $("arFamNote");
  const itemOf = u => { const li = el("li"), t = el("div", null, "t");
    t.appendChild(el("span", u.family, "tag")); t.appendChild(pill(u.status)); t.appendChild(el("b", u.title));
    t.appendChild(el("span", (u.best != null ? "best " + f(u.best, 2) + "× · " : "") + u.arms.length + " arm" + (u.arms.length === 1 ? "" : "s"), "rd")); li.appendChild(t);
    if (u.idea && u.idea.result) li.appendChild(el("div", lead(u.idea.result, 230), "res"));
    const open = u.status === "running", d = fold(["Details and arms"], open);
    const body = el("div", null, "ideabody");
    if (u.idea && u.idea.what) { const p = el("p"); p.appendChild(el("b", "What: ")); p.appendChild(document.createTextNode(u.idea.what)); body.appendChild(p); }
    if (u.idea && u.idea.result) { const p = el("p"); p.appendChild(el("b", "Result: ")); p.appendChild(document.createTextNode(u.idea.result)); body.appendChild(p); }
    if (u.arms.length) { const a = fold(["Arms of this idea", mh(u.arms.length + " rows")], false); a.appendChild(armsTable(u.arms)); body.appendChild(a); }
    d.appendChild(body); li.appendChild(d); return li; };
  const drawIdeas = by => { const host = $("arFam"); host.textContent = "";
    const keyOf = by === "family" ? (u => cap(u.family)) : (u => u.status);
    const groups = {}; units.forEach(u => { (groups[keyOf(u)] = groups[keyOf(u)] || []).push(u); });
    const names = Object.keys(groups).sort((a, b) => by === "family" ? (a === "Reference" ? -1 : b === "Reference" ? 1 : a === "Baseline" ? -1 : b === "Baseline" ? 1 : 0) : (SR[a] ?? 9) - (SR[b] ?? 9));
    names.forEach(n => { const us = groups[n].slice().sort((a, b) => (SR[a.status] ?? 9) - (SR[b.status] ?? 9) || (b.best || 0) - (a.best || 0));
      const d = fold([by === "family" ? n : (OUT[n] || cap(n)), mh(us.length + (us.length === 1 ? " idea" : " ideas"))], true, "qg"); const ul = el("ul", null, "ql"); us.forEach(u => ul.appendChild(itemOf(u))); d.appendChild(ul); host.appendChild(d); }); };
  drawIdeas("outcome");
  const sw = by => { $("arGO").setAttribute("aria-pressed", by === "outcome"); $("arGF").setAttribute("aria-pressed", by === "family"); drawIdeas(by); };
  $("arGO").onclick = () => sw("outcome"); $("arGF").onclick = () => sw("family");
  setN("arIdeasN", ideasAll.length + " ideas · " + cntTxt + (units.length > ideasAll.length ? " · " + (units.length - ideasAll.length) + " arm groups with no queue entry" : "") + " · " + B.length + " arms listed");
  if (es.length) famNote.textContent += " ⚠ eval_size " + es.join(" / ") + " held-out rows (16 per task and length; below 500, so differences under about two paired SEs are noise). Arm tables: italic rows are references, sorted by speedup over the recipe at the same loss. Memory columns were removed on 11-17: the logged values were wrong; valid memory measurements are in the log and the idea results.";

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
    new Chart($("arBestChart"), { type: "line", data: { labels: BH.map(r => r.t), datasets: [{ label: "best FLOP speedup", data: BH.map(r => r.x), borderColor: css("--c1"), backgroundColor: css("--c1"), tension: 0, stepped: "before", pointRadius: 4 }] },
      options: { ...base, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => BH[c.dataIndex].arm + ": " + f(c.raw, 1) + "×", afterLabel: c => lead(BH[c.dataIndex].note, 160) } } }, scales: { x: { ticks: { color: mu, maxRotation: 0, autoSkip: true }, grid: { display: false } }, y: axis("FLOP speedup (×)", { beginAtZero: true }) } } });
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
