"""Add / replace the "Auto-research loop" part of the FLOP-drop hub page (idempotent).

    python3 build_autoresearch_tab.py hub.html [--state loop_state.json]

v2 (2026-11-17, hub reorganisation). The hub is a grouped page: each `<div class="part" id=...>` declares
`data-group`, `data-order`, `data-short` (and optionally `data-badge`), and the page builds its two-level navigation
from them at run time. So this builder no longer touches the navigation. It owns two marker-delimited blocks:

    <!-- AR:PART:START -->   <div class="part" id="autoresearch" ...> ... </div>   <!-- AR:PART:END -->
    <!-- AR:SCRIPT:START -->  <script> ...state JSON + renderer... </script>        <!-- AR:SCRIPT:END -->

Re-running replaces both blocks in place. First run on a page without them: the part goes before the `provenance`
part, the script after the last `</script>`. Legacy v1 blocks (<!-- AUTORESEARCH:START --> ... <!-- AUTORESEARCH:END -->,
three of them: nav link, part, script) are stripped; the part's position is reused. The overview part's `#ovLive` box
is filled by the same script (status, running ideas).

Reads the same `loop_state.json` keys as v1: status, updated, setup, log, ideas, best_by_hour, depth, board, curves,
validation, pilot. Board rows map to ideas through `board[i].idea` (the idea family, set from hub_tab/arms.json by
make_loop_state.py), then by the row's `tag` (tag == idea id, or a name in ALIAS below), then, when a family has
exactly one idea, to that idea. A row may also carry `idea_id` to pin it. Rows that match no idea stay in their family
under "Other arms".
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

PART = PS + '''
<div class="part" id="autoresearch" data-group="g-tried" data-order="1" data-short="Auto-research loop" data-badge="live">
<style>
.pill { display: inline-block; font: 600 .66rem/1 var(--mono); letter-spacing: .05em; text-transform: uppercase; border: 1px solid var(--muted); color: var(--muted); border-radius: 999px; padding: 3px 7px; white-space: nowrap; vertical-align: 1px; }
.pill.s-running { border-color: var(--c1); color: var(--c1); } .pill.s-done { border-color: var(--c3); color: var(--c3); } .pill.s-failed { border-color: var(--c2); color: var(--c2); }
.famhead { display: grid; gap: 2px; margin: 8px 0 6px; } .famhead h3 { font: 600 1.02rem/1.25 var(--display); margin: 0; text-transform: none; }
.fams { display: grid; gap: 4px; }
details.idea > summary .vl { display: block; color: var(--muted); font-size: .82rem; margin-top: 3px; } details.idea > summary .ih { display: inline; }
.ideabody { display: grid; gap: 8px; padding: 10px 12px 12px; min-width: 0; } .ideabody p { margin: 0; max-width: 90ch; }
.ideabody details.more { background: var(--bg); } .ideabody .tbl { margin: 0; }
.arctl { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin-bottom: 8px; }
.arctl select { font: .85rem var(--body); padding: 6px 8px; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); color: var(--fg); max-width: 100%; }
.arstat { display: grid; gap: 6px; } .arstat p { margin: 0; }
.arlog td:first-child { white-space: nowrap; }
.arwide td:last-child { min-width: 22rem; } .arwide td:nth-child(2) { min-width: 11rem; }
</style>
<div class="parthead"><span class="eyebrow">Tried · was tab 8d · live</span><h2 class="ptitle">Auto-research loop: FLOPs to reach the dense loss at 4k-16k</h2><p class="verdict" id="arVerdict"></p><p class="note" id="arUpdated"></p></div>
<section id="arSumSec"><h2 id="ar-summary">Summary: best so far</h2>
  <div class="arstat"><details class="more" id="arStatusBox"><summary>Full status text <span class="mh">from the loop</span></summary><p class="note" id="arStatus" style="margin:10px 12px 12px"></p></details></div>
  <div id="arBest"></div>
  <div class="figs"><figure style="grid-column: 1 / -1"><div class="chart" style="height:220px"><canvas id="arBestChart" aria-label="best FLOP speedup over time"></canvas></div><figcaption>Best FLOP speedup over the dense reference found so far, by time of the loop.</figcaption></figure></div></section>
<section id="arDepthSec"><h2 id="ar-depth">FLOPs to reach a held-out loss</h2><p class="note">Against the full-epoch dense-LoRA curve. Ratio above 1 means fewer training FLOPs than dense LoRA needs for the same held-out loss. ⚠ eval_size 184 held-out rows (below 500: differences under about two paired SEs are noise); one seed per row unless the name says otherwise.</p><div id="arDepth"></div></section>
<section id="arIdeasSec"><h2 id="ar-ideas">Ideas by family, with their arms</h2><p class="note" id="arFamNote">Each idea shows its verdict first. Running ideas are open; closed and negative ones are folded. Arms are the scored runs of the idea, with the same columns as before.</p><div id="arFam" class="fams"></div></section>
<section id="arCurveSec"><h2 id="ar-curves">Loss vs training FLOPs</h2>
  <div class="arctl"><label for="arSel">Also draw the arms of</label><select id="arSel"></select><span class="note" id="arSelNote"></span></div>
  <div class="figs"><figure style="grid-column: 1 / -1"><div class="chart" style="height:380px"><canvas id="arCurves" aria-label="held-out loss vs training PFLOPs"></canvas></div><figcaption id="arCurvesCap">Held-out dense CE against cumulative training PFLOPs (log axis); step 0 is omitted. Reference runs are thick and dashed.</figcaption></figure></div></section>
<section><h2 id="ar-validation">Larger FLOP-matched runs</h2><div id="arVal"></div></section>
<section><h2 id="ar-scoring">How it is scored</h2><details class="more" id="arSetupBox"><summary>Scoring rules and data caveats <span class="mh" id="arSetupN"></span></summary><ul id="arSetup" style="margin:10px 12px 12px"></ul></details></section>
<section><h2 id="ar-log">Log, newest first</h2><div id="arLog"></div></section>
<section><h2 id="ar-pilot">Earlier 8k pilot (old task set)</h2><details class="more"><summary>Show the pilot table <span class="mh">kept for context</span></summary><p class="note" style="margin:10px 12px">Contradiction, nq, outlier, oolong at 8k, 50 steps, eval_size 256; ⚠ below 500.</p><div id="arPilot"></div></details></section>
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

  /* header */
  const sents = sentences(ST.status);
  $("arVerdict").textContent = sents[0] || ""; $("arUpdated").textContent = ST.updated ? "updated " + ST.updated : "";
  if (sents.length > 1) $("arStatus").textContent = ST.status; else $("arStatusBox").style.display = "none";

  /* summary: best by hour */
  const BH = ST.best_by_hour || [];
  if (!BH.length) empty($("arBest"), "No 16k result yet"); else
    $("arBest").appendChild(table(["time", "method", "idea", "FLOP speedup", "worst task", "transient memory vs dense LoRA", "note"],
      BH.map(r => [r.t, r.arm, r.idea, xs(r.x) + (r.level == null || r.level === "" ? "" : (isNaN(r.level) ? " @ " : " @ step ") + r.level), r.xw == null ? "n/a" : xs(r.xw), r.mem == null ? "n/a" : f(r.mem, 2) + "×", r.note]), [4, 5], "arwide"));
  if (!BH.length) { const fg = $("arBestChart"); if (fg) fg.closest("figure").style.display = "none"; }

  /* FLOPs to reach a held-out loss */
  const DP = ST.depth;
  if (!DP || !(DP.runs || []).length) { const s = $("arDepthSec"); if (s) s.style.display = "none"; } else {
    const lvls = DP.levels || [], key = l => Number(l).toFixed(3);
    const cell = (x, pf) => { const s = el("span"); s.appendChild(document.createTextNode(x == null ? "not reached" : f(x, 1) + "×")); if (pf != null) { s.appendChild(el("br")); s.appendChild(el("small", f(pf, 0) + " PF", "note")); } return s; };
    let mn = Infinity; const rm = (DP.ref || []).map(q => [q[0], mn = Math.min(mn, q[1])]);
    const refPF = l => { for (let i = 0; i < rm.length; i++) if (rm[i][1] <= l) { if (i === 0) return rm[0][0]; const [p0, c0] = rm[i - 1], [p1, c1] = rm[i]; return c0 === c1 ? p1 : p0 + (p1 - p0) * (c0 - l) / (c0 - c1); } return null; };
    const rows = [[two("dense LoRA, full epoch (reference)"), "", "1.00×", ""].concat(lvls.map(l => { const p = refPF(l); return p == null ? cell(null) : cell(1, p); }))];
    DP.runs.forEach(r => rows.push([r.run, r.steps, r.x_dense == null ? "" : f(r.x_dense, 2) + "×", f(r.last_ce, 3)].concat(lvls.map(l => { const q = (r.levels || {})[key(l)]; return q ? cell(q.x_vs_dense_lora, q.pf) : cell(null); }))));
    $("arDepth").appendChild(table(["run", "steps", "cost per step vs dense", "last CE"].concat(lvls.map(l => "CE " + key(l))), rows, [1, 2, 3].concat(lvls.map((_, i) => 4 + i))));
  }

  /* board rows -> ideas -> families */
  function HIDE(r) { return r.tag + "/" + r.arm === "hp/denselora-lr2e5" || r.tag === "hp/denselora-lr2e5" || r.arm === "hp/denselora-lr2e5"; }
  const ALIAS = { rowsbk: "rows", "goldrand-lam": "goldrand", grmix50: "goldrand", prefixl: "prefix", "ffnskip-lam": "ffnskip", "earlyloss-lam": "earlyloss", hp: "hparams", mq: "multiq" };
  const ideasAll = ST.ideas || [], byId = {}, famOfIdea = {}; ideasAll.forEach(i => { byId[i.id] = i; });
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
  const armsTable = rows => { const head = ["method", "seeds", "steps", "held-out CE", "cost / step vs dense"].concat(levels.map(l => "× @ step " + l), ["× vs dense LoRA at own loss", "× vs recipe at own loss", "worst task @ step " + deep, "train s"]);
    const body = rows.map(r => { const ce = r.ce_last == null ? "" : f(r.ce_last, 3) + (r.d_vs_ref_se != null && !r.is_ref ? "  (Δ " + f(r.d_vs_ref, 3) + " ±" + f(r.d_vs_ref_se, 3) + ")" : "");
      return [two((r.is_ref ? "[ref] " : "") + r.arm, r.desc), (r.seeds || []).length, r.steps, ce, r.pf_per_step_vs_dense == null ? "" : f(r.pf_per_step_vs_dense, 2) + "×"]
        .concat(levels.map(l => xs((r.x || {})[l])), [iso(r), isr(r), xs((r.xw || {})[deep]), f(r.train_s, 0)]); });
    const nc = [1, 2, 3, 4].concat(levels.map((_, i) => 5 + i), [5 + levels.length, 6 + levels.length, 7 + levels.length, 8 + levels.length]);
    const t = table(head, body, nc); [...t.querySelectorAll("tr")].slice(1).forEach((tr, i) => { if (rows[i].is_ref) { tr.style.background = "var(--surface)"; tr.style.fontStyle = "italic"; } }); return t; };
  const best = rows => rows.filter(r => !r.is_ref && r.iso != null).sort((a, b) => b.iso - a.iso)[0];

  const FAMS = {}; const famList = [];
  const fam = n => FAMS[n] || (famList.push(FAMS[n] = { name: n, ideas: famIdeas[n] || [], arms: [], byIdea: {} }), FAMS[n]);
  ideasAll.forEach(i => fam(i.family));
  B.forEach(r => { const F = fam(famOf(r)), i = ideaOf(r); if (i) (F.byIdea[i.id] = F.byIdea[i.id] || []).push(r); else F.arms.push(r); F.all = (F.all || []).concat([r]); });
  const SR = { running: 0, done: 1, queued: 2, dropped: 3, failed: 4 };
  const frank = F => F.name === "reference" ? 0 : F.name === "baseline" ? 1 : F.name === "unassigned" ? 9 : F.ideas.some(i => i.status === "running") ? 2 : 3;
  famList.sort((a, b) => frank(a) - frank(b));
  const host = $("arFam"); let nArms = 0;
  if (!famList.length) empty(host, "No ideas listed.");
  famList.forEach(F => {
    const cnt = {}; F.ideas.forEach(i => { cnt[i.status] = (cnt[i.status] || 0) + 1; });
    const parts = Object.keys(cnt).sort((a, b) => (SR[a] ?? 9) - (SR[b] ?? 9)).map(s => cnt[s] + " " + s);
    const all = F.all || [], b = best(all);
    const hd = el("div", null, "famhead"); hd.appendChild(el("h3", cap(F.name)));
    hd.appendChild(el("p", (F.ideas.length ? F.ideas.length + (F.ideas.length > 1 ? " ideas: " : " idea: ") + parts.join(", ") : "no entry in the idea queue") + " · " + all.length + " arms" + (b ? " · best arm " + b.arm + ", " + iso(b) + " vs dense LoRA at its own loss" : ""), "note"));
    host.appendChild(hd);
    F.ideas.forEach(i => { const arms = F.byIdea[i.id] || [], run = i.status === "running";
      const ss = sentences(i.result); let vs = ss[0] || ""; for (let k = 1; k < ss.length && vs.length < 70; k++) vs += " " + ss[k];
      const ih = el("span", null, "ih"); ih.appendChild(pill(i.status)); ih.appendChild(document.createTextNode(" ")); ih.appendChild(el("b", i.title)); ih.appendChild(mh(arms.length + " arm" + (arms.length === 1 ? "" : "s")));
      const d = fold([ih, el("span", vs.length > 200 ? vs.slice(0, 200).replace(/\s+\S*$/, "") + " …" : vs, "vl")], run, "idea");
      const body = el("div", null, "ideabody");
      if (i.what) { const p = el("p"); p.appendChild(el("b", "What: ")); p.appendChild(document.createTextNode(i.what)); body.appendChild(p); }
      if (i.result) { const p = el("p"); p.appendChild(el("b", "Result: ")); p.appendChild(document.createTextNode(i.result)); body.appendChild(p); }
      if (arms.length) { const a = fold(["Arms of this idea", mh(arms.length + " rows")], run); a.appendChild(armsTable(arms)); body.appendChild(a); nArms += arms.length; }
      d.appendChild(body); host.appendChild(d); });
    if (F.arms.length) { const a = fold([F.ideas.length ? "Other arms in this family" : "Arms", mh(F.arms.length + " rows")], !F.ideas.length && (F.name === "reference" || F.name === "baseline")); a.appendChild(armsTable(F.arms)); host.appendChild(a); nArms += F.arms.length; }
  });
  const es = [...new Set(B.map(r => r.eval_size).filter(v => v != null))];
  if (es.length) $("arFamNote").textContent += " ⚠ eval_size " + es.join(" / ") + " held-out rows (16 per task and length; below 500, so differences under about two paired SEs are noise). Italic rows are references. Rows are sorted by speedup over the recipe at the same loss. Memory columns were removed on 11-17: the logged values were wrong; valid memory measurements are in the log and the idea results. " + nArms + " of " + B.length + " scored arms are listed below.";

  /* validation */
  const V = ST.validation || [];
  if (!V.length) empty($("arVal"), "None launched yet (planned every 2-3 hours for the most promising methods)."); else
    $("arVal").appendChild(table(["run", "method", "rows", "status", "predicted", "actual", "note"], V.map(r => [r.run, r.arm, r.rows, r.status, r.predicted, r.actual, r.note]), [], "arwide"));

  /* how it is scored */
  const SU = ST.setup || []; SU.forEach(s => $("arSetup").appendChild(el("li", s))); $("arSetupN").textContent = SU.length + " points";

  /* pilot */
  const P = ST.pilot || [];
  if (P.length) { const pl = [...new Set(P.flatMap(r => Object.keys(r.x || {})))].sort((a, b) => a - b);
    $("arPilot").appendChild(table(["method", "seeds", "steps", "held-out CE", "cost / step vs dense"].concat(pl.map(l => "× @ step " + l)),
      P.map(r => [(r.is_ref ? "[ref] " : "") + r.arm, (r.seeds || []).length, r.steps, f(r.ce_last, 3), r.pf_per_step_vs_dense == null ? "" : f(r.pf_per_step_vs_dense, 2) + "×"].concat(pl.map(l => xs((r.x || {})[l])))), [1, 2, 3, 4].concat(pl.map((_, i) => 5 + i)))); }

  /* log, newest first; the first five open, the rest folded */
  const L = (ST.log || []).slice().reverse(), LH = ["time", "entry"], lrows = a => a.map(r => [r.t, r.text]);
  if (!L.length) empty($("arLog"), "No entries."); else {
    $("arLog").appendChild(table(LH, lrows(L.slice(0, 5)), [], "arlog arwide"));
    if (L.length > 5) { const d = fold(["Older entries", mh((L.length - 5) + " entries")], false); d.appendChild(table(LH, lrows(L.slice(5)), [], "arlog arwide")); $("arLog").appendChild(d); } }

  /* overview box: what is running now */
  const ov = $("ovLive"); if (ov) { ov.textContent = "";
    const p = el("p", sents.slice(0, 2).join(" "), "lr"); ov.appendChild(p);
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
    new Chart($("arBestChart"), { type: "line", data: { labels: BH.map(r => r.t), datasets: [{ label: "best FLOP speedup", data: BH.map(r => r.x), borderColor: css("--c1"), backgroundColor: css("--c1"), tension: 0, pointRadius: 3 }] },
      options: { ...base, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => BH[c.dataIndex].arm + ": " + f(c.raw, 1) + "×" } } }, scales: { x: { ticks: { color: mu, maxRotation: 0, autoSkip: true }, grid: { display: false } }, y: axis("FLOP speedup (×)", { beginAtZero: true }) } } });
  }
  const C = (ST.curves || []).filter(c => !HIDE(c));
  if ($("arCurves")) {
    if (!C.length) { $("arCurves").closest("figure").style.display = "none"; $("arSel").parentNode.style.display = "none"; empty($("arCurves").closest("section"), "No curves yet."); }
    else {
      const rowBy = {}; B.forEach(r => { rowBy[r.tag + "|" + r.arm] = r; });
      const order = []; B.forEach(r => { if (!order.includes(r.arm)) order.push(r.arm); });
      const rank = a => { const i = order.indexOf(a); return i < 0 ? 999 : i; };
      const keepArms = new Set(order.slice(0, 12)); C.forEach(c => { if (c.is_ref) keepArms.add(c.arm); });
      /* the selector: one option per idea (and per family with arms that match no idea) */
      const sel = $("arSel"); sel.appendChild(new Option("no idea (references and top arms only)", ""));
      famList.forEach(F => { const og = document.createElement("optgroup"); og.label = cap(F.name);
        F.ideas.forEach(i => { if ((F.byIdea[i.id] || []).length) og.appendChild(new Option(i.title + " (" + F.byIdea[i.id].length + " arms)", "i:" + i.id)); });
        if (F.arms.length) og.appendChild(new Option("Other arms of " + F.name + " (" + F.arms.length + ")", "f:" + F.name));
        if (og.children.length) sel.appendChild(og); });
      const inSel = (c, v) => { const r = rowBy[c.tag + "|" + c.arm]; if (!r || !v) return false; if (v.startsWith("i:")) { const i = ideaOf(r); return !!i && i.id === v.slice(2); } return !ideaOf(r) && famOf(r) === v.slice(2); };
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
    """Replace the existing block in place (callable repl: block text is not a regex template); else use fallback(h)."""
    m = rx.search(h)
    if m:
        return h[: m.start()] + block + h[m.end():]
    return fallback(h, block)


def legacy_part_pos(h):
    """Position of the legacy part block (so the new one keeps its place); strips all legacy blocks."""
    pos = None
    for m in LEG.finditer(h):
        if 'class="part"' in m.group(1):
            pos = m.start()
            break
    if pos is None:
        return h, None
    # remove legacy blocks back to front, remembering the part's offset in the shrunken string
    spans = [(m.start(), m.end()) for m in LEG.finditer(h)]
    out, last, newpos = [], 0, None
    for a, b in spans:
        out.append(h[last:a])
        if a == pos:
            newpos = sum(len(x) for x in out)
        last = b
    out.append(h[last:])
    return "".join(out), newpos


if LEG.search(h):
    h, pos = legacy_part_pos(h)
    if pos is not None and not V2P.search(h):
        h = h[:pos] + PART + h[pos:]
    h = LEG.sub("", h)


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
