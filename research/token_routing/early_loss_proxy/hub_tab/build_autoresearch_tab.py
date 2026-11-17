"""Add/replace the "8d · Auto-research loop" tab in a hub.html (idempotent).

Usage: python3 build_autoresearch_tab.py hub.html [--state loop_state.json]
Inserts a part block (before the provenance tab), a nav link, and a script, each between
<!-- AUTORESEARCH:START --> and <!-- AUTORESEARCH:END --> markers. Re-running replaces them.
"""
import json, os, re, sys

args = sys.argv[1:]
state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loop_state.json")
if "--state" in args:
    i = args.index("--state"); state_path = args[i + 1]; del args[i:i + 2]
H = args[0]
h = open(H).read()
STATE = json.load(open(state_path))

S, E = "<!-- AUTORESEARCH:START -->", "<!-- AUTORESEARCH:END -->"
h = re.sub(re.escape(S) + r".*?" + re.escape(E) + r"\n?", "", h, flags=re.S)

TAB = S + '''
<div class="part" id="autoresearch">
<div class="parthead"><span class="eyebrow">8d · Auto-research loop</span><h2 class="ptitle">FLOPs to reach the dense loss at 4k-16k: search log</h2><p class="verdict" id="arVerdict"></p><p class="note" id="arUpdated"></p></div>
<section><h2>Best so far, by hour</h2><div id="arBest"></div>
  <div class="figs"><figure style="grid-column: 1 / -1"><div class="chart" style="height:220px"><canvas id="arBestChart" aria-label="best FLOP speedup over time"></canvas></div><figcaption>Best FLOP speedup over the dense reference found so far, by time of the loop.</figcaption></figure></div></section>
<section><h2>Leaderboard</h2><div id="arBoard"></div><p class="note" id="arBoardNote"></p></section>
<section><h2>Loss vs training FLOPs</h2>
  <div class="figs"><figure style="grid-column: 1 / -1"><div class="chart" style="height:380px"><canvas id="arCurves" aria-label="held-out loss vs training PFLOPs"></canvas></div><figcaption id="arCurvesCap">Held-out dense CE against cumulative training PFLOPs (log axis); step 0 is omitted. Reference runs are thick and dashed.</figcaption></figure></div></section>
<section><h2>Idea queue</h2><div id="arIdeas"></div></section>
<section><h2>Larger FLOP-matched runs</h2><div id="arVal"></div></section>
<section><h2>How it is scored</h2><ul id="arSetup"></ul></section>
<section><details><summary><b>Earlier 8k pilot (old task set)</b></summary><p class="note">Contradiction, nq, outlier, oolong at 8k, 50 steps, eval_size 256; kept for context.</p><div id="arPilot"></div></details></section>
<section><h2>Log</h2><div id="arLog"></div></section>
</div>
''' + E + "\n"

JS = S + r'''
<script>
(() => { const ST = __ARDATA__;
  const el = (tag, txt, cls) => { const e = document.createElement(tag); if (txt != null) e.textContent = txt; if (cls) e.className = cls; return e; };
  const $ = id => document.getElementById(id); if (!$("arBoard")) return;
  const f = (v, d) => (v == null || v === "" || isNaN(v)) ? "" : Number(v).toFixed(d);
  const xs = v => v == null ? "not reached" : f(v, 1) + "×";
  const table = (head, rows, numCols) => { const w = el("div", null, "tbl"), t = el("table"), tr = el("tr");
    head.forEach((x, i) => tr.appendChild(el("th", x, numCols.includes(i) ? "n" : ""))); t.appendChild(tr);
    rows.forEach(r => { const q = el("tr"); r.forEach((c, i) => { const td = el("td", null, numCols.includes(i) ? "n" : "");
      if (c && c.nodeType) td.appendChild(c); else td.textContent = c == null ? "" : c; q.appendChild(td); }); t.appendChild(q); });
    w.appendChild(t); return w; };
  const empty = (box, msg) => box.appendChild(el("p", msg, "note"));
  const two = (a, b) => { const s = el("span"); s.appendChild(document.createTextNode(a || "")); if (b) { s.appendChild(el("br")); s.appendChild(el("span", b, "note")); } return s; };
  // header
  $("arVerdict").textContent = ST.status || ""; $("arUpdated").textContent = ST.updated ? "updated " + ST.updated : "";
  // best by hour
  const BH = ST.best_by_hour || [];
  if (!BH.length) empty($("arBest"), "No 16k result yet"); else
    $("arBest").appendChild(table(["time", "method", "idea", "FLOP speedup", "worst task", "transient memory vs dense LoRA", "note"],
      BH.map(r => [r.t, r.arm, r.idea, xs(r.x) + (r.level != null ? " @ step " + r.level : ""), xs(r.xw), r.mem == null ? "" : f(r.mem, 2) + "×", r.note]), [3, 4, 5]));
  if (!BH.length) { const fg = $("arBestChart"); if (fg) fg.closest("figure").style.display = "none"; }
  // leaderboard
  const tg = r => { const s = el("span"); s.appendChild(document.createTextNode(f(r.transient_gib, 1))); if (r.peak_gib != null && r.peak_gib !== "") s.appendChild(el("span", " (" + f(r.peak_gib, 1) + ")", "note")); return s; };
  const rk = r => r.arm === "recipe" ? 0 : r.is_ref ? 1 : 2, ir = r => r.iso_rec == null ? -Infinity : r.iso_rec;
  const B = (ST.board || []).map((r, i) => [r, i]).sort((a, b) => rk(a[0]) - rk(b[0]) || ir(b[0]) - ir(a[0]) || a[1] - b[1]).map(x => x[0]), lv = new Set();
  const iso = r => r.iso == null ? "n/a" : (r.iso_lb ? ">= " : "") + f(r.iso, 1) + "×", isr = r => r.iso_rec == null ? "n/a" : f(r.iso_rec, 2) + "×";
  B.forEach(r => Object.keys(r.x || {}).forEach(k => lv.add(k)));
  const levels = [...lv].sort((a, b) => a - b), deep = levels[levels.length - 1];
  if (!B.length) empty($("arBoard"), "No scored methods yet."); else {
    const head = ["method", "idea", "seeds", "steps", "held-out CE", "cost / step vs dense"].concat(levels.map(l => "× @ step " + l), ["× vs dense LoRA at own loss", "× vs recipe at own loss", "worst task @ step " + deep, "transient GiB (peak)", "transient vs dense LoRA", "train s"]);
    const rows = B.map(r => { const ce = r.ce_last == null ? "" : f(r.ce_last, 3) + (r.d_vs_ref_se != null && !r.is_ref ? "  (Δ " + f(r.d_vs_ref, 3) + " ±" + f(r.d_vs_ref_se, 3) + ")" : "");
      const m = two((r.is_ref ? "[ref] " : "") + r.arm, r.desc);
      return [m, r.idea, (r.seeds || []).length, r.steps, ce, r.pf_per_step_vs_dense == null ? "" : f(r.pf_per_step_vs_dense, 2) + "×"]
        .concat(levels.map(l => xs((r.x || {})[l])), [iso(r), isr(r), xs((r.xw || {})[deep]), tg(r), r.transient_vs_ref == null ? "" : f(r.transient_vs_ref, 2) + "×", f(r.train_s, 0)]); });
    const t = table(head, rows, [2, 3, 4, 5].concat(levels.map((_, i) => 6 + i), [6 + levels.length, 7 + levels.length, 8 + levels.length, 9 + levels.length, 10 + levels.length, 11 + levels.length]));
    [...t.querySelectorAll("tr")].slice(1).forEach((tr, i) => { if (B[i].is_ref) tr.style.background = "var(--surface)"; if (B[i].is_ref) tr.style.fontStyle = "italic"; });
    $("arBoard").appendChild(t);
    const es = [...new Set(B.map(r => r.eval_size).filter(v => v != null))];
    $("arBoardNote").textContent = es.length ? "eval_size " + es.join(" / ") + " held-out rows (16 per task and length); below 500, so differences under about two paired SEs are noise. Italic rows are references; sorted with references and the recipe first, then by speedup over the recipe at the same loss." : "";
  }
  // idea queue
  const ORD = { running: 0, done: 1, queued: 2, dropped: 3, failed: 4 };
  const I = (ST.ideas || []).map((r, i) => [r, i]).sort((a, b) => (ORD[a[0].status] ?? 9) - (ORD[b[0].status] ?? 9) || a[1] - b[1]).map(x => x[0]);
  const PC = { running: "--c1", done: "--c3", queued: "--muted", dropped: "--muted", failed: "--c2" };
  const pill = s => { const p = el("span", s, "pill"); const c = getComputedStyle(document.documentElement).getPropertyValue(PC[s] || "--muted").trim();
    p.style.cssText = "border:1px solid " + c + ";color:" + c + ";border-radius:999px;padding:1px 8px;font-size:11px;white-space:nowrap"; return p; };
  if (!I.length) empty($("arIdeas"), "No ideas listed."); else
    $("arIdeas").appendChild(table(["status", "id", "family", "idea", "what", "result"], I.map(r => [pill(r.status), r.id, r.family, r.title, r.what, r.result]), []));
  // validation
  const V = ST.validation || [];
  if (!V.length) empty($("arVal"), "None launched yet (planned every 2-3 hours for the most promising methods)."); else
    $("arVal").appendChild(table(["run", "method", "rows", "status", "predicted", "actual", "note"], V.map(r => [r.run, r.arm, r.rows, r.status, r.predicted, r.actual, r.note]), []));
  // setup
  ($("arSetup") && (ST.setup || [])).forEach(s => $("arSetup").appendChild(el("li", s)));
  // pilot
  const P = ST.pilot || [];
  if (P.length) { const pl = [...new Set(P.flatMap(r => Object.keys(r.x || {})))].sort((a, b) => a - b);
    $("arPilot").appendChild(table(["method", "seeds", "steps", "held-out CE", "cost / step vs dense"].concat(pl.map(l => "× @ step " + l)),
      P.map(r => [(r.is_ref ? "[ref] " : "") + r.arm, (r.seeds || []).length, r.steps, f(r.ce_last, 3), r.pf_per_step_vs_dense == null ? "" : f(r.pf_per_step_vs_dense, 2) + "×"].concat(pl.map(l => xs((r.x || {})[l])))), [1, 2, 3, 4].concat(pl.map((_, i) => 5 + i)))); }
  // log
  const L = ST.log || [];
  if (!L.length) empty($("arLog"), "No entries."); else
    $("arLog").appendChild(table(["time", "entry"], L.slice().reverse().map(r => [r.t, r.text]), []));
  // charts
  if (!window.Chart) return;
  const fg = css("--fg"), mu = css("--muted"), ru = css("--rule");
  const leg = { position: "bottom", labels: { color: fg, boxWidth: 10, font: { size: 10 } } };
  const base = { responsive: true, maintainAspectRatio: false, animation: false };
  const axis = (lab, extra) => Object.assign({ title: { display: true, text: lab, color: mu }, ticks: { color: mu }, grid: { color: ru } }, extra || {});
  if (BH.length && $("arBestChart")) {
    const pts = BH.map((r, i) => ({ x: i, y: r.x })); 
    new Chart($("arBestChart"), { type: "line", data: { labels: BH.map(r => r.t), datasets: [{ label: "best FLOP speedup", data: BH.map(r => r.x), borderColor: css("--c1"), backgroundColor: css("--c1"), tension: 0, pointRadius: 3 }] },
      options: { ...base, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => BH[c.dataIndex].arm + ": " + f(c.raw, 1) + "×" } } }, scales: { x: { ticks: { color: mu, maxRotation: 0, autoSkip: true }, grid: { display: false } }, y: axis("FLOP speedup (×)", { beginAtZero: true }) } } });
  }
  const C = ST.curves || [];
  if ($("arCurves")) {
    if (!C.length) { $("arCurves").closest("figure").style.display = "none"; empty($("arCurves").closest("section"), "No curves yet."); }
    else {
      const order = []; B.forEach(r => { if (!order.includes(r.arm)) order.push(r.arm); });
      const rank = a => { const i = order.indexOf(a); return i < 0 ? 999 : i; };
      const keepArms = new Set(order.slice(0, 12)); C.forEach(c => { if (c.is_ref) keepArms.add(c.arm); });
      const shown = C.filter(c => keepArms.has(c.arm) || order.length === 0).sort((a, b) => rank(a.arm) - rank(b.arm));
      const hidden = new Set(C.filter(c => !shown.includes(c)).map(c => c.arm)).size;
      const pal = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6"], col = {}; let k = 0;
      shown.forEach(c => { if (!(c.arm in col)) col[c.arm] = css(pal[k++ % pal.length]); });
      const ds = shown.map(c => { const p = (c.pts || []).filter(q => q[1] > 0);
        return { label: c.arm + " s" + c.seed, arm: c.arm, seed: c.seed, raw: p, data: p.map(q => ({ x: q[1], y: q[2] })), borderColor: col[c.arm], backgroundColor: col[c.arm],
          borderWidth: c.is_ref ? 3 : 1.5, borderDash: c.is_ref ? [6, 4] : [], pointRadius: 2, tension: 0 }; });
      new Chart($("arCurves"), { type: "line", data: { datasets: ds }, options: { ...base, parsing: false, interaction: { mode: "nearest", intersect: true },
        plugins: { legend: leg, tooltip: { callbacks: { label: c => { const d = c.dataset, q = d.raw[c.dataIndex]; return d.arm + " seed " + d.seed + ": step " + q[0] + ", " + f(q[1], 2) + " PF, CE " + f(q[2], 4); } } } },
        scales: { x: axis("cumulative training PFLOPs", { type: "logarithmic" }), y: axis("held-out dense CE") } } });
      if (hidden) $("arCurvesCap").textContent += " " + hidden + " further method(s) not drawn (top 12 by leaderboard order plus references are shown).";
    }
  }
})();
</script>
''' + E + "\n"
JS = JS.replace("__ARDATA__", json.dumps(STATE, ensure_ascii=False).replace("</", "<\\/"))
NAV = S + '  <a href="#autoresearch"><span>8d</span>Auto-research loop</a>\n' + E + "\n"


def ins(before, block):
    global h
    assert h.count(before) == 1, (before, h.count(before))
    h = h.replace(before, block + before, 1)


ins('<div class="part" id="provenance">', TAB)
ins('  <a href="#provenance">', NAV)
i = h.rindex("</script>") + len("</script>")
h = h[:i] + JS + h[i:]
open(H, "w").write(h)
print("built", len(h))
