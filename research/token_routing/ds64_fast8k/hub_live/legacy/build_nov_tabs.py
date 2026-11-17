import json, sys
H = sys.argv[1]
h = open(H).read()
if 'id="selection"' in h: sys.exit("already built")
O = json.load(open("sel_ops.json"))
T = ["contradiction", "nq", "outlier", "oolong"]
def mean2(a, b, rung): return [round((O[a][rung][t] + O[b][rung][t]) / 2, 4) for t in T]
def one(a, rung): return [O[a][rung][t] for t in T]
task8 = {"labels": ["contradiction", "nq", "outlier", "oolong"], "series": [
  ["dense", one("dense", "8k")], ["recipe (router, 2 seeds)", mean2("recipe s0", "recipe s1", "8k")],
  ["per-row router (2 seeds)", mean2("perrow s0", "perrow s1", "8k")], ["position-id reset", one("posid", "8k")],
  ["tail 20%→7%", one("tail20to7", "8k")], ["tail 10%", one("tail10", "8k")]]}
# arm, family, 8k mean, 2k mean, body kept (stage 1, %)
ALL = [
 ["dense", "ref", 0.843, 0.933, 100], ["recipe: per-document router (2 seeds)", "router", 0.846, 0.9205, 20.6],
 ["per-row router (2 seeds)", "router", 0.843, 0.921, 20], ["per-row + floor 1 (2 seeds)", "router", 0.831, 0.9145, 20],
 ["per-row + floor 2 (2 seeds)", "router", 0.8405, 0.9045, 20], ["cross-row step threshold (2 seeds)", "router", 0.8325, 0.9145, None],
 ["position-id reset", "router", 0.820, 0.881, 20.6],
 ["fixed tail 20%→7%", "fixed", 0.781, 0.851, 20.6], ["fixed tail 10%", "fixed", 0.747, 0.815, 10.6],
 ["selective backprop, sample 30% (2 seeds)", "rows", 0.7385, 0.8735, None], ["selective backprop, top 30% (2 seeds)", "rows", 0.6835, 0.7955, None],
 ["oracle per-doc (orc1doc)", "oracle", 0.850, 0.902, 35.5], ["oracle per-doc + random (orc1docr2)", "oracle", 0.867, 0.933, 94],
 ["oracle grown to routed cos 0.95 (orc1batchrc)", "oracle", 0.837, 0.936, 71.3], ["oracle batch + random (orc1batchr2)", "oracle", 0.806, 0.922, 30.2],
 ["oracle batch, 2 cuts (orcbatch)", "oracle", 0.793, 0.824, 34], ["oracle batch (orc1batch)", "oracle", 0.756, 0.804, 19.1],
 ["oracle batch whole-grad, 2 cuts (orcbatchall)", "oracle", 0.736, 0.846, 23], ["oracle per-doc, 2 cuts (orcdoc)", "oracle", 0.716, 0.829, None],
 ["oracle per-row (orc1row)", "oracle", 0.674, 0.816, 29.1]]
LEN = {"labels": ["64k", "128k", "256k ⚠ eval_size 125"], "series": [["recipe, 1/4 epoch (1,696 PF)", [0.046, 0.033, 0.054]], ["router otr10M7, 1/2 epoch", [0.040, 0.042, 0.023]], ["router, full epoch", [0.025, 0.013, -0.005]]]}
SP = [["router, base: layer-4 feats → gnorm@emb ⚠", 0.51, 0.61], ["router, base: layer-8 → gnorm@L8 ⚠", 0.65, 0.67],
 ["router, trained recipe: block 7", 0.54, 0.54], ["router, trained recipe: block 16", 0.57, 0.57],
 ["true: block 3 vs 7", 0.88, 0.88], ["true: block 7 vs 16", 0.57, 0.61], ["true: block 7 vs 24 (falls w/ training)", 0.34, 0.57], ["true: block 7 vs 31", 0.33, 0.39]]
NORM = [["attribution: global oracle", 0.945, "oracle"], ["attribution: per-row quota", 0.94, "oracle"], ["attribution: per-doc quota", 0.915, "oracle"],
 ["routed: per-doc oracle (35%)", 1.25, "routed"], ["routed: batch oracle (19%)", 1.24, "routed"], ["routed: per-row oracle (29%)", 0.81, "routed"],
 ["routed: oracle to cos 0.95 (71%)", 1.007, "routed"], ["routed: recipe 20%→7%", 1.4, "routed"]]
DATA = {"task8": task8, "all": ALL, "len": LEN, "sp": SP, "norm": NORM}

SEL = '''<div class="part" id="selection">
<div class="parthead"><span class="eyebrow">6c · Selection ablations</span><h2 class="ptitle">Which tokens to keep: allocation, position ids, fixed rules, oracles</h2><p class="verdict">Nothing beats the per-document online router at the recipe's budget. A fixed tail loses 0.065 at 8k; renumbering position ids loses 0.026; per-row and cross-row allocation tie at best; gradient-oracle selection only ties dense when it keeps per-document shares or about 70% of the body. The router works through co-adaptation to a stable rule, not by matching the dense gradient.</p></div>
<section>
  <h2>Every selection rule at 1,952 rows/task</h2>
  <div class="figs">
    <figure style="grid-column: 1 / -1"><h3>4-task mean, 8k (primary) and 2k</h3><div class="chart" style="height:620px"><canvas id="selAll" aria-label="all selection rules"></canvas></div>
      <figcaption>Bars sorted within family: references, router variants, fixed rules, row selection (cross-row session), gradient oracles (oracle-selection session; each step a dense backward picks the tokens, ≈2.3–3.4× dense cost per step, so an upper bound, not a method). Dashed line = dense 8k (0.843). eval_size 500 per cell, no-cot; 1 seed unless marked. SE ≈ ±0.02 at 0.6–0.7, ±0.012 at 0.92.</figcaption></figure>
    <figure><h3>8k by task: where the losses sit</h3><div class="chart"><canvas id="selTask" aria-label="8k by task"></canvas></div>
      <figcaption>Outlier carries nearly every loss: fixed tail −0.15 / −0.24, position reset −0.10 vs the recipe. oolong is flat across rules.</figcaption></figure>
    <figure><h3>Body tokens kept (stage 1) vs 8k mean</h3><div class="chart"><canvas id="selKeep" aria-label="kept vs score"></canvas></div>
      <figcaption>Per-document allocation (router, per-doc oracle) sits at or above dense; per-row and batch-level oracles collapse on outlier at 19–34% kept. Only keeping ≈70%+ makes a batch oracle safe.</figcaption></figure>
  </div>
</section>
<section><h2>Per-row token budget (+ per-document floor)</h2><div data-logfig="Per-row token budget"></div></section>
<section><h2>Cross-row allocation and selective backprop</h2><div data-logfig="Cross-row allocation"></div></section>
<section><h2>Renumbered position ids after each cut</h2><div data-logfig="Ablation: renumber position ids"></div></section>
<section><h2>Fixed tail rule: does the model just adapt to the drop shape?</h2><div data-logfig="Tail rule: keep only"></div></section>
<section><h2>Oracle-selection training</h2><div data-logfig="Oracle-selection training"></div></section>
<section><h2>Whole-token oracle under routing (probe, no training)</h2><div data-logfig="Whole-token oracle"></div></section>
</div>
'''
PRIOR = '''<div class="part" id="priorwork">
<div class="parthead"><span class="eyebrow">8b · Prior work</span><h2 class="ptitle">How the recipe compares with published methods</h2><p class="verdict">Mid-stack token dropping with dense inference is prior art; the online gradient-norm router appears new but is a weak headline because random nearly ties it. The strongest claim is the size of the saving: no published long-context fine-tuning method reports more than ≈1.4–2× at parity.</p></div>
<section>
  <h2>Reported savings of the closest methods</h2>
  <div class="figs"><figure><h3>Best reported training saving (×)</h3><div class="chart" style="height:260px"><canvas id="litBar" aria-label="reported savings"></canvas></div>
    <figcaption>Each paper's own headline number, so settings differ (speed, compute or memory); ours is dense FLOPs-to-match at 8k, 1,952 rows/task. Some numbers come from abstracts only; check the paper before quoting.</figcaption></figure></div>
  <div data-logfig="Literature review: efficient long-context"></div>
</section>
</div>
'''
MT11 = '''<section>
  <h2>Length generalization past the 64k training length (128k, 256k)</h2>
  <div class="figs"><figure><h3>Router gain over dense at matched FLOPs, 6-task mean</h3><div class="chart"><canvas id="lenGen" aria-label="gain at 64k-256k"></canvas></div>
    <figcaption>Evaluated at 64k / 128k / 256k with YaRN ×2. eval_size 500 at 64k and 128k, ⚠ 125 at 256k (±0.03–0.045). Single seed.</figcaption></figure></div>
  <div data-logfig="11-task 64k: length-generalization"></div>
</section>
'''
MECH = '''<section>
  <h2>How well does the router rank tokens, and how big is the routed gradient?</h2>
  <div class="figs">
    <figure><h3>Rank correlation (Spearman) of token gradient importance</h3><div class="chart" style="height:340px"><canvas id="rankSp" aria-label="router ranking accuracy"></canvas></div>
      <figcaption>Bars span the range over tasks/checkpoints; ⚠ = base-model probe on 32 rows/task. The router reaches 0.51–0.67 against its own target; even a perfect block-7 ranker would rank the top layers at 0.33–0.39, which is what the block-16 re-selection compensates for. Never measured: the trained router's picks against the true gradient at blocks 16/24/31.</figcaption></figure>
    <figure><h3>Gradient norm relative to the full / dense gradient</h3><div class="chart" style="height:340px"><canvas id="gnorm" aria-label="gradient norm ratio"></canvas></div>
      <figcaption>Purple = attribution subsets at cos 0.95 under the dense forward; orange = routed training, mean over steps. Attribution subsets keep 91–95% of the norm at cos 0.95. In routed training the gradient is not smaller: dropping context raises the loss on nearly solved rows, which then push harder. Adam normalizes step size, so direction matters more than norm.</figcaption></figure>
  </div>
  <div data-logfig="Router ranking accuracy across layers"></div>
</section>
'''
JS = r'''<script>
(() => { if (!window.Chart) return; const DS = __SELDATA__;
  const esc = t => String(t == null ? "" : t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fg = css("--fg"), mu = css("--muted"), ru = css("--rule");
  const PAL = ["--c2", "--c1", "--c3", "--c4", "--c5", "--c7", "--c6", "--dense"];
  const refPlug = { id: "refl", afterDatasetsDraw(ch, _a, o) { const R = (o && o.lines) || []; if (!R.length) return;
    const hz = ch.options.indexAxis === "y", sc = hz ? ch.scales.x : ch.scales.y, a = ch.chartArea, x = ch.ctx;
    x.save(); x.setLineDash([5, 4]); x.lineWidth = 1.5; x.strokeStyle = css("--dense"); x.fillStyle = mu; x.font = "10px " + css("--body");
    R.forEach((r, i) => { const p = sc.getPixelForValue(r.value); x.beginPath();
      if (hz) { x.moveTo(p, a.top); x.lineTo(p, a.bottom); x.stroke(); x.fillText(r.name + " " + r.value, p + 3, a.top + 10 + i * 11); }
      else { x.moveTo(a.left, p); x.lineTo(a.right, p); x.stroke(); x.fillText(r.name + " " + r.value, a.left + 4 + i * 120, p - 3); } }); x.restore(); } };
  const axes = (hz, lo, hi, lab) => { const v = { min: lo, max: hi, title: { display: !!lab, text: lab || "", color: mu }, ticks: { color: mu }, grid: { color: ru } },
      c = { ticks: { color: fg, font: { size: 10 }, autoSkip: false }, grid: { display: false } }; return hz ? { x: v, y: c } : { x: c, y: v }; };
  function bar(el, labels, sets, o = {}) { if (!el) return; const hz = !!o.hz;
    return new Chart(el, { type: "bar", plugins: [refPlug], data: { labels, datasets: sets.map((s, i) => ({ label: s.label, data: s.data,
        backgroundColor: s.colors || css(s.col || PAL[i % PAL.length]), borderRadius: 2, barPercentage: 0.85, categoryPercentage: 0.8 })) },
      options: { indexAxis: hz ? "y" : "x", responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { refl: { lines: o.refs || [] }, legend: { display: sets.length > 1 || !!o.legend, position: "bottom", labels: { color: fg, boxWidth: 10, font: { size: 10 } } },
          tooltip: { callbacks: { label: c => `${c.dataset.label}: ${Array.isArray(c.raw) ? c.raw[0] + "–" + c.raw[1] : c.raw}` } } },
        scales: axes(hz, o.lo, o.hi, o.lab) } }); }
  // per-entry blocks, rendered from the experiment log so they stay current
  let k = 0;
  document.querySelectorAll("[data-logfig]").forEach(host => { const pre = host.dataset.logfig;
    const e = (typeof D_LOG !== "undefined" ? D_LOG : []).find(x => x.title.startsWith(pre));
    if (!e) { host.innerHTML = '<p class="note">(log entry not found)</p>'; return; }
    let s = `<p class="note"><b>${esc(e.status)}</b> · ${esc(e.date)} · ${esc(e.who)}${e.record ? " · record <code>" + esc(e.record) + "</code>" : ""}${e.branch ? " · branch <code>" + esc(e.branch) + "</code>" : ""}</p><p class="lr" style="margin-bottom:10px">${esc(e.result)}</p>`;
    const c = e.chart, id = "lf" + (k++);
    if (c || (e.figs || []).length) { s += '<div class="figs">';
      if (c) { const n = (c.labels || []).length * Math.max(1, (c.series || []).length), hz = (c.labels || []).length > 8 || (c.labels || []).some(l => String(l).length > 12);
        s += `<figure><h3>${esc(c.title)}</h3><div class="chart" style="height:${hz ? Math.max(240, n * 20 + 90) : 300}px"><canvas id="${id}" data-hz="${hz ? 1 : ""}"></canvas></div>${c.unit ? `<figcaption>${esc(c.unit)}</figcaption>` : ""}</figure>`; }
      (e.figs || []).forEach(f => { s += `<figure><h3>${esc(f.cap)}</h3><a href="${esc(f.src)}" target="_blank" rel="noopener"><img src="${esc(f.src)}" loading="lazy" alt="${esc(f.cap)}" style="max-width:100%;height:auto"></a></figure>`; });
      s += "</div>"; }
    (e.tables || []).forEach(t => { s += `<h3 style="margin-top:14px">${esc(t.title)}</h3><div class="tbl"><table><tr>${t.cols.map((x, i) => `<th${i ? ' class="n"' : ""}>${esc(x)}</th>`).join("")}</tr>${t.rows.map(r => `<tr>${r.map((v, i) => `<td${i ? ' class="n"' : ""}>${esc(v == null ? "–" : v)}</td>`).join("")}</tr>`).join("")}</table></div>${t.note ? `<p class="note">${esc(t.note)}</p>` : ""}`; });
    host.innerHTML = s;
    if (c) { const el = document.getElementById(id); bar(el, c.labels, (c.series || []).map(x => ({ label: x.name, data: x.values })), { hz: !!el.dataset.hz, lo: c.ymin, hi: c.ymax, refs: c.refs, legend: (c.series || []).length > 0 }); } });
  // summary figures
  const FC = { ref: "--dense", router: "--c2", fixed: "--c1", rows: "--c3", oracle: "--c4" };
  const A = DS.all, el = document.getElementById("selAll");
  if (el) new Chart(el, { type: "bar", plugins: [refPlug], data: { labels: A.map(a => a[0]), datasets: [
      { label: "8k mean (primary)", data: A.map(a => a[2]), backgroundColor: A.map(a => css(FC[a[1]])), borderRadius: 2 },
      { label: "2k mean", data: A.map(a => a[3]), backgroundColor: A.map(a => css(FC[a[1]]) + "66"), borderRadius: 2 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { refl: { lines: [{ name: "dense 8k", value: 0.843 }] }, legend: { position: "bottom", labels: { color: fg, boxWidth: 10, font: { size: 10 },
        generateLabels: () => [["dense", "ref"], ["router variants", "router"], ["fixed rules", "fixed"], ["row selection", "rows"], ["gradient oracles", "oracle"]].map(([t, f]) => ({ text: t, fillStyle: css(FC[f]), strokeStyle: css(FC[f]), fontColor: fg }))
          .concat([{ text: "solid = 8k, pale = 2k", fillStyle: "transparent", strokeStyle: "transparent", fontColor: mu }]) } } },
      scales: axes(true, 0.6, 0.96, "4-task mean score") } });
  bar(document.getElementById("selTask"), DS.task8.labels, DS.task8.series.map(([l, v]) => ({ label: l, data: v })), { lo: 0.5, hi: 1.0, lab: "8k f1 / score" });
  const ke = document.getElementById("selKeep");
  if (ke) new Chart(ke, { type: "scatter", plugins: [refPlug], data: { datasets: ["router", "fixed", "oracle"].map(f => ({ label: { router: "router variants", fixed: "fixed rules", oracle: "gradient oracles" }[f],
      data: A.filter(a => a[1] === f && a[4] != null).map(a => ({ x: a[4], y: a[2], n: a[0] })), backgroundColor: css(FC[f]), borderColor: css(FC[f]), pointRadius: 5 })) },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { refl: { lines: [{ name: "dense", value: 0.843 }] },
        legend: { position: "bottom", labels: { color: fg, boxWidth: 10, font: { size: 10 } } }, tooltip: { callbacks: { label: c => `${c.raw.n}: ${c.raw.y} at ${c.raw.x}% kept` } } },
      scales: { x: { type: "logarithmic", min: 8, max: 100, title: { display: true, text: "body tokens kept at block 7 (%, log)", color: mu }, ticks: { color: mu, callback: v => [10, 20, 30, 50, 70, 100].includes(v) ? v : "" }, grid: { color: ru } },
        y: { min: 0.65, max: 0.88, title: { display: true, text: "8k 4-task mean", color: mu }, ticks: { color: mu }, grid: { color: ru } } } } });
  bar(document.getElementById("lenGen"), DS.len.labels, DS.len.series.map(([l, v]) => ({ label: l, data: v })), { lo: -0.02, hi: 0.07, lab: "Δ vs dense at matched FLOPs", refs: [{ name: "dense", value: 0 }] });
  bar(document.getElementById("rankSp"), DS.sp.map(s => s[0]), [{ label: "Spearman (range)", data: DS.sp.map(s => s[1] === s[2] ? [s[1] - 0.008, s[2] + 0.008] : [s[1], s[2]]),
      colors: DS.sp.map(s => css(s[0].startsWith("router") ? "--c2" : "--c6")) }], { hz: true, lo: 0, hi: 1, lab: "Spearman" });
  bar(document.getElementById("gnorm"), DS.norm.map(s => s[0]), [{ label: "norm vs full / dense", data: DS.norm.map(s => s[1]), colors: DS.norm.map(s => css(s[2] === "oracle" ? "--c4" : "--c1")) }], { hz: true, lo: 0, hi: 1.5, lab: "ratio", refs: [{ name: "equal", value: 1 }] });
  bar(document.getElementById("litBar"), ["LeMo / Jenga (speed)", "Centrifuge (end-to-end)", "TokenDrop", "SparseLoRA (compute)", "ours: FLOPs-to-match, 8k"], [{ label: "reported saving (×)", data: [1.36, 1.5, 1.5, 2.2, 4.5],
      colors: ["--c6", "--c6", "--c6", "--c6", "--c2"].map(css) }], { hz: true, lo: 0, hi: 5, lab: "× saving", refs: [{ name: "parity", value: 1 }] });
})();
</script>
'''
JS = JS.replace("__SELDATA__", json.dumps(DATA, ensure_ascii=False))
def ins(before, block, count=1):
    global h; assert h.count(before) == count, (before, h.count(before)); h = h.replace(before, block + before, 1)
ins('<div class="part" id="ruledout">', SEL)
ins('<div class="part" id="provenance">', PRIOR)
ins('</div>\n<div class="part" id="models">', MT11)
ins('</div>\n<div class="part" id="gradmaps">', MECH)
ins('  <a href="#ruledout">', '  <a href="#selection"><span>6c</span>Selection ablations</a>\n')
ins('  <a href="#provenance">', '  <a href="#priorwork"><span>8b</span>Prior work</a>\n')
i = h.rindex("</script>") + len("</script>"); h = h[:i] + "\n" + JS + h[i:]
open(H, "w").write(h); print("built", len(h))
