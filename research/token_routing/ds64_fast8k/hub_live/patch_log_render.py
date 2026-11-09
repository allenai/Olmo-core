import sys
JS = r'''  const escA = t => esc(t).replace(/"/g, "&quot;");
  const SC = ["var(--c2)", "var(--c1)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c7)", "var(--c6)"];
  const cellN = (v, i) => `<td${i ? ' class="n"' : ""}>${esc(v == null ? "–" : v)}</td>`;
  function xtab(t) { return `<div class="lxt"><div class="lxh">${esc(t.title || "")}</div><div class="tbl"><table><tr>${(t.cols || []).map((c, i) => `<th${i ? ' class="n"' : ""}>${esc(c)}</th>`).join("")}</tr>${(t.rows || []).map(r => `<tr>${r.map(cellN).join("")}</tr>`).join("")}</table></div>${t.note ? `<p class="note">${esc(t.note)}</p>` : ""}</div>`; }
  function xchart(c) { const lo = c.ymin ?? 0, hi = c.ymax ?? 1, pc = v => Math.max(0, Math.min(100, (v - lo) / (hi - lo) * 100));
    const S = c.series || [], R = c.refs || [];
    const refl = R.map(r => `<span class="rf" style="left:${pc(r.value)}%" title="${escA(r.name + " " + r.value)}"></span>`).join("");
    let rows = ""; (c.labels || []).forEach((lab, li) => S.forEach((s, si) => { const v = s.values[li], p = v == null ? 0 : pc(v);
      const lbl = v == null ? "" : (p > 84 ? `<span class="v in" style="right:calc(${100 - p}% + 4px)">${esc(v)}</span>` : `<span class="v" style="left:calc(${p}% + 4px)">${esc(v)}</span>`);
      rows += `<div class="hl">${si === 0 ? esc(lab) : ""}</div><div class="tr${si === S.length - 1 ? " last" : ""}">${v == null ? "" : `<span class="b" style="width:${p}%;background:${SC[si % SC.length]}"></span>`}${refl}${lbl}</div>`; }));
    const leg = (S.length > 1 ? S.map((s, si) => `<span><i style="background:${SC[si % SC.length]}"></i>${esc(s.name)}</span>`).join("") : "") + R.map(r => `<span><i class="rfi"></i>${esc(r.name)} ${esc(r.value)}</span>`).join("");
    return `<div class="lxt"><div class="lxh">${esc(c.title || "")} <span class="note">(${c.unit ? esc(c.unit) + "; " : ""}axis ${lo}–${hi})</span></div><div class="hb">${rows}</div>${leg ? `<div class="hleg">${leg}</div>` : ""}</div>`; }
  function extras(e) { const T = e.tables || [], F = e.figs || []; if (!T.length && !e.chart && !F.length) return "";
    const what = [e.chart ? "chart" : "", T.length ? T.length + (T.length > 1 ? " tables" : " table") : "", F.length ? F.length + (F.length > 1 ? " figures" : " figure") : ""].filter(Boolean).join(", ");
    return `<details class="lx"><summary>Show ${what}</summary>${e.chart ? xchart(e.chart) : ""}${T.map(xtab).join("")}${F.map(f => `<figure class="lxf"><a href="${escA(f.src)}" target="_blank" rel="noopener"><img src="${escA(f.src)}" loading="lazy" alt="${escA(f.cap || "")}"></a><figcaption class="note">${esc(f.cap || "")}</figcaption></figure>`).join("")}</details>`; }
'''
CSS = r'''.lx { margin-top: 6px; } .lx > summary { cursor: pointer; color: var(--c2); font-size: .85rem; font-weight: 600; }
.lxt { margin: 10px 0 4px; } .lxh { font-weight: 600; font-size: .85rem; margin-bottom: 4px; }
.hb { display: grid; grid-template-columns: minmax(80px, 34%) minmax(0, 1fr); gap: 2px 8px; font-size: .78rem; align-items: center; max-width: 760px; }
.hb .hl { color: var(--muted); text-align: right; line-height: 1.2; }
.hb .tr { position: relative; height: 15px; background: color-mix(in srgb, var(--rule) 45%, transparent); }
.hb .tr.last { margin-bottom: 4px; } .hb .b { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 0 2px 2px 0; }
.hb .rf { position: absolute; top: -2px; bottom: -2px; border-left: 2px dashed var(--dense); opacity: .7; }
.hb .v { position: absolute; top: 0; line-height: 15px; font-family: var(--mono); font-size: .72rem; color: var(--fg); white-space: nowrap; }
.hb .v.in { color: var(--surface); }
.hleg { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: .75rem; color: var(--muted); margin-top: 4px; }
.hleg i { display: inline-block; width: 10px; height: 10px; margin-right: 4px; vertical-align: -1px; border-radius: 2px; }
.hleg i.rfi { width: 0; height: 12px; border-left: 2px dashed var(--dense); border-radius: 0; }
.lxf { margin: 10px 0; } .lxf img { max-width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 4px; }
'''
for p in sys.argv[1:]:
    h = open(p).read()
    if "function extras(e)" in h: print("already", p); continue
    a = '  function draw() { const rows = L.filter('
    assert h.count(a) == 1, p; h = h.replace(a, JS + a)
    b = '<p class="lr">${esc(hum(e.result))}</p></div></li>'
    assert h.count(b) == 1, p; h = h.replace(b, '<p class="lr">${esc(hum(e.result))}</p>${extras(e)}</div></li>')
    c = '.lr { color: var(--fg); font-size: .9rem; max-width: 90ch; margin: 0; }\n'
    assert h.count(c) == 1, p; h = h.replace(c, c + CSS)
    open(p, "w").write(h); print("patched", p)
