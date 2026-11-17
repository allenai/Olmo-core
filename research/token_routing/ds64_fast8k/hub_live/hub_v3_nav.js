/* PAIL-style navigation: site title, row 1 = groups (Overview first), row 2 = the active group's sections; pager; hash router.
   Parts declare data-group / data-order / data-short (and optional data-badge); groups come from #hubgroups. */
(() => {
  const G = JSON.parse(document.getElementById("hubgroups").textContent);
  const parts = [...document.querySelectorAll(".part")], byId = {}; parts.forEach(p => { byId[p.id] = p; });
  const t1 = document.querySelector("#toc .tabs1"), t2 = document.querySelector("#toc .tabs2"), cur = document.querySelector("#sitehead .cur");
  const label = p => p.dataset.short || (p.querySelector(".ptitle") || {}).textContent || p.id;
  const groups = G.map(g => ({ ...g, parts: parts.filter(p => p.dataset.group === g.id).sort((a, b) => a.dataset.order - b.dataset.order) }));
  const loose = parts.filter(p => p.id !== "overview" && !G.some(g => g.id === p.dataset.group));
  if (loose.length) groups.push({ id: "g-other", title: "More", short: "More", blurb: "", parts: loose });
  const live = groups.filter(g => g.parts.length);
  const flat = [byId.overview, ...live.flatMap(g => g.parts)].filter(Boolean);
  const num = {}; live.forEach((g, i) => g.parts.forEach((p, j) => { num[p.id] = (i + 1) + "." + (j + 1); }));
  const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  t1.innerHTML = (byId.overview ? `<a href="#overview" data-g="home">Overview</a>` : "") + live.map(g => `<a href="#${g.parts[0].id}" data-g="${g.id}" title="${esc(g.title)}">${esc(g.short || g.title)}</a>`).join("");
  const gOf = id => (id === "overview" ? "home" : (live.find(g => g.parts.some(p => p.id === id)) || {}).id);
  const link = p => `<a href="#${p.id}" data-p="${p.id}">${esc(label(p))}${p.dataset.badge ? `<i class="badge">${esc(p.dataset.badge)}</i>` : ""}</a>`;
  /* pager */
  flat.forEach((p, i) => { const f = document.createElement("nav"); f.className = "pager"; f.setAttribute("aria-label", "Pages");
    const t = q => (num[q.id] ? num[q.id] + " " : "") + label(q);
    f.insertAdjacentHTML("beforeend", i > 0 ? `<a href="#${flat[i - 1].id}">← ${esc(t(flat[i - 1]))}</a>` : "<span></span>");
    if (i < flat.length - 1) f.insertAdjacentHTML("beforeend", `<a href="#${flat[i + 1].id}">${esc(t(flat[i + 1]))} →</a>`);
    p.appendChild(f); });
  /* section bars: stable ids <part>--<slug> on each bar's heading, plus expand / collapse all */
  const used = new Set([...document.querySelectorAll("[id]")].map(e => e.id));
  parts.forEach(p => { const hs = [...p.querySelectorAll(":scope > details.sec > summary > h2, :scope > section > h2")];
    hs.forEach(h => { if (h.id) return; let s = p.id + "--" + (h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "s"), k = s, n = 2;
      while (used.has(k)) k = s + "-" + n++; h.id = k; used.add(k); });
    const bars = p.querySelectorAll(":scope > details.sec"); if (bars.length < 3) return;
    const c = document.createElement("div"); c.className = "secctl";
    c.innerHTML = '<button type="button" data-a="open">Expand all</button><button type="button" data-a="close">Collapse all</button>';
    c.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; bars.forEach(d => { d.open = b.dataset.a === "open"; }); window.dispatchEvent(new Event("resize")); });
    const ph = p.querySelector(".parthead"); ph.parentNode.insertBefore(c, ph.nextSibling); });
  const show = (id, el) => { if (!byId[id]) id = "overview";
    parts.forEach(p => { p.hidden = p.id !== id; });
    const g = gOf(id), grp = live.find(x => x.id === g);
    t1.querySelectorAll("a").forEach(a => a.classList.toggle("on", a.dataset.g === g));
    t2.innerHTML = grp ? grp.parts.map(link).join("") : "";
    t2.querySelectorAll("a").forEach(a => a.classList.toggle("on", a.dataset.p === id));
    cur.textContent = " · " + (id === "overview" ? "Overview" : label(byId[id]));
    const act = t2.querySelector("a.on"); if (act && act.scrollIntoView) act.scrollIntoView({ block: "nearest", inline: "center" });
    for (let d = el && el.closest("details"); d; d = d.parentElement && d.parentElement.closest("details")) d.open = true;
    window.dispatchEvent(new Event("resize"));
    if (el) el.scrollIntoView(); else { window.scrollTo(0, 0); setTimeout(() => window.scrollTo(0, 0), 60); } };
  const route = () => { const h = decodeURIComponent(location.hash.slice(1));
    const grp = live.find(g => g.id === h); if (grp) return show(grp.parts[0].id);
    if (h && !byId[h]) { const el = document.getElementById(h), p = el && el.closest(".part"); if (p) return show(p.id, el); }
    show(h || "overview"); };
  window.addEventListener("hashchange", route); route();
  /* folds: re-measure scroll hints and charts when one opens */
  document.addEventListener("toggle", () => window.dispatchEvent(new Event("resize")), true);
})();
