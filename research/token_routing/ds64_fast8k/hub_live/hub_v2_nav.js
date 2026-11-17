/* two-level navigation + pager + hash router. Parts declare data-group / data-order / data-short; groups come from #hubgroups. */
(() => {
  const G = JSON.parse(document.getElementById("hubgroups").textContent);
  const parts = [...document.querySelectorAll(".part")], byId = {}; parts.forEach(p => { byId[p.id] = p; });
  const tocin = document.querySelector("#toc .tocin"), secbar = document.querySelector("#toc .secbar");
  const label = p => p.dataset.short || (p.querySelector(".ptitle") || {}).textContent || p.id;
  const groups = G.map(g => ({ ...g, parts: parts.filter(p => p.dataset.group === g.id).sort((a, b) => a.dataset.order - b.dataset.order) }));
  const loose = parts.filter(p => p.id !== "overview" && !G.some(g => g.id === p.dataset.group));
  if (loose.length) groups.push({ id: "g-other", title: "More", blurb: "", parts: loose });
  const live = groups.filter(g => g.parts.length);
  const flat = [byId.overview, ...live.flatMap(g => g.parts)].filter(Boolean);
  const num = {}; live.forEach((g, i) => g.parts.forEach((p, j) => { num[p.id] = (i + 1) + "." + (j + 1); }));
  const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const link = p => `<a href="#${p.id}" data-p="${p.id}"><span>${num[p.id]}</span>${esc(label(p))}${p.dataset.badge ? `<i class="badge">${esc(p.dataset.badge)}</i>` : ""}</a>`;
  let html = byId.overview ? `<div class="grp home" data-g="home"><a class="gh" href="#overview" data-p="overview">Overview</a></div>` : "";
  live.forEach((g, i) => { html += `<div class="grp" data-g="${g.id}"><a class="gh" href="#${g.parts[0].id}"><span>${i + 1}</span><i class="gl">${esc(g.title)}</i><i class="gs">${esc(g.short || g.title)}</i></a><div class="secs">${g.parts.map(link).join("")}</div></div>`; });
  tocin.innerHTML = html;
  const gOf = id => (id === "overview" ? "home" : (live.find(g => g.parts.some(p => p.id === id)) || {}).id);
  /* pager */
  flat.forEach((p, i) => { const f = document.createElement("nav"); f.className = "pager"; f.setAttribute("aria-label", "Pages");
    const t = q => (num[q.id] ? num[q.id] + " " : "") + label(q);
    f.insertAdjacentHTML("beforeend", i > 0 ? `<a href="#${flat[i - 1].id}">← ${esc(t(flat[i - 1]))}</a>` : "<span></span>");
    if (i < flat.length - 1) f.insertAdjacentHTML("beforeend", `<a href="#${flat[i + 1].id}">${esc(t(flat[i + 1]))} →</a>`);
    p.appendChild(f); });
  /* in-page jump chips from the section headings (stable ids: <part>--<slug>) */
  const used = new Set([...document.querySelectorAll("[id]")].map(e => e.id));
  parts.forEach(p => { if (p.id === "overview") return;
    const hs = [...p.querySelectorAll(":scope > section > h2")]; if (hs.length < 3) return;
    const as = hs.map(h => { if (!h.id) { let s = p.id + "--" + (h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "s"), k = s, n = 2;
        while (used.has(k)) k = s + "-" + n++; h.id = k; used.add(k); }
      return `<a href="#${h.id}">${esc(h.textContent.trim())}</a>`; });
    const j = document.createElement("nav"); j.className = "jump"; j.setAttribute("aria-label", "On this page");
    j.innerHTML = "<span>On this page</span>" + as.join(""); const ph = p.querySelector(".parthead"); ph.parentNode.insertBefore(j, ph.nextSibling); });
  const links = [...document.querySelectorAll("nav.toc a[data-p]")];
  const show = (id, el) => { if (!byId[id]) id = "overview";
    parts.forEach(p => { p.hidden = p.id !== id; });
    const g = gOf(id);
    links.forEach(a => a.classList.toggle("on", a.dataset.p === id));
    tocin.querySelectorAll(".grp").forEach(d => d.classList.toggle("on", d.dataset.g === g));
    const on = tocin.querySelector(`.grp[data-g="${g}"] .secs`); secbar.innerHTML = on ? on.innerHTML : "";
    secbar.querySelectorAll("a").forEach(a => a.classList.toggle("on", a.dataset.p === id));
    const act = secbar.querySelector("a.on") || tocin.querySelector(".grp.on"); if (act && act.scrollIntoView) act.scrollIntoView({ block: "nearest", inline: "center" });
    for (let d = el && el.closest("details"); d; d = d.parentElement && d.parentElement.closest("details")) d.open = true;
    window.dispatchEvent(new Event("resize"));
    if (el) el.scrollIntoView(); else window.scrollTo(0, 0); };
  const route = () => { const h = decodeURIComponent(location.hash.slice(1));
    const grp = live.find(g => g.id === h); if (grp) return show(grp.parts[0].id);
    if (h && !byId[h]) { const el = document.getElementById(h), p = el && el.closest(".part"); if (p) return show(p.id, el); }
    show(h || "overview"); };
  window.addEventListener("hashchange", route); route();
  /* folded tables: re-measure scroll hints when a fold opens */
  document.addEventListener("toggle", () => window.dispatchEvent(new Event("resize")), true);
})();
