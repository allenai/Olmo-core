/* overview: map of the page, built from each part's own verdict line */
(() => {
  const build = () => { const host = document.getElementById("ovMap"); if (!host || host.childElementCount) return;
    const G = JSON.parse(document.getElementById("hubgroups").textContent), parts = [...document.querySelectorAll(".part")];
    const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    const cut = (t, n) => { t = t.replace(/\s+/g, " ").trim(); if (t.length <= n) return t; return t.slice(0, n).replace(/\s+\S*$/, "").replace(/[,;:]$/, "") + " …"; };
    G.forEach((g, i) => { const ps = parts.filter(p => p.dataset.group === g.id).sort((a, b) => a.dataset.order - b.dataset.order); if (!ps.length) return;
      const d = document.createElement("div"); d.className = "mapgrp";
      d.innerHTML = `<h3><a href="#${ps[0].id}">${i + 1} · ${esc(g.title)}</a></h3><p class="note">${esc(g.blurb)}</p><ul>` + ps.map((p, j) => {
        const v = p.querySelector(".parthead .verdict"); return `<li><a href="#${p.id}">${i + 1}.${j + 1} ${esc(p.dataset.short || p.id)}</a> ${v ? esc(cut(v.textContent, 190)) : ""}</li>`; }).join("") + "</ul>";
      host.appendChild(d); }); };
  if (document.readyState === "complete") build(); else window.addEventListener("load", build);
})();
