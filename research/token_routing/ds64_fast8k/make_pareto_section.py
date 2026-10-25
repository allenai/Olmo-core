"""Cross-task Pareto section of the tokdrop_lr artifact, from collect_pareto_points.py's JSON.

For every rule run and every (task, rung) cell: the score, and its delta against the dense curve (LR 5e-6 dense
runs on the same mix, any data size) interpolated linearly in log x at the run's x, where x is training PF
(``flop_meter/actual_pflops``) or train wall-clock seconds (same cluster only). The section plots, per x-axis,
the mean score over the cells against the dense curve, the WORST-cell delta (a universal rule needs it >= 0), and
one small panel per cell. Writes an HTML fragment that replaces the text between
``<!--PARETO-BEGIN-->`` and ``<!--PARETO-END-->`` in the page.

    python research/token_routing/ds64_fast8k/make_pareto_section.py --points results/pareto_points.json --page PAGE.html
"""

import argparse
import json
import math
import re

CELLS = [(t, r) for r in ("8k", "2k") for t in ("contradiction", "nq", "outlier", "oolong")]
#: the dense reference is, per cell, the BEST of these LRs' curves at the same x (no single LR wins every cell:
#: at 122 steps LR 2e-6 gives outlier@2k 0.830 vs 0.557 at 5e-6, while 5e-6 wins contradiction@8k).
DENSE_LRS = (2e-6, 5e-6)
DENSE_LR = 5e-6  # the plotted dense line


def family(arm):
    if arm == "dense":
        return "dense"
    if arm == "densepf":
        return "dense (pad-free path)"
    if re.match(r"otr\d+M\d+", arm):
        return "online router, mid-stack drop"
    if re.match(r"otr\d+L\d+", arm):
        return "online router, input drop"
    if re.match(r"dh\d+M\d+", arm):
        return "hidden-state-change rule, mid-stack"
    if re.match(r"(rnd|fl)\d+", arm):
        return "random / first-last (router's budget)"
    if re.search(r"-(sk\d+|ldl?\d+)$", arm):
        return "+ train-time layer skip"
    if arm.startswith("tier"):
        return "two-tier (gold + decoys at a larger budget)"
    if arm.startswith("bfd"):
        return "gold + decoys whole"
    if arm.startswith("gfl"):
        return "gold only whole (shortcut)"
    if arm.startswith("gb"):
        return "gold-blind"
    return "other"


def interp(curve, x):
    # runs at the same PF (e.g. the Lambda and Beaker copies of one dense point) are averaged into one point
    agg = {}
    for a, b in curve:
        agg.setdefault(round(math.log(a), 6), []).append(b)
    pts = sorted((k, sum(v) / len(v)) for k, v in agg.items())
    lx = math.log(x)
    if len(pts) == 1:
        return pts[0][1], True
    if lx <= pts[0][0]:
        (a, fa), (b, fb), ext = pts[0], pts[1], lx < pts[0][0]
    elif lx >= pts[-1][0]:
        (a, fa), (b, fb), ext = pts[-2], pts[-1], lx > pts[-1][0]
    else:
        i = max(j for j in range(len(pts)) if pts[j][0] <= lx)
        (a, fa), (b, fb), ext = pts[i], pts[i + 1], False
    return fa + (fb - fa) * (lx - a) / (b - a), ext


def build(d, mix="mt4", cells=None):
    cells = cells or CELLS
    runs, pts = d["runs"], d["points"]
    score = {}
    for p in pts:
        score.setdefault(p["run"], {})[(p["task"], p["rung"])] = (p["score"], p["eval_size"])
    out = []
    for xkey in ("pf", "train_seconds"):
        dense = [r for r in runs.values() if r["arm"] == "dense" and r.get(xkey) and r.get("mix", "mt4") == mix
                 and any(abs(r["lr"] - l) < 1e-12 for l in DENSE_LRS)]
        for r in runs.values():
            if r["dense"] and abs(r["lr"] - DENSE_LR) > 1e-12:
                continue
            if "~lrbbug" in r["arm"] or not r.get(xkey) or r["run"] not in score or r.get("mix", "mt4") != mix:
                continue
            same = [q for q in dense if xkey == "pf" or q["cluster"] == r["cluster"]]
            row = dict(run=r["run"], arm=r["arm"], family=family(r["arm"]), x=r[xkey], xkey=xkey,
                       rows=r["rows"], epochs=r.get("epochs", 1), cluster=r["cluster"], cells={})
            for c in cells:
                if c not in score[r["run"]]:
                    continue
                v, n = score[r["run"]][c]
                cell = dict(v=v, n=n)
                best = None
                for lr in DENSE_LRS:  # envelope: the better dense LR at this x
                    curve = sorted((q[xkey], score[q["run"]][c][0]) for q in same
                                   if abs(q["lr"] - lr) < 1e-12 and q["run"] in score and c in score[q["run"]])
                    if len(curve) >= 2 and not r["dense"]:
                        dv, ext = interp(curve, r[xkey])
                        if best is None or dv > best[0]:
                            best = (dv, ext, lr)
                if best:
                    cell.update(dense=round(best[0], 4), delta=round(v - best[0], 4), extrap=best[1], dense_lr=best[2])
                row["cells"][f"{c[0]}@{c[1]}"] = cell
            ds = [c["delta"] for c in row["cells"].values() if "delta" in c]
            vs = [c["v"] for c in row["cells"].values()]
            row["complete"] = len(row["cells"]) == len(cells)
            row["mean"] = round(sum(vs) / len(vs), 4) if vs else None
            if ds and row["complete"]:
                row["mean_delta"], row["worst_delta"] = round(sum(ds) / len(ds), 4), round(min(ds), 4)
            out.append(row)
    return out


FRAGMENT = r"""<!--PARETO__SFX__-BEGIN-->
<section id="pareto__SFX__">
  <h2>__TITLE__</h2>
  <p>__INTRO__</p>
  <div class="ckpick" role="group" aria-label="x axis">
    <button id="px-pf__SFX__" aria-pressed="true">x = training PF</button>
    <button id="px-wall__SFX__" aria-pressed="false">x = train wall-clock (Lambda A100, 2 GPUs)</button>
  </div>
  <div class="legend" id="pareto-legend__SFX__"></div>
  <div class="charts3" style="overflow-x:auto"><div class="panel" id="pareto-mean__SFX__"></div><div class="panel" id="pareto-worst__SFX__"></div></div>
  <h3>Per task and rung</h3>
  <div class="charts" id="pareto-cells__SFX__"></div>
  <h3>Frontier table (sorted by worst-cell Δ vs dense, matched PF)</h3>
  <div class="tbl"><table id="pareto-table__SFX__"></table></div>
  <p class="note">Δ = score − dense at the run's x (per cell, the better of dense LR 2e-6 and 5e-6, each interpolated linearly in log x; a curve needs ≥2 points); "(ext)" = outside the dense range. Run-to-run noise from 3 repeat runs is about ±0.02–0.03 per cell, so a worst-cell Δ within −0.03 is a tie. Updated __UPDATED__.</p>
</section>
<script>
(function () {
const PR = __DATA__;
const PCELLS = __CELLS__;
const FAMC = {"dense": "var(--dense)", "two-tier (gold + decoys at a larger budget)": "var(--teal)", "gold + decoys whole": "var(--warn)",
              "gold-blind": "var(--good)", "gold only whole (shortcut)": "var(--bad)", "+ train-time layer skip": "var(--bar)", "online router, mid-stack drop": "#7b4fd6",
              "online router, input drop": "#b48ce8", "random / first-last (router's budget)": "#8a8f8c", "hidden-state-change rule, mid-stack": "#c2410c",
              "dense (pad-free path)": "var(--dense)", "other": "var(--muted)"};
document.getElementById("pareto-legend__SFX__").innerHTML = Object.entries(FAMC).filter(([f]) => PR.some(r => r.family === f))
  .map(([f, c]) => `<span><span class="sw" style="background:${c}"></span>${f}</span>`).join("");
function pmark(r, x, y, sz) {
  const c = FAMC[r.family] || "var(--muted)", rad = sz * (r.rows > 488 ? 1.45 : 1);
  const fill = r.epochs > 1 ? "var(--surface)" : c;
  const tip = `<title>${r.arm} · ${r.rows} rows/task${r.epochs > 1 ? " × " + r.epochs + " epochs" : ""} · ${r.cluster}\n` +
    Object.entries(r.cells).map(([k, v]) => `${k}: ${v.v.toFixed(3)}${v.delta !== undefined ? " (Δ " + (v.delta > 0 ? "+" : "") + v.delta.toFixed(3) + (v.extrap ? " ext" : "") + ")" : ""}`).join("\n") + `</title>`;
  return `<circle cx="${x}" cy="${y}" r="${rad}" fill="${fill}" stroke="${c}" stroke-width="1.8">${tip}</circle>`;
}
function axes(W, H, L, R, T, B, xlo, xhi, ylo, yhi, xlab, ylab, yticks, zero) {
  const xs = v => L + (Math.log(v) - Math.log(xlo)) / (Math.log(xhi) - Math.log(xlo)) * (W - L - R);
  const ys = v => T + (yhi - Math.min(Math.max(v, ylo), yhi)) / (yhi - ylo) * (H - T - B);
  let g = "";
  yticks.forEach(v => { g += `<line x1="${L}" x2="${W - R}" y1="${ys(v)}" y2="${ys(v)}" stroke="var(--rule)" stroke-width="${zero && v === 0 ? 1.8 : 1}"/><text x="${L - 5}" y="${ys(v) + 3}" text-anchor="end">${v.toFixed(2)}</text>`; });
  const dec = Math.pow(10, Math.floor(Math.log10(xlo)));
  for (let m = dec; m <= xhi * 1.01; m *= 10) [1, 2, 5].forEach(k => { const v = m * k; if (v >= xlo && v <= xhi) g += `<line x1="${xs(v)}" x2="${xs(v)}" y1="${T}" y2="${H - B}" stroke="var(--rule)" stroke-width="0.6"/><text x="${xs(v)}" y="${H - B + 13}" text-anchor="middle">${v >= 1000 ? (v / 1000) + "k" : v}</text>`; });
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${xlab}</text><text transform="translate(11 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${ylab}</text>`;
  return {g, xs, ys};
}
function drawPareto(xkey) {
  const rs = PR.filter(r => r.xkey === xkey && (xkey === "pf" || r.cluster.startsWith("lambda")));
  const xlab = xkey === "pf" ? "training PF (log)" : "train wall-clock seconds (log)";
  const xsAll = rs.map(r => r.x), xlo = Math.min(...xsAll) * 0.85, xhi = Math.max(...xsAll) * 1.15;
  const dense = rs.filter(r => r.family === "dense").sort((a, b) => a.x - b.x);
  // mean score
  const full = rs.filter(r => r.complete), ms = full.map(r => r.mean);
  const ylo = Math.floor(Math.min(...ms) * 20) / 20 - 0.02, yhi = Math.ceil(Math.max(...ms) * 20) / 20 + 0.02;
  let W = 760, H = 320, A = axes(W, H, 48, 16, 12, 34, xlo, xhi, ylo, yhi, xlab, "mean score, 4 tasks × 2 rungs",
                                 [0, 1, 2, 3, 4].map(i => ylo + i * (yhi - ylo) / 4), false);
  let g = A.g; const dd = dense.filter(r => r.complete);
  if (dd.length > 1) g += `<polyline fill="none" stroke="var(--dense)" stroke-width="2.2" points="${dd.map(r => A.xs(r.x) + "," + A.ys(r.mean)).join(" ")}"/>`;
  full.forEach(r => { g += pmark(r, A.xs(r.x), A.ys(r.mean), 4.2); });
  document.getElementById("pareto-mean__SFX__").innerHTML = `<h4>Mean score vs ${xkey === "pf" ? "training FLOPs" : "wall-clock"} · up-and-left of the dense line = better</h4><svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="mean score vs ${xlab}">${g}</svg>`;
  // worst-cell delta
  const wr = rs.filter(r => r.worst_delta !== undefined), wv = wr.map(r => r.worst_delta);
  const wlo = Math.min(-0.15, Math.floor(Math.min(...wv) * 20) / 20), whi = Math.max(0.1, Math.ceil(Math.max(...wv) * 20) / 20);
  A = axes(W, H, 48, 16, 12, 34, xlo, xhi, wlo, whi, xlab, "worst cell: score − dense at same x",
           [wlo, wlo / 2, 0, whi / 2, whi].map(v => Math.round(v * 100) / 100), true);
  g = A.g + `<text x="${54}" y="${A.ys(0) - 5}" style="fill:var(--fg)">beats dense in every cell ↑</text>`;
  g += `<rect x="48" y="${A.ys(0)}" width="${W - 64}" height="${A.ys(-0.03) - A.ys(0)}" fill="var(--chip-bg)" opacity="0.7"><title>within run-to-run noise (−0.03..0)</title></rect>`;
  wr.forEach(r => { g += pmark(r, A.xs(r.x), A.ys(r.worst_delta), 4.2); });
  const top = wr.slice().sort((a, b) => b.worst_delta - a.worst_delta).slice(0, 6);
  top.forEach(r => { g += `<text x="${A.xs(r.x) + 7}" y="${A.ys(r.worst_delta) - 6}" style="fill:var(--fg)">${r.arm}${r.rows > 488 ? " ·" + r.rows : ""}${r.epochs > 1 ? " ×" + r.epochs : ""}</text>`; });
  document.getElementById("pareto-worst__SFX__").innerHTML = `<h4>Worst-cell Δ vs dense at matched ${xkey === "pf" ? "FLOPs" : "wall-clock"} · a universal rule needs ≥ 0</h4><svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="worst cell delta vs ${xlab}">${g}</svg>`;
  // per cell
  document.getElementById("pareto-cells__SFX__").innerHTML = PCELLS.map(c => {
    const have = rs.filter(r => r.cells[c]); if (!have.length) return "";
    const vs = have.map(r => r.cells[c].v), lo = Math.max(0, Math.floor(Math.min(...vs) * 10) / 10), hi = Math.min(1, Math.ceil(Math.max(...vs) * 10) / 10);
    const B = axes(240, 170, 34, 8, 8, 30, xlo, xhi, lo, hi === lo ? lo + 0.1 : hi, xkey === "pf" ? "PF" : "seconds", "", [lo, (lo + hi) / 2, hi], false);
    let s = B.g; const dc = dense.filter(r => r.cells[c]);
    if (dc.length > 1) s += `<polyline fill="none" stroke="var(--dense)" stroke-width="2" points="${dc.map(r => B.xs(r.x) + "," + B.ys(r.cells[c].v)).join(" ")}"/>`;
    have.forEach(r => { s += pmark(r, B.xs(r.x), B.ys(r.cells[c].v), 3); });
    return `<div class="panel"><h4>${c.replace("@", " · ")}</h4><svg viewBox="0 0 240 170" width="100%" role="img" aria-label="${c} score vs ${xlab}">${s}</svg></div>`;
  }).join("");
}
(function table() {
  const rs = PR.filter(r => r.xkey === "pf" && r.worst_delta !== undefined).sort((a, b) => b.worst_delta - a.worst_delta);
  const dmax = Math.max(...PR.filter(r => r.xkey === "pf" && r.family === "dense" && r.rows === 488).map(r => r.x));
  const cls = v => v >= -0.005 ? "pos" : v >= -0.03 ? "mid" : "neg";
  let h = `<tr><th>rule</th><th class="n">rows/task</th><th class="n">PF ÷ dense-122</th>${PCELLS.map(c => `<th class="n">${c.replace("contradiction", "contra").replace("outlier", "outl")}</th>`).join("")}<th class="n">mean Δ</th><th class="n">worst Δ</th></tr>`;
  rs.forEach(r => { h += `<tr><td class="mono">${r.arm}${r.epochs > 1 ? " ×" + r.epochs + "ep" : ""}</td><td class="n">${r.rows}</td><td class="n">${(r.x / dmax).toFixed(2)}</td>` +
    PCELLS.map(c => { const v = r.cells[c]; return `<td class="n ${v && v.delta !== undefined ? cls(v.delta) : ""}">${v && v.delta !== undefined ? (v.delta > 0 ? "+" : "") + v.delta.toFixed(3) + (v.extrap ? "*" : "") : "…"}</td>`; }).join("") +
    `<td class="n ${cls(r.mean_delta)}">${(r.mean_delta > 0 ? "+" : "") + r.mean_delta.toFixed(3)}</td><td class="n ${cls(r.worst_delta)} best">${(r.worst_delta > 0 ? "+" : "") + r.worst_delta.toFixed(3)}</td></tr>`; });
  document.getElementById("pareto-table__SFX__").innerHTML = h;
})();
const pbF = document.getElementById("px-pf__SFX__"), pbW = document.getElementById("px-wall__SFX__");
pbF.onclick = () => { pbF.setAttribute("aria-pressed", "true"); pbW.setAttribute("aria-pressed", "false"); drawPareto("pf"); };
pbW.onclick = () => { pbW.setAttribute("aria-pressed", "true"); pbF.setAttribute("aria-pressed", "false"); drawPareto("train_seconds"); };
drawPareto("pf");
})();
</script>
<!--PARETO__SFX__-END-->"""


INTRO4 = ("Each point is one training run, scored on contradiction, nq, outlier and oolong at the 8k and 2k rungs (500 eval "
          "examples each, no-cot). The line is dense (LR 5e-6); deltas use, per cell, the better of dense LR 2e-6 and 5e-6 at "
          "the same training FLOPs. A universal rule must sit <b>above</b> dense in every cell. Larger markers trained on more "
          "fresh rows (976 or 1,464 per task), ringed markers on repeated epochs. Rules hit by the 10-21 flag-drop bug are left out.")
INTRO11 = ("The 11-task fast8k mix (setA tasks, query before AND after the documents), scored on 10 tasks at 8k and 2k with the "
           "CTC-suite eval rows (500 each, no-cot, ctc-spec grading; rerank has no suite rows). Dense is LR 5e-6 only here. "
           "<b>Reorder@8k is left out:</b> the ctc grader scores 0 unless the output is an exact permutation of all 57 ids, and at 8k "
           "every run, dense included, repeats or invents a few (a lenient Kendall tau over common items would be about 0.20). "
           "Reorder@2k is kept.")


def section(d, mix, sfx, title, intro, updated, cells=None):
    if cells is None:
        tasks = sorted({p["task"] for p in d["points"] if d["runs"].get(p["run"], {}).get("mix") == mix})
        # reorder@8k is excluded: the ctc reorder grader scores 0 unless the output is an exact permutation of all
        # ids, and at 8k every run (dense included) repeats or invents a few, so the cell is a constant 0
        cells = [(t, r) for r in ("8k", "2k") for t in tasks if (t, r) != ("reorder", "8k")]
    rows = build(d, mix, cells)
    frag = (FRAGMENT.replace("__DATA__", json.dumps(rows, separators=(",", ":")))
            .replace("__CELLS__", json.dumps([f"{t}@{r}" for t, r in cells])).replace("__UPDATED__", updated)
            .replace("__SFX__", sfx).replace("__TITLE__", title).replace("__INTRO__", intro))
    return frag, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--updated", default="")
    ap.add_argument("--after", default='</section>\n\n<section id="screen">',
                    help="first-time insertion point (the section goes right before #screen)")
    a = ap.parse_args()
    d = json.load(open(a.points))
    frag, rows = section(d, "mt4", "", "Cross-task Pareto, 4-task mix (every run, matched FLOPs or wall-clock)", INTRO4,
                         a.updated, CELLS)
    page = open(a.page).read()
    if "<!--PARETO-BEGIN-->" in page:
        page = re.sub(r"<!--PARETO-BEGIN-->.*?<!--PARETO-END-->", lambda m: frag, page, flags=re.S)
    else:
        assert page.count(a.after) == 1, "insertion point not found"
        head, tail = a.after.split("\n\n", 1)
        page = page.replace(a.after, f"{head}\n\n{frag}\n\n{tail}")
    if any(r.get("mix") == "mt11" and r["arm"] == "dense" for r in d["runs"].values()):
        frag11, rows11 = section(d, "mt11", "11", "Cross-task Pareto, 11-task mix (10 tasks scored)", INTRO11, a.updated)
        if "<!--PARETO11-BEGIN-->" in page:
            page = re.sub(r"<!--PARETO11-BEGIN-->.*?<!--PARETO11-END-->", lambda m: frag11, page, flags=re.S)
        else:
            page = page.replace("<!--PARETO-END-->", "<!--PARETO-END-->\n\n" + frag11, 1)
        for r in sorted([r for r in rows11 if r["xkey"] == "pf" and "worst_delta" in r], key=lambda r: -r["worst_delta"]):
            print(f"[mt11] {r['arm']:22} rows={r['rows']:4} PF={r['x']:6.1f} mean_d={r['mean_delta']:+.3f} worst_d={r['worst_delta']:+.3f}")
    open(a.page, "w").write(page)
    full = [r for r in rows if r["xkey"] == "pf" and "worst_delta" in r]
    for r in sorted(full, key=lambda r: -r["worst_delta"])[:8]:
        print(f"{r['arm']:24} rows={r['rows']:5} ep={r['epochs']} PF={r['x']:7.1f} mean_d={r['mean_delta']:+.3f} worst_d={r['worst_delta']:+.3f}")


if __name__ == "__main__":
    main()
