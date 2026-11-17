"""Splice the length-generalization section into the mt11_64k artifact page.

    python3 research/token_routing/mt11_64k/lengen_page.py --base <saved artifact html> --out <page.html>

Reads results_lengen/lengen.json (lengen_pool.py), results_lengen/short16.json (16k-window comparison, optional),
the mt11-64k collector's points.json (legacy 2k..64k scores) and progress.json (free-text status lines). The section
sits between <!-- lengen:start --> / <!-- lengen:end --> right after the verdict; re-running replaces it.
"""
import argparse
import html
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_lengen")
POINTS = ("/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/mt11-64k/research/token_routing/"
          "mt11_64k/results/points.json")
TASKS = ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction"]
RUNGS = ["r64k", "r128k", "r256k"]
FULL = {"r64k": 500, "r128k": 500, "r256k": 125}
DENSE = ["m64-dense-f16-lr5e-06", "m64-dense-f8-lr5e-06", "m64-dense-f4-lr5e-06", "m64-dense-f2-lr5e-06",
         "m64-dense-f1-lr5e-06"]
METHODS = {"m64-lora64r20to7f16-f4-lr5e-06": "recipe f4", "m64-otr10M7-f2-lr5e-06": "otr10M7 f2",
           "m64-otr10M7-f1-lr5e-06": "otr10M7 f1"}
SHORT = {"m16-dense-s380-lr5e-06": "16k-data dense (f4 PF)", "m16-dense-s770-lr5e-06": "16k-data dense (f2 PF)",
         "m16-lora64r20to7f16-s380-lr5e-06": "16k-data recipe",
         "m32-dense-s304-lr5e-06": "32k-data dense (f4 PF)", "m32-dense-s610-lr5e-06": "32k-data dense (f2 PF)",
         "m32-lora64r20to7f16-s304-lr5e-06": "32k-data recipe"}
CHART_PNG = os.path.join(HERE, "lengen_rows16_32_64.png")
STYLE = """<style>
:root { --short: #0e7490; --short32: #7c3aed; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --short: #5cc8d8; --short32: #b79cf5; } }
:root[data-theme="dark"] { --short: #5cc8d8; --short32: #b79cf5; }
#lengen .pend { color: var(--muted); font-style: italic; }
#lengen .lg-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
svg .short-pt { fill: var(--short); stroke: var(--panel); stroke-width: 1; }
svg .short32-pt { fill: var(--short32); stroke: var(--panel); stroke-width: 1; }
</style>"""


def esc(x):
    return html.escape(str(x))


def interp(curve, x):
    pts = sorted(curve)
    if len(pts) < 2 or not x:
        return None, True
    lx = math.log(x)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (lx - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + w * (y1 - y0), False
    (x0, y0), (x1, y1) = (pts[0], pts[1]) if x < pts[0][0] else (pts[-2], pts[-1])
    w = (lx - math.log(x0)) / (math.log(x1) - math.log(x0))
    return y0 + w * (y1 - y0), True


def bracketed(by, pf, t, rung, run):
    """Score of dense interpolated at run's PF from its TRUE neighbours in DENSE (both must have the cell), else None."""
    x = pf.get(run)
    pts = sorted((pf[d], d) for d in DENSE if d in pf)
    lo = [p for p in pts if p[0] <= x] if x else []
    hi = [p for p in pts if p[0] >= x] if x else []
    if not lo or not hi:
        return None
    a, b = lo[-1][1], hi[0][1]
    if (a, t, rung) not in by or (b, t, rung) not in by:
        return None
    if a == b:
        return by[(a, t, rung)]["value"]
    ref, _ = interp([(pf[a], by[(a, t, rung)]["value"]), (pf[b], by[(b, t, rung)]["value"])], x)
    return ref


def dcell(d, extra=""):
    if d is None:
        return '<td class="na">pending</td>'
    cls = "pos" if d > 0.005 else "neg" if d < -0.005 else "tie"
    return f'<td class="{cls}">{d:+.3f}{extra}</td>'


def chart(rung, means, pf, w=260, h=170):
    """Task-mean score vs log PF: dense line, methods (router colour), 16k-data runs (teal)."""
    dense = sorted((pf[r], means[r]) for r in DENSE if r in means and r in pf)
    others = [(r, pf[r], means[r]) for r in list(METHODS) + list(SHORT) if r in means and r in pf]
    allpts = dense + [(p, v) for _, p, v in others]
    if len(dense) < 2:
        return f'<div class="panel empty"><div class="ptitle">{rung}</div><div class="na pend">in progress</div></div>'
    xs = [math.log(p) for p, _ in allpts]
    ys = [v for _, v in allpts]
    x0, x1 = min(xs) - 0.15, max(xs) + 0.15
    y0, y1 = min(ys) - 0.03, max(ys) + 0.03
    L, R, T, B = 30, 8, 10, 22
    X = lambda lx: L + (lx - x0) / (x1 - x0) * (w - L - R)  # noqa: E731
    Y = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (h - T - B)  # noqa: E731
    g = []
    for k in range(4):
        v = y0 + k * (y1 - y0) / 3
        g.append(f'<line class="grid" x1="{L}" x2="{w - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="tick" x="2" y="{Y(v) + 3:.1f}">{v:.2f}</text>')
    for p in (1e3, 3e3, 1e4):
        if x0 <= math.log(p) <= x1:
            g.append(f'<text class="tick" x="{X(math.log(p)) - 8:.1f}" y="{h - 6}">{p / 1e3:g}k</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(math.log(p)):.1f},{Y(v):.1f}" for i, (p, v) in enumerate(dense))
    g.append(f'<path class="dense-line" d="{path}"/>')
    g += [f'<circle class="dense-pt" cx="{X(math.log(p)):.1f}" cy="{Y(v):.1f}" r="2.6"><title>dense {p:.0f} PF: '
          f'{v:.3f}</title></circle>' for p, v in dense]
    for r, p, v in others:
        cls = ("short32-pt" if r.startswith("m32-") else "short-pt") if r in SHORT else "router-pt"
        lab = SHORT.get(r) or METHODS.get(r)
        g.append(f'<circle class="{cls}" cx="{X(math.log(p)):.1f}" cy="{Y(v):.1f}" r="4"><title>{esc(lab)} {p:.0f} PF:'
                 f' {v:.3f}</title></circle>')
    return (f'<div class="panel"><div class="ptitle">{rung} · 6-task mean vs training PF</div>'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{rung} task-mean score vs training PF">'
            + "".join(g) + "</svg></div>")


def short16_table():
    """Rows <= 16k vs rows <= 64k at matched FLOPs: mean delta per rung (native 2k..64k ladder, then olmo-eval long rungs)."""
    p = os.path.join(RES, "short16.json")
    if not os.path.exists(p):
        return ""
    d = json.load(open(p))
    cells = d["native"] + d["long"]
    rungs = ["2k", "8k", "16k", "32k", "64k", "r64k", "r128k", "r256k"]
    lab = {"2k": "2k", "8k": "8k", "16k": "16k", "32k": "32k", "64k": "64k", "r64k": "64k*", "r128k": "128k*",
           "r256k": "256k* ⚠125"}
    rows = []
    for run, name in SHORT.items():
        if not any(c["run"] == run for c in cells):
            rows.append(f"<tr><th>{esc(name)}</th><td class='na' colspan='{len(rungs) * 2}'>pending</td></tr>")
            continue
        for key, what in (("d_dense", "vs 64k-data dense at same PF"), ("d_recipe", "vs 64k-data recipe"),
                          ("d_m16", "vs 16k-data twin")):
            if key == "d_recipe" and "lora64" not in run:
                continue
            if key == "d_m16" and not run.startswith("m32-"):
                continue
            tds = []
            for rg in rungs:
                ds = [c[key] for c in cells if c["run"] == run and c["rung"] == rg and c.get(key) is not None]
                tds.append(dcell(sum(ds) / len(ds), f" <span class='muted'>({len(ds)})</span>") if ds
                           else '<td class="na">pending</td>')
            pfv = next((c["pf"] for c in cells if c["run"] == run), float("nan"))
            rows.append(f"<tr><th>{esc(name)} · {pfv:,.0f} PF<br><span class='muted'>{what}</span></th>{''.join(tds)}</tr>")
    head = "".join(f"<th>{lab[r]}</th>" for r in rungs)
    return ("<p>Mean Δ over the tasks scored at each rung (count in brackets). 2k–64k: native ladder, 10 tasks "
            "(no-signal cells reorder 8k/16k and grouping 32k excluded); * = olmo-eval path, 6 tasks. "
            "16k-data runs: rows ≤16,384 tokens, mostly ≤ the 8k rung; 32k-data runs: rows ≤32,768 tokens (5% of rows, "
            "23% of tokens above 16k, mostly 16–24k); the 64k-data runs saw rows to 65,536.</p>"
            f"<div class='tw'><table><thead><tr><th>run</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def section():
    lg = json.load(open(os.path.join(RES, "lengen.json"))) if os.path.exists(os.path.join(RES, "lengen.json")) else {}
    rows, pf = lg.get("rows", []), lg.get("pf", {})
    prog = json.load(open(os.path.join(RES, "progress.json"))) if os.path.exists(os.path.join(RES, "progress.json")) else {}
    by = {(r["run"], r["task"], r["rung"]): r for r in rows}
    means = {}
    for rung in RUNGS:
        means[rung] = {}
        for run in set(r["run"] for r in rows):
            vals = [by[(run, t, rung)]["value"] for t in TASKS if (run, t, rung) in by]
            if len(vals) == len(TASKS):
                means[rung][run] = sum(vals) / len(vals)
    # matched-FLOP deltas per method x rung
    trs = []
    for run, lab in METHODS.items():
        cells = []
        for rung in RUNGS:
            ds = []
            for t in TASKS:
                ref = bracketed(by, pf, t, rung, run)
                if (run, t, rung) in by and ref is not None and by[(run, t, rung)]["eval_size"] == FULL[rung]:
                    ds.append(by[(run, t, rung)]["value"] - ref)
            if len(ds) == len(TASKS):
                ahead = sum(d > 0 for d in ds)
                cells.append(dcell(sum(ds) / len(ds), f" <span class='muted'>({ahead}/6)</span>"))
            else:
                cells.append(f'<td class="na">{"partial " + str(len(ds)) + "/6" if ds else "pending"}</td>')
        trs.append(f"<tr><th>{esc(lab)}</th><td>{pf.get(run, float('nan')):,.0f}</td>{''.join(cells)}</tr>")
    # per-task detail at the longest rungs
    det = []
    for run, lab in METHODS.items():
        for rung in RUNGS[1:]:
            tds = []
            for t in TASKS:
                ref = bracketed(by, pf, t, rung, run)
                if (run, t, rung) in by and ref is not None and by[(run, t, rung)]["eval_size"] == FULL[rung]:
                    tds.append(dcell(by[(run, t, rung)]["value"] - ref))
                else:
                    tds.append('<td class="na">·</td>')
            det.append(f"<tr><th>{esc(lab)} · {rung}</th>{''.join(tds)}</tr>")
    # score table (all checkpoints)
    srows = []
    for run in DENSE + list(METHODS) + list(SHORT):
        if not any(k[0] == run for k in by):
            continue
        tds = []
        for rung in RUNGS:
            for t in TASKS:
                r = by.get((run, t, rung))
                if r is None:
                    tds.append('<td class="na">·</td>')
                else:
                    flag = "" if r["eval_size"] >= 500 else f'<sup title="eval_size {r["eval_size"]}, ±{r["se"]:.3f}">⚠{r["eval_size"]}</sup>'
                    tds.append(f"<td>{r['value']:.3f}{flag}</td>")
        lab = METHODS.get(run) or SHORT.get(run) or run.replace("m64-", "").replace("-lr5e-06", "")
        srows.append(f"<tr><th>{esc(lab)}</th><td>{pf.get(run, float('nan')):,.0f}</td>{''.join(tds)}</tr>")
    head_tasks = "".join(f"<th>{t[:7]}</th>" for t in TASKS)
    # anchor: legacy 64k vs new-path 64k (6-task mean)
    leg = {}
    if os.path.exists(POINTS):
        for s in json.load(open(POINTS))["scores"]:
            if s["rung"] == "64k" and s["task"] in TASKS:
                leg.setdefault(s["train_run"], {})[s["task"]] = s["value"]
    anc = []
    for run in DENSE + list(METHODS):
        if run in means["r64k"] and len(leg.get(run, {})) == len(TASKS):
            lm = sum(leg[run].values()) / len(TASKS)
            anc.append(f"<tr><th>{esc(METHODS.get(run, run.replace('m64-', '')))}</th><td>{lm:.3f}</td>"
                       f"<td>{means['r64k'][run]:.3f}</td>{dcell(means['r64k'][run] - lm)}</tr>")
    short_html = prog.get("short16_html", "") or short16_table()
    if os.path.exists(CHART_PNG):
        import base64
        b64 = base64.b64encode(open(CHART_PNG, "rb").read()).decode()
        short_html += ('<figure style="margin:12px 0"><img style="max-width:100%;background:#fff;border-radius:6px" '
                       f'alt="6-task mean vs eval length, rows ≤16k/≤32k/≤64k at matched FLOPs" src="data:image/png;base64,{b64}">'
                       '<figcaption class="muted">6-task mean vs eval length. 2k–32k native ladder, 64k–256k olmo-eval '
                       '(256k eval_size 125). The ≤64k line is the real 64k-data checkpoint nearest each budget.</figcaption></figure>')
    short_html += prog.get("short32_html", "")
    status = "".join(f"<li>{esc(x)}</li>" for x in prog.get("status", []))
    verdict = prog.get("verdict", "Results pending: the long-rung eval wave is running.")
    return f"""<!-- lengen:start -->
{STYLE}
<section id="lengen">
<h2>Length generalization: 128k and 256k <span class="pill">{esc(prog.get('pill', 'in progress'))}</span></h2>
<div class="verdict prose" style="border-left-color:var(--short)"><strong>Verdict so far.</strong> {verdict}</div>
<div class="prose" style="margin-top:10px">
<p>Prasann (2026-11-04): do the matched-FLOP gains persist when evaluating beyond the 64k training length, and how much does
a model trained only on rows ≤16k (more rows, same FLOPs) lose at long eval lengths? Follow-up (2026-11-12): the same
with rows ≤32k ("32k training with more data such that FLOPs match the 64k runs"). Six tasks with 128k/256k rows (nq,
hotpotqa, qdmatch_nq, outlier, oolong, contradiction), scored on olmo-eval's CTC launcher (chat format + doc markers,
full attention; r256k with YaRN ×2 and 131,072-token chunked prefill). The 2k–64k ladder above came from the native
docchunk evaluator, which cannot go past ~77k on one H100, so r64k is re-scored on the new path for every checkpoint
as the anchor. r64k/r128k: eval_size 500 per cell; <strong>r256k: eval_size 125 per cell (all the suite ships), ±~0.04</strong>.
Dense reference = dense LR 5e-6, f16→f1, interpolated linearly in log PF. No-cot. Single seed.</p>
<ul>{status}</ul>
</div>
<h3>Matched-FLOP Δ (method − dense at the method's PF), 6-task mean (tasks ahead)</h3>
<div class="tw"><table><thead><tr><th>method</th><th>PF</th><th>r64k</th><th>r128k</th><th>r256k ⚠125</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<div class="lg-grid" style="margin-top:12px">{''.join(chart(r, means[r], pf) for r in RUNGS)}</div>
<div class="legend" style="margin-top:6px"><span><span class="sw" style="background:var(--dense)"></span>dense (LR 5e-6, f16…f1)</span>
<span><span class="sw" style="background:var(--router)"></span>router / recipe</span>
<span><span class="sw" style="background:var(--short)"></span>trained on rows ≤16k</span>
<span><span class="sw" style="background:var(--short32)"></span>trained on rows ≤32k</span></div>
<h3>Per-task matched-FLOP Δ at 128k and 256k</h3>
<div class="tw"><table><thead><tr><th></th>{head_tasks}</tr></thead><tbody>{''.join(det)}</tbody></table></div>
<h3>Anchor: 64k on the two eval paths (6-task mean)</h3>
<div class="tw"><table><thead><tr><th>checkpoint</th><th>native ladder</th><th>olmo-eval</th><th>Δ</th></tr></thead>
<tbody>{''.join(anc) or '<tr><td class="na" colspan="4">pending</td></tr>'}</tbody></table></div>
<h3>Training on rows ≤16k and ≤32k at matched FLOPs</h3>
{short_html or '<p class="pend">Training (mt11L16: every setA s42 row ≤16,384 tokens, 192,554 rows / 716M tokens; seq 16,384): dense at the f4 and f2 FLOP budgets and the universal recipe at its budget. Evals follow.</p>'}
<h3>All long-rung scores</h3>
<div class="tw"><table><thead><tr><th>checkpoint</th><th>PF</th>{''.join(f'<th colspan="6">{r}</th>' for r in RUNGS)}</tr>
<tr><th></th><th></th>{head_tasks * 3}</tr></thead><tbody>{''.join(srows) or '<tr><td class="na">pending</td></tr>'}</tbody></table></div>
<p class="muted" style="margin-top:8px">Code: branches <code>claude/mt11-lengen-bed337</code> (≤16k, long rungs) and
<code>claude/mt11-lengen-32k-56ed58</code> (≤32k: <code>lengen_short16.py</code>, <code>lengen_chart.py</code>),
<code>research/token_routing/mt11_64k/lengen_{{evals,collect_beaker.sh,pool,page}}.py</code>; olmo-eval branch
<code>prasann/ctc-suite-mt11lengen</code>; record <code>records/mt11-64k-length-generalization.md</code>; results in ops
(runs <code>m11lg-*</code>).</p>
</section>
<!-- lengen:end -->"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = open(a.base).read()
    # drop the served wrapper (frame runtime, head) if present: the page source starts at <title>
    if "<!-- /frame-runtime -->" in s:
        s = s[s.index("<title>"):]
        s = re.sub(r"</body>\s*</html>\s*$", "", s)
    sec = section()
    if "<!-- lengen:start -->" in s:
        s = re.sub(r"<!-- lengen:start -->.*?<!-- lengen:end -->", lambda m: sec, s, flags=re.S)
    else:
        i = s.index('<div class="verdict prose">')
        j = s.index("</div>", i) + len("</div>")
        s = s[:j] + "\n" + sec + "\n" + s[j:]
    # one length-generalization pill in the header: drop any old one, add the current state
    s = re.sub(r' <span class="pill" style="border-color:var\(--short\);color:var\(--short\)">length generalization · [^<]*</span>', "", s)
    if True:
        s = s.replace('<span class="pill">complete · universal recipe and otr10M7</span>',
                      '<span class="pill">complete · universal recipe and otr10M7</span> '
                      f'<span class="pill" style="border-color:var(--short);color:var(--short)">length generalization · {esc(json.load(open(os.path.join(RES, "progress.json"))).get("pill", "in progress"))}</span>')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(s)
    print(f"wrote {a.out} ({len(s):,} bytes)")


if __name__ == "__main__":
    main()
