"""Render the mt11_64k summary page (claude.ai artifact) from collect_m64.py's results/points.json + matched.json.

Static HTML with inline SVG: verdict, setup, per-rung Pareto small multiples (score vs log training PF, one panel
per task; dense = per-budget best of the two LRs, both LRs drawn faint; router points marked), the matched-FLOP
delta table (task x rung + the mean over the rungs each task has), cost/wall-clock, limits, provenance.

    python3 research/token_routing/mt11_64k/make_page.py --verdict "..." --out <scratch>/mt11_64k_page.html
"""
import argparse
import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction", "xabsence", "reorder", "strmatch",
         "grouping"]
RUNGS = ["2k", "8k", "16k", "32k", "64k"]
METRIC = {"reorder": "kendall τ", "grouping": "pairwise F1"}
#: cells whose numbers carry no signal (see STATUS: reorder parse failures at 8k/16k for every arm)
UNINFORMATIVE = {("reorder", "8k"), ("reorder", "16k"), ("grouping", "32k")}


def esc(x):
    return html.escape(str(x))


def se(v, n=500):
    return math.sqrt(max(v * (1 - v), 1e-9) / n)


def load():
    pts = json.load(open(os.path.join(HERE, "results", "points.json")))
    m = json.load(open(os.path.join(HERE, "results", "matched.json")))
    train = {t["run"]: t for t in pts["train"]}
    scores = {}
    for s in pts["scores"]:
        if s["final"] and s["train_run"] in train:
            scores[(s["train_run"], s["task"], s["rung"])] = s
    return train, scores, m


def panel(task, rung, train, scores, w=210, h=150):
    """One Pareto panel: x = log PF, y = score."""
    pad_l, pad_r, pad_t, pad_b = 34, 8, 18, 26
    rows = [(train[r], s) for (r, t, g), s in scores.items() if t == task and g == rung and train[r].get("pf")
            and "smk" not in r]
    if not rows:
        return f'<div class="panel empty"><div class="ptitle">{esc(task)}</div><div class="na">no rung</div></div>'
    xs = [tr["pf"] for tr, _ in rows]
    lo, hi = math.log10(min(xs)) - 0.08, math.log10(max(xs)) + 0.08
    ys = [s["value"] for _, s in rows]
    ylo, yhi = max(0.0, min(ys) - 0.08), min(1.0, max(ys) + 0.08)
    if yhi - ylo < 0.2:
        mid = (yhi + ylo) / 2
        ylo, yhi = max(0, mid - 0.1), min(1, mid + 0.1)
    X = lambda pf: pad_l + (math.log10(pf) - lo) / (hi - lo) * (w - pad_l - pad_r)
    Y = lambda v: pad_t + (1 - (v - ylo) / (yhi - ylo)) * (h - pad_t - pad_b)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(task)} at {rung}: score vs training PF">']
    for v in (ylo, (ylo + yhi) / 2, yhi):
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{pad_l - 4}" y="{Y(v) + 3:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    # x ticks at the dense budgets' PF
    for pf in sorted({tr["pf"] for tr, _ in rows if tr["arm"] == "dense"}):
        out.append(f'<text x="{X(pf):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{pf / 1000:.1f}k</text>')
    best = {}
    for tr, s in rows:
        if tr["arm"] == "dense":
            b = tr["budget"]
            if b not in best or s["value"] > best[b][1]:
                best[b] = (tr["pf"], s["value"])
            out.append(f'<circle cx="{X(tr["pf"]):.1f}" cy="{Y(s["value"]):.1f}" r="2.4" class="dense-faint">'
                       f'<title>dense {tr["budget"]} LR {tr["lr"]:g}: {s["value"]:.3f}</title></circle>')
    curve = sorted(best.values())
    if len(curve) > 1:
        out.append('<polyline class="dense-line" points="' +
                   " ".join(f"{X(p):.1f},{Y(v):.1f}" for p, v in curve) + '"/>')
    for p, v in curve:
        out.append(f'<circle cx="{X(p):.1f}" cy="{Y(v):.1f}" r="3.2" class="dense-pt"/>')
    for tr, s in rows:
        if tr["arm"] != "dense":
            x, y = X(tr["pf"]), Y(s["value"])
            out.append(f'<path d="M{x:.1f},{y - 5:.1f} L{x + 5:.1f},{y:.1f} L{x:.1f},{y + 5:.1f} L{x - 5:.1f},{y:.1f}Z" '
                       f'class="router-pt"><title>{esc(tr["arm"])} {tr["budget"]} LR {tr["lr"]:g}: '
                       f'{s["value"]:.3f} at {tr["pf"]:.0f} PF</title></path>')
    out.append("</svg>")
    flag = ' <span class="flag">no signal</span>' if (task, rung) in UNINFORMATIVE else ""
    return (f'<div class="panel"><div class="ptitle">{esc(task)}{flag}</div>{"".join(out)}</div>')


def delta_cell(d, extrap):
    if d is None:
        return '<td class="na">–</td>'
    cls = "pos" if d > 0.02 else "neg" if d < -0.02 else "tie"
    return f'<td class="{cls}">{d:+.3f}{"<sup>e</sup>" if extrap else ""}</td>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdict", default="")
    ap.add_argument("--status", default="in progress")
    a = ap.parse_args()
    train, scores, m = load()
    per = m["per_rung"]
    routers = sorted({p["run"] for p in per})
    five = {(f["run"], f["task"]): f for f in m["five_rung"]}

    # score table (all runs x task x rung)
    runs = sorted(train, key=lambda r: (train[r]["arm"] != "dense", train[r].get("pf") or 0, train[r]["lr"]))
    runs = [r for r in runs if "smk" not in r]
    srows = []
    for r in runs:
        tr = train[r]
        cells = []
        for t in TASKS:
            vals = [scores.get((r, t, g)) for g in RUNGS]
            cells.append("<td>" + " / ".join("·" if v is None else f'{v["value"]:.2f}' for v in vals) + "</td>")
        srows.append(f'<tr><th>{esc(r.replace("m64-", ""))}</th><td class="num">{(tr.get("pf") or 0):,.0f}</td>'
                     + "".join(cells) + "</tr>")

    # matched-FLOP delta tables, one per router run
    dtables = []
    for r in routers:
        rows = []
        for t in TASKS:
            cells = []
            for g in RUNGS:
                p = next((p for p in per if p["run"] == r and p["task"] == t and p["rung"] == g), None)
                if (t, g) in UNINFORMATIVE and p:
                    cells.append('<td class="na">n/s</td>')
                else:
                    cells.append(delta_cell(p and p["delta"], p and p["extrapolated"]))
            f = five.get((r, t))
            cells.append(delta_cell(f and f["delta"], f and f["extrapolated"]).replace("<td", '<td style="font-weight:600"'))
            rows.append(f"<tr><th>{esc(t)}</th>{''.join(cells)}</tr>")
        pf = train[r].get("pf") or 0
        dtables.append(f'<h3>{esc(r.replace("m64-", ""))} <span class="sub">{pf:,.0f} PF</span></h3>'
                       f'<div class="tw"><table class="delta"><thead><tr><th>task</th>'
                       + "".join(f"<th>{g}</th>" for g in RUNGS) + "<th>mean</th></tr></thead><tbody>"
                       + "".join(rows) + "</tbody></table></div>")

    # same-LR vs best-LR summary: mean over each task's rungs of (router - dense interpolated at the router's PF)
    import sys as _sys
    _sys.path.insert(0, HERE)
    from collect_m64 import interp_log
    lr_rows = []
    for r in routers:
        pf = train[r]["pf"]
        for label, pick in (("dense at the same LR", lambda tr, r=r: tr["lr"] == train[r]["lr"]),
                            ("dense, better LR per cell", None)):
            cells = []
            for t in TASKS:
                ds = []
                for g in RUNGS:
                    if (r, t, g) not in scores or (t, g) in UNINFORMATIVE:
                        continue
                    if pick is None:
                        p = next((p for p in per if p["run"] == r and p["task"] == t and p["rung"] == g), None)
                        if p and p["delta"] is not None:
                            ds.append(p["delta"])
                        continue
                    curve = [(train[d]["pf"], scores[(d, t, g)]["value"]) for d in train
                             if train[d]["arm"] == "dense" and pick(train[d]) and (d, t, g) in scores]
                    ref, _ = interp_log(curve, pf)
                    if ref is not None:
                        ds.append(scores[(r, t, g)]["value"] - ref)
                cells.append(delta_cell(sum(ds) / len(ds) if ds else None, False))
            lr_rows.append(f'<tr><th>{esc(r.replace("m64-", ""))} vs {label}</th>{"".join(cells)}</tr>')
    lr_table = ('<div class="tw"><table class="delta"><thead><tr><th>mean Δ over rungs</th>'
                + "".join(f"<th>{esc(t)}</th>" for t in TASKS) + "</tr></thead><tbody>" + "".join(lr_rows)
                + "</tbody></table></div>") if lr_rows else ""

    # seed replicate table: each router seed vs the SEED-AVERAGED better-LR dense curve (f16/f8/f4)
    def dense_ref(t, g, pf):
        pts = {}
        for d, x in train.items():
            if x["arm"] != "dense" or x["budget"] not in ("f16", "f8", "f4") or (d, t, g) not in scores:
                continue
            pts.setdefault((x["budget"], x["lr"]), []).append((x["pf"], scores[(d, t, g)]["value"]))
        best = {}
        for (b, _lr), v in pts.items():
            val = sum(z for _, z in v) / len(v)
            if b not in best or val > best[b][1]:
                best[b] = (v[0][0], val)
        return interp_log(list(best.values()), pf)[0]
    seed_runs = [r for r in ("m64-otr10M7-f4-lr5e-06", "m64-otr10M7-f4-lr5e-06-sd1") if r in train]
    seed_rows = []
    if len(seed_runs) == 2:
        for r in seed_runs:
            cells = []
            for t in TASKS:
                ds = [scores[(r, t, g)]["value"] - dense_ref(t, g, train[r]["pf"]) for g in RUNGS
                      if (r, t, g) in scores and (t, g) not in UNINFORMATIVE and dense_ref(t, g, train[r]["pf"]) is not None]
                cells.append(delta_cell(sum(ds) / len(ds) if ds else None, False))
            seed_rows.append(f'<tr><th>{esc(r.replace("m64-", ""))}</th>{"".join(cells)}</tr>')
    seed_table = ('<div class="tw"><table class="delta"><thead><tr><th>router seed vs seed-averaged dense</th>'
                  + "".join(f"<th>{esc(t)}</th>" for t in TASKS) + "</tr></thead><tbody>" + "".join(seed_rows)
                  + "</tbody></table></div>") if seed_rows else '<p class="muted">Seed replicate not in yet.</p>'

    rung_sections = []
    for g in RUNGS:
        rung_sections.append(f'<section class="rung"><h3>{g} rung</h3><div class="grid">'
                             + "".join(panel(t, g, train, scores) for t in TASKS) + "</div></section>")

    cost = "".join(
        f'<tr><th>{esc(r.replace("m64-", ""))}</th><td class="num">{train[r].get("steps")}</td>'
        f'<td class="num">{(train[r].get("pf") or 0):,.0f}</td>'
        f'<td class="num">{(train[r].get("pf") or 0) / max(train[r].get("steps") or 1, 1):.1f}</td>'
        f'<td class="num">{(train[r].get("wall_s") or 0) / 60:,.0f}</td></tr>' for r in runs)

    page = TEMPLATE.format(
        verdict=a.verdict or "Results are still coming in; see the tables for what has landed.",
        status=esc(a.status), score_rows="".join(srows), task_heads="".join(f"<th>{esc(t)}</th>" for t in TASKS),
        lr_table=lr_table, seed_table=seed_table,
        delta_tables="".join(dtables) or '<p class="muted">No router run has finished its evals yet.</p>',
        rungs="".join(rung_sections), cost_rows=cost)
    open(a.out, "w").write(page)
    print(f"wrote {a.out} ({len(page) / 1024:.0f} KB)")


TEMPLATE = """<title>mt11 64k Router Study</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>
/* layout: one reading column for prose, full-width grids for the per-rung Pareto small multiples */
:root {{
  --bg: #f6f7f9; --panel: #ffffff; --fg: #1d2330; --muted: #5c6577; --rule: #dde1e8;
  --dense: #4a5a73; --dense-faint: #a9b3c3; --router: #c2410c; --pos: #0f7a4a; --neg: #b42318; --tie: #5c6577;
  --display: "Source Serif 4", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12151b; --panel: #1a1f27; --fg: #e4e8ef; --muted: #9aa3b2; --rule: #2c333e;
  --dense: #9fb0c9; --dense-faint: #55606f; --router: #f08a4b; --pos: #4cc38a; --neg: #f07a6e; --tie: #9aa3b2;
  color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #12151b; --panel: #1a1f27; --fg: #e4e8ef; --muted: #9aa3b2; --rule: #2c333e;
  --dense: #9fb0c9; --dense-faint: #55606f; --router: #f08a4b; --pos: #4cc38a; --neg: #f07a6e; --tie: #9aa3b2;
  color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); margin: 0;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 28px 56px; }}
main {{ max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 28px; }}
.prose {{ max-width: 70ch; }}
h1 {{ font: 600 2rem/1.15 var(--display); margin: 0; text-wrap: balance; }}
h2 {{ font: 600 1.35rem/1.2 var(--display); margin: 0 0 10px; text-wrap: balance; }}
h3 {{ font: 600 0.95rem/1.3 var(--body); margin: 18px 0 8px; }}
.sub, .muted {{ color: var(--muted); font-weight: 400; }}
.eyebrow {{ font: 500 0.72rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
.verdict {{ background: var(--panel); border: 1px solid var(--rule); border-left: 4px solid var(--router);
  padding: 14px 18px; border-radius: 4px; max-width: 80ch; }}
.pill {{ display: inline-block; font: 500 0.72rem/1 var(--mono); padding: 4px 8px; border-radius: 99px;
  border: 1px solid var(--router); color: var(--router); }}
p {{ margin: 0 0 10px; }}
ul {{ margin: 0; padding-left: 20px; }} li {{ margin-bottom: 4px; }}
code, .num, td {{ font-family: var(--mono); font-size: 0.82rem; font-variant-numeric: tabular-nums; }}
.tw {{ overflow-x: auto; }}
table {{ border-collapse: collapse; background: var(--panel); }}
th, td {{ border-bottom: 1px solid var(--rule); padding: 5px 9px; text-align: right; white-space: nowrap; }}
th {{ font: 500 0.78rem var(--body); color: var(--muted); }}
tbody th {{ text-align: left; color: var(--fg); font-family: var(--mono); font-size: 0.8rem; }}
td.pos {{ color: var(--pos); }} td.neg {{ color: var(--neg); }} td.tie, td.na {{ color: var(--tie); }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.82rem; color: var(--muted); align-items: center; }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; }}
.panel {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 6px 6px 0; min-width: 0; }}
.panel.empty {{ display: flex; flex-direction: column; justify-content: space-between; min-height: 60px; padding-bottom: 8px; }}
.ptitle {{ font: 500 0.78rem var(--mono); }}
.flag {{ font-size: 0.68rem; color: var(--neg); }}
.panel .na {{ color: var(--muted); font-size: 0.75rem; }}
svg {{ width: 100%; height: auto; display: block; }}
svg .grid {{ stroke: var(--rule); stroke-width: 1; }}
svg .tick {{ fill: var(--muted); font: 9px var(--mono); }}
svg .dense-line {{ fill: none; stroke: var(--dense); stroke-width: 1.6; }}
svg .dense-pt {{ fill: var(--dense); }}
svg .dense-faint {{ fill: var(--dense-faint); }}
svg .router-pt {{ fill: var(--router); stroke: var(--panel); stroke-width: 1; }}
.rung h3 {{ font-family: var(--display); font-size: 1.05rem; }}
</style>
<main>
<header class="prose" style="display:flex;flex-direction:column;gap:8px">
  <span class="eyebrow">mt11_64k · Qwen3.5-4B · 11-task CTC mix to 65,536 tokens</span>
  <h1>Does the mid-stack online token router hold up at 64k with 10× the data?</h1>
  <span><span class="pill">{status}</span></span>
</header>
<div class="verdict prose"><strong>Verdict.</strong> {verdict}</div>

<section class="prose">
<h2>Setup</h2>
<ul>
<li><b>Data</b> <code>mt11L64</code>: setA seed-42 gold shards (same generator as the CTC suite, query position both), rows ≤ 65,536 tokens, ds64's short-heavy token shares 30/20/15/13/12/10 over ≤3k/6k/12k/24k/48k/64k of a 64M-token target per task. 156,587 rows, 662M tokens, 12.3k–15.8k rows per task. Verified: every row hits its gold sidecar.</li>
<li><b>Training</b>: 8×H100, unpacked rows, 128 rows/step, micro-batch 1, flash_2, repaired Qwen3.5-4B base. Dense runs the pad-free path (every document kept). Budgets are fractions of one epoch (1,223 steps) with a complete LR schedule each.</li>
<li><b>Router</b> <code>otr10M7</code>: every token runs blocks 0–6 on the full row; a ridge router refit every step on per-token gradient norms keeps 10% of each document's tokens for blocks 7–31; a charged full-context step every 8. LR 2e-6.</li>
<li><b>Dense reference</b>: LR 2e-6 and 5e-6 at each budget; the curve uses the better of the two per task × rung. Matched-FLOP Δ = router score − dense interpolated linearly in log PF (<sup>e</sup> = extrapolated past the dense range).</li>
<li><b>Eval</b>: CTC-suite rows, 500 per rung (eval_size 500 everywhere; binomial SE ≈ ±0.02 at 0.7), query position both, ctc-spec grading, no-cot, native docchunk ladder. reorder stops at 16k; xabsence, strmatch, grouping at 32k. rerank is trained but has no ladder grader for its suite rows. FLOPs = <code>flop_meter/actual_pflops</code>.</li>
</ul>
</section>

<section>
<h2>Same LR vs better LR</h2>
<p class="prose muted">The learning rate decides the sign. Each cell is the router's score minus dense interpolated to the router's FLOPs, averaged over the task's rungs.</p>
{lr_table}
</section>

<section>
<h2>Seed replicate</h2>
<p class="prose muted">Router ¼ at LR 5e-6, seeds 0 and 1, each against the dense curve averaged over its own two seeds (dense ⅛ and ¼ were replicated; 1/16 has seed 0 only). Mean over each task's rungs.</p>
{seed_table}
</section>

<section>
<h2>Matched-FLOP Δ vs dense, per rung</h2>
<p class="prose muted">Green ≥ +0.02, red ≤ −0.02. Single seed: on the 8k screens run-to-run noise was ±0.02–0.03, and up to ±0.13 on outlier. "n/s" = no signal: reorder 8k/16k (every arm fails the exact-permutation parser) and grouping 32k (every arm scores 0.001-0.005).</p>
{delta_tables}
</section>

<section>
<h2>Pareto plots per rung</h2>
<div class="legend"><span><span class="sw" style="background:var(--dense)"></span>dense, better LR per budget</span>
<span><span class="sw" style="background:var(--dense-faint)"></span>dense, each LR</span>
<span><span class="sw" style="background:var(--router);border-radius:1px;transform:rotate(45deg)"></span>online router</span>
<span>x: training PF (log), y: primary metric (f1; reorder kendall τ; grouping pairwise F1)</span></div>
{rungs}
</section>

<section>
<h2>All scores</h2>
<p class="prose muted">Each cell lists 2k / 8k / 16k / 32k / 64k (· = rung not in the suite for that task). eval_size 500 per cell.</p>
<div class="tw"><table><thead><tr><th>run</th><th>PF</th>{task_heads}</tr></thead><tbody>{score_rows}</tbody></table></div>
</section>

<section>
<h2>Cost and wall-clock</h2>
<div class="tw"><table><thead><tr><th>run</th><th>steps</th><th>PF</th><th>PF / step</th><th>wall min (incl. load, queue restarts)</th></tr></thead><tbody>{cost_rows}</tbody></table></div>
<p class="prose muted" style="margin-top:8px">Dense is the pad-free path, so its wall-clock is the pad-free reference. At micro-batch 1 on 8 GPUs a routed step took ~22 s vs ~20 s for dense: the FLOP saving does not turn into wall-clock in this configuration.</p>
</section>

<section class="prose">
<h2>Limits</h2>
<ul>
<li>Single seed per configuration; router at one LR (2e-6).</li>
<li>38–80% of r64k prompts are 65.5k–70k tokens, slightly past the longest training row; outlier, oolong and rerank train on nothing above ~35k (their 64k-bucket rows exceed 65,536 tokens).</li>
<li>reorder 8k/16k carry no signal: well-formed lists with duplicate ids at 8k, repetition loops in 150/500 rows at 16k.</li>
<li>A routed step costs ~0.47× a dense step, not the ~0.3× the block count suggests (router overhead and the dense step every 8 are charged).</li>
</ul>
</section>

<section class="prose">
<h2>Where things live</h2>
<ul>
<li>Experiment state: <code>ops/experiments/mt11_64k/STATUS.md</code>; results: <code>ops result show --experiment mt11_64k</code>.</li>
<li>Code: branch <code>prasann/mt11-64k</code>, <code>research/token_routing/mt11_64k/</code> (launcher, build, verify, collector, this page).</li>
<li>Shard: weka <code>prasanns/mt11_64k/shards/mt11L64</code>; checkpoints <code>prasanns/ctc_suite/ckpts/m64-*</code>; suite rows <code>prasanns/_eval_bundle/ctc_suite_rows</code>.</li>
<li>wandb group <code>m64-q35-4b</code>.</li>
</ul>
</section>
</main>
"""

if __name__ == "__main__":
    main()
