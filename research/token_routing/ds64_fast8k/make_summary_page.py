"""Build the short summary page (index.html) of the token-drop Pareto artifact, plus a compacted copy of the full
generated page (details.html).

The full page (setA_pareto_page.html, rebuilt by make_pareto_section / make_pertask_section / make_inflight_section)
stays the source of every table; this script only adds a one-screen front page on top of it:

  * one compact chart pair from results/pareto_points.json: 4-task mean / worst-cell delta vs dense at matched
    training FLOPs, for a handful of key recipes (seed copies averaged);
  * curated rows for results that live in other sessions' records (11-task floor, warmup-then-freeze, 64k, CPT),
    each naming its source.

usage: python3 make_summary_page.py --points results/pareto_points.json --page <setA_pareto_page.html> --out-dir <dir>
"""
import argparse
import html
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pareto_section import build  # noqa: E402

# (arm, label, colour group) -- the recipes the front-page chart tracks
KEY = [
    ("otr10M7", "router, keep 10% after block 7", "router"),
    ("otr10M8", "router, keep 10% after block 8", "router"),
    ("otr30M8", "router, keep 30% after block 8", "router"),
    ("otr10M7f16", "router + 16-token floor per doc", "router"),
    ("otr10M7de0", "router, no full-context steps", "aggr"),
    ("fa30M7", "router + freeze early blocks after 25%", "aggr"),
    ("frz0M7", "router + early blocks frozen from step 0", "aggr"),
    ("rnd10M8", "random 10% after block 8", "random"),
    ("tier67a50m64", "best fixed keep rule", "rule"),
]
COL = {"router": "#7b4fd6", "aggr": "#db2777", "random": "#8a8f8c", "rule": "#0f8b8d"}


def key_points(d):
    groups = {}
    for r in build(d, "mt4"):
        if r["xkey"] != "pf" or not r.get("complete") or "mean_delta" not in r:
            continue
        for arm, label, grp in KEY:
            if r["arm"] == arm:
                groups.setdefault((arm, r["rows"]), []).append(r)
    out = []
    for (arm, rows), rs in groups.items():
        label, grp = next((l, g) for a, l, g in KEY if a == arm)
        out.append(dict(arm=arm, label=label, grp=grp, rows=rows, n=len(rs),
                        pf=sum(r["x"] for r in rs) / len(rs),
                        md=sum(r["mean_delta"] for r in rs) / len(rs),
                        wd=sum(r["worst_delta"] for r in rs) / len(rs)))
    return sorted(out, key=lambda p: p["pf"])


def chart(pts, field, title, ylo, yhi):
    W, H, L, R, T, B = 420, 230, 40, 10, 10, 30
    xlo, xhi = math.log(60), math.log(800)
    xs = lambda v: L + (math.log(v) - xlo) / (xhi - xlo) * (W - L - R)
    ys = lambda v: T + (yhi - max(ylo, min(yhi, v))) / (yhi - ylo) * (H - T - B)
    g = [f'<rect x="{L}" y="{ys(0.03):.1f}" width="{W - L - R}" height="{ys(-0.03) - ys(0.03):.1f}" class="band"/>']
    y = ylo
    while y <= yhi + 1e-9:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{ys(y):.1f}" y2="{ys(y):.1f}" class="{"zero" if abs(y) < 1e-9 else "grid"}"/>'
                 f'<text x="{L - 4}" y="{ys(y) + 3:.1f}" text-anchor="end">{y:+.2f}</text>')
        y = round(y + 0.05, 4)
    for v in (100, 200, 400, 800):
        g.append(f'<text x="{xs(v):.1f}" y="{H - 14}" text-anchor="middle">{v}</text>')
    g.append(f'<text x="{(L + W - R) / 2}" y="{H - 2}" text-anchor="middle">training PF (log)</text>')
    for p in pts:
        v = p[field]
        tip = (f'{p["label"]}, {p["rows"]} rows/task, {p["pf"]:.0f} PF: mean Δ {p["md"]:+.3f}, worst Δ {p["wd"]:+.3f}'
               + (f' ({p["n"]} runs averaged)' if p["n"] > 1 else ""))
        clip = "▼" if v < ylo else ""
        g.append(f'<circle cx="{xs(p["pf"]):.1f}" cy="{ys(v):.1f}" r="4" fill="{COL[p["grp"]]}" opacity=".9">'
                 f'<title>{html.escape(tip)}</title></circle>'
                 + (f'<text x="{xs(p["pf"]):.1f}" y="{ys(v) - 6:.1f}" text-anchor="middle">{clip}</text>' if clip else ""))
    return (f'<figure class="fig"><figcaption>{title}</figcaption><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="{html.escape(title)}">{"".join(g)}</svg></figure>')


SCREEN_LABEL = {
    "otr10M7de0": "router, no full steps", "otr10M7de0sk3": "+ skip blocks 20/24/28", "r20to7M7de0": "two-stage: 20% at 7 → 7% at 16",
    "otr10M7de0pt": "per-task router", "fa50M7": "freeze blocks 0–6 at 50%", "fa75M7": "freeze blocks 0–6 at 75%",
    "pfz37M7": "progressive freeze (0–2 @25%, 0–6 @50%)", "lora16M7": "rank-16 adapters on blocks 0–6", "lora64M7": "rank-64 adapters on blocks 0–6",
    "mass80c20M7": "keep by predicted gradient mass", "otr10M7de0f16": "16-token floor per doc", "mod25": "mixture-of-depths 25%",
    "mod25t16": "mixture-of-depths 25%, block-16 target", "lora64fa75M7": "rank-64 adapters, frozen at 75%",
    "lora64fa75r20to7": "adapters (frozen 75%) + two-stage 20%→7%", "lora64fa75r20to10": "adapters (frozen 75%) + two-stage 20%→10%",
    "lora64fa75r3s": "adapters (frozen 75%) + three-stage 20/7/3%", "lora64r20to7": "RECIPE: adapters + two-stage 20%→7%",
    "lora64r20to10": "adapters + two-stage 20%→10%"}


def screen_table(d, tag):
    """matched-FLOP rows for one budget tag (e.g. 's1952-tde', 'f3904-tde'), sorted by worst-cell delta."""
    rows = [r for r in build(d, "mt4") if r["xkey"] == "pf" and tag in r["run"] and r.get("complete") and "mean_delta" in r]
    out = []
    for r in sorted(rows, key=lambda r: -r["worst_delta"]):
        w = min(((v.get("delta", 9), k) for k, v in r["cells"].items()))
        cls = lambda v: "pos" if v >= 0.02 else ("neg" if v <= -0.03 else "tie")
        out.append(f'<tr><td>{html.escape(SCREEN_LABEL.get(r["arm"], r["arm"]))}</td><td class="n">{r["x"]:.0f}</td>'
                   f'<td class="n {cls(r["mean_delta"])}">{r["mean_delta"]:+.3f}</td><td class="n {cls(r["worst_delta"])}">{r["worst_delta"]:+.3f}</td>'
                   f'<td>{w[1].replace("@", " @")}</td></tr>')
    return "\n".join(out) or '<tr><td colspan="5">no complete runs yet</td></tr>'


def fresh_table(d):
    """Fresh mt4f pools: each arm's 8-cell mean, and the dense FLOPs needed to reach it on the fresh dense curve
    (log-linear between dense-f1952 and dense-f3904; '>' when above the largest dense point)."""
    rows = [r for r in build(d, "mt4") if r["xkey"] == "pf" and re.search(r"-f\d+-tde-", r["run"]) and r.get("complete")]
    dense = sorted((r["x"], r["mean"]) for r in rows if r["arm"] == "dense")
    if len(dense) < 2:
        return '<tr><td colspan="6">dense fresh curve not complete yet</td></tr>'
    (x0, y0), (x1, y1) = dense[0], dense[-1]
    out = []
    for r in sorted((r for r in rows if r["arm"] != "dense"), key=lambda r: (int(re.search(r"-f(\d+)-", r["run"]).group(1)), r["x"])):
        rows_n = int(re.search(r"-f(\d+)-", r["run"]).group(1))
        t = (r["mean"] - y0) / (y1 - y0)
        need = x0 * (x1 / x0) ** t
        ftm = (">" if r["mean"] > y1 else "") + f"{(min(need, x1) if r['mean'] > y1 else need) / r['x']:.1f}×"
        out.append(f'<tr><td>{html.escape(SCREEN_LABEL.get(r["arm"], r["arm"]))}</td><td class="n">{rows_n}</td><td class="n">{r["x"]:.0f}</td>'
                   f'<td class="n">{r["mean"]:.3f}</td><td class="n">{r["cells"]["outlier@8k"]["v"]:.3f}</td><td class="n"><b>{ftm}</b></td></tr>')
    head = (f'<tr><td>dense</td><td class="n">1952</td><td class="n">{x0:.0f}</td><td class="n">{y0:.3f}</td><td class="n">–</td><td class="n">1.0×</td></tr>'
            f'<tr><td>dense</td><td class="n">3904</td><td class="n">{x1:.0f}</td><td class="n">{y1:.3f}</td><td class="n">–</td><td class="n">1.0×</td></tr>')
    return head + "\n".join(out)


def best_table(pts):
    pick = {}
    for p in pts:  # most-trained point per recipe
        if p["arm"] not in pick or p["rows"] > pick[p["arm"]]["rows"]:
            pick[p["arm"]] = p
    rows = []
    for arm, label, grp in KEY:
        if arm in pick:
            p = pick[arm]
            cls = lambda v: "pos" if v >= 0.02 else ("neg" if v <= -0.03 else "tie")
            rows.append(f'<tr><td><span class="dot" style="background:{COL[grp]}"></span>{html.escape(label)}</td>'
                        f'<td class="n">{p["rows"]}</td><td class="n">{p["pf"]:.0f}</td>'
                        f'<td class="n {cls(p["md"])}">{p["md"]:+.3f}</td><td class="n {cls(p["wd"])}">{p["wd"]:+.3f}</td>'
                        f'<td class="n">{p["n"]}</td></tr>')
    return "\n".join(rows)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token Drop Pareto</title>
<style>
:root { --bg:#fbfbf9; --surface:#fff; --fg:#1c1d1f; --muted:#62666d; --rule:#e2e2dc; --pos:#127a3e; --neg:#b42318; --band:#eef0ea;
  --body: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; --mono: ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141517; --surface:#1c1e21; --fg:#e7e8ea; --muted:#9a9ea6;
  --rule:#2e3136; --pos:#4cc38a; --neg:#f07a6e; --band:#23262a; } }
:root[data-theme="dark"] { --bg:#141517; --surface:#1c1e21; --fg:#e7e8ea; --muted:#9a9ea6; --rule:#2e3136; --pos:#4cc38a; --neg:#f07a6e; --band:#23262a; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 var(--body); }
.wrap { max-width: 860px; margin: 0 auto; padding: 24px 16px 56px; display: grid; gap: 22px; }
h1 { font-size: 1.5rem; margin: 0; } h2 { font-size: 1.1rem; margin: 0 0 8px; }
.sub { color: var(--muted); margin: 4px 0 0; font-size: .9rem; }
p { margin: 0 0 8px; max-width: 72ch; } ul { margin: 0; padding-left: 20px; } li { margin-bottom: 5px; max-width: 72ch; }
.box { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.fig { margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; max-width: 430px; }
.fig figcaption { font-size: .82rem; font-weight: 600; margin: 0 0 4px; }
svg { width: 100%; height: auto; display: block; }
svg text { fill: var(--muted); font: 10px var(--mono); }
svg .grid { stroke: var(--rule); } svg .zero { stroke: var(--muted); stroke-dasharray: 3 3; } svg .band { fill: var(--band); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: .82rem; color: var(--muted); }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; font-size: .85rem; background: var(--surface); width: 100%; }
th, td { border-bottom: 1px solid var(--rule); padding: 5px 8px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: .78rem; }
td.n, th.n { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.pos { color: var(--pos); } .neg { color: var(--neg); } .tie { color: var(--fg); }
.note { color: var(--muted); font-size: .82rem; }
a { color: inherit; }
</style></head><body><div class="wrap">

<header>
  <h1>Train-time token drop vs dense: where it stands</h1>
  <p class="sub">Goal: one train-time token/layer-drop method that beats dense at matched FLOPs (and memory) on every task, then on long-document CPT. Qwen3.5-4B SFT on CTC tasks unless noted. Updated __UPDATED__ · in progress · <a href="details.html">full tables and every run →</a></p>
</header>

<section class="box">
  <h2>Bottom line (2026-10-25)</h2>
  <ul>
    <li><b>Fresh-data FLOPs-to-match (4-task, new 8,200-row/task pools, one seed each; table below):</b> dense needs about <b>3× the FLOPs</b> to match the cheap methods at 3,904 rows/task (the recipe: 0.912 at 777 PF ≈ dense at 2,370 PF), and <b>≈2–2.2×</b> at 7,808 rows/task, where dense is near its plateau (three-stage + frozen adapters: 0.927 at 1,372 PF, above dense's 0.926 at 3,063 PF). Outlier@8k goes 0.96–0.99 for every cheap method at ≥3,904 rows/task.</li>
    <li><b>One recipe for both mixes (2 seeds each):</b> rank-64 adapters on blocks 0–6 (never frozen), keep 20% of tokens from block 7, rescore to 7% at block 16, no full-context steps; add a 16-token floor per document on the 11-task mix.
      <ul>
        <li>4-task mix, 1,464 rows/task: 0.945 / 0.912 / 0.636 / 0.726 at 0.26× dense FLOPs, ~49 GiB peak.</li>
        <li>11-task mix, 720 rows/task: mean9 0.751 at 0.38×, +0.045 over dense at matched FLOPs (xabsence holds at 0.68–0.71).</li>
        <li>Freezing the adapters is free on the 4-task mix but costs xabsence on the 11-task mix, so the recipe has no freeze.</li>
      </ul></li>
    <li><b>Best combination (2 seeds): rank-64 adapters on blocks 0–6, frozen at 75%, plus two-stage keep (20% of tokens from block 7, rescored to 7% at block 16), no full-context steps.</b> 4-task mix, 1,464 rows/task: 0.938 / 0.922 / 0.610 / 0.765 at 0.24× dense FLOPs and 48.9 GiB peak memory, against 0.937 / 0.902 / 0.623 / 0.682 at 0.30× and 60.3 GiB for the plain no-full-step router. Best quality: two-stage keep alone (2 seeds), 0.956 / 0.909 / 0.644 / 0.791 at 0.31×. Dense on the same data scores 0.956 / 0.936 / 0.630 / 0.719 at 1.0×. Second seed, 11-task and fresh-data (to 7,808 rows/task) runs are training.</li>
    <li><b>New best, compute and memory (2 seeds): router with low-rank adapters on blocks 0–6, no full-context steps.</b> Blocks 0–6 are frozen from step 0 and train through rank-64 adapters; the router keeps 10% of tokens for blocks 7–31. 4-task mix, 1,464 rows/task, LR 5e-6: 0.939 / 0.904 / 0.623 / 0.686, against 0.937 / 0.902 / 0.623 / 0.682 for the same router without adapters. That is the same quality at 0.25× dense FLOPs (vs 0.30×) and 48.4 GiB peak memory (vs 60.3), the first cut that lowers the peak from step 0. Rank 16 is worse (0.924 / 0.882 / 0.605 / 0.579). Fresh-data runs out to 7,808 rows/task and CPT runs are queued.</li>
    <li><b>At its own learning rate (5e-6), the router matches dense trained on the same data at 0.38× the FLOPs.</b> 4-task mix, 1,464 rows/task, 2 seeds: contradiction / nq / oolong / outlier @8k 0.955 / 0.913 / 0.622 / 0.726 vs dense 0.956 / 0.936 / 0.630 / 0.719. Level on 3 of 4; nq −0.023.</li>
    <li><b>Cheaper variants form one smooth curve at 5e-6</b> (table below). There's no free lunch: each step down in FLOPs costs a little, mostly outlier. Full-context step every 16 instead of 8 is the best trade (0.34×, contradiction −0.009). Freezing the early blocks halfway reaches 0.24× and 47.7 GiB. Its apparent free win at LR 2e-6 was mostly the LR.</li>
    <li><b>At 64k with 10× data the router beats dense, replicated:</b> 2 seeds, ahead on 8 of 9 tasks, +0.028 averaged, at 0.40× FLOPs (¼ epoch, LR 5e-6 for both). ½- and 1-epoch runs are training (≈10–14 h).</li>
    <li><b>11-task mix, at matched FLOPs (LR 5e-6 for both):</b> freeze-halfway + 16-token floor, with no full-context steps, scores mean9 0.750 at ≈258 PF. Dense reaches 0.750 only at 483 PF, so that is ≈1.9× fewer FLOPs (+0.045 over dense interpolated to 258 PF). On the same data dense is ahead (0.788 at 715 PF). <b>Correction:</b> the earlier "0.730 vs dense 0.741" used dense at LR 2e-6. <b>Long-document CPT:</b> every token-drop variant loses. Whole-model rank-64 LoRA (no dropping) is the CPT recipe: ≈2–4× FLOPs-to-match from 24 to 240 steps, ≈1.8× at 480, ≈3.4× at 64k, 29% lower peak; every stacked per-step cut lost.</li>
  </ul>
</section>

<section>
  <h2>4-task mix: gap to dense at matched FLOPs</h2>
  <div class="legend"><span><span class="dot" style="background:#7b4fd6"></span>router</span><span><span class="dot" style="background:#db2777"></span>aggressive (no full steps / freeze)</span><span><span class="dot" style="background:#8a8f8c"></span>random tokens</span><span><span class="dot" style="background:#0f8b8d"></span>fixed rule</span><span>grey band = ±0.03 noise; above 0 = beats dense</span></div>
  <div class="figs">__CHARTS__</div>
  <p class="note">Each dot is one recipe at one data size (points left to right = more rows/task); hover for details. Dense reference per cell = the better of dense LR 2e-6 and 5e-6, interpolated in log PF. 8 cells = contradiction, nq, outlier, oolong × 8k, 2k; eval_size 500 each, no-cot. ▼ = below the axis.</p>
  <div class="tbl"><table>
    <tr><th>recipe (most data run)</th><th class="n">rows/task</th><th class="n">PF</th><th class="n">mean Δ</th><th class="n">worst Δ</th><th class="n">runs</th></tr>
    __BEST__
  </table></div>
</section>

<section>
  <h2>Fresh data: FLOPs dense needs to match each method</h2>
  <div class="tbl"><table>
    <tr><th>method (LR 5e-6, no full-context steps, one seed)</th><th class="n">rows/task</th><th class="n">PF</th><th class="n">8-cell mean</th><th class="n">outlier@8k</th><th class="n">dense FLOPs to match</th></tr>
    __FRESH__
  </table></div>
  <p class="note">New fresh 4-task pools (8,200 rows/task, new generator seed; single epoch, no repeated rows). "Dense FLOPs to match" = the training PF dense needs to reach the same 8-cell mean on its fresh curve (1,952 and 3,904 rows/task), divided by the method's PF; "&gt;" = above dense's best point. eval_size 500 per cell, no-cot.</p>
</section>

<section>
  <h2>Overnight screen: 1,952 fresh rows/task, no full-context steps, matched FLOPs</h2>
  <div class="tbl"><table>
    <tr><th>variant (LR 5e-6, one seed)</th><th class="n">PF</th><th class="n">mean Δ</th><th class="n">worst Δ</th><th>worst cell</th></tr>
    __SCREEN1952__
  </table></div>
  <p class="note">Δ vs dense at the same training PF (per cell, the better of dense LR 2e-6 / 5e-6, interpolated in log PF; dense on these rows costs 1,541 PF). Every run's weakest cell is outlier@2k (dense ≈0.80 there), the noisiest cell (seed spread up to 0.19). Fresh-data runs out to 7,808 rows/task are training. <b>Caveat:</b> "within noise of dense" is against the dense curve. Against the same router, layer skips are a small loss (2 seeds at 1,464 rows: −0.03 contradiction, −0.09 outlier for 4% less compute), because only the kept 10% of tokens run blocks 20+.</p>
</section>

<section>
  <h2>Cheaper variants at LR 5e-6 (FLOPs and memory)</h2>
  <div class="tbl"><table>
    <tr><th>variant (4-task, 1,464 rows/task)</th><th class="n">FLOPs × dense</th><th class="n">contra</th><th class="n">nq</th><th class="n">oolong</th><th class="n">outlier</th><th class="n">seeds</th></tr>
    <tr><td>dense, same data</td><td class="n">1.00</td><td class="n">0.956</td><td class="n">0.936</td><td class="n">0.630</td><td class="n">0.719</td><td class="n">1</td></tr>
    <tr><td>router, full-context step every 8</td><td class="n">0.38</td><td class="n">0.955</td><td class="n">0.913</td><td class="n">0.622</td><td class="n">0.726</td><td class="n">2</td></tr>
    <tr><td><b>router, full step every 16</b></td><td class="n">0.34</td><td class="n">0.946</td><td class="n">0.912</td><td class="n">0.625</td><td class="n">0.725</td><td class="n">2</td></tr>
    <tr><td>router, no full-context steps</td><td class="n">0.30</td><td class="n">0.937</td><td class="n">0.902</td><td class="n">0.623</td><td class="n">0.682</td><td class="n">2</td></tr>
    <tr><td>+ freeze blocks 0–6 at 75%</td><td class="n">0.27</td><td class="n">0.930</td><td class="n">0.892</td><td class="n">0.628</td><td class="n">0.646</td><td class="n">2</td></tr>
    <tr><td>+ freeze at 50% (47.7 GiB after)</td><td class="n">0.24</td><td class="n">0.924</td><td class="n">–</td><td class="n">–</td><td class="n">–</td><td class="n">2</td></tr>
    <tr><td>router, keep 5% (budget control)</td><td class="n">0.26</td><td class="n">0.906</td><td class="n">0.890</td><td class="n">0.589</td><td class="n neg">0.460</td><td class="n">1</td></tr>
      <tr><td><b>router, no full steps + rank-64 adapters on blocks 0–6</b></td><td class="n">0.25</td><td class="n">0.939</td><td class="n">0.904</td><td class="n">0.623</td><td class="n">0.686</td><td class="n">2</td></tr>
    <tr><td>… rank-16 adapters</td><td class="n">0.24</td><td class="n">0.924</td><td class="n">0.882</td><td class="n">0.605</td><td class="n">0.579</td><td class="n">2</td></tr>
    <tr><td>progressive freeze (0–3 @25%, 0–6 @50%, 0–11 @75%)</td><td class="n">0.22</td><td class="n">0.926</td><td class="n">0.898</td><td class="n">0.634</td><td class="n">0.555</td><td class="n">2</td></tr>
    <tr><td>per-task router (no full steps)</td><td class="n">0.30</td><td class="n">0.941</td><td class="n">0.890</td><td class="n">0.616</td><td class="n">0.773</td><td class="n">1</td></tr>
    <tr><td>keep by predicted gradient mass (80%, cap 20%)</td><td class="n">0.37</td><td class="n">0.941</td><td class="n">0.918</td><td class="n">0.585</td><td class="n">0.571</td><td class="n">1</td></tr>
    <tr><td>mixture-of-depths, 25% on alternate blocks 8–30</td><td class="n">0.78</td><td class="n">0.957</td><td class="n">0.928</td><td class="n">0.622</td><td class="n neg">0.468</td><td class="n">1</td></tr>
</table></div>
  <p class="note">Peak memory per GPU (H100): 60.3 GiB for every arm without adapters, including the freeze arms before they freeze; 48.1–48.4 GiB with adapters. 8k rung, eval_size 500 each, no-cot. From the router session's record (records/online-token-router.md); seed spread on outlier is up to 0.19, so outlier differences under ~0.1 are noise. At matched FLOPs, freezing beats simply keeping fewer tokens (keep 5% loses outlier). Earlier screen at 488 rows: freezing from step 0, pyramids and merging all lose; the early-exit loss is a no-op on SFT rows (only answer tokens are labelled), so it is CPT-only.</p>
</section>

<section>
  <h2>Other tracks</h2>
  <div class="tbl"><table>
    <tr><th>track</th><th>status</th></tr>
    <tr><td>11-task mix, 720 rows/task (LR 5e-6)</td><td>Dense mean9: 0.697 @ 232 PF, 0.750 @ 483, 0.788 @ 715. Freeze-halfway + floor 16 (no full steps): 0.750 @ ≈258 PF, matching dense at ≈1.9× fewer FLOPs. The floor fixes strmatch (2 seeds). Per-task routers are training.</td></tr>
    <tr><td><a href="https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n">64k, 10× data</a></td><td>Router at LR 5e-6 (¼ epoch, 1,872 PF, 0.40× dense for the same rows) beats dense (better LR per cell, matched FLOPs) on 9 of 10 tasks, 2k–64k; one seed. At LR 2e-6 it trailed, so LR matters. Seed replicates and ½ / 1 epoch budgets training.</td></tr>
    <tr><td><a href="https://claude.ai/artifact/4tZKEXtpSpg5ax6B1YeoLk">Long-document CPT (OLMo 3 7B)</a></td><td><b>Recipe: whole-model LoRA</b> (rank 64 on all 32 blocks, embeddings frozen, LR 1e-5 to 2e-5, no token dropping). 0.70× dense FLOPs per step; 25.0 vs 35.2 GiB peak. Dense FLOPs needed to match it, at 32k by training steps: 12 → 1.65×, 24 → 2.75×, 48 → 3.9×, 120 → 3.45×, 240 → 2.5×, 480 → ≈1.8× (dev CE −0.0012 vs dense at 480). At 64k the edge is larger: −0.0049 at 120 steps, ≈3.4×. Every cut stacked on it lost: token router keep 0.5, partial MLP (random and importance-picked, 50–75%), stopping backprop below block 4/8, 8 rows/step (a tie). Lost earlier: all token dropping, mixture-of-depths, freezing without adapters, all-local attention, 4k-only rows, packed mixes. Next: 64k at 240/480 steps, CTC generation eval vs dense at matched FLOPs.</td></tr>
    <tr><td><a href="https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ">Gradient composition</a></td><td>Done. Blocks 0–6 hold 0.42–0.51 of gradient energy at the base, 0.13–0.25 after training (why freezing early helps late, hurts early). Token importance is smooth in position; rescoring at block 16 adds ≈ +0.08 of gradient mass.</td></tr>
    <tr><td><a href="https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1">Router session page</a></td><td>Design, fidelity probes, all ablations.</td></tr>
  </table></div>
</section>

<section>
  <h2>Training now (all without full-context steps, per Prasann)</h2>
  <ul>
    <li><b>4-task, 1,952 fresh rows/task, LR 5e-6 (Beaker):</b>
      <ul>
        <li>no-full-step router; freeze at 50% / 75%; progressive freeze</li>
        <li>two-stage rescoring (20% → 7% at block 16); router + layer skips; per-document floor 16; per-task router</li>
        <li>keep by predicted gradient mass (80%, cap 20%); low-rank adapters on blocks 0–6 from step 0; mixture-of-depths 25% (± block-16 target)</li>
      </ul></li>
    <li><b>Router session (Lambda, 1,464 rows):</b> the same ideas, a second data point.</li>
    <li><b>11-task, 720 rows:</b> per-task vs shared router, both with the 16-token floor.</li>
    <li><b>Fresh 4-task pools, ~8.2k rows/task:</b> building. Next, dense and the cheap arms out to 7,808 rows/task, to measure FLOPs-to-match-dense.</li>
    <li><b>CPT:</b> gradient analysis; freeze grid without dropping tokens; mixture-of-depths 50% / 25%; progressive freeze; low-rank early stack.</li>
    <li><b>64k:</b> ½- and 1-epoch router and dense runs.</li>
  </ul>
  <p class="note">Code: branch prasann/flopdrop_clean (research/token_routing/ds64_fast8k/), router claude/online-router-1bc01a. Status: ops/experiments/tokdrop_lr, learned_router, mt11_64k, olmo3_cptdrop. This page is private.</p>
</section>
</div></body></html>
"""

DETAILS_CSS = """<style id="compact">
.charts { grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)) !important; gap: 10px !important; }
.charts3 svg { min-width: 0 !important; } .charts3 .panel { max-width: 640px; }
.panel svg { max-height: 230px; }
</style>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--updated", default="")
    a = ap.parse_args()
    d = json.load(open(a.points))
    pts = key_points(d)
    charts = (chart(pts, "md", "Mean Δ vs dense (8 cells)", -0.10, 0.10)
              + chart(pts, "wd", "Worst-cell Δ vs dense", -0.15, 0.05))
    page = (PAGE.replace("__FRESH__", fresh_table(d)).replace("__SCREEN1952__", screen_table(d, "s1952-tde")).replace("__CHARTS__", charts).replace("__BEST__", best_table(pts))
            .replace("__UPDATED__", a.updated or "today"))
    open(os.path.join(a.out_dir, "index.html"), "w").write(page)
    det = open(a.page).read()
    det = re.sub(r'<style id="compact">.*?</style>', "", det, flags=re.S)
    det = re.sub(r"^\s*<!doctype html>\s*<meta[^>]*>\s*<meta[^>]*>", "", det, flags=re.I)
    det = det.replace("</head>", DETAILS_CSS + "</head>", 1) if "</head>" in det else DETAILS_CSS + det
    if not det.lstrip().lower().startswith("<!doctype"):  # served as written, so it needs its own skeleton
        det = ('<!doctype html>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n'
               + det)
    if 'href="index.html"' not in det:
        det = re.sub(r"(<header[^>]*>)", r'\1<p class="sub"><a href="index.html">← summary</a></p>', det, count=1)
    open(os.path.join(a.out_dir, "details.html"), "w").write(det)
    for p in pts:
        print(f'{p["arm"]:14s} rows={p["rows"]:5d} n={p["n"]} PF={p["pf"]:6.0f} md={p["md"]:+.3f} wd={p["wd"]:+.3f}')


if __name__ == "__main__":
    main()
