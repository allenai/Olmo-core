"""Plots-first overview of the train-time token-drop work (2026-10-11 .. 10-25): FLOP-matched Pareto plots for the
4-task mix (fresh pools and the original pool), the 11-task mix, long-document CPT (32k and 64k) and max train length,
plus pointers to every data file, record and launcher behind them.

SFT points come from results/pareto_points.json (collect_pareto_points.py); CPT and memory numbers are hand-entered
below from records/olmo3-cptdrop-results.md and records/online-token-router.md (each block names its source).

usage: python3 make_overview_page.py --points results/pareto_points.json --out <dir>/overview.html
"""
import argparse
import collections
import json
import re

MT4 = [(t, r) for t in ("contradiction", "nq", "outlier", "oolong") for r in ("8k", "2k")]
MT11 = ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction", "xabsence", "strmatch", "grouping"]


def series(d):
    """{(mix, pool, arm, lr): [(pf, score, outlier8k, n_seeds, rows, steps)]}, seed copies averaged."""
    cells = collections.defaultdict(dict)
    for p in d["points"]:
        if p["eval_size"] >= 500:
            cells[p["run"]][(p["task"], p["rung"])] = p["score"]
    agg = collections.defaultdict(list)
    for k, r in d["runs"].items():
        c = cells.get(k, {})
        if not r.get("pf") or (r.get("epochs") or 1) > 1:  # repeated-epoch runs are not on the fresh-data curves
            continue
        if r.get("mix") == "mt4" and all(x in c for x in MT4):
            pool = "f" if re.search(r"-f\d+-tde", k) else "s"
            y = sum(c[x] for x in MT4) / 8
        elif r.get("mix") == "mt11" and all((t, "8k") in c for t in MT11):
            pool, y = "s", sum(c[(t, "8k")] for t in MT11) / 9
        else:
            continue
        arm = re.sub(r"sd\d+$", "", r["arm"])
        tasks = [t for t, _ in MT4[::2]] if r["mix"] == "mt4" else MT11
        agg[(r["mix"], pool, arm, f"{r['lr']:.0e}", r["rows"], r.get("max_steps") or 0)].append(
            (r["pf"], y, c[("outlier", "8k")], {t: c[(t, "8k")] for t in tasks}))
    out = collections.defaultdict(list)
    for (mix, pool, arm, lr, rows, steps), v in agg.items():
        n = len(v)
        tc = {t: round(sum(a[3][t] for a in v) / n, 4) for t in v[0][3]}
        out[(mix, pool, arm, lr)].append((round(sum(a[0] for a in v) / n, 1), round(sum(a[1] for a in v) / n, 4),
                                          round(sum(a[2] for a in v) / n, 4), n, rows, steps, tc))
    return {k: sorted(v) for k, v in out.items()}


def pick(S, mix, pool, arm, lr="5e-06", exclude_partial=False):
    pts = S.get((mix, pool, arm, lr), [])
    if exclude_partial:
        pts = [p for p in pts if not p[5]]
    return [{"x": p[0], "y": p[1], "o": p[2], "n": p[3], "rows": p[4], "steps": p[5], "t": p[6]} for p in pts]


# ---- CPT (OLMo 3 7B, dev CE, lower is better). Source: records/olmo3-cptdrop-results.md + CPT session reports ----
CPT32 = {
    "Dense": [(137, 1.7096), (247, 1.7008), (685, 1.6912), (1344, 1.6880), (3318, 1.6853), (6606, 1.6827),
              (13188, 1.6800), (26349, 1.6762)],
    "Whole-model LoRA (rank 64, all 32 blocks)": [(941, 1.6849), (2323, 1.6819), (4624, 1.6804), (9232, 1.6788), (18444, 1.6774)],
    "Whole-model LoRA, frozen output layer": [(921, 1.6842), (2273, 1.6812), (4525, 1.6795), (9034, 1.6779), (18049, 1.6764)],
    "LoRA rank 64, blocks 0–15": [(305, 1.6942), (587, 1.6895), (1151, 1.6864), (2842, 1.6834), (5660, 1.6820),
                                  (11296, 1.6794)],
    "Token router, keep 10% after block 7": [(335, 1.7308), (643, 1.7138), (1258, 1.7079), (3083, 1.7055)],
    "Mixture-of-depths 50%": [(307, 1.7305), (583, 1.7169), (1134, 1.7050), (2788, 1.6987)],
    "Freeze blocks 0–15 at 50% (no adapters)": [(304, 1.7085), (581, 1.7010), (1135, 1.6921), (2795, 1.6858)],
    "4k windows only": [(38, 1.7117), (73, 1.7110), (143, 1.7121), (353, 1.7161), (704, 1.7192), (1404, 1.7275),
                        (2806, 1.7406)],
}
# 64k: dense PF estimated as LoRA PF / 0.696 (the dense-64k flops file is a partial count)
CPT64 = {"Dense": [(3799, 1.7210), (7598, 1.7170), (15195, 1.7134)],
         "Whole-model LoRA (rank 64, all 32 blocks)": [(2644, 1.7161), (5288, 1.7137), (10576, 1.7114)]}
CPT_M_FH = [(48, 4.8), (120, 4.3), (240, 3.2), (480, 2.1), (960, 1.4)]
CPT_M = [(12, 1.65), (24, 2.75), (48, 3.9), (120, 3.45), (240, 2.5), (480, 1.8), (960, 1.12)]
# ---- peak memory per GPU vs row length, SFT 4B, 2 x A100-80GB, FSDP-2, full AC, steady routed step (router record,
# "B' round 0/2" memprobe tables). Standard flags: fixed + transient at 32k/64k/128k. All memory flags (routed warmup,
# --skip-dry-run, chunked FFN 8192, GDN segments 16384): measured peaks. ----
MEM_STD = {"Dense": [(32, 43.7), (64, 56.0)], "Router keep 10%": [(32, 42.1), (64, 50.3), (128, 66.7)],
           "Recipe: two-stage + adapters 0–6": [(32, 36.4), (64, 44.4), (128, 60.2)],
           "Whole-model LoRA + two-stage": [(32, 23.3), (64, 29.3), (128, 45.2)],
           "Two-stage, no adapters (Beaker)": [(32, 44.1), (64, 52.4), (128, 68.7)],
           "Router keep 20% (Beaker)": [(32, 43.6), (64, 51.8)]}
MEM_FLAGS = {"Dense": [(128, 68.3), (160, 77.5)], "Router keep 10%": [(192, 64.3), (224, 69.9)],
             "Recipe: two-stage + adapters 0–6": [(262, 71.6)],
             "Whole-model LoRA + two-stage": [(262, 56.5), (393, 77.4)],
             "Two-stage, no adapters (Beaker)": [(128, 59.2), (196, 68.8)],
             "Router keep 20% (Beaker)": [(128, 57.6), (196, 67.3)]}
# linear fits (intercept GiB, slope GiB per 1k tokens) from the router record's probe tables: standard = fixed + its
# transient fit; all flags = the record's per-1k slopes through the measured points (recipe slope from round 1)
MEM_STD_FIT = {"Dense": (31.3, 0.386), "Router keep 10%": (34.0, 0.255), "Recipe: two-stage + adapters 0–6": (28.5, 0.248),
               "Whole-model LoRA + two-stage": (13.4, 0.248), "Two-stage, no adapters (Beaker)": (36.1, 0.255),
               "Router keep 20% (Beaker)": (35.2, 0.26)}
MEM_FLAGS_FIT = {"Dense": (32.7, 0.28), "Router keep 10%": (31.8, 0.17), "Recipe: two-stage + adapters 0–6": (16.6, 0.21),
                 "Whole-model LoRA + two-stage": (14.6, 0.16), "Two-stage, no adapters (Beaker)": (39.4, 0.15),
                 "Router keep 20% (Beaker)": (37.9, 0.15)}
# early-block parameter-gradient share over training, three tracks (records/grad-composition.md tables; mt11 rows
# 32-42/task, CPT 48 rows x 32k)
EARLY_SHARE = {"SFT 4-task (LR 5e-6)": [(0, 0.45), (31, 0.31), (61, 0.28), (122, 0.25), (366, 0.23)],
               "SFT 11-task (LR 2e-6)": [(0, 0.45), (165, 0.39), (330, 0.39), (495, 0.34)],
               "CPT OLMo 3 7B (32k)": [(0, 0.42), (12, 0.20), (120, 0.17), (480, 0.16)]}
CPT_CTX = {"base": [(0, 0.99), (8, 0.94), (16, 0.83), (24, 0.79)], "after CPT (12-480 steps)": [(0, 0.94), (8, 0.78), (16, 0.42), (24, 0.21)]}
# ---- max train length at 80 GB, SFT 4B, 2 x A100, FSDP-2, full AC (records/online-token-router.md; linear
# extrapolation from 3-step probes, one sample each) ----
MAXT = [("Dense", 165), ("Router keep 10%", 250), ("Two-stage, no adapters", 255), ("Router keep 20%", 270),
        ("Recipe (adapters 0–6 + two-stage)", 300), ("Whole-model LoRA + two-stage", 400)]

PAGE = r"""<title>Token Drop Results</title>
<style>
/* layout: one reading column; full-width figure grid; tables scroll on their own */
:root {
  --bg: #f7f8f6; --surface: #ffffff; --fg: #1b1f1d; --muted: #5d6661; --rule: #dfe3df;
  --dense: #2f3a35; --c1: #c2410c; --c2: #2563eb; --c3: #0f8b6e; --c4: #9333ea; --c5: #b45309; --c6: #64748b; --c7: #be185d;
  --display: "Fraunces", Georgia, serif; --body: "Public Sans", system-ui, sans-serif; --mono: "JetBrains Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121614; --surface: #1a1f1c; --fg: #e5ebe7; --muted: #9aa59f; --rule: #2c3430;
  --dense: #d7e0da; --c1: #fb923c; --c2: #60a5fa; --c3: #34d399; --c4: #c084fc; --c5: #fbbf24; --c6: #94a3b8; --c7: #f472b6; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #121614; --surface: #1a1f1c; --fg: #e5ebe7; --muted: #9aa59f; --rule: #2c3430;
  --dense: #d7e0da; --c1: #fb923c; --c2: #60a5fa; --c3: #34d399; --c4: #c084fc; --c5: #fbbf24; --c6: #94a3b8; --c7: #f472b6; color-scheme: dark }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }
.wrap { max-width: 1120px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 64px; display: grid; gap: 30px; }
h1 { font: 600 2rem/1.15 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 600 1.3rem/1.2 var(--display); margin: 0 0 6px; text-wrap: balance; }
h3 { font: 600 .95rem/1.3 var(--body); margin: 0 0 4px; }
p, li { max-width: 74ch; margin: 0 0 8px; }
ul { margin: 0; padding-left: 20px; }
.sub { color: var(--muted); }
.eyebrow { font: 500 .72rem/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.lead { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 16px 18px; }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 12px 8px; min-width: 0; }
figcaption { font-size: .82rem; color: var(--muted); margin-top: 6px; }
.chart { position: relative; height: 300px; }
.figs.small { grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); }
.figs.small .chart { height: 210px; }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; background: var(--surface); }
th, td { border-bottom: 1px solid var(--rule); padding: 5px 8px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: .78rem; }
td.n, th.n { text-align: right; font-family: var(--mono); font-size: .8rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
code, .mono { font-family: var(--mono); font-size: .8rem; overflow-wrap: anywhere; }
.note { color: var(--muted); font-size: .84rem; }
.pos { color: var(--c3); } .neg { color: var(--c1); }
a { color: var(--c2); }
.layout { max-width: 1340px; margin: 0 auto; padding-inline: 16px; display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 32px; }
.layout > .wrap { max-width: none; margin: 0; padding-inline: 0; min-width: 0; gap: 34px; }
nav.toc { position: sticky; top: env(safe-area-inset-top, 0px); align-self: start; padding-block: 28px; display: grid; gap: 2px; }
nav.toc a { display: flex; gap: 10px; align-items: baseline; padding: 6px 10px; border-radius: 4px; color: var(--muted); text-decoration: none; font-size: .88rem; line-height: 1.3; }
nav.toc a span { font: 500 .72rem/1 var(--mono); color: var(--muted); min-width: 1ch; }
nav.toc a:hover { color: var(--fg); background: var(--surface); }
nav.toc a.on { color: var(--fg); background: var(--surface); box-shadow: inset 2px 0 0 var(--c2); }
.part { display: grid; gap: 26px; scroll-margin-top: 60px; }
.parthead { border-top: 2px solid var(--fg); padding-top: 14px; display: grid; gap: 6px; }
.ptitle { font: 600 1.7rem/1.15 var(--display); margin: 0; }
.verdict { font-size: 1rem; max-width: 80ch; margin: 0; }
.part section h2 { font-size: 1.15rem; }
.arch li { margin-bottom: 6px; }
.logbar { display: grid; gap: 8px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chips button { font: 500 .78rem/1 var(--body); padding: 6px 10px; border-radius: 999px; border: 1px solid var(--rule); background: var(--surface); color: var(--fg); cursor: pointer; }
.chips button[aria-pressed="true"] { background: var(--fg); color: var(--bg); border-color: var(--fg); }
.chips button:focus-visible, #logQ:focus-visible { outline: 2px solid var(--c2); outline-offset: 2px; }
#logQ { font: .9rem var(--body); padding: 7px 10px; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); color: var(--fg); max-width: 420px; width: 100%; }
ol.log { list-style: none; margin: 0; padding: 0; max-height: 75vh; overflow-y: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
ol.log li { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 4px 14px; padding: 12px 14px; border-bottom: 1px solid var(--rule); max-width: none; margin: 0; }
ol.log li:last-child { border-bottom: 0; }
.ld { font: .78rem var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.lt { font-weight: 600; }
.lm { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0; font-size: .74rem; }
.lm span { padding: 2px 7px; border-radius: 4px; border: 1px solid var(--rule); color: var(--muted); }
.lm .st { color: var(--fg); font-weight: 600; }
.st.running { border-color: var(--c2); color: var(--c2); } .st.done { border-color: var(--c3); color: var(--c3); }
.st.dropped, .st.failed { border-color: var(--c1); color: var(--c1); } .st.proposed, .st.held { border-color: var(--c5); color: var(--c5); }
.st.partial { border-color: var(--c4); color: var(--c4); }
.cnow { border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); padding: 12px 14px; display: grid; gap: 8px; margin-bottom: 10px; }
.cnow .tbl { max-height: 260px; overflow: auto; }
.lm .cp { border-color: var(--c2); color: var(--c2); }
.lr { color: var(--fg); font-size: .9rem; max-width: 90ch; margin: 0; }
@media (max-width: 560px) { ol.log li { grid-template-columns: 1fr; } }
@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; gap: 0; }
  nav.toc { z-index: 20; background: var(--bg); display: flex; overflow-x: auto; padding-block: 8px; border-bottom: 1px solid var(--rule); gap: 4px; }
  nav.toc a { white-space: nowrap; padding: 6px 8px; }
  .part { scroll-margin-top: 56px; }
}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=JetBrains+Mono:wght@400;500&family=Public+Sans:wght@400;600&display=swap">
<div class="layout">
<nav class="toc" aria-label="Sections">
  <a href="#summary"><span>1</span>Summary</a>
  <a href="#log"><span>☰</span>Experiment log</a>
  <a href="#sft4"><span>2</span>SFT 4-task</a>
  <a href="#sft11"><span>3</span>SFT 11-task + long context</a>
  <a href="#models"><span>3b</span>Qwen3 + model scale</a>
  <a href="#cpt"><span>4</span>CPT</a>
  <a href="#memory"><span>5</span>Memory + wall-clock</a>
  <a href="#mechanism"><span>6</span>Mechanism</a>
  <a href="#ruledout"><span>7</span>Ruled out</a>
  <a href="#next"><span>8</span>Next ideas</a>
  <a href="#provenance"><span>9</span>Provenance</a>
</nav>
<main class="wrap">
<header>
  <span class="eyebrow">Train-time token / layer drop · all results in one place · updated __UPDATED__</span>
  <h1>Cheaper training that matches dense at matched FLOPs</h1>
  <p class="sub">Qwen3.5-4B SFT on CTC long-context tasks (4-task and 11-task mixes, 64k) and OLMo 3 7B long-document continued pretraining. Every plot puts training FLOPs (measured by the FLOP meter) on the x-axis; up-and-left of the dense line is better (down-and-left for CPT loss). Private page.</p>
</header>
<div class="part" id="summary">
<div class="parthead"><span class="eyebrow">1 · Summary</span><h2 class="ptitle">Where things stand</h2><p class="verdict">Routing plus LoRA trains ≈2–4× cheaper than dense on SFT at matched quality; in CPT, LoRA alone wins and every token drop loses.</p></div>
<section class="lead">
  <h2>Verdicts</h2>
  <ul>
    <li><b>SFT, 4-task mix:</b> dense needs about <b>3×</b> the training FLOPs to match the cheap methods on fresh data (3,904 rows/task), and <b>≈2×</b> at 7,808 rows/task (now a point estimate: the recipe's 0.926 equals dense at 3,904 rows; dense gains only +0.005 from 3,904 to 7,808, so the 7,808 ratios are good only to ≈1.5–4×). On the original pool, dense needs ≈3.5–3.9× to match the recipe and ≈4.4× for whole-model LoRA + two-stage keep.</li>
    <li><b>SFT, 11-task mix:</b> ≈1.7× for the recipe; whole-model LoRA + two-stage scores 0.762 against dense's 0.697 at the same 232 PF (2 seeds), ≈2.4×. Gains concentrate on xabsence, contradiction and outlier; four retrieval tasks are near dense's ceiling; grouping is the one task the router loses.</li>
    <li><b>SFT, 11-task mix at 64k with 10× the data:</b> the router (keep 10% after block 7, LR 5e-6) is ≈+0.03 task-average over dense at matched FLOPs at every budget from 1/4 to 1 epoch; dense needs ≈1.6–1.7× the FLOPs to match. On the same data it trails dense by ≈0.03. <b>The recipe at 64k (1/4 epoch, one seed): +0.037 task-average at 0.365× dense FLOPs, ahead on all 10 graded tasks, 24% less peak active memory</b> (37.2 vs 48.7 GiB); its gains grow with length (contradiction +0.041 at 8k → +0.119 at 64k).</li>
    <li><b>The recipe:</b> rank-64 LoRA adapters on blocks 0–6 (frozen base), an online gradient-norm router that keeps 20% of tokens from block 7 and rescores to 7% at block 16, no full-context steps, LR 5e-6, plus a 16-token floor per document on the 11-task mix. Whole-model LoRA (all 32 blocks, LR 1e-5) is cheaper still at the same quality.</li>
    <li><b>LoRA vs routing (2×2 ablation):</b> routing does most of the matched-FLOP gain; LoRA-all is a smaller add-on. Dense FLOPs to match, 4-task (original 1,464 / fresh 1,952 / fresh 3,904 rows per task): dense + LoRA-all 1.48 / 1.41 / 1.60×, routing alone 3.45 / 2.85 / 2.89×, both 4.82 / 4.63 / 3.81×. 11-task (240 / 480 / 720 rows per task): 1.39 / 1.15 / 1.06×, 1.79 / 1.67 / 1.56×, 1.83 / 2.09 / 2.30×. LoRA alone is 15–35% of the log multiplier on the 4-task mix; on the 11-task mix its share shrinks with data (54% → 19% → 7%). At matched data all four arms score within ≈0.03, so the gain is cheaper steps, not better learning. Dense is at its best LR throughout. A random token choice nearly matches the router (+0.027 at 1,464 rows, a tie at 3,904), so most of the routing gain comes from the drop structure. ⚠ Mostly 1–3 seeds.</li>
    <li><b>CPT:</b> every token-drop variant loses. Whole-model LoRA with no dropping beats dense at 0.70× metered FLOPs per step (≈0.75× counting checkpoint recompute): ≈2–4× FLOPs-to-match between 24 and 240 steps, ≈1.8× at 480, and parity by 960 steps (dev CE +0.0012 behind dense; ≈1.04× counting recompute). The adapters saturate around 0.25–0.5B tokens while full CPT keeps learning long-range structure, so this is a short-budget recipe. At 64k context the lead is ≈1.4–1.5× the 32k one, and about the same at 128k (frozen-output-layer recipe: −0.0057 / −0.0039 / −0.0026 at 64k after 120 / 240 / 480 steps, −0.0038 / −0.0046 at 128k after 48 / 120; the 64k 240-step run ties dense at 480, ≈2.9×). <b>Freezing the output layer too is the CPT recipe</b> (no full-rank weights trained at all; 0.685× FLOPs per step, 22.3 vs 35.2 GiB peak): never behind dense up to 960 steps. Dense FLOPs to match: 4.8× / 4.3× / 3.2× / 2.1× / ≈1.4× at 48 / 120 / 240 / 480 / 960 steps (dev CE −0.0037 / −0.0041 / −0.0032 / −0.0022 / +0.0002); seed 1 reproduces at 120. It also forgets less on short documents.</li>
    <li><b>Max train length at 80 GB (SFT):</b> ≈1.8× dense for the recipe and ≈2.4× for whole-model LoRA (extrapolated from short probes).</li>
  </ul>
</section>
<section>
  <h2>At a glance</h2>
  <div class="tbl"><table>
    <tr><th>setting</th><th>best method</th><th class="n">FLOPs / step vs dense</th><th class="n">dense FLOPs to match</th><th>notes</th></tr>
    <tr><td>SFT 4-task, fresh 3,904 rows/task</td><td>recipe (LoRA + two-stage keep)</td><td class="n">0.22×</td><td class="n">≈3.0–3.7×</td><td>1–2 seeds</td></tr>
    <tr><td>SFT 4-task, fresh 7,808 rows/task</td><td>recipe</td><td class="n">0.25×</td><td class="n">≈2.0×</td><td>dense near ceiling; good only to ≈1.5–4×</td></tr>
    <tr><td>SFT 11-task, 720 rows/task</td><td>whole-model LoRA + two-stage + floor</td><td class="n">≈0.33×</td><td class="n">≈1.7–2.4×</td><td>2 seeds</td></tr>
    <tr><td>SFT 11-task at 64k, ¼ epoch</td><td>recipe</td><td class="n">0.365×</td><td class="n">+0.037 at matched FLOPs</td><td>10 / 10 tasks ahead, 1 seed</td></tr>
    <tr><td>CPT, OLMo 3 7B, 32k</td><td>whole-model LoRA, frozen output layer</td><td class="n">0.685×</td><td class="n">4.8× → 1.4×</td><td>48 → 960 steps; larger at 64k</td></tr>
    <tr><td>Max train length at 80 GB (SFT)</td><td>whole-model LoRA + router</td><td class="n">–</td><td class="n">≈2.4× dense's max T</td><td>extrapolated probes</td></tr>
  </table></div>
  <p class="note">"FLOPs / step vs dense" is per training step; "dense FLOPs to match" is the Pareto gain (compute dense needs to reach the same score). They differ because cheap methods score a little lower per row.</p>
</section>
<section>
  <h2>Open decisions</h2>
  <ul>
    <li><b>Which ≥5× idea to start</b> (Section 8): shared-context multi-query training or row selection. Prasann's bar (10-29): gains of at least 5–10×; tweaks around 1.2× are not pursued.</li>
    <li><b>CPT checkpoints:</b> run a matched SFT + CTC generation eval (held).</li>
    <li><b>Branches to merge</b> (not merged by choice): <code>claude/vibrant-williams-4b0f66</code> (LoRA × routing 2×2), <code>claude/train-layerskip-35010b</code> (layer skip), <code>claude/online-router-1bc01a</code> (router code, hill-climb, fidelity v2).</li>
  </ul>
</section>
</div>
<div class="part" id="log">
<div class="parthead"><span class="eyebrow">Experiment log</span><h2 class="ptitle">What is running, what was tried</h2><p class="verdict">Every idea across the main session and the parallel sessions, newest first. Filter by status or track, or search.</p></div>
<section>
  <div class="logbar">
    <div class="chips" id="logStatus" role="group" aria-label="Filter by status"></div>
    <div class="chips" id="logTrack" role="group" aria-label="Filter by track"></div>
    <input type="search" id="logQ" placeholder="Search the log" aria-label="Search the log">
  </div>
  <div id="computeNow"></div>
  <p class="note" id="logCount"></p>
  <ol class="log" id="logList"></ol>
</section>
</div>
<div class="part" id="sft4">
<div class="parthead"><span class="eyebrow">2 · SFT, 4-task mix</span><h2 class="ptitle">Pareto, FLOPs-to-match and the LoRA × routing split</h2><p class="verdict">Dense needs ≈3× the FLOPs at 3,904 fresh rows/task and ≈2× at 7,808; routing carries most of that gain.</p></div>
<section>
  <h2>SFT: 4-task mix</h2>
  <div class="figs">
    <figure><h3>Fresh data: mean score vs training FLOPs</h3><div class="chart"><canvas id="f4fresh" aria-label="4-task fresh data Pareto"></canvas></div>
      <figcaption>New 8,200-row/task pools, single epoch, no repeated rows. Points per method = 1,952 / 3,904 / 7,808 rows/task. Score = mean of contradiction, nq, outlier, oolong at 8k and 2k (eval_size 500 each, no-cot). One seed per point.</figcaption></figure>
    <figure><h3>Fresh data: outlier@8k vs training FLOPs</h3><div class="chart"><canvas id="f4out" aria-label="outlier at 8k"></canvas></div>
      <figcaption>Outlier is the least saturated task. Same runs as the left plot.</figcaption></figure>
    <figure><h3>Original pool: mean score vs training FLOPs</h3><div class="chart"><canvas id="f4orig" aria-label="4-task original pool Pareto"></canvas></div>
      <figcaption>ds64 8k pool, 488–1,952 rows/task. Dense includes runs cut short at 31/61 steps for the low end. Seed copies averaged (labels give the seed count).</figcaption></figure>
  </div>
  <div class="tbl" style="margin-top:14px"><table>
    <tr><th>method (fresh data, LR 5e-6, no full-context steps)</th><th class="n">rows/task</th><th class="n">PF</th><th class="n">8-cell mean</th><th class="n">outlier@8k</th><th class="n">dense FLOPs to match</th></tr>
    __FTM__
  </table></div>
  <p class="note">"Dense FLOPs to match" = training PF dense needs to reach the same 8-cell mean on its fresh curve (piecewise log-linear along dense at 1,952 / 3,904 / 5,856 / 7,808 rows/task) divided by the method's PF; "&gt;" when the method is above dense's best point (a lower bound).</p>
</section>
__LORA2X2__
<section>
  <h2>LoRA × routing: Pareto view</h2>
  <div class="figs">
    <figure><h3>4-task, fresh pools (1,952 / 3,904 / 7,808 rows/task)</h3><div class="chart"><canvas id="lr2f" aria-label="LoRA x routing, 4-task fresh"></canvas></div>
      <figcaption>8-cell mean (4 tasks × 2k / 8k), eval_size 500, no-cot. Each arm at its better LR (5e-6 or 1e-5). Points left to right = more rows/task. 1–2 seeds per point.</figcaption></figure>
    <figure><h3>4-task, original pool (488–1,952 rows/task)</h3><div class="chart"><canvas id="lr2o" aria-label="LoRA x routing, 4-task original pool"></canvas></div>
      <figcaption>Same metric. Dense's two cheapest points are runs cut at 31 / 61 steps. Dense + LoRA-all and routing-only exist only at 1,464 rows; both arms at 488–1,952.</figcaption></figure>
    <figure><h3>11-task (240 / 480 / 720 rows/task)</h3><div class="chart"><canvas id="lr2e" aria-label="LoRA x routing, 11-task"></canvas></div>
      <figcaption>Mean over 9 tasks at 8k, eval_size 500, no-cot. Dense's two cheapest points are 240-row runs cut at 32 / 64 steps. Routing arms use the 16-token floor.</figcaption></figure>
  </div>
  <p class="note">Read across at a fixed score (dense FLOPs to match, from the table above): on the 4-task mix routing alone (orange) is ≈2.9–3.5× left of dense and routing + LoRA-all (blue) ≈3.8–4.8×, while dense + LoRA-all (green) is only ≈1.4–1.6×. On the 11-task mix the arms are closer: 1.06–1.39× (LoRA-all), 1.56–1.79× (routing), 1.83–2.30× (both).</p>
</section>
<section>
  <h2>Per task: 4-task mix @8k</h2>
  <p class="note">Same runs and colours as above, one panel per task (8k rung, eval_size 500 each, no-cot). Fresh data on the top row, original pool below.</p>
  <div class="figs small" id="pt4f"></div>
  <div class="figs small" id="pt4o" style="margin-top:12px"></div>
</section>
</div>
<div class="part" id="sft11">
<div class="parthead"><span class="eyebrow">3 · SFT, 11-task mix and long context</span><h2 class="ptitle">11 tasks, 64k, and gain vs length</h2><p class="verdict">≈1.7–2.4× on 11 tasks; at 64k the recipe is ahead on all 10 graded tasks at 0.365× FLOPs; the gain grows to 16–32k and holds at 64k.</p></div>
<section>
  <h2>SFT: 11-task mix</h2>
  <div class="figs">
    <figure><h3>Mean over 9 tasks @8k vs training FLOPs</h3><div class="chart"><canvas id="f11" aria-label="11-task Pareto"></canvas></div>
      <figcaption>720 rows/task unless noted; the recipe also at 240/480/960. Mean over nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction, xabsence, strmatch, grouping at 8k (reorder and rerank excluded: no working grader). Dense below 240 rows/task = 240-row runs cut at 32 and 64 steps.</figcaption></figure>
  </div>
</section>
<section>
  <h2>Per task: 11-task mix @8k</h2>
  <p class="note">Each panel = one task at 8k, same runs and colours as the 11-task plot. Reorder and rerank are not shown (no working grader yet; fixes in progress).</p>
  <div class="figs small" id="pt11"></div>
</section>
<section>
  <h2>SFT: 11-task mix at 64k (10× data)</h2>
  <div class="figs">
    <figure><h3>Task average (2k–64k) vs training FLOPs</h3><div class="chart"><canvas id="m64avg" aria-label="64k task average"></canvas></div>
      <figcaption>mt11L64 data (rows up to 65,536 tokens, 662M tokens), 8× H100, 128 rows/step. Average over the 9 graded tasks and all their rungs (2k–64k; xabsence/strmatch/grouping to 32k), eval_size 500 each, no-cot. Points = 1/16, 1/8, 1/4, 1/2, 1 epoch; the router at 1/4 is 2 seeds.</figcaption></figure>
    <figure><h3>64k rung only</h3><div class="chart"><canvas id="m64r64" aria-label="64k rung"></canvas></div>
      <figcaption>Mean over the 6 tasks with a 64k rung (nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction).</figcaption></figure>
  </div>
  <p class="note">The recipe (adapters + two-stage keep + floor, 1/4 epoch, one seed) is +0.037 over dense at matched FLOPs (dense = seed-averaged, better of two LRs), ahead on all 10 graded tasks: xabsence +0.111, contradiction +0.069, outlier +0.063, grouping +0.027, oolong +0.023, hotpotqa +0.022, strmatch +0.020, reorder (2k) +0.018, qdmatch_nq +0.017, nq +0.004. Peak active memory 37.2 vs 48.7 GiB on 4 H100s.</p>
  <div class="figs small" id="pt64" style="margin-top:12px"></div>
</section>
<section>
  <h2>Does the advantage grow with context length?</h2>
  <div class="figs">
    <figure><h3>Router gain over dense at matched FLOPs, by eval length</h3><div class="chart"><canvas id="lenD" aria-label="gain by eval length"></canvas></div>
      <figcaption>64k study (training rows up to 64k). Each line = one router budget (1/4, 1/2, 1 epoch); y = router score minus dense (LR 5e-6) interpolated to the router's FLOPs, averaged over the same 6 tasks at every length (nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction), so lengths compare like with like.</figcaption></figure>
    <figure><h3>Dense FLOPs needed to match the router, by eval length</h3><div class="chart"><canvas id="lenF" aria-label="FLOPs to match by eval length"></canvas></div>
      <figcaption>Same runs: training PF dense needs to reach the router's score at that length, divided by the router's PF. Above 1 = the router is FLOP-cheaper.</figcaption></figure>
    <figure><h3>CPT: whole-model LoRA vs dense, training at 32k, 64k and 128k</h3><div class="chart"><canvas id="lenC" aria-label="CPT gap by training length"></canvas></div>
      <figcaption>Dev CE of whole-model LoRA minus dense at the same step count, full-row scoring (more negative = better). The gap is ≈1.4–1.5× larger at 64k than at 32k and about the same at 128k as at 64k. On the far half of each row alone (32k–64k, 64k–128k positions) it is essentially the same.</figcaption></figure>
  </div>
  <p class="note" id="lenNote"><b>Reading:</b> at 2k the router barely beats dense (Δ ≈ 0.001–0.004; these tasks are near their ceiling at 2k). The gain rises to +0.015–0.033 at 8k and +0.03–0.05 at 16k–32k, then stays at +0.02–0.04 at 64k; dense FLOPs to match go from ≈1.1–1.3× at 2k to ≈1.4–1.9× at 8k–32k and ≈1.3–1.6× at 64k. So the advantage grows with eval length up to ~16–32k and holds at 64k, rather than growing without bound (1/2- and 1-epoch points are one seed). In CPT (full-row scoring), LoRA's lead over dense is ≈1.4–1.5× larger when training at 64k than at 32k, and about the same at 128k as at 64k: it grows to 64k and then stops.</p>
</section>
</div>
<div class="part" id="models">
<div class="parthead"><span class="eyebrow">3b · Other models</span><h2 class="ptitle">Does the recipe generalize? Qwen3 and model scale</h2><p class="verdict">On all-full-attention Qwen3-4B the recipe still wins, but by less: ≈2.1× dense FLOPs-to-match at 1,952 rows vs ≈4.6× on Qwen3.5 (interim, 1 seed). Across Qwen3.5 sizes the gain grows with scale at fixed data (≈1.7× at 0.8B → ≈2.5× at 2B → ≈3.7–4.1× at 4B).</p></div>
<section>
  <h2>Qwen3-4B (36 full-attention layers)</h2>
  <div class="figs">
    <figure><h3>Qwen3-4B: mean score vs training FLOPs</h3><div class="chart"><canvas id="q3par" aria-label="Qwen3 Pareto"></canvas></div>
      <figcaption>4-task fresh pools, 8-cell mean, eval_size 500, no-cot, 1 seed. Dense at its better LR per size. Recipe arms are named by drop depths: keep 20% after block a, 7% after block b (of 36). Routing-only = the 12/24 split without LoRA (full fine-tuning).</figcaption></figure>
    <figure><h3>Dense FLOPs to match: Qwen3 vs Qwen3.5</h3><div class="chart"><canvas id="q3ftm" aria-label="FLOPs to match Qwen3 vs Qwen3.5"></canvas></div>
      <figcaption>Qwen3: best split per size (12/24 at 1,952 rows; 8/18 at 3,904 and 7,808, where the values are lower bounds until dense at 3,904 / 7,808 is scored). Qwen3.5: the recipe (7/16).</figcaption></figure>
  </div>
  <div class="tbl" style="margin-top:14px"><table>
    <tr><th>Qwen3 run</th><th class="n">rows/task</th><th class="n">PF</th><th class="n">8-cell mean</th><th class="n">outlier@8k</th><th class="n">contra@8k</th><th class="n">dense FLOPs to match</th><th class="n">peak GiB</th></tr>
    <tr><td>dense</td><td class="n">488 / 976 / 1,464 / 1,952</td><td class="n">437 / 878 / 1,317 / 1,761</td><td class="n">0.591 / 0.724 / 0.761 / 0.807</td><td class="n">0.12 / 0.28 / 0.35 / 0.62</td><td class="n">0.48 / 0.81 / 0.90 / 0.94</td><td class="n">1.0×</td><td class="n">55.8</td></tr>
    <tr><td>recipe, split 4/18</td><td class="n">1,952</td><td class="n">292</td><td class="n">0.574</td><td class="n">0.257</td><td class="n">0.281</td><td class="n">≤1.5×</td><td class="n">19.5</td></tr>
    <tr><td>recipe, split 8/18 (proportional)</td><td class="n">1,952 / 3,904 / 7,808</td><td class="n">395 / 789 / 1,578</td><td class="n">0.647 / 0.819 / 0.906</td><td class="n">0.22 / 0.64 / 0.95</td><td class="n">0.67 / 0.88 / 0.95</td><td class="n">1.49 / ≥2.23 / ≥1.12×</td><td class="n">19.5</td></tr>
    <tr><td>recipe, split 12/18</td><td class="n">1,952 / 3,904</td><td class="n">498 / 996</td><td class="n">0.725 / 0.854</td><td class="n">0.35 / 0.76</td><td class="n">0.81 / 0.94</td><td class="n">1.78 / ≥1.77×</td><td class="n">19.5</td></tr>
    <tr><td><b>recipe, split 12/24</b></td><td class="n">1,952</td><td class="n">518</td><td class="n">0.743</td><td class="n">0.403</td><td class="n">0.849</td><td class="n"><b>2.09×</b></td><td class="n">19.5</td></tr>
    <tr><td>recipe, split 16/24 · 16/27</td><td class="n">1,952</td><td class="n">621 · 631</td><td class="n">0.721 · 0.717</td><td class="n">0.30 · 0.31</td><td class="n">0.89 · 0.91</td><td class="n">1.39 · 1.34×</td><td class="n">19.7</td></tr>
    <tr><td>routing only (12/24, no LoRA, LR 5e-6)</td><td class="n">1,952</td><td class="n">718</td><td class="n">0.719</td><td class="n">0.279</td><td class="n">0.836</td><td class="n">1.19×</td><td class="n">57.0</td></tr>
  </table></div>
  <p class="note"><b>Reading:</b> the per-row FLOP cut is the same as on Qwen3.5 (0.22× at split 8/18, 0.29× at 12/24); attention is only ≈20–25% of Qwen3's FLOPs at 8k, so its quadratic saving barely moves the ratio. The difference is accuracy at matched data: −0.16 vs dense at 1,952 rows (split 8/18) against −0.006 on Qwen3.5, mostly outlier and contradiction, shrinking with data (outlier@8k 0.22 → 0.64 → 0.95). Early drops hurt more on Qwen3, so the best split sits deeper (12/24, one and two thirds of depth). Unlike the Qwen3.5 2×2, LoRA matters here: routing alone is 1.19× vs 2.09× with LoRA, and LoRA's frozen base is the memory win (19.5 vs 57 GiB). Still running: dense at 3,904 / 7,808, split 12/24 at 3,904, dense + LoRA-all at 1,952. Single-seed noise ≈ ±0.01–0.02 on the mean. JSON: <code>research/token_routing/ds64_fast8k/q3_flops_to_match.json</code> (branch <code>claude/qwen3-recipe-c876d1</code>).</p>
</section>
<section>
  <h2>Model scale: Qwen3.5 0.8B, 2B and 4B</h2>
  <div class="figs">
    <figure><h3>Dense FLOPs to match vs model size</h3><div class="chart"><canvas id="scftm" aria-label="FLOPs to match vs model size"></canvas></div>
      <figcaption>Recipe vs each size's own dense curve (dense LR 5e-6, recipe LR 1e-5, piecewise log-linear in PF). 0.8B at 1,952 rows is the mean of 3 seeds, 2B of 2. The 0.8B 7,808-row point is a lower bound (the recipe passes dense's best). 4B uses the same method, so it differs slightly from the 4-task table.</figcaption></figure>
    <figure><h3>0.8B: mean score vs training FLOPs</h3><div class="chart"><canvas id="sc08" aria-label="0.8B Pareto"></canvas></div>
      <figcaption>8-cell mean, eval_size 500, no-cot. Dense's cheapest points are runs cut at 122 / 244 steps. Recipe = whole-model LoRA + keep 20% after block 7, 7% after block 12 (of 24).</figcaption></figure>
    <figure><h3>2B: mean score vs training FLOPs</h3><div class="chart"><canvas id="sc2" aria-label="2B Pareto"></canvas></div>
      <figcaption>Same setup as 0.8B (24 blocks).</figcaption></figure>
    <figure><h3>4B: mean score vs training FLOPs</h3><div class="chart"><canvas id="sc4" aria-label="4B Pareto"></canvas></div>
      <figcaption>Fresh pools, recipe 7/16 (of 32), for reference.</figcaption></figure>
  </div>
  <div class="tbl" style="margin-top:14px"><table>
    <tr><th>size</th><th class="n">recipe FLOPs / step vs dense</th><th class="n">peak memory vs dense</th><th class="n">FTM, 1,952 rows</th><th class="n">FTM, 3,904 rows</th><th class="n">FTM, 7,808 rows</th><th class="n">dense ceiling (best mean)</th></tr>
    <tr><td>0.8B</td><td class="n">0.241×</td><td class="n">0.60×</td><td class="n">1.73× (3 seeds)</td><td class="n">1.67×</td><td class="n">&gt;4.16×</td><td class="n">0.743</td></tr>
    <tr><td>2B</td><td class="n">0.247×</td><td class="n">0.57×</td><td class="n">2.48× (2 seeds)</td><td class="n">2.52×</td><td class="n">running</td><td class="n">0.880 (5,856 rows)</td></tr>
    <tr><td>4B</td><td class="n">0.225×</td><td class="n">0.47×</td><td class="n">4.08×</td><td class="n">3.68×</td><td class="n">2.17×</td><td class="n">0.931</td></tr>
    <tr><td>9B</td><td class="n">0.226×</td><td class="n">0.60×</td><td class="n" colspan="3">training / evaluating</td><td class="n">–</td></tr>
  </table></div>
  <p class="note"><b>Reading:</b> at matched small data the recipe's gain grows with scale (≈1.7× → 2.5× → 3.7–4.1× from 0.8B to 4B at 1,952 / 3,904 rows), at the same per-step cut (≈0.23–0.25×). The lower ceilings cut the other way at large data: 0.8B dense plateaus at 0.743 (outlier@8k stuck near 0.3) while the frozen-base LoRA recipe keeps improving (0.806 at 7,808 rows), and at 4B dense's high ceiling (0.93) is what shrinks the multiplier to ≈2× at 7,808 rows. So compare scales at fixed small data. The recipe is less stable at 0.8B (contradiction@8k 0.38–0.76 across 3 seeds; ±0.03 on the mean). Higher LRs (dense 2e-5, recipe 3e-5) are worse at both small sizes, so LR is not the confound. Peak memory is not like-for-like across sizes (different GPU counts). 9B and the 2B 7,808-row points are in flight. JSON: <code>research/token_routing/ds64_fast8k/scale_ablation.json</code> (from <code>collect_scale_ablation.py</code>, branch <code>claude/recipe-scale-ablation-71d007</code>).</p>
</section>
</div>
<div class="part" id="cpt">
<div class="parthead"><span class="eyebrow">4 · Continued pretraining</span><h2 class="ptitle">Long-document CPT, OLMo 3 7B</h2><p class="verdict">Frozen-output-layer whole-model LoRA: never behind dense up to 960 steps, 4.8× → 1.4× FLOPs-to-match; every token-drop variant loses.</p></div>
<section>
  <h2>Long-document CPT (OLMo 3 7B)</h2>
  <div class="figs">
    <figure><h3>32k: dev CE vs training FLOPs (lower is better)</h3><div class="chart"><canvas id="cpt32" aria-label="CPT 32k"></canvas></div>
      <figcaption>dev32k_1024 (1,024 rows of 32k), full-context CE; paired SE ≈ ±0.0001, seed-to-seed ≈ ±0.0001. Dense curve 5–960 steps.</figcaption></figure>
    <figure><h3>LoRA's gap to dense at 32k, 64k and 128k context</h3><div class="chart"><canvas id="cpt64" aria-label="CPT 64k"></canvas></div>
      <figcaption>Whole-model LoRA minus dense dev CE at the same step count, scored on full rows (dev32k_1024 / dev64k_512 / dev128k_512; paired SE ≤ ±0.0003). More negative = LoRA better. Solid = frozen output layer (the CPT recipe, 0.68× FLOPs at every length); dashed = trained output layer. The frozen-head recipe is 0.0003–0.0009 better at every point. At 64k its 240-step run ties dense at 480 steps (≈2.9× FLOPs to match).</figcaption></figure>
    <figure><h3>Whole-model LoRA: dense FLOPs to match, by budget (32k)</h3><div class="chart"><canvas id="cptm" aria-label="CPT FLOPs to match"></canvas></div>
      <figcaption>The edge peaks near 48 steps and decays as dense keeps learning, reaching parity by 960 steps (dev CE +0.0012 behind dense there; far-repeat CE +0.0035). At 64k the gap is ≈1.5× larger at every budget.</figcaption></figure>
  </div>
</section>
</div>
<div class="part" id="memory">
<div class="parthead"><span class="eyebrow">5 · Memory and wall-clock</span><h2 class="ptitle">Max trainable length and real step time</h2><p class="verdict">Routing and LoRA raise max length ≈1.8–2.4×; wall-clock savings lag FLOPs (0.54–0.65× time for 0.22–0.30× FLOPs).</p></div>
<section>
  <h2>Memory: max trainable length at 80 GB</h2>
  <div class="figs">
    <figure><h3>Peak memory vs row length (standard settings)</h3><div class="chart"><canvas id="memstd" aria-label="memory vs length standard"></canvas></div>
      <figcaption>Peak GPU memory per GPU at the steady (routed) training step vs tokens per row. Qwen3.5-4B, 2× A100-80GB, FSDP-2, full checkpointing. The slope is memory per token: dense 0.39 GiB per 1k tokens, routed arms ≈0.25. Dense out of memory at 128k. Markers = measured; dashed = linear fit run out to where it reaches the 80 GB line (the estimated max length); dotted = 79.25 GiB usable.</figcaption></figure>
    <figure><h3>Peak memory vs row length (all memory optimizations)</h3><div class="chart"><canvas id="memflags" aria-label="memory vs length optimized"></canvas></div>
      <figcaption>Same probe with routed warmup, no dry run, chunked FFN recompute and segmented GDN. Slopes: dense 0.28, router 0.17, whole-model LoRA 0.16 GiB per 1k tokens. Markers = measured, dashed = linear fit run out to the 80 GB line (the recipe has one point at this setting, so its line uses the slope from the chunked-FFN round). Two-stage without adapters and the 20% router were measured on Beaker H100s (per-length memory matches the A100 probes to 0.03 GiB).</figcaption></figure>
    <figure><h3>Max train length at 80 GB</h3><div class="chart"><canvas id="maxt" aria-label="max train length"></canvas></div>
      <figcaption>Qwen3.5-4B SFT, 2× A100-80GB, FSDP-2, full activation checkpointing, chunked FFN + segmented GDN. Linear extrapolation from 3-step probes, one sample each (router session record).</figcaption></figure>
  </div>
  <div class="tbl" style="margin-top:14px"><table>
    <tr><th>CPT, OLMo 3 7B (8× H200, FSDP, full checkpointing)</th><th class="n">GiB per +32k tokens</th><th class="n">max T, 80 GB-class</th><th class="n">max T, H200</th></tr>
    <tr><td>dense</td><td class="n">20.1</td><td class="n">≈95k</td><td class="n">≈190k</td></tr>
    <tr><td>whole-model LoRA</td><td class="n">20.3</td><td class="n">≈111k</td><td class="n">≈206k</td></tr>
    <tr><td>dense + activation offload</td><td class="n">12.7</td><td class="n">≈126k</td><td class="n">≈277k</td></tr>
    <tr><td>whole-model LoRA + offload</td><td class="n">11.6</td><td class="n">≈157k</td><td class="n">≈323k</td></tr>
  </table></div>
  <p class="note">LoRA lowers the fixed part (no gradients or Adam state for frozen weights); routing lowers the per-token slope (only kept tokens store block inputs past the split). Offload is an implementation fix that helps every arm.</p>
</section>
<section>
  <h2>Wall-clock vs FLOPs</h2>
  <div class="tbl"><table>
    <tr><th>method</th><th class="n">FLOPs vs dense</th><th class="n">step time vs dense (today)</th><th class="n">ideal (time = FLOPs)</th><th class="n">predicted after packing + fused LoRA + compile</th></tr>
    <tr><td>SFT cheap methods (router, recipe, LoRA-all)</td><td class="n">0.22–0.30×</td><td class="n">0.54–0.65×</td><td class="n">0.22–0.30×</td><td class="n">≈0.29–0.47×</td></tr>
    <tr><td>Dense, skip blocks 20–27</td><td class="n">0.78×</td><td class="n">≈0.83×</td><td class="n">0.78×</td><td class="n">–</td></tr>
    <tr><td>Dense, update only top 25% of layers (backward reaches block 0)</td><td class="n">0.79×</td><td class="n">0.90×</td><td class="n">0.79×</td><td class="n">–</td></tr>
    <tr><td>Dense, update only top 25% (backward stops at block 24)</td><td class="n">0.57×</td><td class="n">0.54×</td><td class="n">0.57×</td><td class="n">–</td></tr>
    <tr><td>CPT whole-model LoRA at 128–192k</td><td class="n">0.70×</td><td class="n">≈0.97×</td><td class="n">0.70×</td><td class="n">attention-bound at this length</td></tr>
  </table></div>
  <p class="note">LoRA saves FLOPs but little time today because its adapter hooks are unfused; routing loses time to gather/scatter and unpacked rows. Comparisons are against pad-free dense on the same cluster; layer-skip timings are same-node H200 A/Bs.</p>
</section>
<section>
  <h2>Per-layer skipping overhead: token vs chunk (microbenchmark)</h2>
  <div class="figs">
    <figure><h3>Time vs a full layer, 32k tokens</h3><div class="chart"><canvas id="skb32" aria-label="skip overhead 32k"></canvas></div>
      <figcaption>Forward + backward of one real Qwen3.5-4B block (GDN = block 0, full attention = block 3), 1 H200, bf16, random weights, one row. "Ideal" = the block on the kept tokens with no gather (what dropping a token for all later layers costs per layer); the others gather the kept tokens, run the block and scatter back into the full-length stream, as per-layer skipping must at every layer.</figcaption></figure>
    <figure><h3>Time vs a full layer, 8k tokens</h3><div class="chart"><canvas id="skb8" aria-label="skip overhead 8k"></canvas></div>
      <figcaption>Same at 8k. Timings here are noisy (10 iterations; a few gathered runs come out faster than "ideal"), but the main point is robust: at 8k the shortened block saves far less than its FLOPs.</figcaption></figure>
  </div>
  <p class="note"><b>Reading:</b> the gather / scatter itself is cheap: at 32k it adds 1–5% of a full layer's time per layer (token level: 4.7% on GDN, 1% on attention; 100- or 128-token chunks: ≈2.5% and ≈1%). Chunking halves the overhead on GDN but there is little to win. The real wall-clock loss is short sequences running inefficiently: at 8k, keeping 25% of tokens makes a GDN block only 30% faster and an attention block 64% faster (vs ≈75–80% fewer FLOPs for both), while at 32k the savings match the FLOPs (GDN 0.26× time at 25% kept; attention 0.12×, close to quadratic). So per-layer and chunked skipping are not limited by data movement; they are limited by the same small-kernel inefficiency as the router, and their quality cost (mixture-of-depths: −0.25 outlier) remains. Script: <code>research/token_routing/chunk_skip_bench/bench_skip_overhead.py</code>; results: <code>results_h200.json</code>.</p>
</section>
</div>
<div class="part" id="mechanism">
<div class="parthead"><span class="eyebrow">6 · Mechanism</span><h2 class="ptitle">Where the gradient lives, and what the router gets right</h2><p class="verdict">Blocks 0–6 must see every token; the router is near a perfect predictor, but token choice matters far less than where and how much is dropped.</p></div>
<section>
  <h2>Where the gradient lives, and how that shifts during training</h2>
  <p class="note">From the gradient-composition probe: one full forward + backward of the answer loss per held-out row on frozen checkpoints along the dense fine-tuning curve (Qwen3.5-4B, 4-task rows, 64 per task ⚠ below 500; differences under ≈±0.03 are noise). Full interactive version: <a href="https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ">gradient composition</a>.</p>
  <div class="figs">
    <figure><h3>Share of the parameter gradient in each block</h3><div class="chart"><canvas id="gcpg" aria-label="parameter gradient by block"></canvas></div>
      <figcaption>‖g‖² share per block (32 blocks + LM head = 1), mean of per-row gradients. At the base the gradient is front-loaded (block 0 alone 0.16); training moves it toward the middle blocks and the LM head.</figcaption></figure>
    <figure><h3>Blocks 0–6 share of the parameter gradient over training</h3><div class="chart"><canvas id="gcearly" aria-label="early block share over training"></canvas></div>
      <figcaption>Most of the drop happens in the first ~30 SFT steps (12 in CPT). This is why freezing the early blocks from step 0 hurts but freezing after a warmup costs little, and why adapters (which keep them learning cheaply) work.</figcaption></figure>
    <figure><h3>Share of each block's hidden-state gradient on document tokens</h3><div class="chart"><canvas id="gcdoc" aria-label="document share by block"></canvas></div>
      <figcaption>‖dL/dh_b‖² on document body tokens (vs query, prompt, markers, answer). After even 31 steps documents dominate the early-mid blocks (0.54 at block 7) and fade with depth (0.11 at block 24 after 366 steps): the case for splitting near block 7 and letting documents leave later.</figcaption></figure>
    <figure><h3>How well block 7's token ranking predicts each later block</h3><div class="chart"><canvas id="gcx7" aria-label="spearman with block 7"></canvas></div>
      <figcaption>Spearman of body-token ‖dL/dh‖ between block 7 and block b. With training the late blocks decorrelate (block 24: 0.57 → 0.34), so one keep decision at block 7 goes stale: the case for rescoring at block 16 (two-stage keep).</figcaption></figure>
    <figure><h3>CPT: how much of a token's gradient comes from serving as context</h3><div class="chart"><canvas id="gcctx" aria-label="CPT context share"></canvas></div>
      <figcaption>OLMo 3 7B, 48 rows × 32k ⚠. Share of each block's hidden-state gradient a token receives as keys/values for later tokens (vs its own label). Early blocks learn almost entirely as context, so dropping tokens early removes their learning signal.</figcaption></figure>
  </div>
</section>
<section>
  <h2>Does the router predict the right tokens, and does it matter?</h2>
  <div class="tbl"><table>
    <tr><th>probe</th><th>router</th><th>perfect predictor</th><th>random</th><th>what it says</th></tr>
    <tr><td>Offline accuracy (base model): rank correlation with per-token gradient norm; share of squared gradient in the predicted top 30%</td><td class="n">0.5–0.6; 0.70–0.95</td><td class="n">1.0; 0.86–0.95</td><td class="n">0; 0.30</td><td>cheap block-8 features predict gradient size well</td></tr>
    <tr><td>Gradient fidelity v1 (trained checkpoint, keep 30%): per-row cosine with the full-pass gradient, drop after block 7 / at the input</td><td class="n">0.78 / 0.45</td><td class="n">0.83 / 0.50</td><td class="n">0.10 / 0.06</td><td>where the drop happens matters far more than the predictor</td></tr>
    <tr><td>Fidelity v2 (recipe checkpoint): cosine of the 8-batch accumulated update</td><td class="n">0.45</td><td class="n">–</td><td class="n">0.08</td><td>random's error is mostly variance per row but does not average out in the probe</td></tr>
    <tr><td>Fidelity v2: perfect predictor, split after block 7 / 5 / 3 / 1 (8-batch cosine)</td><td colspan="3" class="n">0.41 / 0.08 / 0.08 / 0.03</td><td>early drops fail by bias even with perfect knowledge: no router fixes them</td></tr>
    <tr><td>Training outcome: router vs random token choice (whole-model LoRA + two-stage)</td><td colspan="3" class="n">+0.027 at 1,464 rows; tie at 3,904</td><td>the drop structure, not the token choice, carries most of the gain</td></tr>
    <tr><td>Training outcome: MLP router vs linear (online rank correlation 0.54 vs 0.37)</td><td colspan="3" class="n">same score</td><td>better prediction does not buy task quality</td></tr>
  </table></div>
  <p class="note">⚠ Probe sizes: offline accuracy 32 held-out rows/task; fidelity v1 12 rows/task; fidelity v2 64 rows/task (256 total). Single-step fidelity is off-policy (it scores random selection on router-trained checkpoints), which likely explains why it over-predicts the value of token choice. Which tokens each method keeps, on real rows: <a href="https://claude.ai/artifact/KsLw2QqRugmkkXCvXknbBg">router keep masks</a>. Full probe write-ups: <a href="https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1">online router</a> and <code>records/online-token-router.md</code> (router branch).</p>
</section>
<section>
  <h2>Why it works where it works</h2>
  <ul>
    <li><b>Blocks 0–6 must see every token;</b> the router decides at block 7 from features that predict each token's gradient norm (held-out Spearman 0.54–0.63). The gradient-composition study shows rankings drift with depth (Spearman between blocks 7 and 24 falls 0.57 → 0.34), which is why rescoring at block 16 helps. ⚠ probes on 32–64 rows/task.</li>
    <li><b>In SFT only answer tokens carry loss</b>, and the gradient concentrates on few document tokens (top 10% hold ≈0.9 of late-layer gradient energy).</li>
    <li><b>In CPT every token is a label</b> and importance is flat (top 10% hold 0.62); early blocks learn mostly as context for later tokens (0.95 of block-0 gradient). So dropping tokens removes most of the learning signal, while freezing weights behind adapters does not. ⚠ 48 rows × 32k.</li>
  </ul>
</section>
</div>
<div class="part" id="ruledout">
<div class="parthead"><span class="eyebrow">7 · Ruled out</span><h2 class="ptitle">What did not work</h2><p class="verdict">Router-quality tweaks, layer skipping and earlier drops all land at ≤1.3× over the recipe or lose.</p></div>
<section>
  <h2>What failed, and the number that decided it</h2>
  <div class="tbl"><table>
    <tr><th>idea</th><th>track</th><th>result</th></tr>
    <tr><td>Fixed keep rules (gold whole + trimmed context)</td><td>SFT</td><td class="neg">gold shortcut: contradiction@8k 0.03–0.07 at every LR (dense 0.89)</td></tr>
    <tr><td>Gold-blind fixed rules, two-tier rules</td><td>SFT</td><td>best only ties dense at 0.57× FLOPs</td></tr>
    <tr><td>Dropping at the input, or before block 7</td><td>SFT</td><td class="neg">contradiction 0.77–0.85 vs 0.93 when blocks 0–6 see every token</td></tr>
    <tr><td>Activation-based router target</td><td>SFT</td><td class="neg">outlier 0.36 vs 0.64 with the gradient-norm target</td></tr>
    <tr><td>Frozen / no-grad early stack from step 0, pyramid, merge</td><td>SFT</td><td class="neg">contradiction 0.34–0.55 vs 0.66</td></tr>
    <tr><td>Layer skip in the routed stack</td><td>SFT</td><td>−0.03 contradiction, −0.09 outlier for 4% FLOPs (2 seeds)</td></tr>
    <tr><td>Partial MLP width (random or importance-picked)</td><td>SFT + CPT</td><td class="neg">recipe + half MLP: mean 0.75 vs 0.84; CPT +0.08 nats</td></tr>
    <tr><td>Freezing the LoRA stack partway</td><td>SFT 11-task</td><td>xabsence 0.54–0.58 vs 0.68–0.71 unfrozen</td></tr>
    <tr><td>Layer skipping (fixed 20/24/28 or 20–27, LayerDrop on blocks 16–29, gradient-ranked top-25/50% layers)</td><td>SFT 4-task, 1,464 + 3,904 rows</td><td>dense: within noise of dense at 0.78–0.92× FLOPs (Δ −0.03 to +0.02); recipe + skip: −0.007 to −0.13 for 3–8% FLOPs; running only the top 25/50% of layers collapses (records/layer-skip-screen.md, branch claude/train-layerskip-35010b)</td></tr>
    <tr><td>Router hill-climb: threshold keep, MLP router, doc-context features, task one-hot, per-task routers, per-group ridge, keep schedule</td><td>SFT 4-task, 1,464 rows, 1–4 seeds</td><td>all tie the linear router (0.83–0.86 vs 0.852); first stage at block 5 −0.033; adaptive stage-2 budget 0.864 over 3 seeds there, but it ties the recipe on fresh data (0.880 / 0.918 / 0.922 vs 0.883 / 0.915 / 0.924 at 1,952 / 3,904 / 7,808 rows) and loses 0.03 on the 11-task mix (xabsence −0.10, outlier −0.11)</td></tr>
    <tr><td>Dropping before block 7 (after block 1–6)</td><td>SFT 4-task + fidelity probe</td><td>cliff at block 6→7 (outlier 0.38→0.69); even a perfect gradient-norm predictor loses the gradient direction (8-batch cos 0.03–0.08 vs 0.41 at block 7), so no router fixes it</td></tr>
    <tr><td>Earlier first stage on the recipe (after block 1 / 3 / 5), fresh 3,904 rows, 1 seed</td><td>SFT 4-task</td><td>0.839 / 0.872 / 0.880 vs the recipe's 0.916, at 0.61 / 0.73 / 0.87× its FLOPs: −0.02 to −0.04 at matched FLOPs</td></tr>
    <tr><td>Pyramid 50% after block 3 → 20% after 7 → 7% after 16</td><td>SFT 4-task</td><td>0.870 / 0.906 (3 seeds) / 0.925 vs the recipe's 0.883 / 0.916 / 0.924 at 1,952 / 3,904 / 7,808 rows, at 0.83× its FLOPs: a tie, ≈1.2× at best (below the 5× bar)</td></tr>
    <tr><td>Any token dropping (router, input drop, MoD, keep 50%)</td><td>CPT</td><td class="neg">router FLOPs-to-match &lt;0.41 → 0.06; keep 50% + LoRA worse than dense after 4 steps</td></tr>
    <tr><td>Early-exit loss on dropped tokens</td><td>CPT</td><td class="neg">dev CE rises with training (1.737 → 1.771)</td></tr>
    <tr><td>All-local attention, 4k-only rows</td><td>CPT</td><td class="neg">context gain collapses (0.075 → 0.044 / 0.0005)</td></tr>
    <tr><td>Stopping backprop below block B</td><td>CPT</td><td>FLOPs-to-match 2.9 / 1.4 at B = 4 / 8, below whole-model LoRA's 3.45</td></tr>
  </table></div>
</section>
<section>
  <h2>Small-model token selection: cost ceiling</h2>
  <div class="figs">
    <figure><h3>Best-case speedup over the recipe, by selector</h3><div class="chart"><canvas id="smsel" aria-label="small-model selection ceiling"></canvas></div>
      <figcaption>Total training PF at fresh 3,904 rows/task if the 4B drops from block 0 and loses no score, relative to the recipe's 688 PF (above 1 = cheaper than the recipe). Cost model only, no training.</figcaption></figure>
  </div>
  <div class="tbl" style="margin-top:14px"><table>
    <tr><th>selector</th><th class="n">selector PF</th><th class="n">total, keep 20%→7% from block 0</th><th class="n">× recipe</th><th class="n">total, keep 7% from block 0</th><th class="n">× recipe</th></tr>
    <tr><td>Qwen3.5-0.8B, forward only</td><td class="n">138</td><td class="n">510</td><td class="n">1.35</td><td class="n">366</td><td class="n">1.88</td></tr>
    <tr><td>Qwen3.5-0.8B, gradient-norm scoring</td><td class="n">276</td><td class="n">648</td><td class="n">1.06</td><td class="n">503</td><td class="n">1.37</td></tr>
    <tr><td>Qwen3.5-0.8B, first 6 blocks only</td><td class="n">33</td><td class="n">406</td><td class="n">1.70</td><td class="n">261</td><td class="n">2.64</td></tr>
    <tr><td>≈135M model (different tokenizer)</td><td class="n">30</td><td class="n">402</td><td class="n">1.71</td><td class="n">257</td><td class="n">2.67</td></tr>
    <tr><td>Qwen3.5-2B, forward only</td><td class="n">337</td><td class="n">709</td><td class="n">0.97</td><td class="n">564</td><td class="n">1.22</td></tr>
  </table></div>
  <p class="note"><b>Verdict: not worth it.</b> The recipe is 688 PF, of which ≈313 PF (45%) is blocks 0–6 at full length; a 4B that drops from block 0 costs ≈372 PF. Even a nearly free selector keeping 7% from block 0 reaches only ≈2.6× over the recipe (≈8× dense), and only if the score does not drop. Measured early drops already lose 0.04–0.08 at this budget, and random token choice scores like the router (0.914 vs 0.915), so better selection cannot recover it: at its realistic score a block-0 drop with the 0.8B is ≈1.9× dense, below the recipe's ≈3×. Cost model: <code>research/token_routing/ds64_fast8k/small_model_cost.py</code> on branch <code>claude/small-model-select-0a941a</code>.</p>
</section>
<section>
  <h2>KV-only context for dropped tokens: fidelity probe</h2>
  <div class="figs">
    <figure><h3>8-batch gradient cosine, plain drop vs KV-only context (recipe checkpoint)</h3><div class="chart"><canvas id="kvcos" aria-label="KV-only fidelity"></canvas></div>
      <figcaption>Cosine of the 8-batch (128-row) accumulated update with the full-pass update, router selection, trained parameters, recipe checkpoint. "1st @ b" = one-stage keep 10% after block b; "2st @ b" = keep 20% after block b then 7% after 16. ⚠ eval_size 256 (64 rows/task).</figcaption></figure>
  </div>
  <p class="note"><b>Verdict: fails; no training screen.</b> Letting dropped tokens stay as frozen key/value context closes only ≈10–35% of the gap to the block-7 split on the trained recipe checkpoint: after block 1 the 8-batch cosine goes 0.00 → 0.03 (one stage) and −0.00 → 0.04 (two stage); after block 3, 0.04 → 0.11 and 0.12 → 0.15; against 0.24–0.45 at block 7. Even a perfect token picker plus KV-only context stays at 0.06 / 0.14 after blocks 1 / 3. On the base model it recovers more (≈40–65%), but training moves the model away from that. Branch <code>claude/kv-only-context-9ee75b</code>, table <code>research/token_routing/online_router/gfkv_table.md</code>.</p>
</section>
</div>
<div class="part" id="next">
<div class="parthead"><span class="eyebrow">8 · Next ideas</span><h2 class="ptitle">Ideas with a ≥5× ceiling</h2><p class="verdict">Shift from per-token tweaks to per-row savings: shared-context multi-query training and row selection.</p></div>
<section>
  <h2>Candidates for ≥5–10× over dense</h2>
  <div class="tbl"><table>
    <tr><th>idea</th><th>how it saves</th><th class="n">rough ceiling</th><th>status</th></tr>
    <tr><td>Shared-context multi-query training</td><td>k questions per corpus; run the corpus once, train on all k answers</td><td class="n">≈k× on top of routing</td><td>dropped 10-29: no drastic data changes for now</td></tr>
    <tr><td>Row selection</td><td>train only on the hardest 10–20% of fresh rows</td><td class="n">3–5× on top of routing</td><td>dropped 10-29: no drastic data changes for now</td></tr>
    <tr><td>Recipe on Qwen3-4B (all full attention)</td><td>routing saves attention quadratically; more attention may mean larger gains</td><td class="n">?</td><td>running (side session)</td></tr>
    <tr><td>Model-scale ablation (smaller Qwen3.5 sizes)</td><td>does the gain grow or shrink with scale?</td><td class="n">?</td><td>running (side session)</td></tr>
    <tr><td>Small-model front end</td><td>a 0.8B (or smaller) model reads the full context and picks tokens; the 4B sees only the kept ones</td><td class="n">≈1.35× over the recipe (0.8B); ≤2.6× even with a free selector</td><td>tested 10-29 (cost model): dropped, see Section 7</td></tr>
  </table></div>
  <p class="note">Ceilings are back-of-envelope estimates, not measurements. Prasann (10-29): pursue only ideas whose ceiling is at least ≈5× over dense.</p>
</section>
</div>
<div class="part" id="provenance">
<div class="parthead"><span class="eyebrow">9 · Provenance</span><h2 class="ptitle">Data, code, caveats and detail pages</h2><p class="verdict">Every number traces to ops results, records and the files listed here.</p></div>
<section>
  <h2>Caveats</h2>
  <ul>
    <li>Most screen points are one seed; outlier@8k varies up to ±0.19 between seeds. Dense mt4 reference runs at the same rows differ by up to 0.02 mean / 0.07 outlier between Lambda and Beaker copies.</li>
    <li>The FLOP meter counts forward + backward, not the activation-checkpoint recompute. Every 4B SFT run and every CPT run uses full checkpointing, so all are undercounted similarly; adapter runs are undercounted relatively more (CPT whole-model LoRA ≈0.75× real vs 0.70× metered).</li>
    <li>CPT gains are small in absolute terms (0.001–0.005 nats) because dense CPT saturates within ~12 steps; the FLOPs-to-match ratios are real but sit on a flat dense curve. No task-level (generation) eval of CPT checkpoints yet: that needs a matched SFT stage.</li>
    <li>Max train length numbers are linear extrapolations from short probes, one sample each.</li>
  </ul>
</section>
<section>
  <h2>Where everything lives</h2>
  <div class="tbl"><table>
    <tr><th>what</th><th>where</th></tr>
    <tr><td>SFT data (4-task)</td><td><code>weka prasanns/ds64/fast8k/shards/mt4s{488..1952}</code> (ds64 8k pool) and fresh <code>mt4f{1952,3904,5856,7808}</code> (built by <code>build_mt4_fresh_beaker.sh</code>, new generator seed 5310, 8,200 rows/task; zero train/eval example overlap on nq and outlier, not verified for contradiction/oolong)</td></tr>
    <tr><td>SFT data (11-task, 64k)</td><td><code>mt11s{240..960}</code> (setA seed-42 shards); 64k: <code>mt11L64</code>, 156,587 rows, 662M tokens</td></tr>
    <tr><td>SFT eval</td><td>8k and 2k rungs × 500 rows/task, no-cot; 4-task on ds64 ladder files, 11-task on CTC-suite rows (query position both)</td></tr>
    <tr><td>All SFT scores and FLOPs</td><td><code>research/token_routing/ds64_fast8k/results/pareto_points.json</code> (from <code>collect_pareto_points.py</code>); ops results log (<code>ops result show --experiment tokdrop_lr|learned_router</code>)</td></tr>
    <tr><td>Launchers</td><td><code>research/token_routing/ds64_fast8k/launch_fast8k.py</code> (Beaker, -tde runs); router session: <code>research/token_routing/online_router/launch_otr.py</code> (branch <code>claude/online-router-1bc01a</code>)</td></tr>
    <tr><td>Router code</td><td><code>src/olmo_core/nn/online_token_router.py</code>, flags in <code>recipes/train/research_trainer/train_ctc_suite.py</code> (<code>--st-otr-*</code>, <code>--ffn-width-*</code>, <code>--lora-rank</code>)</td></tr>
    <tr><td>Records</td><td><code>records/online-token-router.md</code> (router branch), <code>records/olmo3-cptdrop-results.md</code>, <code>records/grad-composition.md</code>, <code>records/mt11-64k-online-router.md</code>, <code>records/flopdrop-final-figures-plan.md</code></td></tr>
    <tr><td>Status pages</td><td><code>ops/experiments/{tokdrop_lr,learned_router,mt11_64k,olmo3_cptdrop,grad_composition}/STATUS.md</code></td></tr>
    <tr><td>Other pages (also listed above)</td><td><a href="https://claude.ai/artifact/RFpSfNnqnz8M9ygi8kK1U5">cross-task tables</a> · <a href="https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1">online router</a> · <a href="https://claude.ai/artifact/4tZKEXtpSpg5ax6B1YeoLk">CPT</a> · <a href="https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n">64k study</a> · <a href="https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ">gradient composition</a></td></tr>
  </table></div>
</section>
<section>
  <h2>Detail pages</h2>
  <div class="tbl"><table>
    <tr><th>page</th><th>covers</th></tr>
    <tr><td><a href="https://claude.ai/artifact/RFpSfNnqnz8M9ygi8kK1U5">Token Drop Pareto</a></td><td>cross-task tables, matched-FLOP Δ per cell, older screens</td></tr>
    <tr><td><a href="https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n">mt11 64k Router Study</a></td><td>64k per-rung Δ tables, per-rung Pareto small multiples, seed replicate</td></tr>
    <tr><td><a href="https://claude.ai/artifact/4tZKEXtpSpg5ax6B1YeoLk">OLMo 3 CPT Drop Screen</a></td><td>every CPT run, context gain, memory probe, setup</td></tr>
    <tr><td><a href="https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ">Gradient Composition Probe</a></td><td>interactive per-block, per-token gradient maps (SFT and CPT)</td></tr>
    <tr><td><a href="https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1">Gradprobe Follow-ups</a></td><td>online-router design, fidelity probes, every router ablation, max-length probes</td></tr>
    <tr><td><a href="https://claude.ai/artifact/KsLw2QqRugmkkXCvXknbBg">Router Keep Masks</a></td><td>which tokens the router and the fixed rules keep, on real rows</td></tr>
  </table></div>
</section>
<section>
  <h2>Archive: earlier compute-saving work (Aug–Sep)</h2>
  <ul class="arch">
    <li><a href="https://claude.ai/artifact/DeDhCkJLGyMug1cNW5znGs">Sparsity vs Scale</a> · <a href="https://claude.ai/artifact/CkNbfD44b11RuvLApp6v78">Dev-Loss Scheme Grid</a> · <a href="https://claude.ai/artifact/GkXtjWSW11dckFCgR9vs7K">Soft-Token Scheme Grid</a> · <a href="https://claude.ai/artifact/8t9hNP8sKcx14V6NDB8DnD">Slot Construction Tracker</a></li>
    <li><a href="https://claude.ai/artifact/U4torzWNyskquCsY73dkrd">Soft-KV vs Dense Pareto</a> · <a href="https://claude.ai/artifact/RPtvNxRvkqdJu2NXTKsi67">Soft KV, Explained</a></li>
    <li><a href="https://claude.ai/artifact/LAYgcDNAXvE3CdbRDGkUiz">Does Routed FFN Compute Buy Speed?</a> · <a href="https://claude.ai/artifact/UWrbA9bAJfBwMsDPHdZzFC">Where the Router Speedup Runs Out</a> · <a href="https://claude.ai/artifact/MdQ8EhxunnQjW7sFFEypp1">Routed-FFN Ledger</a> · <a href="https://claude.ai/artifact/BSJsa9gpQh1APELNv6GzHC">Routed Compute Axes</a> · <a href="https://claude.ai/artifact/M8qDKSxiBV3Jzktwku45Ra">Routed Compute Briefing</a></li>
    <li><a href="https://claude.ai/artifact/UzrX3T7976coNmpFQvmfLb">Dense vs Sparse per Training Hour</a> · <a href="https://claude.ai/artifact/HMPTs56ofPHYNn9uXhrrCq">Qwen3.5 Cost Anatomy</a> · <a href="https://claude.ai/artifact/Mx6VH85wUEoL5u5umJcf6E">Scaling-Law Fits: Measured vs Extrapolated</a></li>
  </ul>
</section>
</div>
</main>
</div>
<script>
(() => { const links = [...document.querySelectorAll("nav.toc a")];
  const set = id => links.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + id));
  try { const io = new IntersectionObserver(es => { es.filter(e => e.isIntersecting).forEach(e => set(e.target.id)); },
    { rootMargin: "-20% 0px -70% 0px" });
    document.querySelectorAll(".part").forEach(p => io.observe(p)); } catch (e) {}
  if (location.hash) set(location.hash.slice(1)); else set("summary"); })();
</script>
<script>const D_LOG = __LOG__; const D_COMPUTE = __COMPUTE__;</script>
<script>
(() => { const L = (typeof D_LOG !== "undefined") ? D_LOG : []; const list = document.getElementById("logList"); if (!list) return;
  const order = ["running", "proposed", "held", "partial", "done", "dropped", "failed"];
  const st = new Set(), tr = new Set(); let q = "";
  const statuses = order.filter(x => L.some(e => e.status === x)), tracks = [...new Set(L.map(e => e.track))];
  const esc = t => String(t).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  function chips(host, vals, sel) { const h = document.getElementById(host);
    vals.forEach(v => { const b = document.createElement("button"); b.type = "button"; b.textContent = v + " (" + L.filter(e => e.status === v || e.track === v).length + ")";
      b.setAttribute("aria-pressed", "false"); b.onclick = () => { sel.has(v) ? sel.delete(v) : sel.add(v); b.setAttribute("aria-pressed", sel.has(v)); draw(); }; h.appendChild(b); }); }
  const C = (typeof D_COMPUTE !== "undefined" && D_COMPUTE) ? D_COMPUTE : { jobs: [] };
  const byEntry = {}; C.jobs.forEach(j => { const k = j.entry || "(unlinked)"; (byEntry[k] = byEntry[k] || []).push(j); });
  const gp = j => j.gpus == null ? "CPU" : j.gpus + " GPU";
  function summ(js) { const run = js.filter(j => j.state === "RUNNING"), q = js.filter(j => j.state !== "RUNNING");
    const where = [...new Set(js.map(j => j.cluster))].join(", ");
    return `${where}: ${run.length} running (${run.reduce((a, j) => a + (j.gpus || 0), 0)} GPUs), ${q.length} queued`; }
  (() => { const host = document.getElementById("computeNow"); if (!host) return;
    const run = C.jobs.filter(j => j.state === "RUNNING"), q = C.jobs.filter(j => j.state !== "RUNNING");
    const when = C.generated_at ? new Date(C.generated_at).toLocaleString() : "unknown";
    host.innerHTML = `<div class="cnow"><h3>Compute now <span class="note">(snapshot ${esc(when)}, refreshed at every hub update)</span></h3>
      <p class="lr"><b>${run.length} jobs running on ${run.reduce((a, j) => a + (j.gpus || 0), 0)} GPUs, ${q.length} queued</b> (${q.reduce((a, j) => a + (j.gpus || 0), 0)} GPUs requested).</p>
      <div class="tbl"><table><tr><th>job</th><th>log entry</th><th>where</th><th class="n">GPUs</th><th>state</th><th class="n">min</th></tr>
      ${C.jobs.map(j => `<tr><td class="mono">${esc(j.name)}</td><td>${esc(j.entry || "–")}</td><td>${esc(j.cluster)}${j.node ? " · " + esc(j.node) : ""}</td><td class="n">${gp(j)}</td><td>${j.state === "RUNNING" ? "running" : "queued"}</td><td class="n">${j.elapsed_min}</td></tr>`).join("")}
      </table></div></div>`; })();
  function draw() { const rows = L.filter(e => (!st.size || st.has(e.status)) && (!tr.size || tr.has(e.track)) &&
      (!q || (e.title + " " + e.result + " " + e.who + " " + e.track).toLowerCase().includes(q)));
    list.innerHTML = rows.map(e => `<li><span class="ld">${esc(e.date)}</span><div><div class="lt">${esc(e.title)}</div>
      <div class="lm"><span class="st ${esc(e.status)}">${esc(e.status)}</span><span>${esc(e.track)}</span><span>${esc(e.who)}</span>${byEntry[e.title] ? `<span class="cp">${esc(summ(byEntry[e.title]))}</span>` : ""}</div>
      <p class="lr">${esc(e.result)}</p></div></li>`).join("") || '<li><span></span><p class="lr">No entries match.</p></li>';
    document.getElementById("logCount").textContent = rows.length + " of " + L.length + " entries"; }
  chips("logStatus", statuses, st); chips("logTrack", tracks, tr);
  document.getElementById("logQ").addEventListener("input", e => { q = e.target.value.trim().toLowerCase(); draw(); }); draw(); })();
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = ["--dense","--c1","--c2","--c3","--c4","--c5","--c6","--c7"];
function plot(id, sets, ylab, opts = {}) {
  const el = document.getElementById(id); if (!el || !window.Chart) return;
  const fg = css("--fg"), mu = css("--muted"), ru = css("--rule");
  new Chart(el, {
    type: "line",
    data: { datasets: sets.map((s, i) => ({ label: s.label, data: s.pts, showLine: s.pts.length > 1, borderColor: css(s.col || COL[i % COL.length]),
      backgroundColor: css(s.col || COL[i % COL.length]), borderWidth: s.dense ? 2.5 : 1.6, borderDash: s.dense ? [] : (s.dash || []),
      pointRadius: s.dense ? 3 : 4, pointStyle: s.dense ? "circle" : (s.shape || "rectRot"), tension: 0 })) },
    options: { responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { position: "bottom", labels: { color: fg, boxWidth: 10, font: { size: 10 } } },
        tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(opts.dp || 3)} at ${Math.round(c.parsed.x)} PF` + (c.raw.note ? ` (${c.raw.note})` : "") } } },
      scales: { x: { type: opts.xlin ? "linear" : "logarithmic", title: { display: true, text: opts.xlab || "training PF (log)", color: mu },
                     ticks: { color: mu, callback: v => [30,100,300,1000,3000,10000,30000].includes(v) || opts.xlin ? v : "" }, grid: { color: ru } },
                y: { title: { display: true, text: ylab, color: mu }, ticks: { color: mu }, grid: { color: ru }, reverse: !!opts.rev } } } });
}
const pts = (a, f = "y") => a.map(p => ({ x: p.x, y: p[f], note: (p.rows ? p.rows + " rows/task" : "") + (p.steps ? ", " + p.steps + " steps" : "") + (p.n > 1 ? ", " + p.n + " seeds" : "") }));
function draw() {
  plot("f4fresh", D.f4fresh.map(s => ({ label: s.label, pts: pts(s.p), dense: s.dense })), "8-cell mean score");
  plot("f4out", D.f4fresh.map(s => ({ label: s.label, pts: pts(s.p, "o"), dense: s.dense })), "outlier@8k f1");
  plot("f4orig", D.f4orig.map(s => ({ label: s.label, pts: pts(s.p), dense: s.dense })), "8-cell mean score");
  plot("f11", D.f11.map(s => ({ label: s.label, pts: pts(s.p), dense: s.dense })), "mean over 9 tasks @8k");
  const mx = D.maxt; new Chart(document.getElementById("maxt"), { type: "bar",
    data: { labels: mx.map(m => m[0]), datasets: [{ data: mx.map(m => m[1]), backgroundColor: mx.map((m, i) => css(i ? COL[i] : "--dense")) }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "max train length (thousand tokens)", color: css("--muted") }, ticks: { color: css("--muted") }, grid: { color: css("--rule") } },
                y: { ticks: { color: css("--fg"), font: { size: 10 } }, grid: { display: false } } } } });
  const memSets = (M, F, xmax) => {
    const out = [];
    Object.keys(M).forEach((k, i) => {
      const col = COL[i];
      out.push({ label: k + " (measured)", col, dense: i === 0, pts: M[k].map(p => ({ x: p[0], y: p[1] })) });
      const [b, m] = F[k], cross = (79.25 - b) / m;
      const xs = [16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384].filter(x => x < cross).concat([cross]);
      out.push({ label: k + ` (fit; reaches 80 GB at ≈${Math.round(cross)}k)`, col, dash: [5, 4], shape: "line",
                 pts: xs.map(x => ({ x: +x.toFixed(1), y: +(b + m * x).toFixed(1) })) });
    });
    out.push({ label: "80 GB GPU (usable)", col: "--c6", dash: [2, 3], pts: [{ x: 16, y: 79.25 }, { x: xmax, y: 79.25 }] });
    return out;
  };
  plot("memstd", memSets(D.memstd, D.memstdfit, 290), "peak GiB per GPU", { xlin: true, xlab: "tokens per row (thousands)", dp: 1 });
  plot("memflags", memSets(D.memflags, D.memflagsfit, 420), "peak GiB per GPU", { xlin: true, xlab: "tokens per row (thousands)", dp: 1 });
  [["memstd", 290], ["memflags", 420]].forEach(([id, xm]) => { const ch = Chart.getChart(id); if (ch) {
    ch.options.scales.y.min = 0; ch.options.scales.y.max = 85; ch.options.scales.x.min = 0; ch.options.scales.x.max = xm;
    ch.data.datasets.forEach(ds => { if (ds.label.includes("fit")) { ds.pointRadius = 0; } }); ch.update(); } });
  plot("cpt32", Object.entries(D.cpt32).map(([k, v], i) => ({ label: k, pts: v.map(p => ({ x: p[0], y: p[1] })), dense: i === 0 })), "dev CE (nats)", { dp: 4 });
  plot("cpt64", cptLenSets(), "LoRA − dense dev CE (nats)", { xlab: "training steps (log)", dp: 4 });
  if (D.lr2) Object.entries(D.lr2).forEach(([id, pan]) => {
    const lab = { A: "dense", B: "dense + LoRA-all", C: "routing (two-stage)", D: "routing + LoRA-all" }, col = { A: "--dense", B: "--c3", C: "--c1", D: "--c2" };
    plot(id, ["A", "B", "C", "D"].filter(c => pan[c] && pan[c].length).map(c => ({ label: lab[c], dense: c === "A", col: col[c], shape: c === "B" ? "triangle" : "rectRot",
      pts: pan[c].map(p => ({ x: p[0], y: p[1], note: (p[2] === "cut" ? "cut run" : p[2] + " rows/task") + (p[3] > 1 ? ", " + p[3] + " seeds" : "") })) })),
      id === "lr2e" ? "mean of 9 tasks @8k" : "8-cell mean"); });
  (() => { const el = document.getElementById("smsel"); if (!el || !window.Chart) return;
    const labs = ["0.8B forward", "0.8B grad scoring", "0.8B first 6 blocks", "≈135M", "2B forward"];
    new Chart(el, { type: "bar", data: { labels: labs, datasets: [
      { label: "keep 20%→7% from block 0", data: [1.35, 1.06, 1.70, 1.71, 0.97], backgroundColor: css("--c2") },
      { label: "keep 7% from block 0", data: [1.88, 1.37, 2.64, 2.67, 1.22], backgroundColor: css("--c1") }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { position: "bottom", labels: { color: css("--fg"), boxWidth: 10, font: { size: 10 } } } },
        scales: { x: { ticks: { color: css("--muted"), font: { size: 10 } }, grid: { display: false } },
                  y: { min: 0, title: { display: true, text: "× cheaper than the recipe (best case)", color: css("--muted") }, ticks: { color: css("--muted") }, grid: { color: css("--rule") } } } } }); })();
  (() => { if (!window.Chart) return; const R = D.skipbench || [];
    [["skb32", 32768], ["skb8", 8192]].forEach(([id, T]) => { const el = document.getElementById(id); if (!el) return;
      const rows = R.filter(r => r.T === T);
      const labs = rows.map(r => (r.block.startsWith("gdn") ? "GDN" : "attention") + ", keep " + Math.round(r.keep * 100) + "%");
      const ser = [["ideal_ms", "ideal (no gather)", "--dense"], ["tok_ms", "per-token skip", "--c1"], ["chunk100_ms", "100-token chunks", "--c2"], ["chunk128_ms", "128-token chunks", "--c3"]];
      new Chart(el, { type: "bar", data: { labels: labs, datasets: ser.map(([k, l, c]) => ({ label: l, data: rows.map(r => +(r[k] / r.full_ms).toFixed(3)), backgroundColor: css(c) })).concat([
          { type: "line", label: "keep fraction (linear-FLOPs ideal)", data: rows.map(r => r.keep), borderColor: css("--muted"), borderDash: [4, 4], pointRadius: 0, fill: false }]) },
        options: { responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { position: "bottom", labels: { color: css("--fg"), boxWidth: 10, font: { size: 10 } } },
            tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(2)}× full-layer time` } } },
          scales: { x: { ticks: { color: css("--muted"), font: { size: 10 } }, grid: { display: false } },
                    y: { min: 0, title: { display: true, text: "time / full-layer time", color: css("--muted") }, ticks: { color: css("--muted") }, grid: { color: css("--rule") } } } } }); }); })();
  (() => { if (!window.Chart) return; const Q = D.q3 || {}; const el = document.getElementById("q3par");
    if (el && Q.dense_curve) { const cells = Object.values(Q.cells || {}); const arms = {};
      cells.filter(c => c.arm !== "dense").forEach(c => { (arms[c.arm] = arms[c.arm] || []).push(c); });
      const lab = a => a === "q3r20at12s24" ? "routing only 12/24 (no LoRA)" : "recipe " + a.replace("q3lora64all36r20at", "").replace("s", "/") + (a.includes("s2") ? "" : "/18");
      plot("q3par", [{ label: "dense", dense: true, pts: Q.dense_curve.map(p => ({ x: p[0], y: p[1] })) }].concat(
        Object.entries(arms).map(([a, cs], i) => ({ label: lab(a), col: COL[(i % 7) + 1], pts: cs.sort((x, y) => x.pf - y.pf).map(c => ({ x: c.pf, y: c.y, note: c.rows + " rows/task" })) }))), "8-cell mean"); }
    const e2 = document.getElementById("q3ftm"); if (e2) new Chart(e2, { type: "bar", data: { labels: ["1,952 rows", "3,904 rows", "7,808 rows"], datasets: [
        { label: "Qwen3.5-4B (recipe 7/16)", data: [4.63, 3.82, 2.18], backgroundColor: css("--c2") },
        { label: "Qwen3-4B (best split; 3,904 / 7,808 are lower bounds)", data: [2.09, 2.23, 1.12], backgroundColor: css("--c1") }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { position: "bottom", labels: { color: css("--fg"), boxWidth: 10, font: { size: 10 } } } },
        scales: { x: { ticks: { color: css("--muted") }, grid: { display: false } }, y: { min: 0, title: { display: true, text: "dense FLOPs to match (×)", color: css("--muted") }, ticks: { color: css("--muted") }, grid: { color: css("--rule") } } } } });
    const e3 = document.getElementById("kvcos"); if (e3) new Chart(e3, { type: "bar", data: { labels: ["1st @ 1", "1st @ 3", "1st @ 7", "2st @ 1", "2st @ 3", "2st @ 7"], datasets: [
        { label: "plain drop", data: [0.000, 0.040, 0.244, -0.003, 0.116, 0.453], backgroundColor: css("--c6") },
        { label: "KV-only context", data: [0.028, 0.108, 0.314, 0.042, 0.148, 0.471], backgroundColor: css("--c2") },
        { label: "perfect picker + KV-only", data: [0.060, 0.140, 0.466, 0.154, 0.188, 0.580], backgroundColor: css("--c3") }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { position: "bottom", labels: { color: css("--fg"), boxWidth: 10, font: { size: 10 } } } },
        scales: { x: { ticks: { color: css("--muted") }, grid: { display: false } }, y: { title: { display: true, text: "8-batch gradient cosine", color: css("--muted") }, ticks: { color: css("--muted") }, grid: { color: css("--rule") } } } } }); })();
  (() => { if (!window.Chart) return; const S = D.scale || { points: [] };
    const e = document.getElementById("scftm"); if (e) new Chart(e, { type: "bar", data: { labels: ["0.8B", "2B", "4B"], datasets: [
        { label: "1,952 rows/task", data: [1.73, 2.48, 4.08], backgroundColor: css("--c2") },
        { label: "3,904 rows/task", data: [1.67, 2.52, 3.68], backgroundColor: css("--c3") },
        { label: "7,808 rows/task (0.8B: lower bound)", data: [4.16, null, 2.17], backgroundColor: css("--c1") }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { position: "bottom", labels: { color: css("--fg"), boxWidth: 10, font: { size: 10 } } } },
        scales: { x: { ticks: { color: css("--muted") }, grid: { display: false } }, y: { min: 0, title: { display: true, text: "dense FLOPs to match (×)", color: css("--muted") }, ticks: { color: css("--muted") }, grid: { color: css("--rule") } } } } });
    [["sc08", "0.8b"], ["sc2", "2b"], ["sc4", "4b"]].forEach(([id, sz]) => {
      const pts = S.points.filter(p => p.scale === sz);
      const dense = pts.filter(p => p.arm === "dense" && p.lr === 5e-6).sort((a, b) => a.pf - b.pf);
      const rec = pts.filter(p => p.arm !== "dense" && p.lr === 1e-5).sort((a, b) => a.pf - b.pf);
      const nt = p => p.rows + " rows/task" + (p.n_seeds > 1 ? ", " + p.n_seeds + " seeds" : "");
      plot(id, [{ label: "dense (LR 5e-6)", dense: true, pts: dense.map(p => ({ x: p.pf, y: p.y, note: nt(p) })) },
                { label: "recipe (LR 1e-5)", col: "--c1", pts: rec.map(p => ({ x: p.pf, y: p.y, note: nt(p) })) }], "8-cell mean"); }); })();
  plot("cptm", [{ label: "whole-model LoRA, frozen output layer", pts: D.cptmfh.map(p => ({ x: p[0], y: p[1] })), col: "--c3" },
                { label: "whole-model LoRA, trained output layer", pts: D.cptm.map(p => ({ x: p[0], y: p[1] })), col: "--c1", dash: [5, 4] }],
       "dense FLOPs to match (×)", { xlab: "training steps (log)", dp: 2 });
}
function cptLenSets() {
  const fh = Object.entries(D.cptlenfh).map(([k, v], i) => ({ label: k + ", frozen output layer (recipe)", col: COL[i + 1], pts: v.map(p => ({ x: p[0], y: p[1] })) }));
  const th = Object.entries(D.cptlen).map(([k, v], i) => ({ label: k + ", trained output layer", col: COL[i + 1], dash: [5, 4], pts: v.map(p => ({ x: p[0], y: p[1] })) }));
  return fh.concat(th);
}
function multiples(hostId, sets, tasks, prefix) {
  const host = document.getElementById(hostId); if (!host) return;
  tasks.forEach((t, j) => {
    const id = `${prefix}${j}`;
    host.insertAdjacentHTML("beforeend", `<figure><h3>${t}${prefix === "pt4o" ? " (original pool)" : ""}</h3><div class="chart"><canvas id="${id}" aria-label="${t} at 8k"></canvas></div></figure>`);
    plot(id, sets.map(s => ({ label: s.label, dense: s.dense, pts: s.p.map(p => ({ x: p.x, y: p.t[t], note: p.rows + " rows/task" })) })), `${t}@8k`);
    const ch = Chart.getChart(id); if (ch) { ch.options.plugins.legend.display = j === 0; ch.update(); }
  });
}
function drawGC() {
  if (!D.gc) return;
  const C = D.gc.ckpts, cols = ["--c6", "--c5", "--c4", "--c2", "--c1"];
  const blk = (f, n) => C.map((c, i) => ({ label: c.name, col: cols[i], pts: c[f].slice(0, n).map((v, b) => ({ x: b, y: v })) }));
  plot("gcpg", blk("pgrad_share", 32), "share of ‖g‖²", { xlin: true, xlab: "block", dp: 3 });
  plot("gcdoc", blk("doc_share_g2", 33), "document share of ‖dL/dh‖²", { xlin: true, xlab: "block (32 = final output)", dp: 2 });
  plot("gcx7", blk("spearman_b7", 33).map(s => ({ ...s, pts: s.pts.filter(p => p.x >= 7) })), "Spearman with block 7", { xlin: true, xlab: "block", dp: 2 });
  plot("gcearly", Object.entries(D.early).map(([k, v], i) => ({ label: k, col: COL[i + 1], pts: v.map(p => ({ x: p[0] + 1, y: p[1], note: p[0] ? p[0] + " steps" : "base" })) })),
       "blocks 0–6 share", { xlab: "training steps + 1 (log; 1 = base)", dp: 2 });
  plot("gcctx", Object.entries(D.cptctx).map(([k, v], i) => ({ label: k, col: i ? "--c1" : "--c6", pts: v.map(p => ({ x: p[0], y: p[1] })) })),
       "context share of gradient", { xlin: true, xlab: "block", dp: 2 });
}
try { draw(); drawGC(); multiples("pt4f", D.f4fresh, ["contradiction","nq","outlier","oolong"], "pt4f");
      multiples("pt4o", D.f4orig, ["contradiction","nq","outlier","oolong"], "pt4o");
      if (D.m64) {
        const M = [["Dense, LR 5e-6", "dense|5e-06", true], ["Dense, LR 2e-6", "dense|2e-06", false], ["Router keep 10%, LR 5e-6", "otr10M7|5e-06", false], ["Router keep 10%, LR 2e-6", "otr10M7|2e-06", false], ["Recipe: adapters + two-stage + floor", "lora64r20to7f16|5e-06", false]];
        const sets = M.filter(m => D.m64[m[1]]).map(m => ({ label: m[0], dense: m[2], p: D.m64[m[1]] }));
        plot("m64avg", sets.map((s, i) => ({ label: s.label, dense: s.dense, dash: i % 2 ? [5, 4] : [], pts: s.p.map(p => ({ x: p.x, y: p.y, note: p.budget + (p.n > 1 ? ", " + p.n + " seeds" : "") })) })), "task average (2k–64k)");
        plot("m64r64", sets.map((s, i) => ({ label: s.label, dense: s.dense, dash: i % 2 ? [5, 4] : [], pts: s.p.filter(p => p.rung["64k"] != null).map(p => ({ x: p.x, y: p.rung["64k"], note: p.budget })) })), "mean @64k");
        const host = document.getElementById("pt64");
        ["xabsence","contradiction","outlier","strmatch","grouping","hotpotqa","nq","qdmatch_nq","oolong"].forEach((t, j) => {
          host.insertAdjacentHTML("beforeend", `<figure><h3>${t} (64k study, mean over rungs)</h3><div class="chart"><canvas id="pt64${j}" aria-label="${t} 64k"></canvas></div></figure>`);
          plot(`pt64${j}`, sets.map((s, i) => ({ label: s.label, dense: s.dense, dash: i % 2 ? [5, 4] : [], pts: s.p.map(p => ({ x: p.x, y: p.t[t], note: p.budget })) })), t);
          const ch = Chart.getChart(`pt64${j}`); if (ch) { ch.options.plugins.legend.display = j === 0; ch.update(); }
        });
      }
      if (D.bylen) {
        const R = ["2k","8k","16k","32k","64k"], X = { "2k": 2, "8k": 8, "16k": 16, "32k": 32, "64k": 64 };
        const B = Object.keys(D.bylen).sort();
        const lab = k => { const [a, b] = k.split("|"); return a + ", " + ({ f4: "1/4 epoch", f2: "1/2 epoch", f1: "1 epoch" }[b] || b); };
        plot("lenD", B.map((b, i) => ({ label: lab(b), col: COL[i + 1], dash: b.startsWith("router") ? [5, 4] : [], pts: R.filter(r => D.bylen[b][r]).map(r => ({ x: X[r], y: D.bylen[b][r].delta })) })),
             "Δ score vs dense at matched FLOPs", { xlab: "eval length (thousand tokens, log)" });
        plot("lenF", B.map((b, i) => ({ label: lab(b), col: COL[i + 1], dash: b.startsWith("router") ? [5, 4] : [], pts: R.filter(r => D.bylen[b][r] && D.bylen[b][r].ftm).map(r => ({ x: X[r], y: Math.abs(D.bylen[b][r].ftm), note: D.bylen[b][r].ftm < 0 ? "lower bound" : "" })) })),
             "dense FLOPs to match (×)", { xlab: "eval length (thousand tokens, log)", dp: 2 });
        ["lenD","lenF"].forEach(id => { const ch = Chart.getChart(id); if (ch) { ch.options.scales.x.ticks.callback = v => [2,4,8,16,32,64].includes(v) ? v + "k" : ""; ch.update(); } });
      }
      plot("lenC", cptLenSets(),
           "LoRA − dense dev CE (nats)", { xlab: "training steps (log)", dp: 4 });
      multiples("pt11", D.f11, ["xabsence","contradiction","outlier","strmatch","grouping","hotpotqa","nq","qdmatch_nq","oolong"], "pt11"); }
catch (e) { console.error(e); }
</script>
"""


M64_TASKS = ["xabsence", "contradiction", "outlier", "strmatch", "grouping", "hotpotqa", "nq", "qdmatch_nq", "oolong"]
M64_RUNGS = ["2k", "8k", "16k", "32k", "64k"]


def m64_data(path):
    """64k study (mt11L64, rows to 65,536 tokens): per run PF, task-average over all (task, rung) cells except reorder,
    per-rung means and per-task means over rungs; seed copies averaged."""
    try:
        j = json.load(open(path))
    except Exception:
        return None
    pf = {t["run"]: t["pf"] for t in j["train"]}
    cells = collections.defaultdict(dict)
    for s in j["scores"]:
        if s["eval_size"] >= 500 and s["task"] in M64_TASKS and s.get("final", True):
            cells[s["train_run"]][(s["task"], s["rung"])] = s["value"]
    groups = collections.defaultdict(list)
    for run, c in cells.items():
        m = re.match(r"m64-(.+?)-(f\d+)-lr([\d.e-]+?)(?:-sd\d+)?$", run)
        if not m or run not in pf or len(c) < 30:
            continue
        groups[(m.group(1), m.group(3))].append((pf[run], c, m.group(2)))
    out = {}
    for (arm, lr), v in groups.items():
        by_b = collections.defaultdict(list)
        for x, c, b in v:
            by_b[b].append((x, c))
        pts = []
        for b, w in by_b.items():
            keys = set.intersection(*[set(c) for _, c in w])
            avg = {k: sum(c[k] for _, c in w) / len(w) for k in keys}
            rung = {r: round(sum(vv for (t, rr), vv in avg.items() if rr == r) / max(1, sum(1 for (t, rr) in avg if rr == r)), 4)
                    for r in M64_RUNGS if any(rr == r for _, rr in avg)}
            task = {t: round(sum(vv for (tt, _), vv in avg.items() if tt == t) / max(1, sum(1 for (tt, _) in avg if tt == t)), 4)
                    for t in M64_TASKS if any(tt == t for tt, _ in avg)}
            six = ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction"]  # graded at every length 2k-64k
            rung6 = {r: round(sum(avg[(t, r)] for t in six) / 6, 4) for r in M64_RUNGS if all((t, r) in avg for t in six)}
            pts.append({"x": round(sum(x for x, _ in w) / len(w), 1), "y": round(sum(avg.values()) / len(avg), 4),
                        "n": len(w), "budget": b, "rung": rung, "rung6": rung6, "t": task})
        out[f"{arm}|{lr}"] = sorted(pts, key=lambda p: p["x"])
    return out


def by_length(m64):
    """Router vs dense (LR 5e-6) per eval rung: matched-FLOP delta and dense-FLOPs-to-match, per router budget."""
    if not m64 or "dense|5e-06" not in m64:
        return None
    import math
    dense = m64["dense|5e-06"]
    out = {}
    arms = [("router", "otr10M7|5e-06"), ("recipe", "lora64r20to7f16|5e-06")]
    for name, key in arms:
      for rp in m64.get(key, []):
          row = {}
          for r in M64_RUNGS:
              dc = [(p["x"], p["rung6"][r]) for p in dense if r in p["rung6"]]
              if r not in rp["rung6"] or len(dc) < 2:
                  continue
              y = rp["rung6"][r]
              # matched-FLOP delta: dense interpolated in log PF at the router's PF
              lx = math.log(rp["x"])
              seg = next(((a, b) for a, b in zip(dc, dc[1:]) if math.log(a[0]) <= lx <= math.log(b[0])), None)
              if seg is None:
                  continue
              (x0, y0), (x1, y1) = seg
              dmatch = y0 + (y1 - y0) * (lx - math.log(x0)) / (math.log(x1) - math.log(x0))
              # FLOPs to match: first dense segment whose score brackets y
              ftm = None
              for (x0, y0), (x1, y1) in zip(dc, dc[1:]):
                  if min(y0, y1) <= y <= max(y0, y1) and y1 != y0:
                      ftm = x0 * (x1 / x0) ** ((y - y0) / (y1 - y0)) / rp["x"]
                      break
              if ftm is None and y > max(v for _, v in dc):
                  ftm = -max(x for x, _ in dc) / rp["x"]  # negative = lower bound (above dense's best point)
              row[r] = {"delta": round(y - dmatch, 4), "ftm": None if ftm is None else round(ftm, 2)}
          out[f'{name}|{rp["budget"]}'] = row
    return out


def ftm_rows(S):
    dense = pick(S, "mt4", "f", "dense")
    if len(dense) < 2:
        return ""
    dpts = sorted((d["x"], d["y"]) for d in dense)
    x1, y1 = dpts[-1]

    def need_pf(y):
        """dense PF at score y: piecewise log-linear along the fresh dense curve (first crossing)"""
        for (xa, ya), (xb, yb) in zip(dpts, dpts[1:]):
            if min(ya, yb) <= y <= max(ya, yb) and yb != ya:
                return xa * (xb / xa) ** ((y - ya) / (yb - ya))
        (xa, ya), (xb, yb) = dpts[0], dpts[1]  # below the curve: extrapolate the first segment
        return xa * (xb / xa) ** ((y - ya) / (yb - ya))
    names = {"lora64r20to7": "Recipe: adapters 0–6 + two-stage 20%→7%", "lora64fa75r3s": "Adapters (frozen at 75%) + three-stage",
             "lora64fa75r20to7": "Adapters (frozen at 75%) + two-stage", "lora64fa75r20to10": "Adapters (frozen at 75%) + two-stage 20%→10%",
             "r20to7M7de0": "Two-stage 20%→7%, no adapters", "otr10M7de0": "Router keep 10%", "fa50M7": "Router + freeze blocks 0–6 at 50%",
             "lora64M7": "Router + adapters 0–6", "lora64r20to10": "Adapters 0–6 + two-stage 20%→10%"}
    out = [f'<tr><td>Dense</td><td class="n">{p["rows"]}</td><td class="n">{p["x"]:.0f}</td><td class="n">{p["y"]:.3f}</td>'
           f'<td class="n">{p["o"]:.3f}</td><td class="n">1.0×</td></tr>' for p in dense]
    for arm, label in names.items():
        for p in pick(S, "mt4", "f", arm):
            txt = f'&gt;{x1 / p["x"]:.1f}×' if p["y"] > max(y for _, y in dpts) else f'{need_pf(p["y"]) / p["x"]:.1f}×'
            out.append(f'<tr><td>{label}</td><td class="n">{p["rows"]}</td><td class="n">{p["x"]:.0f}</td><td class="n">{p["y"]:.3f}</td>'
                       f'<td class="n">{p["o"]:.3f}</td><td class="n"><b>{txt}</b></td></tr>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--updated", default="")
    ap.add_argument("--gradcomp", default=__file__.rsplit("/", 1)[0] + "/gradcomp_summary.json")
    ap.add_argument("--log", default=__file__.rsplit("/", 1)[0] + "/experiment_log.json", help="experiment log entries")
    ap.add_argument("--lora2x2", default=__file__.rsplit("/", 1)[0] + "/lora_routing_2x2.json",
                    help="LoRA x routing 2x2 (lora_routing_2x2.py); its section is skipped if the file is missing")
    ap.add_argument("--m64", default="/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/mt11-64k/research/token_routing/mt11_64k/results/points.json")
    a = ap.parse_args()
    S = series(json.load(open(a.points)))
    f4fresh = [("Dense", "dense", True), ("Recipe: adapters 0–6 + two-stage", "lora64r20to7", False),
               ("Adapters frozen 75% + three-stage", "lora64fa75r3s", False), ("Adapters frozen 75% + two-stage", "lora64fa75r20to7", False),
               ("Two-stage, no adapters", "r20to7M7de0", False), ("Router keep 10%", "otr10M7de0", False),
               ("Router + freeze 0–6 at 50%", "fa50M7", False)]
    dense_s = pick(S, "mt4", "s", "dense")
    f4orig = [{"label": "Dense", "dense": True, "p": dense_s},
              {"label": "Recipe: adapters 0–6 + two-stage", "dense": False, "p": pick(S, "mt4", "s", "lora64r20to7")},
              {"label": "Whole-model LoRA + two-stage (LR 1e-5)", "dense": False,
               "p": pick(S, "mt4", "s", "lora64K32r20to7", "1e-05") + pick(S, "mt4", "s", "lora64all32r20to7", "1e-05")},
              {"label": "Two-stage, no adapters", "dense": False, "p": pick(S, "mt4", "s", "r20to7M7de0")},
              {"label": "Router keep 10%, no full steps", "dense": False, "p": pick(S, "mt4", "s", "otr10M7de0")},
              {"label": "Router keep 10%, full step every 8", "dense": False, "p": pick(S, "mt4", "s", "otr10M7")}]
    for s in f4orig[2:3]:
        s["p"] = sorted(s["p"], key=lambda p: p["x"])
    f11 = [{"label": "Dense", "dense": True, "p": pick(S, "mt11", "s", "dense")},
           {"label": "Recipe + 16-token floor", "dense": False, "p": pick(S, "mt11", "s", "lora64r20to7f16")},
           {"label": "Whole-model LoRA + two-stage + floor (LR 1e-5)", "dense": False, "p": pick(S, "mt11", "s", "lora64K32r20to7f16", "1e-05")},
           {"label": "Router + freeze at 50% + floor", "dense": False, "p": pick(S, "mt11", "s", "fa247f16M7")},
           {"label": "Two-stage + floor", "dense": False, "p": pick(S, "mt11", "s", "r20to7f16")},
           {"label": "Router keep 10% + floor", "dense": False, "p": pick(S, "mt11", "s", "otr10M7de0f16")}]
    data = {"f4fresh": [{"label": l, "dense": dn, "p": pick(S, "mt4", "f", arm)} for l, arm, dn in f4fresh],
            "f4orig": f4orig, "f11": f11, "m64": m64_data(a.m64),
            "cptlen": {"32k": [(48, -0.0031), (120, -0.0034), (240, -0.0023), (480, -0.0012)], "64k": [(120, -0.0050), (240, -0.0030), (480, -0.0017)],
                       "128k": [(48, -0.0035), (120, -0.0041)]},
            "cptlenfh": {"32k": [(48, -0.0037), (120, -0.0041), (240, -0.0032), (480, -0.0022)], "64k": [(120, -0.0057), (240, -0.0039), (480, -0.0026)],
                         "128k": [(48, -0.0038), (120, -0.0046)]}, "cpt32": CPT32, "cpt64": CPT64, "cptm": CPT_M, "cptmfh": CPT_M_FH, "maxt": MAXT, "memstd": MEM_STD, "memflags": MEM_FLAGS, "memstdfit": MEM_STD_FIT, "memflagsfit": MEM_FLAGS_FIT}
    data["bylen"] = by_length(data["m64"])
    try:
        data["gc"] = json.load(open(a.gradcomp))
    except Exception:
        data["gc"] = None
    data["early"] = EARLY_SHARE
    data["cptctx"] = CPT_CTX
    try:
        data["scale"] = json.load(open(__file__.rsplit("/", 1)[0] + "/scale_ablation.json"))
    except FileNotFoundError:
        data["scale"] = {"points": []}
    try:
        data["q3"] = json.load(open(__file__.rsplit("/", 1)[0] + "/q3_flops_to_match.json"))["q3"]
    except FileNotFoundError:
        data["q3"] = {}
    try:
        data["skipbench"] = json.load(open(__file__.rsplit("/", 2)[0] + "/chunk_skip_bench/results_h200.json"))
    except FileNotFoundError:
        data["skipbench"] = []
    try:
        from make_lora_routing_section import section as lora2x2_section
        lr2 = json.load(open(a.lora2x2))
        lora2x2 = lora2x2_section(lr2)
        cut = {"mt4": [(102, 0.666), (197, 0.743)], "mt11": [(46, 0.500), (92, 0.616)]}  # dense runs cut at 31/61 (32/64) steps
        def arm(mix, pool, c):
            return sorted((t["cells"][c]["pf"], t["cells"][c]["y"], t["rows"], t["cells"][c]["n"]) for t in lr2["tables"]
                          if t["mix"] == mix and t["pool"] == pool and c in t["cells"])
        data["lr2"] = {}
        for key, mix, pool in (("lr2f", "mt4", "f"), ("lr2o", "mt4", "s"), ("lr2e", "mt11", "s")):
            pan = {c: arm(mix, pool, c) for c in "ABCD"}
            if key != "lr2f":
                pan["A"] = sorted([(x, y, "cut", 1) for x, y in cut[mix]] + pan["A"])
            data["lr2"][key] = pan
    except FileNotFoundError:
        lora2x2 = ""
        data["lr2"] = {}
    log = sorted(json.load(open(a.log)), key=lambda e: e["date"], reverse=True)
    try:
        compute = json.load(open(__file__.rsplit("/", 1)[0] + "/compute_now.json"))
    except FileNotFoundError:
        compute = {"generated_at": None, "jobs": []}
    html = PAGE.replace("__LOG__", json.dumps(log).replace("</", "<\\/")).replace("__COMPUTE__", json.dumps(compute).replace("</", "<\\/")).replace("__LORA2X2__", lora2x2).replace("__DATA__", json.dumps(data)).replace("__FTM__", ftm_rows(S)).replace("__UPDATED__", a.updated or "today")
    open(a.out, "w").write(html)
    for k in ("f4fresh", "f4orig", "f11"):
        for s in data[k]:
            print(k, s["label"], [(p["x"], p["y"], p["n"]) for p in s["p"]])


if __name__ == "__main__":
    main()
