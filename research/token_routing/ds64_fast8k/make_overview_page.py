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
CPT_M = [(12, 1.65), (24, 2.75), (48, 3.9), (120, 3.45), (240, 2.5), (480, 1.8), (960, 1.12)]
# ---- peak memory per GPU vs row length, SFT 4B, 2 x A100-80GB, FSDP-2, full AC, steady routed step (router record,
# "B' round 0/2" memprobe tables). Standard flags: fixed + transient at 32k/64k/128k. All memory flags (routed warmup,
# --skip-dry-run, chunked FFN 8192, GDN segments 16384): measured peaks. ----
MEM_STD = {"Dense": [(32, 43.7), (64, 56.0)], "Router keep 10%": [(32, 42.1), (64, 50.3), (128, 66.7)],
           "Recipe: two-stage + adapters 0–6": [(32, 36.4), (64, 44.4), (128, 60.2)],
           "Whole-model LoRA + two-stage": [(32, 23.3), (64, 29.3), (128, 45.2)]}
MEM_FLAGS = {"Dense": [(128, 68.3), (160, 77.5)], "Router keep 10%": [(192, 64.3), (224, 69.9)],
             "Recipe: two-stage + adapters 0–6": [(262, 71.6)],
             "Whole-model LoRA + two-stage": [(262, 56.5), (393, 77.4)]}
# linear fits (intercept GiB, slope GiB per 1k tokens) from the router record's probe tables: standard = fixed + its
# transient fit; all flags = the record's per-1k slopes through the measured points (recipe slope from round 1)
MEM_STD_FIT = {"Dense": (31.3, 0.386), "Router keep 10%": (34.0, 0.255), "Recipe: two-stage + adapters 0–6": (28.5, 0.248),
               "Whole-model LoRA + two-stage": (13.4, 0.248)}
MEM_FLAGS_FIT = {"Dense": (32.7, 0.28), "Router keep 10%": (31.8, 0.17), "Recipe: two-stage + adapters 0–6": (16.6, 0.21),
                 "Whole-model LoRA + two-stage": (14.6, 0.16)}
# ---- max train length at 80 GB, SFT 4B, 2 x A100, FSDP-2, full AC (records/online-token-router.md; linear
# extrapolation from 3-step probes, one sample each) ----
MAXT = [("Dense", 165), ("Router, no full-context steps", 250), ("Recipe (adapters 0–6 + two-stage)", 300),
        ("Whole-model LoRA + two-stage", 400)]

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
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=JetBrains+Mono:wght@400;500&family=Public+Sans:wght@400;600&display=swap">
<div class="wrap">
<header>
  <span class="eyebrow">Train-time token / layer drop · 2026-10-11 to 10-25 · updated __UPDATED__</span>
  <h1>Cheaper training that matches dense at matched FLOPs</h1>
  <p class="sub">Qwen3.5-4B SFT on CTC long-context tasks (4-task and 11-task mixes, 64k) and OLMo 3 7B long-document continued pretraining. Every plot puts training FLOPs (measured by the FLOP meter) on the x-axis; up-and-left of the dense line is better (down-and-left for CPT loss). Private page.</p>
</header>

<section class="lead">
  <h2>Where things stand</h2>
  <ul>
    <li><b>SFT, 4-task mix:</b> dense needs about <b>3×</b> the training FLOPs to match the cheap methods on fresh data (3,904 rows/task), and <b>≈2.2×</b> or more at 7,808 rows/task where dense nears its ceiling. On the original pool, dense needs ≈3.5–3.9× to match the recipe and ≈4.4× for whole-model LoRA + two-stage keep.</li>
    <li><b>SFT, 11-task mix:</b> ≈1.7× for the recipe; whole-model LoRA + two-stage scores 0.762 against dense's 0.697 at the same 232 PF (2 seeds), ≈2.4×. Gains concentrate on xabsence, contradiction and outlier; four retrieval tasks are near dense's ceiling; grouping is the one task the router loses.</li>
    <li><b>SFT, 11-task mix at 64k with 10× the data:</b> the router (keep 10% after block 7, LR 5e-6) is ≈+0.03 task-average over dense at matched FLOPs at every budget from 1/4 to 1 epoch; dense needs ≈1.6–1.7× the FLOPs to match. On the same data it trails dense by ≈0.03. <b>The recipe at 64k (1/4 epoch, one seed): +0.038 task-average at 0.365× dense FLOPs, ahead on all 9 scored tasks, 24% less peak active memory</b> (37.2 vs 48.7 GiB); its gains grow with length (contradiction +0.041 at 8k → +0.119 at 64k).</li>
    <li><b>The recipe:</b> rank-64 LoRA adapters on blocks 0–6 (frozen base), an online gradient-norm router that keeps 20% of tokens from block 7 and rescores to 7% at block 16, no full-context steps, LR 5e-6, plus a 16-token floor per document on the 11-task mix. Whole-model LoRA (all 32 blocks, LR 1e-5) is cheaper still at the same quality.</li>
    <li><b>CPT:</b> every token-drop variant loses. Whole-model LoRA with no dropping beats dense at 0.70× metered FLOPs per step (≈0.75× counting checkpoint recompute): ≈2–4× FLOPs-to-match between 24 and 240 steps, ≈1.8× at 480, and parity by 960 steps (dev CE +0.0012 behind dense; ≈1.04× counting recompute). The adapters saturate around 0.25–0.5B tokens while full CPT keeps learning long-range structure, so this is a short-budget recipe. The edge is larger at 64k.</li>
    <li><b>Max train length at 80 GB (SFT):</b> ≈1.8× dense for the recipe and ≈2.4× for whole-model LoRA (extrapolated from short probes).</li>
  </ul>
</section>

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
  <p class="note">"Dense FLOPs to match" = training PF dense needs to reach the same 8-cell mean on its fresh curve (log-linear between 1,952 and 3,904 rows/task) divided by the method's PF; "&gt;" when the method is above dense's best point (a lower bound). Dense at 5,856 and 7,808 rows/task is training to extend the curve.</p>
</section>

<section>
  <h2>Per task: 4-task mix @8k</h2>
  <p class="note">Same runs and colours as above, one panel per task (8k rung, eval_size 500 each, no-cot). Fresh data on the top row, original pool below.</p>
  <div class="figs small" id="pt4f"></div>
  <div class="figs small" id="pt4o" style="margin-top:12px"></div>
</section>

<section>
  <h2>SFT: 11-task mix</h2>
  <div class="figs">
    <figure><h3>Mean over 9 tasks @8k vs training FLOPs</h3><div class="chart"><canvas id="f11" aria-label="11-task Pareto"></canvas></div>
      <figcaption>720 rows/task unless noted; the recipe also at 240/480/960. Mean over nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction, xabsence, strmatch, grouping at 8k (reorder and rerank excluded: no working grader). Dense below 240 rows/task = 240-row runs cut at 32 and 64 steps.</figcaption></figure>
    <figure><h3>Peak memory vs row length (standard settings)</h3><div class="chart"><canvas id="memstd" aria-label="memory vs length standard"></canvas></div>
      <figcaption>Peak GPU memory per GPU at the steady (routed) training step vs tokens per row. Qwen3.5-4B, 2× A100-80GB, FSDP-2, full checkpointing. The slope is memory per token: dense 0.39 GiB per 1k tokens, routed arms ≈0.25. Dense out of memory at 128k. Markers = measured; dashed = linear fit extended to 512k (crossing the 80 GB line gives the estimated max length); dotted = 79.25 GiB usable.</figcaption></figure>
    <figure><h3>Peak memory vs row length (all memory optimizations)</h3><div class="chart"><canvas id="memflags" aria-label="memory vs length optimized"></canvas></div>
      <figcaption>Same probe with routed warmup, no dry run, chunked FFN recompute and segmented GDN. Slopes: dense 0.28, router 0.17, whole-model LoRA 0.16 GiB per 1k tokens. Markers = measured, dashed = linear fit extended to 512k (the recipe has one point at this setting, so its line uses the slope from the chunked-FFN round). Two-stage without adapters and a 20% router are being measured.</figcaption></figure>
    <figure><h3>Max train length at 80 GB</h3><div class="chart"><canvas id="maxt" aria-label="max train length"></canvas></div>
      <figcaption>Qwen3.5-4B SFT, 2× A100-80GB, FSDP-2, full activation checkpointing, chunked FFN + segmented GDN. Linear extrapolation from 3-step probes, one sample each (router session record).</figcaption></figure>
  </div>
</section>

<section>
  <h2>SFT: 11-task mix at 64k (10× data)</h2>
  <div class="figs">
    <figure><h3>Task average (2k–64k) vs training FLOPs</h3><div class="chart"><canvas id="m64avg" aria-label="64k task average"></canvas></div>
      <figcaption>mt11L64 data (rows up to 65,536 tokens, 662M tokens), 8× H100, 128 rows/step. Average over the 9 graded tasks and all their rungs (2k–64k; xabsence/strmatch/grouping to 32k), eval_size 500 each, no-cot. Points = 1/16, 1/8, 1/4, 1/2, 1 epoch; the router at 1/4 is 2 seeds.</figcaption></figure>
    <figure><h3>64k rung only</h3><div class="chart"><canvas id="m64r64" aria-label="64k rung"></canvas></div>
      <figcaption>Mean over the 6 tasks with a 64k rung (nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction).</figcaption></figure>
  </div>
  <p class="note">The recipe (adapters + two-stage keep + floor, 1/4 epoch, one seed) is +0.038 over dense at matched FLOPs, ahead on all 9 scored tasks (grouping 16k pending a re-run after an NCCL timeout), with 37.2 vs 48.7 GiB peak active memory on 4 H100s.</p>
  <div class="figs small" id="pt64" style="margin-top:12px"></div>
</section>

<section>
  <h2>Does the advantage grow with context length?</h2>
  <div class="figs">
    <figure><h3>Router gain over dense at matched FLOPs, by eval length</h3><div class="chart"><canvas id="lenD" aria-label="gain by eval length"></canvas></div>
      <figcaption>64k study (training rows up to 64k). Each line = one router budget (1/4, 1/2, 1 epoch); y = router score minus dense (LR 5e-6) interpolated to the router's FLOPs, averaged over the same 6 tasks at every length (nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction), so lengths compare like with like.</figcaption></figure>
    <figure><h3>Dense FLOPs needed to match the router, by eval length</h3><div class="chart"><canvas id="lenF" aria-label="FLOPs to match by eval length"></canvas></div>
      <figcaption>Same runs: training PF dense needs to reach the router's score at that length, divided by the router's PF. Above 1 = the router is FLOP-cheaper.</figcaption></figure>
    <figure><h3>CPT: whole-model LoRA vs dense, training at 32k vs 64k</h3><div class="chart"><canvas id="lenC" aria-label="CPT gap by training length"></canvas></div>
      <figcaption>Dev CE of whole-model LoRA minus dense at the same step count (more negative = better), training and evaluating at 32k vs 64k. Paired SE ≈ ±0.0001–0.0002.</figcaption></figure>
  </div>
  <p class="note" id="lenNote"><b>Reading:</b> at 2k the router barely beats dense (Δ ≈ 0.001–0.004; these tasks are near their ceiling at 2k). The gain rises to +0.015–0.033 at 8k and +0.03–0.05 at 16k–32k, then stays at +0.02–0.04 at 64k; dense FLOPs to match go from ≈1.1–1.3× at 2k to ≈1.4–1.9× at 8k–32k and ≈1.3–1.6× at 64k. So the advantage grows with eval length up to ~16–32k and holds at 64k, rather than growing without bound (1/2- and 1-epoch points are one seed). In CPT, training at 64k widens LoRA's lead over dense by ≈1.5× at every budget compared with training at 32k.</p>
</section>

<section>
  <h2>Per task: 11-task mix @8k</h2>
  <p class="note">Each panel = one task at 8k, same runs and colours as the 11-task plot. Reorder and rerank are not shown (no working grader yet; fixes in progress).</p>
  <div class="figs small" id="pt11"></div>
</section>

<section>
  <h2>Long-document CPT (OLMo 3 7B)</h2>
  <div class="figs">
    <figure><h3>32k: dev CE vs training FLOPs (lower is better)</h3><div class="chart"><canvas id="cpt32" aria-label="CPT 32k"></canvas></div>
      <figcaption>dev32k_1024 (1,024 rows of 32k), full-context CE; paired SE ≈ ±0.0001, seed-to-seed ≈ ±0.0001. Dense curve 5–960 steps.</figcaption></figure>
    <figure><h3>64k: dev CE vs training FLOPs</h3><div class="chart"><canvas id="cpt64" aria-label="CPT 64k"></canvas></div>
      <figcaption>dev64k_512 (512 rows). Dense 64k PF estimated as LoRA PF ÷ 0.696 (the dense-64k FLOP file is a partial count). 120 / 240 / 480 steps.</figcaption></figure>
    <figure><h3>Whole-model LoRA: dense FLOPs to match, by budget (32k)</h3><div class="chart"><canvas id="cptm" aria-label="CPT FLOPs to match"></canvas></div>
      <figcaption>The edge peaks near 48 steps and decays as dense keeps learning, reaching parity by 960 steps (dev CE +0.0012 behind dense there; far-repeat CE +0.0035). At 64k the gap is ≈1.5× larger at every budget.</figcaption></figure>
  </div>
</section>

<section>
  <h2>What failed (and the number that decided it)</h2>
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
    <tr><td>Any token dropping (router, input drop, MoD, keep 50%)</td><td>CPT</td><td class="neg">router FLOPs-to-match &lt;0.41 → 0.06; keep 50% + LoRA worse than dense after 4 steps</td></tr>
    <tr><td>Early-exit loss on dropped tokens</td><td>CPT</td><td class="neg">dev CE rises with training (1.737 → 1.771)</td></tr>
    <tr><td>All-local attention, 4k-only rows</td><td>CPT</td><td class="neg">context gain collapses (0.075 → 0.044 / 0.0005)</td></tr>
    <tr><td>Stopping backprop below block B</td><td>CPT</td><td>FLOPs-to-match 2.9 / 1.4 at B = 4 / 8, below whole-model LoRA's 3.45</td></tr>
  </table></div>
</section>

<section>
  <h2>Why it works where it works</h2>
  <ul>
    <li><b>Blocks 0–6 must see every token;</b> the router decides at block 7 from features that predict each token's gradient norm (held-out Spearman 0.54–0.63). The gradient-composition study shows rankings drift with depth (Spearman between blocks 7 and 24 falls 0.57 → 0.34), which is why rescoring at block 16 helps. ⚠ probes on 32–64 rows/task.</li>
    <li><b>In SFT only answer tokens carry loss</b>, and the gradient concentrates on few document tokens (top 10% hold ≈0.9 of late-layer gradient energy).</li>
    <li><b>In CPT every token is a label</b> and importance is flat (top 10% hold 0.62); early blocks learn mostly as context for later tokens (0.95 of block-0 gradient). So dropping tokens removes most of the learning signal, while freezing weights behind adapters does not. ⚠ 48 rows × 32k.</li>
  </ul>
</section>

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
    <tr><td>Other pages</td><td><a href="https://claude.ai/artifact/RFpSfNnqnz8M9ygi8kK1U5">cross-task tables</a> · <a href="https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1">online router</a> · <a href="https://claude.ai/artifact/4tZKEXtpSpg5ax6B1YeoLk">CPT</a> · <a href="https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n">64k study</a> · <a href="https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ">gradient composition</a></td></tr>
  </table></div>
</section>
</div>
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
  const memSets = (M, F) => {
    const out = [];
    Object.keys(M).forEach((k, i) => {
      const col = COL[i];
      out.push({ label: k + " (measured)", col, dense: i === 0, pts: M[k].map(p => ({ x: p[0], y: p[1] })) });
      const [b, m] = F[k], cross = (79.25 - b) / m;
      out.push({ label: k + ` (linear fit; ≈${Math.round(cross)}k at 80 GB)`, col, dash: [5, 4], shape: "line",
                 pts: [16, 64, 128, 256, 384, 512].map(x => ({ x, y: +(b + m * x).toFixed(1) })) });
    });
    out.push({ label: "80 GB GPU (usable)", col: "--c6", dash: [2, 3], pts: [{ x: 16, y: 79.25 }, { x: 512, y: 79.25 }] });
    return out;
  };
  plot("memstd", memSets(D.memstd, D.memstdfit), "peak GiB per GPU", { xlin: true, xlab: "tokens per row (thousands)", dp: 1 });
  plot("memflags", memSets(D.memflags, D.memflagsfit), "peak GiB per GPU", { xlin: true, xlab: "tokens per row (thousands)", dp: 1 });
  ["memstd", "memflags"].forEach(id => { const ch = Chart.getChart(id); if (ch) { ch.options.scales.y.max = 160; ch.options.scales.y.min = 0;
    ch.data.datasets.forEach(ds => { if (ds.label.includes("fit")) { ds.pointRadius = 0; } }); ch.update(); } });
  plot("cpt32", Object.entries(D.cpt32).map(([k, v], i) => ({ label: k, pts: v.map(p => ({ x: p[0], y: p[1] })), dense: i === 0 })), "dev CE (nats)", { dp: 4 });
  plot("cpt64", Object.entries(D.cpt64).map(([k, v], i) => ({ label: k, pts: v.map(p => ({ x: p[0], y: p[1] })), dense: i === 0 })), "dev CE (nats)", { dp: 4 });
  plot("cptm", [{ label: "whole-model LoRA, 32k", pts: D.cptm.map(p => ({ x: p[0], y: p[1] })), col: "--c1" }], "dense FLOPs to match (×)", { xlab: "training steps (log)", dp: 2 });
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
try { draw(); multiples("pt4f", D.f4fresh, ["contradiction","nq","outlier","oolong"], "pt4f");
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
      plot("lenC", Object.entries(D.cptlen).map(([k, v], i) => ({ label: "train + eval at " + k, col: COL[i + 1], pts: v.map(p => ({ x: p[0], y: p[1] })) })),
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
    (x0, y0), (x1, y1) = (dense[0]["x"], dense[0]["y"]), (dense[-1]["x"], dense[-1]["y"])
    names = {"lora64r20to7": "Recipe: adapters 0–6 + two-stage 20%→7%", "lora64fa75r3s": "Adapters (frozen at 75%) + three-stage",
             "lora64fa75r20to7": "Adapters (frozen at 75%) + two-stage", "lora64fa75r20to10": "Adapters (frozen at 75%) + two-stage 20%→10%",
             "r20to7M7de0": "Two-stage 20%→7%, no adapters", "otr10M7de0": "Router keep 10%", "fa50M7": "Router + freeze blocks 0–6 at 50%",
             "lora64M7": "Router + adapters 0–6", "lora64r20to10": "Adapters 0–6 + two-stage 20%→10%"}
    out = [f'<tr><td>Dense</td><td class="n">{p["rows"]}</td><td class="n">{p["x"]:.0f}</td><td class="n">{p["y"]:.3f}</td>'
           f'<td class="n">{p["o"]:.3f}</td><td class="n">1.0×</td></tr>' for p in dense]
    for arm, label in names.items():
        for p in pick(S, "mt4", "f", arm):
            need = x0 * (x1 / x0) ** ((p["y"] - y0) / (y1 - y0))
            txt = f'&gt;{x1 / p["x"]:.1f}×' if p["y"] > y1 else f'{need / p["x"]:.1f}×'
            out.append(f'<tr><td>{label}</td><td class="n">{p["rows"]}</td><td class="n">{p["x"]:.0f}</td><td class="n">{p["y"]:.3f}</td>'
                       f'<td class="n">{p["o"]:.3f}</td><td class="n"><b>{txt}</b></td></tr>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--updated", default="")
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
            "cptlen": {"32k": [(120, -0.0034), (240, -0.0023), (480, -0.0012)], "64k": [(120, -0.0049), (240, -0.0033), (480, -0.0020)]}, "cpt32": CPT32, "cpt64": CPT64, "cptm": CPT_M, "maxt": MAXT, "memstd": MEM_STD, "memflags": MEM_FLAGS, "memstdfit": MEM_STD_FIT, "memflagsfit": MEM_FLAGS_FIT}
    data["bylen"] = by_length(data["m64"])
    html = PAGE.replace("__DATA__", json.dumps(data)).replace("__FTM__", ftm_rows(S)).replace("__UPDATED__", a.updated or "today")
    open(a.out, "w").write(html)
    for k in ("f4fresh", "f4orig", "f11"):
        for s in data[k]:
            print(k, s["label"], [(p["x"], p["y"], p["n"]) for p in s["p"]])


if __name__ == "__main__":
    main()
