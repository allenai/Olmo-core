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
        agg[(r["mix"], pool, arm, f"{r['lr']:.0e}", r["rows"], r.get("max_steps") or 0)].append(
            (r["pf"], y, c[("outlier", "8k")]))
    out = collections.defaultdict(list)
    for (mix, pool, arm, lr, rows, steps), v in agg.items():
        n = len(v)
        out[(mix, pool, arm, lr)].append((round(sum(a[0] for a in v) / n, 1), round(sum(a[1] for a in v) / n, 4),
                                          round(sum(a[2] for a in v) / n, 4), n, rows, steps))
    return {k: sorted(v) for k, v in out.items()}


def pick(S, mix, pool, arm, lr="5e-06", exclude_partial=False):
    pts = S.get((mix, pool, arm, lr), [])
    if exclude_partial:
        pts = [p for p in pts if not p[5]]
    return [{"x": p[0], "y": p[1], "o": p[2], "n": p[3], "rows": p[4], "steps": p[5]} for p in pts]


# ---- CPT (OLMo 3 7B, dev CE, lower is better). Source: records/olmo3-cptdrop-results.md + CPT session reports ----
CPT32 = {
    "Dense": [(137, 1.7096), (247, 1.7008), (685, 1.6912), (1344, 1.6880), (3318, 1.6853), (6606, 1.6827),
              (13188, 1.6800), (26349, 1.6762)],
    "Whole-model LoRA (rank 64, all 32 blocks)": [(941, 1.6849), (2323, 1.6819), (4624, 1.6804), (9232, 1.6788)],
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
CPT_M = [(12, 1.65), (24, 2.75), (48, 3.9), (120, 3.45), (240, 2.5), (480, 1.8)]
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
    <li><b>The recipe:</b> rank-64 LoRA adapters on blocks 0–6 (frozen base), an online gradient-norm router that keeps 20% of tokens from block 7 and rescores to 7% at block 16, no full-context steps, LR 5e-6, plus a 16-token floor per document on the 11-task mix. Whole-model LoRA (all 32 blocks, LR 1e-5) is cheaper still at the same quality.</li>
    <li><b>CPT:</b> every token-drop variant loses. Whole-model LoRA with no dropping beats dense at 0.70× metered FLOPs per step (≈0.75× counting checkpoint recompute): ≈2–4× FLOPs-to-match between 24 and 240 steps, ≈1.8× at 480, and a larger edge at 64k.</li>
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
  <h2>SFT: 11-task mix</h2>
  <div class="figs">
    <figure><h3>Mean over 9 tasks @8k vs training FLOPs</h3><div class="chart"><canvas id="f11" aria-label="11-task Pareto"></canvas></div>
      <figcaption>720 rows/task unless noted; the recipe also at 240/480/960. Mean over nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction, xabsence, strmatch, grouping at 8k (reorder and rerank excluded: no working grader). Dense below 240 rows/task = 240-row runs cut at 32 and 64 steps.</figcaption></figure>
    <figure><h3>Max train length at 80 GB</h3><div class="chart"><canvas id="maxt" aria-label="max train length"></canvas></div>
      <figcaption>Qwen3.5-4B SFT, 2× A100-80GB, FSDP-2, full activation checkpointing, chunked FFN + segmented GDN. Linear extrapolation from 3-step probes, one sample each (router session record).</figcaption></figure>
  </div>
</section>

<section>
  <h2>Long-document CPT (OLMo 3 7B)</h2>
  <div class="figs">
    <figure><h3>32k: dev CE vs training FLOPs (lower is better)</h3><div class="chart"><canvas id="cpt32" aria-label="CPT 32k"></canvas></div>
      <figcaption>dev32k_1024 (1,024 rows of 32k), full-context CE; paired SE ≈ ±0.0001, seed-to-seed ≈ ±0.0001. Dense curve 5–960 steps.</figcaption></figure>
    <figure><h3>64k: dev CE vs training FLOPs</h3><div class="chart"><canvas id="cpt64" aria-label="CPT 64k"></canvas></div>
      <figcaption>dev64k_512 (512 rows). Dense 64k PF estimated as LoRA PF ÷ 0.696 (the dense-64k FLOP file is a partial count). 120 / 240 / 480 steps.</figcaption></figure>
    <figure><h3>Whole-model LoRA: dense FLOPs to match, by budget (32k)</h3><div class="chart"><canvas id="cptm" aria-label="CPT FLOPs to match"></canvas></div>
      <figcaption>The edge peaks near 48 steps and decays as dense keeps learning; the 480-step value is approximate. At 64k the gap is ≈1.5× larger at every budget.</figcaption></figure>
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
  plot("cpt32", Object.entries(D.cpt32).map(([k, v], i) => ({ label: k, pts: v.map(p => ({ x: p[0], y: p[1] })), dense: i === 0 })), "dev CE (nats)", { dp: 4 });
  plot("cpt64", Object.entries(D.cpt64).map(([k, v], i) => ({ label: k, pts: v.map(p => ({ x: p[0], y: p[1] })), dense: i === 0 })), "dev CE (nats)", { dp: 4 });
  plot("cptm", [{ label: "whole-model LoRA, 32k", pts: D.cptm.map(p => ({ x: p[0], y: p[1] })), col: "--c1" }], "dense FLOPs to match (×)", { xlab: "training steps (log)", dp: 2 });
}
try { draw(); } catch (e) { console.error(e); }
</script>
"""


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
            "f4orig": f4orig, "f11": f11, "cpt32": CPT32, "cpt64": CPT64, "cptm": CPT_M, "maxt": MAXT}
    html = PAGE.replace("__DATA__", json.dumps(data)).replace("__FTM__", ftm_rows(S)).replace("__UPDATED__", a.updated or "today")
    open(a.out, "w").write(html)
    for k in ("f4fresh", "f4orig", "f11"):
        for s in data[k]:
            print(k, s["label"], [(p["x"], p["y"], p["n"]) for p in s["p"]])


if __name__ == "__main__":
    main()
