"""Add the "8c · GPU audit" tab to hub.html (run after build_nov_tabs.py; idempotent).

Data: gpu_audit_data.json beside this file, copied verbatim from research/common/gpu_audit/data/ on branch
claude/gpu-workflow-audit-8833c4 (summary.json, daily_gpuh_pool.csv, waits.csv). Text figures are from
records/gpu-usage-workflow-audit-2026-11.md and records/f8k-local-slurm-eval.md on that branch.
Usage: python3 build_gpu_tab.py hub.html
"""
import json, os, sys

H = sys.argv[1]
h = open(H).read()
if 'id="gpuaudit"' in h:
    sys.exit("already built")
DATA = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_audit_data.json")))

TAB = '''<div class="part" id="gpuaudit">
<div class="parthead"><span class="eyebrow">8c · GPU audit</span><h2 class="ptitle">Where the GPU-hours went, 10-20 to 11-10, and what changed</h2><p class="verdict">6,705 GPU-h in three weeks (Beaker 74%). The largest loss is preempted Beaker training restarting from step 0: <b>1,083 GPU-h, 16%</b> of the total (309 attempts, 304 of them at minRuntime 1 h). Beaker jobs spent <b>60%</b> of their lifetime pending, while jsteinhardt ran at 23% of its 8-GPU cap. On 11-10 Prasann adopted run-sized minRuntimes and spilling small short jobs to jsteinhardt, retired the "always unallocated" rule, and deferred jupiter+ceres as a default.</p></div>
<section>
  <h2>Headline numbers</h2>
  <div class="tbl"><table>
    <tr><th>what</th><th class="n">value</th><th>detail</th></tr>
    <tr><td>GPU-h, all pools</td><td class="n">6,705</td><td>Beaker 4,954 (jupiter 2,971, jupiter+ceres 1,864, titan 119) · jsteinhardt 1,190 (preemptive_high 945, preemptive 245) · Lambda 560 · berkeleynlp 1</td></tr>
    <tr><td>by kind</td><td class="n"></td><td>train 4,847 · eval 1,657 · smoke 77 · probe/bench 52 · data 37</td></tr>
    <tr><td>Beaker training preempted, restarted from step 0</td><td class="n">1,083 (16%)</td><td>309 attempts; median attempt ran 1.26 h, 77% preempted before 2 h. Preemption is by allocation balance (group <code>ai2/oe-scaling</code> ~55% over target); the trainer wrote no mid-run checkpoint unless <code>--save-interval</code> was passed</td></tr>
    <tr><td>Beaker pending share of job lifetime</td><td class="n">60%</td><td>Lambda 57%, jsteinhardt 17%. Median / p90 wait: jupiter 0.39 / 2.35 h, jupiter+ceres 0.06 / 0.60 h, jsteinhardt 0.00 / 0.14 h, Lambda 0.10 / 0.78 h</td></tr>
    <tr><td>jsteinhardt use of the 8-GPU preemptive_high cap</td><td class="n">23%</td><td>mean 1.8 GPUs; 375 of 520 h with none of our jobs; 691 GPU-h of headroom unused while ≥4 of our GPUs were pending on Beaker</td></tr>
    <tr><td>Lambda, other users</td><td class="n">20.5 / 24 GPUs</td><td>about 2.4 GPUs free on average at every hour; 29 of their jobs ended within 3 min of one of our preemptive starts</td></tr>
    <tr><td>Beaker allocated share</td><td class="n">98%</td><td>4,871 of 4,991 GPU-h (Beaker's own report), because <code>ai2/flex2</code> now belongs to <code>ai2/oe-scaling</code></td></tr>
    <tr><td>estimates vs actual runtime</td><td class="n">0.21×</td><td>median actual / expected minutes (p90 0.95): estimates run about 5× long</td></tr>
  </table></div>
  <p class="note">Infra audit, no eval metrics. Every number is from the audit record or its data files (see "Where things live").</p>
</section>
<section>
  <h2>GPU-hours over time and what was lost</h2>
  <div class="figs">
    <figure style="grid-column: 1 / -1"><h3>GPU-hours per day by pool (PT days)</h3><div class="chart" style="height:300px"><canvas id="gaDaily" aria-label="GPU-hours per day by pool"></canvas></div>
      <figcaption>From the audit's <code>daily_gpuh_pool.csv</code>. Its day axis lists 11-01 twice and stops at 11-09 (a DST fall-back artifact in <code>analyze.py</code>); 11-01 is shown once here, so the days sum to 6,686 GPU-h vs the 6,705 total (the ~19 GPU-h of 11-10 are missing).</figcaption></figure>
    <figure><h3>GPU-hours lost or unused, by cause</h3><div class="chart" style="height:340px"><canvas id="gaWaste" aria-label="lost GPU-hours by cause"></canvas></div>
      <figcaption>The record's waste table. Grey = capacity left unused (not held by us), not a loss of our hours. Categories can overlap (smokes include failed smokes).</figcaption></figure>
    <figure><h3>Beaker: preempted GPU-hours per day vs all Beaker GPU-hours</h3><div class="chart" style="height:340px"><canvas id="gaPre" aria-label="preempted GPU-hours per day"></canvas></div>
      <figcaption>Preempted = every kind (405 attempts, 1,159 GPU-h; training 1,083), dated by the attempt's end; Beaker total dated by when the hours ran. 10-30 alone lost 311 GPU-h.</figcaption></figure>
  </div>
</section>
<section>
  <h2>Queue wait</h2>
  <div class="figs">
    <figure><h3>Beaker wait by job size: jupiter-only vs jupiter+ceres</h3><div class="chart" style="height:300px"><canvas id="gaSize" aria-label="wait by job size"></canvas></div>
      <figcaption>Median and p90 wait (hours) per first attempt, from <code>waits.csv</code>. 2-GPU p90: 2.97 h jupiter-only vs 0.81 h with ceres allowed. 3,759 of 5,563 attempts were pinned to jupiter.</figcaption></figure>
    <figure><h3>Beaker wait by minRuntime</h3><div class="chart" style="height:300px"><canvas id="gaMinrt" aria-label="wait by minRuntime"></canvas></div>
      <figcaption>A longer minRuntime did not queue longer (2–3 h median 0.15–0.28 h vs 0.28 h at 1 h). Bars with few jobs (4–8 h: 5–27 attempts) are noisy; counts in the tooltip.</figcaption></figure>
    <figure style="grid-column: 1 / -1"><h3>Wait-time histogram by pool (attempts)</h3><div class="chart" style="height:280px"><canvas id="gaHist" aria-label="wait histogram"></canvas></div>
      <figcaption>No Beaker job waited more than 7.2 h in the window; the days-long tail of minRuntime-0 jobs is gone (98% of first attempts set one).</figcaption></figure>
  </div>
</section>
<section>
  <h2>Recommendations and what Prasann decided (2026-11-10)</h2>
  <div class="tbl"><table>
    <tr><th>#</th><th>change</th><th>evidence</th><th>decision</th></tr>
    <tr><td>1</td><td>minRuntime = min(4 h, 1.3 × expected) on every Beaker launch; past 4 h, a rolling resume point</td><td>1,083 GPU-h restarted; 92% of completed Beaker training runs take ≤ 2 h, 99% ≤ 4 h; ~300 GPU-h / week</td><td><b>adopted</b> (CLAUDE.md rule; launcher default <code>--min-runtime auto</code>, <code>--resume-interval</code>)</td></tr>
    <tr><td>2</td><td>constrain Beaker jobs to <code>ai2/jupiter</code> + <code>ai2/ceres</code> by default</td><td>2-GPU p90 ~3 h → ~0.8 h; ~650 job-hours of waiting</td><td><b>deferred</b>: "ask me about this again later"; nothing changed (needs a ceres 2-GPU smoke first)</td></tr>
    <tr><td>3</td><td>allocated vs unallocated Beaker usage</td><td>98% of Beaker GPU-h already allocated</td><td><b>allocated is fine</b>: the "always unallocated" rule is retired</td></tr>
    <tr><td>4</td><td>spill small, short jobs (about ≤ 2 GPUs, ≤ 2 h) to jsteinhardt, or wherever they start soonest, when Beaker is backed up</td><td>jsteinhardt at 23% of cap; 691 GPU-h headroom during Beaker backlogs</td><td><b>adopted</b> ("always a good idea"). jsteinhardt: at most 8 GPUs at <code>preemptive_high</code> across all sessions; more at <code>preemptive</code> when capacity is available (restartable work only)</td></tr>
    <tr><td>5</td><td>one eval job per checkpoint looping over its tasks</td><td>2,665 per-task jobs for 862 checkpoints; median eval 6.6 min</td><td>built for the local path (below)</td></tr>
    <tr><td>6</td><td>show other users' GPUs per node; warn before a start would preempt someone</td><td>Lambda 20.5 / 24 GPUs busy; 29 jobs ended at our starts</td><td><b>adopted</b>: <code>ops capacity</code> (+ <code>--would-preempt</code>)</td></tr>
    <tr><td>7–9</td><td><code>ops orphans</code>; refuse <code>--experiment</code> without a STATUS page; expected_minutes from history; no cross-session GPU lock</td><td>597 completed evals with no result row; estimates ~5× long</td><td>not decided in this audit</td></tr>
  </table></div>
</section>
<section>
  <h2>What was built</h2>
  <ul>
    <li><code>ops capacity</code> (with <code>--would-preempt</code>): other users' load per node on jsteinhardt, berkeleynlp and Lambda.</li>
    <li>minRuntime default: <code>opslib/runtime.py</code> computes min(4 h, 1.3 × expected), 1 h fallback, from <code>--expected-minutes</code> or the history median; <code>beaker_ctc_suite.py --min-runtime auto</code> with an automatic resume point past 4 h; <code>train_ctc_suite.py --resume-interval</code> (also fixes <code>--save-interval</code> being a no-op under <code>--no-final-checkpoint</code>); eval launcher <code>BEAKER_MIN_RUNTIME=auto</code>.</li>
    <li>Local slurm fast8k eval: <code>research/token_routing/ds64_fast8k/local_eval/submit_eval_local.py</code>, one jsteinhardt job per checkpoint at <code>--qos=preemptive</code> with per-cell resume. Parity with Beaker on the recipe checkpoint: <b>8/8 cells within 0.002</b> (eval_size 500/cell, SE 0.007–0.021), 8k mean 0.846 local vs 0.845 Beaker, ~30 min per checkpoint on 2 H200s with no queue. It has since scored the weighted-sampling and docdrop runs.</li>
    <li>The spill rule in <code>where-to-run</code> and allocation wording in beaker.md, skills and the ops README.</li>
  </ul>
  <p class="note">The branch <code>claude/gpu-workflow-audit-8833c4</code> (e790ae6b2) is <b>not merged</b> into <code>prasann/flopdrop_clean</code>: the tooling above lives on that branch.</p>
</section>
<section>
  <h2>Limits</h2>
  <ul>
    <li>In-job GPU utilization is unmeasured (no stored samples), so "allocated but idle" is not in these numbers.</li>
    <li>Restart-from-step-0 is inferred from the trainer default plus the incident log, not observed per job.</li>
    <li>Duplicate detection matches run names, so renamed copies are missed; the 72% of registered runs with no final state limit the waste accounting.</li>
    <li>One three-week window; the audit is to be re-run around 2026-12-01 to measure the effect of the changes. Whether a longer minRuntime slows scheduling is unverified beyond the 2–3 h jobs seen here.</li>
  </ul>
</section>
<section>
  <h2>Where things live</h2>
  <div class="tbl"><table>
    <tr><th>what</th><th>where</th></tr>
    <tr><td>audit record</td><td><code>records/gpu-usage-workflow-audit-2026-11.md</code> (branch <code>claude/gpu-workflow-audit-8833c4</code>)</td></tr>
    <tr><td>local eval record</td><td><code>records/f8k-local-slurm-eval.md</code> (same branch)</td></tr>
    <tr><td>scripts, data</td><td><code>research/common/gpu_audit/</code>: <code>analyze.py</code>, <code>fetch_beaker.py</code>, <code>sessions.py</code>, <code>build_page.py</code>; <code>data/summary.json</code>, <code>jobs.csv</code>, <code>waits.csv</code>, <code>daily_*.csv</code>, <code>workflow_incidents.csv</code>, <code>beaker_report/</code></td></tr>
    <tr><td>this tab's data</td><td><code>research/token_routing/ds64_fast8k/hub_live/gpu_audit_data.json</code> (verbatim copy), built by <code>hub_live/build_gpu_tab.py</code></td></tr>
    <tr><td>experiment state</td><td><code>ops/experiments/gpu_audit/STATUS.md</code> (<code>ops exp show gpu_audit</code>)</td></tr>
    <tr><td>standalone audit page</td><td><a href="https://claude.ai/artifact/AxWXjBS1jDQy51MtGszd3d">GPU usage and workflow audit</a> (private)</td></tr>
  </table></div>
  <div data-logfig="GPU usage + workflow audit"></div>
</section>
</div>
'''

JS = r'''<script>
(() => { if (!window.Chart) return; const G = __GADATA__;
  const fg = css("--fg"), mu = css("--muted"), ru = css("--rule");
  const leg = { position: "bottom", labels: { color: fg, boxWidth: 10, font: { size: 10 } } };
  const ax = (lab, stack) => ({ stacked: !!stack, title: { display: !!lab, text: lab || "", color: mu }, ticks: { color: mu }, grid: { color: ru } });
  const cat = stack => ({ stacked: !!stack, ticks: { color: fg, font: { size: 10 }, autoSkip: false }, grid: { display: false } });
  const mk = (id, cfg) => { const el = document.getElementById(id); if (el) new Chart(el, cfg); };
  const base = { responsive: true, maintainAspectRatio: false, animation: false };
  // 1. daily GPU-h by pool, stacked
  const DP = G.daily_pool, PC = { beaker: "--c2", jsteinhardt: "--c3", lambda: "--c5", berkeleynlp: "--c6" };
  mk("gaDaily", { type: "bar", data: { labels: DP.map(r => r[0].slice(5)), datasets: ["beaker", "jsteinhardt", "lambda", "berkeleynlp"].map((k, i) => ({
      label: k, data: DP.map(r => r[i + 1]), backgroundColor: css(PC[k]), borderRadius: 1 })) },
    options: { ...base, plugins: { legend: leg }, scales: { x: cat(true), y: ax("GPU-h per day", true) } } });
  // 2. waste by cause (record table)
  const W = [["Beaker training preempted, restarted", 1083, 1], ["unused jsteinhardt cap during Beaker backlog", 691, 0], ["cancelled while running", 234, 1], ["failed attempts", 233, 1],
    ["evals with no result row after 24 h", 173, 1], ["preempted eval / smoke / probe", 100, 1], ["smokes (all)", 77, 1], ["same run on two pools", 53, 1], ["slurm / Lambda training preempted", 47, 1], ["unregistered jobs", 36, 1]];
  mk("gaWaste", { type: "bar", data: { labels: W.map(w => w[0]), datasets: [{ label: "GPU-h", data: W.map(w => w[1]), backgroundColor: W.map(w => css(w[2] ? "--c1" : "--c6")), borderRadius: 2 }] },
    options: { ...base, indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: ax("GPU-h, 10-20 to 11-10"), y: cat() } } });
  // 3. Beaker preempted per day vs Beaker total per day
  const PD = G.beaker_preempt_gpuh_by_day, days = DP.map(r => r[0]);
  mk("gaPre", { type: "bar", data: { labels: days.map(d => d.slice(5)), datasets: [
      { label: "all Beaker GPU-h", data: DP.map(r => r[1]), backgroundColor: css("--c2") + "55", borderRadius: 1 },
      { label: "preempted Beaker GPU-h", data: days.map(d => PD[d] ?? null), backgroundColor: css("--c1"), borderRadius: 1 }] },
    options: { ...base, plugins: { legend: leg }, scales: { x: cat(), y: ax("GPU-h per day") } } });
  // 4. wait by size, jupiter vs jupiter+ceres
  const WT = G.waits, pick = (pool, g) => WT.find(r => r[0] === pool && r[1] === g);
  const sizes = ["1", "2", "3-4", "5-8"], pools = [["beaker:jupiter", "jupiter only", "--c1"], ["beaker:multi:jupiter+ceres", "jupiter+ceres", "--c3"]];
  const sets = []; pools.forEach(([p, n, c]) => { sets.push({ label: n + " median", data: sizes.map(s => (pick(p, s) || [])[3] ?? null), backgroundColor: css(c), n: sizes.map(s => (pick(p, s) || [])[2]) });
    sets.push({ label: n + " p90", data: sizes.map(s => (pick(p, s) || [])[4] ?? null), backgroundColor: css(c) + "66", n: sizes.map(s => (pick(p, s) || [])[2]) }); });
  const tip = { callbacks: { label: c => `${c.dataset.label}: ${c.raw} h (${c.dataset.n[c.dataIndex]} attempts)` } };
  mk("gaSize", { type: "bar", data: { labels: sizes.map(s => s + " GPU"), datasets: sets }, options: { ...base, plugins: { legend: leg, tooltip: tip }, scales: { x: cat(), y: ax("wait (h)") } } });
  // 5. wait by minRuntime
  const MR = WT.filter(r => r[0].startsWith("beaker:minRuntime="));
  mk("gaMinrt", { type: "bar", data: { labels: MR.map(r => r[0].split("=")[1]), datasets: [
      { label: "median", data: MR.map(r => r[3]), backgroundColor: css("--c2"), n: MR.map(r => r[2]) },
      { label: "p90", data: MR.map(r => r[4]), backgroundColor: css("--c2") + "66", n: MR.map(r => r[2]) }] },
    options: { ...base, plugins: { legend: leg, tooltip: tip }, scales: { x: { ...cat(), title: { display: true, text: "minRuntime", color: mu } }, y: ax("wait (h)") } } });
  // 6. wait histogram
  const WH = G.wait_hist, HC = { beaker: "--c2", jsteinhardt: "--c3", lambda: "--c5" };
  mk("gaHist", { type: "bar", data: { labels: WH.labels, datasets: ["beaker", "jsteinhardt", "lambda"].map(k => ({ label: k, data: WH[k], backgroundColor: css(HC[k]), borderRadius: 1 })) },
    options: { ...base, plugins: { legend: leg }, scales: { x: { ...cat(), title: { display: true, text: "wait before start", color: mu } }, y: ax("attempts") } } });
})();
</script>
'''
JS = JS.replace("__GADATA__", json.dumps(DATA, ensure_ascii=False).replace("</", "<\\/"))


def ins(before, block):
    global h
    assert h.count(before) == 1, (before, h.count(before))
    h = h.replace(before, block + before, 1)


ins('<div class="part" id="provenance">', TAB)
ins('  <a href="#provenance">', '  <a href="#gpuaudit"><span>8c</span>GPU audit</a>\n')
i = h.rindex("</script>") + len("</script>")
h = h[:i] + "\n" + JS + h[i:]
open(H, "w").write(h)
print("built", len(h))
