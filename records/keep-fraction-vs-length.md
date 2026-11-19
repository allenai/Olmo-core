---
title: Does the recipe's best keep fraction fall as context grows? (keep 70%..1.25% at 2k-64k, dense held-out-loss proxy)
date: 2026-11-17
updated: 2026-11-19
experiment: tokdrop_lr
area: token_routing
kind: result
status: current (11-19: + outlier label validity, label-valid drop and the short-row baseline at 16k; 11-18: 16k gold-drop ablation / more-data runs)
superseded_by: 
artifact: 
summary: No. On single-length pools at 2k-64k scored by the early dense held-out-loss proxy (one epoch, eval_size 144 per length), the best router keep is 20-40% at every length the proxy resolves and does not fall with length (8k 40% tied with 20%; 16k 20-40% tie; 32k 40% beats the 20%/7% recipe at matched FLOPs by 0.01-0.02 CE on both seeds, 70% ties; 64k 20%, 40% +0.015 on both seeds). Low keeps break harder the longer the rows (10%/3.5%: +0.003 CE vs the recipe at 8k, +0.08 at 64k; 2.5% and 1.25% collapse at 32k/64k), so constant fraction fits and a constant kept-token count is rejected; 2k/4k saturate. Gold-aware keep grJ has the biggest mean speedups at 32k/64k (6-8x vs dense full FT) but stalls on outlier at every length, and at 8k never reaches dense's mid-epoch loss on the mean (2 seeds); grJc is outlier-safe at 32k, not at 64k. 16k follow-up (11-19): random gold drop breaks the outlier label (0% of dropped outlier rows keep a valid answer), so every earlier gold-drop number here (grJ, grJc, the 16k w*/r* arms) is for training on broken outlier labels; dropping whole majority topics keeps the label valid and fixes outlier (-0.030 ± 0.006 mean CE, outlier -0.082 ± 0.013, vs random drop at matched FLOPs, 2 seeds, eval_size 144), but the plain recipe trained on natively shorter rows at the same FLOPs is as good (-0.004 ± 0.005 vs label-valid drop).
---
# Does the recipe's best keep fraction fall as context grows? (keepfrac_len)

Prasann (2026-11-17): does the recipe's best keep fraction get lower as context grows? Lengths 2k-64k, each with a
single-length training pool and a same-length held-out set, scored on the cheap dense held-out-loss proxy. Code:
`research/token_routing/early_loss_proxy/keepfrac_len/` (branch `claude/keepfrac-vs-length-4bf10c`, built on
`claude/early-loss-proxy-d6e886`). Brief: `records/tokdrop-handoff-2026-11-17.md` section 2a.

## Verdict

**No. The best keep stays at 20-40% from 8k to 64k (40% at 32k, 20% at 64k) and does not fall with length; low
keeps break harder the longer the rows.** Constant fraction fits; constant kept-token count is rejected.

- **Best keep per length** (headline = held-out CE difference vs the 20%/7% recipe at matched training FLOPs, plus
  FLOP speedup vs dense full fine-tune `dft` at dft's mid-epoch CE):
  - 2k, 4k: **unresolvable**. Every arm drives held-out CE to 0.006-0.036 by the end of the epoch; differences at
    matched FLOPs are ≤ 0.02 and every keep from 70% to 2.5% reaches dft's mid-epoch CE at 2.6-4.1x. Saturated.
  - 8k: **40%/14%**, tied with 20%/7%: ΔCE vs recipe -0.004 / -0.004 (2 seeds), 3.34x vs 3.30x dft (seed means);
    70% is worse (+0.007 / +0.018); 10% and 5% slightly worse (+0.003, +0.012); 2.5% fails (+0.069).
  - 16k: **20-40% tie**: 40% ΔCE -0.001 / +0.014, 10% +0.002 / +0.014, 70% +0.007; speedups 3.0-3.2x (seed means)
    for 10-40%. 5% +0.010, 2.5% +0.036 (never reaches dft's mid-epoch CE).
  - 32k: **40%/14%** beats the recipe at matched FLOPs on both seeds (-0.019 / -0.010; paired SE of a final-CE
    difference 0.005); 70% ties it (-0.022 / -0.002); 10% +0.011 / +0.031, 5% +0.040, 2.5% +0.296 (collapse). Only
    20-70% reach dft's mid-epoch CE (seed means 40% 3.24x, 70% 3.06x, 20% 2.86x); 40% and 70% reach dft's
    end-of-epoch CE on seed 0 (3.95x / 4.09x).
  - 64k: **20%/7%**: 40% is +0.015 / +0.017 at matched FLOPs (both seeds; at equal steps +0.008 / -0.007, a tie),
    10% +0.080 / +0.033, 5% +0.103, 2.5% +0.278, 1.25% +0.602 (collapse). Only 20% and 40% reach dft's mid-epoch CE
    (seed means 3.17x and 2.58x); no router arm reaches dft's end-of-epoch CE (0.211; recipe 0.238 / 0.265). 70%/24.5%
    (1 seed) is +0.047 at matched FLOPs (2.27x at the mid level) but ends lowest at equal steps (0.222) and is the
    only router arm whose every task reaches dft's mid-epoch level (worst task outlier 1.96x).
- **Margin vs seed noise.** Seed-to-seed |Δ mean CE| at the end of the epoch is 0.001-0.018 per arm at 8k-32k
  (k20 at 32k 0.0006, k40 0.005, k10 0.015, k70 0.018) and up to 0.027 at 64k (k20); the speedup at the mid level
  moves by up to 1.0x between seeds (32k k70: 3.57 vs 2.54; 64k k20: 3.61 vs 2.72). So 40% vs 20% at 8k/16k and
  40% vs 70% at 32k are inside the noise; 40% > 20% at 32k (0.010-0.019) and 20% > 40% at 64k (0.015-0.017) have
  the same sign on both seeds but are no larger than the seed spread; the low-keep penalties at 32k/64k (0.03-0.6
  CE, both seeds where run) are far outside it.
- **Which rule fits.** A constant kept-token count anchored at 8k (40% x 8k ≈ 3.3k tokens) predicts best keeps of
  20% / 10% / 5% at 16k / 32k / 64k. Measured: 20-40% / 40-70% / 20% (5% at 64k is +0.10 CE worse than the recipe).
  Constant fraction in the 20-40% band fits every resolvable length. The penalty of keeping 10% grows with length
  (ΔCE vs recipe +0.003 at 8k, +0.002/+0.014 at 16k, +0.011/+0.031 at 32k, +0.080/+0.033 at 64k), so if anything longer rows
  need *more* kept tokens per row, not a fixed count.
- **Why low keeps buy little.** Below 20%/7% the per-step FLOP saving flattens (FLOPs/step vs dense: 64k 0.185 at
  20% -> 0.160 at 10% -> 0.140 at 1.25%; 32k 0.201 -> 0.174 -> 0.154) because blocks 0-6 run dense on every token
  and dominate; the loss penalty grows instead.
- **Per task / worst task.** Contradiction is where low keeps break first at every length (e.g. 32k final CE
  contradiction: dft 0.063, 40% 0.072, 20% 0.122, 10% 0.155, 5% 0.224); outlier is the worst task for most arms at
  64k (no router arm reaches dft's outlier mid-epoch CE there). 70% at 32k seed 0 is the only router run below dft on
  outlier (0.169 vs 0.187; seed 1 0.195).
- **Gold-aware stochastic keep (`grJ`) and the outlier-safe variant (`grJc`)**, 32k and 64k:
  - `grJ` has the largest mean speedups of the study (6.2x at 32k, 8.4x at 64k vs dft at the mid level; 0.080 /
    0.067 of dense FLOPs per step) but **outlier never reaches dft's mid-epoch CE** (final outlier CE 0.249 vs
    0.187 at 32k, 0.284 vs 0.220 at 64k) and at 64k contradiction does not either. The mean hides an outlier stall,
    as in the 4k-16k mixed-pool screen.
  - `grJc` (document drop on contradiction rows only) at 32k: 4.16x / 4.15x on the mean (2 seeds); worst task
    2.82x (contradiction) on seed 0, but on seed 1 contradiction never reaches dft's mid-epoch CE. 40%/14% is 3.24x
    (worst 2.72x / 2.41x). At equal steps `grJc` ends 0.015 / 0.019 CE above 40%/14% (contradiction +0.036 /
    +0.038), so its mean advantage is per-step cheapness, and contradiction is its weak task. At 64k `grJc` is 4.10x
    on the mean but contradiction (0.191 vs dft 0.086) and outlier do not reach dft's mid-epoch level: **outlier-
    safe at 32k, not at 64k** (one seed).
- **Every gold-drop number above uses random distractor drop, which breaks the outlier label** (11-19, section
  "Outlier label validity and the short-row baseline (16k)"). An outlier row's answer is the uniquely smallest topic
  (3 passages vs 23 majority topics of >= 4); dropping random passages leaves 0% of rows with a valid answer. So the
  `grJ` / `grJc` outlier stall and the 16k `w*` / `r*` arms measure training on wrong outlier labels. With a
  label-valid drop (whole majority topics) outlier is fixed at 16k, and gold drop then matches, but does not beat,
  the plain recipe trained on natively shorter rows at the same FLOPs.

The proxy understates gains relative to generation evals (recipe 1.9-2.2x on the proxy vs 3.3x on generation evals,
`records/early-dense-loss-proxy.md`) and cannot resolve final gaps below about 0.03-0.05 CE: use these numbers to
rank keeps, not to quote speedups.

## Gold drop at 8k vs 32k / 64k (Prasann 11-18: "does gold dropping work at 8k at a similar level?")

**No. At 8k `grJ` (gold-aware stochastic whole-document keep on all rows) never reaches dense full FT's mid-epoch
held-out CE on the mean, on both seeds: it stalls on outlier, as at 32k/64k, and at 8k that stall decides the mean.**

- 8k `grJ`, seeds 0 / 1 (Beaker `elp-kl8k-bkRPQTGH7X` / `-bk3M9X4FBD`, eval_size 144 ⚠, 48 per task): step cost
  0.100 / 0.102 of dense (32k 0.080, 64k 0.067). Final mean CE 0.0922 / 0.0918 vs dft 0.0445, recipe 0.0476 /
  0.0519, 40%/14% 0.0324 / 0.0390. Per task (contradiction / outlier / qdmatch_nq): `grJ` 0.045 / 0.182 / 0.050 and
  0.048 / 0.177 / 0.050; dft 0.022 / 0.068 / 0.044; recipe 0.033 / 0.067 / 0.043 and 0.023 / 0.085 / 0.048; 40%
  0.022 / 0.042 / 0.033 and 0.023 / 0.061 / 0.033.
- FLOP speedup vs dft at dft's mid-epoch CE: mean **not reached** (dft level 0.081; `grJ` ends at 0.092), per task
  contradiction 5.06x / 4.54x, outlier **not reached** (level 0.134; `grJ` ends at 0.18), qdmatch_nq 6.63x / 5.80x.
  At dft's end-of-epoch CE: not reached on the mean or on any task.
- Same arm across lengths (seed 0): mean speedup at the mid level 8k not reached, 32k 6.17x, 64k 8.41x; outlier
  never reaches the mid level at any length; its final outlier gap to dft is +0.114 (8k), +0.062 (32k), +0.064
  (64k) CE and its contradiction gap +0.023, +0.043, +0.099. The mean "works" at 32k/64k only because the outlier
  gap is a smaller share of a larger CE scale. Per task: qdmatch_nq reaches dft's mid level at 5.8-21x at every
  length, contradiction at 4.5-5.8x at 8k/32k but not at all at 64k, outlier never; no task reaches dft's
  end-of-epoch CE except 32k `grJc` qdmatch_nq.
- What was dropped (`[gold-doc-drop]`, rank 0's cumulative counts, 1,802 rows each): 8k sd0 1,535 rows dropped /
  267 whole / 0 full / 0 not-selected / 0 miss, doc-token keep 0.417; sd1 1,527 / 275 / 0 / 0 / 0, keep 0.420. 32k
  sd0 1,505 / 297, keep 0.404; 64k sd0 1,541 / 260, keep 0.383. Gold share of documents per row (row_info gold /
  n_docs; contradiction / outlier / qdmatch_nq): 8k 3.2 / 5.3 / 7.1%, 32k 0.9 / 1.6 / 1.9%, 64k 0.5 / 0.8 / 0.9%: all
  below the module's 0.5 cut, so `grJ` drops documents on all three tasks including qdmatch_nq (the module docstring's
  "qdmatch_nq keeps every document" does not describe these pools). `grJc` was not run at 8k (scope set by Prasann).
- Resolution: 8k sits near the proxy's floor (dft ends at 0.045, the routed arms at 0.03-0.05), so the proxy cannot
  rank dft / 20% / 40% there (their gaps are ≤ 0.02). The `grJ` failure is resolvable: its outlier gap (+0.11 CE)
  and mean gap (+0.040 to +0.060 vs recipe / 40%, paired SE 0.006, both seeds) are well above the 0.03-0.05 floor.

Gold-drop comparison table (per seed; x = FLOP speedup vs dft at dft's own CE level; levels in the last row of each
block; eval_size 144 per length, 48 per task ⚠):

| length | arm | seed | FLOPs/step vs dense | x dft @ mid CE (mean) | x dft @ end CE (mean) | x @ mid: contradiction / outlier / qdmatch_nq | x @ end: contradiction / outlier / qdmatch_nq | final CE mean | final CE contradiction / outlier / qdmatch_nq |
|---|---|---|---|---|---|---|---|---|---|
| 8k | dft | 0 | 1.000 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.0445 | 0.022 / 0.068 / 0.044 |
| 8k | k40 | 0 | 0.284 | 3.57 | 4.41 | 2.50 / 4.86 / 3.66 | not reached / 4.52 / 3.18 | 0.0324 | 0.022 / 0.042 / 0.033 |
| 8k | k40 | 1 | 0.284 | 3.11 | 4.01 | 2.92 / 7.02 / 3.67 | not reached / 3.77 / 3.39 | 0.0390 | 0.023 / 0.061 / 0.033 |
| 8k | k20 | 0 | 0.224 | 3.37 | not reached | 2.47 / 5.98 / 3.21 | not reached / 6.58 / 2.81 | 0.0476 | 0.033 / 0.067 / 0.043 |
| 8k | k20 | 1 | 0.224 | 3.23 | not reached | 2.96 / 4.05 / 2.78 | 4.49 / not reached / not reached | 0.0519 | 0.023 / 0.085 / 0.048 |
| 8k | grJ | 0 | 0.100 | not reached | not reached | 5.06 / not reached / 6.63 | not reached / not reached / not reached | 0.0922 | 0.045 / 0.182 / 0.050 |
| 8k | grJ | 1 | 0.102 | not reached | not reached | 4.54 / not reached / 5.80 | not reached / not reached / not reached | 0.0918 | 0.048 / 0.177 / 0.050 |
| 8k | levels | | | mid (dft step 100) 0.0808 | end 0.0445 | 0.051 / 0.134 / 0.058 | 0.022 / 0.068 / 0.044 | | |
| 32k | dft | 0 | 1.000 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.1440 | 0.063 / 0.187 / 0.182 |
| 32k | k40 | 0 | 0.259 | 3.50 | 3.95 | 2.72 / 2.89 / 4.22 | not reached / 3.96 / 4.52 | 0.1438 | 0.072 / 0.184 / 0.175 |
| 32k | k40 | 1 | 0.259 | 2.98 | not reached | 2.41 / 2.81 / 3.95 | not reached / 3.27 / not reached | 0.1487 | 0.079 / 0.181 / 0.186 |
| 32k | k20 | 0 | 0.201 | 2.70 | not reached | not reached / 2.77 / 3.95 | not reached / not reached / not reached | 0.1722 | 0.122 / 0.192 / 0.202 |
| 32k | k20 | 1 | 0.201 | 3.01 | not reached | not reached / 4.22 / 3.80 | not reached / 3.70 / not reached | 0.1728 | 0.119 / 0.189 / 0.211 |
| 32k | grJ | 0 | 0.080 | 6.17 | not reached | 5.80 / not reached / 12.87 | not reached / not reached / not reached | 0.1860 | 0.106 / 0.249 / 0.204 |
| 32k | grJc | 0 | 0.158 | 4.16 | not reached | 2.82 / 4.46 / 6.47 | not reached / not reached / 7.97 | 0.1590 | 0.108 / 0.197 / 0.172 |
| 32k | grJc | 1 | 0.158 | 4.15 | not reached | not reached / 3.64 / 5.55 | not reached / not reached / not reached | 0.1673 | 0.117 / 0.199 / 0.185 |
| 32k | levels | | | mid (dft step 100) 0.1922 | end 0.1440 | 0.109 / 0.204 / 0.264 | 0.063 / 0.187 / 0.182 | | |
| 64k | dft | 0 | 1.000 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.2110 | 0.086 / 0.220 / 0.327 |
| 64k | k40 | 0 | 0.240 | 2.87 | not reached | 2.82 / not reached / 2.98 | not reached / not reached / not reached | 0.2464 | 0.117 / 0.240 / 0.382 |
| 64k | k40 | 1 | 0.240 | 2.30 | not reached | 2.94 / not reached / 1.98 | not reached / not reached / not reached | 0.2578 | 0.124 / 0.232 / 0.418 |
| 64k | k20 | 0 | 0.185 | 3.61 | not reached | 3.57 / not reached / 4.02 | not reached / not reached / not reached | 0.2382 | 0.129 / 0.243 / 0.343 |
| 64k | k20 | 1 | 0.185 | 2.72 | not reached | 2.67 / not reached / 2.85 | not reached / not reached / not reached | 0.2650 | 0.166 / 0.234 / 0.395 |
| 64k | grJ | 0 | 0.067 | 8.41 | not reached | not reached / not reached / 21.13 | not reached / not reached / not reached | 0.2652 | 0.185 / 0.284 / 0.327 |
| 64k | grJc | 0 | 0.144 | 4.10 | not reached | not reached / not reached / 6.84 | not reached / not reached / not reached | 0.2544 | 0.191 / 0.235 / 0.337 |
| 64k | levels | | | mid (dft step 100) 0.2785 | end 0.2110 | 0.184 / 0.227 / 0.424 | 0.086 / 0.220 / 0.327 | | |

## Gold-drop ablation at 16k, and more data at matched FLOPs (Prasann 11-18)

Arms = the 20%/7% router plus gold-aware whole-document drop (`--st-gdd-keep-range lo,hi --st-gdd-whole-frac w`,
`--st-gdd-keep-power` for the skewed ones), named `w<whole%> r<lo>to<hi>`. One seed each, ⚠ eval_size 144 (48 per
task, kl16k held-out), so the paired SE of a final-CE difference is about 0.005-0.006 and seed-to-seed spread at 16k is
up to 0.014. The 1x arms are Prasann's runs: sneetches `results/kl16kabl/` (slurm 5204526 / 5204533 / 5204565; I checked
every number against the JSONs) and Lambda tag kl16kabl (jobs 115238-115240). Step-0 CE agrees across Beaker H100,
sneetches H200 and Lambda A100 to within 0.003 per task.

**Does the aggressive arm (w0 r05to20) catch up when given more data for the same FLOPs? No.** On the 6x pool it
reached 0.221 (running min) at 138 PF and 0.220 at 157 PF, step 600 of 1,350, when Prasann cancelled it. Its outlier
CE was 0.34-0.42 across those probes. At the same 138 PF: the 1x w15 r05to50 arm ends at 0.142, the recipe is at
0.184 / 0.167 (two seeds) and dft at 0.311. The run is partial (its LR had not annealed), but it trails every other
gold-drop arm by more than 0.07 CE at matched FLOPs, and outlier does not move.

**More data does help the baseline.** w15 r05to50 on the 2.4x pool (540 steps, 334 PF) ends at 0.1143 (contradiction
0.041, outlier 0.205, qdmatch_nq 0.096). Compared with:
- the recipe at the same 333 PF (0.1265 / 0.1203): -0.012 ± 0.005 and -0.006 ± 0.006 paired, i.e. as good or slightly
  better;
- 40%/14% at 426 PF (0.1143): equal;
- dft at 1,531 PF (0.1230): -0.009 ± 0.007.

It reaches dft's end-of-epoch CE with **5.7x** fewer FLOPs, and the recipe's best (seed 0) with 1.31x fewer. Its
outlier (0.205) is at dft's level (0.213); at 16k this arm shows no outlier stall, unlike `grJ` at 8k / 32k / 64k.

Per task at matched FLOPs (contradiction / outlier / qdmatch_nq):
- At 138 PF: 6x aggressive 0.146 / 0.343 / 0.169; 1x w15 0.073 / 0.224 / 0.127; recipe 0.184 / 0.222 / 0.143
  (seed 0).
- At 333 PF: 2.4x w15 0.041 / 0.204 / 0.095; recipe 0.067 / 0.207 / 0.106 and 0.061 / 0.211 / 0.089; dft
  0.167 / 0.242 / 0.248.

**Without the router (`gdnr`, 11-19 write-up of Prasann's runs).** Three arms ran gold drop with
`--st-otr-dense-every 1`, i.e. every training step dense after the drop, so they cut FLOPs only by removing documents
(sneetches slurm 5204584, 8 x H200, code b379a1dd4; numbers re-read from the JSONs). At matched FLOPs each is far
behind its router counterpart: `gdnr w5 r05to50` ends at 0.134 at 372 PF and is at 0.203 at 138 PF, against 0.1515 at
116 PF for `w5 r05to50` with the router; at 59 PF `gdnr w0 r05to20` is at 0.346 against 0.272 for its router version
(which ends there), and it ends at 0.264 after 171 PF. The router is still worth its cost on the kept documents.

Not run: w0 r05to50 on the 3.2x pool (slurm 5204547). Prasann put it on hold, and it was still held when this was
written. The skewed `p0.5` arm of 5204565 has no result (that job was cancelled).

Speedups ("x vs dft", "x vs recipe sd0") are the FLOPs the reference needs to reach the arm's lowest CE, divided by
the arm's FLOPs. When the arm gets below the reference's own best, the level is the reference's best instead. CE at
matched FLOPs reads each run's running-minimum curve, interpolated linearly between probes ("-" = the run spent more
than 2% less than that budget). Per-task values use per-task running minima.

| arm (whole% / keep range / skew) | pool | where | steps | FLOPs/step vs dense | PF | final CE mean | final CE contra / outlier / qdmatch | CE @ 59 PF | CE @ 138 PF | CE @ 333 PF | x vs dft | x vs recipe sd0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dft (dense full FT) | kl16k (1x) | Beaker 2 x H100 | 225 | 1.000 | 1531 | 0.1230 | 0.057 / 0.213 / 0.099 | 0.4923 | 0.3112 | 0.2236 | 1.00 | 0.24 |
| recipe 20%/7%, seed 0 | kl16k (1x) | Beaker 2 x H100 | 225 | 0.214 | 333 | 0.1265 | 0.066 / 0.207 / 0.105 | 0.2704 | 0.1843 | 0.1265 | 4.10 | 1.00 |
| recipe 20%/7%, seed 1 | kl16k (1x) | Beaker 2 x H100 | 225 | 0.214 | 333 | 0.1203 | 0.061 / 0.211 / 0.089 | 0.2835 | 0.1670 | 0.1203 | 5.14 | 1.20 |
| 40%/14%, seed 0 | kl16k (1x) | Beaker 2 x H100 | 225 | 0.273 | 426 | 0.1143 | 0.050 / 0.204 / 0.089 | 0.2566 | 0.1719 | 0.1251 | 4.41 | 1.03 |
| 40%/14%, seed 1 | kl16k (1x) | Beaker 2 x H100 | 225 | 0.273 | 426 | 0.1188 | 0.051 / 0.206 / 0.100 | 0.2522 | 0.1703 | 0.1341 | 4.13 | 0.93 |
| `w0 r05to20` | kl16k (1x) | sneetches 8 x H200 | 225 | 0.038 | 59 | 0.2862 | 0.211 / 0.423 / 0.226 | 0.2722 | - | - | 3.59 | 1.26 |
| `w0 r05to50` | kl16k (1x) | sneetches 8 x H200 | 225 | 0.067 | 105 | 0.1774 | 0.103 / 0.287 / 0.143 | 0.2062 | - | - | 4.39 | 1.47 |
| `w15 r05to50` | kl16k (1x) | sneetches 8 x H200 | 225 | 0.088 | 138 | 0.1416 | 0.073 / 0.225 / 0.127 | 0.1939 | 0.1416 | - | 6.52 | 1.78 |
| `w15 r25to75` | kl16k (1x) | Lambda 2 x A100 | 225 | 0.126 | 197 | 0.1299 | 0.070 / 0.226 / 0.094 | 0.2056 | 0.1452 | - | 5.52 | 1.55 |
| `w30 r05to50` | kl16k (1x) | Lambda 2 x A100 | 225 | 0.110 | 172 | 0.1366 | 0.077 / 0.219 / 0.113 | 0.1893 | 0.1468 | - | 5.59 | 1.50 |
| `w5 r05to50 p2` | kl16k (1x) | sneetches 8 x H200 | 225 | 0.061 | 95 | 0.1552 | 0.094 / 0.258 / 0.114 | 0.1847 | - | - | 5.83 | 1.92 |
| `w5 r05to50 p3` | kl16k (1x) | sneetches 8 x H200 | 225 | 0.054 | 84 | 0.1703 | 0.108 / 0.263 / 0.140 | 0.1825 | - | - | 5.71 | 1.94 |
| `w5 r05to50` | kl16k (1x) | sneetches 8 x H200 | 225 | 0.075 | 116 | 0.1515 | 0.076 / 0.269 / 0.110 | 0.1901 | - | - | 5.16 | 1.60 |
| `w50 r05to50` | kl16k (1x) | Lambda 2 x A100 | 225 | 0.140 | 218 | 0.1257 | 0.055 / 0.210 / 0.113 | 0.2181 | 0.1508 | - | 6.45 | 1.57 |
| `gdnr w5 r05to50` (no router: dense every step) | kl16k (1x) | sneetches 8 x H200 | 225 | 0.239 | 372 | 0.1344 | 0.059 / 0.249 / 0.096 | 0.2772 | 0.2026 | 0.1404 | 2.65 | 0.73 |
| `gdnr w5 r05to20` (no router) | kl16k (1x) | sneetches 8 x H200 | 225 | 0.141 | 220 | 0.1685 | 0.104 / 0.266 / 0.136 | 0.3048 | 0.2239 | - | 2.20 | 0.75 |
| `gdnr w0 r05to20` (no router) | kl16k (1x) | sneetches 8 x H200 | 225 | 0.110 | 171 | 0.2636 | 0.162 / 0.437 / 0.192 | 0.3463 | 0.2780 | - | 1.05 | 0.36 |
| `w0 r05to20` ⚠ CANCELLED at step 600 of 1,350 (partial, LR not annealed) | kl16k_x6 (6x) | sneetches 8 x H200 | 600 | 0.038 | 157 | 0.2200 | 0.135 / 0.364 / 0.161 | 0.2879 | 0.2211 | - | 2.20 | 0.57 |
| `w15 r05to50` | kl16k_x2p4 (2.4x) | sneetches 8 x H200 | 540 | 0.089 | 334 | 0.1143 | 0.041 / 0.205 / 0.096 | 0.2074 | 0.1666 | 0.1141 | 5.73 | 1.31 |

Pools (sneetches `/data/prasann/early_loss_proxy/shards/`, built by `keepfrac_len/grow_pool.py`, slurm 5204543):
- `kl16k_x6` (7,200 rows per task), `kl16k_x3p2` (3,840), `kl16k_x2p4` (2,880).
- Each is kl16k's 1,200 rows per task first (same part order), then mtb16kx16's other 16k rows of the same generator
  cells (seed 42; n_docs 324 / 96 / 174). All 3,600 kl16k rows are in mtb16kx16, by fingerprint.
- Not a fresh builder run: a fresh run would have taken more than 1 h, mostly outlier generation.
- Held-out hygiene against the kl16k held-out (heldout_v3 gen16k):
  - mtb16kx16_overlap.json reports 0 prompt hits and 0 gold-as-gold hits;
  - outlier rows containing any held-out outlier passage were dropped, as build_kl_pools does (7 rows). The passage
    hash is computed from decoded tokens and matched the source-text hash on 1,920 of 1,920 kl16k documents.
- Not re-checked: build_kl_pools' other exclusions (suite eval rungs, heldout_v4 candidates). Those come from
  mtb16kx16's own builder.
- Mean row length is 14,932 tokens, the same as kl16k.
- The trainer shuffles the whole pool. "Contains kl16k" means a superset, not a prefix in training order.

Runs: tag kl16kx, sneetches 8 x H200, `code_e5c1dfe30` (the code of the 1x ablation), one epoch of each pool, so the LR
schedule runs over each run's own length. Probes at 10, 25, 50, 75 and then every 75 steps. Results:
`sneetches:/data/prasann/early_loss_proxy/results/kl16kx/`. Scored by `keepfrac_len/ablation_16k.py` into
`keepfrac_len/ablation_16k.json`.

## Outlier label validity and the short-row baseline (16k, 11-18 / 11-19)

Question (Prasann): is gold drop weak on outlier because dropping distractors invalidates the label, and once the label
is kept valid, is gold drop better than simply training on shorter rows? Proxy as everywhere in this record: dense
held-out answer CE, ⚠ eval_size 144 (48 per task) unless marked, one epoch, 16 rows per step.

**Verdict.**
- **Random distractor drop breaks every outlier label.** A label-valid drop (keep whole majority topics) fixes outlier:
  against random drop at the same rows, steps and FLOPs it is -0.030 ± 0.006 lower in mean CE and -0.082 ± 0.013 on
  outlier (two seeds, paired, final probe at ~333 PF).
- **But at matched FLOPs, gold drop is about equal to the plain recipe trained on natively shorter rows** (arm E):
  E - A = -0.004 ± 0.005 (two seeds). Label-valid drop and the short-row mix both reach dense full FT's final CE at
  about 8-9x fewer FLOPs and the recipe's at 1.8-1.9x fewer; random drop 5.2x / 1.15x.
- **0% whole rows and keep 2-50% do not help** (one seed each): at 300 PF, 0% whole 0.104 and keep 2-50% 0.099 vs
  0.095 for the label-valid 5%-whole / 5-50% arm.

### Why random drop breaks the outlier label

A 16k outlier row (`ctc/src/ctc/tasks/outlier/sources/wiki100w.py`, mode "mixed") has 96 passages: 3 outlier passages
from one Wikipedia article plus 93 passages in 23 majority topics of >= 4 passages each. The outlier is defined only as
the uniquely smallest topic group (3 vs >= 4). The drop keeps every gold passage and a random share of the others, so
the majority topics shrink and the answer stops being the smallest group. Monte Carlo over the 1,188 mapped kl16k
outlier rows (`keepfrac_len/sim_outlier_label.py`, 20 draws per row):

| keep share r of distractors | label still valid (outlier uniquely smallest) | outlier topic strictly the LARGEST | largest incl. ties |
|---|---|---|---|
| U(0.05, 0.50) (the `w* r05to50` arms) | 0.0% | 33.8% | 76.4% |
| U(0.05, 0.20) | 0.0% | 82.7% | 99.1% |
| U(0.25, 0.75) | 0.0% | 2.1% | 30.6% |

Fix: `--st-gdd-groups <sidecar>` (`src/olmo_core/nn/gold_doc_drop.py`, branch `claude/elp-lambda-kl`, commits
38e4a1346 and 13bf548f4; `--st-gdd-keep-power` is b379a1dd4). A dropped outlier row keeps all gold passages plus a
share r ~ U(lo, hi) of the majority TOPICS (at least 2) with all their passages, so every kept majority topic still has
>= 4 passages. Sidecar = `{row fingerprint: [topic id per document]}`; a `null` entry passes the row through whole;
rows absent from the sidecar (contradiction, qdmatch_nq) get the ordinary drop unchanged; no sidecar = old behaviour.
The topic of each passage comes from matching its text to the article titles in `outlier.seed.jsonl.gz`
(`outlier_topic_map_text.py` for kl16k from the generator jsonl; `outlier_topic_keys.py` + `outlier_topic_map.py` from
the decoded pool rows for kl16k_x3p2, whose generator jsonl no longer exists). Mapped: 1,188 / 1,200 kl16k rows with
the first map (its 12 rejects are all valid under the second); 3,835 / 3,840 kl16k_x3p2 rows (3,436 / 3,440 in the
slice used below), and the two maps agree on 1,187 / 1,188 rows. Unmapped rows pass through whole.

### Outlier-only training (does the label fix help outlier alone?)

Pool `kl16k_outlier` (kl16k's 1,200 outlier rows), one epoch = 75 steps, 2 x H200 on sneetches, recipe router on
every arm, seed 0. ⚠ eval_size 48 only (the kl16k outlier held-out rows; SE about ±0.008-0.018 at step 75).
"random" = old per-passage drop, "topic" = label-valid drop.

| arm | FLOPs/step vs dense | epoch PF | outlier CE at step 0 / 10 / 25 / 50 / 75 | slurm |
|---|---|---|---|---|
| recipe (k20) | 0.208 | 104.6 | 1.233 / 0.271 / 0.274 / 0.243 / 0.229 | 5207278 |
| random, 15% whole, keep 5-50% | 0.086 | 43.3 | 1.233 / 0.298 / 0.262 / 0.250 / 0.240 | 5207279 |
| random, 5% whole, keep 5-50% | 0.074 | 36.9 | 1.233 / 0.281 / 0.283 / 0.243 / 0.243 | 5207424 |
| random, 0% whole, keep 5-50% | 0.065 | 32.4 | 1.233 / 0.445 / 0.336 / 0.232 / 0.238 | 5207280 |
| topic, 15% whole, keep 5-50% | 0.087 | 43.4 | 1.233 / 0.276 / 0.237 / 0.232 / 0.227 | 5207403 |
| topic, 5% whole, keep 5-50% | 0.073 | 36.7 | 1.233 / 0.306 / 0.272 / 0.256 / 0.235 | 5207411 |
| topic, 5% whole, keep 2-50% | 0.071 | 35.6 | 1.233 / 0.271 / 0.247 / 0.258 / 0.223 | 5207412 |
| topic, 0% whole, keep 5-50% | 0.066 | 33.2 | 1.233 / 0.250 / 0.268 / 0.231 / 0.239 | 5207375 |

Read: with every row's label broken (random, 0% whole) learning starts slower (0.445 vs 0.250 at step 10), but after one
epoch every arm sits at 0.223-0.243, inside the noise of eval_size 48. Outlier-only training at 16k does not show the
stall seen at 32k/64k; the difference appears in the three-task, larger-budget runs below.

### Three-task 16k at the recipe's FLOPs: label-valid drop, random drop, and the short-row baseline

Rows: 645 steps x 16 = 10,320 rows, 3,440 per task. Arms A-D train on `kl16k_x3p2_n3440` (the first 3,440 rows per
task of `kl16k_x3p2`, `slice_prefix_pool.py`), so the LR schedule covers exactly 645 steps; the same rows and order for
A-D at a given seed. Router 20%/7% on every arm. Code 13bf548f4. 4 x H200 per arm (seed 0 of A and B on sneetches,
everything else on cubbins). Probes at 0, 25, 50, 100, 150, 225, 300, 375, 450, 525, 600, 645.
- **A**: label-valid drop on outlier rows + ordinary drop on contradiction / qdmatch_nq, 5% whole, keep 5-50%.
- **B**: ordinary random drop on all three tasks, same flags (the control).
- **C**: as A with 0% whole. **D**: as A with keep 2-50%. One seed each; cheaper per step, so they end below 333 PF.
- **E**: no gold drop: the plain recipe on a LENGTH MIX of native rows with A's FLOPs (below).

**E pool** (`kl16k_lenmixE_172_1138_1200_930`, `build_lenmix_pool.py`): per task 172 rows at 16k (5%, from the x3p2
slice), 1,138 at 2k, 1,200 at 4k, 930 at 8k (from the single-length pools kl2k / kl4k / kl8k: the same seed-42 generator
examples at shorter lengths, so E sees the same questions with fewer distractors); 10,320 rows, mean 5,137 tokens per
row, shuffled within each task part (the loader also shuffles across parts). Mean row length (contradiction / outlier /
qdmatch_nq): 2k 2,587 / 2,202 / 1,860; 4k 4,149 / 4,311 / 3,817; 8k 8,366 / 8,634 / 7,740; 16k 14,486 / 14,458 /
15,828. FLOP calibration: a first mix (968 / 1,200 / 1,100 rows at 2k / 4k / 8k, 5,435 tokens per row) measured
0.549 PF per step, 6% above A, and was cancelled at step 89 (slurm 5208182); the final mix measured 0.521 PF per step
(A 0.516). 4k is used whole (1,200 rows per task), so the short-row mix is 35 / 37 / 28% (2k / 4k / 8k) rather than
the 30 / 40 / 30% a uniform 5-50% keep would suggest; no rows were generated. Rows are EOS-padded to 16,384; the
router trims the right padding to the longest real row of each 2-row micro-batch, so GPU compute pads to that
length, while the FLOP meter charges each example at its own real length with padding excluded (`--flop-per-example`
default), the same accounting as the gold-drop arms.

Mean held-out CE (contradiction / outlier / qdmatch_nq at the final probe). Two-seed rows average the two seeds' curves
probe by probe. Speedups = FLOPs the reference needs to reach the arm's lowest CE / the arm's FLOPs (the reference's
best when the arm gets below it); "recipe" = the two-seed mean curve of the 1x recipe (0.1234 at 333 PF), dft = dense
full FT (0.1230 at 1,531 PF). CE at matched PF from the running-minimum curve. ⚠ eval_size 144 (48 per task); the
paired SE of a final-CE difference is 0.005-0.009.

| arm | seed | slurm | PF/step (dense 16k step = 6.80) | PF total | final CE | final CE contra / outlier / qdmatch | CE @ 138 / 200 / 300 / 333 PF | x vs dft | x vs recipe |
|---|---|---|---|---|---|---|---|---|---|
| A label-valid, 5% whole, keep 5-50% | 0 | 5207720 | 0.517 | 333.2 | 0.0885 | 0.042 / 0.138 / 0.086 | 0.140 / 0.122 / 0.094 / 0.0875 | 8.09 | 1.82 |
| A | 1 | 5207985 | 0.515 | 332.0 | 0.0900 | 0.038 / 0.135 / 0.098 | 0.148 / 0.119 / 0.096 / 0.090 | 8.14 | 1.78 |
| **A, two-seed mean** | | | | | **0.0892** | 0.040 / 0.136 / 0.092 | 0.144 / 0.120 / 0.095 / 0.089 | **8.1** | **1.78** |
| B random, 5% whole, keep 5-50% | 0 | 5207674 | 0.513 | 330.8 | 0.1200 | 0.043 / 0.221 / 0.096 | 0.164 / 0.143 / 0.124 / 0.120 | 5.00 | 1.10 |
| B | 1 | 5207986 | 0.515 | 332.0 | 0.1186 | 0.043 / 0.215 / 0.098 | 0.153 / 0.139 / 0.121 / 0.119 | 5.73 | 1.25 |
| **B, two-seed mean** | | | | | **0.1193** | 0.043 / 0.218 / 0.097 | 0.158 / 0.142 / 0.122 / 0.119 | **5.2** | **1.15** |
| E recipe on length mix | 0 | 5208228 | 0.521 | 336.1 | 0.0870 | 0.042 / 0.123 / 0.097 | 0.141 / 0.111 / 0.093 / 0.087 | 9.06 | 1.98 |
| E | 1 | 5208321 | 0.521 | 335.9 | 0.0827 | 0.046 / 0.098 / 0.104 | 0.139 / 0.120 / 0.095 / 0.084 | 8.05 | 1.76 |
| **E, two-seed mean** | | | | | **0.0849** | 0.044 / 0.110 / 0.101 | 0.141 / 0.116 / 0.095 / 0.086 | **8.6** | **1.89** |
| C label-valid, 0% whole, keep 5-50% | 0 | 5208183 | 0.465 | 299.8 | 0.1044 | 0.041 / 0.163 / 0.109 | 0.147 / 0.128 / 0.104 / - | 6.04 | 1.32 |
| D label-valid, 5% whole, keep 2-50% | 0 | 5208304 | 0.499 | 321.7 | 0.0968 | 0.040 / 0.156 / 0.095 | 0.151 / 0.128 / 0.099 / - | 6.74 | 1.48 |
| ref: recipe 1x (kl16k, 225 steps), two-seed mean | | | 1.48 | 333.3 | 0.1234 | | 0.176 / 0.148 / 0.127 / 0.123 | 4.5 | 1.00 |
| ref: `w15 r05to50` random drop on kl16k_x2p4 | 0 | 5204546 | 0.618 | 334 | 0.1143 | 0.041 / 0.205 / 0.096 | 0.167 / 0.134 / 0.117 / 0.114 | 5.73 | 1.25 |

Paired differences at the final probe (mean over the 144 rows; per task over 48):

| comparison | mean | contradiction | outlier | qdmatch_nq |
|---|---|---|---|---|
| A - B, seed 0 | -0.032 ± 0.007 | -0.002 ± 0.005 | -0.083 ± 0.018 | -0.010 ± 0.007 |
| A - B, seed 1 | -0.029 ± 0.007 | -0.005 ± 0.007 | -0.080 ± 0.017 | 0.000 ± 0.008 |
| A - B, two-seed mean | **-0.030 ± 0.006** | -0.004 ± 0.005 | **-0.082 ± 0.013** | -0.005 ± 0.005 |
| E - A, two-seed mean | **-0.004 ± 0.005** | +0.004 ± 0.006 | -0.026 ± 0.013 | +0.009 ± 0.007 |
| E - B, two-seed mean | -0.034 ± 0.007 | +0.001 ± 0.008 | -0.108 ± 0.014 | +0.003 ± 0.008 |
| D - A, seed 0 (D at 322 PF) | +0.008 ± 0.007 | -0.002 ± 0.010 | +0.018 ± 0.018 | +0.009 ± 0.009 |

What was dropped (`[gold-doc-drop]`, rank 0's cumulative counts over 2,562 rows): A seed 0: 137 whole, 2,423 dropped, 849
outlier rows grouped, 2 unmapped passed whole, doc-token keep 0.337; A seed 1: 877 grouped, 0 unmapped, keep 0.331; B:
keep 0.331 / 0.334; C: 0 whole, 2,560 dropped, keep 0.301; D: keep 0.324. Contradiction and qdmatch_nq rows took the
ordinary drop path in A, C and D (full 0, not-selected 0, as in B).

Read:
- The label is the outlier problem. Holding everything else fixed (rows, steps, FLOPs, drop rate), keeping the outlier
  answer valid removes 0.08 CE of outlier loss on both seeds; contradiction and qdmatch_nq move by less than their SE.
  The earlier "gold drop stalls on outlier" findings (`grJ` at 8k / 32k / 64k, the 16k `w*` arms) are a property of
  training on wrong answers, not of gold drop itself.
- Gold drop is not better than shorter data. The recipe trained on a length mix of native rows with the same FLOPs (E)
  ties label-valid drop on the mean (-0.004 ± 0.005) and is, if anything, ahead on outlier (-0.026 ± 0.013, two SE).
  What gold drop buys at this budget is what cheaper, shorter rows buy: more rows per FLOP.
- Dropping more does not help: 0% whole (C) and keep 2-50% (D) trail A at matched FLOPs (at 300 PF 0.104 and 0.099
  vs 0.095).

Limits: 16k only (the 32k / 64k `grJ` arms were not rerun with a label-valid drop); proxy CE, not generation evals;
⚠ eval_size 144 (48 per task), and 48 for the outlier-only table; two seeds for A, B, E, one for C and D; E's FLOPs are
1% above A's (336 vs 333 PF); the topic map is recovered from article titles, not recorded by the generator.

Files: results `results/kl16kol/` (sneetches) and `results/kl16kx/elp-kl16kx-{tpA,gdB,tpC,tpD,E2_lenmix_k20}*` (seed 0
of A and B on sneetches, the rest on cubbins), all under `/data/prasann/early_loss_proxy/`. Pools on both nodes:
`shards/kl16k_outlier` (+ `gdd_groups.json`), `shards/kl16k_x3p2_n3440` (+ `gdd_groups.json`),
`shards/kl16k_lenmixE_172_1138_1200_930`; held-out `heldout_kl16k_outlier.npz` (sneetches). Scripts in
`keepfrac_len/`: `sim_outlier_label.py`, `outlier_topic_map_text.py`, `outlier_topic_keys.py`, `outlier_topic_map.py`,
`make_outlier_only_pool.py`, `slice_prefix_pool.py`, `build_lenmix_pool.py`, `outlier_only_summary.py`,
`curves_lib.py` + `ab_compare.py` (curves, speedups, paired differences), `check_gdd_groups.py` (CPU check of the
groups path), `launch_outlier_only.sh`, `launch_x3p2n.sh`. Recorded with `ops result add --check` (experiment
`tokdrop_lr`, metric `heldout_ce_dense_final`, run ids `elp-kl16kol-<arm>-sd0-probe` and `elp-kl16kx-<arm>-sd<seed>-probe`;
only eval_size warnings).

## Setup

- Base `q35-4b-base-markerfix` (Qwen3.5-4B, marker-repaired). Training pools `kl{2k,4k,8k,16k,32k,64k}`
  (`build_kl_pools.py`): three tasks (contradiction, outlier, qdmatch_nq), 1,200 rows each, one length per pool.
  Row lengths (mean / max): 2k 2,215 / 3,367; 4k 4,092 / 5,354; 8k 8,246 / 9,819; 16k 14,932 / 16,308; 32k
  29,210 / 31,483; 64k 58,549 / 62,307 (the short pools are rung-labelled and overshoot their label; trainer
  --seq-len 4,096 / 6,144 / 10,240 there).
- Held-out: `heldout_kls.npz` (sets kl2k/4k/8k) and `heldout_kl.npz` (kl16k/32k/64k), 48 rows per task (eval_size
  144 per length, ⚠ below 500: paired SE of a CE difference between two arms ≈ 0.003-0.006 at the end of the epoch),
  gold not in training (builder checks).
- One epoch = 3,600 rows / 16 rows per step = 225 steps, the trainer's real LR schedule over that epoch; dense
  probes at steps 0, 10, 25, 50, ..., 225 (teacher-forced answer CE of the DENSE forward, no generation).
- Arms: `dft` (dense full fine-tune, LR 5e-6, the reference), `ref` (dense LoRA r64, LR 1e-5), router keeps
  `k70/k40/k20/k10/k5/k2p5/k1p25` = the recipe's online router with cut 1 (after block 7) keeping 70/40/20/10/5/2.5/
  1.25% and cut 2 (after block 16) keeping 0.35x that (24.5/14/7/3.5/1.75/0.9/0.45%), `grJ` (recipe + gold-aware
  stochastic whole-document keep, `--st-gdd-keep-range 0.05,0.5 --st-gdd-whole-frac 0.15`), `grJc` (`grJ` with the
  document drop on contradiction rows only). LR 1e-5 for all LoRA arms.
- Compute: Beaker 2 x H100 (jupiter/ceres), micro-batch 2 (1 at 64k and for `dft` at 32k; 2 GPUs at micro 2 ran
  out of memory there); `dft` and `ref` at 64k on 8 x H100 (2 x H200 reserves 78 GiB and takes 76 s/step for `dft`).
  Quick reads `q32k` / `q64k` on mooney 2 x H200. Beaker vs mooney agree: 64k `dft` step-10 CE 0.528 (8 x H100)
  vs 0.527 (2 x H200), `k10` 0.569 vs 0.572.
- Metrics (`analyze_kl.py`, `summarize_kl.py`): PF = training FLOPs actually spent (the trainer's meter).
  (1) **ΔCE vs recipe at matched FLOPs**: mean held-out CE of the arm minus the recipe's, both read (running-min
  curves, linear interpolation) at the smaller of the two runs' total training FLOPs; negative = better than the
  recipe. (2) **x dft @ mid CE**: PF_dft / PF_arm to reach dft's own mean CE at step 100 (the "mid-epoch" level),
  "not reached" if the arm never gets there. (3) the same at dft's end-of-epoch CE. Worst task = min over the three
  tasks of the per-task speedup at dft's per-task mid-epoch CE.

## Quick reads (40 steps, one seed)

- `q32k` (pool `mtb64k_r32k`, 960 rows = 60-step epoch, held-out `heldout_v4` gen32k, eval_size 72): 40%/14% and
  20%/7% tie (CE 0.370 vs 0.380 ±0.015); 10% and below plateau higher. Full table `keepfrac_len/INTERIM_q32k.md`.
- `q64k` (pool kl64k, held-out kl64k, eval_size 144; 225-step schedule, stopped at 40): held-out CE at step 40: dft
  0.404, dense LoRA 0.370, 40% 0.452, 20% 0.471, 10% 0.496, 5% 0.499, 2.5% 1.052, 1.25% 1.051 (both diverged
  after step 10). Same ordering as the full 64k epoch. Slurm jobs 5201918-5201921 (mooney, completed); scored
  `keepfrac_len/q64k_scored.json`.

## Memory and time (for reference only; micro-batch and GPU count differ between dense arms)

Peak GiB (max over ranks) for router arms is flat in keep at each length (2k 24-25, 8k 27-28, 16k 32, 32k 46.5,
64k 46 on 2 x H100), i.e. it is set by the dense early blocks and full-length activations, not by the kept share.
Train s/step on 2 x H100: 16k recipe 6.1, 40% 7.4, 70% 8.4, dft 13.9; 32k recipe 9.8, 40% 12.3, 70% 15.3.

## Per-length tables

Seed values are separated by " / ". ⚠ eval_size 144 per length (48 rows per task), below 500: the paired SE of a CE
difference between two arms on the same rows is 0.003-0.006 at the end of the epoch; seed-to-seed spread is up to
0.015 CE and 0.6x in the mid-level speedup. "not reached" = the arm never gets down to that dft CE level.

<!-- tables:begin (finalize.sh) -->

**2k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.0233; end of epoch 0.0103)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0204 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.0103 | 0.017 / 0.003 / 0.010 |
| dense LoRA | 0 | 0.746 | +0.0068 | 1.85 | 1.75 | 1.71 (outlier) | 1.90 | 1.71 | 1.79 | 0.0070 | 0.015 / 0.002 / 0.004 |
| 70% / 24.5% | 0 | 0.407 | +0.0001 | 2.64 | 3.15 | 1.91 (qdmatch_nq) | 2.65 | 2.90 | 1.91 | 0.0057 | 0.007 / 0.003 / 0.007 |
| 40% / 14% | 0 | 0.319 | +0.0008 | 4.05 | not reached | 3.18 (qdmatch_nq) | 4.05 | 4.81 | 3.18 | 0.0130 | 0.021 / 0.008 / 0.010 |
| 20% / 7% (recipe) | 0 | 0.262 | +0.0000 | 2.78 | not reached | 2.57 (qdmatch_nq) | 2.95 | 3.95 | 2.57 | 0.0138 | 0.012 / 0.020 / 0.009 |
| 10% / 3.5% | 0 | 0.233 | -0.0066 | 3.22 | not reached | 2.82 (qdmatch_nq) | 3.62 | 4.29 | 2.82 | 0.0110 | 0.015 / 0.012 / 0.006 |
| 5% / 1.75% | 0 | 0.219 | -0.0033 | 3.06 | not reached | not reached (qdmatch_nq) | 2.47 | 5.06 | not reached | 0.0144 | 0.027 / 0.005 / 0.011 |
| 2.5% / 0.9% | 0 | 0.212 | -0.0030 | 2.88 | not reached | not reached (qdmatch_nq) | 2.86 | 3.98 | not reached | 0.0147 | 0.019 / 0.016 / 0.009 |

**4k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.0429; end of epoch 0.0129)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0403 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.0129 | 0.012 / 0.012 / 0.014 |
| dense LoRA | 0 | 0.745 | +0.0318 | 1.49 | not reached | 1.13 (qdmatch_nq) | 1.71 | 1.43 | 1.13 | 0.0169 | 0.016 / 0.016 / 0.018 |
| 70% / 24.5% | 0 | 0.390 | +0.0190 | 2.63 | not reached | not reached (qdmatch_nq) | 2.95 | 1.44 | not reached | 0.0270 | 0.027 / 0.024 / 0.031 |
| 40% / 14% | 0 | 0.299 | +0.0059 | 3.25 | not reached | 2.16 (qdmatch_nq) | 2.40 | 2.31 | 2.16 | 0.0214 | 0.029 / 0.014 / 0.021 |
| 20% / 7% (recipe) | 0 | 0.240 | +0.0000 | 3.63 | not reached | not reached (qdmatch_nq) | 3.02 | 2.22 | not reached | 0.0187 | 0.018 / 0.014 / 0.025 |
| 10% / 3.5% | 0 | 0.210 | +0.0048 | 3.82 | not reached | not reached (qdmatch_nq) | 3.30 | 2.02 | not reached | 0.0258 | 0.026 / 0.027 / 0.024 |
| 5% / 1.75% | 0 | 0.195 | +0.0033 | 3.09 | not reached | not reached (qdmatch_nq) | 2.70 | 2.08 | not reached | 0.0266 | 0.033 / 0.022 / 0.024 |
| 2.5% / 0.9% | 0 | 0.188 | +0.0119 | 2.99 | not reached | not reached (qdmatch_nq) | 2.90 | 2.15 | not reached | 0.0364 | 0.033 / 0.047 / 0.030 |

**8k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.0808; end of epoch 0.0445)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0749 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.0445 | 0.022 / 0.068 / 0.044 |
| dense LoRA | 0 | 0.743 | +0.0571 | 1.15 | 1.62 | 0.84 (contradiction) | 0.84 | 1.63 | 1.06 | 0.0365 | 0.021 / 0.053 / 0.036 |
| 70% / 24.5% | 0,1 | 0.377 | +0.0065 / +0.0178 | 2.76 (2.53-2.99) | 3.35 (2.98-3.72) | 2.69 (contradiction) / 2.16 (outlier) | 2.69 / 2.83 | 3.53 / 2.16 | 3.39 / 2.98 | 0.0306 / 0.0396 | 0.017 / 0.043 / 0.031 ; 0.020 / 0.069 / 0.030 |
| 40% / 14% | 0,1 | 0.284 | -0.0035 / -0.0043 | 3.34 (3.11-3.57) | 4.21 (4.01-4.41) | 2.50 (contradiction) / 2.92 (contradiction) | 2.50 / 2.92 | 4.86 / 7.02 | 3.66 / 3.67 | 0.0324 / 0.0390 | 0.022 / 0.042 / 0.033 ; 0.023 / 0.061 / 0.033 |
| 20% / 7% (recipe) | 0,1 | 0.224 | +0.0000 / +0.0000 | 3.30 (3.23-3.37) | not reached / not reached | 2.47 (contradiction) / 2.78 (qdmatch_nq) | 2.47 / 2.96 | 5.98 / 4.05 | 3.21 / 2.78 | 0.0476 / 0.0519 | 0.033 / 0.067 / 0.043 ; 0.023 / 0.085 / 0.048 |
| 10% / 3.5% | 0 | 0.195 | +0.0030 | 3.21 | not reached | not reached (contradiction) | not reached | 5.75 | 2.83 | 0.0584 | 0.052 / 0.077 / 0.046 |
| 5% / 1.75% | 0 | 0.180 | +0.0119 | 3.30 | not reached | not reached (contradiction) | not reached | 4.35 | not reached | 0.0712 | 0.057 / 0.092 / 0.065 |
| 2.5% / 0.9% | 0 | 0.173 | +0.0690 | not reached | not reached | not reached (contradiction) | not reached | 2.87 | not reached | 0.1295 | 0.130 / 0.127 / 0.132 |
| grJ (gold-aware stochastic keep) | 0,1 | 0.100 | +0.0071 / -0.0027 | not reached / not reached | not reached / not reached | not reached (outlier) / not reached (outlier) | 5.06 / 4.54 | not reached / not reached | 6.63 / 5.80 | 0.0922 / 0.0918 | 0.045 / 0.182 / 0.050 ; 0.048 / 0.177 / 0.050 |

**16k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.1455; end of epoch 0.1230)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0971 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.1230 | 0.057 / 0.213 / 0.099 |
| dense LoRA | 0 | 0.739 | +0.0325 | 1.30 | 1.79 | 0.94 (outlier) | 1.37 | 0.94 | 1.73 | 0.1138 | 0.039 / 0.215 / 0.088 |
| 70% / 24.5% | 0 | 0.367 | +0.0067 | 2.94 | 3.56 | 2.21 (qdmatch_nq) | 3.10 | 2.86 | 2.21 | 0.1107 | 0.047 / 0.207 / 0.079 |
| 40% / 14% | 0,1 | 0.273 | -0.0014 / +0.0138 | 3.01 (2.76-3.27) | 4.27 (4.13-4.41) | 2.80 (contradiction) / 2.59 (outlier) | 2.80 / 2.96 | 4.02 / 2.59 | 3.57 / 2.88 | 0.1143 / 0.1188 | 0.050 / 0.204 / 0.089 ; 0.051 / 0.206 / 0.100 |
| 20% / 7% (recipe) | 0,1 | 0.214 | +0.0000 / +0.0000 | 3.19 (2.89-3.49) | not reached / 5.14 | 2.72 (contradiction) / 3.15 (contradiction) | 2.72 / 3.15 | 4.23 / 8.61 | 3.03 / 3.34 | 0.1265 / 0.1203 | 0.066 / 0.207 / 0.105 ; 0.061 / 0.211 / 0.089 |
| 10% / 3.5% | 0,1 | 0.185 | +0.0015 / +0.0140 | 2.97 (2.80-3.13) | not reached / not reached | not reached (contradiction) / not reached (contradiction) | not reached / not reached | 6.07 / 3.29 | 3.06 / 2.64 | 0.1335 / 0.1386 | 0.088 / 0.210 / 0.102 ; 0.095 / 0.213 / 0.108 |
| 5% / 1.75% | 0 | 0.170 | +0.0099 | 2.59 | not reached | not reached (contradiction) | not reached | 3.54 | 2.68 | 0.1450 | 0.109 / 0.216 / 0.110 |
| 2.5% / 0.9% | 0 | 0.163 | +0.0357 | not reached | not reached | not reached (contradiction) | not reached | not reached | 2.11 | 0.1738 | 0.173 / 0.223 / 0.125 |

**32k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.1922; end of epoch 0.1440)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.0917 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.1440 | 0.063 / 0.187 / 0.182 |
| dense LoRA | 0 | 0.733 | +0.0377 | 1.84 | 1.65 | 1.08 (outlier) | 1.47 | 1.08 | 2.32 | 0.1329 | 0.061 / 0.184 / 0.154 |
| 70% / 24.5% | 0,1 | 0.354 | -0.0220 / -0.0022 | 3.06 (2.54-3.57) | 3.51 (2.93-4.09) | 2.52 (contradiction) / 2.11 (contradiction) | 2.52 / 2.11 | 2.61 / 2.99 | 4.39 / 3.45 | 0.1236 / 0.1413 | 0.054 / 0.169 / 0.148 ; 0.071 / 0.195 / 0.158 |
| 40% / 14% | 0,1 | 0.259 | -0.0190 / -0.0104 | 3.24 (2.98-3.50) | 3.95 / not reached | 2.72 (contradiction) / 2.41 (contradiction) | 2.72 / 2.41 | 2.89 / 2.81 | 4.22 / 3.95 | 0.1438 / 0.1487 | 0.072 / 0.184 / 0.175 ; 0.079 / 0.181 / 0.186 |
| 20% / 7% (recipe) | 0,1 | 0.201 | +0.0000 / +0.0000 | 2.85 (2.70-3.01) | not reached / not reached | not reached (contradiction) / not reached (contradiction) | not reached / not reached | 2.77 / 4.22 | 3.95 / 3.80 | 0.1722 / 0.1728 | 0.122 / 0.192 / 0.202 ; 0.119 / 0.189 / 0.211 |
| 10% / 3.5% | 0,1 | 0.174 | +0.0114 / +0.0306 | not reached / not reached | not reached / not reached | not reached (contradiction) / not reached (contradiction) | not reached / not reached | not reached / not reached | 4.16 / 3.86 | 0.1938 / 0.2090 | 0.155 / 0.211 / 0.215 ; 0.185 / 0.227 / 0.216 |
| 5% / 1.75% | 0 | 0.160 | +0.0400 | not reached | not reached | not reached (contradiction) | not reached | not reached | 3.18 | 0.2336 | 0.224 / 0.222 / 0.255 |
| 2.5% / 0.9% | 0 | 0.154 | +0.2961 | not reached | not reached | not reached (contradiction) | not reached | not reached | not reached | 0.4955 | 0.608 / 0.404 / 0.474 |
| grJ (gold-aware stochastic keep) | 0 | 0.080 | -0.1110 | 6.17 | not reached | not reached (outlier) | 5.80 | not reached | 12.87 | 0.1860 | 0.106 / 0.249 / 0.204 |
| grJc (grJ, doc drop on contradiction only) | 0,1 | 0.158 | -0.0362 / -0.0175 | 4.16 (4.15-4.16) | not reached / not reached | 2.82 (contradiction) / not reached (contradiction) | 2.82 / not reached | 4.46 / 3.64 | 6.47 / 5.55 | 0.1590 / 0.1673 | 0.108 / 0.197 / 0.172 ; 0.117 / 0.199 / 0.185 |

**64k** (eval_size 144 ⚠ < 500, 48 rows/task; matched level = dft's mean held-out CE at step 100 = 0.2785; end of epoch 0.2110)

| arm | seeds | FLOPs/step vs dense | ΔCE vs recipe @ matched FLOPs | x dft @ mid CE | x dft @ end CE | worst task x @ mid (task) | contradiction x @ mid | outlier x @ mid | qdmatch_nq x @ mid | final CE | final CE contra / outlie / qdmatc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense full FT | 0 | 1.000 | +0.1634 | 1.00 | 1.00 | 1.00 (contradiction) | 1.00 | 1.00 | 1.00 | 0.2110 | 0.086 / 0.220 / 0.327 |
| dense LoRA | 0 | 0.723 | +0.0961 | 1.34 | 1.84 | 1.34 (qdmatch_nq) | 1.71 | 1.83 | 1.34 | 0.1994 | 0.089 / 0.212 / 0.297 |
| 70% / 24.5% | 0 | 0.335 | +0.0468 | 2.27 | not reached | 1.96 (outlier) | 2.30 | 1.96 | 2.35 | 0.2217 | 0.106 / 0.215 / 0.344 |
| 40% / 14% | 0,1 | 0.240 | +0.0145 / +0.0170 | 2.58 (2.30-2.87) | not reached / not reached | not reached (outlier) / not reached (outlier) | 2.82 / 2.94 | not reached / not reached | 2.98 / 1.98 | 0.2464 / 0.2578 | 0.117 / 0.240 / 0.382 ; 0.124 / 0.232 / 0.418 |
| 20% / 7% (recipe) | 0,1 | 0.185 | +0.0000 / +0.0000 | 3.17 (2.72-3.61) | not reached / not reached | not reached (outlier) / not reached (outlier) | 3.57 / 2.67 | not reached / not reached | 4.02 / 2.85 | 0.2382 / 0.2650 | 0.129 / 0.243 / 0.343 ; 0.166 / 0.234 / 0.395 |
| 10% / 3.5% | 0,1 | 0.160 | +0.0797 / +0.0329 | not reached / not reached | not reached / not reached | not reached (contradiction) / not reached (contradiction) | not reached / not reached | not reached / not reached | not reached / not reached | 0.3311 / 0.3128 | 0.285 / 0.253 / 0.456 ; 0.237 / 0.255 / 0.446 |
| 5% / 1.75% | 0 | 0.148 | +0.1030 | not reached | not reached | not reached (contradiction) | not reached | not reached | not reached | 0.3548 | 0.304 / 0.278 / 0.482 |
| 2.5% / 0.9% | 0 | 0.142 | +0.2778 | not reached | not reached | not reached (contradiction) | not reached | not reached | not reached | 0.5318 | 0.638 / 0.425 / 0.533 |
| 1.25% / 0.45% | 0 | 0.140 | +0.6016 | not reached | not reached | not reached (contradiction) | not reached | not reached | not reached | 1.0407 | 1.201 / 0.839 / 1.083 |
| grJ (gold-aware stochastic keep) | 0 | 0.067 | -0.1207 | 8.41 | not reached | not reached (contradiction) | not reached | not reached | 21.13 | 0.2652 | 0.185 / 0.284 / 0.327 |
| grJc (grJ, doc drop on contradiction only) | 0 | 0.144 | +0.0018 | 4.10 | not reached | not reached (contradiction) | not reached | not reached | 6.84 | 0.2544 | 0.191 / 0.235 / 0.337 |

<!-- tables:end -->

## Still running

Nothing is running. Label-validity / short-row runs (11-19) are all finished. slurm 5204547 (w0 r05to50 on `kl16k_x3p2`, 720 steps) is on user hold by Prasann and has not
run. Re-score from scratch: `bash research/token_routing/early_loss_proxy/keepfrac_len/finalize.sh`, `gold_table.py`,
`ablation_16k.py --root <dir with kl16k, kl16kabl, kl16kx>`.

## Limits

- Proxy, not generation evals: teacher-forced answer CE of the dense forward on 48 held-out rows per task; it ranks
  arms (Spearman 0.83-0.95 vs final scores, `records/early-dense-loss-proxy.md`), understates the recipe's gain vs
  generation evals and cannot resolve final gaps below ~0.03-0.05 CE. Every 8k/16k keep difference between 20% and
  40% is below that.
- One epoch of 3,600 rows; budget-dependence (does 40% vs 20% flip with more data?) is untested here.
- Two seeds only on the deciding arms; `dft` (the reference) has one seed per length.
- The keep schedule is two cuts with cut 2 = 0.35 x cut 1; one-cut-at-a-time arms (`c1k10`, `c2k3p5`, ...) exist in
  `launch_kl.py` but were not run.
- Memory: router-arm peaks are flat in keep at every length here; this study says nothing about flat-memory
  training at 256k+.

## Files

- Code: `research/token_routing/early_loss_proxy/keepfrac_len/` — `launch_kl.py` (arms; `--local` for jsteinhardt),
  `analyze_kl.py` (matched-loss scoring), `summarize_kl.py` (headline metrics + chart), `paired_se.py`,
  `make_tables.py`, `record_results.py` (`ops result add` rows), `score_all.sh` / `finalize.sh`.
- Scored: `keepfrac_len/kl_results.json` (all curves and levels), `kl_summary.json` (the tables above),
  `q32k_scored.json`, `q64k_scored.json`, `INTERIM_q32k.md`; chart `keepfrac_len/keepfrac_vs_length.png`; hub block
  `keepfrac_len/hub_keeplen_block.html`; log entries `keepfrac_len/experiment_log_entry.json`.
- Raw run JSONs + logs: weka `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/early_loss_proxy/results/kl<L>/
  elp-kl<L>-<arm>-sd<seed>.json`, mirrored to `s3://ai2-llm/checkpoints/prasanns/early_loss_proxy/results/kl<L>/`
  (`../fetch_beaker_results.sh kl<L>`); quick reads on mooney `/data/prasann/early_loss_proxy/results/q{32,64}k/`.
- Pools / held-out: mooney `/data/prasann/early_loss_proxy/shards/kl*`, `heldout_kl{,s}.npz`; weka copies under
  `.../prasanns/early_loss_proxy/` (staged by Beaker `elp-stage-weka-b47e`, 01M7NW1V51Y38BC14Q2KXR6MXX).
- 16k gold-drop ablation, outlier label validity and the short-row baseline: see those sections (scripts
  `ablation_16k.py` -> `ablation_16k.json`, now including the three `gdnr` arms; `ab_compare.py`, `curves_lib.py`,
  `sim_outlier_label.py`, the topic-map and pool builders).
- Numbers recorded with `ops result add --check` (experiment `tokdrop_lr`, metrics `heldout_ce_dense_final`,
  `flop_x_vs_dft_at_mid_ce`, `flop_x_vs_dft_at_mid_ce_worst_task`; run ids `elp-kl<L>-<arm>-sd<seed>-probe`). `--check` warnings, all investigated: every row has eval_size 48
  / 144 (flagged above); the "deviation vs prior pass" warnings on q32k / q64k rows are two different arms that ran
  back to back in one slurm job (e.g. 5201920 = 40% then 2.5%), which ops reads as two passes of one training run. The
  same holds for the three `gdnr` arms (one slurm job, 5204584).
