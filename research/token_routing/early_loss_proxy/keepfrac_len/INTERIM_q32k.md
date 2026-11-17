# Interim: quick 32k read (tag q32k), scored 2026-11-17 15:40 PT

**Verdict (provisional, one seed, 40 of 225 steps, eval_size 72):** at 32k there is no sign that the best keep falls
below the recipe. 40%/14% and 20%/7% tie (held-out CE 0.370 vs 0.380 at step 40, ±0.015 each) and are the only router
arms that reach the second matched-loss level; 10%/3.5% and below plateau higher (0.440 / 0.448 / 0.469) and stall on
contradiction. At the shallow level all router arms reach (CE 0.463) every keep from 40% to 5% is 1.9-2.7x fewer FLOPs
than dense full fine-tune, i.e. the early read cannot separate them there; it separates them by how deep they get.

⚠ eval_size = 72 (24 rows per task, `heldout_v4` set `gen32k`), well below 500: SE of a mean held-out CE ≈ ±0.015.
The proxy cannot resolve gaps under ~0.03-0.05 CE and understates gains vs generation evals: rank only.

Setup: pool `mtb64k_r32k` (the 960 32k rows of `mtb64k`; NOT the `kl32k` pool of the full wave), 16 rows/step, 2 x H200
on mooney (jsteinhardt), probes at steps 0/10/20/30/40, one seed, code `code_7c6b5d877`. Score = analyze_kl.py
`--ref dft` (dense full fine-tune); levels = dft's own mean CE at steps 10-40.

| arm (keep cut1/cut2) | FLOPs/step vs dense | held-out CE @ step 40 (±SE) | PF spent | x dft @ CE 0.463 | x dft @ CE 0.379 | worst task (CE @40) | contra / outlier / qdmatch_nq CE @40 |
|---|---|---|---|---|---|---|---|
| dft (dense full FT) | 1.00 | 0.285 ±0.016 | 585 | 1.00 | 1.00 | qdmatch_nq 0.431 | 0.187 / 0.236 / 0.431 |
| ref (dense LoRA) | 0.73 | 0.255 ±0.015 | 436 | 1.44 | 2.05 | qdmatch_nq 0.315 | 0.197 / 0.254 / 0.315 |
| k40 (40% / 14%) | 0.27 | 0.370 ±0.017 | 160 | 2.63 | 1.98 | qdmatch_nq 0.457 | 0.388 / 0.265 / 0.457 |
| k20 (20% / 7%, recipe) | 0.21 | 0.380 ±0.015 | 126 | 1.92 | — (0.380, just misses) | qdmatch_nq 0.457 | 0.444 / 0.239 / 0.457 |
| k10 (10% / 3.5%) | 0.19 | 0.440 ±0.015 | 110 | 2.65 | — | contradiction 0.565 | 0.565 / 0.293 / 0.462 |
| k5 (5% / 1.75%) | 0.17 | 0.448 ±0.014 | 103 | 1.87 | — | contradiction 0.566 | 0.566 / 0.310 / 0.469 |
| k2p5 (2.5% / 0.9%) | 0.17 | 0.469 ±0.014 | 99 | — (never reaches 0.463) | — | contradiction 0.575 | 0.575 / 0.361 / 0.471 |

Step-0 CE is 1.216 for every arm. Per-task at the contradiction level reached by dft at step 20 (0.338): only dft/ref
reach it; k40 ends at 0.388, k20 at 0.444, k10-k2p5 at 0.57-0.58 (contradiction is where low keeps break first).
Below 20%/7% the per-step FLOP saving flattens (0.21 -> 0.17x dense: the dense blocks 0-6 and the head dominate), so
lower keeps buy little compute and lose depth. Dense LoRA beats dense full FT at this short budget (0.255 vs 0.285, 2x
at CE 0.379); peak memory: dft 63.9 GiB, ref 40.2, every router arm 37.4 (the router's peak does not move with keep
at 32k; memory values from this run are after the 11-17 13:10 fix).

Files: mooney `/data/prasann/early_loss_proxy/results/q32k/elp-q32k-{dft,ref,k40,k20,k10,k5,k2p5}-sd0.json` (+ `.log`,
`spec_<job>.tsv`), slurm jobs 5201844-5201847, 5201858 (all COMPLETED). Scored JSON:
`keepfrac_len/q32k_scored.json` (analyze_kl.py --tag-prefix q --lens 32k --ref dft).
