# Auto-research loop: brief for idea agents

You implement ONE idea for cutting the training FLOPs (and peak GPU memory) of long-context SFT on Qwen3.5-4B, run it
through the 50-100 step dense-loss proxy, and report numbers. The orchestrator (another Claude session) combines
ideas and decides what to scale up. Read this file, `README.md` beside it and the repo `CLAUDE.md` (cluster rules are
hook-enforced) before anything else.

## Goal (Prasann, 2026-11-16)
At least 10-20x fewer training FLOPs than dense to reach the same dense-forward held-out loss, and 5-10x lower peak
training memory (so 5-10x longer contexts fit). The current recipe is at 2-4x. Inference stays dense over the full
context. Small tweaks (1.2x) are not interesting; ambitious ideas are. FLOPs first, wall-clock is optimised later.
Preference: principled / end-to-end over heuristics; a learned router must not be warm-started from a heuristic.

## Setting
- Model `q35-4b-base-markerfix` (36 blocks, hybrid full attention + Gated DeltaNet), trainer
  `recipes/train/research_trainer/train_ctc_suite.py`, 16 rows/step, micro-batch 2, 2 GPUs (FSDP), sequence 16,384,
  one epoch LR schedule (linear warmup over the first 3% of steps, then linear decay).
- Pool `mtb16k` on the node (`/data/prasann/early_loss_proxy/shards/mtb16k`): contradiction, outlier, reorder,
  qdmatch_nq; rows at about 4k, 8k and 16k tokens; documents wrapped in marker tokens 248049 / 248050; gold-document
  sidecar beside it. Held-out probe rows `/data/prasann/early_loss_proxy/heldout_v2.npz` (sets suite4k, suite8k,
  suite16k). See `POOL_V2.md` beside this file once it exists for exact paths and stats.
- Current recipe (`RECIPE` in `launch_proxy.py`): whole-model LoRA r64, online ridge token router, all tokens run
  blocks 0-6, keep 20% of each document's body tokens after block 7 and 7% after block 16 (query, answer, prompt and
  markers always kept), LR 1e-5. Its per-step cost is about 0.23-0.31x dense; blocks 0-6 at full length are a floor.
- What was already tried at 8k on the old task set (read before designing):
  `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.claude/worktrees/gallant-robinson-adc54a/records/tokdrop-handoff-2026-11-09.md`
  (sections 4 and 5), and `/accounts/projects/berkeleynlp/prasann/ops/experiments/tokdrop_lr/STATUS.md`. Related
  experiment pages: `ops/experiments/{softdetach,pooled_kv,ffndrop,kv_route,learned_router}/STATUS.md`. Known trap:
  keeping ONLY the gold documents is a shortcut and collapses (contradiction 0.07 vs 0.89).

## The proxy (what you run)
`dense_probe.py` runs the unmodified trainer with your flags and, at step 0 and each probe step, scores the DENSE
forward (model.eval(): no dropping, adapters on) on the held-out rows (teacher-forced answer CE). Per step it logs
cumulative FLOPs from the flop meter, peak GPU memory (max over ranks) and wall-clock.

```bash
# 1. commit your change in YOUR worktree (git archive stages HEAD), then stage it on the node you were assigned
D=$(bash research/token_routing/early_loss_proxy/stage_code.sh <node> | tail -1)
# 2. launch (one slurm job per call; arms in one call run back to back). Registered with ops automatically.
python3 research/token_routing/early_loss_proxy/launch_proxy.py --code $D --node <node> --qos preemptive_high \
    --tag <your-idea-tag> --seeds 0 --steps 25,50,75,100 --max-train-s 600 \
    --arms myarm --custom 'myarm=softtoken,1e-5,<trainer flags>'
# 3. wait WITHOUT a foreground poll loop: run this with run_in_background=true and wait for the notification
timeout 2400 bash -c 'until grep -q "ELP_DONE\|FATAL\|CANCELLED" /net/<node>/data/prasann/joblogs/elp-<tag>-<jobid>.out 2>/dev/null; do sleep 20; done'; grep -a "dense-probe\|=== \|result\|Error" /net/<node>/data/prasann/joblogs/elp-<tag>-<jobid>.out | cut -c1-300 | tail -30
# 4. score against the reference (dense + dense LoRA live in results/v2 on mooney)
python3 research/token_routing/early_loss_proxy/score.py --results /net/mooney/data/prasann/early_loss_proxy/results/v2 \
    /net/<node>/data/prasann/early_loss_proxy/results/<your-idea-tag> --out /tmp/.../board.json
```
- `--custom NAME=VARIANT,LR,FLAGS`: VARIANT is the trainer `--variant` (`softtoken` for the router / drop paths, `full`
  for dense), FLAGS are appended to the trainer command (later flags override earlier ones, e.g. `--global-batch 8`,
  `--base-checkpoint ...`). Baseline flag sets are in `launch_proxy.py` (`RECIPE`, `LORA`, `ARMS`).
- `--max-train-s 600` stops a run (after a final probe) once training has taken 10 minutes; `--stop-ce X` stops once
  the mean held-out CE is <= X. Keep every run under 15 minutes of wall-clock. Cheap methods should run MORE steps
  (add probe steps, e.g. 25,50,100,150,200) since the score is FLOPs to reach the reference loss, not loss at step 50.
- The step-0 probe is cached per base (`step0_cache.json`); a run whose step-0 model differs from the base is detected
  and re-scored automatically.
- Smoke-test every code change first with 3-4 steps (`--steps 2,4 --minutes 8 --tag smoke-<idea>`); look at the log
  for the flop-meter line and the probe lines before spending a full run.
- GPUs: 2 per run, `--qos preemptive_high` on your assigned node (slurm caps the user at 8 GPUs there; extra jobs
  queue). Never more than 2 of your jobs submitted at once. Do not use berkeleynlp nodes. Raw results stay on the
  node under `/data/prasann/early_loss_proxy/results/<tag>/` (inspect lightly through `/net/<node>/...`).

## Score (what counts)
`score.py` prints per arm: held-out CE at the last probe, cost per step vs dense (`step/dense`), `x@k` = FLOP
speedup at the deepest reference level reached (cheapest dense-reference FLOPs to reach the reference's step-k loss
divided by your FLOPs to reach it), `xw@k` = the same requiring every task to reach its own reference level, peak
GiB and its ratio to the reference. A method only counts if it reaches a reference level; "cheap but never gets
there" is a negative result. eval_size is 192 held-out rows (16 per task and length), below 500: differences under
about two paired SEs (`d_vs_ref_se`) are noise. Seed 0 for screening; rerun seed 1 only for an arm that looks >= 5x.

## FLOP and memory accounting (must be right, or the result is void)
- FLOPs come from `src/olmo_core/train/callbacks/flop_meter.py` (`flop_meter/actual_pflops`): it prices each example
  at its own (compacted) length per block, with token exit / staged drops, LoRA adjustments and layer drop handled,
  and adds whatever the model accumulates in `model._flop_extra_actual` (extra passes, scoring, fits). If your method
  adds a forward or backward pass, skips the backward for some blocks or tokens, or changes which tokens run which
  blocks, make the meter charge exactly that (forward-only work is about 1/3 of forward+backward; say what you
  charged). Never report a FLOP saving the meter does not show. State the per-step cost formula in your report.
- Memory: `peak_gib` in the run JSON is `torch.cuda.max_memory_allocated` per training step, max over ranks; it
  excludes the dense probe. Report it at the same micro-batch as the reference.

## Rules
- Work only in your own git worktree; commit there (branch stays local, do not push, never touch `main` or
  `prasann/flopdrop_clean`). Minimal additive hunks: new flags default to off so every existing arm is unchanged.
- No CPU-heavy work, tests or training on the login node; node-local interpreter and `/data` paths in jobs only;
  no foreground sleep/poll loops; register launches (launch_proxy.py does). Run GPU unit tests, if any, via srun.
- `n` means corpus size (documents per row); write `eval_size` or "rows" for counts. Flag eval sets under 500 inline.
- Never fabricate or extrapolate a number. If something fails, say so with the error.
- Budget: about 90 minutes. Then stop and report, even if partial.

## Report (your final message, under 300 words, facts only)
1. One-line verdict: best arm, its cost per step vs dense, x@level, xw@level, peak GiB vs reference.
2. Table of every arm you ran: name, exact flags, steps, held-out CE, cost/step vs dense, x@k, xw@k, peak GiB, train s.
3. What the code change is (files, flags), the per-step FLOP formula charged, your worktree path, branch and commit.
4. What did not work and why; what you would try next; which other ideas it should combine with.

## Smoke config while the new pool is not on your node yet
Until `mtb16k` / `heldout_v2.npz` exist on the node, smoke-test on mooney with the old 8k pool by adding
`--data /data/prasann/oracle/shards/mt4f1952 --heldout /data/prasann/early_loss_proxy/heldout_v1.npz --sets suite8k:8 --seq-len 12288`
to `launch_proxy.py`. Real runs use the defaults (new pool, 16,384).

## Where to run (updated 2026-11-17 00:40)
- Berkeley: `launch_proxy.py --node sneetches|mooney` (8 GPUs at preemptive_high in total across all agents; the
  references live on sneetches: `/net/sneetches/data/prasann/early_loss_proxy/results/v2`). Pending GPU jobs on a node
  block CPU-only jobs there, so do not leave jobs queued on mooney.
- Lambda (2 x A100, ~2.3x slower, startup ~4 min): `launch_proxy_lambda.py`, see `LAMBDA.md`. Cheap arms only.
- Beaker (2 x H100, about recipe speed of H200 x 1.2, starts in 20-30 s once warm): `launch_proxy_beaker.py`, see
  `BEAKER.md`. Needs your branch PUSHED (`git push -u origin <your-branch>`; never main / prasann/flopdrop_clean) and
  merged with `origin/claude/early-loss-proxy-d6e886`. Arms that read files only on Berkeley `/data` are refused.
- Data and held-out rows: `POOL_V2.md`. Memory to report: `transient_gib` (peak minus pre-step baseline).

## Priority (Prasann, 2026-11-17 12:00): the 16k held-out loss, and ideas that gain more as rows get longer
- Judge every arm first on the 16k held-out rows (`python3 iso_set.py set/suite16k`; with `heldout_v3.npz` the sets
  `gen16k` = fresh in-range 16k rows and `rung16k` = uncapped 16k-rung eval rows), then on all lengths.
- The plain recipe is 2.1-2.7x vs dense LoRA at 16k; short-heavy pools add only ~1.2x there. Prefer levers whose
  saving grows with length (attention is quadratic in length; the kept-token share can fall as rows grow).
- Memory (Prasann 11-17 12:15): savings at 16k do not matter (transient ~3 GiB vs ~12 GiB of weights + optimizer
  state). What matters is an algorithm whose peak stays nearly flat at 256k, 1M tokens or longer: report
  `transient_gib` per 1k tokens and say what would hold the peak at 256k+.
