# random_doc_drop handoff (branch claude/random-doc-drop-nogold)

Question (Prasann 2026-11-18): does RANDOM whole-document dropping at the input, like gold drop, help on tasks with
no gold documents (reorder / oolong / grouping), with the label rewritten for the kept documents?
Record: `records/random-doc-drop-no-gold.md`. Experiment `tokdrop_lr`.

## Data (mooney:/data/prasann/early_loss_proxy, staged to weka .../prasanns/early_loss_proxy, Beaker 01M7RME6M3DNWGKJRMWWKKWMXF)
- `shards/ng16k`: t0..t2 = reorder / oolong / grouping, 1,200 rows each, cap 16,384, means 14.4k / 14.2k / 13.6k
  tokens (`ng16k_stats.json`). No gold sidecar.
- `heldout_ng.npz` set `ng16k`: 48 rows/task (eval_size 144). reorder = heldout_v3 gen16k rows (no shared book);
  oolong = item-disjoint EVAL split (1.3% of held-out item lines also in training); grouping = n88 rows 3700+
  (disjoint rows, but 89% of held-out abstracts also occur in some training row: the OpenAlex pool is reused).
  0 held-out prompts equal a training prompt.
- Variant pools `ng16k_rdd{A,B}_s{0,1}`: per slot, 15% whole, else keep r~U(.05,.5) (A) / U(.25,.75) (B) of the
  documents (oolong: items) with the label rewritten (`relabel.py`). Token keep: A 0.39-0.40, B 0.57-0.58.
  oolong: 47 rows of A_s0 fell back to whole (degenerate subsets after 20 redraws).
- `ng16k_rddAnr_s0`: NO-REWRITE control: the same kept sets as rddA_s0, the dropped documents' token spans cut out of
  the full row, question / answer / ids / oolong preamble N untouched (token keep 0.406).
- Loader order is identical across ng16k and all variant pools (checked, `build_ng_pools.py order`).

## Commands
- Build: `build_ng.sbatch` (STAGES / VARIANTS / NOREWRITE / ORDER env) on mooney, code staged with ctc included.
- Launch: `python3 random_doc_drop/launch_ng.py --arms dft,k20,rddA,rddB,rddAnr --seeds 0` (Beaker 2 x H100).
- Fetch: `bash ../fetch_beaker_results.sh ng16k`; score: `python3 analyze_ng.py --root <dir>/ng16k --out ng_results.json --png ng_loss_vs_pf.png`.

## State (2026-11-18 ~19:00 PT): DONE
- Seed 0: dft, k20, rddA, rddB, rddAnr. Seed 1: k20, rddA, rddAnr (pools rddA_s1 / rddB_s1 / rddAnr_s1 are built and
  staged). All Beaker runs have finished and are marked done in ops; 52 rows were added with `ops result add --check`.
- Verdict (record): random doc drop is not reliably ahead of the recipe on no-gold tasks. Savings at own lowest loss
  vs k20 are 1.25x / 0.90x for rddA. Reorder is behind at matched FLOPs, grouping is level, and oolong is ahead but
  noisy. The no-rewrite control beats the rewritten arm on every task (-0.016 / -0.014 mean CE).
- Open, if wanted: rddB seed 1 (its pool is staged); deeper budgets (multi-epoch / bigger pool) to see whether the
  shallow-level advantage survives; a fresh OpenAlex pool for grouping to remove the 89% held-out abstract overlap.
