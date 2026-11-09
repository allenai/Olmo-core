---
experiment: ffndrop
title: Drop-CPT — CPT with random per-token null FFNs, then routed-FFN (ffnmoe-t10) SFT, Qwen3.5-4B
status: active
updated: 2026-10-31
branch: prasann/landmark
launcher: research/ffn_compute/ffndrop/launch_ffndrop_cpt.py
records:
  - records/ffndrop-cpt-plan.md
  - records/flop-scaling-report-2026-09-02.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ffndrop_cpt/pipeline_state.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core-legacy/debug/ffndrop_cpt/pipeline.log
  - research/ffn_compute/ffndrop/LAUNCH_LEDGER.tsv
  - research/ffn_compute/flop_scaling/LAUNCH_LEDGER.tsv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/results35.csv
wandb_group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/fdcpt-q35-4b
next_step: Launch dense SFT from the drop-CPT base on contra 28M / nq 32M (FS35_BASE=<export> FS35_BASE_TAG=-bfdrop launch_grid35.py --arms dense) to separate the drop effect from the CPT data
---

# ffndrop

## Question
Routed FFN (nested-width router + null rung) was never compute-optimal against dense SFT. If a one-time
CPT teaches the base to tolerate randomly dropped FFNs (AdaMoE-style null expert, no router), does routed
SFT from that base reach the dense frontier at matched SFT FLOPs?

## Decisions
- 2026-09-29 (Prasann): the drop-CPT checkpoint is a new BASE, so its CPT cost is amortized and not charged to per-task SFT FLOPs. Also report the break-even.
- 2026-10-31 (agent, weka_cleanup): the drop-CPT checkpoint `ctc_suite/ckpts/fdcpt-q35-4b-drop75l12-u1B` (pipeline.py CPT_RUN, the FS35_BASE for the next step) was DELETED by mistake in the 10-30 weka cleanup (misclassified as unlabeled; see weka_cleanup STATUS). No other copy. The next step (dense SFT from the drop-CPT base) needs the CPT re-run first: launch_ffndrop_cpt.py drop75l12 from q35-4b-base-markerfix on softdetach_cpt/shards/cpt_u1B (both intact), 1 node x 8 H100, >=5 h. Existing bfdrop SFT results are unaffected.
- 2026-09-29 (agent): arm `drop75l12`. Per row r ~ U[0, 0.75]; tokens skip each FFN of layers >= 1 with prob r; each (row, layer) drops the whole row's FFN with prob 0.125; no 1/(1-p) rescale. CPT on the 1B-token `softdetach_cpt/shards/cpt_u1B` shard, base `q35-4b-base-markerfix`, lr 3e-5, 8 x 64k rows/step.
- 2026-09-30 (Prasann): dense-CPT control CANCELED ("you don't need the control"). The comparison uses the existing fs35 dense/t10 points from the unmodified base. Scope stays at "one or two settings": contra 28M, nq 32M.
- 2026-10-03 (agent): verdict for this round. Drop-CPT lifts routed SFT only at 16k/32k and stays below the dense frontier. Confounded by the long-context CPT data.

## Current best numbers
SFT = routed `ffnmoe-t10` (layers 12+, target 0.10), routing ON at eval, fs35 ladders 2k/8k/16k/32k.
Means pool the 4 rungs (approximate SE).

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| base -> routed t10 (515 PF) | contra 28M / mean | f1=0.808 | 4x500 | 0.009 | unknown | results/flop_scaling/results35.csv |
| **drop-CPT -> routed t10 (~541 PF)** | contra 28M / mean | f1=0.824 | 4x500 | 0.009 | unknown | records/ffndrop-cpt-plan.md |
| drop-CPT vs base, routed | contra / 32k | f1=0.687 vs 0.627 | 500 | 0.021 / 0.022 | unknown | records/ffndrop-cpt-plan.md |
| dense SFT 28M (674 PF), 3 rungs only | contra / mean 2k-16k | f1=0.924 | 3x500 | 0.007 | unknown | results/flop_scaling/results35.csv |
| base -> routed t10 (572 PF) | nq 32M / mean | f1=0.853 | 4x600 | 0.007 | unknown | results/flop_scaling/results35.csv |
| **drop-CPT -> routed t10 (~586 PF)** | nq 32M / mean | f1=0.879 | 4x600 | 0.007 | unknown | records/ffndrop-cpt-plan.md |
| drop-CPT vs base, routed | nq / 32k | f1=0.778 vs 0.700 | 600 | 0.017 / 0.019 | unknown | records/ffndrop-cpt-plan.md |
| dense SFT 16M / 32M (379 / 758 PF) | nq / mean | f1=0.878 / 0.907 | 4x600 | 0.007 / 0.006 | unknown | results/flop_scaling/results35.csv |
| drop-robustness probe, base -> drop-CPT | cpt_dev CE penalty at r=0.5 | +2.73 -> +0.20 nats | 32 rows ⚠ eval_size=32 | n/a | no-cot (TF CE) | records/ffndrop-cpt-plan.md |

Reading: nq sits ~0.017 below log-interpolated dense at ~586 PF; base-routed was ~0.04 below. Contra on
2k-16k is 0.870 vs ~0.895 interpolated dense, the same as base-routed. All of the gain is at 32k.

## Next step
Dense SFT from the drop-CPT base on the same two settings. If dense SFT also gains at 16k/32k, the effect
is the CPT data and not drop-robustness. Not launched: no `dense ... -bfdrop` row exists in
`research/ffn_compute/flop_scaling/LAUNCH_LEDGER.tsv`, which was last written 2026-10-03 04:27. The `ffndrop` variant has been
ported into the setA recipes on `prasann/flopdrop_clean` (`recipes/train/setA/README.md`,
`--variant ffndrop`). That branch is local only.

## Open issues / traps
- Confound: there is no dense-CPT control, so the 16k/32k gains may come from the long-context CPT data (cpt_u1B) and not from drop-robustness.
- The FLOP meter charges dense FFN cost on the drop arm. CPT is amortized, so this only affects wall-clock. Drop-CPT SFT meters at actual/dense 0.900/0.885, vs 0.880/0.876 for the base runs, so it costs about 1-2% more FLOPs.
- The drop is training-only. The export is a plain dense checkpoint with no router keys.
- Gantry `UnpushedChangesError`: another session's unpushed HEAD (ec8242072) made every pipeline launch fail on 2026-10-03 04:23. See `memory:gantry-needs-pushed-commit`.
- Dev-loss probe ⚠ eval_size=32 rows (~2M tokens). Treat it as a direction, not a precise penalty.
- Mechanism: `olmo_core.nn.ffn_token_drop`. Trainer flags are `--ffn-drop-max-rate/--ffn-drop-layer-prob/--ffn-drop-start-layer`.
- Background, not superseded: `memory:moe-vs-null-routing-ffn-time` (null routing beats top-k MoE on FFN time), `memory:adamoe-vs-nested-experts`, `memory:nested-ffn-moe-task`, `memory:flop-scaling-ffn-kv-study`.

## Verified vs unverified
- Verified 2026-10-12 against `debug/ffndrop_cpt/pipeline_state.json` and `pipeline.log`:
  - CPT finished with exit 0 at 10-03 04:23.
  - The dev-loss eval was submitted (01M40R7VV0W7ATH7K2T2E0DT1R).
  - Both SFTs finished with exit 0: contra 01M40R8PBE5C192A0TFS7PE9NY, nq 01M40R9FN39TNRVGPJKXV3CM2S.
  - Both evals finished with exit 0 (01M40SF6HZKBNWY7PD2FB257JA, 01M40SGAGAKFAS5CSMHXENJDPD). PIPELINE DONE at 05:28.
- Verified: the dense-CPT arm was CANCELED (ffndrop ledger). The base-routed t10 points (.808 / .853) and the dense points (.924*, .878, .907) match `results/flop_scaling/results35.csv`.
- Verified absent: no dense SFT from the drop-CPT base appears in any ledger.
- UNVERIFIED: the drop-CPT SFT numbers (.824 / .879 and the per-rung values) come only from the record (commit ddc22f44b). No `bfdrop` row exists in `results35.csv` and no eval JSON is in the repo, presumably because they are on weka.
- UNVERIFIED (cot label): the fs35 native ladder eval does not state a cot mode in its drivers.
- UNVERIFIED — confirm with Prasann: is dense-SFT-from-drop-CPT still wanted, or does the setA port replace this line of work?
