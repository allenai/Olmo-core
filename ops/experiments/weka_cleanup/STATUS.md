---
experiment: weka_cleanup
title: INFRA — weka / S3 / node-local checkpoint storage cleanup (redundant stepN dirs, relay duplicates, ZFS snapshot lag)
status: active
updated: 2026-11-05
branch: prasann/flopdrop_clean
launcher: /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/ops/experiments/weka_cleanup/launch_weka_apply.py
records:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/records/weka-checkpoint-cleanup.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/ops/experiments/weka_cleanup/manifests_20261105/apply_20261105.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/ops/experiments/weka_cleanup/manifests_20261105/APPROVED_W1_W2_W3.tsv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/ops/experiments/weka_cleanup/manifests_20261030/apply_20261030.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/ops/experiments/weka_cleanup/manifests_20261030/APPROVED_A_B_C.tsv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/storage_cleanup/DELETION_LOG.tsv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/storage_cleanup/CLEANUP_PLAN.md
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/weka_cleanup/manifest_relaydup_20261012.tsv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/storage_cleanup/weka_stepn_audit.tsv
artifacts:
  - https://claude.ai/artifact/QCgqoE2w5QVKXyecTYU9ds
wandb_group:
next_step: Wave 2 (11-05) done, weka 10.84 TiB. <10 TiB needs Prasann + the FLOP-drop sessions to pick among recipe-family seeds (1.84 TiB) or FLOP-drop dense baselines. ffndrop drop-CPT rebuild still Prasann's call.
---

# weka_cleanup

## Question
Not an experiment: keep `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/` (and S3 relay copies and
Berkeley node-local `/data`) from filling, by deleting only provably redundant checkpoint copies with
auditable, two-stage (plan → literal manifest → validated apply) tooling.

## Decisions
- 2026-07-28 (Prasann): deletes name exact paths, never globs; two-stage plan/apply; manifests archived to `s3://ai2-llm/checkpoints/prasanns/_inventory/manifest_<phase>[_TAG].txt`.
- 2026-07-28 (agent): Phase 1 (all stepN in 74 `ctc_suite/ckpts` runs that have `model_and_optim/`) 11.22 TB + Phase 2 (keepfinal on 49 top-level runs) 3.04 TB → 21.37 → 7.2 TB. `export_olmo_to_hf.py:latest_step_dir()` falls back to `model_and_optim`.
- 2026-07-28 (agent): optimizer-strip of the 97 surviving top-level finals (~3.1 TB) NOT done — needs a GPU load-and-resave per run, irreversible, and outlier_datascale_sweep evals were owed.
- 2026-08-29 (agent): root cause of regrowth to 23 TB = `beaker_ctc_suite.py` never passed `--no-final-checkpoint` (weights stored twice); fixed as default-suppress with `--keep-final-checkpoint` opt-out (present in the launcher now).
- 2026-09-01 (agent): node-local deletes on mooney/horton/lorax/thidwick; ZFS weekly snapshots defer reclaim up to ~4 weeks — never promise immediate `df` gains; destroying snapshots on shared nodes is an infra decision (sysadmin request drafted for cubbins 2.63 TB USEDSNAP).
- 2026-09-08 (agent): S3 `ckpt_relay` cleanup (90 entries) + weka round logged in DELETION_LOG.tsv.
- 2026-10-12 (agent, via storage-audit skill): round 3 — weka had regrown to 28.8 TB; 96 stepN deleted (`outlier_lengthmix/ckpt_relay` 85 / ~4.5 TB, top-level keepfinal 10 / 0.40 TB incl. setA 32k/256k, `ckpt_slm_p8k4000` 1 / 0.05 TB) + relay-dedupe of 85 `ckpt_relay` runs that duplicated `ctc_suite/ckpts/lmx-*` (1.50 TiB); sneetches /data 58 stepN 2.20 TB + 6 approved dirs 1.20 TB. Weka ends ~22.3 TB.
- 2026-10-12 (Prasann): bf16 re-save of finals (would ~halve weka) declined for now; S3 offload tooling NOT built while Prasann checks AI2 S3 policy.
- 2026-10-30 (agent, storage-audit): weka had regrown to 43.02 TiB in 18 days (olmo3_cptdrop 11.6 TiB, FLOP-drop arms 8.4 TiB). Tiered by experiment status from the run registry (`ops runs`) + INDEX.md: A done/retired 13.1, B paused + unlabeled 7.7, C olmo3_cptdrop non-final stepN 5.1 + per-run base/data copies 1.8 (verified on S3 `_transfer/olmo3_cptdrop/`), D/D2 FLOP-drop non-dense arms 2.1 / 3.7.
- 2026-10-30 (Prasann): approved A, B ("the 7.7") and C. NOT approved: D, D2.
- 2026-10-30 (agent): agent-excluded from A/B before apply (0.37 TiB): `ctc_olmo3/bases`, `*/shards`, `*/hf`, `pyserini_cache`, setA 32k/256k dense, singletask ladders and other data dirs. New tooling (`weka_apply.py` + `launch_weka_apply.py`, modes wholedir/step/copydir, verify-then-apply, Beaker dataset input) replaces the archived debug/weka_cleanup scripts (commit 33349d64b). Verify job 01M68AXZT2FTRKWT0CDQT18EEM: 1129 OK / 4 refused (no checkpoint files). Apply job 01M68B15W8HDDRFQBSCJ5P8Z90: 1129 deleted, 27.2 TiB, 0 errors. Post index 01M68C0MBS5XA2NSVXQ7EG4ZH5: 43.02 -> 16.04 TiB; bucket free 7.82 -> 36.55 T.
- 2026-10-31 (agent): MISTAKE in that round. Tier B's "unlabeled" bucket (0.7 TiB) held checkpoints of experiments my classifier did not map to Done/Paused, including ACTIVE ones: `fdcpt-q35-4b-drop75l12-u1B` (ffndrop's drop-CPT base, the only copy; its next step SFTs from it) and 6 old ds64_fast8k arms (`f8k-outlier-{dense-8M,dense-16M,dense-20M,cc00-16M,cpi33-16M,kvgb50-16M}`, idle 45d, evaluated). No S3 / node-local copy. Re-creatable: launch_ffndrop_cpt.py from q35-4b-base-markerfix on softdetach_cpt/shards/cpt_u1B (both intact), 1 node x 8 H100, >=5 h.

- 2026-11-05 (agent, wave 2): weka 16.79 TiB. Stricter tiering (tiers2.py): candidate only if the experiment is positively Done/Paused/retired, idle >=14d, unused by any launch in 14d, and not named in active code/STATUS; unknown = keep. That left 0.70 TiB (W1, done-experiment dense baselines kept on 10-30). Plus W2 = olmo3_cptdrop stepN in 63 runs that also hold model-only `model_and_optim` (2.90 TiB; weights kept, resume-with-optimizer lost) and W3 = FLOP-drop non-dense arms last used 8-14d, minus the recipe/lead families `r20to7*` and `otr10M7` (133 dirs, 2.36 TiB). Prasann approved W1+W2+W3. New applier mode `stepmo` (requires non-empty run/model_and_optim/.metadata; fixture-tested). Verify 01M6QTPP2ZQTPTHRQ1CW9V9BC5: 227/227 OK. Apply 01M6QTSQ59BB1KHTH4TSTF1SWY: 227 deleted, 5.96 TiB, 0 errors. Post index 01M6QV097HD7TX9JKADCBGD4H0: 16.79 -> 10.84 TiB; bucket free 32.44 -> 37.86 T. Survivors checked: bases, xlong5, hpqaret, sparsity_scale dense, 126 r20to7 + 36 otr10M7 + 180 dense ckpts, olmo3_cptdrop 1.86 TiB of model-only finals.

## Current best numbers
Storage state (no eval metrics):

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| weka prasanns | total | 21.37 TB (07-28) → 7.2 TB → 23 TB (08-29) → 28.8 TB → ~22.3 TB (10-12) | — | — | — | memory:weka-ckpt-cleanup, memory:beaker-launcher-missing-nofinalckpt |
| weka prasanns | total (10-30) | 43.02 TiB -> 16.04 TiB (27.2 TiB / 1129 dirs deleted) | — | — | — | manifests_20261030/apply_20261030.json, storage-audit index 01M68C0MBS5XA2NSVXQ7EG4ZH5 |
| weka prasanns | total (11-05) | 16.79 TiB -> 10.84 TiB (5.96 TiB / 227 dirs, wave 2) | — | — | — | manifests_20261105/apply_20261105.json, index 01M6QV097HD7TX9JKADCBGD4H0 |
| weka bucket | free | 3.8 T → 9.8 T after 10-12 round; 7.82 T -> 36.55 T after 10-30 | — | — | — | memory:weka-ckpt-cleanup |
| DELETION_LOG 2026-10-12 | rows | weka:phase3 85, weka:relay-dedupe 85, weka:phase2 10, weka:phase4 1, sneetches:/data 64 | — | — | — | debug/storage_cleanup/DELETION_LOG.tsv |
| node /data USEDSNAP (09-01/09-08) | deferred reclaim | cubbins 2.63 TB (0 avail), mooney 5.81 TB, horton 9.36 TB | — | — | — | debug/storage_cleanup/sysadmin_request.md |
| horton → S3 | throughput | 73 MiB/s up / 81 MiB/s down (aws-cli v1) | — | — | — | memory:weka-ckpt-cleanup (s3_speedtest.sh) |

## Next step
10-30 round done (16.04 TiB). Open for <10 TiB: FLOP-drop arms D/D2 (manifests ready, not approved) or bf16 re-save. Prasann: re-run the ffndrop drop-CPT? Earlier open items: (1) S3 offload of cold finals vs keep on weka (AI2 S3 policy); (2) bf16 re-save (declined for
now); (3) optimizer-strip of the old top-level keepfinal finals (~3.1 TB as of 07-28; re-measure first — some were
deleted since). Routine: run the `storage-audit` skill (`--refresh`) before large writes; repeat plan/apply rounds as needed.

## Open issues / traps
- Tiering trap (10-30): only delete whole run dirs whose experiment is POSITIVELY known Done/Paused/retired. An 'unclassified' bucket must default to KEEP, and anything mapped to an Active experiment (INDEX.md) is never a candidate, however idle. The 10-30 tiers.py put unmatched names (incl. ffndrop, ds64_fast8k) into a delete tier.
- Growth rate 10-12 -> 10-30 was ~1.1 TiB/day: olmo3_cptdrop keeps several 82 GiB full stepN per run plus a 27 GiB base copy and data copy per arm dir; FLOP-drop arms are fp32 ~18 GiB each.
- Naming collision: memory index says "Phase 3 (optim-strip, ~3.1TB) still open", but the 2026-10-12 manifests also use `phase3` for the `outlier_lengthmix/ckpt_relay` stepN round. They are different things — the optim-strip has never run.
- Do NOT delete: `ctc_suite/bases/*`, `ctc_olmo3/bases/*-fixmark`, `xlong5_2k256k_qwen35/*` (weka is the only copy), `contra_mix_*_10k_128k` shards, `_eval_bundle*`, `ctc-olmo3-7b-hpqaret-*` (retrain owed; the 5 hpqaret runs keep stepN on purpose).
- Freshness-guard trap: deleting a stepN bumps its PARENT mtime; the applier pre-computes parent verdicts once. Resuming an interrupted apply needs a lowered `FRESH_MIN`.
- Plan-tool bug (fixed 10-12): a non-numeric `step*` dir put a bare RUN DIR in the manifest; all planners now use `-regex '.*/step[0-9]+'`.
- Weka is not mounted at Berkeley: plan/apply run as 0-GPU gantry jobs with the worker shipped inline (no pushed commit needed) — memory:gantry-needs-pushed-commit.
- ZFS: node-local deletes free nothing for up to ~4 weeks — memory:zfs-snapshots-delay-node-local-reclaim. `zfs destroy -nv` dry-runs fine; real destroys need sysadmins.
- Possible regrowth sources still writing stepN: the setA recipe (its 32k/256k runs had stepN deleted on 10-12) and any `--keep-final-checkpoint` launch.
- `/accounts` 20 GB quota is a separate problem — memory:accounts-quota-avoid-writes.

## Verified vs unverified
- Verified 2026-10-12: `beaker_ctc_suite.py` now passes `--no-final-checkpoint` unless `--keep-final-checkpoint` (lines ~191/281); DELETION_LOG.tsv row counts per pool/date (09-01: 330, 09-08: 369, 10-12: 245); the 10-12 tooling files (dedupe_relay.py, launch_dedupe_relay.py, dir_cleanup.sh, manifest_relaydup_20261012.tsv); sysadmin_request.md USEDSNAP figures; record's "Not done: optimizer strip" text.
- Read only in memory: all TB totals and bucket-free figures (DELETION_LOG `bytes` is blank on many rows, so it under-counts and cannot confirm them), sneetches 2.20 + 1.20 TB, S3 throughput.
- UNVERIFIED — confirm with Prasann: whether the cubbins/mooney/horton snapshot request was ever sent/acted on; whether the optimizer-strip is still wanted (and its current size); whether the setA launcher should also default to `--no-final-checkpoint`.
