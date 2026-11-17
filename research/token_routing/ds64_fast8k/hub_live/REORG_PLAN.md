# Hub reorganisation plan (2026-11-17)

Artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (private). Source of the old page: version 1794947058-c158
(717 KB, 16 parts, flat nav, one part shown at a time by a hash router). No result, number, chart or table is removed.

## Inventory of the old page (part id, nav tag, question it answers)

| id | old tab | question | tables / charts |
|---|---|---|---|
| summary | 1 | where do things stand? | 1 / 0 |
| log | log | what is running, what was tried? (filterable) | log list + compute snapshot |
| sft4 | 2 | SFT 4-task: Pareto, FLOPs-to-match, LoRA x routing | 7 / 8 |
| sft11 | 3 | SFT 11-task, 64k, gain vs length, 128k-256k | 0 / 7 (+ log figs) |
| models | 3b | does it generalize to Qwen3 and other sizes? | 3 / 8 |
| cpt | 4 | long-document CPT on OLMo 3 7B | 0 / 3 |
| memory | 5 | max trainable length, wall-clock | 2 / 5 |
| mechanism | 6 | where the gradient lives, router quality | 8 / 10 |
| gradmaps | 6b | per-token, per-layer gradient pictures | 0 / 0 (66 images, 5 details) |
| selection | 6c | which tokens to keep (rule ablations) | 0 / 3 (+ log figs) |
| ruledout | 7 | what did not work | 3 / 3 |
| next | 8 | ideas with a >=5x ceiling | 2 / 1 |
| priorwork | 8b | how it compares with published methods | 0 / 1 |
| gpuaudit | 8c | where the GPU-hours went | 3 / 6 |
| autoresearch | 8d | auto-research loop (live, rewritten hourly) | built from loop_state.json |
| provenance | 9 | data, code, caveats, detail pages | 2 / 0 |

## New structure (every old id kept; the nav is built at run time from `data-group` / `data-order` on each `.part`)

- **Overview** (`overview`, new landing): bottom line, five numbers, what is running now (live from the auto-research state), map of the page.
- **1 Best method and how it is measured** (`g-best`): `summary` Verdicts, `sft4` SFT 4-task.
- **2 Scale and length** (`g-scale`): `sft11` 11 tasks / 64k / length, `models` other models and sizes, `cpt` continued pretraining.
- **3 What was tried and what is next** (`g-tried`): `autoresearch` (live), `selection`, `ruledout`, `next`, `priorwork`.
- **4 Mechanism and analysis** (`g-mech`): `mechanism`, `gradmaps`.
- **5 Cost and infrastructure** (`g-cost`): `memory`, `gpuaudit`.
- **6 Log and provenance** (`g-log`): `log`, `provenance`.

## Cleanup rules applied by `reorg_hub.py`

1. Two-level nav (group, then section) on a sidebar; on a phone a chip row of groups plus a second row of that group's sections. "On this page" jump chips per section from its H2s (stable ids `<part>--<slug>`).
2. Tables with 12 or more rows go inside `<details>` ("Table: ..."); notes longer than 450 characters become `<details>` with their first sentence as the summary, unless they carry a warning sign. Notes that start with "Verdict" move to the top of their section.
3. Headings shortened, eyebrows show the group and the old tab number ("was 8d"). Part order follows the nav, not the DOM.
4. The auto-research part is rebuilt by `build_autoresearch_tab.py` (see README): summary, FLOPs-to-reach table, ideas grouped by family with their arms nested, chart limited to references plus top arms, validation, scoring, log newest first, pilot collapsed.
5. Verification: text blocks, ids, canvases, figures, images, tables and script data hashes compared before and after (`verify_reorg.py`).
