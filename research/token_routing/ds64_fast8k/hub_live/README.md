# Live tokdrop_lr hub

`hub.html` is the page published at https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (private). Reorganized 2026-11-17
(old flat 16-tab page: version 1794947058-c158, kept in `REORG_PLAN.md` terms). **This file is the source: edit it
(or run the small builders below) rather than regenerating from `../make_overview_page.py`**, which has none of the later tabs.
`hub.html` is the authored page (no `<!doctype>` skeleton); `action: "read"` returns the same page wrapped in the skeleton,
and every builder below works on either form.

## Structure

An **Overview** landing part, then six groups; each old tab is a section (`.part`) and **keeps its old id**, so old links work.

| group (id) | sections (ids) |
|---|---|
| Overview (`overview`) | bottom line, five numbers, running now (filled by the auto-research script), map of the page |
| 1 Best method and how it is measured (`g-best`) | `summary`, `sft4` |
| 2 Scale and length (`g-scale`) | `sft11`, `models`, `cpt` |
| 3 What was tried and what is next (`g-tried`) | `autoresearch` (live), `selection`, `ruledout`, `next`, `priorwork` |
| 4 Mechanism and analysis (`g-mech`) | `mechanism`, `gradmaps` |
| 5 Cost and infrastructure (`g-cost`) | `memory`, `gpuaudit` |
| 6 Log and provenance (`g-log`) | `log`, `provenance` |

Navigation is built at run time: a sidebar tree on desktop, a group chip row plus a section row on a phone. It reads
`data-group`, `data-order`, `data-short` (and optional `data-badge`) from each `<div class="part" id="...">`, and the group
list from `<script type="application/json" id="hubgroups">`. Deep links: `#<part id>`, `#<group id>` (opens its first
section), and `#<part>--<heading-slug>` for any section heading (ids are generated on load). Each section's "On this page"
row comes from its `<section><h2>` headings. Long tables (12+ rows) and long notes are inside `<details class="more">`.

**Add a section:** append `<div class="part" id="myid" data-group="g-tried" data-order="6" data-short="Nav label">` with a
`.parthead` (eyebrow, `h2.ptitle`, `p.verdict`: the verdict first) and `<section><h2>` blocks. Nothing else needs editing.
**Add a group:** add an entry to `#hubgroups` (`id`, `title`, `short`, `blurb`). A part with no known group lands under "More".
Charts are `<canvas>` + data in the page's `const D` script (Chart.js 4.4.1 from cdnjs); the router already fires a resize
when a part is shown, so charts in hidden sections size correctly.

## Refresh the auto-research tab (hourly; done by the loop's hub-tab agent)

Source of truth: `research/token_routing/early_loss_proxy/hub_tab/` (branch `claude/early-loss-proxy-d6e886`):
`build_autoresearch_tab.py` (copy also here) and `loop_state.json` (keys: status, updated, setup, log, ideas, best_by_hour,
depth, board, curves, validation, pilot). Copy of the state: `autoresearch_state.json`.

```bash
python3 research/token_routing/early_loss_proxy/hub_tab/build_autoresearch_tab.py HUB.html --state research/token_routing/early_loss_proxy/hub_tab/loop_state.json
```

then `node --check` the inline scripts and publish HUB.html to the same artifact URL (read the live page first; if a newer
version exists, run the builder on that version, never force). The builder replaces the blocks between
`<!-- AR:PART:START -->`/`END` and `<!-- AR:SCRIPT:START -->`/`END` in place (first run: part before `provenance`, script at the
end; legacy `AUTORESEARCH` blocks are stripped). It is idempotent, does not touch the navigation, and also fills the
overview's "Running now" box. It lays the tab out as: status line and hourly-best table, FLOPs-to-reach table, ideas by family
(running ideas open, closed ones folded, each with its arms table), the loss-vs-FLOPs chart (references and top arms, plus a
selector for one idea's arms), validation runs, how it is scored, log (newest first, first five open), 8k pilot folded.
Board rows map to ideas by `board[i].idea` (the family, from `hub_tab/arms.json`), then by `tag` (== idea id, or an alias in
the script), then by the family's single idea; set `idea_id` on a row to pin it.

## Other refreshes

- Experiment log: `python3 update_log.py hub.html` splices `../experiment_log.json` into `const D_LOG`.
- Verify a rebuild lost nothing: `python3 verify_reorg.py OLD.html NEW.html [DOM_DUMP.html STATE.json]`
  (counts, ids, text blocks, data-script hashes; with `chrome --headless --dump-dom NEW#autoresearch` it checks every board row).
- Gradient-map PNGs are published under `gradmaps/` from `research/token_routing/grad_concentration/gradmaps/`.

## Files

- `reorg_hub.py`, `hub_v2.css`, `hub_v2_nav.js`, `hub_v2_map.js`: the one-off transform that produced the grouped page from the
  old one (kept for provenance; the CSS and the nav / overview scripts now live inside `hub.html`). `REORG_PLAN.md`: inventory and plan.
- `legacy/`: `rebuild_hub.sh`, `build_nov_tabs.py`, `build_gpu_tab.py`, `patch_log_render.py`. They assembled the old flat page from
  a v95 backup and edit the old navigation, so do not run them on the grouped page. Their data (`log_extras.json`, `sel_ops.json`,
  `gpu_audit_data.json`) stay here; the tabs they built are already in `hub.html`.
- Known stale item: the "Compute now" box on the log tab is a snapshot from 2026-10-30 (`const D_COMPUTE`); nothing refreshes it.
