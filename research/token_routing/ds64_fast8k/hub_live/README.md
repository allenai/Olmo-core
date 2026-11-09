# Live tokdrop_lr hub snapshot

`hub.html` is the page published at https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (private). It has drifted from
`../make_overview_page.py`: the Gradient maps (6b), Selection ablations (6c) and Prior work (8b) tabs, the
concentration / router-accuracy sections and the per-entry log tables exist only here. Edit this file (or rerun the
builders) rather than regenerating from `make_overview_page.py`, which would drop them.

- `../experiment_log.json` is spliced into `const D_LOG` on every update. Entries may carry `tables`, `chart` and `figs`;
  the log renders them under "Show ...", and `<div data-logfig="<title prefix>">` blocks in the tabs render them as figures.
- `build_nov_tabs.py` adds the 6c / 8b tabs and the 128k-256k and router-accuracy sections (data: `sel_ops.json`,
  numbers from `ops result show`); `log_extras.json` holds the per-entry tables extracted from the records (2026-11-09).
- Gradient-map PNGs are published under `gradmaps/` from `research/token_routing/grad_concentration/gradmaps/`
  (branch `claude/grad-support-tokens`).
