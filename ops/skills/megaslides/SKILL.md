---
name: megaslides
description: Use when adding or updating slides in Prasann's "Prasann Mega-Slides" weekly meeting deck (Google Slides) — "add slides for X to the megaslides", "put this week's results in the deck", "make 2-3 slides on <experiment>", "update the Claude slides". Builds slides in the deck's own format (dash bullets, figure/table on the right, "[Added by Claude]" titles) from a short markdown spec, with native bar/line charts, in ~4 tool calls.
---

# megaslides (v1, 2026-10-20)

Token budget matters: one masked read saved to a file, one write, one stacked thumbnail. Never pull deck
JSON into context; the script does all index and layout arithmetic.

Files here: `megaslides.py` (locate / build / thumbs), `deck.json` (deck IDs, layout constants, palette:
edit this to restyle), `s3presign.py` (PNG upload route, NOT enabled: see Figures).
State: `~/.cache/megaslides/state.json`. Run with plain `python3` (stdlib; `thumbs` needs PIL), login node is fine.

## The deck

- `1G7gWetx74tB1YCAl4tg4TrIyMWFC3cn9Mx20WIRoAes`, 10 x 5.625 in, Simple Light, ~250 slides. **Newest week at the
  top**: slide 0 is the "Week #N" example; then `Week #19` divider, Project Overview, Summary of Week, Meeting /
  Success, content slides, `End of Week 19`, then older weeks.
- Content slides: TITLE_AND_BODY, short Title-Case label title, **dash bullets** with one nested level, a
  figure/table screenshot on the right, little else. Our slides go just before "End of Week N" of the newest week.
- Titles get the prefix `[Added by Claude] ` automatically (deck.json `title_prefix`).

## Workflow

1. **Content first.** For each slide pick ONE experiment/result. Pull numbers only from the experiment's
   STATUS.md "Current best numbers"/verdict or its record (`ops exp show NAME`), never from memory. Flag
   eval_size < 500 inline (⚠ eval_size=100) and say no-cot/cot where relevant. 2-3 slides unless asked.
2. **Read + locate** (the result is ~400 KB, so the tool saves it to a file; pass that path):
   `read_presentation` with fields
   `["revisionId,slides(objectId,pageElements(objectId,size,transform,shape(placeholder.type,text.textElements(startIndex,endIndex,paragraphMarker.bullet.nestingLevel,textRun.content))))"]`
   then `python3 <skill>/megaslides.py locate <saved file>` → rev, newest week, insert index, the week's titles
   (check nothing you plan already exists), template groups.
3. **Write a spec** (markdown, in the scratchpad):
   ```
   ## Length Gen: RoPE Reset / Hashed IDs
   - Trained on setA at ≤32k: do these help at 64k / 128k?
     - RoPE reset at each doc and the query
   - Neither helps (±0.02 of baseline)
     - Only hint: contradiction@128k 0.35 → 0.43 / 0.45
   chart: {"type":"bar","categories":["Baseline","RoPE reset"],"series":{"64k":[0.70,0.68],"128k":[0.53,0.54]},"ymax":0.8}
   > Mean over 7 CTC rows, no-cot. ⚠ eval_size=100 per cell (±0.04–0.05).
   ```
   - `## ` title (no prefix), `- ` bullet, `  - ` sub-bullet, `> ` caption under the right panel,
     `id: <5+ chars>` optional fixed slide id (default: hash of the title, so a rebuild finds it).
   - Right panel, at most one: a markdown table (`| a | b |` rows, first row = header; numbers right-aligned,
     first column bold) OR `chart:` one-line JSON. No panel → body is full width.
   - Charts: `bar` {categories, series{name:[...]}, ymax?, fmt?} (grouped, value labels, legend if >1 series);
     `line` {x, series, xlog?, xlabel?, ymin?, ymax?, fmt?} (markers, direct series labels, 3 y ticks).
     `fmt` is a Python format like `"{:.0%}"`.
   - Bullet shape limit: the body is cut from a template whose levels run 0,1,0,1,0,1, so at most **3 top-level
     bullets that have sub-bullets** (any number of lines per group). Top-level lines ≈ 34 chars per line on a
     half-width body, sub-bullets ≈ 45: keep each to ≤ 2 lines.
4. **Build + send:** `python3 <skill>/megaslides.py build spec.md [--replace] [--at N|end]` prints `REV <id>` and
   one compact JSON line. Send it once with `update_presentation` (`requests` = that JSON,
   `writeControl.requiredRevisionId` = REV). Do not print it twice. `--replace` deletes and rebuilds slides whose
   id already exists. A table slide is ~5 KB, a chart slide ~10 KB.
5. **Verify:** `read_slide_page_thumbnail` (MEDIUM) per new slide, then
   `python3 <skill>/megaslides.py thumbs URL1 URL2 ...` and Read the one stacked PNG. Check: wrapped table headers,
   orphan words, caption overlap, text past the bottom.
6. **Fix small things cheaply:** wording → `replaceAllText` with `pageObjectIds` (no revision needed); geometry →
   a few `updatePageElementTransform`/`updateTableColumnProperties` after a `fields: ["revisionId"]` read. Bigger
   changes: edit the spec and rebuild with `--replace` (re-run locate first; the state's rev is stale after a write).
7. Tell the user the slide titles, where they landed, and that the deck link is unchanged.

## Figures

- **Native charts (works now):** `chart:` lines above, drawn from shapes, so they are editable and vector. Element
  cloning (prototype + duplicate + RELATIVE transform) keeps request size down.
- **Existing PNG figures (not enabled):** Slides `createImage` needs a URL Google can fetch. `s3presign.py` uploads
  to `s3://ai2-llm/checkpoints/prasanns/megaslides/` and makes a 1 h presigned URL (Google copies the image at
  insert time). The auto-mode classifier blocked running it (2026-10-20, "data exfiltration"), so it is untested and
  needs Prasann's explicit OK / a permission rule before use. Never route images through third-party hosts.
- **Sheets charts (blocked):** `createSheetsChart` (LINKED or NOT_LINKED_IMAGE) fails with "Permission denied for
  document" through this Slides connector, even on a sheet we own. A test sheet "MegaSlides figures (Claude)"
  exists (`1_Fmn0O_JLUDhCIbZWBkJu_98AJBsFAGHVJTMQ75wnOo`); revisit if the connector gains Sheets scope.

## API traps (learned the hard way)

- The deck's "-" bullets cannot be created through the API: `createParagraphBullets` only has presets (●○■) and
  inserted text gets no bullets. Hence the template trick: duplicate slide `g3f5eb7b6944_0_102` ("Next Steps",
  Week #10) and rewrite its paragraphs, which keep their dash and nesting. **Nobody should edit that slide**; if it
  changes, `locate` re-reads its structure, and if it is deleted pick another dash-list slide in deck.json.
- Typed tabs never become nesting; `insertText` with `\n` inside a paragraph clones that paragraph's level.
- The shape's final newline cannot be deleted (`end index > text length`); to drop trailing paragraphs delete from
  the previous paragraph's newline to `text_len`, and the kept paragraph keeps its own level.
- Slides re-normalizes a new shape's stored size, so ABSOLUTE transforms computed from the size you sent are wrong
  for clones; use RELATIVE.
- A line runs top-left → bottom-right of its box; `scaleY: -1` flips it upward.
- Big masked reads auto-save to a file; small ones land in context, so keep masks tight for single slides.
- Thumbnail `contentUrl`s download without auth (urllib/curl).
- Native table headers wrap mid-word ("Rando m") if a column is too narrow; the script sizes columns by text length.

## Changelog / ideas for next versions

- v1 (2026-10-20): locate/build/thumbs, template bullets, tables, bar + line charts. Built while adding the three
  Week #19 slides (token-drop gradients, RoPE/hash length gen, tiny AI-text classifier).
- Next: PNG figures once a host is approved; new-week scaffolding (divider + overview + end slide); speaker notes;
  deeper templates (level-2 bullets, more groups); auto-fit check before sending; even smaller chart requests.
