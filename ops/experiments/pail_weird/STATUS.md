---
experiment: pail_weird
title: Weird-web finder: surface unusual / harmful AI text among Pangram's AI-labelled core100k pages
status: active
updated: 2026-11-13
branch: PAIL prasann/weird-web (worktree /accounts/projects/berkeleynlp/prasann/projects/PAIL-weird), scripts/classify/weird/
launcher: scripts/tinyclf/beaker.sh --gpus 1 --vllm --mem 96GiB -- scripts/classify/weird/{annotate,build}.py
records:
results: s3://ai2-llm/checkpoints/prasanns/PAIL/commoncrawl/ccdev_v3/devsets/p41dev_v1/weird/v1/ (core100k tags), weird/v2/ (ann/ = next-500k tags, site/ = the 600k tab data)
artifacts: https://claude.ai/artifact/U8tdBvrbWe1EDAZQFoK2wj (PAIL Text Workbench, "Weird" tab = weird.html; v1 versions 167-168, v2 600k versions 220-221)
wandb_group:
next_step: Prasann reviews v2 (Agents & misalignment lens first). Options: a stronger agent read prompt (frontier stays low for benign agent activity), more crawls, DseWiki-style tooling net on more pages
---

# pail_weird

## Question
Among the pages Pangram 4.1 calls AI or Mixed in Jenna's ccdev_v3 core100k (5,781 pages), which show new, unusual or
harmful AI behaviour on the web (DseWiki-style agent output, leaked prompts, injected spam, scams), and can a cheap
LLM-tagging + clustering pass make them easy for Prasann to browse?

## Decisions
- 2026-11-09 (Prasann): build it as a "weird" tab in the PAIL Text Workbench; look at Jenna's Pangram labels.
- 2026-11-09 (Claude): "AI-labelled" = p4_label AI or Mixed (the p41dev rule fraction_ai > 0 makes 79% AI). Controls
  for base rates: 2,000 pages with fraction_ai = 0, and all 539 DseWiki pages (agent vs human) as the known-weird check.
  Tagger: Qwen3.8-27B, thinking off, temperature 0, on URL + first 800 chars + Pangram's AI windows (6,000 chars).
- 2026-11-09 (Claude): cluster lift = share of AI pages / share of human controls (the first formula capped at 1.35);
  recomputed on the built index locally (arithmetic only) and in build.py for reruns.

- 2026-11-13 (Prasann): scale to the 600k labels (core100k + Jenna's next 500k, p41next500k_v1). Claude: tag the next
  500k's AI / Mixed pages plus 8,000 more fully-human controls (v1 tags reused, same prompt); K = 100 clusters.
- 2026-11-13 (Claude, after Prasann saw farm pages that did not look alike): the v1 farms were wrong. Checked: 24 of
  30 v1 farms shared no real passage (the two largest, 22 and 18 pages, share at most 5 words a pair); char-5-gram
  TF-IDF matched shared vocabulary and union-find chained unrelated pages. v2 farms: word-8-gram shingles, a pair
  needs >= 25 shared covering >= 20% of the smaller AI excerpt, every member linked to >= half the group, the shared
  passage stored as evidence and an LLM label (content farm / shared template / mirror). On v1 data: 13 groups, every
  one with a real copied passage (36-295 words).
- 2026-11-13 (Claude): added "untagged themes" (weird or harmful pages with only broad tags, their reasons clustered
  and named, with a suggested new tag) and 95% Wilson intervals on tag rates, after Prasann asked what the tag set misses.
- 2026-11-13 (Prasann): focus on new / long-tail behaviour, especially rogue agents, misalignment and sophisticated
  coordination. Claude: (1) net.py, phrase families over ALL 600k pages whatever Pangram says (agent runtime, agent web
  tooling, agent self-talk, agent platforms, AI-addressed, injection, chatbot residue, misalignment talk, agent
  economy); on v1 texts the web-tooling family (URL-to-markdown proxies, raw archive fetches, request catchers) hits
  115 / 396 DseWiki agent pages and 0 human pages, the other families ~0 on DseWiki (its agents do not talk about
  themselves). (2) deep.py, a focused second read with thinking on (frame: agent output vs agent workspace vs
  AI-addressed vs ordinary AI content vs people discussing AI vs fiction; autonomy, AI-to-AI, misalignment *shown by an
  AI*, coordination, injection, novelty 0-3; frontier 0-10; named agents; evidence) on net hits, first-pass agent /
  prompt / weird>=7 pages, all DseWiki and 300+300 random calibration pages. (3) build: Agents & misalignment lens,
  calibration table, the same named agent on >= 2 sites.

## Current best numbers
| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| DseWiki agent pages | weirdness >= AI p95 (8) | 52.3% (human pages 0%) | 396 / 143 | | | weird/v1/site/weird/index.json meta.dse_check |
| DseWiki agent pages | tagged agent_activity | 59.1% (human 0%) | 396 / 143 | | | same |
| AI or Mixed core | weirdness >= 6 / harm >= 5 | AI 14.6% / 9.7%, Mixed 6.2% / 4.7%, human controls 6.5% / 7.7% | 1054 / 4727 / 1999 | | | meta.rates |
| AI or Mixed core | agent_activity / chatbot_residue / ai_disclosure / prompt_or_placeholder | 13 / 24 / 64 / 66 pages (controls 1 / 0 / 0 / 35) | 5781 | | | docs |
| tag rates AI vs control | mass_template 45.7 vs 6.1%, fabricated_specifics 6.1 vs 0.1%, fake_persona 3.4 vs 0.1%, hacked_or_injected 6.6 vs 7.0%, adult 1.3 vs 7.3% | | 5781 / 1999 | | | meta.rates |
| farms (v1, WRONG) | cross-site groups by char-5-gram cosine >= 0.5 | 30 groups; only ~6 share a real passage; superseded by v2 farms | 5781 | | | v1 meta.farms |

Notable: a "compromised Chinese industrial sites" cluster of 78 human-control pages (weirdness 7.8): keyword-swap spam
Pangram calls human. Harmful web spam is not mostly AI by Pangram's reading.

## v2 results (2026-11-13, 600k labelled pages)
| arm | metric | value | eval_size | source |
|---|---|---|---|---|
| first pass | pages tagged | 45,607 (AI 6,084, Mixed 28,798, controls 9,996, DseWiki 539, net-only 190) | | weird/v2/site meta |
| phrase net | pages with any hit over 600k | ~300 (most: agent_platform / llms.txt 167, misalignment talk 50) | 600,000 | weird/v2/net/tally.json |
| focused read | DseWiki agent content pages put in an agent frame | 83.5% (227 / 272) | 272 | meta.deep.calib |
| focused read | first-pass pages put in an agent frame | 1.9% | 3,744 | meta.deep.calib |
| focused read | frontier >= 6 | 1 page of 5,158 (frontier is anchored on harm; ranking uses agent_score instead) | 5,158 | deep/*.jsonl.gz |
| farms v2 | copied-passage groups | 127 (many shared cookie / checkout boilerplate, LLM-labelled) | 34,882 | meta.farms |
| untagged themes | weird / harmful pages with no specific tag | 136 in 5 themes (one new: contextual mismatch spam) | | meta.themes |
Top of the Agents lens: agentbazaar.tech (agent-authored research in an "AI society"), supercraft.host (game wiki with an
API for AI agents), drwho.virtadpt.net (prompt-injection + Anthropic magic strings aimed at AI readers), zap-ros.ru
(rules for AI agents scraping the shop), uaix.org (AGENTS.md standard).
Traps: the build crashed once on OpenBLAS threads (fixed: cap at 32); another session republished the Workbench while
this ran (merged on its index.html and nav).

## Next step
Prasann reviews the tab. Options: run on the 500k fresh labels when Jenna has them; a second tagger (e.g. another model
family) for agreement on weirdness / harm; a db-backed "interesting / not" flag in the page to steer the next round.

## Open issues / traps
- LLM tags are leads for a human to read, not labels; mass_template is still broad (46% of AI pages) after tightening.
- The artifact host refuses raw U+FFFD in published files: build.py writes it as the JSON escape.
- zsh: `set -- $x` does not split words (ops register failed once); write arguments out.
- A Monitor loop using bash `declare -A` stayed silent for 30 min; use a background until-loop with an exit bound.

## Verified vs unverified
- Verified (2026-11-09): smoke test read by hand (50 pages); build CPU path tested on smoke rows with GPU steps stubbed;
  tab screenshotted headless (desktop and 400px) on the real data before publish; nav link diff = only the link on all 12 pages.
- UNVERIFIED: tagger precision on a random sample of its high-weirdness calls (not hand-checked beyond the top ~25).
