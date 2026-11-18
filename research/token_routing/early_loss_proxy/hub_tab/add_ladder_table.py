"""Post-process the hub page AFTER hub_live/build_autoresearch_tab.py: add the 11-18 phase table (hub_tab/ladder.json) as the
first section bar of the auto-research part, and mark the old best-so-far chart as superseded. Idempotent (the builder
regenerates the part, so run this after every build).

    python3 hub_tab/add_ladder_table.py HUB.html [--ladder hub_tab/ladder.json]

Interim: the hub-reorg session owns the tab builder; when it is reachable again this table should move into the builder.
"""
import html, json, os, re, sys

args = sys.argv[1:]
lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ladder.json")
if "--ladder" in args:
    i = args.index("--ladder"); lp = args[i + 1]; del args[i:i + 2]
H = args[0]
h = open(H, encoding="utf-8").read()
L = json.load(open(lp))
STEPS = [50, 100, 150, 200, 300, 400, 488]

DESC = {  # arm -> plain description
    "dense": "dense full fine-tune (reference)", "denselora": "dense LoRA (baseline)", "dense-mix80x3": "dense full fine-tune, 80/20 re-mix x3 (baseline)",
    "recipe": "LoRA, keep 20% then 7% (old recipe)", "lo-c16k50": "LoRA, keep 50% after block 16", "lo-r50to20": "LoRA, keep 50% then 20%",
    "lo-r35to12": "LoRA, keep 35% then 12%", "lo-r50to10": "LoRA, keep 50% then 10%", "lo-p50-20-7": "LoRA, keep 50% / 20% / 7% at blocks 7 / 16 / 24",
    "lo-e50-30-15": "LoRA, keep 50% / 30% / 15% at blocks 3 / 7 / 16", "lo-c7ramp": "LoRA, keep ramp 20% to 90% (block 7)",
    "lo-c7ramp10to60": "LoRA, keep ramp 10% to 60% (block 7)", "rc-de4": "LoRA 20%/7% + every 4th step dense", "rc-tail20": "LoRA 20%/7% + last 20% of steps dense",
    "ft-c16k50": "full FT, keep 50% after block 16", "ft-c7k50": "full FT, keep 50% after block 7", "ft-r50to20": "full FT, keep 50% then 20%",
    "ft-r20to7": "full FT, keep 20% then 7%", "ft-r20to7-de4": "full FT 20%/7% + every 4th step dense", "ft-r20to7-tail20": "full FT 20%/7% + last 20% dense",
    "ft-c7ramp": "full FT, keep ramp 20% to 90% (block 7)", "ft-c7ramp10to90": "full FT, keep ramp 10% to 90% (block 7)",
    "ft-m99": "full FT, adaptive keep covering 99% of gradient mass", "ft-m95": "full FT, adaptive keep 95%", "ft-m90": "full FT, adaptive keep 90%",
    "ft-mod8s2": "full FT, 80% of tokens skip every other block from 8", "ft-mod8d3": "full FT, 80% of tokens skip 3 of every 4 blocks from 8",
}


def arm_seed(run):
    arm, _, sd = run.split("-", 1)[1].rpartition("-sd")
    return arm, sd


rows = []
for r in L["runs"]:
    arm, sd = arm_seed(r["run"])
    if "mix" in arm and not arm.startswith("dense"): continue  # old re-mix recipe runs: baseline-side lever, no dense partner yet
    if arm == "denselora2e5": continue
    tr = {s: (ce, b) for s, ce, b in r["track"]}
    if not any(s in tr for s in STEPS): continue  # 1,464-step runs: shown when their baseline lands
    fam = 0 if arm.startswith("dense") else (1 if arm.startswith("ft-") else 2)
    rows.append((fam, r.get("cost_vs_dense_ft") or 9, r, arm, sd, tr))
rows.sort(key=lambda x: (x[0], -x[1] if x[0] else x[1]))

e = html.escape
out = ['<tr><th>arm</th><th class="n">seed</th><th class="n">step cost vs dense full FT</th>' + "".join(f'<th class="n">step {s}</th>' for s in STEPS)
       + '<th class="n">deepest same-adapter dense level reached</th><th class="n">FLOP ratio there</th></tr>']
for fam, _, r, arm, sd, tr in rows:
    cells = []
    for s in STEPS:
        if s not in tr: cells.append('<td class="n">' + ("running" if not r["done"] else "") + "</td>"); continue
        ce, b = tr[s]
        d = "" if b is None else f' <span class="{"neg" if ce - b > 0.01 else "sub"}">({ce - b:+.3f})</span>'
        cells.append(f'<td class="n">{ce:.3f}{d}</td>')
    v = r.get("vs_same")
    deep = f'step {v["ref_step"]} of {v["ref_steps"]} (CE {v["level"]:.3f})' if v else ("baseline" if fam == 0 else "")
    x = (f'<b>{v["x"]:.2f}×</b>' + ("" if v["final"] else " (not the final level)")) if v else ""
    c = r.get("cost_vs_dense_ft")
    out.append(f'<tr><td>{e(DESC.get(arm, arm))}<br><code>{e(arm)}</code></td><td class="n">{e(sd)}</td><td class="n">{"" if c is None else f"{c:.2f}×"}</td>' + "".join(cells)
               + f'<td class="n">{deep}</td><td class="n">{x}</td></tr>')

block = f'''<!-- AR:LADDER:START -->
<details class="sec" open id="ar-ladder-sec"><summary><h2 id="ar-ladder">Current phase (rules of 11-18): dynamic token dropping against the same-adapter dense baseline</h2><span class="n">1 table · {len(rows)} runs</span></summary><div class="secbody">
<p class="note sv">Rules since 2026-11-18: LoRA and fixed data re-mixes count as the dense baseline, so only dynamic FLOP dropping during training is a method. Each drop arm is compared with the dense run that uses the same adapter (LoRA arms with dense LoRA, full fine-tune arms with dense full fine-tuning) at the same training step; a method counts only at the deepest loss the dense baseline reaches. The search starts from drop settings that track dense and lowers the step cost from there.</p>
<p class="note">16k held-out CE (gen16k + rung16k of heldout_v3) at each probe step; in brackets the difference to the same-adapter dense baseline at that step (red when more than 0.01 behind). ⚠ eval_size 320, below 500; one seed per row, and two seeds of the same arm differ by about 0.005, so differences under 0.01 are noise. Full epoch on the standard pool (488 steps of 16 rows). Full fine-tune arms have no baseline cells until the dense full fine-tune run finishes. Step cost is PF per step relative to dense full fine-tuning (4.06 PF).</p>
<div class="tbl"><table>
{chr(10).join(out)}
</table></div>
</div></details>
<!-- AR:LADDER:END -->
'''
h = re.sub(r"<!-- AR:LADDER:START -->.*?<!-- AR:LADDER:END -->\n?", "", h, flags=re.S)
anchor = '<details class="sec" open id="ar-progress-sec">'
if anchor not in h:
    anchor = '<details class="sec" id="ar-progress-sec">'
assert h.count(anchor) == 1, "progress section anchor not found"
h = h.replace(anchor, block + '<details class="sec" id="ar-progress-sec">')
h = h.replace('<h2 id="ar-progress">Best so far, over time</h2>', '<h2 id="ar-progress">Best so far, over time (before the 11-18 rules; superseded)</h2>')
open(H, "w", encoding="utf-8").write(h)
print(f"ladder table: {len(rows)} rows -> {H} ({len(h)} bytes)")
