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
    "ft-sbw-r20to7": "full FT, dense forward, gradient only through kept 20% / 7%", "ft-sbw-c7k20": "full FT, dense forward, gradient only through kept 20% (block 7)",
    "ft-sbw-c0k20": "full FT, dense forward, gradient only through kept 20% (input)", "ft-c3ramp": "full FT, keep ramp 20% to 90% (block 3)",
    "ft-c7ramp10to60": "full FT, keep ramp 10% to 60% (block 7)", "ft-c7ramp-tail10": "full FT, keep ramp 20% to 90% + last 10% dense",
    "ft-r50to20-ex50": "full FT, keep 50% then 20%, half of the budget random", "ft-r50to20-rnd": "full FT, keep 50% then 20%, random tokens",
    "lo-p50-10-5": "LoRA, keep 50% / 10% / 5% at blocks 7 / 16 / 24",
    "ft-hold50": "full FT, 20% / 7% held for half the run, then ramp to 90% / 90%", "ft-hold70": "full FT, 20% / 7% held for 70% of the run, then ramp to dense",
    "ft-r2ramp": "full FT, both cuts ramp 20% / 7% to 90% / 90%", "ft-rlow": "full FT, both cuts ramp 20% / 7% to 60% / 40%",
    "r20to7-invkeep": "full FT 20% / 7%, gradient x 1/keep at the cuts", "r20to7-invmass": "full FT 20% / 7%, gradient x 1/covered mass", "r20to7-sqrt": "full FT 20% / 7%, gradient x sqrt(1/keep)",
    "r50to20-invkeep": "full FT 50% / 20%, gradient x 1/keep at the cuts", "r50to20-invmass": "full FT 50% / 20%, gradient x 1/covered mass", "lo-q50-30-15-7": "LoRA, keep 50% / 30% / 15% / 7% at blocks 7 / 12 / 16 / 24",
}


def arm_seed(run):
    arm, _, sd = run.split("-", 1)[1].rpartition("-sd")
    return arm, sd


rows = []
for r in L["runs"]:
    arm, sd = arm_seed(r["run"])
    if "mix" in arm and not arm.startswith("dense"): continue  # old re-mix recipe runs: baseline-side lever, no dense partner yet
    if arm == "denselora2e5" or arm.endswith("-x4"): continue  # equal-FLOP runs on the 4x pool get their own table when they finish
    tr = {s: (ce, b) for s, ce, b in r["track"]}
    if not any(s in tr for s in STEPS): continue  # 1,464-step runs: shown when their baseline lands
    fam = 0 if arm.startswith("dense") else (1 if arm.startswith("ft-") else 2)
    rows.append((fam, r.get("cost_vs_dense_ft") or 9, r, arm, sd, tr))
rows.sort(key=lambda x: (x[0], -x[1] if x[0] else x[1]))

e = html.escape
out = ['<tr><th>arm</th><th class="n">seed</th><th class="n">step cost vs dense full FT</th>' + "".join(f'<th class="n">step {s}</th>' for s in STEPS)
       + '<th class="n">deepest same-adapter dense level reached</th><th class="n">FLOP ratio there</th>'
       + '<th class="n">deepest dense full fine-tune level reached</th><th class="n">FLOP ratio there</th></tr>']
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
    w = r.get("vs_dense_ft")
    deepw = f'step {w["ref_step"]} of {w["ref_steps"]} (CE {w["level"]:.3f})' if w else ("reference" if arm == "dense" else "")
    xw = (f'<b>{w["x"]:.2f}×</b>' + (" ✓ final" if w["final"] else "")) if w else ""
    c = r.get("cost_vs_dense_ft")
    out.append(f'<tr><td>{e(DESC.get(arm, arm))}<br><code>{e(arm)}</code></td><td class="n">{e(sd)}</td><td class="n">{"" if c is None else f"{c:.2f}×"}</td>' + "".join(cells)
               + f'<td class="n">{deep}</td><td class="n">{x}</td><td class="n">{deepw}</td><td class="n">{xw}</td></tr>')

block = f'''<!-- AR:LADDER:START -->
<details class="sec" open id="ar-ladder-sec"><summary><h2 id="ar-ladder">Current phase (rules of 11-18): dynamic token dropping against the same-adapter dense baseline</h2><span class="n">1 table · {len(rows)} runs</span></summary><div class="secbody">
<p class="note sv">Rules since 2026-11-18: LoRA and fixed data re-mixes count as the dense baseline, so only dynamic FLOP dropping during training is a method. Each drop arm is compared with the dense run that uses the same adapter (LoRA arms with dense LoRA, full fine-tune arms with dense full fine-tuning) at the same training step; a method counts only at the deepest loss the dense baseline reaches. The search starts from drop settings that track dense and lowers the step cost from there.</p>
<p class="note">16k held-out CE (gen16k + rung16k of heldout_v3) at each probe step; in brackets the difference to the same-adapter dense baseline at that step (red when more than 0.01 behind). ⚠ eval_size 320, below 500; one seed per row, and two seeds of the same arm differ by about 0.005, so differences under 0.01 are noise. Full epoch on the standard pool (488 steps of 16 rows). The last two columns score every arm against dense full fine-tuning, the overall reference (16k CE 0.114 at step 488, 1,982 PF): the deepest level of its curve the arm reaches and how many times fewer FLOPs the arm needed there; "✓ final" marks arms that reach its final loss. Step cost is PF per step relative to dense full fine-tuning (4.06 PF). Rows marked running show live probes.</p>
<div class="tbl"><table>
{chr(10).join(out)}
</table></div>
</div></details>
<!-- AR:LADDER:END -->
'''
# ---- equal-FLOP table (hub_tab/eqflop.json, written by eqflop.py): runs on the 4x-row pool against dense full fine-tune ----
ep = os.path.join(os.path.dirname(lp), "eqflop.json")
if os.path.exists(ep):
    E = json.load(open(ep)); G = [g for g in E["grid"] if any(r["at_pf"].get(str(g)) is not None for r in E["runs"])]
    XD = {"w1-dense-sd0": "dense full fine-tune, standard pool, one epoch (seed 0)", "w1-dense-sd1": "dense full fine-tune, standard pool, one epoch (seed 1)",
          "x4-dense-x4-sd0": "dense full fine-tune, 976 steps on the 4x pool", "x4-dense-x4full-sd0": "dense full fine-tune, whole 4x pool (1,952 steps)",
          "x4-denselora-x4-sd0": "dense LoRA, 642 steps (baseline at 1,982 PF)", "x4-denselora-x4full-sd0": "dense LoRA, whole 4x pool (1,952 steps)",
          "v3-denselora-sd0": "dense LoRA, standard pool, one epoch (seed 0)", "w2-denselora-sd1": "dense LoRA, standard pool, one epoch (seed 1)",
          "x4-recipe-x4-sd0": "LoRA, keep 20% then 7%, whole 4x pool (1,952 steps)", "x4-lo-r50to20-x4-sd0": "LoRA, keep 50% then 20%, 1,525 steps",
          "x4-lo-p50-20-7-x4-sd0": "LoRA, keep 50% / 20% / 7%, 1,683 steps", "x4-lo-c16k50-x4-sd0": "LoRA, keep 50% after block 16, 905 steps",
          "x4-ft-r50to20-x4-sd0": "full FT, keep 50% then 20%, 1,108 steps", "x4-ft-c7ramp-x4-sd0": "full FT, keep ramp 20% to 90%, 762 steps"}
    er = ['<tr><th>run</th>' + "".join(f'<th class="n">{g:,} PF</th>' for g in G) + '<th class="n">last probe: step, PF, CE</th><th class="n">dense full FT PF for that loss</th><th class="n">FLOP ratio</th></tr>']
    for r in E["runs"]:
        cells = "".join('<td class="n">' + (f'{r["at_pf"][str(g)]:.3f}' if r["at_pf"].get(str(g)) is not None else "") + "</td>" for g in G)
        isd = "dense" in r["run"] and "lora" not in r["run"]
        need = "" if isd else (f'{r["dense_pf_same_loss"]:,.0f}' if r["dense_pf_same_loss"] else "dense has not reached it yet")
        x = "" if isd else (f'<b>{r["x"]:.2f}×</b>' if r["x"] else (f'&gt; {r["x_lower_bound"]:.2f}× (lower bound)' if r["x_lower_bound"] else ""))
        er.append(f'<tr><td>{e(XD.get(r["run"], r["run"]))}<br><code>{e(r["run"])}</code>{"" if r["done"] else " · running"}</td>{cells}'
                  f'<td class="n">{r["last_step"]:,}, {r["last_pf"]:,.0f}, {r["last_ce"]:.3f}</td><td class="n">{need}</td><td class="n">{x}</td></tr>')
    block = block.replace("<!-- AR:LADDER:END -->", f'''<details class="sec" open id="ar-eqflop-sec"><summary><h2 id="ar-eqflop">Equal FLOPs, more rows: loss against cumulative FLOPs on the 4x-row pool</h2><span class="n">1 table · {len(E["runs"])} runs</span></summary><div class="secbody">
<p class="note sv">The test of the deepest-loss rule: each drop arm gets the FLOPs of one dense full fine-tune epoch (1,982 PF) on a pool with 4x the rows (its learning-rate schedule stretched over its own step count), so cheap steps are traded for more rows. Dense references with more rows (976 and 1,952 steps) are running to extend the dense curve past 1,982 PF.</p>
<p class="note">16k held-out CE at equal cumulative PF (linear interpolation between probes). ⚠ eval_size 320, below 500; one seed. Rows marked running come from live probes and are mid-schedule (their learning rate is still high); for those, PF is the step count times the arm's PF per step measured on its finished one-epoch run. The last columns give the PF dense full fine-tuning needs to reach the run's latest loss and the ratio; where dense has not reached that loss in any run yet, the ratio is a lower bound from the deepest dense point. 12 of the 48 generated contradiction held-out rows have a gold claim that appears as filler (not as gold) in some row of the 4x pool, 2 for outlier; 0 on the standard pool.</p>
<div class="tbl"><table>
{chr(10).join(er)}
</table></div>
</div></details>
<!-- AR:LADDER:END -->''')
h = re.sub(r"<!-- AR:LADDER:START -->.*?<!-- AR:LADDER:END -->\n?", "", h, flags=re.S)
anchor = '<details class="sec" open id="ar-progress-sec">'
if anchor not in h:
    anchor = '<details class="sec" id="ar-progress-sec">'
assert h.count(anchor) == 1, "progress section anchor not found"
h = h.replace(anchor, block + '<details class="sec" id="ar-progress-sec">')
h = h.replace('<h2 id="ar-progress">Best so far, over time</h2>', '<h2 id="ar-progress">Best so far, over time (before the 11-18 rules; superseded)</h2>')
open(H, "w", encoding="utf-8").write(h)
print(f"ladder table: {len(rows)} rows -> {H} ({len(h)} bytes)")
