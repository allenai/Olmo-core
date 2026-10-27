"""HTML section for the tokdrop_lr overview page: the LoRA x routing 2x2 ablation (records/lora-vs-routing-ablation.md).

Reads results/lora_routing_2x2.json (lora_routing_2x2.py) and returns one <section> in the overview page's markup
(.tbl / .n / .note classes). make_overview_page.py can splice it in with

    from make_lora_routing_section import section
    html = html.replace("<!--LORA_ROUTING-->", section(json.load(open(".../lora_routing_2x2.json"))))

or run this file standalone to print the section.
"""
import json
import sys

CELL = {"A": "dense", "B": "dense + LoRA-all", "C": "two-stage routing", "D": "routing + LoRA-all"}
SIZE = {("mt4", "f", 1952): "4-task, fresh 1,952 rows/task", ("mt4", "f", 3904): "4-task, fresh 3,904 rows/task",
        ("mt4", "s", 1464): "4-task, original pool 1,464 rows/task", ("mt11", "s", 240): "11-task, 240 rows/task",
        ("mt11", "s", 480): "11-task, 480 rows/task", ("mt11", "s", 720): "11-task, 720 rows/task"}
TAG = "best_any_lr"  # matched-FLOP reference: dense at its best LR (incl. 2e-6), the conservative curve


def _x(e, top_pf):
    """above dense's best point: the multiplier is at least dense's largest PF / the cell's PF"""
    return f"&gt;{top_pf / e['pf']:.2f}×" if e[f"xgt_{TAG}"] else f"{e[f'x_{TAG}']:.2f}×"


def section(d):
    rows = []
    attr = []
    for t in d["tables"]:
        key = (t["mix"], t["pool"], t["rows"])
        if key not in SIZE or len(t["cells"]) < 4:
            continue
        top_pf = max(p["pf"] for p in d["dense_curve"][f"{t['mix']}:{TAG}"])
        for c in "ABCD":
            e = t["cells"][c]
            seeds = f"{e['n']}" + (f" (sd {e['y_sd']:.3f})" if e.get("y_sd") is not None else "")
            other = ", ".join(f"{lr}: {y:.3f}" for lr, y in e["other_lr"].items()) or "–"
            rows.append(f"<tr><td>{SIZE[key] if c == 'A' else ''}</td><td>{c}. {CELL[c]}</td><td>{e['lr']}</td>"
                        f"<td class='n'>{other}</td><td class='n'>{seeds}</td><td class='n'>{e['pf']:.0f}</td>"
                        f"<td class='n'>{e['y']:.3f}</td><td class='n'>{e['o8']:.3f}</td><td class='n'>{e['c8']:.3f}</td>"
                        f"<td class='n'>{e[f'd_{TAG}']:+.3f}</td><td class='n'>{_x(e, top_pf)}</td></tr>")
        a, lx, md = t.get(f"attr_{TAG}"), t.get(f"attr_log2x_{TAG}"), t.get("matched_data")
        if a:
            f = lambda k: f"{a[k]:+.3f}" + (f" / {2 ** lx[k]:.2f}×" if lx else "")
            attr.append(f"<tr><td>{SIZE[key]}</td><td class='n'>{f('lora_no_route')}</td><td class='n'>{f('lora_with_route')}</td>"
                        f"<td class='n'>{f('route_no_lora')}</td><td class='n'>{f('route_with_lora')}</td>"
                        f"<td class='n'>{f('interaction')}</td><td class='n'>{f('total')}</td>"
                        f"<td class='n'>B {md['pf_ratio']['B']:.2f} · C {md['pf_ratio']['C']:.2f} · D {md['pf_ratio']['D']:.2f}</td></tr>")
    return f"""<section>
  <h2>LoRA vs routing: the 2×2 ablation</h2>
  <p class="note">Qwen3.5-4B SFT, every cell at LR 5e-6 and 1e-5 (the better one shown, the other in the next column), no
  full-context steps. LoRA-all = rank-64 adapters on all 32 blocks over a frozen base. Routing = online gradient-norm
  router, keep 20% after block 7 and 7% after block 16 (+16-token per-document floor on the 11-task mix). eval_size 500
  per task and rung, no-cot. "vs dense" = score minus dense at the same training PF, dense taken at its best LR per
  data size (2e-6, 5e-6 or 1e-5) and interpolated in log PF; "× dense" = dense PF needed to match ÷ the cell's PF.
  ⚠ Most cells are one seed; the 8-cell / 9-task mean moves ≈0.01–0.02 between seeds and outlier@8k up to ±0.1–0.19.</p>
  <div class="tbl"><table>
    <tr><th>data</th><th>cell</th><th>LR</th><th class="n">other LR</th><th class="n">seeds</th><th class="n">PF</th>
    <th class="n">mean</th><th class="n">outlier@8k</th><th class="n">contra@8k</th><th class="n">vs dense</th><th class="n">× dense</th></tr>
    {"".join(rows)}
  </table></div>
  <h3 style="margin-top:14px">Attribution at matched FLOPs (Δ score / FLOP multiplier)</h3>
  <div class="tbl"><table>
    <tr><th>data</th><th class="n">LoRA, no routing (B−A)</th><th class="n">LoRA on routing (D−C)</th>
    <th class="n">routing, no LoRA (C−A)</th><th class="n">routing on LoRA (D−B)</th><th class="n">interaction</th>
    <th class="n">total (D−A)</th><th class="n">PF / dense, same data</th></tr>
    {"".join(attr)}
  </table></div>
</section>"""


def markdown(d, tag=TAG):
    """The same two tables as Markdown, for records/lora-vs-routing-ablation.md."""
    out = ["| data | cell | best LR | other LR | seeds | PF | mean | outlier@8k | contra@8k | vs dense | × dense |",
           "|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    att = ["| data | LoRA, no routing (B−A) | LoRA on routing (D−C) | routing, no LoRA (C−A) | routing on LoRA (D−B) "
           "| interaction | total (D−A) | PF ÷ dense PF, same data (B · C · D) |",
           "|---|--:|--:|--:|--:|--:|--:|---|"]
    for t in d["tables"]:
        key = (t["mix"], t["pool"], t["rows"])
        if key not in SIZE or len(t["cells"]) < 4:
            continue
        top_pf = max(p["pf"] for p in d["dense_curve"][f"{t['mix']}:{tag}"])
        for c in "ABCD":
            e = t["cells"][c]
            seeds = f"{e['n']}" + (f" (sd {e['y_sd']:.3f})" if e.get("y_sd") is not None else "")
            other = ", ".join(f"{lr}: {y:.3f}" for lr, y in e["other_lr"].items()) or "–"
            x = f">{top_pf / e['pf']:.2f}×" if e[f"xgt_{tag}"] else f"{e[f'x_{tag}']:.2f}×"
            out.append(f"| {SIZE[key] if c == 'A' else ''} | {c}. {CELL[c]} | {e['lr']} | {other} | {seeds} | {e['pf']:.0f} "
                       f"| {e['y']:.3f} | {e['o8']:.3f} | {e['c8']:.3f} | {e[f'd_{tag}']:+.3f} | {x} |")
        a, lx, md = t.get(f"attr_{tag}"), t.get(f"attr_log2x_{tag}"), t.get("matched_data")
        if a:
            f = lambda k: f"{a[k]:+.3f}" + (f" / {2 ** lx[k]:.2f}×" if lx else "")
            att.append(f"| {SIZE[key]} | {f('lora_no_route')} | {f('lora_with_route')} | {f('route_no_lora')} "
                       f"| {f('route_with_lora')} | {f('interaction')} | {f('total')} "
                       f"| {md['pf_ratio']['B']:.2f} · {md['pf_ratio']['C']:.2f} · {md['pf_ratio']['D']:.2f} |")
    return "\n".join(out), "\n".join(att)


if __name__ == "__main__":
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/lora_routing_2x2.json"))
    if "--md" in sys.argv:
        print("\n\n".join(markdown(d)))
    else:
        print(section(d))
