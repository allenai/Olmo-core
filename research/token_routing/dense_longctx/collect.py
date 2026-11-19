"""Tabulate dense_step_bench.py result JSONs:  python collect.py data/bench/*.json"""
import json
import sys

rows = []
for p in sys.argv[1:]:
    d = json.load(open(p))
    a = d["args"]
    name = p.split("/")[-1][:-5]
    cp = a["cp"]
    rows_per_step = d["world"] // cp
    mode = "fwd-only" if a.get("forward_only") else ("LoRA r%d" % a["lora_rank"] if a["lora_rank"] else "full FT")
    opts = [f"AC {a['ac']}"] + (["int32 guard"] if a["gdn_safe"] else []) + (["base+real row"] if a["init"] != "random" else [])
    if "oom" in d:
        rows.append((name, d["T"], d["world"], cp, mode, "OOM", "-", "-", "-", ", ".join(opts)))
        continue
    st = d["steps"]
    pk = max(r["peak_alloc_gib"] for r in st)
    res = max(r["peak_reserved_gib"] for r in st)
    s = d.get("s_per_step", st[-1]["s"])
    pf = d.get("pf", {}).get("total_pf")
    gpu_s_row = s * d["world"] / rows_per_step
    rows.append((name, d["T"], d["world"], cp, mode, f"{pk:.1f} / {res:.1f}", f"{s:.1f}", f"{gpu_s_row:.0f}",
                 f"{pf:.1f}" if pf else "-", ", ".join(opts) + f"; loss {st[-1]['loss']:.3f}"))
print("| run | T | GPUs | CP | mode | peak GiB alloc / reserved | s/step | GPU-s/row | PF/row (analytic) | notes |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r[4], r[1], r[2])):
    print("| " + " | ".join(str(x) for x in r) + " |")
