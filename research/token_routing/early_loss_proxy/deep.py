"""Deepest-level scoring (Prasann 2026-11-17: only methods that reach the deepest dense losses count; extra steps are fine).
Reference = full-epoch dense LoRA (tag full, arm denselora) on the mean held-out CE (heldout_v2, eval_size 184). For every
run in hub_tab/board_v2.json: the deepest reference probe level it reaches, the FLOP ratio there, whether it reaches the
reference's final loss, and whether its loss was still falling when it stopped.   python3 deep.py [--json hub_tab/deep.json]"""
import json, sys
b = json.load(open("hub_tab/board_v2.json"))
# --ref TAG/ARM picks the reference run. Prasann 2026-11-17: the reference is dense FULL fine-tuning, not dense LoRA.
# Until a full-epoch dense full-FT probe curve exists, v2/dense (steps 10-100 only) is all there is for it.
RT, RA = (sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "full/denselora").split("/")
ref = next(r for r in b["runs"] if r["tag"] == RT and r["arm"] == RA)
def env(c):  # running-min envelope of (step, pf, ce)
    out, best = [], 9.0
    for s, pf, ce in c: best = min(best, ce); out.append((s, pf, best))
    return out
def reach(c, lvl):
    e = env(c)
    for a, q in zip(e, e[1:]):
        if q[2] <= lvl: return q[1] if a[2] <= q[2] else a[1] + (a[2] - lvl) / (a[2] - q[2]) * (q[1] - a[1])
    return None
R = env(ref["curve"])[1:]; final = R[-1][2]
rows = []
for r in b["runs"]:
    if r is ref or r["arm"] == "dense" or len(r["curve"]) < 2: continue
    c = r["curve"]; hit = [(s, pf, ce, reach(c, ce)) for s, pf, ce in R]; hit = [h for h in hit if h[3]]
    last, prev = c[-1], c[-2]
    d = dict(tag=r["tag"], arm=r["arm"], seed=r["seed"], steps=last[0], pf=last[1], ce_last=last[2], eval_size=r.get("eval_size"),
             reaches_final=bool(hit and hit[-1][2] <= final + 1e-9), still_falling=bool(last[2] < min(x[2] for x in c[:-1]) - 0.002))
    if hit: s, pf, ce, mine = hit[-1]; d.update(deep_ref_step=s, deep_ce=ce, deep_x=pf / mine, deep_pf=mine)
    rows.append(d)
rows.sort(key=lambda d: (-(d.get("deep_ref_step") or 0), -(d.get("deep_x") or 0)))
out = dict(ref_run=f"{RT}/{RA}", ref=[dict(step=s, pf=pf, ce=ce) for s, pf, ce in R], final_ce=final, final_pf=R[-1][1], runs=rows)
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
print(f"reference {RT}/{RA}: final CE {final:.3f} at {R[-1][1]:.0f} PF; levels (step:CE) " + " ".join(f"{s}:{ce:.3f}" for s, _, ce in R))
print(f"{len(rows)} runs; reaching the final level: {sum(d['reaches_final'] for d in rows)}")
for d in rows[:int(next((a for a in sys.argv[1:] if a.isdigit()), 30))]:
    print(f"{d['tag'][:10]:10s} {d['arm'][:22]:22s} sd{d['seed']} steps {d['steps']:4d} PF {d['pf']:6.0f} last CE {d['ce_last']:.3f} | deepest dense level: step {d.get('deep_ref_step','-')} (CE {d.get('deep_ce',0):.3f}) x {d.get('deep_x',0):.2f} {'still falling' if d['still_falling'] else 'flat at end'}")
