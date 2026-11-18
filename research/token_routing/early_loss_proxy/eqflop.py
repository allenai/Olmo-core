"""Equal-FLOP view of the runs on the 4x-row pool (Beaker tag x4) against the dense references (rules of 2026-11-18).

For every x4 run and the dense curves: 16k held-out CE (heldout_v3 gen16k + rung16k, eval_size 320) as a function of cumulative
PF. Dense reference curve = lower envelope over PF of every dense FULL fine-tune run (w1-dense sd0/sd1 one epoch on mtb16k,
x4-dense-x4 976 steps, x4-dense-x4full 1,952 steps); dense LoRA runs are listed as baseline arms. Per run: loss at fixed PF
points, and for its LAST probe the PF dense full fine-tune needs for the same loss (None when dense has not reached it yet:
then the ratio is a lower bound from the deepest dense point).

Unfinished runs come from live logs, which carry no PF: PF = step x the arm's PF/step measured on its finished one-epoch run
(COST below); keep-ramp arms use the ramp's measured cumulative-PF shape rescaled to the run's length.

    python3 eqflop.py [--json hub_tab/eqflop.json]
"""
import contextlib, io, json, sys

sys.argv, argv = [sys.argv[0], "--live"], sys.argv
with contextlib.redirect_stdout(io.StringIO()):
    import ladder
runs = ladder.runs
COST = {"recipe-x4": 442 / 488, "lo-r50to20-x4": 643 / 488, "lo-c16k50-x4": 1066 / 488, "ft-r50to20-x4": 869 / 488, "lo-p50-20-7-x4": 575 / 488,
        "dense-x4": 4.061, "dense-x4full": 4.061, "denselora-x4": 1496 / 488, "denselora-x4full": 1496 / 488}
RAMP = {"ft-c7ramp-x4": ("w1-ft-c7ramp-sd0", 762)}  # arm -> (finished one-epoch run with the same schedule shape, this run's steps)


def curve(name):
    r = runs[name]; arm = name.split("-", 1)[1].rsplit("-sd", 1)[0]; out = []
    for s, pf, ce, _ in r["curve"]:
        if s == 0: continue
        if pf is None:
            if arm in RAMP:
                ref, n = RAMP[arm]; c = [(a, b) for a, b, _, _ in runs[ref]["curve"]]; x = s * 488 / n
                for (s0, p0), (s1, p1) in zip(c, c[1:]):
                    if s0 <= x <= s1: pf = (p0 + (x - s0) / (s1 - s0) * (p1 - p0)) * n / 488; break
            elif arm in COST: pf = s * COST[arm]
        if pf is not None: out.append((s, pf, ce))
    return out


def at_pf(c, pf):  # loss at a PF point (linear interpolation, running min), None outside the run
    best, prev = 9.0, None
    for s, p, ce in c:
        b0 = best; best = min(best, ce)
        if p >= pf:
            if prev is None: return None
            return b0 + (pf - prev) / (p - prev) * (best - b0) if best < b0 else b0
        prev = p
    return None


def pf_to_reach(c, lvl):
    best, prev = 9.0, 0.0
    for s, p, ce in c:
        if ce <= lvl: return p if best <= ce else prev + (best - lvl) / (best - ce) * (p - prev)
        best, prev = min(best, ce), p
    return None


dense_names = [k for k in ("w1-dense-sd0", "w1-dense-sd1", "x4-dense-x4-sd0", "x4-dense-x4full-sd0") if k in runs]
dense = {k: curve(k) for k in dense_names}
GRID = [500, 750, 1000, 1250, 1500, 1750, 1982, 3000, 4000, 6000, 7900]
names = dense_names + [k for k in sorted(runs) if k.startswith("x4-") and k not in dense_names] + [k for k in ("v3-denselora-sd0", "w2-denselora-sd1") if k in runs]
out = []
print("16k held-out CE at equal cumulative PF (eval_size 320, below 500). * = run finished.  dense full FT one epoch = 1,982 PF")
print(f"{'run':24s} " + " ".join(f"{g:>6d}" for g in GRID) + " | last: step PF CE | dense full-FT PF for that loss -> ratio")
for k in names:
    c = curve(k)
    if not c: continue
    cells = [at_pf(c, g) for g in GRID]; s, p, ce = c[-1]
    need = [pf_to_reach(d, ce) for d in dense.values()]; need = [x for x in need if x]
    deepest = max((d[-1][1] for d in dense.values() if d), default=0)
    ratio = (min(need) / p) if need else None; lb = (deepest / p) if not need and k not in dense else None
    out.append(dict(run=k, done=runs[k]["done"], last_step=s, last_pf=p, last_ce=ce, at_pf=dict(zip(GRID, cells)), dense_pf_same_loss=min(need) if need else None,
                    x=ratio, x_lower_bound=lb))
    print(f"{k[:23]:23s}{'*' if runs[k]['done'] else ' '} " + " ".join(f"{v:6.3f}" if v is not None else "     -" for v in cells)
          + f" | {s:5d} {p:6.0f} {ce:.3f} | " + (f"{min(need):6.0f} -> x{ratio:.2f}" if need and k not in dense else (f"dense not there yet (> {deepest:.0f} PF) -> >x{lb:.2f}" if lb else "")))
if "--json" in argv:
    json.dump(dict(grid=GRID, runs=out, eval_size=320), open(argv[argv.index("--json") + 1], "w"), indent=1)
