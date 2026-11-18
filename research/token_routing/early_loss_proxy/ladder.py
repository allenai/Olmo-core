"""Scoring under the 2026-11-18 rules (heldout_v3 runs; primary key = mean CE over the 16k sets gen16k + rung16k).

LoRA and fixed re-mixes are part of the dense baseline, so every drop arm is compared with the dense run that uses the
SAME adapter and pool:   ft-* -> w1/dense (dense full fine-tune, mtb16k),   everything else on mtb16k -> v3/denselora,
*-mix80x3 -> w1/dense-mix80x3.   Per arm: step cost vs dense full FT, the 16k CE at each shared probe step next to its
baseline (per-step tracking), and the FLOP ratio at the deepest baseline level the arm reaches (running-min envelope,
linear interpolation in PF). A second ratio is against dense full fine-tuning (the overall reference).

    python3 ladder.py [--json hub_tab/ladder.json] [--live]      (--live also parses [dense-probe] lines of unfinished logs)
Sources: /tmp/claude-3018/elp_beaker_results/{v3,w*}/ (fetch_beaker_results.sh) and /net/{sneetches,mooney}/.../results/w*/.
"""
import glob, json, os, re, sys
import numpy as np

SRC = [p for pat in ("/tmp/claude-3018/elp_beaker_results/v3", "/tmp/claude-3018/elp_beaker_results/w[0-9]*",
                     "/tmp/claude-3018/elp_beaker_results/x4", "/tmp/claude-3018/elp_beaker_results/sbw", "/tmp/claude-3018/elp_beaker_results/ks", "/tmp/claude-3018/elp_beaker_results/gc", "/tmp/claude-3018/elp_beaker_results/bud", "/tmp/claude-3018/elp_beaker_results/x16",
                     "/net/sneetches/data/prasann/early_loss_proxy/results/w[0-9]*",
                     "/net/mooney/data/prasann/early_loss_proxy/results/w[0-9]*") for p in sorted(glob.glob(pat))]
W16 = {"gen16k": 48, "rung16k": 32}
DENSE_PF_STEP = 4.06  # dense full fine-tune on mtb16k, PF per step (flop meter)


def load_json(p):
    d = json.load(open(p))
    S = np.array([r["set"] for r in d["rows"]]); m = np.isin(S, list(W16))
    if not m.any(): return None
    f = lambda q: (q.get("flops") or {})
    c = [(q["step"], f(q).get("actual_pf", 0.0), float(np.asarray(q["ce"])[m].mean()), float(np.mean(q["ce"]))) for q in d["probes"]]
    last = f(d["probes"][-1])
    return dict(curve=c, done=bool(d.get("done")), eval_size=int(m.sum()),
                xd=last.get("actual_pf", 0) / max(1e-9, last.get("dense_pf", 0)) if last else None)


def load_log(p):  # unfinished run: 16k mean from the per-set/task means in the log (PF unknown -> None)
    c = []
    for line in open(p, errors="replace"):
        if not line.startswith("[dense-probe] step "): continue
        step = int(line.split()[2]); num = den = 0.0
        for k, v in re.findall(r"(\w+)/\w+=([0-9.]+)", line):
            if k in W16: num += W16[k] * float(v); den += W16[k]
        if den: c.append((step, None, num / den, None))
    return dict(curve=c, done=False, eval_size=320, xd=None) if len(c) > 1 else None


runs = {}
for src in SRC:
    for p in sorted(glob.glob(f"{src}/elp-*.json")):
        r = load_json(p)
        if r: runs[os.path.basename(p)[4:-5]] = r
    if "--live" in sys.argv:
        for p in sorted(glob.glob(f"{src}/elp-*.log")):
            k = os.path.basename(p)[4:-4]
            if k not in runs:
                r = load_log(p)
                if r: runs[k] = r


def base_of(name):
    arm = name.split("-", 1)[1].rsplit("-sd", 1)[0]
    if "mix80x3" in arm: return "w1-dense-mix80x3-sd0"
    if arm.endswith("-x4"):  # equal-FLOP runs on the 4x pool: dense run on the same pool with the same adapter (compared in PF)
        return "x4-dense-x4-sd0" if arm.startswith(("ft-", "dense-")) else "x4-denselora-x4-sd0"
    if arm.startswith(("ft-", "dense")) and "lora" not in arm: return "w1-dense-sd0"
    if "mix" in arm: return None  # other re-mix pools: no dense baseline run yet
    return "v3-denselora-sd0"


def env(c):
    out, best = [], 9.0
    for s, pf, ce, _ in c:
        best = min(best, ce); out.append((s, pf, best))
    return out


def reach(c, lvl):
    e = env(c)
    for a, q in zip(e, e[1:]):
        if q[2] <= lvl and q[1] is not None and a[1] is not None:
            return q[1] if a[2] <= q[2] else a[1] + (a[2] - lvl) / (a[2] - q[2]) * (q[1] - a[1])
    return None


def versus(r, ref):
    """deepest level of ref that r reaches: (ref step, level, x = ref PF / my PF, reaches ref's final level)"""
    if not ref: return None
    R = [e for e in env(ref["curve"])[1:] if e[1] is not None]
    hit = [(s, ce, pf / mine) for s, pf, ce in R for mine in [reach(r["curve"], ce)] if mine]
    if not hit: return None
    s, ce, x = hit[-1]
    return dict(ref_step=s, level=ce, x=x, final=bool(ce <= R[-1][2] + 1e-9), ref_final=R[-1][2], ref_steps=R[-1][0])


REF = runs.get("w1-dense-sd0")
out = []
for name, r in sorted(runs.items()):
    b = base_of(name)
    if b:  # same-seed baseline if one exists under any tag (e.g. w2-denselora-sd1, w1-dense-sd1)
        want = b.split("-", 1)[1].replace("-sd0", "-sd" + name.rsplit("-sd", 1)[1])
        b = next((k for k in sorted(runs) if k.split("-", 1)[1] == want and len(runs[k]["curve"]) > 2), b)
    B = runs.get(b) if b and b != name else None
    bs = {s: ce for s, _, ce, _ in B["curve"]} if B else {}
    track = [(s, ce, bs.get(s)) for s, _, ce, _ in r["curve"][1:]]
    last = r["curve"][-1]
    cost = (last[1] / last[0] / DENSE_PF_STEP) if last[1] and last[0] else None
    out.append(dict(run=name, baseline=b if B else None, done=r["done"], eval_size=r["eval_size"], steps=last[0], pf=last[1],
                    cost_vs_dense_ft=cost, ce16k_last=last[2], track=track, vs_same=versus(r, B), vs_dense_ft=versus(r, REF) if REF is not r else None))
    v = out[-1]["vs_same"]; w = out[-1]["vs_dense_ft"]
    fmt = lambda v: f"step {v['ref_step']:4d}/{v['ref_steps']} x{v['x']:.2f}{'*' if v['final'] else ' '}" if v else "      -          "
    print(f"{name[:26]:26s} {'D' if r['done'] else '.'} cost {cost if cost is None else round(cost, 2)!s:5s} | same-adapter {fmt(v)} | dense FT {fmt(w)} | "
          + " ".join(f"{s}:{ce:.3f}" + (f"({ce - b0:+.3f})" if b0 is not None else "") for s, ce, b0 in track))
print("\ncolumns: cost = PF/step vs dense full FT (4.06); 'step a/b xR' = deepest baseline probe level reached (a of b steps) and FLOP ratio there, * = baseline's final level;\n"
      "track = step:16k CE (difference vs the same-adapter dense baseline at that step). 16k eval_size 320 (below 500).")
if "--json" in sys.argv:
    json.dump(dict(runs=out, eval_size=320, note="16k sets gen16k + rung16k of heldout_v3"), open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
