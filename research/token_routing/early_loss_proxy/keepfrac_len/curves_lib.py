"""Probe-curve helpers for the label-validity runs: curve(json) -> [(step, PF, mean CE, {task: CE})]; running-min
interpolation (ce_at), FLOPs to reach a level (pf_to_reach). Reference curves are read from RF (local copies of the
kl16k dft / recipe JSONs)."""
import json, sys, numpy as np
RF = "/tmp/claude-3018/keepfrac_len_results"
T = ["contradiction", "outlier", "qdmatch_nq"]
def curve(p):
    d = json.load(open(p)); rows = d["rows"]
    out = []
    for pr in d["probes"]:
        ce = np.array(pr["ce"]); f = pr.get("flops") or {}
        bt = {t: float(np.mean([c for c, r in zip(ce, rows) if r["task"] == t])) for t in T}
        out.append((pr["step"], (f.get("actual_pf") or 0.0), float(ce.mean()), bt))
    return out, d
def runmin(c):
    m, o = 9, []
    for s, pf, ce, bt in c:
        m = min(m, ce); o.append((pf, m))
    return o
def pf_to_reach(c, level):
    rm = runmin(c)
    for i, (pf, m) in enumerate(rm):
        if m <= level:
            if i == 0: return pf
            p0, m0 = rm[i - 1]
            return p0 + (pf - p0) * (m0 - level) / (m0 - m) if m0 > m else pf
    return None
def ce_at(c, pf):
    rm = runmin(c)
    if rm[-1][0] < pf * 0.98: return None
    for i, (p, m) in enumerate(rm):
        if p >= pf:
            p0, m0 = rm[i - 1]; return m0 + (m - m0) * (pf - p0) / (p - p0)
    return rm[-1][1]
