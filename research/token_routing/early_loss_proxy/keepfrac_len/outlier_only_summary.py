"""Outlier-only 16k arms (results/kl16kol): cost per step, epoch PF, held-out CE per probe, last [gold-doc-drop] line."""
import json, glob, os, re, math, numpy as np, sys
R = "/net/sneetches/data/prasann/early_loss_proxy/results/kl16kol"
out = []
for p in sorted(glob.glob(f"{R}/*.json")):
    d = json.load(open(p))
    arm = os.path.basename(p)[len("elp-kl16kol-"):-len("-sd0.json")]
    ces = {pr["step"]: (float(np.mean(pr["ce"])), float(np.std(pr["ce"], ddof=1) / math.sqrt(len(pr["ce"])))) for pr in d["probes"]}
    st = d["steps"]
    fl = st[-1]["flops"] if st else {}
    log = open(p[:-5] + ".log", errors="ignore").read()
    gl = re.findall(r"\[gold-doc-drop\].*", log)
    cost = fl.get("actual_pf", float("nan")) / fl.get("dense_pf", float("nan")) if fl else float("nan")
    print(f"{arm:20s} done={d.get('done')} steps={len(st)} cost/dense={cost:.4f} actual_PF={fl.get('actual_pf', 0):.1f} dense_PF={fl.get('dense_pf', 0):.1f} eval_size={len(d['probes'][0]['ce'])}")
    print("   CE " + "  ".join(f"s{k}={v[0]:.4f}±{v[1]:.3f}" for k, v in sorted(ces.items())))
    if gl: print("   " + gl[-1])
    out.append({"arm": arm, "json": p.replace("/net/sneetches", ""), "cost": cost, "flops": fl, "ce": ces, "eval_size": len(d['probes'][0]['ce']), "gdd": gl[-1] if gl else "", "job": (d.get("slurm") or {}).get("SLURM_JOB_ID"), "code": (d.get("slurm") or {}).get("CODE")})
json.dump(out, open(sys.argv[1], "w"), indent=1)
