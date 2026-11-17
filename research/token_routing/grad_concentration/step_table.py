"""Routed-vs-dense gradient agreement along a training run: one analyze_gap.py JSON per checkpoint (labels e<step>).

    python3 research/token_routing/grad_concentration/step_table.py results/gap_e0.json:e0 results/gap_e8.json:e8 ... \
        [--json results/step_table.json]
"""
import json
import sys

SELS = ("rtr_recipe", "rnd_recipe", "og_recipe", "rtr_L7_k0.1", "rtr_L16_k0.03")
TASKS = ("contradiction", "nq", "outlier", "oolong")


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else ""
    if out_json in args:
        args.remove(out_json)
    rows = []
    for arg in args:
        p, lab = arg.rsplit(":", 1)
        d = json.load(open(p))[lab]
        step = int(lab.lstrip("e"))
        for s in SELS:
            if s not in d["raw"]:
                continue
            r, a = d["raw"][s]["whole"], d["adam"][s]["whole"]
            ce = d["ce"][s]
            rows.append(dict(step=step, sel=s, eval_size=d["rows"], exp_cos_raw=r["exp_cos_routed"],
                             exp_cos_adam=a["exp_cos_routed"], adam_proj=a["exp_proj_routed"],
                             norm_ratio_raw=r["exp_norm_ratio_routed"], norm_ratio_adam=a["exp_norm_ratio_routed"],
                             routed_cos_raw=r["routed_cos"][0], pair_cos_R=r["pair_cos_routed"][0],
                             pair_cos_D=r["pair_cos_dense"][0],
                             ce_routed={t: ce[t]["ce_routed"][0] for t in TASKS},
                             ce_dense={t: ce[t]["ce_dense"][0] for t in TASKS}))
    rows.sort(key=lambda x: (SELS.index(x["sel"]), x["step"]))
    print("| sel | step | exp cos raw | exp cos Adam | Adam proj | ‖E[R]‖/‖E[D]‖ raw | per-step cos raw | pair cos R-R (D-D) "
          "| routed CE c/nq/out/ool (dense) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        ce = "/".join(f"{x['ce_routed'][t]:.3f}" for t in TASKS)
        cd = "/".join(f"{x['ce_dense'][t]:.3f}" for t in TASKS)
        print(f"| {x['sel']} | {x['step']} | {x['exp_cos_raw']:.3f} | {x['exp_cos_adam']:.3f} | {x['adam_proj']:.2f} | "
              f"{x['norm_ratio_raw']:.2f} | {x['routed_cos_raw']:.3f} | {x['pair_cos_R']:.3f} ({x['pair_cos_D']:.3f}) | "
              f"{ce} ({cd}) |")
    flag = {x["eval_size"] for x in rows}
    print(f"\neval_size per checkpoint: {sorted(flag)} rows" + (" (< 500: flagged)" if min(flag) < 500 else ""))
    for s in SELS:
        xs = [x for x in rows if x["sel"] == s]
        hit = [x["step"] for x in xs if x["exp_cos_raw"] > 0.8 and x["exp_cos_adam"] > 0.8]
        print(f"{s}: first step with exp cos > 0.8 (raw and Adam): {hit[0] if hit else 'never (in these checkpoints)'}")
    if out_json:
        json.dump(rows, open(out_json, "w"), indent=1)
        print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
