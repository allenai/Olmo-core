"""Markdown tables for records/early-dense-loss-proxy.md from results_<tag>.json + cost_<tag>.json (no hand copying)."""
import json
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
d = json.load(open(f"results_{tag}.json"))
cost = json.load(open(f"cost_{tag}.json"))
T = ["contradiction", "nq", "outlier", "oolong"]
s = "suite8k"
print("| arm | final 8k | CE step 10 | CE step 20 | excess vs dense LoRA 10 / 20 | outlier drop 20 | contra drop 20 | "
      "frac of dense drop, same step 10 / 20 | frac, same FLOPs 10 | s / step |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a, A in d["arms"].items():
    ff = A["frac_flops"][s]["10"]["frac"]
    print(f"| {a} | {A['final_8k']:.3f} | {A['ce'][s]['10']:.3f} | {A['ce'][s]['20']:.3f} | "
          f"{A['excess_vs_denselora'][s]['10']:+.3f} / {A['excess_vs_denselora'][s]['20']:+.3f} | "
          f"{A['drop_task'][s]['outlier']['20']:.3f} | {A['drop_task'][s]['contradiction']['20']:.3f} | "
          f"{A['frac_step'][s]['10']:.2f} / {A['frac_step'][s]['20']:.2f} | {ff:.1f} | {cost[a]['step_med']:.1f} |")
print()
print("| held-out set | step | Spearman, 4-task mean (all arms) | methods only | contradiction | nq | outlier | oolong |")
print("|---|---|---|---|---|---|---|---|")
for hs in ("suite8k", "suite2k", "train"):
    for st in d["steps"]:
        if st == 0:
            continue
        c = d["corr"]
        g = lambda k: f"{c[f'{hs}/step{st}/{k}']['spearman']:+.2f}"  # noqa: E731
        print(f"| {hs} | {st} | {g('mean')} (n={c[f'{hs}/step{st}/mean']['n_arms']}) | {g('mean_methods_only')} | "
              + " | ".join(g(t) for t in T) + " |")
