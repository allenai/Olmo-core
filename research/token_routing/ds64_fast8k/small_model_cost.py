"""
Cost ceiling for small-model token selection (2026-10-29): a small model reads every token of a row
and picks which tokens the 4B trains on, so the 4B drops from block 0 instead of block 7.

Priced in the FLOP meter's own convention, at fresh mt4f3904 (976 steps, 15,616 rows):

- 4B side from MEASURED ``flop_meter/actual_pflops`` (pareto_points.json): the recipe drops at
  block 7; r20at1/3/5 move the first drop earlier with the same 20% / 7% budgets. The block-0 point
  is the linear extrapolation of those four (slope ~45 PF per full-length block).
- small model from the meter's per-block formula (``Block.num_flops_per_token``, meta device):
  forward-only = 1/3 of forward + backward (the router's prefix-pass convention); gradient scoring =
  forward + activation backward without weight gradients = 2/3. No LM head for hidden-state
  features; the head on answer tokens only is negligible.

Run with the launcher interpreter from the repo root (meta device only, seconds).
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from olmo_core.nn.transformer.config import TransformerConfig  # noqa: E402

PARETO = Path("/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/research/token_routing/"
              "ds64_fast8k/results/pareto_points.json")
V = 248320
L_MEAN = 7300  # mean row length implied by dense 3,063 PF / (15,616 rows x fpt_4B)


def build(name, n_layers=None):
    kw = {"n_layers": n_layers} if n_layers else {}
    with torch.device("meta"):
        return getattr(TransformerConfig, name)(vocab_size=V, **kw).build(init_device="meta")


def main():
    d = json.loads(PARETO.read_text())
    runs, pts = d["runs"], d["points"]

    def arm(a, lr="1e-05"):
        r = f"f8k-mt4-{a}-f3904-tde-lr{lr}"
        s = [p["score"] for p in pts if p["run"] == r]
        return runs[r]["pf"], sum(s) / len(s)

    dense_pf, _ = arm("dense", "5e-06")
    m4 = build("qwen3_5_4B")
    fpt4 = m4.num_flops_per_token(L_MEAN)
    tok = dense_pf * 1e15 / fpt4  # real tokens trained, dense
    print(f"dense {dense_pf:.0f} PF -> {tok / 1e6:.1f}M tokens (mean row {tok / 15616:.0f})")

    measured = {k: arm(a) for k, a in (("at1", "lora64all32r20at1"), ("at3", "lora64all32r20at3"),
                                       ("at5", "lora64all32r20at5"), ("at7 (recipe)", "lora64all32r20to7"),
                                       ("at7 random choice", "lora64all32rnd20to7"))}
    for k, (pf, s) in measured.items():
        print(f"  4B measured {k:18s} {pf:6.1f} PF  8-cell {s:.3f}")
    xs, ys = [1, 3, 5, 7], [measured[k][0] for k in ("at1", "at3", "at5", "at7 (recipe)")]
    slope = (sum((x - 4) * (y - sum(ys) / 4) for x, y in zip(xs, ys)) / sum((x - 4) ** 2 for x in xs))
    at0 = sum(ys) / 4 - 4 * slope
    recipe = measured["at7 (recipe)"][0]
    # 7% from block 0 (no 20% stage): blocks 0-15 move from the stage-1 to the stage-2 kept length
    # (real kept frac ~0.26 -> ~0.10 incl. query / answer / exploration tokens)
    blk_full = slope / 0.8  # one frozen-LoRA block at full length
    at0_7 = at0 - 16 * blk_full * (0.26 - 0.10)
    print(f"  4B extrapolated block-0 drop (20%/7%): {at0:.0f} PF; 7% from block 0: ~{at0_7:.0f} PF")

    small = {}
    for label, name, nl in (("Qwen3.5-0.8B", "qwen3_5_0_8B", None), ("Qwen3.5-0.8B first 6 blocks", "qwen3_5_0_8B", 6),
                            ("Qwen3.5-2B", "qwen3_5_2B", None)):
        m = build(name, nl)
        blocks = sum(b.num_flops_per_token(L_MEAN) for b in m.blocks.values())
        small[label] = {"fwd": tok * blocks / 3 / 1e15, "grad": tok * blocks * 2 / 3 / 1e15}
    # a non-Qwen3.5 tokenizer model (e.g. ~135M SmolLM-class): ~0.1B non-embedding params, ~1.3x the tokens
    small["~135M other tokenizer"] = {"fwd": tok * 1.3 * 2 * 0.1e9 / 1e15, "grad": tok * 1.3 * 4 * 0.1e9 / 1e15}

    print(f"\n{'selector':30s} {'mode':5s} {'small PF':>8s} | {'20%/7% from blk0':>16s} {'x recipe':>8s} | "
          f"{'7% from blk0':>12s} {'x recipe':>8s}")
    out = {"dense_pf": dense_pf, "recipe_pf": recipe, "at0_pf": at0, "at0_7pct_pf": at0_7, "rows": {}}
    for label, c in small.items():
        for mode in ("fwd", "grad"):
            a, b = at0 + c[mode], at0_7 + c[mode]
            out["rows"][f"{label}/{mode}"] = {"small_pf": c[mode], "total_20_7": a, "total_7": b}
            print(f"{label:30s} {mode:5s} {c[mode]:8.0f} | {a:16.0f} {recipe / a:8.2f} | {b:12.0f} {recipe / b:8.2f}")
    print(json.dumps(out, indent=1), file=open("research/token_routing/ds64_fast8k/small_model_cost.json", "w"))


if __name__ == "__main__":
    main()
