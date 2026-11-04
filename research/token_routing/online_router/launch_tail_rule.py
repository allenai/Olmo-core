"""Content-blind "keep each document's tail" arms on the token-drop recipe (tokdrop_lr, 2026-11-04).

Question (Prasann): does the model simply adapt to whatever drop shape it trains with? Both arms use the recipe's code
path and base (whole-model LoRA r64, cut after block 7, original position ids, LR 1e-5, mt4, 16 rows/step), with the
router replaced by a fixed rule (``--st-otr-score tail``): each document keeps its LAST body tokens; non-body tokens
(prompt, markers, query, answer) are always kept; no exploration; router FLOPs not charged.

- ``tail10``: one cut at block 7, each document keeps its last 10% (ceil, at least 1).
- ``tail20to7``: the recipe's two-cut shape, last 20% at block 7 then last 7% (nested) at block 16 (FLOP-matched).

    python research/token_routing/online_router/launch_tail_rule.py --arms tail10,tail20to7 --budget f1952 launch
    python research/token_routing/online_router/launch_tail_rule.py --runs NAME eval
"""
import os

import launch_perrow_floor as lpf

HERE = os.path.dirname(os.path.abspath(__file__))
lpf.STATE = f"{HERE}/tail_rule_state.json"
_STAGES = "--st-otr-stages 7:0.2,16:0.07"
assert _STAGES in lpf.RECIPE
lpf.ARMS = {
    "recipe": lpf.ARMS["recipe"],
    "tail10": ("lora64all32t10", lpf.RECIPE.replace(_STAGES, "--st-otr-stages 7:0.1").format(
        share="1.0", extra=" --st-otr-score tail")),
    "tail20to7": ("lora64all32t20to7", lpf.RECIPE.format(share="1.0", extra=" --st-otr-score tail")),
}

if __name__ == "__main__":
    lpf.main()
