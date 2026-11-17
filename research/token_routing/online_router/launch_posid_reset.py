"""Position-id reset ablation on the token-drop recipe (tokdrop_lr, 2026-11-04).

The recipe (``lora64all32r20to7``) keeps each kept token's ORIGINAL position id after each cut (block 7: 20% of every
document's body, block 16: 7%). This arm adds ``--st-otr-reset-positions``: after each cut the kept row is renumbered
0..k-1, as if the dropped tokens never existed (RoPE in the full-attention blocks sees compacted distances; eval is
dense with true positions). Everything else is the recipe's Beaker command verbatim (``launch_perrow_floor.py``:
fast8k mt4 fresh pool, seq 12288, 16 rows/step, micro 2, 2 GPUs, LR 1e-5, repaired base, suffix -tde).

    python research/token_routing/online_router/launch_posid_reset.py --arms posreset --budget f1952 launch
    python research/token_routing/online_router/launch_posid_reset.py eval
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import launch_perrow_floor as lpf  # noqa: E402

lpf.STATE = f"{HERE}/posid_reset_state.json"
lpf.ARMS = {
    "recipe": lpf.ARMS["recipe"],
    "posreset": ("lora64all32r20to7pr", lpf.RECIPE.format(share="1.0", extra=" --st-otr-reset-positions")),
}

if __name__ == "__main__":
    lpf.main()
