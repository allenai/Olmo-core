"""How the recipe's step cost scales with row length (lenscale Part 1). Run on a compute node (imports torch).

Uses the flop meter's own per-block formulas (``block.num_flops_per_token(L)``, the LM head, the LoRA adapter charge
``6 r (in + out)`` per adapted linear) on a meta-device Qwen3.5-4B and the online router's pricing rule
(``OnlineTokenRouter._mid_compact``): every block is charged x2/3 (frozen base under whole-model LoRA) at the length it
runs at, blocks [0, 7) at the full row length L, blocks [7, 16) at L1 = f1 L, blocks [16, 32) + head at L2 = f2 L,
adapters at the same lengths. f1 / f2 are the measured mean kept-row fractions of the v2 recipe run (mid_row_frac
0.248, s1 0.513 -> 0.127 of the row; prompt / query / answer / markers ~6% of a row are always kept).

    python flop_scaling.py --out flop_scaling.json
"""

import argparse
import json


def causal_pairs_per_token(L):
    return (L + 1) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="flop_scaling.json")
    ap.add_argument("--f1", type=float, default=0.248, help="kept row fraction after block 7 (measured, v2 recipe)")
    ap.add_argument("--f2", type=float, default=0.127, help="kept row fraction after block 16 (measured)")
    ap.add_argument("--prompt", type=int, default=200, help="tokens before the first document (local mask)")
    ap.add_argument("--doc", type=int, default=150, help="mean document length in tokens (local mask)")
    ap.add_argument("--qa", type=int, default=100, help="query + answer tokens that keep full attention")
    a = ap.parse_args()

    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.early_lora import early_lora_flops_per_block, install_early_lora
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320, attn_backend=AttentionBackendName.flash_2)
    m = cfg.build(init_device="meta")
    install_early_lora(m, len(m.blocks), 64, 32.0, check_calls=False)
    blocks = list(m.blocks.values())
    nb = len(blocks)
    ab = early_lora_flops_per_block(m)
    head = float(m.lm_head.num_flops_per_token(0))
    kinds = []
    for b in blocks:
        att = b.attention
        kinds.append("attn" if type(att).__name__ == "Attention" else type(att).__name__)
    print("mixer types:", sorted(set(type(b.attention).__name__ for b in blocks)))
    full = [i for i, k in enumerate(kinds) if k == "attn"]
    a0 = blocks[full[0]].attention
    attn_coef = 12 * a0.n_heads * a0.head_dim  # FLOPs per (query, key) pair, fwd+bwd (the meter's formula)
    K1, K2, FAC = 7, 16, 2.0 / 3.0

    def bf(i, L):
        return float(blocks[i].num_flops_per_token(int(L)))

    rows = []
    for L in (4096, 8192, 16384, 32768, 65536, 131072):
        L1, L2 = a.f1 * L, a.f2 * L
        attn_ctx = attn_coef * causal_pairs_per_token(L)  # length term of one full-attention block, per token
        attn_par = float(a0.num_flops_per_token(0))
        gdn_i = next(i for i, k in enumerate(kinds) if k != "attn")
        gdn = float(blocks[gdn_i].attention.num_flops_per_token(int(L)))
        ffn = float(blocks[0].feed_forward.num_flops_per_token(int(L)))
        dense = float(m.num_flops_per_token(int(L)))  # per token, full fine-tune
        dlora = FAC * sum(bf(i, L) for i in range(nb)) + head + sum(ab)
        # recipe per token of the FULL row (router pricing rule)
        a_ = sum(FAC * bf(i, L) + ab[i] for i in range(K1))
        a_quad = sum(FAC * (bf(i, L) - bf(i, 0)) for i in range(K1) if i in full)
        b_ = a.f1 * sum(FAC * bf(i, L1) + ab[i] for i in range(K1, K2))
        b_quad = a.f1 * sum(FAC * (bf(i, L1) - bf(i, 0)) for i in range(K1, K2) if i in full)
        c_ = a.f2 * (sum(FAC * bf(i, L2) + ab[i] for i in range(K2, nb)) + head)
        c_quad = a.f2 * sum(FAC * (bf(i, L2) - bf(i, 0)) for i in range(K2, nb) if i in full)
        d_ = 0.0  # router: scoring 2F/token + ridge ~2F^2 x 2.5e4 samples/step: < 1e-3 of a step (see report)
        rec = a_ + b_ + c_ + d_
        # document-local attention in the early full-attention blocks: pairs per token ~ prompt + doc/2,
        # plus the query / answer tokens attending the whole row (qa * L / 2 pairs spread over L tokens)
        loc_pairs = a.prompt + a.doc / 2.0 + a.qa * (L / 2.0) / L
        loc_pairs = min(loc_pairs, causal_pairs_per_token(L))
        a_local_quad = sum(FAC * attn_coef * loc_pairs for i in range(K1) if i in full)
        rec_local = rec - a_quad + a_local_quad
        # local in [0,7) AND in the kept tokens of [7,16): kept pairs per token ~ f1 x (prompt-ish) -> local too
        kept_pairs_local = min(a.prompt + a.f1 * a.doc / 2.0 + a.qa / 2.0, causal_pairs_per_token(L1))
        b_local_quad = a.f1 * sum(FAC * attn_coef * kept_pairs_local for i in range(K1, K2) if i in full)
        rec_local2 = rec_local - b_quad + b_local_quad
        rows.append(dict(
            L=L, attn_block_len_term=attn_ctx, attn_block_rest=attn_par, gdn_block_mixer=gdn, ffn=ffn,
            dense_per_token=dense, dense_lora_frac=dlora / dense, recipe_frac=rec / dense,
            a_blocks0_6=a_ / dense, a_quad=a_quad / dense, b_blocks7_15=b_ / dense, b_quad=b_quad / dense,
            c_blocks16plus_head=c_ / dense, c_quad=c_quad / dense, d_router=d_ / dense,
            early_attn_share_of_recipe=a_quad / rec, recipe_frac_local0_7=rec_local / dense,
            recipe_frac_local0_15=rec_local2 / dense,
            all_quad_share_of_recipe=(a_quad + b_quad + c_quad) / rec,
        ))
    out = dict(n_blocks=nb, full_attention_blocks=full, kinds=kinds, head_per_token=head, lora_per_token=sum(ab),
               attn_coef_per_pair=attn_coef, assumptions=vars(a), rows=rows)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    hdr = ("L", "attnLen", "attnRest", "gdn", "ffn", "dense", "dLoRA", "recipe", "a", "a_quad", "b", "b_quad", "c",
           "c_quad", "earlyShare", "rec_loc7", "rec_loc16", "allQuad")
    print(" ".join(f"{h:>9}" for h in hdr))
    for r in rows:
        v = (r["L"], r["attn_block_len_term"] / 1e6, r["attn_block_rest"] / 1e6, r["gdn_block_mixer"] / 1e6,
             r["ffn"] / 1e6, r["dense_per_token"] / 1e9, r["dense_lora_frac"], r["recipe_frac"], r["a_blocks0_6"],
             r["a_quad"], r["b_blocks7_15"], r["b_quad"], r["c_blocks16plus_head"], r["c_quad"],
             r["early_attn_share_of_recipe"], r["recipe_frac_local0_7"], r["recipe_frac_local0_15"],
             r["all_quad_share_of_recipe"])
        print(" ".join(f"{x:9.4g}" for x in v))


if __name__ == "__main__":
    main()
