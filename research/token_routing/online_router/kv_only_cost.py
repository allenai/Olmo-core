"""FLOP cost model for KV-only context (2026-10-29): what keeping dropped tokens as stale K/V context costs.

The online router (plain drop) deletes a dropped token: blocks >= L run on the kept tokens only. KV-only context
keeps every dropped token frozen at its h_L and still feeds it to every later block's sequence mixer:
  attention block   the dropped token's K/V projections (+ their LoRA adapters) run; kept queries attend ALL keys
                    (causal pairs ~ kept x T instead of kept x kept);
  GDN block         the dropped token's q/k/v/a/b projections + short convs (+ adapters) run (q because the kept
                    tokens' q conv windows reach their dropped neighbours; an upper bound) and the chunked recurrent
                    kernel runs over the full row (all 4 state terms charged for every token; an upper bound);
  FFN, output gate / out projection, LM head: kept tokens only (as plain drop).
Modes: kv = gradient flows through the stale context (dropped projections pay forward + input gradient: the frozen
base's 2/3, adapters full); kvdet = the stale context is detached (base projections forward only, 1/3; adapters
forward + weight gradient, 2/3). kva / kvadet = the same at the attention blocks only: GDN blocks run on the kept tokens alone (the recurrent state
never sees a dropped token), so the context costs only K/V projections + kept x all attention scores.
Everything else follows the FLOP meter's convention for the recipe
(online_token_router._block_factors): frozen-base (LoRA) blocks pay 2/3 of the block (attention scores included),
adapters 6 r (in + out) per adapted linear, the trained LM head at the kept positions. Dense = full fine-tuning at
the full row length.

Rows: the 4 mt4 tasks' mean (T, body) from the gradfid2 probe rows (8k pool), and the same layout stretched to 32k.
Non-body tokens (headers, question, answer) are always kept and placed at the row end for the attention pair count.

    python kv_only_cost.py --out kv_only_cost.json      # CPU only; builds Qwen3.5-4B on the meta device
"""
from __future__ import annotations

import argparse
import json

# mt4 per-task mean (T, n_body) of the gradfid2 held-out rows (mt4f1952, 64 rows per task)
ROWS_8K = {"contradiction": (8372.6, 7858.0), "nq": (7663.2, 7502.9), "oolong": (3846.6, 3525.9),
           "outlier": (8611.7, 8362.2)}
SETTINGS = {
    "recipe 7:0.2,16:0.07": [(7, 0.2), (16, 0.07)],
    "1:0.2,16:0.07": [(1, 0.2), (16, 0.07)],
    "3:0.2,16:0.07": [(3, 0.2), (16, 0.07)],
    "1:0.1": [(1, 0.1)],
    "3:0.1": [(3, 0.1)],
    "7:0.1": [(7, 0.1)],
    "1:0.1,16:0.07": [(1, 0.1), (16, 0.07)],
    "1:0.05": [(1, 0.05)],
}
RANK = 64


def model_spec():
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320, attn_backend=AttentionBackendName("flash_2"))
    m = cfg.build(init_device="meta")
    blocks = []
    lin = lambda mod: (mod.in_features, mod.out_features)  # noqa: E731
    for i, b in enumerate(m.blocks.values()):
        a = b.attention
        ffn = sum(p.numel() for p in b.feed_forward.parameters())
        ffn_lin = [lin(x) for x in b.feed_forward.modules() if x.__class__.__name__ == "Linear"]
        if hasattr(a, "w_a"):  # gated delta net
            ctx = list(a._qkv_projs()) + [a.w_a, a.w_b]
            out = [a.w_g, a.w_out]
            conv = 2 * a.conv_size * (2 * a.key_dim + a.value_dim)  # forward, per token
            blocks.append(dict(kind="gdn", ctx=[lin(x) for x in ctx], out=[lin(x) for x in out], conv=conv,
                               state=a.n_v_heads * a.head_k_dim * a.head_v_dim, ffn=ffn, ffn_lin=ffn_lin))
        else:
            blocks.append(dict(kind="attn", q=[lin(a.w_q)], kv=[lin(a.w_k), lin(a.w_v)], out=[lin(a.w_out)],
                               n_heads=a.n_heads, head_dim=a.head_dim, ffn=ffn, ffn_lin=ffn_lin))
    d = m.embeddings.weight.shape[1]
    head = 6 * m.embeddings.weight.shape[0] * d  # fwd+bwd per position (lm_head w_out)
    return dict(blocks=blocks, head=head, d_model=d)


def P(ls):  # params of a list of (in, out) linears
    return sum(i * o for i, o in ls)


def A(ls):  # LoRA adapter fwd+bwd FLOPs per token
    return sum(6.0 * RANK * (i + o) for i, o in ls)


def row_cost(spec, T, Tb, stages, mode, lora=True):
    """FLOPs (fwd + bwd) of one row. mode: dense (full FT, no drop), plain, kv, kvdet, kva, kvadet."""
    att_only = mode in ("kva", "kvadet")
    det = mode in ("kvdet", "kvadet")
    Tn = T - Tb
    fac = 2.0 / 3.0 if lora else 1.0
    at = dict(stages)
    keep_b = 1.0
    out = dict(total=0.0, ctx_extra=0.0)
    for i, b in enumerate(spec["blocks"]):
        if mode != "dense" and i in at:
            keep_b = at[i]
        n = Tn + keep_b * Tb  # tokens updating at this block
        dropped = T - n
        allp = (P(b["q"]) + P(b["kv"]) + P(b["out"])) if b["kind"] == "attn" else (P(b["ctx"]) + P(b["out"]))
        lin_all = (b["q"] + b["kv"] + b["out"] + b["ffn_lin"]) if b["kind"] == "attn" else (b["ctx"] + b["out"] + b["ffn_lin"])
        base = n * (6.0 * (allp + b["ffn"]))
        if b["kind"] == "gdn":
            base += n * 3.0 * (b["conv"] + 8.0 * b["state"])
        if b["kind"] == "attn":
            # causal (query, key) pairs: kept body uniformly spread, non-body at the row end
            if mode in ("kv", "kvdet", "kva", "kvadet"):
                pairs = keep_b * Tb * (Tb / 2.0) + Tn * T
            else:
                pairs = keep_b * Tb * (keep_b * Tb / 2.0) + Tn * n
            base += 12.0 * b["n_heads"] * b["head_dim"] * pairs
        c = fac * base + (n * A(lin_all) if lora else 0.0)
        extra = 0.0
        if mode in ("kv", "kvdet", "kva", "kvadet") and dropped > 0 and not (att_only and b["kind"] == "gdn"):
            pf, af = (fac, 1.0) if not det else (1.0 / 3.0, 2.0 / 3.0)
            if not lora:
                pf, af = (1.0, 0.0) if not det else (2.0 / 3.0, 0.0)  # full FT: kvdet skips only the dgrad
            if b["kind"] == "attn":
                extra = dropped * (pf * 6.0 * P(b["kv"]) + (af * A(b["kv"]) if lora else 0.0))
            else:
                extra = dropped * (pf * 6.0 * P(b["ctx"]) + pf * 3.0 * b["conv"] + (af * A(b["ctx"]) if lora else 0.0)
                                   + fac * 3.0 * 8.0 * b["state"])  # kernel over the full row (gradient flows)
        out["total"] += c + extra
        out["ctx_extra"] += extra
    n_head = T if mode == "dense" else Tn + keep_b * Tb
    out["total"] += n_head * spec["head"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kv_only_cost.json")
    a = ap.parse_args()
    spec = model_spec()
    kinds = "".join("A" if b["kind"] == "attn" else "g" for b in spec["blocks"])
    print(f"blocks: {kinds}  d_model={spec['d_model']}")
    rows = {"8k": ROWS_8K, "32k": {t: (32768.0, 32768.0 - (T - Tb)) for t, (T, Tb) in ROWS_8K.items()}}
    res = {}
    for length, rr in rows.items():
        dense = sum(row_cost(spec, T, Tb, [], "dense", lora=False)["total"] for T, Tb in rr.values())
        res[length] = {"dense_pflops_per_4rows": dense / 1e15}
        print(f"\n=== {length}: dense full-FT {dense / 4 / 1e12:.1f} TF/row (mean of the 4 task rows); "
              f"x16 rows/step = {dense / 4 * 16 / 1e15:.2f} PF/step")
        print(f"{'setting':22s} {'plain':>7s} {'kv':>7s} {'kvdet':>7s} {'kva':>7s} {'kvadet':>7s}  (x dense)")
        for name, st in SETTINGS.items():
            r = {}
            for mode in ("plain", "kv", "kvdet", "kva", "kvadet"):
                c = [row_cost(spec, T, Tb, st, mode) for T, Tb in rr.values()]
                tot = sum(x["total"] for x in c)
                r[mode] = dict(x_dense=tot / dense, tf_per_row=tot / 4 / 1e12, pf_per_step16=tot / 4 * 16 / 1e15,
                               ctx_extra_share=sum(x["ctx_extra"] for x in c) / tot)
            res[length][name] = r
            print(f"{name:22s} " + " ".join(f"{r[m]['x_dense']:7.3f}" for m in r)
                  + f"  | ctx share kv {r['kv']['ctx_extra_share']:.0%} kva {r['kva']['ctx_extra_share']:.0%} | PF/step "
                  + " ".join(f"{m} {r[m]['pf_per_step16']:.2f}" for m in r))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
