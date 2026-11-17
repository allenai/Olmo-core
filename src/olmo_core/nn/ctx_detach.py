"""Context forward-only training ("ctx detach", early-loss-proxy idea ``detach``, 2026-11-16).

Training forward for one-example padded rows ``[context ... last doc-close marker][suffix = query + answer]``:

- blocks ``[0, K)`` run the FULL row with gradient (as usual);
- blocks ``[K, n)`` run in two passes:
  1. a PREFIX pass over the context tokens only, under ``torch.no_grad``; every attention layer records its
     post-RoPE K/V and every Gated-DeltaNet layer its final recurrent state and short-conv window. Optional
     random drops (``stages``: block -> keep fraction of each row's document-BODY tokens, nested, markers and
     non-document tokens always kept) compact the prefix stream in place (survivors keep their original
     positions);
  2. a SUFFIX pass with gradient over the query + answer tokens only, reading the recorded K/V (varlen flash
     attention, bottom-right causal) and continuing the GDN recurrence / conv from the recorded state. The
     recorded context states are constants: no backward and no stored activations for the context in
     blocks ``>= K``.

Exact up to the drops: with no stages the forward equals the plain causal forward (the prefix of a causal
model does not depend on the suffix). The LM head runs only on the suffix.

FLOP accounting (charged through ``model._flop_extra_actual`` on top of the meter's dense-LoRA price at each
row's length): per row with ``n_ctx`` context tokens and real length ``L``, for every block ``j >= K``
``- n_ctx (2/3 f_j(L) + a_j)`` (the dense-LoRA price of those token-blocks: frozen base, adapters a_j) plus
``+ c_j (1/3 f_j(L_j) + a_j / 3)`` (forward only for the ``c_j`` context tokens that ran block j, attention at
the compacted length ``L_j = c_j + n_suffix``).
"""

from typing import Any, Dict, List, Optional

import torch


def _row_lengths(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    is_pad = input_ids == pad_id
    is_pad[:, 0] = False
    has = is_pad.any(dim=1)
    first = is_pad.int().argmax(dim=1)
    return torch.where(has, first + 1, torch.full_like(first, input_ids.shape[1]))


def clear(model) -> None:
    for block in model.blocks.values():
        att = getattr(block, "attention", None)
        if att is not None:
            att._ctx_detach_state = None


def attn_varlen_with_prefix(q, k, v, kp, vp, lens_p, scale):
    """``q, k, v``: (B, S, H, D) suffix (post-RoPE); ``kp, vp``: (B, P, Hkv, D) recorded prefix, row b valid on
    ``[0, lens_p[b])``. Each suffix query attends to its row's prefix plus the causal suffix."""
    from flash_attn import flash_attn_varlen_func

    B, S, H, D = q.shape
    ks, vs, cu = [], [], [0]
    lp = [int(x) for x in lens_p.tolist()]
    for b in range(B):
        ks.append(torch.cat([kp[b, : lp[b]].to(k.dtype), k[b]], dim=0))
        vs.append(torch.cat([vp[b, : lp[b]].to(v.dtype), v[b]], dim=0))
        cu.append(cu[-1] + lp[b] + S)
    kk, vv = torch.cat(ks, 0), torch.cat(vs, 0)
    cu_k = torch.tensor(cu, dtype=torch.int32, device=q.device)
    cu_q = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=q.device)
    out = flash_attn_varlen_func(
        q.reshape(B * S, H, D), kk, vv, cu_q, cu_k, S, max(lp) + S, causal=True, softmax_scale=scale
    )
    return out.view(B, S, H, D)


def ctx_detach_forward(model, input_ids, labels, *, ignore_index, loss_reduction, z_loss_multiplier,
                       loss_div_factor, return_logits, **kwargs):
    from olmo_core.nn.early_lora import early_lora_flops_per_block

    cfg = model._ctx_detach
    K = int(cfg["layer"])
    stages: Dict[int, float] = dict(cfg.get("stages") or {})
    dev = model.device
    input_ids = input_ids.to(dev)
    ids, labels, abk, pbk, lmk = model._prepare_inputs(
        input_ids, labels, ignore_index=ignore_index, loss_reduction=loss_reduction,
        z_loss_multiplier=z_loss_multiplier, loss_div_factor=loss_div_factor, return_logits=return_logits,
        **kwargs,
    )
    if any(k not in ("loss_div_factor",) for k in abk) or pbk:
        raise NotImplementedError(f"ctx detach supports plain padded rows only (block kwargs {list(abk)})")
    B, T = ids.shape
    L = _row_lengths(ids, cfg["pad_id"])  # (B,)
    is_end = ids == cfg["doc_end_id"]
    ar = torch.arange(T, device=dev)
    last_end = torch.where(is_end, ar[None, :], torch.full_like(ids, -1)).max(dim=1).values
    # suffix start = first token after the last doc-close marker, minus an optional grad TAIL of the context
    s = (last_end + 1 - int(cfg.get("tail", 0))).clamp(min=0)
    S_b = L - s
    if labels is not None:
        lab_ctx = ((labels != ignore_index) & (ar[None, :] < s[:, None])).sum()
        if int(lab_ctx) > 0:
            raise RuntimeError(f"ctx detach: {int(lab_ctx)} supervised positions inside the context")
    S = int(S_b.max())
    P = int(s.max())

    h = model.embeddings(ids)
    if model.embed_scale is not None:
        h = h * model.embed_scale
    if model.embedding_norm is not None:
        h = model.embedding_norm(h)
    blocks = list(model.blocks.items())
    for key, blk in blocks:
        if int(key) >= K:
            break
        h = blk(h, **abk)

    # suffix stream (B, S): original positions, padded tail clamped to the row's last token (no labels there)
    sidx = s[:, None] + torch.arange(S, device=dev)[None, :]
    s_valid = sidx < L[:, None]
    sidx_c = torch.minimum(sidx, (L - 1)[:, None]).clamp(max=T - 1)
    h_s = h.gather(1, sidx_c[..., None].expand(-1, -1, h.shape[-1]))
    lab_s = None
    if labels is not None:
        lab_s = labels.gather(1, sidx_c).masked_fill(~s_valid, ignore_index)

    # prefix stream (no grad): context tokens [0, s_b), right-padded to P
    roles_body = None
    if stages:
        # document body = strictly inside a <start> ... <end> span (markers themselves excluded)
        st_c = torch.cumsum((ids == cfg["doc_start_id"]).int(), 1)
        en_c = torch.cumsum(is_end.int(), 1)
        inside = (st_c - en_c) > 0
        roles_body = inside & (ids != cfg["doc_start_id"])
        g = torch.Generator(device="cpu")
        g.manual_seed(int(cfg.get("seed", 0)) * 1000003 + int(cfg["_calls"]))
        u = torch.rand(B, T, generator=g).to(dev)
    cfg["_calls"] = int(cfg["_calls"]) + 1
    clear(model)

    f_cache: Dict[int, List[float]] = {}

    def fj(j, Lx):
        Lx = int(Lx)
        if Lx not in f_cache:
            f_cache[Lx] = [float(b.num_flops_per_token(Lx)) for _, b in blocks]
        return f_cache[Lx][j]

    ab = early_lora_flops_per_block(model)
    extra = 0.0
    with torch.no_grad():
        hp = h[:, :P].detach()
        pos = torch.arange(P, device=dev)[None, :].expand(B, -1).contiguous()
        lens_p = s.clone()
        counts = lens_p.clone()
        for key, blk in blocks:
            j = int(key)
            if j < K:
                continue
            if j in stages:
                frac = float(stages[j])
                valid = torch.arange(hp.shape[1], device=dev)[None, :] < lens_p[:, None]
                body = roles_body.gather(1, pos) & valid
                keep = valid & (~body | (u.gather(1, pos) < frac))
                new_len = keep.sum(1)
                Pn = int(new_len.max()) if B else 0
                order = torch.argsort((~keep).int(), dim=1, stable=True)[:, :Pn]
                hp = hp.gather(1, order[..., None].expand(-1, -1, hp.shape[-1]))
                pos = pos.gather(1, order)
                lens_p = new_len
                counts = new_len
            blk.attention._ctx_detach_state = {"mode": "record", "lens": lens_p}
            hp = blk(hp, position_ids=pos, **abk)
            for b in range(B):
                n_ctx = int(s[b])
                Lb, Sb = int(L[b]), int(S_b[b])
                extra -= n_ctx * ((2.0 / 3.0) * fj(j, Lb) + ab[j])
                c = int(counts[b])
                extra += c * ((1.0 / 3.0) * fj(j, c + Sb) + ab[j] / 3.0)
        # one more record step is not needed: every block >= K recorded its own state above
    del hp
    # mark every recorded layer as "use" (records stay attached for AC recompute in backward)
    for key, blk in blocks:
        att = getattr(blk, "attention", None)
        stt = getattr(att, "_ctx_detach_state", None) if att is not None else None
        if stt is not None:
            stt["mode"] = "use"
    for key, blk in blocks:
        if int(key) < K:
            continue
        h_s = blk(h_s, position_ids=sidx_c, **abk)
    model._flop_extra_actual = float(getattr(model, "_flop_extra_actual", 0.0)) + extra
    st = cfg.setdefault("_stats", {"ctx": 0, "suffix": 0, "kept_last": 0})
    st["ctx"] += int(s.sum())
    st["suffix"] += int(S_b.sum())
    if lab_s is not None:
        lmk["labels"] = lab_s
    return model.lm_head(h_s, **lmk)
