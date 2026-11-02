"""Parity of the soft-keep gate path at g = 1 against the dense flash forward (support_probe.py's gate score).

Per row (one per task), answer CE (summed) of: dense flash (routed_probe.run_row); soft_keep = 1 in blocks >= L with
the default SDPA kernel (forward only), the math kernel, soft_keep only on attention blocks / only on GDN blocks
(math), and the default kernel on a sequence padded at the END to a multiple of --pad (causal, so earlier logits are
unchanged) with backward (does the memory-efficient backward run there?).

    python research/token_routing/grad_concentration/soft_parity.py --ckpt <dir> --shards <dir> --out x.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe as P  # noqa: E402
import routed_probe as RP  # noqa: E402
import support_probe as SP  # noqa: E402


def soft_ce(model, r, L, gval, kernel=None, only=None, pad=0, backward=False):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    x, pred, tgt = r["x"], r["pred"], r["tgt"]
    body = r["body"]
    T0 = x.numel()
    if pad:
        Tp = -(-T0 // pad) * pad
        x = torch.cat([x, x.new_full((Tp - T0,), int(x[-1]))])
        body = torch.cat([body, body.new_zeros(Tp - T0)])
    T = x.numel()
    _, _, abk, pbk, lmk = model._prepare_inputs(x[None], None, logits_to_keep=pred[None])
    bidx = torch.nonzero(body, as_tuple=True)[0]
    z = torch.full((bidx.numel(),), float(gval), device=x.device, requires_grad=backward)
    ctx = SP.fp32_attn() if kernel == "FP32" else (sdpa_kernel([getattr(SDPBackend, kernel)]) if kernel else nullcontext())
    with ctx:
        with torch.set_grad_enabled(backward):
            with torch.no_grad():
                h = model.embeddings(x[None])
            with set_checkpoint_early_stop(False):
                for key, block in model.blocks.items():
                    b = int(key)
                    kw = {**abk, **pbk.get(b, {})}
                    is_gdn = type(block.attention).__name__ == "GatedDeltaNet"
                    use = b >= L and (only is None or (only == "gdn") == is_gdn)
                    if not use:
                        if backward and b >= L:
                            h = checkpoint(lambda hh, bl=block, k=kw: bl(hh, **k), h, use_reentrant=False)
                        else:
                            with torch.no_grad() if not backward or b < L else nullcontext():
                                h = block(h, **kw)
                        continue
                    sk = torch.ones(T, device=x.device, dtype=torch.float32).index_put((bidx,), z)
                    if backward:
                        h = checkpoint(lambda hh, s, bl=block, k=kw: bl(hh, soft_keep=s[None], **k), h, sk,
                                       use_reentrant=False)
                    else:
                        h = block(h, soft_keep=sk[None], **kw)
            ce = F.cross_entropy(model.lm_head(h, **lmk)[0].float(), tgt, reduction="sum")
        gn = None
        if backward:
            ce.backward()
            gn = float(z.grad.abs().sum())
    return float(ce.detach()), gn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", default=P.SHARDS)
    ap.add_argument("--label", default="")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--split", type=int, default=7)
    ap.add_argument("--pad", type=int, default=64)
    ap.add_argument("--ref", action="store_true", help="fp32 reference: torch attention backend, dense bf16 / dense "
                    "fp32 / soft g=1 fp32 (whole model in fp32)")
    ap.add_argument("--out", required=True)
    a, _ = ap.parse_known_args()
    if a.ref:
        return ref_main(a)
    model = P.build_model(P.find_ckpt(a.ckpt), "flash_2")
    lins = P.list_linears(model)
    RP.merge_lora(model, lins, P.find_ckpt(a.ckpt), 32.0)
    tasks = ["contradiction", "nq", "outlier", "oolong"]
    seen = [os.path.join(a.shards, s) for s in ("mt4f7808", "mt4s1952")]
    rbt, _ = P.load_rows(a.shards, tasks, a.rows, 7808, 12288, seen)
    tap = argparse.Namespace(cur={})
    out = []
    L = a.split
    for t in tasks:
        for rr in rbt[t]:
            r = RP.prep_row(rr, t)
            t0 = time.time()
            with torch.no_grad():
                dense, _, _ = RP.run_row(model, tap, r, ckpt=False)
            res = {"task": t, "T": r["T"], "n_answer": int(r["pred"].numel()), "dense_flash": float(dense)}
            for name, kw in (("soft1_fp32_fwd", {"kernel": "FP32"}), ("soft1_fp32_bwd", {"kernel": "FP32", "backward": True}),
                             ("soft05_fp32_bwd", {"kernel": "FP32", "backward": True, "g": 0.5}),
                             ("soft1_fp32_attn_only", {"kernel": "FP32", "only": "attn"}),
                             ("soft1_default_fwd", {}), ("soft1_math_fwd", {"kernel": "MATH"}),
                             ("soft1_efficient_fwd", {"kernel": "EFFICIENT_ATTENTION"}),
                             ("soft1_math_attn_only", {"kernel": "MATH", "only": "attn"}),
                             ("soft1_math_gdn_only", {"kernel": "MATH", "only": "gdn"}),
                             ("soft1_efficient_gdn_only", {"kernel": "EFFICIENT_ATTENTION", "only": "gdn"}),
                             ("soft1_efficient_pad_bwd", {"kernel": "EFFICIENT_ATTENTION", "pad": a.pad, "backward": True}),
                             ("soft05_efficient_pad_bwd", {"kernel": "EFFICIENT_ATTENTION", "pad": a.pad, "backward": True,
                                                           "g": 0.5}),
                             ("soft05_math_bwd", {"kernel": "MATH", "backward": True, "g": 0.5})):
                g = kw.pop("g", 1.0)
                try:
                    ce, gn = soft_ce(model, r, L, g, **kw)
                    res[name] = ce
                    if gn is not None:
                        res[name + "_gradsum"] = gn
                except Exception as exc:  # noqa: BLE001
                    res[name] = f"ERR {type(exc).__name__}: {str(exc)[:120]}"
                torch.cuda.empty_cache()
            res["seconds"] = time.time() - t0
            print(json.dumps(res), flush=True)
            out.append(res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


def ref_main(a):
    tasks = ["contradiction", "nq", "outlier", "oolong"]
    seen = [os.path.join(a.shards, s) for s in ("mt4f7808", "mt4s1952")]
    rbt, _ = P.load_rows(a.shards, tasks, a.rows, 7808, 12288, seen)
    tap = argparse.Namespace(cur={})
    res = {}
    for dt in ("flash_bf16", "torch_bf16", "torch_fp32"):
        model = P.build_model(P.find_ckpt(a.ckpt), "flash_2" if dt == "flash_bf16" else "torch")
        RP.merge_lora(model, P.list_linears(model), P.find_ckpt(a.ckpt), 32.0)
        if dt == "torch_fp32":
            model = model.float()
        for t in tasks:
            for j, rr in enumerate(rbt[t]):
                r = RP.prep_row(rr, t)
                o = res.setdefault(f"{t}{j}", {"task": t, "T": r["T"], "n_answer": int(r["pred"].numel())})
                for name, fn in (("dense", lambda: RP.run_row(model, tap, r, ckpt=False)[0]),
                                 ("soft1", lambda: soft_ce(model, r, a.split, 1.0, kernel="FP32")[0])):
                    try:
                        with torch.no_grad():
                            v = fn()
                        o[f"{name}_{dt}"] = float(v)
                    except Exception as exc:  # noqa: BLE001
                        o[f"{name}_{dt}"] = f"ERR {type(exc).__name__}: {str(exc)[:160]}"
                    torch.cuda.empty_cache()
                print(json.dumps(o), flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(list(res.values()), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
