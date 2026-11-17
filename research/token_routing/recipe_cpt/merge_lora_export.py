"""
Turn a recipe-CPT checkpoint (whole-model LoRA r64 / scale 32 on a frozen base, trained LM head) into a plain
Qwen3.5-4B base that any arm can start from with ``--base-checkpoint <out>/model_and_optim``.

    W_merged = W + scale * B @ A        (every adapted nn.Linear, fp32; early_lora.py: y = W x + scale * B (A x))

Every other tensor (embeddings, norms, GDN params, the trained LM head) is copied as is. The output holds exactly
the keys of a plain ``TransformerConfig.qwen3_5_4B`` model, so it loads strictly into the dense ``full`` arm and
tolerantly into the softtoken / router arms, the same as the marker-repaired base it replaces. The embeddings are
frozen during recipe CPT, so the marker rows stay repaired (checked below against the original base if given).

Input: a step checkpoint (``<run>/stepN/model_and_optim``, model + optim + train state; only the model is read) or a
post-fit model-only export (``<run>/model_and_optim``). The input MUST hold LoRA keys unless ``--no-lora`` (dense
control checkpoints: then this is a model-only re-save, dropping the optimizer, so SFT reads 8 GB, not ~55 GB).

    python merge_lora_export.py --ckpt <run>/step242/model_and_optim --out <dir> [--base <base>/model_and_optim]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.distributed.checkpoint import FileSystemReader

from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.nn.early_lora import install_early_lora
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320


def log(m):
    print(f"[merge-lora] {m}", flush=True)


def ckpt_keys(path):
    return list(FileSystemReader(path).read_metadata().state_dict_metadata)


def build():
    cfg = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB)
    return cfg, cfg.build(init_device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".../model_and_optim (distcp) of the CPT run")
    ap.add_argument("--out", required=True, help="output dir; writes <out>/model_and_optim + <out>/config.json")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--scale", type=float, default=32.0)
    ap.add_argument("--n-blocks", type=int, default=32)
    ap.add_argument("--no-lora", action="store_true", help="dense checkpoint: model-only re-save, no merge")
    ap.add_argument("--base", default="", help="original base model_and_optim: report how far the merged weights moved")
    a = ap.parse_args()
    t0 = time.time()
    keys = ckpt_keys(a.ckpt)
    n_lora = sum(k.endswith(".lora_A") for k in keys)
    log(f"{a.ckpt}: {len(keys)} keys, {n_lora} LoRA adapters")
    if a.no_lora and n_lora:
        raise SystemExit("--no-lora but the checkpoint holds LoRA adapters: refusing (would drop them)")
    if not a.no_lora and n_lora == 0:
        raise SystemExit("no LoRA keys in the checkpoint: pass --no-lora for a dense checkpoint")
    cfg, model = build()
    if not a.no_lora:
        mods = install_early_lora(model, a.n_blocks, a.rank, a.scale, check_calls=False)
        if len(mods) != n_lora:
            raise SystemExit(f"installed {len(mods)} adapters but the checkpoint has {n_lora}")
    load_model_and_optim_state(a.ckpt, model)
    log(f"loaded in {time.time() - t0:.0f}s")
    sd = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.endswith((".lora_A", ".lora_B"))}
    if not a.no_lora:
        name = {m: n for n, m in model.named_modules()}
        dn = []
        for m in model._early_lora["modules"]:
            k = name[m] + ".weight"
            w = sd[k].float()
            delta = m._lora_scale * (m.lora_B.float() @ m.lora_A.float())
            sd[k] = (w + delta).to(sd[k].dtype)
            dn.append(float(delta.norm() / w.norm()))
        log(f"merged {len(dn)} linears: ||scale B A|| / ||W|| mean {sum(dn) / len(dn):.2e} max {max(dn):.2e}")
    _, plain = build()
    plain.load_state_dict(sd, strict=True)
    if a.base:
        _, base = build()
        load_model_and_optim_state(a.base, base)
        bsd = base.state_dict()
        moved = {k: float((sd[k].float() - bsd[k].float()).norm() / (bsd[k].float().norm() + 1e-12)) for k in bsd}
        emb = [k for k in moved if "embeddings" in k]
        log(f"vs base: embeddings moved {max(moved[k] for k in emb):.2e} (frozen in recipe CPT -> 0 expected); "
            f"largest relative change {max(moved.values()):.2e} ({max(moved, key=moved.get)})")
    os.makedirs(a.out, exist_ok=True)
    save_model_and_optim_state(os.path.join(a.out, "model_and_optim"), plain, save_overwrite=True)
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.ckpt).rstrip("/")))
    src_cfg = None
    for cand in (os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "config.json"), os.path.join(run_dir, "config.json")):
        if os.path.exists(cand):
            src_cfg = json.load(open(cand))
            break
    out_cfg = src_cfg or {"model": cfg.as_config_dict()}
    for k in list(out_cfg):
        if k not in ("model", "dataset"):
            out_cfg[k] = None  # merged: no adapters, no routing to re-enable at eval
    out_cfg["merged_from"] = {"ckpt": os.path.abspath(a.ckpt), "lora": None if a.no_lora else
                              {"rank": a.rank, "scale": a.scale, "n_blocks": a.n_blocks}}
    json.dump(out_cfg, open(os.path.join(a.out, "config.json"), "w"))
    log(f"wrote {a.out}/model_and_optim (+config.json) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
