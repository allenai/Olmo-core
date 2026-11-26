"""Beaker side of the checkpoint staging (``stage_ckpts.sh``): load a LoRA training checkpoint (dense LoRA ``--lora-rank``
or the recipe's early LoRA; both are ``olmo_core.nn.early_lora`` adapters, y = W x + scale * B (A x), scale 32 for both
arms here), merge W += scale * B A into the base weights, and re-save a MODEL-ONLY bf16 olmo distcp. The merged model
has exactly the trained forward, and every block linear's per-token weight-gradient term (delta_t x_t^T) is unchanged by
the merge (the adapter is a hook on the same nn.Linear: same input, same output gradient).

    python merge_lora_ckpt.py <ckpt dir with model_and_optim/> <out dir> [--scale 32]
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("ck")
ap.add_argument("out")
ap.add_argument("--scale", type=float, default=32.0)
a = ap.parse_args()
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys, load_model_and_optim_state, save_model_and_optim_state
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

mo = os.path.join(a.ck, "model_and_optim")
keys = list(get_checkpoint_metadata(mo).state_dict_metadata)
lk = sorted(k for k in keys if k.endswith("lora_A") and k.startswith("model."))
print(f"{len(keys)} keys, {len(lk)} LoRA adapters; sample {lk[:2]}", flush=True)
lora = {k: tuple(load_keys(mo, [k, k[:-1] + "B"])) for k in lk}  # before init_process_group (load_keys refuses dist)
dist.init_process_group("gloo", rank=0, world_size=1)
t = time.time()
m = TransformerConfig.qwen3_5_4B(vocab_size=248320, attn_backend=AttentionBackendName("flash_2")).build(init_device="cpu")
load_model_and_optim_state(mo, m)
sd = dict(m.named_parameters())
blocks = set()
dn = 0.0
for k in lk:
    pname = k.split("model.", 1)[-1][:-len(".lora_A")] + ".weight"
    A, B = lora[k]
    with torch.no_grad():
        d = a.scale * (B.float() @ A.float())
        dn += float(d.square().sum())
        sd[pname].copy_((sd[pname].float() + d).to(sd[pname].dtype))
    blocks.add(pname.split(".")[1])
print(f"merged {len(lk)} adapters over {len(blocks)} blocks, |sum dW|_F {dn ** 0.5:.3f}", flush=True)
m = m.to(torch.bfloat16)
save_model_and_optim_state(os.path.join(a.out, "model_and_optim"), m, save_overwrite=True)
json.dump({"source": a.ck, "lora_adapters": len(lk), "blocks": sorted(blocks, key=int), "scale": a.scale,
           "dW_fro": dn ** 0.5}, open(os.path.join(a.out, "merge_info.json"), "w"), indent=1)
print(f"resaved {a.out} ({sum(p.numel() for p in m.parameters()) / 1e9:.2f}B params) in {time.time() - t:.0f}s", flush=True)
