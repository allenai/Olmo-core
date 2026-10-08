"""GPU: per-block hidden states of the OLMo Core reference (strict fp32) for the parity tokens."""
import sys, runpy, json, numpy as np, torch, triton.language as tl
sys.argv = ["x"]  # import the reference script as a module without running main
ref = runpy.run_path("src/scripts/maxtext/olmoe3_reference_logits.py", run_name="ref")
for name, mod in list(sys.modules.items()):
    if name.startswith("fla.") and getattr(mod, "SOLVE_TRIL_DOT_PRECISION", None) is not None:
        mod.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
torch.set_float32_matmul_precision("highest")
from olmo_core.nn.maxtext.checkpoint import load_olmo_core_weights, load_olmo_core_config
from olmo_core.nn.maxtext.olmoe3 import OLMoE3Geometry
ckpt, out = "/root/work/small/olmo", "/root/work/small/blocks.npz"
cfg = load_olmo_core_config(ckpt)["model"]
g = OLMoE3Geometry.from_olmo_core_config(cfg)
model = ref["build_model"](cfg, torch.device("cuda"))
model.load_state_dict({k: torch.from_numpy(v) for k, v in load_olmo_core_weights(ckpt, g).items()})
model.eval()
tokens = np.load("/root/work/small/ref192.npz")["tokens"][:1].astype(np.int64)
acts = {}
model.embedding_norm.register_forward_hook(lambda m, i, o: acts.__setitem__("embed", o.detach().float().cpu().numpy()))
for i, blk in enumerate(model.blocks.values()):
    blk.register_forward_hook(lambda m, inp, o, i=i: acts.__setitem__(f"block{i}", (o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy()))
with torch.no_grad():
    logits = model(input_ids=torch.from_numpy(tokens).cuda()).float().cpu().numpy()
np.savez(out, tokens=tokens, logits=logits, **acts)
print("saved", sorted(acts)[:3], len(acts), logits.shape)
