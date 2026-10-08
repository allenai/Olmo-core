"""CPU/JAX: run each MaxText layer on OLMo Core's input to that layer and report the local error."""
import os, json, numpy as np, jax, jax.numpy as jnp
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from olmo_core.nn.maxtext.olmoe3 import OLMoE3Geometry, maxtext_overrides
W = "/root/work/small"
g = OLMoE3Geometry.from_olmo_core_config(json.load(open(f"{W}/mt-unscanned/olmo_core_config.json"))["model"])
pool = g.emo_pools[2]
acts = np.load(f"{W}/blocks.npz"); S = acts["tokens"].shape[1]
cfg = pyconfig.initialize(["", os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"), "model_name=olmo35-tiny", "override_model_config=True",
    *maxtext_overrides(g), f"emo_min_document_expert_pool={pool}", f"emo_max_document_expert_pool={pool}", "run_name=d", "enable_checkpointing=True",
    f"load_parameters_path={W}/mt-unscanned/0/items", "scan_layers=False", "skip_jax_distributed_system=True", f"max_target_length={S}",
    "dtype=float32", "weight_dtype=float32", "matmul_precision=highest", "use_tokamax_kda=False", "per_device_batch_size=1"])
mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
dec = model.decoder
seg = jnp.ones((1, S), jnp.int32); pos = jnp.arange(S, dtype=jnp.int32)[None]
def rel(a, b): return float(np.abs(a - b).max() / np.abs(b).max())
# embedding: scale by sqrt(d) then embedding_norm
emb = np.asarray(dec.embedding_norm(model.token_embedder(jnp.asarray(acts["tokens"], jnp.int32)) * np.sqrt(g.d_model)))
print(f"embed      rel={rel(emb, acts['embed']):.2e}")
prev = acts["embed"]
for i in range(g.n_layers):
    layer = getattr(dec, f"layers_{i}")
    out, _ = layer(jnp.asarray(prev), seg, pos, True, MODEL_MODE_TRAIN)
    kind = "attn" if g.layers[i].attention else "kda "
    print(f"layer {i:2d} {kind} rel={rel(np.asarray(out), acts[f'block{i}']):.2e}  delta-rel={rel(np.asarray(out) - prev, acts[f'block{i}'] - prev):.2e}")
    prev = acts[f"block{i}"]
logits = np.asarray(dec.logits_dense(dec.decoder_norm(jnp.asarray(prev))))
print(f"head       rel={rel(logits, acts['logits']):.2e}")
