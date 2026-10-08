"""Real-weights KDA: MaxText chunked delta rule vs its exact token scan, plus variants."""
import os, json, sys
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.models import olmoe3
from olmo_core.nn.maxtext.olmoe3 import OLMoE3Geometry, maxtext_overrides

R = "/weka/olmo-3p5-checkpoints/scratch/calebo/olmo35-small-hero-20260907-emo"
g = OLMoE3Geometry.from_olmo_core_config(json.load(open(f"{R}/maxtext-step834466/olmo_core_config.json"))["model"])
tokens = np.load("/root/work/real_tokens.npy")[:1, :256]
S = tokens.shape[1]; pool = g.emo_pools[2]
cfg = pyconfig.initialize(["", os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"), "model_name=olmo35-tiny", "override_model_config=True",
    *maxtext_overrides(g), f"emo_min_document_expert_pool={pool}", f"emo_max_document_expert_pool={pool}", "run_name=d", "enable_checkpointing=True",
    f"load_parameters_path={R}/maxtext-step834466/0/items", "scan_layers=True", "skip_jax_distributed_system=True", f"max_target_length={S}",
    "dtype=float32", "weight_dtype=float32", "matmul_precision=highest", "use_tokamax_kda=False", "per_device_batch_size=1", "megablox=False"])
mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
dec = model.decoder
x = dec.embedding_norm(model.token_embedder(jnp.asarray(tokens, jnp.int32)) * np.sqrt(g.d_model))
cycle0 = dec.layers_0
seg = jnp.ones((1, S), jnp.int32); pos = jnp.arange(S, dtype=jnp.int32)[None]

def kda_inputs(mixer, h):
    """Mirror OLMoE3KimiDeltaAttention.__call__ up to the delta rule (unfused path)."""
    b, s, _ = h.shape; H, dk, dv = mixer_dims
    q = olmoe3.causal_depthwise_conv(mixer.w_q(h), mixer.q_conv[...], seg).reshape(b, s, H, dk)
    k = olmoe3.causal_depthwise_conv(mixer.w_k(h), mixer.k_conv[...], seg).reshape(b, s, H, dk)
    v = olmoe3.causal_depthwise_conv(mixer.w_v(h), mixer.v_conv[...], seg).reshape(b, s, H, dv)
    raw_g = mixer.f_proj_2(mixer.f_proj_1(h)).reshape(b, s, H, dk)
    beta = 2.0 * jax.nn.sigmoid(mixer.w_b(h).astype(jnp.float32))
    q = olmoe3._l2_normalize(q.astype(jnp.float32)) * (dk ** -0.5)
    k = olmoe3._l2_normalize(k.astype(jnp.float32))
    log_decay = -jnp.exp(mixer.A_log[...]).reshape(1, 1, H, 1) * jax.nn.softplus(raw_g.astype(jnp.float32) + mixer.dt_bias[...].reshape(1, 1, H, dk))
    return q, k, v.astype(jnp.float32), log_decay, beta

mixer_dims = (g.kda_heads, g.kda_head_dim, g.kda_head_v_dim)
resets = jnp.zeros((1, S), bool)
def rel(a, b): a, b = np.asarray(a, np.float64), np.asarray(b, np.float64); return np.abs(a - b).max() / np.abs(b).max()

h = x
for j in range(7):  # layers 0..6 (all KDA)
    layer = getattr(cycle0, f"layer_{j}")
    hin = layer.attn_in_norm(h)
    q, k, v, ld, beta = kda_inputs(layer.mixer, hin)
    A = np.exp(np.asarray(layer.mixer.A_log[...]))
    print(f"layer {j}: exp(A_log) range [{A.min():.3g}, {A.max():.3g}], log_decay min {float(ld.min()):.3g}, beta max {float(beta.max()):.3f}")
    scan = olmoe3._delta_rule_scan(q, k, v, jnp.exp(ld), beta, resets)
    for c in (64, 16):
        ch = olmoe3._delta_rule_chunked(q, k, v, ld, beta, resets, c, "float32")
        print(f"   chunked c={c:2d} fp32 vs scan: rel={rel(ch, scan):.2e}")
    # where along the sequence does the chunked output diverge?
    ch = np.asarray(olmoe3._delta_rule_chunked(q, k, v, ld, beta, resets, 64, "float32"), np.float64); sc = np.asarray(scan, np.float64)
    err = np.abs(ch - sc).max(axis=(0, 2, 3)) / np.abs(sc).max()
    print("   per-64-block max err:", " ".join(f"{err[i:i+64].max():.1e}" for i in range(0, S, 64)))
    h, _ = layer(h, seg, pos, True, MODEL_MODE_TRAIN)
