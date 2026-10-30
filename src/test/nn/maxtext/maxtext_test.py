"""
Tests of the OLMo Core <-> MaxText mappings against MaxText itself. Skipped unless MaxText (with
JAX) is installed; CPU JAX is enough.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("maxtext")

from olmo_core.nn.maxtext import (  # noqa: E402
    convert_state_to_maxtext,
    get_maxtext_config,
)
from olmo_core.nn.maxtext.checkpoint import (  # noqa: E402
    initialize_maxtext_config,
    load_maxtext_params,
    save_maxtext_params,
    scan_params,
)
from olmo_core.nn.maxtext.convert import maxtext_shapes  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

from .convert_test import load_fixture  # noqa: E402

FIXTURES = {
    "dense": "dense_small_config.json",
    "hybrid_moe": "hybrid_moe_small_config.json",
}


def maxtext_param_shapes(config, scan_layers: bool) -> dict:
    """The parameter shapes MaxText's own init builds, via ``jax.eval_shape``."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from maxtext.layers import quantizations
    from maxtext.models import models
    from maxtext.utils import maxtext_utils

    mt_config = initialize_maxtext_config(
        config,
        "run_name=shapes",
        "enable_checkpointing=False",
        "max_target_length=64",
        "megablox=False",
        "sparse_matmul=False",
        scan_layers=scan_layers,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(mt_config), mt_config.mesh_axes)
    model = models.transformer_as_linen(
        mt_config, mesh, quant=quantizations.configure_quantization(mt_config)
    )
    rng = jax.random.PRNGKey(0)
    ids = jnp.ones((1, 64), jnp.int32)
    params = jax.eval_shape(
        lambda: model.init({"params": rng, "dropout": rng}, ids, ids, enable_dropout=False)
    )
    shapes = {}
    for path, x in jax.tree_util.tree_flatten_with_path(params["params"])[0]:
        keys = [str(getattr(k, "key", k)) for k in path]
        # Flax keeps NNX variables' values under a trailing ".value" key.
        shapes["/".join(k for k in keys if k != ".value")] = tuple(x.shape)
    return shapes


@pytest.mark.parametrize("scan_layers", [False, True], ids=["unscanned", "scanned"])
@pytest.mark.parametrize("name", FIXTURES)
def test_converted_tree_matches_maxtext(name, scan_layers):
    config = get_maxtext_config(load_fixture(FIXTURES[name]))
    shapes = maxtext_shapes(config)
    if scan_layers:
        meta = {k: torch.empty(s, device="meta") for k, s in shapes.items()}
        shapes = {
            k: tuple(v.shape)
            for k, v in scan_params(meta, config.scan_layout, config.n_layers).items()
        }
    assert shapes == maxtext_param_shapes(config, scan_layers)


def maxtext_logits(config, items_path: str, tokens: np.ndarray, scan_layers: bool) -> np.ndarray:
    """
    Run a converted checkpoint in MaxText, in fp32 at the highest matmul precision, and return
    its ``[B, S, V]`` logits. Weights load through MaxText's own ``load_parameters_path``, and
    every loaded parameter is checked against the checkpoint so a model that fell back to its
    own init can't pass.
    """
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.utils import maxtext_utils, model_creation_utils

    B, S = tokens.shape
    mt_config = initialize_maxtext_config(
        config,
        "run_name=olmo_core_parity",
        "enable_checkpointing=True",
        f"load_parameters_path={items_path}",
        f"max_target_length={S}",
        "dtype=float32",
        "weight_dtype=float32",
        "matmul_precision=highest",
        scan_layers=scan_layers,
    )
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(mt_config), mt_config.mesh_axes)
    model = model_creation_utils.from_pretrained(mt_config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)

    stored = load_maxtext_params(items_path)
    loaded = {
        "/".join(str(p) for p in path): np.asarray(v[...])
        for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }
    assert loaded.keys() == stored.keys()
    for k, v in loaded.items():
        assert np.array_equal(v, stored[k].numpy()), f"{k}: MaxText isn't running the checkpoint"

    global_batch = mt_config.global_batch_size_to_load
    positions = jnp.broadcast_to(jnp.arange(S, dtype=jnp.int32), (global_batch, S))
    segments = jnp.ones((global_batch, S), dtype=jnp.int32)
    logits = []
    for row in tokens:
        # One sequence per call, replicated across the global batch.
        ids = jnp.broadcast_to(jnp.asarray(row, dtype=jnp.int32), (global_batch, S))
        out = model(
            decoder_input_tokens=ids,
            decoder_positions=positions,
            decoder_segment_ids=segments,
            enable_dropout=False,
        )
        logits.append(np.asarray(out[0], dtype=np.float32))
    return np.stack(logits)


@pytest.mark.parametrize("scan_layers", [False, True], ids=["unscanned", "scanned"])
def test_dense_logits_match(tmp_path, scan_layers):
    """
    A random dense model gives the same logits in OLMo Core and, after conversion, in MaxText.
    Both run on CPU in fp32.
    """
    model_config = load_fixture("dense_small_config.json")
    config = get_maxtext_config(model_config)
    model = TransformerConfig.from_dict(model_config).build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    with torch.no_grad():
        gen = torch.Generator().manual_seed(0)
        for p in model.parameters():
            # Norm gains start at one; make them distinct so a mix-up can't hide.
            p.add_(0.1 * torch.randn(p.shape, generator=gen))
    model.eval()

    # Longer than the sliding window, so the window matters.
    tokens = np.random.default_rng(0).integers(0, config.vocab_size, size=(2, 32))
    with torch.no_grad():
        expected = model(input_ids=torch.from_numpy(tokens)).float().numpy()

    state = {k: v.detach() for k, v in model.named_parameters()}
    params = convert_state_to_maxtext(config, state)
    if scan_layers:
        params = scan_params(params, config.scan_layout, config.n_layers)
    save_maxtext_params(str(tmp_path), params)
    actual = maxtext_logits(config, str(tmp_path / "0" / "items"), tokens, scan_layers)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max())
    assert np.array_equal(actual.argmax(-1), expected.argmax(-1))
