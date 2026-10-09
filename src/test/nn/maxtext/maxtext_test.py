"""
Tests of the OLMo Core <-> MaxText mappings against MaxText itself. Skipped unless MaxText (with
JAX) is installed; CPU JAX is enough.
"""

import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("maxtext")

from olmo_core.nn.maxtext import (  # noqa: E402
    get_maxtext_config,
    maxtext_shapes,
    save_maxtext_params,
    scan_params,
)
from olmo_core.nn.maxtext.convert import convert_state_to_maxtext  # noqa: E402
from olmo_core.nn.maxtext.parity import (  # noqa: E402
    compare_summaries,
    default_positions,
    maxtext_logit_summary,
    summarize_logits,
)
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
    from maxtext.configs import pyconfig
    from maxtext.layers import quantizations
    from maxtext.models import models
    from maxtext.utils import maxtext_utils
    from maxtext.utils.globals import MAXTEXT_PKG_DIR

    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        mt_config = pyconfig.initialize(
            [
                "",
                os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
                f"model_name={config.model_name}",
                "override_model_config=True",
                *config.overrides(),
                "run_name=shapes",
                "enable_checkpointing=False",
                f"scan_layers={scan_layers}",
                "skip_jax_distributed_system=True",
                "per_device_batch_size=1",
                "max_target_length=64",
                "megablox=False",
                "sparse_matmul=False",
            ]
        )
    finally:
        sys.argv = argv
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
    positions = default_positions(tokens.shape[1])
    with torch.no_grad():
        logits = model(input_ids=torch.from_numpy(tokens)).float().numpy()
    reference = summarize_logits(tokens, logits, positions)

    state = {k: v.detach() for k, v in model.named_parameters()}
    params = convert_state_to_maxtext(config, state)
    if scan_layers:
        params = scan_params(params, config.scan_layout, config.n_layers)
    save_maxtext_params(str(tmp_path), params)
    actual = maxtext_logit_summary(
        config, str(tmp_path / "0" / "items"), tokens, positions, scan_layers=scan_layers
    )

    metrics = compare_summaries(reference, actual)
    assert metrics["max_rel_logit_err"] < 1e-5, metrics
    assert metrics["top1_agreement"] == 1.0, metrics
