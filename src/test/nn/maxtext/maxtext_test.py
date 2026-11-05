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
    AdamState,
    _flatten,
    _maxtext_adam_state,
    _unflatten,
    initialize_maxtext_config,
    load_maxtext_checkpoint,
    load_maxtext_params,
    maxtext_checkpoint_items,
    save_maxtext_checkpoint,
    save_maxtext_params,
    scan_params,
)
from olmo_core.nn.maxtext.convert import maxtext_shapes  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

from .convert_test import assert_states_equal, distinct, load_fixture  # noqa: E402

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


# ------------------------------------------------------------------------------------------------
# Optimizer state
# ------------------------------------------------------------------------------------------------

#: Overrides for building a MaxText training state on CPU.
TRAIN_OVERRIDES = (
    "max_target_length=64",
    "megablox=False",
    "sparse_matmul=False",
    "opt_type=adamw",
    "dtype=float32",
    "weight_dtype=float32",
)


def path_str(path) -> str:
    """A JAX key path as a ``/``-joined string, without NNX's ``value`` wrappers."""
    keys = [str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k)))) for k in path]
    return "/".join(k for k in keys if k not in (".value", "value"))


def without_params_collection(flat: dict) -> dict:
    """Drop the ``params`` collection that the on-disk layout wraps the moments in."""
    out = {}
    for k, v in flat.items():
        parts = k.split("/")
        for i, p in enumerate(parts[:-1]):
            if p in ("mu", "nu") and parts[i + 1] == "params":
                del parts[i + 1]
                break
        out["/".join(parts)] = v
    return out


def scanned_state(config, scan_layers: bool, step: int = 37):
    """Distinct MaxText parameters and AdamW state in the given layout."""
    shapes = maxtext_shapes(config)
    params = distinct(shapes)
    optim = AdamState(step, distinct(shapes, seed=10), distinct(shapes, seed=11))
    if scan_layers:
        scan = lambda m: scan_params(m, config.scan_layout, config.n_layers)  # noqa: E731
        params = scan(params)
        optim = AdamState(step, scan(optim.exp_avg), scan(optim.exp_avg_sq))
    return params, optim


@pytest.mark.parametrize("scan_layers", [False, True], ids=["unscanned", "scanned"])
@pytest.mark.parametrize("name", FIXTURES)
def test_opt_state_matches_maxtext_optimizer(name, scan_layers):
    """The opt_state we write has the leaves MaxText's own adamw optimizer state has."""
    import jax
    from maxtext.optimizers import optimizers
    from maxtext.utils import maxtext_utils

    config = get_maxtext_config(load_fixture(FIXTURES[name]))
    mt_config = initialize_maxtext_config(
        config,
        "run_name=opt_state",
        "enable_checkpointing=False",
        *TRAIN_OVERRIDES,
        scan_layers=scan_layers,
    )
    abstract_params = _unflatten(
        {
            k: jax.ShapeDtypeStruct(s, np.float32)
            for k, s in maxtext_param_shapes(config, scan_layers).items()
        }
    )
    tx = optimizers.get_optimizer(mt_config, maxtext_utils.create_learning_rate_schedule(mt_config))
    expected = {
        path_str(path): tuple(x.shape)
        for path, x in jax.tree_util.tree_flatten_with_path(
            jax.eval_shape(tx.init, abstract_params)
        )[0]
    }

    params, optim = scanned_state(config, scan_layers)
    ours = without_params_collection(_flatten(maxtext_checkpoint_items(params, optim)["opt_state"]))
    assert {k: tuple(np.shape(v)) for k, v in ours.items()} == expected


def test_full_state_orbax_round_trip(tmp_path):
    config = get_maxtext_config(load_fixture(FIXTURES["hybrid_moe"]))
    params, optim = scanned_state(config, scan_layers=True)
    save_maxtext_checkpoint(str(tmp_path), params, optim)
    loaded_params, loaded_optim = load_maxtext_checkpoint(str(tmp_path / "0" / "items"))
    assert_states_equal(loaded_params, params)
    assert loaded_optim is not None
    assert loaded_optim.step == optim.step
    assert_states_equal(loaded_optim.exp_avg, optim.exp_avg)
    assert_states_equal(loaded_optim.exp_avg_sq, optim.exp_avg_sq)


@pytest.mark.parametrize("scan_layers", [False, True], ids=["unscanned", "scanned"])
def test_maxtext_resumes_from_converted_optim_state(tmp_path, scan_layers):
    """
    MaxText's own training-state setup restores a converted full-state checkpoint with our moments
    and step. MaxText keeps its fresh init for anything a restore doesn't find, so this guards
    against a layout mismatch silently resetting the optimizer.
    """
    import jax
    from flax import nnx
    from maxtext.common import train_state_nnx
    from maxtext.utils import maxtext_utils, model_creation_utils, train_utils

    config = get_maxtext_config(load_fixture(FIXTURES["dense"]))
    params, optim = scanned_state(config, scan_layers)
    save_maxtext_checkpoint(str(tmp_path), params, optim)

    mt_config = initialize_maxtext_config(
        config,
        "run_name=optim_resume",
        "enable_checkpointing=False",
        f"load_full_state_path={tmp_path}/0/items",
        *TRAIN_OVERRIDES,
        scan_layers=scan_layers,
    )
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(mt_config), mt_config.mesh_axes)
    create_model, model = model_creation_utils.create_nnx_abstract_model(mt_config, mesh)
    _, tx = train_utils.create_training_optimizer(mt_config, model, mesh=mesh)

    def init_state_fn():
        m = create_model()
        return train_state_nnx.TrainStateNNX(m, nnx.Optimizer(m, tx, wrt=nnx.Param))

    state, *_, was_restored = maxtext_utils.setup_initial_state(
        None, mt_config, mesh, None, init_state_fn
    )
    assert was_restored
    restored = train_state_nnx.to_checkpoint_dict(state)
    assert int(restored["step"]) == optim.step
    found = _maxtext_adam_state(_flatten(restored["opt_state"]), "restored")
    assert found.step == optim.step
    assert_states_equal(found.exp_avg, optim.exp_avg)
    assert_states_equal(found.exp_avg_sq, optim.exp_avg_sq)
    assert_states_equal(
        {
            k: torch.from_numpy(np.asarray(v))
            for k, v in _flatten(restored["params"]["params"]).items()
        },
        params,
    )
