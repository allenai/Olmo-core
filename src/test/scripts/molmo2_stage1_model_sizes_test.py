"""``Molmo2-Stage1.py``'s ``--model_size`` variants: the Qwen3-0.6B / Qwen3-1.7B baselines."""

import importlib.util
import sys

import pytest

from olmo_core.exceptions import OLMoConfigurationError


def _load_stage1_module():
    try:
        import olmo_core.internal.common  # noqa: F401  (needs a recent beaker-py)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Molmo2-Stage1.py imports fail here: {e}")
    spec = importlib.util.spec_from_file_location(
        "_stage1_model_sizes", "src/scripts/train/Molmo2-Stage1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_model_sizes"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


@pytest.mark.parametrize(
    "model_size, d_model, connector_hidden", [("0.6b", 1024, 3072), ("1.7b", 2048, 6144)]
)
def test_small_model_sizes_build_from_scratch(model_size: str, d_model: int, connector_hidden: int):
    mod = _load_stage1_module()
    assert mod._resolve_model_spec([f"--model_size={model_size}"]) == (model_size, "scratch")
    cfg = mod._build_model_config(model_size, "scratch")
    assert cfg.lm.d_model == d_model
    assert cfg.connector.mlp_hidden_size == connector_hidden
    assert cfg.lm.block.sequence_mixer.rope.theta == 1_000_000
    assert cfg.lm.block.masked_dropout == mod.RESPONSE_RESIDUAL_DROPOUT


@pytest.mark.parametrize("model_size", ["0.6b", "1.7b"])
def test_small_model_sizes_reject_init_from_molmo2(model_size: str):
    mod = _load_stage1_module()
    with pytest.raises(OLMoConfigurationError, match="no released Molmo2"):
        mod._resolve_model_spec([f"--model_size={model_size}", "--init_from=molmo2"])


@pytest.mark.parametrize("model_size", ["0.6b", "1.7b"])
def test_small_model_sizes_match_the_qwen3_backbone(model_size: str):
    """The LM config matches the base checkpoint's HF config (only needs the config cached)."""
    transformers = pytest.importorskip("transformers")
    mod = _load_stage1_module()
    spec = mod.MODEL_SIZES[model_size]
    try:
        hf = transformers.AutoConfig.from_pretrained(spec.scratch_lm_id, local_files_only=True)
    except OSError:
        pytest.skip(f"{spec.scratch_lm_id} is not in the local HF cache")
    lm = mod._build_model_config(model_size, "scratch").lm
    assert lm.d_model == hf.hidden_size
    assert lm.n_layers == hf.num_hidden_layers
    assert lm.vocab_size == hf.vocab_size
    assert lm.block.feed_forward.hidden_size == hf.intermediate_size
    # Newer transformers keep the RoPE base in `rope_parameters`.
    rope_theta = getattr(hf, "rope_parameters", None) or {"rope_theta": hf.rope_theta}
    assert lm.block.sequence_mixer.rope.theta == rope_theta["rope_theta"]
    assert lm.tie_word_embeddings is hf.tie_word_embeddings
