"""``MultimodalLMConfig.molmo2_4B`` must stay identical to the HF-derived config.

The native factory exists so training from base checkpoints needs nothing from the
released ``allenai/Molmo2-4B`` repo. That is only safe while it produces exactly what
:func:`molmo2_config_from_hf_config` produces, so this test pins the two together.
"""

import pytest

from olmo_core.nn.vision import MultimodalLMConfig

from ._molmo2_common import _hf_cache_has

transformers = pytest.importorskip("transformers")

VARIANTS = [
    ("allenai/Molmo2-4B", "molmo2_4B"),
    ("allenai/Molmo2-8B", "molmo2_8B"),
]


def _flatten(d, prefix=""):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


@pytest.mark.parametrize("model_id, factory_name", VARIANTS)
def test_native_config_matches_hf_derived(model_id: str, factory_name: str):
    if not _hf_cache_has(model_id):
        pytest.skip(f"{model_id} not in local HF cache")

    from transformers import AutoConfig

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_config_from_hf_config,
    )

    ensure_default_rope_registered()
    hf_derived = molmo2_config_from_hf_config(
        AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    )
    native = getattr(MultimodalLMConfig, factory_name)()

    expected, actual = _flatten(hf_derived.as_config_dict()), _flatten(native.as_config_dict())
    differences = {
        key: (expected.get(key), actual.get(key))
        for key in expected.keys() | actual.keys()
        if expected.get(key) != actual.get(key)
    }
    assert not differences, f"{factory_name} diverged from the HF-derived config: {differences}"


def test_rope_theta_defaults_match_the_released_checkpoints():
    """Molmo2-4B is the only released variant whose base differs from its Qwen3 backbone's."""
    assert MultimodalLMConfig.molmo2_4B().lm.block.sequence_mixer.rope.theta == 5_000_000
    assert MultimodalLMConfig.molmo2_8B().lm.block.sequence_mixer.rope.theta == 1_000_000
    # Both must accept the base-Qwen3 value used when initialising from scratch.
    for factory in (MultimodalLMConfig.molmo2_4B, MultimodalLMConfig.molmo2_8B):
        assert factory(rope_theta=1_000_000).lm.block.sequence_mixer.rope.theta == 1_000_000


@pytest.mark.parametrize(
    "factory_name, d_model, ffn_hidden_size, tied",
    [("molmo2_4B", 2560, 9728, True), ("molmo2_8B", 4096, 12288, False)],
)
def test_native_config_shape(factory_name: str, d_model: int, ffn_hidden_size: int, tied: bool):
    """Layout invariants that do not need the HF checkpoint cached."""
    cfg = getattr(MultimodalLMConfig, factory_name)()
    assert cfg.lm.d_model == d_model
    assert cfg.lm.n_layers == 36
    assert cfg.lm.block.feed_forward.hidden_size == ffn_hidden_size
    # The table is split: `vocab_size` is the base vocab, the 128 image-special tokens live
    # in a separate `embeddings.extra_weight` block, and the LM head spans the base only —
    # so no logit masking (`output_vocab_size`) is needed.
    assert cfg.lm.vocab_size == 151_936
    assert cfg.lm.n_extra_vocab == 128
    assert cfg.lm.tie_word_embeddings is tied
    assert cfg.output_vocab_size is None
    # The vision stack is shared across variants.
    assert cfg.vision.image_num_layers == 25  # SigLIP2 blocks 0..24 of 27
    assert cfg.vit_layers == (24, 18)
    assert max(cfg.vit_layers) == cfg.vision.image_num_layers - 1
    assert cfg.connector.num_input_layers == len(cfg.vit_layers)
    assert cfg.connector.output_dim == cfg.lm.d_model
    assert cfg.connector.mlp_hidden_size == ffn_hidden_size
