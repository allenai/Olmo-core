import json
from typing import Any, Callable

import pytest
from cached_path import cached_path

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)


def _block(**norm_kwargs) -> TransformerBlockConfig:
    return TransformerBlockConfig(
        name=TransformerBlockType.default,
        sequence_mixer=AttentionConfig(name=AttentionType.default, n_heads=8),
        feed_forward=FeedForwardConfig(hidden_size=512),
        **norm_kwargs,
    )


def test_block_config_layer_norm_back_compat():
    """Legacy ``layer_norm=`` is mirrored into both split norm fields and not stored itself."""
    ln = LayerNormConfig()
    block = _block(layer_norm=ln)
    assert block.attention_norm is ln
    assert block.feed_forward_norm is ln

    # ``layer_norm`` is an InitVar, so it must not survive serialization.
    serialized = block.as_config_dict()
    assert "layer_norm" not in serialized
    assert "attention_norm" in serialized and "feed_forward_norm" in serialized

    # Round-trip should be lossless and still resolve to the split fields.
    roundtripped = TransformerBlockConfig.from_dict(serialized)
    assert roundtripped.attention_norm is not None
    assert roundtripped.feed_forward_norm is not None


def test_block_config_split_norms_independent():
    """The two norm fields can be configured independently."""
    attn_norm = LayerNormConfig(bias=False)
    ff_norm = LayerNormConfig(bias=True)
    block = _block(attention_norm=attn_norm, feed_forward_norm=ff_norm)
    assert block.attention_norm is attn_norm
    assert block.feed_forward_norm is ff_norm


def test_block_config_layer_norm_conflict_raises():
    """Specifying both the legacy and split fields is an error."""
    with pytest.raises(OLMoConfigurationError):
        _block(layer_norm=LayerNormConfig(), attention_norm=LayerNormConfig())


def test_llama_factory_sets_split_norms():
    """Factory methods populate the split norm fields (not the legacy alias)."""
    config = TransformerConfig.llama2_271M(vocab_size=1024, dtype=DType.float32)
    assert isinstance(config.block, TransformerBlockConfig)
    assert config.block.attention_norm is not None
    assert config.block.feed_forward_norm is not None
    assert "layer_norm" not in config.block.as_config_dict()


# OLMO3_7B_CHECKPOINT = "https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage1/step0"
OLMO3_7B_CHECKPOINT = (
    "https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step0"
)


def test_load_olmo3_7b_config():
    """Verify that old checkpoint configs with a single block (not a dict) still load correctly."""
    config_path = cached_path(f"{OLMO3_7B_CHECKPOINT}/config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = TransformerConfig.from_dict(config_dict["model"])

    assert config.d_model == 4096
    assert config.n_layers == 32
    assert config.vocab_size == 100352
    assert isinstance(config.block, TransformerBlockConfig)
    assert config.block.name == "reordered_norm"

    # Round-trip through as_config_dict / from_dict should be lossless.
    roundtripped = TransformerConfig.from_dict(config.as_config_dict())
    assert roundtripped.as_config_dict() == config.as_config_dict()


def test_legacy_fused_attention_config():
    """Load legacy attention fields, including a per-layer override, into V2."""
    import pytest

    from olmo_core.nn.attention import (
        AttentionBackendName,
        AttentionType,
        FusedAttentionV2,
    )
    from olmo_core.nn.layer_norm import LayerNormType
    from olmo_core.nn.rope import RoPEType

    config = TransformerConfig.llama_like(
        d_model=32,
        vocab_size=64,
        n_layers=2,
        n_heads=2,
        fused_ops=True,
        use_flash=True,
        layer_norm_name=LayerNormType.rms,
    )
    assert isinstance(config.block, TransformerBlockConfig)
    attention = config.block.sequence_mixer
    assert attention.name == AttentionType.fused_v2
    assert attention.backend == AttentionBackendName.flash_2
    assert attention.use_flash is None
    assert attention.rope.name == RoPEType.default

    serialized = config.as_config_dict()
    block = serialized["block"]
    block["attention"] = block.pop("sequence_mixer")
    block["attention"].update(name="fused", backend="torch")
    block["attention"]["rope"]["name"] = "fused"
    serialized["block_overrides"] = {"1": block}
    restored = TransformerConfig.from_dict(serialized)
    with pytest.warns(UserWarning, match="not numerically identical"):
        model = restored.build()
    assert all(isinstance(block.attention, FusedAttentionV2) for block in model.blocks.values())


def test_factory_legacy_use_flash_emits_backend():
    """Factories accept the old keyword but produce configs with explicit backends."""
    import pytest

    from olmo_core.nn.attention import AttentionBackendName, AttentionConfig

    factories: list[tuple[Callable[..., TransformerConfig], dict[str, Any]]] = [
        (TransformerConfig.llama_like, {}),
        (TransformerConfig.ngpt_like, {}),
        (TransformerConfig.gemma3_like, {"n_kv_heads": 2, "hidden_size": 128}),
        (
            TransformerConfig.qwen3_5_like,
            {"n_kv_heads": 2, "head_dim": 32, "intermediate_size": 128},
        ),
    ]
    for factory, kwargs in factories:
        with pytest.warns(DeprecationWarning, match="use_flash"):
            config = factory(
                d_model=64, vocab_size=128, n_layers=4, n_heads=2, use_flash=True, **kwargs
            )
        blocks = config.block.values() if isinstance(config.block, dict) else [config.block]
        for block in blocks:
            attention = block.sequence_mixer
            if isinstance(attention, AttentionConfig):
                assert attention.backend == AttentionBackendName.flash_2
                assert attention.use_flash is None
        config = factory(
            d_model=64,
            vocab_size=128,
            n_layers=4,
            n_heads=2,
            use_flash=False,
            attn_backend=AttentionBackendName.flash_3,
            **kwargs,
        )
        blocks = config.block.values() if isinstance(config.block, dict) else [config.block]
        for block in blocks:
            attention = block.sequence_mixer
            if isinstance(attention, AttentionConfig):
                assert attention.backend == AttentionBackendName.flash_3
                assert attention.use_flash is None
