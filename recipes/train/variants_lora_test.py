"""The --lora-rank modifier (recipes/train/variants.py LoRA): install on [B, K), freeze the base, export the early_lora
block the eval re-installs from."""
import pytest


def test_lora_modifier_installs_exports_and_validates():
    import torch  # noqa: F401
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    import variants as V

    v = V.VARIANTS["lora"]
    assert v.new_params
    o = V.LoRAOptions(lora_rank=4, lora_start=1, lora_scale=8.0)
    v.validate(o)
    m = TransformerConfig.olmo2_190M(vocab_size=1000, n_layers=4, fused_ops=False, dtype=DType.float32).build(
        init_device="cpu")
    v.post_build_model(m, o, None)
    assert {mm._lora_block for mm in m._early_lora["modules"]} == {1, 2, 3}
    assert not any(p.requires_grad for n, p in m.named_parameters() if n.startswith("blocks.") and "lora_" not in n)
    assert v.export_config(o) == {"early_lora": {"n_blocks": 4, "rank": 4, "scale": 8.0, "start": 1}}
    with pytest.raises(ValueError):
        v.validate(V.LoRAOptions(lora_rank=4, lora_blocks=2, lora_start=2))
