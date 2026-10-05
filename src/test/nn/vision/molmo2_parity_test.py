"""
Numerical-parity tests for the HF Molmo2 → :class:`MultimodalLM`
state-dict converter.

Loads a real Molmo2 checkpoint from the local HuggingFace cache, runs the
converter, loads the converted weights into our model, and verifies:

1. Every parameter our model expects gets a tensor of the right shape.
2. The vision encoder produces numerically identical hidden states given the
   same pre-patchified input (after permuting the input from HF's spatial-first
   patch convention to our channel-first convention).

The full multimodal forward (LM + connector) is not asserted here because it
requires holding both models in memory at once (the 8B variant is ~32GB in
fp32 and our model would need another ~30GB). Loading-success + vision-parity
is sufficient evidence the converter is correct end-to-end, given the unit
tests in ``molmo2_loader_test.py`` already verified the QKV split, SwiGLU
chunk, and patch-embedding permute semantically.

These tests auto-skip when the corresponding HF checkpoint is not cached
locally — they're not meant for default CI.
"""


import pytest
import torch

from olmo_core.nn.vision import MultimodalLM
from olmo_core.nn.vision.molmo2_loader import (
    ensure_default_rope_registered,
    molmo2_config_from_hf_config,
    molmo2_hf_state_dict_to_multimodal_lm,
    reinit_rope_buffers,
    retie_word_embeddings,
)
from olmo_core.testing import requires_gpu

transformers = pytest.importorskip("transformers")

from ._molmo2_common import MOLMO2_VARIANTS, _hf_cache_has  # noqa: F401 (re-exported)


def _load_hf(model_id: str):
    """Load an HF Molmo2 model from the local cache, fp32. Skips on failure."""
    ensure_default_rope_registered()
    from transformers import AutoModelForImageTextToText

    try:
        hf = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load {model_id}: {e}")
    reinit_rope_buffers(hf)
    return hf


def _patchify_spatial_to_c_first(img_spatial: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert a pre-patchified tensor from HF's ``(kh, kw, c)`` flatten order
    to our ``(c, kh, kw)`` order. Both correspond to the same pixels — only
    the per-patch channel-vs-spatial ordering differs.

    Mirrors the inverse of :func:`molmo2_loader._convert_patch_embedding`:
    the converter permutes the *weight* from spatial-first to C-first; here
    we permute the *activations* the other way so the dot product gives the
    same result on either side.
    """
    B, N, _ = img_spatial.shape
    return (
        img_spatial.reshape(B, N, patch_size, patch_size, 3)
        .permute(0, 1, 4, 2, 3)
        .reshape(B, N, 3 * patch_size * patch_size)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("model_id", MOLMO2_VARIANTS)
def test_molmo2_converter_loads_and_vision_matches(model_id: str):
    """Load HF Molmo2 → run converter → load into our model → confirm
    parameter coverage + vision-encoder forward-pass parity.

    Vision encoder is the only piece moved to CUDA for the forward pass; the
    full LM + connector stay on CPU to avoid OOM when both HF and our model
    must coexist (Molmo2-8B is ~32GB in fp32).
    """
    if not _hf_cache_has(model_id):
        pytest.skip(f"{model_id} not in HF cache")

    device = torch.device("cuda")
    hf = _load_hf(model_id)

    # Build our config from the HF config and convert weights.
    cfg = molmo2_config_from_hf_config(hf.config)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), cfg)

    # Materialize our model on CPU and load the converted state dict.
    model = MultimodalLM(cfg, init_device="meta")
    model.to_empty(device=torch.device("cpu"))
    missing, unexpected = model.load_state_dict(converted, strict=False)
    retie_word_embeddings(model)  # `to_empty` breaks the tied-embedding share (Molmo2-4B)
    del converted

    # Strict on parameters: no LM / vision / connector weight may be unloaded.
    param_keys = {k for k, _ in model.named_parameters()}
    missing_params = set(missing) & param_keys
    assert not missing_params, (
        f"{model_id}: converter didn't load these params: " f"{sorted(missing_params)[:10]}"
    )
    unexpected_params = set(unexpected) & param_keys
    assert not unexpected_params, (
        f"{model_id}: converter produced unknown params: " f"{sorted(unexpected_params)[:10]}"
    )

    # Move just the vision encoders to CUDA for the forward pass.
    hf_vit = hf.model.vision_backbone.image_vit.eval().to(device)
    our_vit = model.vision.eval().to(device)

    # Build a random pre-patchified image and a matching C-first-permuted
    # copy for our side.
    torch.manual_seed(0)
    n_patches = cfg.vision.image_num_pos  # 729 for SO400M/14-378
    patch_size = cfg.vision.image_patch_size
    img_hf = torch.randn(
        1, n_patches, 3 * patch_size * patch_size, dtype=torch.float32, device=device
    )
    img_ours = _patchify_spatial_to_c_first(img_hf, patch_size)

    h, w = cfg.vision.image_default_input_size
    patch_num = (h // patch_size, w // patch_size)

    with torch.inference_mode():
        hf_hidden = hf_vit(img_hf)
        our_hidden = our_vit(img_ours, patch_num=patch_num)

    # Compare every ViT layer that the connector actually reads. ~5e-3 is
    # well within fp32 noise for 25 transformer layers under SDPA on GPU,
    # while still tight enough to catch real architecture mismatches.
    for layer in cfg.vit_layers:
        diff = (hf_hidden[layer] - our_hidden[layer]).abs().max().item()
        assert diff < 5e-3, (
            f"{model_id}: vision layer {layer} max abs diff = {diff:.2e} " f"(threshold 5e-3)"
        )
