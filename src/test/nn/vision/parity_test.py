"""
Numerical-parity tests between our vision encoders and the HuggingFace
reference implementations.

These tests load real checkpoints (``openai/clip-vit-large-patch14-336``,
``google/siglip-so400m-patch14-384``, ``google/siglip2-so400m-patch14-384``)
and assert that running the same pixel input through both models produces
matching pre-``post_layernorm`` hidden states.  Requires the ``transformers``
package and the checkpoints to be available locally (or downloadable) —
otherwise the tests skip.
"""

from typing import Dict

import pytest
import torch

from olmo_core.nn.vision import (
    VisionEncoderConfig,
    VisionEncoderType,
    VisionTransformer,
    siglip_state_dict_to_vision_encoder,
)

transformers = pytest.importorskip("transformers")

# Float32 accumulation error across 24–27 transformer layers.  Two independent
# implementations of identical ops can differ by O(1e-4)–O(1e-3) due to kernel
# fusion differences and activation approximations — not from any precision loss.
# Measured worst-cases: CLIP 5e-4, SigLIP 2.8e-3, SigLIP2 3e-4.
# These tolerances verify fp32-level equivalence; they would NOT pass fp16
# inference (errors ~1e-2), which is intentional.
_RTOL = 1e-3
_ATOL = 3e-3  # SigLIP 27-layer worst case; CLIP and SigLIP2 are ~10x lower


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pixel ``(B, 3, H, W)`` → patches ``(B, N, 3 * p * p)``.

    Uses the C-major flatten order that matches a HuggingFace ``Conv2d`` patch
    embedding with ``kernel_size = stride = patch_size``: each output channel's
    ``Conv2d`` weight, reshaped to ``(D, -1)``, dotted with the flat patch
    pixels equals the convolution at that spatial location.
    """
    B, C, H, W = image.shape
    p = patch_size
    assert H % p == 0 and W % p == 0
    x = image.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, H/p, W/p, C, p, p)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


def _convert_clip_state_dict(hf_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map a HuggingFace ``CLIPVisionTransformer`` state dict to ours."""
    out: Dict[str, torch.Tensor] = {}
    out["class_embedding"] = hf_sd["embeddings.class_embedding"]
    # Conv2d (D, 3, p, p) → Linear (D, 3*p*p), C-major flatten.
    out["patch_embedding.weight"] = hf_sd["embeddings.patch_embedding.weight"].reshape(
        hf_sd["embeddings.patch_embedding.weight"].shape[0], -1
    )
    out["positional_embedding"] = hf_sd["embeddings.position_embedding.weight"]
    out["pre_ln.weight"] = hf_sd["pre_layrnorm.weight"]
    out["pre_ln.bias"] = hf_sd["pre_layrnorm.bias"]

    # Per-layer projections.
    n_layers = (
        max(int(k.split(".")[2]) for k in hf_sd.keys() if k.startswith("encoder.layers.")) + 1
    )
    for i in range(n_layers):
        src = f"encoder.layers.{i}"
        dst = f"blocks.{i}"
        for hf_name, our_name in (
            ("layer_norm1", "attn_norm"),
            ("layer_norm2", "ffn_norm"),
        ):
            out[f"{dst}.{our_name}.weight"] = hf_sd[f"{src}.{hf_name}.weight"]
            out[f"{dst}.{our_name}.bias"] = hf_sd[f"{src}.{hf_name}.bias"]
        for hf_name, our_name in (
            ("q_proj", "wq"),
            ("k_proj", "wk"),
            ("v_proj", "wv"),
            ("out_proj", "wo"),
        ):
            out[f"{dst}.attn.{our_name}.weight"] = hf_sd[f"{src}.self_attn.{hf_name}.weight"]
            out[f"{dst}.attn.{our_name}.bias"] = hf_sd[f"{src}.self_attn.{hf_name}.bias"]
        for hf_name, our_name in (("fc1", "w1"), ("fc2", "w2")):
            out[f"{dst}.ffn.{our_name}.weight"] = hf_sd[f"{src}.mlp.{hf_name}.weight"]
            out[f"{dst}.ffn.{our_name}.bias"] = hf_sd[f"{src}.mlp.{hf_name}.bias"]
    return out


def _try_load_hf(model_cls, model_id: str):
    """Load an HF model, preferring the local cache.

    Tries ``local_files_only=True`` first so a cached checkpoint never triggers a
    network download (which would turn this unit test into a slow integration
    download). Only if the checkpoint isn't cached do we fall back to a normal
    ``from_pretrained`` that may download. Skips the test if neither succeeds.
    """
    try:
        return model_cls.from_pretrained(model_id, local_files_only=True).eval()
    except Exception:  # noqa: BLE001  # not cached locally; fall back to download
        pass
    try:
        return model_cls.from_pretrained(model_id).eval()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not load {model_id}: {e}")


# ---------------------------------------------------------------------------
# CLIP parity
# ---------------------------------------------------------------------------


def test_clip_parity():
    """Our :class:`VisionTransformer` matches HF's ``CLIPVisionModel`` numerically.

    Uses ``openai/clip-vit-large-patch14-336`` (default config). HF has 24
    blocks; our default config sets ``image_num_layers=23`` (the final block is
    unused when reading from layer ``-2``), so we override to 24 for the parity test.

    ``CLIPVisionModel`` is the vision-only model in ``transformers``; it exposes
    ``forward(pixel_values=…)`` directly and does not have a nested
    ``vision_model`` attribute.
    """
    hf = _try_load_hf(transformers.CLIPVisionModel, "openai/clip-vit-large-patch14-336")

    cfg = VisionEncoderConfig(image_num_layers=24)  # match HF's 24-layer checkpoint
    assert cfg.name == VisionEncoderType.openai
    ours = VisionTransformer(cfg, init_device="cpu").eval()
    ours.load_state_dict(_convert_clip_state_dict(hf.state_dict()))

    torch.manual_seed(0)
    pixel_values = torch.randn(1, 3, 336, 336)

    with torch.inference_mode():
        # hidden_states[0] is the patch-embedding output; hidden_states[-1] is
        # the output of the final encoder block, i.e. pre-post_layernorm.
        hf_out = hf(pixel_values=pixel_values, output_hidden_states=True)
        hf_last = hf_out.hidden_states[-1]

        patches = _patchify(pixel_values, cfg.image_patch_size)
        ours_last = ours(patches, cfg.image_num_patch)[-1]

    torch.testing.assert_close(ours_last, hf_last, rtol=_RTOL, atol=_ATOL)


# ---------------------------------------------------------------------------
# SigLIP parity
# ---------------------------------------------------------------------------


def test_siglip_parity():
    """Our :class:`VisionTransformer` matches HF's ``SiglipVisionModel``.

    Uses ``google/siglip-so400m-patch14-384``, which truncates to a 27×27
    patch grid (floor(384/14) = 27). Our :meth:`VisionEncoderConfig.siglip_so400m`
    config uses input size 378 (= 27×14) to match the same 27×27 grid exactly.

    ``SiglipVisionModel`` exposes ``forward(pixel_values=…)`` directly; it does
    not have a nested ``vision_model`` attribute.
    """
    hf = _try_load_hf(transformers.SiglipVisionModel, "google/siglip-so400m-patch14-384")

    cfg = VisionEncoderConfig.siglip_so400m()
    ours = VisionTransformer(cfg, init_device="cpu").eval()
    ours.load_state_dict(siglip_state_dict_to_vision_encoder(hf.state_dict()))

    torch.manual_seed(0)
    # 378 = 27×14: both HF and our model produce a 27×27 patch grid.
    pixel_values = torch.randn(1, 3, 378, 378)

    with torch.inference_mode():
        hf_out = hf(pixel_values=pixel_values, output_hidden_states=True)
        hf_last = hf_out.hidden_states[-1]  # output of final encoder block

        patches = _patchify(pixel_values, cfg.image_patch_size)
        ours_last = ours(patches, cfg.image_num_patch)[-1]

    torch.testing.assert_close(ours_last, hf_last, rtol=_RTOL, atol=_ATOL)


# ---------------------------------------------------------------------------
# SigLIP2 parity
# ---------------------------------------------------------------------------


def test_siglip2_parity():
    """Our :class:`VisionTransformer` matches HF's SigLIP2 checkpoint.

    Uses ``google/siglip2-so400m-patch14-384``.  Despite the "384" in the
    model ID, both 378×378 and 384×384 inputs produce a 27×27 patch grid
    (floor(378/14) = floor(384/14) = 27).  We use 378 to match our config's
    ``image_default_input_size``.

    SigLIP2 shares the same ``SiglipVisionModel`` class in ``transformers`` as
    SigLIP — only the training recipe differs — so the same state-dict
    converter applies.
    """
    hf = _try_load_hf(transformers.SiglipVisionModel, "google/siglip2-so400m-patch14-384")

    cfg = VisionEncoderConfig.siglip2_so400m_patch14_378()
    ours = VisionTransformer(cfg, init_device="cpu").eval()
    ours.load_state_dict(siglip_state_dict_to_vision_encoder(hf.state_dict()))

    torch.manual_seed(0)
    # 378 = 27×14: both HF and our model produce a 27×27 = 729 patch grid.
    pixel_values = torch.randn(1, 3, 378, 378)

    with torch.inference_mode():
        hf_out = hf(pixel_values=pixel_values, output_hidden_states=True)
        hf_last = hf_out.hidden_states[-1]  # output of final encoder block

        patches = _patchify(pixel_values, cfg.image_patch_size)
        ours_last = ours(patches, cfg.image_num_patch)[-1]

    torch.testing.assert_close(ours_last, hf_last, rtol=_RTOL, atol=_ATOL)
