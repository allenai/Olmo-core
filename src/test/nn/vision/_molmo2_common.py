"""Shared helpers for the Molmo2 GPU parity / generation tests.

Cache-gating (``_hf_cache_has``) and the released-checkpoint variant list
(``MOLMO2_VARIANTS``) were previously duplicated across every ``molmo2_*_test.py``;
they live here so each test module imports one copy.
"""

import os

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

MOLMO2_VARIANTS = [
    "allenai/Molmo2-4B",
    "allenai/Molmo2-8B",
    "allenai/Molmo2-O-7B",
]


def _hf_cache_has(model_id: str) -> bool:
    """True if ``model_id`` is present in a local HF cache (``~/.cache/huggingface/hub``
    or ``$HF_HOME/hub``) — used to skip GPU parity tests when the checkpoint isn't cached."""
    suffix = "models--" + model_id.replace("/", "--")
    candidates = [os.path.expanduser("~/.cache/huggingface/hub")]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(os.path.join(hf_home, "hub"))
    return any(os.path.isdir(os.path.join(root, suffix)) for root in candidates if root)


def _load_hf(model_id: str):
    """Load an HF Molmo2 model (fp32, CPU) from the local cache; skips the test on failure."""
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


def _build_ours(model_id: str, device, dtype):
    """Load HF Molmo2 and build our model from its converted weights. Returns
    (hf_model on CPU, ours on `device` with `dtype`, our config)."""
    hf = _load_hf(model_id)
    cfg = molmo2_config_from_hf_config(hf.config)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), cfg)
    ours = MultimodalLM(cfg, init_device="meta")
    ours.to_empty(device=torch.device("cpu"))
    ours.load_state_dict(converted, strict=False)
    retie_word_embeddings(ours)  # `to_empty` breaks the tied-embedding share (Molmo2-4B)
    del converted
    ours = ours.to(device=device, dtype=dtype).eval()
    return hf, ours, cfg
