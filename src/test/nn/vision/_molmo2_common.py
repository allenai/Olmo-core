"""Shared helpers for the Molmo2 GPU parity / generation tests.

Cache-gating (``_hf_cache_has``) and the released-checkpoint variant list
(``MOLMO2_VARIANTS``) were previously duplicated across every ``molmo2_*_test.py``;
they live here so each test module imports one copy.
"""

import os

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
