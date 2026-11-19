"""Shared data selection and resource settings for the LeakySFT sweep."""

ARMS = {"a00": 0.0, "a01": 0.1, "a03": 0.3, "a05": 0.5, "a07": 0.7, "a10": 1.0}
NORMALIZED_ARMS = {arm: alpha for arm, alpha in ARMS.items() if alpha < 1.0}
SEQUENCE_LENGTH = 32768
GPUS = 2
# Prep 01M7QMF46AHRKA5344ZEHMFZDN, 2026-11-18: 4,986 windows, no tail.
PACKED_WINDOWS = 4986
DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/"
    "xlong5_2k256k_qwen35/shards_full/contradiction_train"
)
CLUSTER = "ai2/jupiter-cirrascale-2"
