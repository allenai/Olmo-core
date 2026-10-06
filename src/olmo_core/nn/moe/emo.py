from dataclasses import dataclass
from typing import Literal, Optional

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError


@dataclass
class EmoRouterConfig(Config):
    """Document-level expert-pool policy shared by EMO router implementations."""

    eos_token_id: int
    min_document_expert_pool: int
    max_document_expert_pool: int
    eval_document_expert_pool: Optional[int] = None
    segment_ids_from: Literal["eos", "doc_lens"] = "eos"
    """
    Where training takes each token's document from. ``"eos"`` (the default) scans ``input_ids``
    for ``eos_token_id``. ``"doc_lens"`` uses the batch's ``doc_lens``, the same boundaries as
    intra-document attention masking; set it when those come from metadata rather than from EOS
    (``use_array_if_local=False``), or EMO pools would still split and merge documents at EOS.
    """

    def validate_for_router(self, *, num_experts: int, top_k: int) -> None:
        if self.segment_ids_from not in ("eos", "doc_lens"):
            raise OLMoConfigurationError(
                f"EMO segment_ids_from must be 'eos' or 'doc_lens', got {self.segment_ids_from!r}"
            )
        if not 0 < self.min_document_expert_pool <= self.max_document_expert_pool:
            raise OLMoConfigurationError(
                "EMO document expert pools must satisfy 0 < min_pool <= max_pool"
            )
        if self.max_document_expert_pool > num_experts:
            raise OLMoConfigurationError(
                "EMO max_document_expert_pool cannot exceed the number of routed experts"
            )
        if self.min_document_expert_pool < top_k:
            raise OLMoConfigurationError(
                "EMO min_document_expert_pool must be greater than or equal to top_k"
            )
        if self.eval_document_expert_pool is not None and not (
            top_k <= self.eval_document_expert_pool <= num_experts
        ):
            raise OLMoConfigurationError(
                "EMO eval_document_expert_pool must be between top_k and num_experts"
            )

    def eval_pool_size(self) -> int:
        if self.eval_document_expert_pool is not None:
            return self.eval_document_expert_pool
        return (self.min_document_expert_pool + self.max_document_expert_pool) // 2
