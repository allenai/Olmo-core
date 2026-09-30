from dataclasses import dataclass
from typing import Dict, Optional

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError


@dataclass
class EmoRouterConfig(Config):
    """Document-level expert-pool policy shared by EMO router implementations."""

    eos_token_id: int
    min_document_expert_pool: int
    max_document_expert_pool: int
    eval_document_expert_pool: Optional[int] = None
    full_pool: bool = False
    """Use ordinary token routing in both train and eval while retaining EMO ancestry."""
    source_config: Optional[Dict[str, Optional[int]]] = None
    """Original HF EMO settings before an explicit full-pool RL override."""

    def validate_for_router(self, *, num_experts: int, top_k: int) -> None:
        if self.full_pool and self.eval_document_expert_pool != num_experts:
            raise OLMoConfigurationError(
                "Full-pool EMO requires eval_document_expert_pool=num_experts"
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
