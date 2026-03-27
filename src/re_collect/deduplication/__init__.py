"""Claim deduplication module.

Simple two-step pipeline:
1. Find similar claims using embedding cosine similarity
2. Ask the LLM what to do (ADD, UPDATE, DELETE, NONE)
3. Execute the action against storage

Example:
    from re_collect.deduplication import ClaimDeduplicator

    deduplicator = ClaimDeduplicator(
        storage=storage,
        embedding_provider=embedder,
        llm_provider=llm,
    )

    # Finds duplicates, asks LLM, stores/updates/deletes — all in one call
    result = deduplicator.process(new_claim)
    print(result.action)  # ADD, UPDATE, DELETE, or NONE
"""

from re_collect.deduplication.deduplicator import (
    ClaimDeduplicator,
    DeduplicationResult,
)
from re_collect.deduplication.merger import (
    LLMMerger,
    MergeDecision,
)
from re_collect.deduplication.similarity import (
    EmbeddingSimilarity,
    SimilarityCalculator,
    claim_to_text,
)

__all__ = [
    "ClaimDeduplicator",
    "DeduplicationResult",
    "LLMMerger",
    "MergeDecision",
    "EmbeddingSimilarity",
    "SimilarityCalculator",
    "claim_to_text",
]
