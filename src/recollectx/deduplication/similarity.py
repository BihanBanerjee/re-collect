"""Similarity calculator for claim deduplication.

Uses embedding-based cosine similarity to find candidate duplicates.
Requires an EmbeddingProvider implementation.
"""

import math
from typing import Protocol

from recollectx.claims import Claim, EpisodicClaim, SemanticClaim
from recollectx.llm.base import EmbeddingProvider


class SimilarityCalculator(Protocol):
    """Protocol for calculating similarity between claims."""

    def calculate(self, claim1: Claim, claim2: Claim) -> float:
        """Calculate similarity score between two claims.

        Returns:
            Similarity score in [0.0, 1.0]
        """
        ...


class EmbeddingSimilarity:
    """Calculate similarity using embedding cosine similarity.

    Converts claims to text, generates embeddings, and computes
    cosine similarity to find candidate duplicates.
    """

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedder = embedding_provider

    def calculate(self, claim1: Claim, claim2: Claim) -> float:
        """Calculate cosine similarity between claim embeddings."""
        if type(claim1) is not type(claim2):
            return 0.0

        text1 = claim_to_text(claim1)
        text2 = claim_to_text(claim2)

        embeddings = self.embedder.embed_batch([text1, text2])
        return _cosine_similarity(embeddings[0], embeddings[1])


def claim_to_text(claim: Claim) -> str:
    """Convert a claim to text for embedding."""
    if isinstance(claim, SemanticClaim):
        return f"{claim.subject} {claim.predicate} {claim.object}"
    elif isinstance(claim, EpisodicClaim):
        return claim.summary
    return str(claim)


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
