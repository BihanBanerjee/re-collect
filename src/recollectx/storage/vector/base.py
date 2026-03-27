"""Abstract interface for vector backends.

Vector backends provide semantic search capabilities for belief recall.
They return belief IDs only - never belief objects or truth decisions.

Design principles:
1. Vectors return IDs only - prevents authority inversion
2. Vectors never decide truth - only suggest candidates
3. Vectors never bypass write policies
4. Storage + belief graph remain authoritative
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector storage backends.

    Vector backends provide semantic similarity search for belief recall.
    They are advisory only - returning candidate IDs for the authoritative
    storage layer to fetch and filter.

    Example:
        # Vectors suggest candidates
        candidate_ids = vectors.search("user preferences", k=10)

        # Storage fetches authoritative beliefs
        beliefs = [storage.get(id) for id in candidate_ids]
    """

    def upsert(self, belief_id: str, text: str) -> None:
        """Insert or update a belief's vector representation."""
        ...

    def delete(self, belief_id: str) -> None:
        """Remove a belief's vector representation."""
        ...

    def search(self, query: str, k: int = 10) -> list[str]:
        """Search for semantically similar beliefs.

        Returns:
            List of belief IDs, ordered by semantic similarity.
        """
        ...
