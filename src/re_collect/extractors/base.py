"""Base protocol for claim extraction from text.

This module defines the abstract interface for extracting claims from natural language text.
Implementations can use different strategies:
- LLM-based extraction (sophisticated, requires API)
- Hybrid approaches
"""

from typing import Any, Protocol

from re_collect.claims import Claim


class ClaimExtractor(Protocol):
    """Abstract interface for extracting claims from text.

    Extractors analyze natural language text and identify factual claims,
    converting them into structured Claim objects (episodic or semantic).

    Example:
        Input text: "I love pizza on Fridays"
        Output claims:
        - SemanticClaim(subject="user", predicate="likes", object="pizza")
        - EpisodicClaim(summary="user mentioned Friday preference", ...)
    """

    def extract(
        self, text: str, context: dict[str, Any] | None = None
    ) -> list[Claim]:
        """Extract claims from a single text.

        Args:
            text: Natural language text to analyze
            context: Optional context dict with additional information:
                - user_id: Identifier for the user
                - timestamp: When the text was created
                - conversation_id: Related conversation
                - Any other domain-specific context

        Returns:
            List of extracted claims (EpisodicClaim, SemanticClaim)

        Example:
            claims = extractor.extract(
                text="I love pizza on Fridays",
                context={"user_id": "user_123", "timestamp": "2024-01-24"}
            )
            for claim in claims:
                print(f"{claim.type}: {claim}")
        """
        ...

    def extract_batch(
        self, texts: list[str], context: dict[str, Any] | None = None
    ) -> list[list[Claim]]:
        """Extract claims from multiple texts in batch.

        This can be more efficient than calling extract() multiple times,
        especially for LLM-based extractors that can batch API calls.

        Args:
            texts: List of texts to analyze
            context: Optional shared context for all texts

        Returns:
            List of claim lists (one list per input text)

        Example:
            texts = ["I love pizza", "I prefer pasta"]
            results = extractor.extract_batch(texts)
            for i, claims in enumerate(results):
                print(f"Text {i}: {len(claims)} claims")
        """
        ...
