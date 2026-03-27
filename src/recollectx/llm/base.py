"""Base protocols for LLM and embedding providers.

This module defines abstract interfaces for LLM and embedding providers,
allowing recollect to work with any LLM service (OpenAI, Anthropic, Google, Ollama, etc.)
without hard dependencies.

All provider implementations are optional - the core recollect library works
without any LLM integration.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMResponse:
    """Standardized response from an LLM provider.

    Attributes:
        content: The generated text content
        usage: Optional token usage statistics (prompt_tokens, completion_tokens, total_tokens)
        model: Optional model identifier used for generation
        metadata: Optional additional metadata from the provider
    """

    content: str
    usage: dict[str, int] | None = None
    model: str | None = None
    metadata: dict[str, Any] | None = None


class LLMProvider(Protocol):
    """Abstract interface for any LLM provider.

    This protocol defines the contract for LLM providers. Implementations can use
    any backend (OpenAI, Anthropic, Google, local models via Ollama, etc.).

    Example:
        class MyLLMProvider:
            def generate(self, prompt: str, **kwargs) -> LLMResponse:
                # Implementation using your preferred LLM service
                ...

            def generate_structured(self, prompt: str, schema: dict, **kwargs) -> dict:
                # Implementation for structured output
                ...
    """

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text completion from a prompt.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt to set behavior/context
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing generated text and metadata

        Example:
            response = llm.generate(
                prompt="Extract facts from: I love pizza",
                system_prompt="You are a fact extraction assistant",
                temperature=0.3
            )
            print(response.content)
        """
        ...

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate structured output matching a JSON schema.

        This is used for extracting claims with specific fields, ensuring
        the LLM response follows a predictable structure.

        Args:
            prompt: The user prompt
            schema: JSON schema defining the expected output structure
            system_prompt: Optional system prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary matching the provided schema

        Example:
            schema = {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        }
                    }
                }
            }
            result = llm.generate_structured(
                prompt="Extract facts from: User likes pizza",
                schema=schema
            )
        """
        ...


class EmbeddingProvider(Protocol):
    """Abstract interface for embedding generation.

    Embeddings are used for semantic similarity search and deduplication.
    Providers can use any embedding model (OpenAI, Google, local sentence-transformers, etc.).

    Example:
        class MyEmbeddingProvider:
            def embed(self, text: str) -> List[float]:
                # Generate embedding vector
                ...

            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                # Batch embedding generation
                ...

            @property
            def dimension(self) -> int:
                return 768  # Embedding dimension
    """

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Example:
            embedding = embedder.embed("User likes pizza")
            print(f"Dimension: {len(embedding)}")
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single call.

        This is more efficient than calling embed() multiple times.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per input text)

        Example:
            texts = ["User likes pizza", "User prefers pasta"]
            embeddings = embedder.embed_batch(texts)
            print(f"Generated {len(embeddings)} embeddings")
        """
        ...

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.

        This is used by vector stores to configure their storage.
        Common dimensions:
        - 768: sentence-transformers (all-MiniLM-L6-v2)
        - 1536: OpenAI text-embedding-3-small
        - 3072: OpenAI text-embedding-3-large

        Returns:
            Integer dimension of embedding vectors
        """
        ...
