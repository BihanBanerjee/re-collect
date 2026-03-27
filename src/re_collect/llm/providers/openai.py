"""OpenAI provider for LLM and embedding generation.

This module provides OpenAI implementations of the LLMProvider and
EmbeddingProvider protocols, supporting both GPT models and embeddings.

Installation:
    pip install openai

Example:
    from re_collect.llm.providers import OpenAIProvider

    # Create provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Generate text
    response = provider.generate("What is 2+2?")
    print(response.content)

    # Generate embeddings
    embedding = provider.embed("Hello world")
    print(f"Dimension: {len(embedding)}")

    # Structured output
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = provider.generate_structured("What is 2+2?", schema)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from re_collect.llm.base import LLMResponse

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIProvider:
    """OpenAI provider implementing LLMProvider and EmbeddingProvider protocols.

    This provider supports:
    - Text generation with GPT-3.5, GPT-4, and newer models
    - Structured output with JSON schema validation
    - Embedding generation with text-embedding models

    Example:
        provider = OpenAIProvider(api_key="sk-...")

        # Text generation
        response = provider.generate(
            prompt="Extract facts from: I love pizza",
            system_prompt="You are a fact extractor",
            temperature=0.3,
        )

        # Embeddings
        embedding = provider.embed("User likes pizza")

    Attributes:
        model: The LLM model to use (default: gpt-4o-mini)
        embedding_model: The embedding model to use (default: text-embedding-3-small)
    """

    # Default models
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    # Embedding dimensions by model
    EMBEDDING_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: LLM model to use (default: gpt-4o-mini)
            embedding_model: Embedding model to use (default: text-embedding-3-small)
            base_url: Optional custom API base URL (for Azure or proxies)
            organization: Optional OpenAI organization ID
            timeout: Request timeout in seconds

        Raises:
            ImportError: If openai package is not installed
        """
        self._api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self._base_url = base_url
        self._organization = organization
        self._timeout = timeout

        # Lazy client initialization
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenAIProvider. "
                    "Install it with: pip install openai"
                ) from e

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                organization=self._organization,
                timeout=self._timeout,
            )

        return self._client

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
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            LLMResponse containing generated text and metadata

        Example:
            response = provider.generate(
                prompt="Extract facts from: I love pizza",
                system_prompt="You are a fact extraction assistant",
                temperature=0.3
            )
            print(response.content)
        """
        client = self._get_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        content = response.choices[0].message.content or ""

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            metadata={"id": response.id, "finish_reason": response.choices[0].finish_reason},
        )

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate structured output matching a JSON schema.

        Uses OpenAI's JSON mode or response_format for structured output.

        Args:
            prompt: The user prompt
            schema: JSON schema defining the expected output structure
            system_prompt: Optional system prompt
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Dictionary matching the provided schema

        Example:
            schema = {
                "type": "object",
                "properties": {
                    "claims": {"type": "array", "items": {"type": "object"}}
                }
            }
            result = provider.generate_structured(
                prompt="Extract facts from: User likes pizza",
                schema=schema
            )
        """
        client = self._get_client()

        # Build messages
        messages: list[dict[str, str]] = []

        # Add schema instruction to system prompt
        schema_instruction = (
            f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        )
        full_system = (system_prompt or "") + schema_instruction
        messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": prompt})

        # Use JSON mode for compatible models
        response = client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=messages,
            response_format={"type": "json_object"},
            temperature=kwargs.pop("temperature", 0.3),
            max_tokens=kwargs.pop("max_tokens", 2000),
            **kwargs,
        )

        content = response.choices[0].message.content or "{}"

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            return self._extract_json(content)

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Try to extract JSON from text that may contain extra content."""
        # Try to find JSON in code blocks
        import re

        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object directly
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Example:
            embedding = provider.embed("User likes pizza")
            print(f"Dimension: {len(embedding)}")
        """
        client = self._get_client()

        response = client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )

        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single call.

        More efficient than calling embed() multiple times.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per input text)

        Example:
            texts = ["User likes pizza", "User prefers pasta"]
            embeddings = provider.embed_batch(texts)
        """
        if not texts:
            return []

        client = self._get_client()

        response = client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    @property
    def dimension(self) -> int:
        """Return the dimensionality of embedding vectors.

        Returns:
            Integer dimension based on the embedding model
        """
        return self.EMBEDDING_DIMENSIONS.get(self.embedding_model, 1536)

    def close(self) -> None:
        """Close the OpenAI client."""
        if self._client is not None:
            self._client.close()
            self._client = None
