"""Ollama provider for local LLM and embedding generation.

This module provides an Ollama implementation of the LLMProvider and
EmbeddingProvider protocols, enabling completely local AI without API costs.

Ollama supports many open-source models including:
- llama3.2, llama3.1, llama2
- mistral, mixtral
- codellama
- nomic-embed-text (embeddings)

Installation:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3.2
    3. pip install httpx

Example:
    from recollect.llm.providers import OllamaProvider

    # Create provider (uses local Ollama server)
    provider = OllamaProvider(model="llama3.2")

    # Generate text
    response = provider.generate("What is 2+2?")
    print(response.content)

    # Generate embeddings (requires embedding model)
    provider = OllamaProvider(
        model="llama3.2",
        embedding_model="nomic-embed-text"
    )
    embedding = provider.embed("Hello world")
"""

from __future__ import annotations

import json
from typing import Any

from re_collect.llm.base import LLMResponse


class OllamaProvider:
    """Ollama provider implementing LLMProvider and EmbeddingProvider protocols.

    This provider enables completely local AI without API costs by using
    Ollama to run open-source models.

    Features:
    - Local execution (no API costs, no data leaves your machine)
    - Support for many open-source models
    - Both text generation and embeddings

    Example:
        provider = OllamaProvider(model="llama3.2")

        # Text generation
        response = provider.generate(
            prompt="Extract facts from: I love pizza",
            system_prompt="You are a fact extractor",
            temperature=0.3,
        )

        # Embeddings (with embedding model)
        provider = OllamaProvider(embedding_model="nomic-embed-text")
        embedding = provider.embed("User likes pizza")

    Attributes:
        model: The LLM model to use (default: llama3.2)
        embedding_model: The embedding model to use (default: nomic-embed-text)
        base_url: Ollama server URL (default: http://localhost:11434)
    """

    # Default models
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

    # Common embedding model dimensions
    EMBEDDING_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            model: LLM model to use (default: llama3.2)
            embedding_model: Embedding model to use (default: nomic-embed-text)
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (longer for local models)

        Raises:
            ImportError: If httpx package is not installed
        """
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

        self._client: Any | None = None
        self._embedding_dimension: int | None = None

    def _get_client(self) -> Any:
        """Get or create the httpx client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "httpx package is required for OllamaProvider. "
                    "Install it with: pip install httpx"
                ) from e

            self._client = httpx.Client(timeout=self._timeout)

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
            **kwargs: Additional Ollama-specific parameters

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

        # Build request
        request_data: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            request_data["system"] = system_prompt

        # Add any extra options
        if kwargs:
            request_data["options"].update(kwargs)

        url = f"{self.base_url}/api/generate"

        response = client.post(url, json=request_data)
        response.raise_for_status()
        data = response.json()

        content = data.get("response", "")

        # Extract usage info if available
        usage = None
        if "prompt_eval_count" in data or "eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }

        return LLMResponse(
            content=content,
            usage=usage,
            model=data.get("model", self.model),
            metadata={
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "eval_duration": data.get("eval_duration"),
            },
        )

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate structured output matching a JSON schema.

        Args:
            prompt: The user prompt
            schema: JSON schema defining the expected output structure
            system_prompt: Optional system prompt
            **kwargs: Additional Ollama-specific parameters

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
        # Add schema instruction to system prompt
        schema_instruction = (
            f"\n\nYou must respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Respond with ONLY the JSON object, no other text or explanation."
        )
        full_system = (system_prompt or "") + schema_instruction

        # Use lower temperature for structured output
        response = self.generate(
            prompt=prompt,
            system_prompt=full_system,
            temperature=kwargs.pop("temperature", 0.1),
            max_tokens=kwargs.pop("max_tokens", 2000),
            **kwargs,
        )

        content = response.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            return self._extract_json(content)

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Try to extract JSON from text that may contain extra content."""
        import re

        # Try to find JSON in code blocks
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

        Requires an embedding model like nomic-embed-text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Example:
            embedding = provider.embed("User likes pizza")
            print(f"Dimension: {len(embedding)}")
        """
        client = self._get_client()

        request_data = {
            "model": self.embedding_model,
            "prompt": text,
        }

        url = f"{self.base_url}/api/embeddings"

        response = client.post(url, json=request_data)
        response.raise_for_status()
        data = response.json()

        embedding = data.get("embedding", [])

        # Cache dimension if not set
        if self._embedding_dimension is None and embedding:
            self._embedding_dimension = len(embedding)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Note: Ollama doesn't have native batch embedding, so this calls
        embed() for each text sequentially.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []

        # Ollama doesn't support batch embeddings natively
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)

        return embeddings

    @property
    def dimension(self) -> int:
        """Return the dimensionality of embedding vectors.

        Returns:
            Integer dimension based on the embedding model
        """
        # Return cached dimension if available
        if self._embedding_dimension is not None:
            return self._embedding_dimension

        # Return known dimension or default
        return self.EMBEDDING_DIMENSIONS.get(self.embedding_model, 768)

    def list_models(self) -> list[str]:
        """List available models on the Ollama server.

        Returns:
            List of model names

        Example:
            models = provider.list_models()
            print(f"Available: {models}")
        """
        client = self._get_client()

        url = f"{self.base_url}/api/tags"

        response = client.get(url)
        response.raise_for_status()
        data = response.json()

        return [model["name"] for model in data.get("models", [])]

    def close(self) -> None:
        """Close the httpx client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OllamaProvider:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
