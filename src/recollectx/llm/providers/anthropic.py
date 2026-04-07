"""Anthropic Claude provider for LLM generation.

This module provides an Anthropic implementation of the LLMProvider protocol,
supporting Claude models for text and structured output generation.

Note: Anthropic does not provide embedding models. For embeddings, use
OpenAIProvider, LocalEmbedder, or another embedding provider.

Installation:
    pip install anthropic

Example:
    from recollectx.llm.providers import AnthropicProvider

    # Create provider
    provider = AnthropicProvider(api_key="your-api-key")

    # Generate text
    response = provider.generate("What is 2+2?")
    print(response.content)

    # Structured output
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = provider.generate_structured("What is 2+2?", schema)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, cast

from recollectx.llm.base import LLMResponse

if TYPE_CHECKING:
    from anthropic import Anthropic


class AnthropicProvider:
    """Anthropic Claude provider implementing LLMProvider protocol.

    This provider supports:
    - Text generation with Claude models (claude-3-opus, claude-3-sonnet, etc.)
    - Structured output with JSON schema validation

    Note:
        Anthropic does not provide embedding models. Use OpenAIProvider,
        LocalEmbedder, or another provider for embeddings.

    Example:
        provider = AnthropicProvider(api_key="sk-ant-...")

        # Text generation
        response = provider.generate(
            prompt="Extract facts from: I love pizza",
            system_prompt="You are a fact extractor",
            temperature=0.3,
        )

    Attributes:
        model: The Claude model to use (default: claude-3-5-sonnet-latest)
    """

    # Default model
    DEFAULT_MODEL = "claude-3-5-sonnet-latest"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Claude model to use (default: claude-3-5-sonnet-latest)
            base_url: Optional custom API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests

        Raises:
            ImportError: If anthropic package is not installed
        """
        self._api_key = api_key
        self.model = model
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries

        # Lazy client initialization
        self._client: Anthropic | None = None

    def _get_client(self) -> Anthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install it with: pip install anthropic"
                ) from e

            kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": self._timeout,
                "max_retries": self._max_retries,
            }
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = Anthropic(**kwargs)

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
            **kwargs: Additional Anthropic-specific parameters

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
        request_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        # Add any extra kwargs
        request_kwargs.update(kwargs)

        response = client.messages.create(**request_kwargs)

        # Extract content
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            metadata={
                "id": response.id,
                "stop_reason": response.stop_reason,
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
            **kwargs: Additional Anthropic-specific parameters

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
            f"Respond with ONLY the JSON object, no other text."
        )
        full_system = (system_prompt or "") + schema_instruction

        response = self.generate(
            prompt=prompt,
            system_prompt=full_system,
            temperature=kwargs.pop("temperature", 0.3),
            max_tokens=kwargs.pop("max_tokens", 2000),
            **kwargs,
        )

        content = response.content.strip()

        try:
            return cast(dict[str, Any], json.loads(content))
        except json.JSONDecodeError:
            # Try to extract JSON from response
            return self._extract_json(content)

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Try to extract JSON from text that may contain extra content."""
        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                return cast(dict[str, Any], json.loads(json_match.group(1)))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object directly
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return cast(dict[str, Any], json.loads(brace_match.group(0)))
            except json.JSONDecodeError:
                pass

        return {}

    def close(self) -> None:
        """Close the Anthropic client."""
        if self._client is not None:
            self._client.close()
            self._client = None
