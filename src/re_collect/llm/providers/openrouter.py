"""OpenRouter provider for LLM generation via OpenAI-compatible API.

OpenRouter (https://openrouter.ai) is an API aggregator that provides access
to 200+ models (OpenAI, Anthropic, Meta, Google, Mistral, etc.) through a
single OpenAI-compatible endpoint.

Installation:
    pip install openai  # Uses the openai SDK under the hood

Example:
    from re_collect.llm.providers import OpenRouterProvider

    provider = OpenRouterProvider(api_key="sk-or-...")

    # Uses openrouter default model routing
    response = provider.generate("What is 2+2?")

    # Or pick a specific model
    provider = OpenRouterProvider(
        api_key="sk-or-...",
        model="anthropic/claude-sonnet-4-20250514",
    )

    # Embeddings (via OpenRouter's embedding models)
    provider = OpenRouterProvider(
        api_key="sk-or-...",
        embedding_model="openai/text-embedding-3-small",
    )
    embedding = provider.embed("Hello world")
"""

from __future__ import annotations

from re_collect.llm.providers.openai import OpenAIProvider

# OpenRouter API endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider — access 200+ models via one API key.

    Subclasses OpenAIProvider since OpenRouter is fully OpenAI-compatible.
    All generate(), generate_structured(), embed(), and embed_batch()
    methods work identically.

    Example:
        provider = OpenRouterProvider(api_key="sk-or-...")

        # Use any model available on OpenRouter
        provider = OpenRouterProvider(
            api_key="sk-or-...",
            model="meta-llama/llama-3.1-70b-instruct",
        )

        response = provider.generate("Extract facts from: I love pizza")
    """

    DEFAULT_MODEL = "openai/gpt-4o-mini"
    DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        timeout: float = 60.0,
        app_name: str | None = None,
    ) -> None:
        """Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key (starts with sk-or-)
            model: Model to use (default: openai/gpt-4o-mini).
                See https://openrouter.ai/models for available models.
            embedding_model: Embedding model (default: openai/text-embedding-3-small)
            timeout: Request timeout in seconds
            app_name: Optional app name sent in HTTP-Referer header
                for OpenRouter's rankings and rate limits.
        """
        super().__init__(
            api_key=api_key,
            model=model,
            embedding_model=embedding_model,
            base_url=OPENROUTER_BASE_URL,
            timeout=timeout,
        )
        self._app_name = app_name

    def _get_client(self):
        """Get or create the OpenAI client with OpenRouter headers."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenRouterProvider. "
                    "Install it with: pip install openai"
                ) from e

            default_headers = {}
            if self._app_name:
                default_headers["HTTP-Referer"] = self._app_name

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                default_headers=default_headers or None,
            )

        return self._client
