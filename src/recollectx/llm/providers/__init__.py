"""LLM and embedding provider implementations.

This module provides ready-to-use implementations of the LLMProvider and
EmbeddingProvider protocols for popular services.

Available Providers:
- OpenAIProvider: OpenAI GPT models + embeddings (requires API key)
- AnthropicProvider: Anthropic Claude models (requires API key)
- OpenRouterProvider: 200+ models via OpenRouter (requires API key)
- OllamaProvider: Local models via Ollama (free, local)
- LocalEmbedder: Local embeddings via sentence-transformers (free, local)

Installation:
    # For OpenAI
    pip install openai

    # For Anthropic
    pip install anthropic

    # For Ollama
    pip install httpx
    # Also install Ollama: https://ollama.ai

    # For local embeddings
    pip install sentence-transformers

Example (OpenAI - cloud):
    from recollectx.llm.providers import OpenAIProvider

    provider = OpenAIProvider(api_key="sk-...")

    # Text generation
    response = provider.generate("What is 2+2?")
    print(response.content)

    # Embeddings
    embedding = provider.embed("Hello world")

Example (Anthropic - cloud):
    from recollectx.llm.providers import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-...")
    response = provider.generate("What is 2+2?")

Example (Ollama - local, free):
    from recollectx.llm.providers import OllamaProvider

    # Requires: ollama pull llama3.2
    provider = OllamaProvider(model="llama3.2")
    response = provider.generate("What is 2+2?")

Example (Local embeddings - free):
    from recollectx.llm.providers import LocalEmbedder

    embedder = LocalEmbedder()  # Uses all-MiniLM-L6-v2
    embedding = embedder.embed("Hello world")

Example (with extractors):
    from recollectx.llm.providers import OpenAIProvider
    from recollectx.extractors import LLMExtractor

    provider = OpenAIProvider(api_key="sk-...")
    extractor = LLMExtractor(provider)
    claims = extractor.extract("I love pizza on Fridays")

Example (with vector backends):
    from recollectx.llm.providers import LocalEmbedder
    from recollectx.storage.vector import QdrantBackend

    embedder = LocalEmbedder()
    vectors = QdrantBackend.from_provider(
        url="http://localhost:6333",
        collection_name="beliefs",
        embedding_provider=embedder,
    )
"""

from recollectx.llm.providers.anthropic import AnthropicProvider
from recollectx.llm.providers.local import CachedEmbedder, LocalEmbedder
from recollectx.llm.providers.ollama import OllamaProvider
from recollectx.llm.providers.openai import OpenAIProvider
from recollectx.llm.providers.openrouter import OpenRouterProvider

__all__ = [
    # Cloud providers
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    # Local providers
    "OllamaProvider",
    "LocalEmbedder",
    # Utilities
    "CachedEmbedder",
]
