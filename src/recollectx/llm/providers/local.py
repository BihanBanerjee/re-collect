"""Local embedding provider using sentence-transformers.

This module provides a local implementation of the EmbeddingProvider protocol
using sentence-transformers, enabling completely local embeddings without API costs.

Installation:
    pip install sentence-transformers

Example:
    from recollectx.llm.providers import LocalEmbedder

    # Create embedder with default model
    embedder = LocalEmbedder()

    # Generate embeddings
    embedding = embedder.embed("Hello world")
    print(f"Dimension: {len(embedding)}")

    # Batch embeddings (more efficient)
    embeddings = embedder.embed_batch(["Hello", "World"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Local embedding provider using sentence-transformers.

    This provider enables completely local embeddings without API costs.
    It uses sentence-transformers models that run on your machine.

    Popular models:
    - all-MiniLM-L6-v2: Fast, good quality (384 dimensions)
    - all-mpnet-base-v2: Higher quality, slower (768 dimensions)
    - paraphrase-multilingual-MiniLM-L12-v2: Multilingual support

    Example:
        # Default model (all-MiniLM-L6-v2)
        embedder = LocalEmbedder()

        # Custom model
        embedder = LocalEmbedder(model_name="all-mpnet-base-v2")

        # Generate embedding
        embedding = embedder.embed("User likes pizza")

        # Use with vector backend
        from recollectx.storage.vector import QdrantBackend
        vectors = QdrantBackend.from_provider(
            url="http://localhost:6333",
            collection_name="beliefs",
            embedding_provider=embedder,
        )

    Attributes:
        model_name: The sentence-transformers model name
        device: Device to run on (cpu, cuda, mps)
    """

    # Default model - good balance of speed and quality
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    # Known model dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-distilroberta-v1": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "multi-qa-mpnet-base-dot-v1": 768,
    }

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        cache_folder: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the local embedder.

        Args:
            model_name: sentence-transformers model name
            device: Device to use (cpu, cuda, mps). None for auto-detect.
            cache_folder: Custom folder for model cache
            normalize_embeddings: Normalize embeddings to unit length

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        self.model_name = model_name
        self._device = device
        self._cache_folder = cache_folder
        self._normalize = normalize_embeddings

        # Lazy model initialization
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    def _get_model(self) -> SentenceTransformer:
        """Get or create the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers package is required for LocalEmbedder. "
                    "Install it with: pip install sentence-transformers"
                ) from e

            kwargs: dict[str, Any] = {}
            if self._device:
                kwargs["device"] = self._device
            if self._cache_folder:
                kwargs["cache_folder"] = self._cache_folder

            self._model = SentenceTransformer(self.model_name, **kwargs)

            # Cache dimension
            self._dimension = self._model.get_sentence_embedding_dimension()

        return self._model

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
        model = self._get_model()
        embedding = model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return cast(list[float], embedding.tolist())

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single call.

        This is more efficient than calling embed() multiple times
        as it batches the computation.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per input text)

        Example:
            texts = ["User likes pizza", "User prefers pasta"]
            embeddings = embedder.embed_batch(texts)
        """
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return cast(list[list[float]], embeddings.tolist())

    @property
    def dimension(self) -> int:
        """Return the dimensionality of embedding vectors.

        Returns:
            Integer dimension of the embedding model
        """
        # Return cached dimension if available
        if self._dimension is not None:
            return self._dimension

        # Try known dimensions
        if self.model_name in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.model_name]

        # Load model to get dimension
        model = self._get_model()
        self._dimension = model.get_sentence_embedding_dimension() or 384
        return self._dimension

    def close(self) -> None:
        """Release resources."""
        self._model = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        pass


class CachedEmbedder:
    """Wrapper that adds caching to any EmbeddingProvider.

    This reduces redundant embedding computations by caching results
    in memory. Useful when the same texts are embedded multiple times.

    Example:
        from recollectx.llm.providers import LocalEmbedder, CachedEmbedder

        base_embedder = LocalEmbedder()
        embedder = CachedEmbedder(base_embedder, max_size=10000)

        # First call computes embedding
        emb1 = embedder.embed("Hello world")

        # Second call returns cached result
        emb2 = embedder.embed("Hello world")  # Instant!

    Attributes:
        provider: The underlying embedding provider
        max_size: Maximum number of cached embeddings
    """

    def __init__(
        self,
        provider: Any,
        max_size: int = 10000,
    ) -> None:
        """Initialize the cached embedder.

        Args:
            provider: Any object implementing EmbeddingProvider protocol
            max_size: Maximum cache size (LRU eviction when exceeded)
        """
        self._provider = provider
        self._max_size = max_size
        self._cache: dict[str, list[float]] = {}
        self._access_order: list[str] = []

    def embed(self, text: str) -> list[float]:
        """Generate embedding with caching.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if text in self._cache:
            # Move to end of access order (most recently used)
            self._access_order.remove(text)
            self._access_order.append(text)
            return self._cache[text]

        # Compute embedding
        embedding: list[float] = cast(list[float], self._provider.embed(text))

        # Cache with LRU eviction
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[text] = embedding
        self._access_order.append(text)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        results: list[list[float]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if text in self._cache:
                results.append(self._cache[text])
                # Update access order
                self._access_order.remove(text)
                self._access_order.append(text)
            else:
                results.append([])  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Compute uncached embeddings
        if uncached_texts:
            new_embeddings = self._provider.embed_batch(uncached_texts)

            for text, embedding, idx in zip(uncached_texts, new_embeddings, uncached_indices):
                results[idx] = embedding

                # Cache with LRU eviction
                if len(self._cache) >= self._max_size:
                    oldest = self._access_order.pop(0)
                    del self._cache[oldest]

                self._cache[text] = embedding
                self._access_order.append(text)

        return results

    @property
    def dimension(self) -> int:
        """Return the dimensionality of embedding vectors."""
        return cast(int, self._provider.dimension)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._access_order.clear()

    @property
    def cache_size(self) -> int:
        """Return current cache size."""
        return len(self._cache)
