"""Pinecone vector backend for production semantic search.

Installation:
    pip install pinecone-client

Example:
    from recollectx.storage.vector import PineconeBackend

    vectors = PineconeBackend(
        api_key="your-api-key",
        index_name="beliefs",
        embedding_fn=my_embed_function,
        dimension=384,
    )

    vectors.upsert("belief-1", "user likes pizza")
    results = vectors.search("food preferences", k=5)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone import Index  # type: ignore[attr-defined]

    from recollectx.llm.base import EmbeddingProvider

EmbedFn = Callable[[str], list[float]]


class PineconeBackend:
    """Pinecone vector backend for production semantic search.

    Implements the VectorBackend protocol and returns belief IDs only.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding_fn: EmbedFn,
        dimension: int = 384,
        host: str | None = None,
        namespace: str = "",
        metric: str = "cosine",
        auto_create_index: bool = True,
        cloud: str = "aws",
        region: str = "us-east-1",
        spec_type: str = "serverless",
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._embed_fn = embedding_fn
        self._dimension = dimension
        self._host = host
        self._namespace = namespace
        self._metric = metric
        self._auto_create = auto_create_index
        self._cloud = cloud
        self._region = region
        self._spec_type = spec_type

        self._index: Index | None = None
        self._initialized = False

    @classmethod
    def from_provider(
        cls,
        api_key: str,
        index_name: str,
        embedding_provider: EmbeddingProvider,
        namespace: str = "",
        metric: str = "cosine",
        auto_create_index: bool = True,
        **kwargs: Any,
    ) -> PineconeBackend:
        """Create a PineconeBackend using an EmbeddingProvider."""
        return cls(
            api_key=api_key,
            index_name=index_name,
            embedding_fn=embedding_provider.embed,
            dimension=embedding_provider.dimension,
            namespace=namespace,
            metric=metric,
            auto_create_index=auto_create_index,
            **kwargs,
        )

    def _get_index(self) -> Index:
        """Get or create the Pinecone index."""
        if self._index is not None:
            return self._index

        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as e:
            raise ImportError(
                "pinecone-client is required for PineconeBackend. "
                "Install it with: pip install pinecone-client"
            ) from e

        pc = Pinecone(api_key=self._api_key)

        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]

        if self._index_name not in index_names:
            if self._auto_create:
                pc.create_index(
                    name=self._index_name,
                    dimension=self._dimension,
                    metric=self._metric,
                    spec=ServerlessSpec(
                        cloud=self._cloud,
                        region=self._region,
                    ),
                )
                self._wait_for_index_ready(pc)

        if self._host:
            self._index = pc.Index(name=self._index_name, host=self._host)
        else:
            self._index = pc.Index(name=self._index_name)

        self._initialized = True
        return self._index

    def _wait_for_index_ready(self, pc: Any, max_wait: int = 120) -> None:
        """Wait for the index to be ready."""
        start = time.time()
        while time.time() - start < max_wait:
            try:
                index_info = pc.describe_index(self._index_name)
                if index_info.status.ready:
                    return
            except Exception:
                pass
            time.sleep(2)

    def upsert(self, belief_id: str, text: str) -> None:
        """Insert or update a belief's vector representation."""
        if not text.strip():
            return

        index = self._get_index()
        vector = self._embed_fn(text)

        index.upsert(
            vectors=[
                {
                    "id": belief_id,
                    "values": vector,
                    "metadata": {"text": text},
                }
            ],
            namespace=self._namespace,
        )

    def upsert_batch(
        self,
        items: list[tuple[str, str]],
        batch_size: int = 100,
    ) -> None:
        """Batch upsert multiple beliefs."""
        if not items:
            return

        index = self._get_index()

        vectors_data = []
        for belief_id, text in items:
            if text.strip():
                vector = self._embed_fn(text)
                vectors_data.append({
                    "id": belief_id,
                    "values": vector,
                    "metadata": {"text": text},
                })

        for i in range(0, len(vectors_data), batch_size):
            batch = vectors_data[i : i + batch_size]
            index.upsert(vectors=batch, namespace=self._namespace)

    def delete(self, belief_id: str) -> None:
        """Remove a belief's vector representation."""
        index = self._get_index()
        index.delete(ids=[belief_id], namespace=self._namespace)

    def delete_batch(self, belief_ids: list[str]) -> None:
        """Batch delete multiple beliefs."""
        if not belief_ids:
            return
        index = self._get_index()
        index.delete(ids=belief_ids, namespace=self._namespace)

    def search(self, query: str, k: int = 10) -> list[str]:
        """Search for semantically similar beliefs. Returns IDs only."""
        if not query.strip():
            return []

        index = self._get_index()
        query_vector = self._embed_fn(query)

        results = index.query(
            vector=query_vector,
            top_k=k,
            namespace=self._namespace,
            include_metadata=False,
        )

        return [match.id for match in results.matches]

    def search_with_scores(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search for similar beliefs, returning IDs with similarity scores."""
        if not query.strip():
            return []

        index = self._get_index()
        query_vector = self._embed_fn(query)

        results = index.query(
            vector=query_vector,
            top_k=k,
            namespace=self._namespace,
            include_metadata=False,
        )

        return [(match.id, match.score) for match in results.matches]

    def close(self) -> None:
        """Close the Pinecone connection."""
        self._index = None
        self._initialized = False

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        if self._index is None:
            return {}

        stats = self._index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
            "dimension": stats.dimension,
        }
