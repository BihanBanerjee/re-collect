"""Qdrant vector backend for production semantic search.

Installation:
    pip install qdrant-client

Example:
    from recollectx.storage.vector import QdrantBackend

    vectors = QdrantBackend(
        url="http://localhost:6333",
        collection_name="beliefs",
        embedding_fn=my_embed_function,
    )

    vectors.upsert("belief-1", "user likes pizza")
    results = vectors.search("food preferences", k=5)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from recollectx.llm.base import EmbeddingProvider

EmbedFn = Callable[[str], list[float]]


class QdrantBackend:
    """Qdrant vector backend for production semantic search.

    Implements the VectorBackend protocol and returns belief IDs only.
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        embedding_fn: EmbedFn,
        dimension: int = 384,
        api_key: str | None = None,
        distance: str = "cosine",
        auto_create_collection: bool = True,
        prefer_grpc: bool = False,
        timeout: float | None = None,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._embed_fn = embedding_fn
        self._dimension = dimension
        self._api_key = api_key
        self._distance = distance
        self._auto_create = auto_create_collection
        self._prefer_grpc = prefer_grpc
        self._timeout = timeout

        self._client: QdrantClient | None = None
        self._initialized = False

    @classmethod
    def from_provider(
        cls,
        url: str,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        api_key: str | None = None,
        distance: str = "cosine",
        auto_create_collection: bool = True,
        **kwargs: Any,
    ) -> QdrantBackend:
        """Create a QdrantBackend using an EmbeddingProvider."""
        return cls(
            url=url,
            collection_name=collection_name,
            embedding_fn=embedding_provider.embed,
            dimension=embedding_provider.dimension,
            api_key=api_key,
            distance=distance,
            auto_create_collection=auto_create_collection,
            **kwargs,
        )

    def _get_client(self) -> QdrantClient:
        """Get or create the Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise ImportError(
                    "qdrant-client is required for QdrantBackend. "
                    "Install it with: pip install qdrant-client"
                ) from e

            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                prefer_grpc=self._prefer_grpc,
                timeout=self._timeout,
            )

        return self._client

    def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        if self._initialized:
            return

        client = self._get_client()

        collections = client.get_collections()
        exists = any(c.name == self._collection_name for c in collections.collections)

        if not exists and self._auto_create:
            from qdrant_client.models import Distance, VectorParams

            distance_map = {
                "cosine": Distance.COSINE,
                "euclid": Distance.EUCLID,
                "dot": Distance.DOT,
            }
            distance = distance_map.get(self._distance.lower(), Distance.COSINE)

            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=distance,
                ),
            )

        self._initialized = True

    def upsert(self, belief_id: str, text: str) -> None:
        """Insert or update a belief's vector representation."""
        if not text.strip():
            return

        self._ensure_collection()
        client = self._get_client()

        vector = self._embed_fn(text)

        from qdrant_client.models import PointStruct

        point = PointStruct(
            id=belief_id,
            vector=vector,
            payload={"text": text},
        )

        client.upsert(
            collection_name=self._collection_name,
            points=[point],
        )

    def delete(self, belief_id: str) -> None:
        """Remove a belief's vector representation."""
        self._ensure_collection()
        client = self._get_client()

        from qdrant_client.models import PointIdsList

        client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=[belief_id]),
        )

    def search(self, query: str, k: int = 10) -> list[str]:
        """Search for semantically similar beliefs. Returns IDs only."""
        if not query.strip():
            return []

        self._ensure_collection()
        client = self._get_client()

        query_vector = self._embed_fn(query)

        results = client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=k,
        )

        return [str(point.id) for point in results]

    def search_with_scores(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search for similar beliefs, returning IDs with similarity scores."""
        if not query.strip():
            return []

        self._ensure_collection()
        client = self._get_client()

        query_vector = self._embed_fn(query)

        results = client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=k,
        )

        return [(str(point.id), point.score) for point in results]

    def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._initialized = False
