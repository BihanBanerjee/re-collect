"""Vector storage backends for semantic similarity search.

Vector backends provide fuzzy recall for beliefs based on semantic similarity.
They are advisory only - returning candidate IDs for the authoritative
storage layer to fetch and filter.

CRITICAL: Vectors are subordinate to belief logic.
- Vectors suggest candidates
- Storage fetches beliefs
- Graph + confidence decides what survives

Available backends:
- FAISSBackend: Local FAISS-based search (no server required)
- QdrantBackend: Production Qdrant vector database
- PineconeBackend: Production Pinecone managed service

Example (FAISS, local):
    from recollect.storage.vector import FAISSBackend

    vectors = FAISSBackend(
        embed_fn=model.encode,
        dimension=384,
    )
    candidate_ids = vectors.search("user preferences", k=10)

Example (Qdrant):
    from recollect.storage.vector import QdrantBackend

    vectors = QdrantBackend(
        url="http://localhost:6333",
        collection_name="beliefs",
        embedding_fn=model.encode,
        dimension=384,
    )

Example (Pinecone):
    from recollect.storage.vector import PineconeBackend

    vectors = PineconeBackend(
        api_key="your-api-key",
        index_name="beliefs",
        embedding_fn=model.encode,
        dimension=384,
    )
"""

from .base import VectorBackend
from .faiss import FAISSBackend
from .pinecone import PineconeBackend
from .qdrant import QdrantBackend

__all__ = [
    "VectorBackend",
    "FAISSBackend",
    "QdrantBackend",
    "PineconeBackend",
]
