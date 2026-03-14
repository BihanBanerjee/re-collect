"""FAISS vector backend for local semantic search.

FAISS (Facebook AI Similarity Search) provides high-performance vector
search without requiring an external server.

Installation:
    pip install faiss-cpu

Example:
    from re_collect.storage.vector import FAISSBackend

    vectors = FAISSBackend(
        embed_fn=model.encode,
        dimension=384,
    )

    vectors.upsert("belief-1", "user likes pizza")
    results = vectors.search("food preferences", k=5)
"""

from __future__ import annotations

from collections.abc import Callable

# Type for embedding functions (sync only)
EmbedFn = Callable[[str], list[float]]


class FAISSBackend:
    """FAISS vector backend for local semantic search.

    Uses FAISS for fast approximate nearest neighbor search without
    requiring an external server. Implements the VectorBackend protocol
    and returns belief IDs only.

    Features:
    - No server required — runs in-process
    - Fast indexed search (IndexFlatIP with normalized vectors = cosine similarity)
    - Supports save/load for persistence
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        dimension: int,
        index_path: str | None = None,
    ) -> None:
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is required for FAISSBackend. "
                "Install it with: pip install faiss-cpu"
            ) from e

        self._embed_fn = embed_fn
        self._dimension = dimension
        self._index_path = index_path

        self._index = faiss.IndexFlatIP(dimension)

        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._next_idx = 0

        if index_path:
            self._try_load(index_path)

    def _try_load(self, path: str) -> None:
        """Try to load a saved index from disk."""
        import os
        import json
        import faiss

        if os.path.exists(path):
            self._index = faiss.read_index(path)
            mapping_path = path + ".mapping.json"
            if os.path.exists(mapping_path):
                with open(mapping_path) as f:
                    data = json.load(f)
                self._id_to_idx = data["id_to_idx"]
                self._idx_to_id = {int(k): v for k, v in data["idx_to_id"].items()}
                self._next_idx = data["next_idx"]

    def save(self) -> None:
        """Save the index and ID mapping to disk."""
        if not self._index_path:
            raise ValueError("No index_path configured. Pass index_path to __init__.")

        import json
        import faiss

        faiss.write_index(self._index, self._index_path)

        mapping_path = self._index_path + ".mapping.json"
        with open(mapping_path, "w") as f:
            json.dump({
                "id_to_idx": self._id_to_idx,
                "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
                "next_idx": self._next_idx,
            }, f)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        """L2-normalize a vector for cosine similarity via inner product."""
        norm = sum(x * x for x in vector) ** 0.5
        if norm == 0:
            return vector
        return [x / norm for x in vector]

    def upsert(self, belief_id: str, text: str) -> None:
        """Insert or update a belief's vector representation."""
        if not text.strip():
            return

        import numpy as np

        vector = self._embed_fn(text)
        normalized = self._normalize(vector)
        vec_array = np.array([normalized], dtype=np.float32)

        if belief_id in self._id_to_idx:
            old_idx = self._id_to_idx[belief_id]
            del self._idx_to_id[old_idx]

        idx = self._next_idx
        self._index.add(vec_array)
        self._id_to_idx[belief_id] = idx
        self._idx_to_id[idx] = belief_id
        self._next_idx += 1

    def delete(self, belief_id: str) -> None:
        """Remove a belief's vector representation."""
        if belief_id in self._id_to_idx:
            idx = self._id_to_idx.pop(belief_id)
            self._idx_to_id.pop(idx, None)

    def search(self, query: str, k: int = 10) -> list[str]:
        """Search for semantically similar beliefs."""
        results = self.search_with_scores(query, k=k)
        return [belief_id for belief_id, _ in results]

    def search_with_scores(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search for similar beliefs, returning IDs with similarity scores."""
        if not query.strip() or not self._id_to_idx:
            return []

        import numpy as np

        query_vector = self._embed_fn(query)
        normalized = self._normalize(query_vector)
        query_array = np.array([normalized], dtype=np.float32)

        fetch_k = min(k * 3, self._index.ntotal)
        if fetch_k == 0:
            return []

        scores, indices = self._index.search(query_array, fetch_k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            belief_id = self._idx_to_id.get(int(idx))
            if belief_id is not None:
                results.append((belief_id, float(score)))
                if len(results) >= k:
                    break

        return results

    def __len__(self) -> int:
        """Return the number of active vectors."""
        return len(self._id_to_idx)

    def clear(self) -> None:
        """Clear all stored vectors and reset the index."""
        import faiss

        self._index = faiss.IndexFlatIP(self._dimension)
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._next_idx = 0
