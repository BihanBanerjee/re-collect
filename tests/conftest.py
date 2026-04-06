"""Shared fixtures for recollectx tests.

All tests use:
- In-memory SQLite (no files on disk)
- A simple mock VectorBackend (dict-based, no embeddings)
- A configurable mock LLMProvider (returns preset JSON)
"""

import json
import pytest

from recollectx.claims import EpisodicClaim, SemanticClaim
from recollectx.graph.edges import BeliefEdge
from recollectx.llm.base import LLMResponse


# ---------------------------------------------------------------------------
# Mock Vector Backend
# ---------------------------------------------------------------------------

class MockVectorBackend:
    """In-memory vector backend — stores texts, returns insertion-order IDs."""

    def __init__(self):
        self._store: dict[str, str] = {}  # id → text

    def upsert(self, belief_id: str, text: str) -> None:
        self._store[belief_id] = text

    def delete(self, belief_id: str) -> None:
        self._store.pop(belief_id, None)

    def search(self, query: str, k: int = 10) -> list[str]:
        # Return all stored IDs (simple keyword match or just all)
        query_lower = query.lower()
        matches = [
            bid for bid, text in self._store.items()
            if query_lower in text.lower()
        ]
        # Fall back to all IDs if no keyword match
        if not matches:
            matches = list(self._store.keys())
        return matches[:k]


# ---------------------------------------------------------------------------
# Mock Embedding Provider
# ---------------------------------------------------------------------------

class MockEmbeddingProvider:
    """Deterministic embeddings based on text content.

    Returns a 4-dimensional unit vector where each dimension is derived
    from the characters in the text. Identical texts → identical vectors
    (similarity = 1.0). Completely different texts → near-zero similarity.
    """

    def embed(self, text: str) -> list[float]:
        import math
        # Produce a repeatable 4-d vector from text
        seed = [float(ord(c)) for c in (text or " ")]
        dims = 4
        v = [sum(seed[i::dims]) for i in range(dims)]
        # Normalise
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 4


# ---------------------------------------------------------------------------
# Mock LLM Provider
# ---------------------------------------------------------------------------

class MockLLMProvider:
    """Returns a pre-configured JSON response for any prompt."""

    def __init__(self, response_json: dict | None = None):
        self._response = response_json or {"memories": []}

    def set_response(self, response_json: dict) -> None:
        self._response = response_json

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        return LLMResponse(content=json.dumps(self._response))

    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict:
        return self._response


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    """Fresh in-memory SQLite session for each test."""
    from recollectx.db.database import reset_engine, create_tables, SessionLocal
    reset_engine()
    create_tables(":memory:")
    db = SessionLocal(":memory:")
    yield db
    db.close()
    reset_engine()


@pytest.fixture
def mock_vectors():
    return MockVectorBackend()


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def mock_embedder():
    return MockEmbeddingProvider()


@pytest.fixture
def memory_store(db_session, mock_vectors):
    from recollectx.storage.memory_store import MemoryStore
    return MemoryStore(db_session, mock_vectors)


# ---------------------------------------------------------------------------
# Sample claims
# ---------------------------------------------------------------------------

@pytest.fixture
def episodic():
    return EpisodicClaim(
        summary="Had coffee this morning",
        confidence=0.9,
        evidence=("Had coffee this morning",),
    )


@pytest.fixture
def semantic():
    return SemanticClaim(
        subject="user",
        predicate="works_as",
        object="software engineer",
        confidence=0.95,
        evidence=("user works_as software engineer",),
    )


@pytest.fixture
def low_confidence_claim():
    return EpisodicClaim(
        summary="Maybe had a snack",
        confidence=0.2,
        evidence=("Maybe had a snack",),
    )
