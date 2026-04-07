"""MemoryStore — the main storage layer for re-collect.

Combines SQLite (authoritative storage via SQLAlchemy) with vector search
(semantic recall) into a single interface. All CRUD goes through SQLite;
vectors accelerate semantic queries.

Architecture:
    User query
        |
    Vector search (candidate belief IDs)
        |
    Storage fetch (authoritative beliefs)
        |
    Graph filtering (contradictions)
        |
    Confidence + decay filtering
        |
    Final belief set
"""

import logging
import math
import time
from typing import Any

from sqlalchemy.orm import Session

from ..claims import Claim, EpisodicClaim, SemanticClaim
from ..db.converters import (
    claim_to_model,
    confidence_event_to_model,
    edge_to_model,
    model_to_claim,
    model_to_confidence_event,
    model_to_edge,
)
from ..db.models.belief_edge import BeliefEdgeModel
from ..db.models.claim import ClaimModel
from ..db.models.confidence_history import ConfidenceHistoryModel
from ..graph.edges import BeliefEdge
from ..propagation import ConfidenceChangeEvent
from .vector.base import VectorBackend

logger = logging.getLogger(__name__)


def belief_to_text(belief: Claim) -> str:
    """Convert a belief to normalized text for embedding.

    Produces clean, structured text suitable for embedding.
    Never embeds raw conversation, logs, or chain-of-thought.

    Args:
        belief: The belief to convert

    Returns:
        Normalized text representation, or empty string if not embeddable.
    """
    if isinstance(belief, SemanticClaim):
        return f"{belief.subject} {belief.predicate} {belief.object}"

    if isinstance(belief, EpisodicClaim):
        return belief.summary

    return ""


def _apply_recency_boost(
    claims: list[Claim],
    boost_factor: float,
) -> list[Claim]:
    """Re-rank claims using importance-modulated exponential decay scoring.

    Combines semantic similarity (position), importance, and temporal decay
    into a single score:
        score = position_score × exp(-λ × hours_elapsed × (1 - importance))

    Decay rates by claim type:
        episodic: λ = 0.001  (faster decay — events are time-bound)
        semantic: λ = 0.0001 (slower decay — facts are durable)

    High-importance claims decay much more slowly regardless of type.
    The boost_factor scales both λ values proportionally.
    """
    _LAMBDA_EPISODIC = 0.001
    _LAMBDA_SEMANTIC = 0.0001

    now = time.time()
    n = len(claims)

    scored: list[tuple[float, Claim]] = []
    for i, claim in enumerate(claims):
        position_score = 1.0 - (i / n) if n > 1 else 1.0
        hours_elapsed = (now - claim.created_at) / 3600
        lam = _LAMBDA_EPISODIC if claim.type == "episodic" else _LAMBDA_SEMANTIC
        lam *= boost_factor  # scale by caller's bias (default 1.0 → no change)
        decay = math.exp(-lam * hours_elapsed * (1.0 - claim.importance))
        combined = position_score * decay
        scored.append((combined, claim))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [claim for _, claim in scored]


class MemoryStore:
    """The main storage layer for re-collect.

    Combines a SQLAlchemy Session (SQLite) with a vector backend for
    semantic search. SQLite is authoritative for all CRUD operations;
    vectors provide fuzzy recall.

    Example:
        from recollectx.db import SessionLocal, create_tables
        from recollectx.storage import MemoryStore
        from recollectx.storage.vector import FAISSBackend

        create_tables()
        db = SessionLocal()
        vectors = FAISSBackend(embed_fn=my_embed, dimension=384)
        store = MemoryStore(db, vectors)

        store.put(belief)
        results = store.semantic_query("user preferences")
    """

    def __init__(
        self,
        db: Session,
        vectors: VectorBackend,
        *,
        similarity_threshold: float = 0.7,
        max_connections: int = 5,
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.similarity_threshold = similarity_threshold
        self.max_connections = max_connections

    # =========================================================================
    # Claim CRUD
    # =========================================================================

    def put(self, item: Claim) -> None:
        """Store a belief in both storage and vectors."""
        model = claim_to_model(item)
        self.db.merge(model)
        self.db.commit()

        text = belief_to_text(item)
        if text:
            try:
                self.vectors.upsert(item.id, text)
            except Exception:
                pass

            self._create_similarity_edges(item.id, text)

    def update(self, item: Claim) -> None:
        """Update a belief in both storage and vectors."""
        model = claim_to_model(item)
        self.db.merge(model)
        self.db.commit()

        text = belief_to_text(item)
        if text:
            try:
                self.vectors.upsert(item.id, text)
            except Exception:
                pass

    def get(self, id: str) -> Claim | None:
        """Retrieve a belief by ID from storage."""
        model = self.db.get(ClaimModel, id)
        return model_to_claim(model) if model else None

    def delete(self, id: str) -> bool:
        """Delete a belief from storage, vectors, and similarity edges."""
        model = self.db.get(ClaimModel, id)
        if model is None:
            return False

        self.db.delete(model)
        self.db.commit()

        try:
            self.vectors.delete(id)
        except Exception:
            pass

        return True

    def query(
        self,
        type: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[Claim]:
        """Query beliefs by type/confidence (no semantic search)."""
        q = self.db.query(ClaimModel).filter(ClaimModel.confidence >= min_confidence)

        if type is not None:
            q = q.filter(ClaimModel.type == type)

        return [model_to_claim(m) for m in q.all()]

    def count(self, type: str | None = None) -> int:
        """Count stored claims, optionally filtered by type."""
        q = self.db.query(ClaimModel)
        if type is not None:
            q = q.filter(ClaimModel.type == type)
        return q.count()

    # =========================================================================
    # Semantic (vector) query
    # =========================================================================

    def semantic_query(
        self,
        query: str,
        *,
        type: str | None = None,
        min_confidence: float = 0.0,
        k: int = 10,
        recency_bias: float = 0.0,
        episodic_ttl_days: float | None = None,
    ) -> list[Claim]:
        """Query beliefs using semantic similarity.

        Pipeline:
        1. Vectors suggest candidate IDs (over-fetch by 3x)
        2. Storage fetches authoritative beliefs
        3. Hard filters apply (type, confidence)
        4. Soft expiry: drop low-importance old episodic claims (if episodic_ttl_days set)
        5. Importance-modulated exponential decay re-ranking (if recency_bias > 0)
        6. Results are limited

        Args:
            episodic_ttl_days: Base TTL for episodic claims. Effective TTL per claim is
                               ttl_days * (1 + importance), so important episodes survive
                               longer. Semantic claims are never expired by this filter.
        """
        if not query.strip():
            return []

        candidate_ids = self.vectors.search(query, k=k * 3)

        candidates: list[Claim] = []
        for bid in candidate_ids:
            belief = self.get(bid)
            if belief is not None:
                candidates.append(belief)

        filtered = [
            b for b in candidates
            if (type is None or b.type == type)
            and b.confidence >= min_confidence
        ]

        # Soft expiry: episodic claims older than their importance-scaled TTL are dropped
        if episodic_ttl_days is not None:
            now = time.time()
            ttl_seconds = episodic_ttl_days * 86400
            filtered = [
                b for b in filtered
                if b.type != "episodic"
                or (now - b.created_at) <= ttl_seconds * (1.0 + b.importance)
            ]

        if recency_bias > 0 and filtered:
            filtered = _apply_recency_boost(filtered, recency_bias)

        return filtered[:k]

    # =========================================================================
    # Similarity edges
    # =========================================================================

    def _create_similarity_edges(self, claim_id: str, text: str) -> None:
        """Find similar existing claims and create bidirectional 'similar' edges."""
        try:
            if hasattr(self.vectors, "search_with_scores"):
                scored = self.vectors.search_with_scores(text, k=self.max_connections + 1)
                similar_ids = [
                    sid for sid, score in scored
                    if sid != claim_id and score >= self.similarity_threshold
                ][:self.max_connections]
            else:
                candidates = self.vectors.search(text, k=self.max_connections + 1)
                similar_ids = [
                    sid for sid in candidates if sid != claim_id
                ][:self.max_connections]

            for similar_id in similar_ids:
                edge = BeliefEdge(claim_id, similar_id, "similar")
                self.put_edge(edge)
                reverse_edge = BeliefEdge(similar_id, claim_id, "similar")
                self.put_edge(reverse_edge)

            if similar_ids:
                logger.debug(
                    f"Created {len(similar_ids)} similarity connections for {claim_id}"
                )
        except Exception as e:
            logger.debug(f"Similarity connection failed for {claim_id}: {e}")

    def get_connections(self, claim_id: str) -> list[Claim]:
        """Get all claims connected to this one by similarity."""
        edges = self.get_edges(src_id=claim_id, relation="similar")
        reverse_edges = self.get_edges(dst_id=claim_id, relation="similar")

        connected_ids: set[str] = set()
        for e in edges:
            connected_ids.add(e.dst_id)
        for e in reverse_edges:
            connected_ids.add(e.src_id)

        claims: list[Claim] = []
        for cid in connected_ids:
            claim = self.get(cid)
            if claim is not None:
                claims.append(claim)
        return claims

    # =========================================================================
    # Belief edges
    # =========================================================================

    def put_edge(self, edge: BeliefEdge) -> None:
        """Store a belief graph edge. Duplicates are ignored."""
        existing = (
            self.db.query(BeliefEdgeModel)
            .filter_by(src_id=edge.src_id, dst_id=edge.dst_id, relation=edge.relation)
            .first()
        )
        if existing is None:
            model = edge_to_model(edge)
            self.db.add(model)
            self.db.commit()

    def get_edges(
        self,
        src_id: str | None = None,
        dst_id: str | None = None,
        relation: Any = None,
    ) -> list[BeliefEdge]:
        """Query belief graph edges."""
        q = self.db.query(BeliefEdgeModel)
        if src_id is not None:
            q = q.filter(BeliefEdgeModel.src_id == src_id)
        if dst_id is not None:
            q = q.filter(BeliefEdgeModel.dst_id == dst_id)
        if relation is not None:
            q = q.filter(BeliefEdgeModel.relation == relation)
        return [model_to_edge(m) for m in q.all()]

    def delete_edge(self, src_id: str, dst_id: str, relation: str) -> bool:
        """Delete a specific edge."""
        count = (
            self.db.query(BeliefEdgeModel)
            .filter_by(src_id=src_id, dst_id=dst_id, relation=relation)
            .delete()
        )
        self.db.commit()
        return count > 0

    def get_all_edges(self) -> list[BeliefEdge]:
        """Get all belief graph edges."""
        return [model_to_edge(m) for m in self.db.query(BeliefEdgeModel).all()]

    # =========================================================================
    # Confidence history
    # =========================================================================

    def put_confidence_event(self, event: ConfidenceChangeEvent) -> None:
        """Store a confidence change event."""
        model = confidence_event_to_model(event)
        self.db.add(model)
        self.db.commit()

    def get_confidence_history(self, claim_id: str) -> list[ConfidenceChangeEvent]:
        """Get confidence change history for a claim, newest first."""
        models = (
            self.db.query(ConfidenceHistoryModel)
            .filter(ConfidenceHistoryModel.claim_id == claim_id)
            .order_by(ConfidenceHistoryModel.timestamp.desc())
            .all()
        )
        return [model_to_confidence_event(m) for m in models]
