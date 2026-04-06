"""Tests for MemoryStore CRUD and belief_to_text (storage/memory_store.py)."""

import pytest

from recollectx.claims import EpisodicClaim, SemanticClaim
from recollectx.graph.edges import BeliefEdge
from recollectx.propagation import ConfidenceChangeEvent
from recollectx.storage.memory_store import belief_to_text, _apply_recency_boost


# ---------------------------------------------------------------------------
# belief_to_text
# ---------------------------------------------------------------------------

class TestBeliefToText:
    def test_semantic_formats_spo(self):
        c = SemanticClaim(subject="user", predicate="likes", object="pizza", confidence=0.9)
        assert belief_to_text(c) == "user likes pizza"

    def test_episodic_returns_summary(self):
        c = EpisodicClaim(summary="Had coffee this morning", confidence=0.8)
        assert belief_to_text(c) == "Had coffee this morning"

    def test_base_claim_returns_empty(self):
        from recollectx.claims import Claim
        c = Claim(type="", confidence=0.5)
        assert belief_to_text(c) == ""


# ---------------------------------------------------------------------------
# _apply_recency_boost
# ---------------------------------------------------------------------------

class TestRecencyBoost:
    def test_empty_list_returns_empty(self):
        assert _apply_recency_boost([], 1.0) == []

    def test_single_item_unchanged(self):
        c = EpisodicClaim(summary="x", confidence=0.8)
        result = _apply_recency_boost([c], 0.5)
        assert result == [c]

    def test_more_important_claim_ranks_higher(self):
        import time
        now = time.time()
        low = EpisodicClaim(summary="low", confidence=0.8, importance=0.1, created_at=now)
        high = EpisodicClaim(summary="high", confidence=0.8, importance=0.9, created_at=now)
        # Both same age — higher importance should win
        result = _apply_recency_boost([high, low], 0.1)
        assert result[0] == high


# ---------------------------------------------------------------------------
# MemoryStore CRUD
# ---------------------------------------------------------------------------

class TestMemoryStoreCRUD:
    def test_put_and_get(self, memory_store, episodic):
        memory_store.put(episodic)
        retrieved = memory_store.get(episodic.id)
        assert retrieved is not None
        assert retrieved.id == episodic.id

    def test_get_nonexistent_returns_none(self, memory_store):
        assert memory_store.get("no-such-id") is None

    def test_put_semantic_claim(self, memory_store, semantic):
        memory_store.put(semantic)
        retrieved = memory_store.get(semantic.id)
        assert isinstance(retrieved, SemanticClaim)
        assert retrieved.subject == "user"
        assert retrieved.predicate == "works_as"
        assert retrieved.object == "software engineer"

    def test_delete_existing(self, memory_store, episodic):
        memory_store.put(episodic)
        deleted = memory_store.delete(episodic.id)
        assert deleted is True
        assert memory_store.get(episodic.id) is None

    def test_delete_nonexistent_returns_false(self, memory_store):
        assert memory_store.delete("ghost-id") is False

    def test_update_claim(self, memory_store, episodic):
        memory_store.put(episodic)
        from dataclasses import replace
        updated = replace(episodic, confidence=0.55)
        memory_store.update(updated)
        retrieved = memory_store.get(episodic.id)
        assert retrieved.confidence == pytest.approx(0.55, abs=0.001)

    def test_count_all(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        assert memory_store.count() == 2

    def test_count_by_type(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        assert memory_store.count("episodic") == 1
        assert memory_store.count("semantic") == 1

    def test_count_empty(self, memory_store):
        assert memory_store.count() == 0


# ---------------------------------------------------------------------------
# MemoryStore query
# ---------------------------------------------------------------------------

class TestMemoryStoreQuery:
    def test_query_all(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        all_claims = memory_store.query()
        assert len(all_claims) == 2

    def test_query_by_type_episodic(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        results = memory_store.query(type="episodic")
        assert all(c.type == "episodic" for c in results)
        assert len(results) == 1

    def test_query_by_type_semantic(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        results = memory_store.query(type="semantic")
        assert all(c.type == "semantic" for c in results)
        assert len(results) == 1

    def test_query_min_confidence(self, memory_store, episodic, low_confidence_claim):
        memory_store.put(episodic)
        memory_store.put(low_confidence_claim)
        results = memory_store.query(min_confidence=0.5)
        ids = {c.id for c in results}
        assert episodic.id in ids
        assert low_confidence_claim.id not in ids

    def test_query_empty_store(self, memory_store):
        assert memory_store.query() == []


# ---------------------------------------------------------------------------
# MemoryStore semantic_query
# ---------------------------------------------------------------------------

class TestMemoryStoreSemanticQuery:
    def test_semantic_query_empty_string_returns_empty(self, memory_store, episodic):
        memory_store.put(episodic)
        result = memory_store.semantic_query("")
        assert result == []

    def test_semantic_query_finds_matching_claim(self, memory_store, episodic):
        memory_store.put(episodic)
        result = memory_store.semantic_query("coffee")
        assert any(c.id == episodic.id for c in result)

    def test_semantic_query_respects_type_filter(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        result = memory_store.semantic_query("user", type="semantic")
        assert all(c.type == "semantic" for c in result)

    def test_semantic_query_respects_k_limit(self, memory_store):
        for i in range(10):
            memory_store.put(EpisodicClaim(summary=f"event {i}", confidence=0.8))
        result = memory_store.semantic_query("event", k=3)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# MemoryStore edges
# ---------------------------------------------------------------------------

class TestMemoryStoreEdges:
    def test_put_and_get_edge(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        edge = BeliefEdge(episodic.id, semantic.id, "supports")
        memory_store.put_edge(edge)
        edges = memory_store.get_edges(src_id=episodic.id)
        assert len(edges) == 1
        assert edges[0].dst_id == semantic.id

    def test_duplicate_edge_not_stored_twice(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        edge = BeliefEdge(episodic.id, semantic.id, "supports")
        memory_store.put_edge(edge)
        memory_store.put_edge(edge)  # duplicate
        edges = memory_store.get_edges(src_id=episodic.id, relation="supports")
        assert len(edges) == 1

    def test_delete_edge(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        edge = BeliefEdge(episodic.id, semantic.id, "supports")
        memory_store.put_edge(edge)
        deleted = memory_store.delete_edge(episodic.id, semantic.id, "supports")
        assert deleted is True
        assert memory_store.get_edges(src_id=episodic.id) == []

    def test_get_all_edges(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        e1 = BeliefEdge(episodic.id, semantic.id, "supports")
        e2 = BeliefEdge(semantic.id, episodic.id, "derives")
        memory_store.put_edge(e1)
        memory_store.put_edge(e2)
        all_edges = memory_store.get_all_edges()
        assert len(all_edges) == 2

    def test_get_edges_by_relation(self, memory_store, episodic, semantic):
        memory_store.put(episodic)
        memory_store.put(semantic)
        memory_store.put_edge(BeliefEdge(episodic.id, semantic.id, "supports"))
        memory_store.put_edge(BeliefEdge(episodic.id, semantic.id, "contradicts"))
        supports = memory_store.get_edges(relation="supports")
        assert all(e.relation == "supports" for e in supports)


# ---------------------------------------------------------------------------
# MemoryStore confidence history
# ---------------------------------------------------------------------------

class TestConfidenceHistory:
    def test_store_and_retrieve_event(self, memory_store, episodic):
        memory_store.put(episodic)
        event = ConfidenceChangeEvent(
            claim_id=episodic.id,
            old_confidence=0.9,
            new_confidence=0.75,
            reason="Contradicted",
            change_type="contradiction",
        )
        memory_store.put_confidence_event(event)
        history = memory_store.get_confidence_history(episodic.id)
        assert len(history) == 1
        assert history[0].old_confidence == pytest.approx(0.9, abs=0.001)
        assert history[0].new_confidence == pytest.approx(0.75, abs=0.001)

    def test_empty_history_returns_empty(self, memory_store, episodic):
        memory_store.put(episodic)
        assert memory_store.get_confidence_history(episodic.id) == []
