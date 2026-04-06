"""Tests for MemoryUpdater (updater.py)."""

import json
import pytest

from recollectx.claims import EpisodicClaim, SemanticClaim
from recollectx.updater import MemoryUpdater, UpdateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_updater(memory_store, response_json: dict):
    """Return a MemoryUpdater with a mock LLM returning a preset decision."""
    from tests.conftest import MockLLMProvider
    llm = MockLLMProvider(response_json)
    return MemoryUpdater(memory_store, llm, similarity_k=5)


# ---------------------------------------------------------------------------
# ADD path (no similar claims)
# ---------------------------------------------------------------------------

class TestUpdaterAdd:
    def test_add_when_no_similar_found(self, memory_store):
        """With empty store → no similar → ADD without calling LLM."""
        updater = make_updater(memory_store, {})  # LLM won't be called
        claim = EpisodicClaim(summary="Had breakfast", confidence=0.8)
        result = updater.process(claim)
        assert result.action == "ADD"
        assert result.claim is not None
        assert result.claim.id == claim.id
        assert memory_store.get(claim.id) is not None

    def test_add_decision_from_llm(self, memory_store, episodic, semantic):
        """With similar claims → LLM says ADD → new claim is stored."""
        memory_store.put(episodic)
        memory_store.put(semantic)

        new_claim = EpisodicClaim(summary="Ate dinner at 8pm", confidence=0.85)
        updater = make_updater(
            memory_store,
            {
                "decisions": [
                    {
                        "new_memory": "Ate dinner at 8pm",
                        "action": "ADD",
                        "target_id": None,
                        "merged_content": None,
                        "reason": "New information",
                    }
                ],
                "relationships": [],
            },
        )
        result = updater.process(new_claim)
        assert result.action == "ADD"
        assert memory_store.get(new_claim.id) is not None


# ---------------------------------------------------------------------------
# UPDATE path
# ---------------------------------------------------------------------------

class TestUpdaterUpdate:
    def test_update_merges_into_existing(self, memory_store):
        """LLM says UPDATE → existing claim gets new object value."""
        existing = SemanticClaim(
            subject="user", predicate="likes", object="pizza", confidence=0.8,
            evidence=("user likes pizza",)
        )
        memory_store.put(existing)

        new_claim = SemanticClaim(
            subject="user", predicate="likes", object="pepperoni pizza", confidence=0.85,
            evidence=("user likes pepperoni pizza",)
        )
        updater = make_updater(
            memory_store,
            {
                "decisions": [
                    {
                        "new_memory": "user likes pepperoni pizza",
                        "action": "UPDATE",
                        "target_id": existing.id,
                        "merged_content": "pepperoni pizza",
                        "reason": "More specific",
                    }
                ],
                "relationships": [],
            },
        )
        result = updater.process(new_claim)
        assert result.action == "UPDATE"
        updated = memory_store.get(existing.id)
        assert updated is not None
        assert isinstance(updated, SemanticClaim)
        assert updated.object == "pepperoni pizza"

    def test_update_falls_back_to_add_without_target(self, memory_store, episodic):
        """LLM says UPDATE but no valid target_id → falls back to ADD."""
        memory_store.put(episodic)
        new_claim = EpisodicClaim(summary="Had dinner", confidence=0.8)
        updater = make_updater(
            memory_store,
            {
                "decisions": [
                    {
                        "new_memory": "Had dinner",
                        "action": "UPDATE",
                        "target_id": "nonexistent-id",
                        "merged_content": "some content",
                        "reason": "Update attempt",
                    }
                ],
                "relationships": [],
            },
        )
        result = updater.process(new_claim)
        # Falls back to ADD since target_id not in similar
        assert result.action == "ADD"


# ---------------------------------------------------------------------------
# DELETE path
# ---------------------------------------------------------------------------

class TestUpdaterDelete:
    def test_delete_removes_old_and_stores_new(self, memory_store):
        """LLM says DELETE → old claim deleted, new one stored."""
        old_claim = SemanticClaim(
            subject="user", predicate="works_as", object="student", confidence=0.9,
            evidence=("user works_as student",)
        )
        memory_store.put(old_claim)

        new_claim = SemanticClaim(
            subject="user", predicate="works_as", object="engineer", confidence=0.95,
            evidence=("user works_as engineer",)
        )
        updater = make_updater(
            memory_store,
            {
                "decisions": [
                    {
                        "new_memory": "user works_as engineer",
                        "action": "DELETE",
                        "target_id": old_claim.id,
                        "merged_content": None,
                        "reason": "Role changed",
                    }
                ],
                "relationships": [],
            },
        )
        result = updater.process(new_claim)
        assert result.action == "DELETE"
        assert old_claim.id in result.deleted_ids
        assert memory_store.get(old_claim.id) is None
        assert memory_store.get(new_claim.id) is not None

    def test_delete_nonexistent_target_still_adds_new(self, memory_store, episodic):
        """DELETE with bad target_id still stores the new claim."""
        memory_store.put(episodic)
        new_claim = EpisodicClaim(summary="Something else", confidence=0.8)
        updater = make_updater(
            memory_store,
            {
                "decisions": [
                    {
                        "new_memory": "Something else",
                        "action": "DELETE",
                        "target_id": "ghost-id",
                        "merged_content": None,
                        "reason": "Delete attempt",
                    }
                ],
                "relationships": [],
            },
        )
        result = updater.process(new_claim)
        assert result.action == "DELETE"
        assert memory_store.get(new_claim.id) is not None


# ---------------------------------------------------------------------------
# NONE path
# ---------------------------------------------------------------------------

class TestUpdaterNone:
    def test_none_skips_duplicate(self, memory_store, episodic):
        """LLM says NONE → nothing stored, result.claim is None."""
        memory_store.put(episodic)
        duplicate = EpisodicClaim(summary="Had coffee this morning", confidence=0.9)
        updater = make_updater(
            memory_store,
            {
                "decisions": [
                    {
                        "new_memory": "Had coffee this morning",
                        "action": "NONE",
                        "target_id": None,
                        "merged_content": None,
                        "reason": "Duplicate",
                    }
                ],
                "relationships": [],
            },
        )
        result = updater.process(duplicate)
        assert result.action == "NONE"
        assert result.claim is None
        assert memory_store.get(duplicate.id) is None


# ---------------------------------------------------------------------------
# Edge creation
# ---------------------------------------------------------------------------

class TestUpdaterEdges:
    def test_creates_supports_edge(self, memory_store, episodic, semantic):
        """LLM can specify relationships → stored as BeliefEdges."""
        memory_store.put(episodic)
        memory_store.put(semantic)

        new_claim = EpisodicClaim(summary="New event", confidence=0.8)
        updater = make_updater(
            memory_store,
            {
                "decisions": [
                    {
                        "new_memory": "New event",
                        "action": "ADD",
                        "target_id": None,
                        "merged_content": None,
                        "reason": "New",
                    }
                ],
                "relationships": [
                    {"existing_id": episodic.id, "relation": "supports"}
                ],
            },
        )
        result = updater.process(new_claim)
        assert result.action == "ADD"
        assert len(result.edges) == 1
        assert result.edges[0].relation == "supports"
        assert result.edges[0].dst_id == episodic.id

    def test_ignores_invalid_relation_type(self, memory_store, episodic):
        """Relationships with unknown relation types are silently ignored."""
        memory_store.put(episodic)
        new_claim = EpisodicClaim(summary="Another event", confidence=0.8)
        updater = make_updater(
            memory_store,
            {
                "decisions": [
                    {
                        "new_memory": "Another event",
                        "action": "ADD",
                        "target_id": None,
                        "merged_content": None,
                        "reason": "New",
                    }
                ],
                "relationships": [
                    {"existing_id": episodic.id, "relation": "invented_relation"}
                ],
            },
        )
        result = updater.process(new_claim)
        assert len(result.edges) == 0

    def test_ignores_self_referential_edge(self, memory_store, episodic):
        """An edge from a claim to itself is ignored."""
        memory_store.put(episodic)
        new_claim = EpisodicClaim(summary="Self ref", confidence=0.8)
        updater = make_updater(
            memory_store,
            {
                "decisions": [
                    {
                        "new_memory": "Self ref",
                        "action": "ADD",
                        "target_id": None,
                        "merged_content": None,
                        "reason": "New",
                    }
                ],
                "relationships": [
                    # new_claim.id won't be in similar_ids since it's the new claim
                    {"existing_id": new_claim.id, "relation": "supports"}
                ],
            },
        )
        result = updater.process(new_claim)
        assert len(result.edges) == 0


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

class TestUpdaterResponseParsing:
    def test_llm_unavailable_defaults_to_add(self, memory_store, episodic):
        """If LLM raises, updater falls back to ADD."""
        from recollectx.llm.base import LLMResponse

        class FailingLLM:
            def generate(self, *args, **kwargs):
                raise RuntimeError("LLM down")

            def generate_structured(self, *args, **kwargs):
                raise RuntimeError("LLM down")

        memory_store.put(episodic)
        new_claim = EpisodicClaim(summary="Fallback claim", confidence=0.8)
        updater = MemoryUpdater(memory_store, FailingLLM(), similarity_k=5)  # type: ignore[arg-type]
        result = updater.process(new_claim)
        assert result.action == "ADD"
