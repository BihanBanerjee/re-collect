"""Tests for the Memory class (memory.py) — the main public API."""

import pytest

from recollectx.claims import EpisodicClaim, SemanticClaim
from recollectx.graph.edges import BeliefEdge
from recollectx.memory import Memory
from recollectx.policies.static import MinConfidence, MinEvidence
from recollectx.propagation import PropagationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_memory(memory_store, policy=None, updater=None, propagation_config=None):
    return Memory(
        storage=memory_store,
        write_policy=policy,
        updater=updater,
        propagation_config=propagation_config,
    )


# ---------------------------------------------------------------------------
# store() — basic (no policy, no updater)
# ---------------------------------------------------------------------------

class TestMemoryStore:
    def test_store_persists_episodic(self, memory_store, episodic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        assert memory_store.get(episodic.id) is not None

    def test_store_persists_semantic(self, memory_store, semantic):
        mem = make_memory(memory_store)
        mem.store(semantic)
        retrieved = memory_store.get(semantic.id)
        assert isinstance(retrieved, SemanticClaim)
        assert retrieved.subject == "user"

    def test_store_multiple_claims(self, memory_store, episodic, semantic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        mem.store(semantic)
        assert memory_store.count() == 2


# ---------------------------------------------------------------------------
# store() — with write policy
# ---------------------------------------------------------------------------

class TestMemoryStoreWithPolicy:
    def test_policy_accepts_high_confidence(self, memory_store):
        mem = make_memory(memory_store, policy=MinConfidence(episodic=0.5))
        claim = EpisodicClaim(summary="x", confidence=0.8)
        mem.store(claim)
        assert memory_store.get(claim.id) is not None

    def test_policy_rejects_low_confidence(self, memory_store):
        mem = make_memory(memory_store, policy=MinConfidence(episodic=0.7))
        claim = EpisodicClaim(summary="x", confidence=0.3)
        mem.store(claim)
        assert memory_store.get(claim.id) is None

    def test_composed_policy_rejects_missing_evidence(self, memory_store):
        policy = MinEvidence(2) & MinConfidence(episodic=0.5)
        mem = make_memory(memory_store, policy=policy)
        claim = EpisodicClaim(summary="x", confidence=0.8, evidence=("e1",))
        mem.store(claim)
        assert memory_store.get(claim.id) is None

    def test_composed_policy_accepts_both_satisfied(self, memory_store):
        policy = MinEvidence(1) & MinConfidence(episodic=0.5)
        mem = make_memory(memory_store, policy=policy)
        claim = EpisodicClaim(summary="x", confidence=0.8, evidence=("e1",))
        mem.store(claim)
        assert memory_store.get(claim.id) is not None


# ---------------------------------------------------------------------------
# store() — with updater (mocked)
# ---------------------------------------------------------------------------

class TestMemoryStoreWithUpdater:
    def _make_updater(self, memory_store, action: str, existing=None):
        from tests.conftest import MockLLMProvider
        from recollectx.updater import MemoryUpdater
        llm = MockLLMProvider({
            "decisions": [
                {
                    "new_memory": "x",
                    "action": action,
                    "target_id": existing.id if existing else None,
                    "merged_content": "merged" if action == "UPDATE" else None,
                    "reason": f"test {action}",
                }
            ],
            "relationships": [],
        })
        return MemoryUpdater(memory_store, llm, similarity_k=5)

    def test_updater_add_stores_claim(self, memory_store, episodic):
        memory_store.put(episodic)  # need existing so updater triggers LLM
        updater = self._make_updater(memory_store, "ADD")
        mem = make_memory(memory_store, updater=updater)
        new_claim = EpisodicClaim(summary="New event", confidence=0.8)
        mem.store(new_claim)
        assert memory_store.get(new_claim.id) is not None

    def test_updater_none_skips_claim(self, memory_store, episodic):
        memory_store.put(episodic)
        updater = self._make_updater(memory_store, "NONE")
        mem = make_memory(memory_store, updater=updater)
        duplicate = EpisodicClaim(summary="Had coffee this morning", confidence=0.9)
        mem.store(duplicate)
        assert memory_store.get(duplicate.id) is None

    def test_updater_edges_added_to_graph(self, memory_store, episodic):
        memory_store.put(episodic)
        from tests.conftest import MockLLMProvider
        from recollectx.updater import MemoryUpdater
        llm = MockLLMProvider({
            "decisions": [
                {
                    "new_memory": "x",
                    "action": "ADD",
                    "target_id": None,
                    "merged_content": None,
                    "reason": "New",
                }
            ],
            "relationships": [
                {"existing_id": episodic.id, "relation": "supports"}
            ],
        })
        updater = MemoryUpdater(memory_store, llm, similarity_k=5)
        mem = make_memory(memory_store, updater=updater)
        new_claim = EpisodicClaim(summary="New event", confidence=0.8)
        mem.store(new_claim)
        # Edge should be in the in-memory graph
        assert episodic.id in mem.graph.supports(new_claim.id) or \
               new_claim.id in [e.src_id for e in mem.graph.outgoing_edges(new_claim.id)]


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------

class TestMemoryRetrieve:
    def test_retrieve_all(self, memory_store, episodic, semantic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        mem.store(semantic)
        results = mem.retrieve()
        assert len(results) == 2

    def test_retrieve_by_type(self, memory_store, episodic, semantic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        mem.store(semantic)
        episodics = mem.retrieve(type="episodic")
        assert all(c.type == "episodic" for c in episodics)

    def test_retrieve_by_min_confidence(self, memory_store, episodic, low_confidence_claim):
        mem = make_memory(memory_store)
        mem.store(episodic)
        mem.store(low_confidence_claim)
        results = mem.retrieve(min_confidence=0.5)
        ids = {c.id for c in results}
        assert episodic.id in ids
        assert low_confidence_claim.id not in ids

    def test_retrieve_empty(self, memory_store):
        mem = make_memory(memory_store)
        assert mem.retrieve() == []


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------

class TestMemoryExplain:
    def test_explain_returns_belief(self, memory_store, semantic):
        mem = make_memory(memory_store)
        mem.store(semantic)
        result = mem.explain(semantic.id)
        assert result is not None
        assert result["belief"].id == semantic.id

    def test_explain_nonexistent_returns_none(self, memory_store):
        mem = make_memory(memory_store)
        assert mem.explain("ghost-id") is None

    def test_explain_shows_supported_by(self, memory_store, episodic, semantic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        mem.store(semantic)
        mem.add_support(episodic.id, semantic.id)
        result = mem.explain(semantic.id)
        assert episodic.id in result["supported_by"]

    def test_explain_shows_contradicted_by(self, memory_store, episodic, semantic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        mem.store(semantic)
        edge = BeliefEdge(episodic.id, semantic.id, "contradicts")
        memory_store.put_edge(edge)
        mem.graph.add(edge)
        result = mem.explain(semantic.id)
        assert episodic.id in result["contradicted_by"]


# ---------------------------------------------------------------------------
# add_support()
# ---------------------------------------------------------------------------

class TestMemoryAddSupport:
    def test_add_support_creates_edge_in_graph(self, memory_store, episodic, semantic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        mem.store(semantic)
        mem.add_support(episodic.id, semantic.id)
        assert episodic.id in mem.graph.supports(semantic.id)

    def test_add_support_persists_edge(self, memory_store, episodic, semantic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        mem.store(semantic)
        mem.add_support(episodic.id, semantic.id)
        edges = memory_store.get_edges(src_id=episodic.id, relation="supports")
        assert len(edges) == 1

    def test_add_support_missing_src_raises(self, memory_store, semantic):
        mem = make_memory(memory_store)
        mem.store(semantic)
        with pytest.raises(KeyError):
            mem.add_support("ghost-id", semantic.id)

    def test_add_support_missing_dst_raises(self, memory_store, episodic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        with pytest.raises(KeyError):
            mem.add_support(episodic.id, "ghost-id")

    def test_add_support_with_propagation_boosts_confidence(self, memory_store):
        config = PropagationConfig(support_boost=0.10)
        mem = make_memory(memory_store, propagation_config=config)
        supporter = EpisodicClaim(summary="a", confidence=0.8, evidence=("a",))
        supported = EpisodicClaim(summary="b", confidence=0.7, evidence=("b",))
        mem.store(supporter)
        mem.store(supported)
        mem.add_support(supporter.id, supported.id)
        updated = memory_store.get(supported.id)
        assert updated.confidence > 0.7


# ---------------------------------------------------------------------------
# explain_deep()
# ---------------------------------------------------------------------------

class TestMemoryExplainDeep:
    def test_explain_deep_nonexistent_returns_none(self, memory_store):
        mem = make_memory(memory_store)
        assert mem.explain_deep("ghost-id") is None

    def test_explain_deep_root_only(self, memory_store, episodic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        result = mem.explain_deep(episodic.id)
        assert result is not None
        assert result.total_nodes == 1
        assert result.root.belief.id == episodic.id
        assert result.max_depth_reached == 0
        assert not result.cycle_detected

    def test_explain_deep_with_supporter(self, memory_store, episodic, semantic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        mem.store(semantic)
        mem.add_support(episodic.id, semantic.id)
        result = mem.explain_deep(semantic.id)
        assert result is not None
        assert result.total_nodes == 2
        assert result.max_depth_reached == 1
        assert len(result.root.children) == 1
        assert result.root.children[0].belief.id == episodic.id

    def test_explain_deep_respects_max_depth(self, memory_store):
        mem = make_memory(memory_store)
        claims = [EpisodicClaim(summary=f"e{i}", confidence=0.8) for i in range(4)]
        for c in claims:
            mem.store(c)
        # chain: claims[0] supports claims[1] supports claims[2] supports claims[3]
        for i in range(3):
            mem.add_support(claims[i].id, claims[i + 1].id)
        result = mem.explain_deep(claims[3].id, max_depth=2)
        assert result.max_depth_reached <= 2


# ---------------------------------------------------------------------------
# explain_confidence_history()
# ---------------------------------------------------------------------------

class TestMemoryConfidenceHistory:
    def test_history_empty_initially(self, memory_store, episodic):
        mem = make_memory(memory_store)
        mem.store(episodic)
        assert mem.explain_confidence_history(episodic.id) == []

    def test_history_records_support_event(self, memory_store):
        config = PropagationConfig(support_boost=0.10)
        mem = make_memory(memory_store, propagation_config=config)
        supporter = EpisodicClaim(summary="a", confidence=0.8, evidence=("a",))
        supported = EpisodicClaim(summary="b", confidence=0.7, evidence=("b",))
        mem.store(supporter)
        mem.store(supported)
        mem.add_support(supporter.id, supported.id)
        history = mem.explain_confidence_history(supported.id)
        assert len(history) >= 1
        assert history[0].change_type == "support"


# ---------------------------------------------------------------------------
# Graph hydration on init
# ---------------------------------------------------------------------------

class TestMemoryHydration:
    def test_hydrates_existing_edges_from_storage(self, memory_store, episodic, semantic):
        """Edges persisted in DB should appear in graph when Memory is created."""
        memory_store.put(episodic)
        memory_store.put(semantic)
        edge = BeliefEdge(episodic.id, semantic.id, "supports")
        memory_store.put_edge(edge)

        # Create a fresh Memory — it should hydrate the graph
        mem = make_memory(memory_store)
        assert episodic.id in mem.graph.supports(semantic.id)
