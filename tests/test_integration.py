"""End-to-end integration tests — multiple components working together."""

import pytest

from recollectx.claims import EpisodicClaim, SemanticClaim
from recollectx.memory import Memory
from recollectx.policies.static import MinConfidence
from recollectx.propagation import PropagationConfig
from recollectx.updater import MemoryUpdater
from recollectx.extractors.llm import LLMExtractor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_memory(memory_store, policy=None, updater=None, propagation_config=None):
    return Memory(
        storage=memory_store,
        write_policy=policy,
        updater=updater,
        propagation_config=propagation_config,
    )


# ---------------------------------------------------------------------------
# Store → retrieve round-trip
# ---------------------------------------------------------------------------

class TestStoreRetrieveRoundTrip:
    def test_episodic_survives_round_trip(self, memory_store):
        mem = make_memory(memory_store)
        claim = EpisodicClaim(summary="Attended a conference", confidence=0.85)
        mem.store(claim)
        results = mem.retrieve(type="episodic")
        assert any(c.id == claim.id for c in results)

    def test_semantic_survives_round_trip(self, memory_store):
        mem = make_memory(memory_store)
        claim = SemanticClaim(
            subject="user", predicate="speaks", object="Spanish", confidence=0.9
        )
        mem.store(claim)
        results = mem.retrieve(type="semantic")
        assert any(c.id == claim.id for c in results)
        retrieved = next(c for c in results if c.id == claim.id)
        assert isinstance(retrieved, SemanticClaim)
        assert retrieved.subject == "user"
        assert retrieved.object == "Spanish"

    def test_multiple_types_coexist(self, memory_store):
        mem = make_memory(memory_store)
        ep = EpisodicClaim(summary="Ran 5km today", confidence=0.9)
        sem = SemanticClaim(
            subject="user", predicate="enjoys", object="running", confidence=0.8
        )
        mem.store(ep)
        mem.store(sem)
        assert len(mem.retrieve()) == 2
        assert len(mem.retrieve(type="episodic")) == 1
        assert len(mem.retrieve(type="semantic")) == 1


# ---------------------------------------------------------------------------
# Policy + store
# ---------------------------------------------------------------------------

class TestPolicyStorePipeline:
    def test_only_high_confidence_claims_stored(self, memory_store):
        policy = MinConfidence(episodic=0.7, semantic=0.8)
        mem = make_memory(memory_store, policy=policy)

        high = EpisodicClaim(summary="Certain event", confidence=0.9)
        low = EpisodicClaim(summary="Uncertain event", confidence=0.4)
        mem.store(high)
        mem.store(low)

        results = mem.retrieve()
        ids = {c.id for c in results}
        assert high.id in ids
        assert low.id not in ids

    def test_rejected_count_stays_zero(self, memory_store):
        policy = MinConfidence(episodic=0.95)
        mem = make_memory(memory_store, policy=policy)
        for i in range(5):
            mem.store(EpisodicClaim(summary=f"event {i}", confidence=0.5))
        assert memory_store.count() == 0


# ---------------------------------------------------------------------------
# Propagation + store + add_support
# ---------------------------------------------------------------------------

class TestPropagationPipeline:
    def test_support_boosts_persisted_confidence(self, memory_store):
        config = PropagationConfig(support_boost=0.10)
        mem = make_memory(memory_store, propagation_config=config)

        supporter = SemanticClaim(
            subject="user", predicate="knows", object="Python",
            confidence=0.9, evidence=("e",)
        )
        supported = SemanticClaim(
            subject="user", predicate="is_a", object="developer",
            confidence=0.7, evidence=("e",)
        )
        mem.store(supporter)
        mem.store(supported)
        mem.add_support(supporter.id, supported.id)

        # Fetch from DB — confidence must have been updated
        in_db = memory_store.get(supported.id)
        assert in_db.confidence > 0.7

    def test_contradiction_lowers_persisted_confidence(self, memory_store):
        config = PropagationConfig(contradiction_decay=0.15, symmetric_contradiction=True)
        mem = make_memory(memory_store, propagation_config=config)

        old = SemanticClaim(
            subject="user", predicate="likes", object="tea",
            confidence=0.85, evidence=("e",)
        )
        new = SemanticClaim(
            subject="user", predicate="likes", object="coffee",
            confidence=0.85, evidence=("e",)
        )
        mem.store(old)
        mem.store(new)

        # Manually add contradiction edge and propagate
        from recollectx.graph.edges import BeliefEdge
        edge = BeliefEdge(new.id, old.id, "contradicts")
        memory_store.put_edge(edge)
        mem.graph.add(edge)
        mem._apply_propagation(edge)

        old_in_db = memory_store.get(old.id)
        assert old_in_db.confidence < 0.85

    def test_support_event_in_confidence_history(self, memory_store):
        config = PropagationConfig(support_boost=0.10)
        mem = make_memory(memory_store, propagation_config=config)

        supporter = EpisodicClaim(summary="a", confidence=0.8, evidence=("a",))
        supported = EpisodicClaim(summary="b", confidence=0.7, evidence=("b",))
        mem.store(supporter)
        mem.store(supported)
        mem.add_support(supporter.id, supported.id)

        history = mem.explain_confidence_history(supported.id)
        assert any(h.change_type == "support" for h in history)


# ---------------------------------------------------------------------------
# Updater + Memory: full write pipeline
# ---------------------------------------------------------------------------

class TestUpdaterMemoryPipeline:
    def _make_updater(self, memory_store, action: str, target_id=None, merged=None,
                      relationship=None):
        from tests.conftest import MockLLMProvider
        response = {
            "decisions": [
                {
                    "new_memory": "x",
                    "action": action,
                    "target_id": target_id,
                    "merged_content": merged,
                    "reason": f"test {action}",
                }
            ],
            "relationships": [relationship] if relationship else [],
        }
        llm = MockLLMProvider(response)
        return MemoryUpdater(memory_store, llm, similarity_k=5)

    def test_updater_deduplicates_via_none(self, memory_store):
        existing = EpisodicClaim(summary="Coffee every morning", confidence=0.85)
        memory_store.put(existing)
        updater = self._make_updater(memory_store, "NONE")
        mem = make_memory(memory_store, updater=updater)
        duplicate = EpisodicClaim(summary="Coffee every morning", confidence=0.85)
        mem.store(duplicate)
        # Only the original should exist, not the duplicate
        assert memory_store.get(duplicate.id) is None
        assert memory_store.count() == 1

    def test_updater_delete_replaces_claim(self, memory_store):
        old = SemanticClaim(
            subject="user", predicate="works_as", object="student",
            confidence=0.9, evidence=("e",)
        )
        memory_store.put(old)
        updater = self._make_updater(memory_store, "DELETE", target_id=old.id)
        mem = make_memory(memory_store, updater=updater)
        new = SemanticClaim(
            subject="user", predicate="works_as", object="engineer",
            confidence=0.95, evidence=("e",)
        )
        mem.store(new)
        assert memory_store.get(old.id) is None
        assert memory_store.get(new.id) is not None

    def test_updater_edges_propagate_via_memory(self, memory_store):
        """Edges produced by the updater are reflected in Memory's belief graph."""
        existing = EpisodicClaim(summary="existing event", confidence=0.8)
        memory_store.put(existing)
        updater = self._make_updater(
            memory_store, "ADD",
            relationship={"existing_id": existing.id, "relation": "supports"}
        )
        mem = make_memory(memory_store, updater=updater)
        new = EpisodicClaim(summary="new supporting event", confidence=0.8)
        mem.store(new)
        # The edge should be in the graph (new → existing via supports)
        outgoing = mem.graph.outgoing_edges(new.id)
        assert any(e.relation == "supports" and e.dst_id == existing.id for e in outgoing)


# ---------------------------------------------------------------------------
# Extractor + Memory end-to-end
# ---------------------------------------------------------------------------

class TestExtractorMemoryPipeline:
    def test_extracted_claims_stored_in_memory(self, memory_store):
        from tests.conftest import MockLLMProvider
        llm = MockLLMProvider({
            "memories": [
                {
                    "subject": "user",
                    "predicate": "works_as",
                    "object": "ML engineer",
                    "type": "semantic",
                    "confidence": 0.92,
                },
                {
                    "content": "Started new job this week",
                    "type": "episodic",
                    "confidence": 0.85,
                },
            ]
        })
        extractor = LLMExtractor(llm, min_confidence=0.5)
        mem = make_memory(memory_store)

        claims = extractor.extract("I started a new job as an ML engineer this week.")
        for claim in claims:
            mem.store(claim)

        assert memory_store.count("semantic") == 1
        assert memory_store.count("episodic") == 1

    def test_extractor_filtered_claims_not_stored(self, memory_store):
        """Claims below min_confidence from extractor should never reach Memory."""
        from tests.conftest import MockLLMProvider
        llm = MockLLMProvider({
            "memories": [
                {"content": "low confidence event", "type": "episodic", "confidence": 0.2}
            ]
        })
        extractor = LLMExtractor(llm, min_confidence=0.5)
        mem = make_memory(memory_store)

        claims = extractor.extract("some vague text")
        for claim in claims:
            mem.store(claim)

        assert memory_store.count() == 0


# ---------------------------------------------------------------------------
# LLMExtractor — fallback JSON parsing (regression for missing `import re`)
# ---------------------------------------------------------------------------

class TestExtractorFallbackParsing:
    def test_parses_json_from_markdown_block(self):
        """The generate() fallback path (re.search) must not NameError."""
        import json
        from recollectx.llm.base import LLMResponse

        payload = {"memories": [{"content": "test event", "type": "episodic", "confidence": 0.9}]}
        markdown_response = f"```json\n{json.dumps(payload)}\n```"

        class MarkdownLLM:
            """LLM that has no generate_structured — forces the re fallback."""
            def generate(self, *args, **kwargs):
                return LLMResponse(content=markdown_response)

        extractor = LLMExtractor(MarkdownLLM(), min_confidence=0.5)  # type: ignore[arg-type]
        # Before the fix this raised NameError: name 're' is not defined
        claims = extractor.extract("text")
        assert len(claims) == 1
        assert claims[0].summary == "test event"

    def test_parses_raw_json_object(self):
        """Bare JSON object in response text is also parsed correctly."""
        import json
        from recollectx.llm.base import LLMResponse

        payload = {"memories": [{"content": "raw event", "type": "episodic", "confidence": 0.8}]}

        class RawJsonLLM:
            def generate(self, *args, **kwargs):
                return LLMResponse(content=json.dumps(payload))

        extractor = LLMExtractor(RawJsonLLM(), min_confidence=0.5)  # type: ignore[arg-type]
        claims = extractor.extract("text")
        assert len(claims) == 1

    def test_unparseable_response_returns_empty(self):
        """Completely garbage LLM output returns empty list, not an exception."""
        from recollectx.llm.base import LLMResponse

        class GarbageLLM:
            def generate(self, *args, **kwargs):
                return LLMResponse(content="this is definitely not json at all!!!")

        extractor = LLMExtractor(GarbageLLM(), min_confidence=0.5)  # type: ignore[arg-type]
        claims = extractor.extract("text")
        assert claims == []
