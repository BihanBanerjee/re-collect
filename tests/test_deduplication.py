"""Tests for the deduplication module (deduplication/)."""

import math
import pytest

from recollectx.claims import EpisodicClaim, SemanticClaim
from recollectx.deduplication.similarity import (
    EmbeddingSimilarity,
    claim_to_text,
    _cosine_similarity,
)
from recollectx.deduplication.merger import LLMMerger, MergeDecision, _rebuild_claim
from recollectx.deduplication.deduplicator import ClaimDeduplicator


# ---------------------------------------------------------------------------
# claim_to_text
# ---------------------------------------------------------------------------

class TestClaimToText:
    def test_semantic_formats_spo(self):
        c = SemanticClaim(subject="user", predicate="likes", object="pizza", confidence=0.9)
        assert claim_to_text(c) == "user likes pizza"

    def test_episodic_returns_summary(self):
        c = EpisodicClaim(summary="Had lunch today", confidence=0.8)
        assert claim_to_text(c) == "Had lunch today"


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_return_zero(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert _cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_return_minus_one(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert _cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_norm_first_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_norm_second_returns_zero(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_partial_similarity(self):
        v1 = [1.0, 1.0]
        v2 = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine_similarity(v1, v2) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# EmbeddingSimilarity
# ---------------------------------------------------------------------------

class TestEmbeddingSimilarity:
    def test_identical_claims_have_high_similarity(self, mock_embedder):
        calc = EmbeddingSimilarity(mock_embedder)
        c1 = EpisodicClaim(summary="Had coffee", confidence=0.8)
        c2 = EpisodicClaim(summary="Had coffee", confidence=0.9)
        score = calc.calculate(c1, c2)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_different_type_claims_return_zero(self, mock_embedder):
        calc = EmbeddingSimilarity(mock_embedder)
        ep = EpisodicClaim(summary="Had coffee", confidence=0.8)
        sem = SemanticClaim(subject="user", predicate="likes", object="coffee", confidence=0.9)
        assert calc.calculate(ep, sem) == 0.0

    def test_similar_text_has_higher_score_than_unrelated(self, mock_embedder):
        calc = EmbeddingSimilarity(mock_embedder)
        base = EpisodicClaim(summary="Had coffee this morning", confidence=0.8)
        similar = EpisodicClaim(summary="Had coffee this morning", confidence=0.9)
        unrelated = EpisodicClaim(summary="xyz pqr abc", confidence=0.8)
        score_similar = calc.calculate(base, similar)
        score_unrelated = calc.calculate(base, unrelated)
        assert score_similar > score_unrelated

    def test_score_in_range(self, mock_embedder):
        calc = EmbeddingSimilarity(mock_embedder)
        c1 = SemanticClaim(subject="u", predicate="likes", object="tea", confidence=0.8)
        c2 = SemanticClaim(subject="u", predicate="hates", object="coffee", confidence=0.8)
        score = calc.calculate(c1, c2)
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# LLMMerger.decide()
# ---------------------------------------------------------------------------

class TestLLMMergerDecide:
    def _make_merger(self, response_json: dict):
        from tests.conftest import MockLLMProvider
        return LLMMerger(MockLLMProvider(response_json))

    def test_decide_add(self):
        merger = self._make_merger({
            "decisions": [
                {
                    "new_memory": "x",
                    "action": "ADD",
                    "target_id": None,
                    "merged_content": None,
                    "reason": "New info",
                }
            ]
        })
        existing = EpisodicClaim(summary="old", confidence=0.8)
        new = EpisodicClaim(summary="new", confidence=0.8)
        decision = merger.decide(new, [existing])
        assert decision.action == "ADD"
        assert decision.target_id is None

    def test_decide_none(self):
        merger = self._make_merger({
            "decisions": [
                {
                    "new_memory": "x",
                    "action": "NONE",
                    "target_id": None,
                    "merged_content": None,
                    "reason": "Duplicate",
                }
            ]
        })
        existing = EpisodicClaim(summary="old", confidence=0.8)
        new = EpisodicClaim(summary="old", confidence=0.8)
        decision = merger.decide(new, [existing])
        assert decision.action == "NONE"

    def test_decide_update(self):
        existing = EpisodicClaim(summary="old info", confidence=0.8)
        merger = self._make_merger({
            "decisions": [
                {
                    "new_memory": "updated info",
                    "action": "UPDATE",
                    "target_id": existing.id,
                    "merged_content": "merged info",
                    "reason": "More specific",
                }
            ]
        })
        new = EpisodicClaim(summary="updated info", confidence=0.85)
        decision = merger.decide(new, [existing])
        assert decision.action == "UPDATE"
        assert decision.target_id == existing.id
        assert decision.merged_content == "merged info"

    def test_decide_delete(self):
        existing = EpisodicClaim(summary="old", confidence=0.8)
        merger = self._make_merger({
            "decisions": [
                {
                    "new_memory": "replacement",
                    "action": "DELETE",
                    "target_id": existing.id,
                    "merged_content": None,
                    "reason": "Obsolete",
                }
            ]
        })
        new = EpisodicClaim(summary="replacement", confidence=0.9)
        decision = merger.decide(new, [existing])
        assert decision.action == "DELETE"
        assert decision.target_id == existing.id

    def test_decide_llm_failure_defaults_to_add(self):
        class FailingLLM:
            def generate_structured(self, *args, **kwargs):
                raise RuntimeError("LLM down")

        merger = LLMMerger(FailingLLM())  # type: ignore[arg-type]
        new = EpisodicClaim(summary="x", confidence=0.8)
        decision = merger.decide(new, [])
        assert decision.action == "ADD"

    def test_decide_empty_decisions_defaults_to_add(self):
        merger = self._make_merger({"decisions": []})
        new = EpisodicClaim(summary="x", confidence=0.8)
        decision = merger.decide(new, [])
        assert decision.action == "ADD"


# ---------------------------------------------------------------------------
# LLMMerger.apply_decision()
# ---------------------------------------------------------------------------

class TestLLMMergerApplyDecision:
    def _make_merger(self):
        from tests.conftest import MockLLMProvider
        return LLMMerger(MockLLMProvider({}))

    def test_apply_add_returns_new_claim(self):
        merger = self._make_merger()
        new = EpisodicClaim(summary="new", confidence=0.8)
        decision = MergeDecision(action="ADD", target_id=None, merged_content=None, reason="")
        result = merger.apply_decision(decision, new, [])
        assert result is new

    def test_apply_none_returns_none(self):
        merger = self._make_merger()
        new = EpisodicClaim(summary="x", confidence=0.8)
        decision = MergeDecision(action="NONE", target_id=None, merged_content=None, reason="")
        result = merger.apply_decision(decision, new, [])
        assert result is None

    def test_apply_update_rebuilds_with_merged_content(self):
        merger = self._make_merger()
        existing = SemanticClaim(
            subject="user", predicate="likes", object="pizza", confidence=0.8,
            evidence=("e1",)
        )
        decision = MergeDecision(
            action="UPDATE",
            target_id=existing.id,
            merged_content="pepperoni pizza",
            reason="More specific",
        )
        result = merger.apply_decision(decision, existing, [existing])
        assert isinstance(result, SemanticClaim)
        assert result.object == "pepperoni pizza"

    def test_apply_delete_returns_new_claim(self):
        merger = self._make_merger()
        new = EpisodicClaim(summary="replacement", confidence=0.9)
        existing = EpisodicClaim(summary="old", confidence=0.8)
        decision = MergeDecision(
            action="DELETE", target_id=existing.id, merged_content=None, reason=""
        )
        result = merger.apply_decision(decision, new, [existing])
        assert result is new


# ---------------------------------------------------------------------------
# _rebuild_claim
# ---------------------------------------------------------------------------

class TestRebuildClaim:
    def test_rebuilds_semantic_object(self):
        original = SemanticClaim(
            subject="user", predicate="likes", object="pizza",
            confidence=0.8, evidence=("e1",), support_count=2
        )
        rebuilt = _rebuild_claim(original, "sushi")
        assert isinstance(rebuilt, SemanticClaim)
        assert rebuilt.object == "sushi"
        assert rebuilt.id == original.id
        assert rebuilt.subject == original.subject
        assert rebuilt.support_count == original.support_count + 1

    def test_rebuilds_episodic_summary(self):
        original = EpisodicClaim(
            summary="Had coffee", confidence=0.8, evidence=("e1",), support_count=1
        )
        rebuilt = _rebuild_claim(original, "Had espresso")
        assert isinstance(rebuilt, EpisodicClaim)
        assert rebuilt.summary == "Had espresso"
        assert rebuilt.id == original.id
        assert rebuilt.support_count == 2


# ---------------------------------------------------------------------------
# ClaimDeduplicator (integration with mock embedder + mock LLM)
# ---------------------------------------------------------------------------

class TestClaimDeduplicator:
    def _make_deduplicator(self, memory_store, mock_embedder, llm_response: dict,
                           threshold: float = 0.5):
        from tests.conftest import MockLLMProvider
        llm = MockLLMProvider(llm_response)
        return ClaimDeduplicator(
            memory_store, mock_embedder, llm, similarity_threshold=threshold
        )

    def test_add_when_no_similar(self, memory_store, mock_embedder):
        """No existing claims → ADD without LLM."""
        deduplicator = self._make_deduplicator(memory_store, mock_embedder, {})
        claim = EpisodicClaim(summary="Had breakfast", confidence=0.8)
        result = deduplicator.process(claim)
        assert result.action == "ADD"
        assert memory_store.get(claim.id) is not None

    def test_add_with_dissimilar_existing(self, memory_store, mock_embedder):
        """Existing claims below threshold → ADD without LLM."""
        existing = EpisodicClaim(summary="xyz pqr abc", confidence=0.8)
        memory_store.put(existing)
        # Use high threshold so nothing matches
        deduplicator = self._make_deduplicator(
            memory_store, mock_embedder, {}, threshold=0.9999
        )
        new_claim = EpisodicClaim(summary="totally different text!", confidence=0.8)
        result = deduplicator.process(new_claim)
        assert result.action == "ADD"

    def test_none_skips_exact_duplicate(self, memory_store, mock_embedder):
        """Identical text hits threshold → LLM says NONE → not stored."""
        text = "Had coffee this morning"
        existing = EpisodicClaim(summary=text, confidence=0.8)
        memory_store.put(existing)
        deduplicator = self._make_deduplicator(
            memory_store,
            mock_embedder,
            {
                "decisions": [
                    {
                        "new_memory": text,
                        "action": "NONE",
                        "target_id": None,
                        "merged_content": None,
                        "reason": "Exact duplicate",
                    }
                ]
            },
            threshold=0.5,
        )
        duplicate = EpisodicClaim(summary=text, confidence=0.8)
        result = deduplicator.process(duplicate)
        assert result.action == "NONE"
        assert result.claim is None

    def test_delete_removes_old(self, memory_store, mock_embedder):
        """Similar claim found → LLM says DELETE → old removed, new stored."""
        text = "Had coffee"
        existing = EpisodicClaim(summary=text, confidence=0.8)
        memory_store.put(existing)
        deduplicator = self._make_deduplicator(
            memory_store,
            mock_embedder,
            {
                "decisions": [
                    {
                        "new_memory": text,
                        "action": "DELETE",
                        "target_id": existing.id,
                        "merged_content": None,
                        "reason": "Replacing old",
                    }
                ]
            },
            threshold=0.5,
        )
        new_claim = EpisodicClaim(summary=text, confidence=0.9)
        result = deduplicator.process(new_claim)
        assert result.action == "DELETE"
        assert existing.id in result.deleted_ids
        assert memory_store.get(existing.id) is None
        assert memory_store.get(new_claim.id) is not None

    def test_different_type_claims_not_compared(self, memory_store, mock_embedder):
        """Episodic claim should not be compared to semantic claim."""
        existing = SemanticClaim(
            subject="user", predicate="likes", object="coffee",
            confidence=0.8, evidence=("e",)
        )
        memory_store.put(existing)
        deduplicator = self._make_deduplicator(
            memory_store, mock_embedder, {}, threshold=0.0
        )
        new_ep = EpisodicClaim(summary="Had coffee", confidence=0.8)
        result = deduplicator.process(new_ep)
        # No similar found (different types) → ADD without LLM
        assert result.action == "ADD"
