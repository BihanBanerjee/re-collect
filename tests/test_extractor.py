"""Tests for LLMExtractor (extractors/llm.py)."""

import json
import pytest

from recollectx.claims import EpisodicClaim, SemanticClaim
from recollectx.extractors.llm import LLMExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_extractor(response_json: dict, min_confidence: float = 0.5):
    """Return an LLMExtractor backed by a mock LLM."""
    from tests.conftest import MockLLMProvider
    llm = MockLLMProvider(response_json)
    return LLMExtractor(llm, min_confidence=min_confidence)


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------

class TestExtractorBasic:
    def test_extracts_episodic_claim(self):
        extractor = make_extractor({
            "memories": [
                {"content": "Had coffee this morning", "type": "episodic", "confidence": 0.9}
            ]
        })
        claims = extractor.extract("I had coffee this morning.")
        assert len(claims) == 1
        assert isinstance(claims[0], EpisodicClaim)
        assert claims[0].summary == "Had coffee this morning"

    def test_extracts_semantic_claim(self):
        extractor = make_extractor({
            "memories": [
                {
                    "subject": "user",
                    "predicate": "works_as",
                    "object": "data scientist",
                    "type": "semantic",
                    "confidence": 0.95,
                }
            ]
        })
        claims = extractor.extract("I work as a data scientist.")
        assert len(claims) == 1
        assert isinstance(claims[0], SemanticClaim)
        assert claims[0].subject == "user"
        assert claims[0].predicate == "works_as"
        assert claims[0].object == "data scientist"

    def test_extracts_multiple_claims(self):
        extractor = make_extractor({
            "memories": [
                {"content": "Had lunch today", "type": "episodic", "confidence": 0.8},
                {
                    "subject": "user",
                    "predicate": "likes",
                    "object": "sushi",
                    "type": "semantic",
                    "confidence": 0.85,
                },
            ]
        })
        claims = extractor.extract("I had sushi for lunch today.")
        assert len(claims) == 2
        types = {type(c).__name__ for c in claims}
        assert "EpisodicClaim" in types
        assert "SemanticClaim" in types

    def test_empty_response_returns_empty(self):
        extractor = make_extractor({"memories": []})
        claims = extractor.extract("Nothing interesting here.")
        assert claims == []

    def test_returns_empty_on_extraction_failure(self):
        """If LLM raises an exception, extractor returns empty list."""
        from recollectx.llm.base import LLMResponse

        class FailingLLM:
            def generate_structured(self, *args, **kwargs):
                raise RuntimeError("LLM unavailable")

        extractor = LLMExtractor(FailingLLM(), min_confidence=0.5)  # type: ignore[arg-type]
        claims = extractor.extract("text that triggers failure")
        assert claims == []


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------

class TestExtractorConfidenceFilter:
    def test_filters_low_confidence_claims(self):
        extractor = make_extractor(
            {
                "memories": [
                    {"content": "high confidence", "type": "episodic", "confidence": 0.9},
                    {"content": "low confidence", "type": "episodic", "confidence": 0.2},
                ]
            },
            min_confidence=0.5,
        )
        claims = extractor.extract("text")
        assert len(claims) == 1
        assert claims[0].summary == "high confidence"

    def test_accepts_at_exact_threshold(self):
        extractor = make_extractor(
            {"memories": [{"content": "borderline", "type": "episodic", "confidence": 0.5}]},
            min_confidence=0.5,
        )
        claims = extractor.extract("text")
        assert len(claims) == 1

    def test_all_filtered_returns_empty(self):
        extractor = make_extractor(
            {"memories": [{"content": "low", "type": "episodic", "confidence": 0.1}]},
            min_confidence=0.8,
        )
        claims = extractor.extract("text")
        assert claims == []


# ---------------------------------------------------------------------------
# Max claims limit
# ---------------------------------------------------------------------------

class TestExtractorMaxClaims:
    def test_respects_max_claims_per_text(self):
        memories = [
            {"content": f"event {i}", "type": "episodic", "confidence": 0.9}
            for i in range(10)
        ]
        extractor = make_extractor({"memories": memories}, min_confidence=0.5)
        extractor.max_claims = 3
        claims = extractor.extract("text")
        assert len(claims) <= 3


# ---------------------------------------------------------------------------
# Semantic claim — missing fields filtered
# ---------------------------------------------------------------------------

class TestExtractorSemanticValidation:
    def test_semantic_missing_predicate_skipped(self):
        extractor = make_extractor({
            "memories": [
                {
                    "subject": "user",
                    "predicate": "",
                    "object": "pizza",
                    "type": "semantic",
                    "confidence": 0.9,
                }
            ]
        })
        claims = extractor.extract("text")
        assert claims == []

    def test_semantic_missing_object_skipped(self):
        extractor = make_extractor({
            "memories": [
                {
                    "subject": "user",
                    "predicate": "likes",
                    "object": "",
                    "type": "semantic",
                    "confidence": 0.9,
                }
            ]
        })
        claims = extractor.extract("text")
        assert claims == []

    def test_semantic_uses_default_subject(self):
        """If subject is missing, default is 'user'."""
        extractor = make_extractor({
            "memories": [
                {
                    "predicate": "likes",
                    "object": "pizza",
                    "type": "semantic",
                    "confidence": 0.9,
                }
            ]
        })
        claims = extractor.extract("text")
        assert len(claims) == 1
        assert isinstance(claims[0], SemanticClaim)
        assert claims[0].subject == "user"

    def test_semantic_uses_context_user_id(self):
        extractor = make_extractor({
            "memories": [
                {
                    "predicate": "likes",
                    "object": "coffee",
                    "type": "semantic",
                    "confidence": 0.9,
                }
            ]
        })
        claims = extractor.extract("text", context={"user_id": "alice"})
        assert isinstance(claims[0], SemanticClaim)
        assert claims[0].subject == "alice"

    def test_unknown_type_skipped(self):
        extractor = make_extractor({
            "memories": [
                {"content": "something", "type": "procedural", "confidence": 0.9}
            ]
        })
        claims = extractor.extract("text")
        assert claims == []


# ---------------------------------------------------------------------------
# extract_batch
# ---------------------------------------------------------------------------

class TestExtractBatch:
    def test_batch_returns_list_per_text(self):
        extractor = make_extractor({"memories": []})
        results = extractor.extract_batch(["text1", "text2", "text3"])
        assert len(results) == 3

    def test_batch_each_item_is_list(self):
        extractor = make_extractor({"memories": []})
        results = extractor.extract_batch(["text"])
        assert isinstance(results[0], list)


# ---------------------------------------------------------------------------
# Evidence auto-generated
# ---------------------------------------------------------------------------

class TestExtractorEvidence:
    def test_semantic_claim_has_evidence(self):
        extractor = make_extractor({
            "memories": [
                {
                    "subject": "user",
                    "predicate": "likes",
                    "object": "tea",
                    "type": "semantic",
                    "confidence": 0.9,
                }
            ]
        })
        claims = extractor.extract("text")
        assert len(claims[0].evidence) > 0

    def test_episodic_claim_has_evidence(self):
        extractor = make_extractor({
            "memories": [
                {"content": "Had tea", "type": "episodic", "confidence": 0.9}
            ]
        })
        claims = extractor.extract("text")
        assert len(claims[0].evidence) > 0
