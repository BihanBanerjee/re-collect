"""Tests for claim dataclasses (claims.py)."""

import pytest

from recollectx.claims import Claim, EpisodicClaim, SemanticClaim, VALID_CLAIM_TYPES


class TestClaimBase:
    def test_default_id_is_generated(self):
        c = EpisodicClaim(summary="test", confidence=0.8)
        assert c.id
        assert len(c.id) == 36  # UUID4 format

    def test_two_claims_get_different_ids(self):
        a = EpisodicClaim(summary="a", confidence=0.8)
        b = EpisodicClaim(summary="b", confidence=0.8)
        assert a.id != b.id

    def test_custom_id_is_preserved(self):
        c = EpisodicClaim(id="my-id", summary="test", confidence=0.8)
        assert c.id == "my-id"

    def test_confidence_lower_bound(self):
        with pytest.raises(ValueError, match="confidence"):
            EpisodicClaim(summary="x", confidence=-0.01)

    def test_confidence_upper_bound(self):
        with pytest.raises(ValueError, match="confidence"):
            EpisodicClaim(summary="x", confidence=1.01)

    def test_confidence_boundary_values_accepted(self):
        EpisodicClaim(summary="x", confidence=0.0)
        EpisodicClaim(summary="x", confidence=1.0)

    def test_importance_lower_bound(self):
        with pytest.raises(ValueError, match="importance"):
            EpisodicClaim(summary="x", confidence=0.8, importance=-0.01)

    def test_importance_upper_bound(self):
        with pytest.raises(ValueError, match="importance"):
            EpisodicClaim(summary="x", confidence=0.8, importance=1.01)

    def test_importance_boundary_values_accepted(self):
        EpisodicClaim(summary="x", confidence=0.8, importance=0.0)
        EpisodicClaim(summary="x", confidence=0.8, importance=1.0)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="type"):
            Claim(type="unknown", confidence=0.5)

    def test_valid_types_accepted(self):
        for t in ("episodic", "semantic"):
            assert t in VALID_CLAIM_TYPES

    def test_negative_support_count_raises(self):
        with pytest.raises(ValueError, match="support_count"):
            EpisodicClaim(summary="x", confidence=0.8, support_count=-1)

    def test_zero_support_count_accepted(self):
        c = EpisodicClaim(summary="x", confidence=0.8, support_count=0)
        assert c.support_count == 0

    def test_evidence_list_converted_to_tuple(self):
        c = EpisodicClaim(
            summary="x",
            confidence=0.8,
            evidence=["evidence 1", "evidence 2"],  # type: ignore[arg-type]
        )
        assert isinstance(c.evidence, tuple)
        assert c.evidence == ("evidence 1", "evidence 2")

    def test_claim_is_immutable(self):
        c = EpisodicClaim(summary="x", confidence=0.8)
        with pytest.raises(Exception):
            c.confidence = 0.5  # type: ignore[misc]

    def test_timestamps_are_set_automatically(self):
        import time
        before = time.time()
        c = EpisodicClaim(summary="x", confidence=0.8)
        after = time.time()
        assert before <= c.created_at <= after
        assert before <= c.last_reinforced_at <= after

    def test_default_importance_is_half(self):
        c = EpisodicClaim(summary="x", confidence=0.8)
        assert c.importance == 0.5

    def test_default_support_count_is_one(self):
        c = EpisodicClaim(summary="x", confidence=0.8)
        assert c.support_count == 1


class TestEpisodicClaim:
    def test_type_is_episodic(self):
        c = EpisodicClaim(summary="hello", confidence=0.8)
        assert c.type == "episodic"

    def test_summary_stored(self):
        c = EpisodicClaim(summary="Had lunch", confidence=0.9)
        assert c.summary == "Had lunch"

    def test_empty_summary_allowed(self):
        c = EpisodicClaim(summary="", confidence=0.8)
        assert c.summary == ""


class TestSemanticClaim:
    def test_type_is_semantic(self):
        c = SemanticClaim(subject="user", predicate="likes", object="pizza", confidence=0.9)
        assert c.type == "semantic"

    def test_spo_fields_stored(self):
        c = SemanticClaim(subject="sky", predicate="has_color", object="blue", confidence=0.95)
        assert c.subject == "sky"
        assert c.predicate == "has_color"
        assert c.object == "blue"

    def test_empty_spo_allowed(self):
        c = SemanticClaim(confidence=0.5)
        assert c.subject == ""
        assert c.predicate == ""
        assert c.object == ""

    def test_custom_fields(self):
        c = SemanticClaim(
            subject="user",
            predicate="works_as",
            object="engineer",
            confidence=0.9,
            importance=0.8,
            support_count=3,
            evidence=("user works_as engineer",),
        )
        assert c.importance == 0.8
        assert c.support_count == 3
        assert c.evidence == ("user works_as engineer",)
