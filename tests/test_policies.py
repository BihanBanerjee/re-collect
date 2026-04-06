"""Tests for write policies (policies/)."""

import pytest

from recollectx.claims import EpisodicClaim, SemanticClaim
from recollectx.policies.base import Decision, Policy, AndPolicy
from recollectx.policies.static import MinEvidence, MinConfidence


# ---------------------------------------------------------------------------
# MinEvidence
# ---------------------------------------------------------------------------

class TestMinEvidence:
    def test_accepts_enough_evidence(self):
        policy = MinEvidence(2)
        claim = EpisodicClaim(
            summary="x", confidence=0.8, evidence=("e1", "e2")
        )
        assert policy(claim, None) == Decision.ACCEPT  # type: ignore[arg-type]

    def test_accepts_exactly_n_evidence(self):
        policy = MinEvidence(1)
        claim = EpisodicClaim(summary="x", confidence=0.8, evidence=("e1",))
        assert policy(claim, None) == Decision.ACCEPT  # type: ignore[arg-type]

    def test_rejects_insufficient_evidence(self):
        policy = MinEvidence(3)
        claim = EpisodicClaim(summary="x", confidence=0.8, evidence=("e1",))
        assert policy(claim, None) == Decision.REJECT  # type: ignore[arg-type]

    def test_rejects_empty_evidence(self):
        policy = MinEvidence(1)
        claim = EpisodicClaim(summary="x", confidence=0.8, evidence=())
        assert policy(claim, None) == Decision.REJECT  # type: ignore[arg-type]

    def test_zero_n_always_accepts(self):
        policy = MinEvidence(0)
        claim = EpisodicClaim(summary="x", confidence=0.8, evidence=())
        assert policy(claim, None) == Decision.ACCEPT  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MinConfidence
# ---------------------------------------------------------------------------

class TestMinConfidence:
    def test_accepts_episodic_above_threshold(self):
        policy = MinConfidence(episodic=0.3, semantic=0.6)
        claim = EpisodicClaim(summary="x", confidence=0.5)
        assert policy(claim, None) == Decision.ACCEPT  # type: ignore[arg-type]

    def test_rejects_episodic_below_threshold(self):
        policy = MinConfidence(episodic=0.5, semantic=0.6)
        claim = EpisodicClaim(summary="x", confidence=0.3)
        assert policy(claim, None) == Decision.REJECT  # type: ignore[arg-type]

    def test_accepts_semantic_above_threshold(self):
        policy = MinConfidence(episodic=0.3, semantic=0.6)
        claim = SemanticClaim(
            subject="user", predicate="likes", object="tea", confidence=0.8
        )
        assert policy(claim, None) == Decision.ACCEPT  # type: ignore[arg-type]

    def test_rejects_semantic_below_threshold(self):
        policy = MinConfidence(episodic=0.3, semantic=0.7)
        claim = SemanticClaim(
            subject="user", predicate="likes", object="tea", confidence=0.5
        )
        assert policy(claim, None) == Decision.REJECT  # type: ignore[arg-type]

    def test_accepts_at_exact_threshold(self):
        policy = MinConfidence(episodic=0.5)
        claim = EpisodicClaim(summary="x", confidence=0.5)
        assert policy(claim, None) == Decision.ACCEPT  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AndPolicy (& composition)
# ---------------------------------------------------------------------------

class TestAndPolicy:
    def test_both_accept(self):
        policy = MinEvidence(1) & MinConfidence(episodic=0.3)
        claim = EpisodicClaim(
            summary="x", confidence=0.8, evidence=("e1",)
        )
        assert policy(claim, None) == Decision.ACCEPT  # type: ignore[arg-type]

    def test_first_rejects(self):
        policy = MinEvidence(5) & MinConfidence(episodic=0.3)
        claim = EpisodicClaim(
            summary="x", confidence=0.8, evidence=("e1",)
        )
        assert policy(claim, None) == Decision.REJECT  # type: ignore[arg-type]

    def test_second_rejects(self):
        policy = MinEvidence(1) & MinConfidence(episodic=0.9)
        claim = EpisodicClaim(
            summary="x", confidence=0.5, evidence=("e1",)
        )
        assert policy(claim, None) == Decision.REJECT  # type: ignore[arg-type]

    def test_both_reject(self):
        policy = MinEvidence(5) & MinConfidence(episodic=0.9)
        claim = EpisodicClaim(
            summary="x", confidence=0.3, evidence=()
        )
        assert policy(claim, None) == Decision.REJECT  # type: ignore[arg-type]

    def test_chaining_three_policies(self):
        policy = MinEvidence(1) & MinConfidence(episodic=0.3) & MinEvidence(1)
        claim = EpisodicClaim(
            summary="x", confidence=0.8, evidence=("e1",)
        )
        assert policy(claim, None) == Decision.ACCEPT  # type: ignore[arg-type]
