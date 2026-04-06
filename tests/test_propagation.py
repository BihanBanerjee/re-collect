"""Tests for confidence propagation (propagation.py)."""

import pytest

from recollectx.claims import EpisodicClaim, SemanticClaim
from recollectx.propagation import ConfidencePropagator, PropagationConfig, ConfidenceChangeEvent


# ---------------------------------------------------------------------------
# PropagationConfig validation
# ---------------------------------------------------------------------------

class TestPropagationConfig:
    def test_defaults_are_valid(self):
        cfg = PropagationConfig()
        assert cfg.contradiction_decay == 0.15
        assert cfg.support_boost == 0.10
        assert cfg.min_confidence == 0.01
        assert cfg.max_confidence == 0.99
        assert cfg.symmetric_contradiction is True

    def test_invalid_contradiction_decay(self):
        with pytest.raises(ValueError, match="contradiction_decay"):
            PropagationConfig(contradiction_decay=1.5)

    def test_invalid_support_boost(self):
        with pytest.raises(ValueError, match="support_boost"):
            PropagationConfig(support_boost=-0.1)

    def test_min_must_be_less_than_max(self):
        with pytest.raises(ValueError):
            PropagationConfig(min_confidence=0.9, max_confidence=0.5)

    def test_min_equal_to_max_invalid(self):
        with pytest.raises(ValueError):
            PropagationConfig(min_confidence=0.5, max_confidence=0.5)


# ---------------------------------------------------------------------------
# ConfidencePropagator — support
# ---------------------------------------------------------------------------

class TestPropagatorSupport:
    def setup_method(self):
        """Create a minimal mock storage for propagation tests."""
        self._updates: list = []

    def _make_store(self):
        class MockStore:
            def __init__(self, tracker):
                self._tracker = tracker

            def update(self, claim):
                self._tracker.append(claim)
        return MockStore(self._updates)

    def test_support_increases_confidence(self):
        store = self._make_store()
        config = PropagationConfig(support_boost=0.10)
        propagator = ConfidencePropagator(store, config)  # type: ignore[arg-type]

        supporter = EpisodicClaim(summary="a", confidence=0.8)
        supported = EpisodicClaim(summary="b", confidence=0.7)

        updated = propagator.on_support(supporter, supported)
        assert updated.confidence == pytest.approx(0.8, abs=0.001)

    def test_support_clamped_at_max(self):
        store = self._make_store()
        config = PropagationConfig(support_boost=0.10, max_confidence=0.99)
        propagator = ConfidencePropagator(store, config)  # type: ignore[arg-type]

        supporter = EpisodicClaim(summary="a", confidence=0.9)
        supported = EpisodicClaim(summary="b", confidence=0.95)

        updated = propagator.on_support(supporter, supported)
        assert updated.confidence <= 0.99

    def test_support_emits_event(self):
        store = self._make_store()
        events: list[ConfidenceChangeEvent] = []
        propagator = ConfidencePropagator(
            store,  # type: ignore[arg-type]
            on_event=events.append,
        )

        supporter = EpisodicClaim(summary="a", confidence=0.8)
        supported = EpisodicClaim(summary="b", confidence=0.7)
        propagator.on_support(supporter, supported)

        assert len(events) == 1
        assert events[0].change_type == "support"
        assert events[0].claim_id == supported.id
        assert events[0].caused_by_id == supporter.id

    def test_support_updates_storage(self):
        store = self._make_store()
        propagator = ConfidencePropagator(store)  # type: ignore[arg-type]

        supporter = EpisodicClaim(summary="a", confidence=0.8)
        supported = EpisodicClaim(summary="b", confidence=0.7)
        propagator.on_support(supporter, supported)

        assert len(self._updates) == 1


# ---------------------------------------------------------------------------
# ConfidencePropagator — contradiction
# ---------------------------------------------------------------------------

class TestPropagatorContradiction:
    def setup_method(self):
        self._updates: list = []

    def _make_store(self):
        class MockStore:
            def __init__(self, tracker):
                self._tracker = tracker

            def update(self, claim):
                self._tracker.append(claim)
        return MockStore(self._updates)

    def test_contradiction_lowers_confidence(self):
        store = self._make_store()
        config = PropagationConfig(contradiction_decay=0.15)
        propagator = ConfidencePropagator(store, config)  # type: ignore[arg-type]

        claim_a = SemanticClaim(
            subject="u", predicate="likes", object="tea", confidence=0.8
        )
        claim_b = SemanticClaim(
            subject="u", predicate="likes", object="coffee", confidence=0.8
        )

        updated_a, updated_b = propagator.on_contradiction(claim_a, claim_b)
        assert updated_b.confidence == pytest.approx(0.65, abs=0.001)

    def test_symmetric_contradiction_lowers_both(self):
        store = self._make_store()
        config = PropagationConfig(
            contradiction_decay=0.15, symmetric_contradiction=True
        )
        propagator = ConfidencePropagator(store, config)  # type: ignore[arg-type]

        claim_a = EpisodicClaim(summary="a", confidence=0.8)
        claim_b = EpisodicClaim(summary="b", confidence=0.9)
        updated_a, updated_b = propagator.on_contradiction(claim_a, claim_b)

        assert updated_a.confidence < 0.8
        assert updated_b.confidence < 0.9

    def test_asymmetric_contradiction_only_lowers_b(self):
        store = self._make_store()
        config = PropagationConfig(
            contradiction_decay=0.15, symmetric_contradiction=False
        )
        propagator = ConfidencePropagator(store, config)  # type: ignore[arg-type]

        claim_a = EpisodicClaim(summary="a", confidence=0.8)
        claim_b = EpisodicClaim(summary="b", confidence=0.9)
        updated_a, updated_b = propagator.on_contradiction(claim_a, claim_b)

        assert updated_a.confidence == claim_a.confidence  # unchanged
        assert updated_b.confidence < 0.9

    def test_contradiction_clamped_at_min(self):
        store = self._make_store()
        config = PropagationConfig(
            contradiction_decay=0.5, min_confidence=0.01
        )
        propagator = ConfidencePropagator(store, config)  # type: ignore[arg-type]

        claim_a = EpisodicClaim(summary="a", confidence=0.1)
        claim_b = EpisodicClaim(summary="b", confidence=0.1)
        updated_a, updated_b = propagator.on_contradiction(claim_a, claim_b)

        assert updated_a.confidence >= 0.01
        assert updated_b.confidence >= 0.01

    def test_contradiction_emits_two_events_when_symmetric(self):
        store = self._make_store()
        events: list[ConfidenceChangeEvent] = []
        config = PropagationConfig(symmetric_contradiction=True)
        propagator = ConfidencePropagator(
            store,  # type: ignore[arg-type]
            config,
            on_event=events.append,
        )

        claim_a = EpisodicClaim(summary="a", confidence=0.8)
        claim_b = EpisodicClaim(summary="b", confidence=0.8)
        propagator.on_contradiction(claim_a, claim_b)

        assert len(events) == 2
        change_types = {e.change_type for e in events}
        assert change_types == {"contradiction"}
