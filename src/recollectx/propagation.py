"""Confidence propagation for belief relationships.

Key behaviors:
- Contradictions reduce confidence of conflicting beliefs
- Supports increase confidence of the supported belief
- Confidence is clamped to configurable min/max bounds
- Confidence changes are recorded as ConfidenceChangeEvent for history tracking
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .claims import Claim
    from .storage.memory_store import MemoryStore

# Type alias for event callback
EventCallback = Callable[["ConfidenceChangeEvent"], None]


@dataclass(frozen=True)
class ConfidenceChangeEvent:
    """Record of a confidence change for a claim."""

    claim_id: str
    old_confidence: float
    new_confidence: float
    reason: str
    change_type: Literal["contradiction", "support", "manual"]
    timestamp: float = field(default_factory=time.time)
    caused_by_id: str | None = None


@dataclass(frozen=True)
class PropagationConfig:
    """Configuration for confidence propagation."""

    contradiction_decay: float = 0.15
    support_boost: float = 0.10
    min_confidence: float = 0.01
    max_confidence: float = 0.99
    symmetric_contradiction: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.contradiction_decay <= 1.0:
            raise ValueError(
                f"contradiction_decay must be in [0.0, 1.0], got {self.contradiction_decay}"
            )
        if not 0.0 <= self.support_boost <= 1.0:
            raise ValueError(
                f"support_boost must be in [0.0, 1.0], got {self.support_boost}"
            )
        if not 0.0 <= self.min_confidence < self.max_confidence <= 1.0:
            raise ValueError(
                f"min_confidence ({self.min_confidence}) must be < max_confidence "
                f"({self.max_confidence}), both in [0.0, 1.0]"
            )


class ConfidencePropagator:
    """Propagates confidence changes when belief relationships change."""

    def __init__(
        self,
        storage: "MemoryStore",
        config: PropagationConfig | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.storage = storage
        self.config = config or PropagationConfig()
        self._on_event = on_event

    def _emit_event(self, event: "ConfidenceChangeEvent") -> None:
        if self._on_event is not None:
            self._on_event(event)

    def _clamp_confidence(self, confidence: float) -> float:
        return max(
            self.config.min_confidence, min(self.config.max_confidence, confidence)
        )

    def on_contradiction(
        self,
        claim_a: "Claim",
        claim_b: "Claim",
    ) -> tuple["Claim", "Claim"]:
        """Handle a contradiction between two claims."""
        now = time.time()

        new_conf_b = self._clamp_confidence(
            claim_b.confidence - self.config.contradiction_decay
        )
        updated_b = replace(claim_b, confidence=new_conf_b)
        self.storage.update(updated_b)

        self._emit_event(
            ConfidenceChangeEvent(
                claim_id=claim_b.id,
                old_confidence=claim_b.confidence,
                new_confidence=new_conf_b,
                reason=f"Contradicted by claim {claim_a.id}",
                change_type="contradiction",
                timestamp=now,
                caused_by_id=claim_a.id,
            )
        )

        if self.config.symmetric_contradiction:
            new_conf_a = self._clamp_confidence(
                claim_a.confidence - self.config.contradiction_decay
            )
            updated_a = replace(claim_a, confidence=new_conf_a)
            self.storage.update(updated_a)

            self._emit_event(
                ConfidenceChangeEvent(
                    claim_id=claim_a.id,
                    old_confidence=claim_a.confidence,
                    new_confidence=new_conf_a,
                    reason=f"Contradicted by claim {claim_b.id}",
                    change_type="contradiction",
                    timestamp=now,
                    caused_by_id=claim_b.id,
                )
            )

            return (updated_a, updated_b)

        return (claim_a, updated_b)

    def on_support(
        self,
        supporting_claim: "Claim",
        supported_claim: "Claim",
    ) -> "Claim":
        """Handle a support relationship."""
        new_conf = self._clamp_confidence(
            supported_claim.confidence + self.config.support_boost
        )
        updated = replace(supported_claim, confidence=new_conf)
        self.storage.update(updated)

        self._emit_event(
            ConfidenceChangeEvent(
                claim_id=supported_claim.id,
                old_confidence=supported_claim.confidence,
                new_confidence=new_conf,
                reason=f"Supported by claim {supporting_claim.id}",
                change_type="support",
                caused_by_id=supporting_claim.id,
            )
        )

        return updated
