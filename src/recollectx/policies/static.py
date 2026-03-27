"""Static policy implementations.

- MinEvidence: Requires a minimum number of evidence items
- MinConfidence: Requires minimum confidence thresholds per claim type
"""

from typing import TYPE_CHECKING

from .base import Decision, Policy

if TYPE_CHECKING:
    from ..claims import Claim
    from ..memory import Memory


class MinEvidence(Policy):
    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, claim: "Claim", memory: "Memory") -> Decision:
        return Decision.ACCEPT if len(claim.evidence) >= self.n else Decision.REJECT


class MinConfidence(Policy):
    def __init__(
        self,
        episodic: float = 0.3,
        semantic: float = 0.6,
    ) -> None:
        self.thresholds: dict[str, float] = {
            "episodic": episodic,
            "semantic": semantic,
        }

    def __call__(self, claim: "Claim", memory: "Memory") -> Decision:
        threshold = self.thresholds.get(claim.type)
        if threshold is None:
            return Decision.REJECT
        return Decision.ACCEPT if claim.confidence >= threshold else Decision.REJECT
