"""Base policy classes for filtering claims."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..claims import Claim
    from ..memory import Memory


class Decision(Enum):
    ACCEPT = 1
    REJECT = 0


class Policy:
    """Base class for write policies.

    Policies determine whether a claim should be stored or rejected.
    Compose with the `&` operator.

    Example:
        policy = MinEvidence(2) & MinConfidence()
    """

    def __call__(self, claim: "Claim", memory: "Memory") -> Decision:
        raise NotImplementedError

    def __and__(self, other: "Policy") -> "AndPolicy":
        return AndPolicy(self, other)


class AndPolicy(Policy):
    """A policy that requires both sub-policies to accept."""

    def __init__(self, a: Policy, b: Policy) -> None:
        self.a = a
        self.b = b

    def __call__(self, claim: "Claim", memory: "Memory") -> Decision:
        if self.a(claim, memory) == Decision.REJECT:
            return Decision.REJECT
        return self.b(claim, memory)
