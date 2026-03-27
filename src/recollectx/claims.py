"""Claim dataclasses representing different types of beliefs.

This module provides immutable dataclass representations for two types of claims:
- EpisodicClaim: Events or experiences with a summary
- SemanticClaim: Factual knowledge as subject-predicate-object triples
"""

import time
import uuid
from dataclasses import dataclass, field

VALID_CLAIM_TYPES = frozenset({"", "episodic", "semantic"})


@dataclass(frozen=True)
class Claim:
    """Base class for all claim types.

    Attributes:
        id: Unique identifier for the claim (auto-generated UUID if not provided)
        type: The claim type (episodic, semantic)
        confidence: Confidence score in range [0.0, 1.0] — how sure are we this is true?
        importance: Importance score in range [0.0, 1.0] — how useful is this for
                    future conversations? (0.3 = minor detail, 0.5 = normal, 0.9 = critical)
        evidence: Tuple of evidence strings supporting this claim
        created_at: Unix timestamp when the claim was created
        last_reinforced_at: Unix timestamp when the claim was last reinforced
        support_count: Number of times this claim has been supported
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    confidence: float = 0.0
    importance: float = 0.5
    evidence: tuple[str, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)
    last_reinforced_at: float = field(default_factory=time.time)
    support_count: int = 1

    def __post_init__(self) -> None:
        """Validate claim fields after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance must be in [0.0, 1.0], got {self.importance}")
        if self.type not in VALID_CLAIM_TYPES:
            raise ValueError(f"type must be one of {VALID_CLAIM_TYPES}, got {self.type!r}")
        if self.support_count < 0:
            raise ValueError(f"support_count must be non-negative, got {self.support_count}")
        # Convert list to tuple if passed (for backward compatibility)
        if isinstance(self.evidence, list):
            object.__setattr__(self, 'evidence', tuple(self.evidence))


@dataclass(frozen=True)
class EpisodicClaim(Claim):
    """A claim representing an event or experience.

    Attributes:
        summary: A text summary of the episode
    """
    type: str = "episodic"
    summary: str = ""


@dataclass(frozen=True)
class SemanticClaim(Claim):
    """A claim representing factual knowledge as a subject-predicate-object triple.

    Example: SemanticClaim(subject="sky", predicate="has_color", object="blue")
    represents the fact "the sky has the color blue".

    Attributes:
        subject: The subject of the fact
        predicate: The relationship or property
        object: The object or value
    """
    type: str = "semantic"
    subject: str = ""
    predicate: str = ""
    object: str = ""