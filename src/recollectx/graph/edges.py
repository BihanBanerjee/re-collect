"""Belief graph edge definitions.

This module defines the BeliefEdge dataclass and the Relation type
for representing relationships between beliefs.
"""

from dataclasses import dataclass
from typing import Literal

Relation = Literal["supports", "contradicts", "derives", "similar"]


@dataclass(frozen=True)
class BeliefEdge:
    """An edge in the belief graph representing a relationship between beliefs.

    Attributes:
        src_id: The ID of the source belief
        dst_id: The ID of the destination belief
        relation: The type of relationship (supports, contradicts, derives, similar)
    """
    src_id: str
    dst_id: str
    relation: Relation
