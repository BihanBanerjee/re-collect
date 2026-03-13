"""Explanation data structures for deep explainability.

This module provides dataclasses for representing recursive explanations
and justification chains for beliefs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..claims import Claim
    from .edges import Relation


@dataclass
class ExplanationNode:
    """A node in a recursive explanation tree.

    Represents a belief and its relationships in a justification chain.

    Attributes:
        belief: The full Claim object
        depth: How many hops from the root belief (0 = root)
        relation: How this node relates to its parent
            ("supports", "contradicts", "derives", or None for root)
        children: Child ExplanationNodes (beliefs that support/contradict this one)
    """

    belief: Claim
    depth: int
    relation: Relation | None
    children: list[ExplanationNode] = field(default_factory=list)


@dataclass
class ExplanationResult:
    """Result of a deep explanation query.

    Contains the full explanation tree and metadata about the traversal.

    Attributes:
        root: The ExplanationNode for the queried belief
        max_depth_reached: The maximum depth actually traversed
        cycle_detected: Whether cycles were detected (and skipped)
        total_nodes: Total number of nodes in the explanation tree
    """

    root: ExplanationNode
    max_depth_reached: int
    cycle_detected: bool
    total_nodes: int