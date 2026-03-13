"""Belief graph module for tracking relationships between beliefs.

This module provides:
- BeliefGraph: Graph for tracking belief relationships
- BeliefEdge: Edge representing a relationship between beliefs
- Relation: Type alias for valid relationship types
- ExplanationNode: Node in a recursive explanation tree
- ExplanationResult: Result of a deep explanation query
"""

from .edges import BeliefEdge, Relation
from .explanation import ExplanationNode, ExplanationResult
from .graph import BeliefGraph

__all__ = [
    "BeliefGraph",
    "BeliefEdge",
    "Relation",
    "ExplanationNode",
    "ExplanationResult",
]
