"""Belief graph for tracking relationships between beliefs.

This module provides the BeliefGraph class for tracking support,
contradiction, and derivation relationships between beliefs.
"""

from collections import defaultdict
from dataclasses import dataclass

from .edges import BeliefEdge, Relation


@dataclass
class TraversalNode:
    """A node encountered during graph traversal.

    Attributes:
        belief_id: ID of the belief at this node
        depth: Distance from the root belief (0 = root)
        relation: How this belief relates to its parent (None for root)
        parent_id: ID of the parent belief (None for root)
    """

    belief_id: str
    depth: int
    relation: Relation | None
    parent_id: str | None


class BeliefGraph:
    """A directed graph tracking relationships between beliefs.

    The graph stores edges representing how beliefs relate to each other:
    - supports: One belief provides evidence for another
    - contradicts: One belief conflicts with another
    - derives: One belief is derived from another
    - similar: Two beliefs are semantically similar (auto-discovered)

    Attributes:
        outgoing: Edges indexed by source belief ID
        incoming: Edges indexed by destination belief ID
    """

    def __init__(self) -> None:
        """Initialize an empty belief graph."""
        self.outgoing: defaultdict[str, list[BeliefEdge]] = defaultdict(list)
        self.incoming: defaultdict[str, list[BeliefEdge]] = defaultdict(list)

    def add(self, edge: BeliefEdge) -> None:
        """Add an edge to the graph.

        Args:
            edge: The edge to add
        """
        self.outgoing[edge.src_id].append(edge)
        self.incoming[edge.dst_id].append(edge)

    def contradictions(self, belief_id: str) -> list[str]:
        """Get IDs of beliefs that contradict the given belief.

        Args:
            belief_id: The ID of the belief to query

        Returns:
            List of belief IDs that have a 'contradicts' edge to this belief
        """
        return [
            e.src_id for e in self.incoming[belief_id]
            if e.relation == "contradicts"
        ]

    def supports(self, belief_id: str) -> list[str]:
        """Get IDs of beliefs that support the given belief.

        Args:
            belief_id: The ID of the belief to query

        Returns:
            List of belief IDs that have a 'supports' edge to this belief
        """
        return [
            e.src_id for e in self.incoming[belief_id]
            if e.relation == "supports"
        ]

    def derives(self, belief_id: str) -> list[str]:
        """Get IDs of beliefs that this belief derives from.

        Args:
            belief_id: The ID of the belief to query

        Returns:
            List of belief IDs that have a 'derives' edge to this belief
        """
        return [
            e.src_id for e in self.incoming[belief_id]
            if e.relation == "derives"
        ]

    def similar(self, belief_id: str) -> list[str]:
        """Get IDs of beliefs that are similar to the given belief.

        Checks both incoming and outgoing edges since similarity is bidirectional.

        Args:
            belief_id: The ID of the belief to query

        Returns:
            List of belief IDs connected by 'similar' edges
        """
        ids: list[str] = []
        for e in self.outgoing[belief_id]:
            if e.relation == "similar":
                ids.append(e.dst_id)
        for e in self.incoming[belief_id]:
            if e.relation == "similar":
                ids.append(e.src_id)
        return ids

    def outgoing_edges(self, belief_id: str) -> list[BeliefEdge]:
        """Get all outgoing edges from a belief.

        Args:
            belief_id: The ID of the belief to query

        Returns:
            List of edges originating from this belief
        """
        return list(self.outgoing[belief_id])

    def incoming_edges(self, belief_id: str) -> list[BeliefEdge]:
        """Get all incoming edges to a belief.

        Args:
            belief_id: The ID of the belief to query

        Returns:
            List of edges pointing to this belief
        """
        return list(self.incoming[belief_id])

    def traverse_recursive(
        self,
        belief_id: str,
        max_depth: int = 3,
    ) -> tuple[list[TraversalNode], bool]:
        """Traverse the graph recursively from a starting belief.

        Performs a breadth-first traversal of incoming edges (beliefs that
        support, contradict, or derive this belief) up to max_depth.

        Args:
            belief_id: The starting belief ID
            max_depth: Maximum depth to traverse (default 3)

        Returns:
            Tuple of (list of TraversalNodes, cycle_detected bool)
        """
        result: list[TraversalNode] = []
        visited: set[str] = set()
        cycle_detected = False

        # Start with root node
        root = TraversalNode(
            belief_id=belief_id,
            depth=0,
            relation=None,
            parent_id=None,
        )
        result.append(root)
        visited.add(belief_id)

        # BFS queue: (belief_id, current_depth)
        queue: list[tuple[str, int]] = [(belief_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_depth >= max_depth:
                continue

            # Get all incoming edges (beliefs that relate to this one)
            for edge in self.incoming[current_id]:
                if edge.src_id in visited:
                    cycle_detected = True
                    continue

                visited.add(edge.src_id)
                node = TraversalNode(
                    belief_id=edge.src_id,
                    depth=current_depth + 1,
                    relation=edge.relation,
                    parent_id=current_id,
                )
                result.append(node)
                queue.append((edge.src_id, current_depth + 1))

        return result, cycle_detected

    def all_edges(self) -> list[BeliefEdge]:
        """Get all edges in the graph.

        Returns:
            List of all edges
        """
        edges: list[BeliefEdge] = []
        for edge_list in self.outgoing.values():
            edges.extend(edge_list)
        return edges
