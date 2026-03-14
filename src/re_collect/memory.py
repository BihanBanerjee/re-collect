"""Central memory management system for belief storage and retrieval.

This module provides the Memory class, which is the main interface for
storing, retrieving, and explaining beliefs in the recollect system.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from .graph.edges import BeliefEdge
from .graph.explanation import ExplanationNode, ExplanationResult
from .graph.graph import BeliefGraph
from .policies.base import Decision

if TYPE_CHECKING:
    from .claims import Claim
    from .policies.base import Policy
    from .propagation import ConfidenceChangeEvent, ConfidencePropagator, PropagationConfig
    from .storage.memory_store import MemoryStore
    from .updater import MemoryUpdater

logger = logging.getLogger(__name__)


class Memory:
    """Central memory management system for belief storage and retrieval.

    The Memory class provides an interface for storing, retrieving,
    and explaining beliefs. It supports:
    - Claim storage with write policies
    - Automatic contradiction detection for semantic claims
    - Belief graph tracking for supports/contradicts relationships

    Example:
        from recollect.db import SessionLocal, create_tables
        from recollect.storage import MemoryStore
        from recollect.storage.vector import FAISSBackend

        create_tables()
        db = SessionLocal()
        vectors = FAISSBackend(embed_fn=my_embed, dimension=384)
        store = MemoryStore(db, vectors)

        memory = Memory(storage=store)
        claim = SemanticClaim(subject="sky", predicate="has_color",
                              object="blue", confidence=0.9)
        memory.store(claim)
        results = memory.retrieve(type="semantic")
    """

    def __init__(
        self,
        *,
        storage: "MemoryStore",
        write_policy: Optional["Policy"] = None,
        updater: Optional["MemoryUpdater"] = None,
        propagation_config: Optional["PropagationConfig"] = None,
    ) -> None:
        self.storage = storage
        self.policy = write_policy
        self._updater = updater
        self.graph = BeliefGraph()

        self._propagator: "ConfidencePropagator | None" = None  # noqa: UP037
        if propagation_config is not None:
            from .propagation import ConfidencePropagator

            self._propagator = ConfidencePropagator(
                storage,
                propagation_config,
                on_event=self._on_confidence_event,
            )

        # Hydrate graph from storage
        self._hydrate_graph()

    def _on_confidence_event(self, event: Any) -> None:
        """Callback for storing confidence change events."""
        try:
            self.storage.put_confidence_event(event)
        except Exception as e:
            logger.warning(f"Failed to store confidence event: {e}")

    def _hydrate_graph(self) -> None:
        """Load persisted edges into the belief graph."""
        try:
            edges = self.storage.get_all_edges()
            for edge in edges:
                self.graph.add(edge)
            if edges:
                logger.debug(f"Hydrated belief graph with {len(edges)} edges")
        except Exception as e:
            logger.warning(f"Failed to hydrate belief graph: {e}")

    def store(self, claim: "Claim") -> None:
        """Store a claim.

        Applies write policy (if configured), then either sends through
        the LLM updater or stores directly.

        Args:
            claim: The claim to store
        """
        # Apply write policy if configured
        if self.policy:
            try:
                decision = self.policy(claim, self)
                if decision == Decision.REJECT:
                    logger.debug(f"Claim {claim.id} rejected by policy")
                    return
            except Exception as e:
                logger.error(f"Policy evaluation failed for claim {claim.id}: {e}")
                return

        # Store the claim — either through LLM updater or directly
        try:
            if self._updater:
                result = self._updater.process(claim)
                logger.debug(
                    f"Claim {claim.id}: updater decided {result.action} — {result.reason}"
                )
                if result.action == "NONE":
                    return

                for edge in result.edges:
                    self.graph.add(edge)
                    if self._propagator:
                        self._apply_propagation(edge)
            else:
                self.storage.put(claim)
                logger.debug(f"Claim {claim.id} stored successfully")
        except Exception as e:
            logger.error(f"Failed to store claim {claim.id}: {e}")

    def _apply_propagation(self, edge: BeliefEdge) -> None:
        """Apply confidence propagation for a relationship edge."""
        if not self._propagator:
            return

        try:
            src = self.storage.get(edge.src_id)
            dst = self.storage.get(edge.dst_id)
            if src is None or dst is None:
                return

            if edge.relation == "supports":
                self._propagator.on_support(src, dst)
            elif edge.relation == "contradicts":
                self._propagator.on_contradiction(src, dst)
        except Exception as e:
            logger.warning(
                f"Propagation failed for {edge.relation} "
                f"{edge.src_id} -> {edge.dst_id}: {e}"
            )

    def retrieve(self, **kwargs: Any) -> list["Claim"]:
        """Retrieve claims matching the given criteria.

        Args:
            **kwargs: Query parameters passed to the storage backend
                (e.g., type="semantic", min_confidence=0.5)

        Returns:
            List of claims matching the criteria
        """
        return self.storage.query(**kwargs)

    def explain(self, belief_id: str) -> dict[str, Any] | None:
        """Get an explanation for a belief including its relationships.

        Returns:
            A dictionary containing:
            - belief: The belief object
            - supported_by: List of belief IDs that support this belief
            - contradicted_by: List of belief IDs that contradict this belief
            Returns None if the belief is not found.
        """
        try:
            belief = self.storage.get(belief_id)
        except Exception as e:
            logger.error(f"Failed to retrieve belief {belief_id}: {e}")
            return None

        if not belief:
            return None

        supports = self.graph.supports(belief_id)
        contradictions = self.graph.contradictions(belief_id)

        return {
            "belief": belief,
            "supported_by": supports,
            "contradicted_by": contradictions,
        }

    def add_support(self, src_id: str, dst_id: str) -> None:
        """Add a support relationship between two beliefs."""
        src_claim = self.storage.get(src_id)
        dst_claim = self.storage.get(dst_id)

        if src_claim is None:
            raise KeyError(f"Supporting belief {src_id} not found")
        if dst_claim is None:
            raise KeyError(f"Supported belief {dst_id} not found")

        edge = BeliefEdge(src_id, dst_id, "supports")
        self.graph.add(edge)
        try:
            self.storage.put_edge(edge)
        except Exception as e:
            logger.warning(f"Failed to persist support edge: {e}")
        logger.debug(f"Support relationship added: {src_id} -> {dst_id}")

        if self._propagator:
            try:
                self._propagator.on_support(src_claim, dst_claim)
            except Exception as e:
                logger.warning(
                    f"Propagation failed for support {src_id} -> {dst_id}: {e}"
                )

    def explain_deep(
        self,
        belief_id: str,
        max_depth: int = 3,
    ) -> ExplanationResult | None:
        """Get a deep recursive explanation for a belief."""
        root_belief = self.storage.get(belief_id)
        if root_belief is None:
            return None

        nodes, cycle_detected = self.graph.traverse_recursive(belief_id, max_depth)

        node_map: dict[str, ExplanationNode] = {}

        for tnode in nodes:
            belief = self.storage.get(tnode.belief_id)
            if belief is None:
                continue

            exp_node = ExplanationNode(
                belief=belief,
                depth=tnode.depth,
                relation=tnode.relation,
                children=[],
            )
            node_map[tnode.belief_id] = exp_node

            if tnode.parent_id is not None and tnode.parent_id in node_map:
                node_map[tnode.parent_id].children.append(exp_node)

        root_node = node_map.get(belief_id)
        if root_node is None:
            return None

        max_depth_reached = max(n.depth for n in nodes) if nodes else 0

        return ExplanationResult(
            root=root_node,
            max_depth_reached=max_depth_reached,
            cycle_detected=cycle_detected,
            total_nodes=len(node_map),
        )

    def explain_confidence_history(
        self,
        belief_id: str,
    ) -> list["ConfidenceChangeEvent"]:
        """Get the confidence change history for a belief."""
        try:
            return self.storage.get_confidence_history(belief_id)
        except Exception as e:
            logger.warning(f"Failed to get confidence history for {belief_id}: {e}")
            return []
