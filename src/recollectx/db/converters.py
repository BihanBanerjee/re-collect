"""Converters between domain Claim objects and ORM ClaimModel rows."""

import json
from typing import Literal, cast

from recollectx.claims import Claim, EpisodicClaim, SemanticClaim
from recollectx.db.models.belief_edge import BeliefEdgeModel
from recollectx.db.models.claim import ClaimModel
from recollectx.db.models.confidence_history import ConfidenceHistoryModel
from recollectx.graph.edges import BeliefEdge, Relation
from recollectx.propagation import ConfidenceChangeEvent


def claim_to_model(claim: Claim) -> ClaimModel:
    """Convert a domain Claim to an ORM ClaimModel."""
    model = ClaimModel(
        id=claim.id,
        type=claim.type,
        confidence=claim.confidence,
        importance=claim.importance,
        evidence=json.dumps(list(claim.evidence)),
        created_at=claim.created_at,
        last_reinforced_at=claim.last_reinforced_at,
        support_count=claim.support_count,
    )

    if isinstance(claim, EpisodicClaim):
        model.summary = claim.summary
    elif isinstance(claim, SemanticClaim):
        model.subject = claim.subject
        model.predicate = claim.predicate
        model.object = claim.object
    return model


def model_to_claim(model: ClaimModel) -> Claim:
    """Convert an ORM ClaimModel to a domain Claim."""
    claim_id: str = model.id
    confidence: float = model.confidence
    importance: float = model.importance if model.importance is not None else 0.5
    evidence: tuple[str, ...] = tuple(json.loads(model.evidence))
    created_at: float = model.created_at
    last_reinforced_at: float = model.last_reinforced_at
    support_count: int = model.support_count

    if model.type == "episodic":
        return EpisodicClaim(
            id=claim_id,
            confidence=confidence,
            importance=importance,
            evidence=evidence,
            created_at=created_at,
            last_reinforced_at=last_reinforced_at,
            support_count=support_count,
            summary=model.summary or "",
        )
    elif model.type == "semantic":
        return SemanticClaim(
            id=claim_id,
            confidence=confidence,
            importance=importance,
            evidence=evidence,
            created_at=created_at,
            last_reinforced_at=last_reinforced_at,
            support_count=support_count,
            subject=model.subject or "",
            predicate=model.predicate or "",
            object=model.object or "",
        )
    else:
        return Claim(
            id=claim_id,
            confidence=confidence,
            importance=importance,
            evidence=evidence,
            created_at=created_at,
            last_reinforced_at=last_reinforced_at,
            support_count=support_count,
        )


def edge_to_model(edge: BeliefEdge) -> BeliefEdgeModel:
    """Convert a domain BeliefEdge to an ORM BeliefEdgeModel."""
    return BeliefEdgeModel(
        src_id=edge.src_id,
        dst_id=edge.dst_id,
        relation=edge.relation,
    )


def model_to_edge(model: BeliefEdgeModel) -> BeliefEdge:
    """Convert an ORM BeliefEdgeModel to a domain BeliefEdge."""
    return BeliefEdge(
        src_id=model.src_id,
        dst_id=model.dst_id,
        relation=cast(Relation, model.relation),
    )


def confidence_event_to_model(event: ConfidenceChangeEvent) -> ConfidenceHistoryModel:
    """Convert a domain ConfidenceChangeEvent to an ORM model."""
    return ConfidenceHistoryModel(
        claim_id=event.claim_id,
        old_confidence=event.old_confidence,
        new_confidence=event.new_confidence,
        reason=event.reason,
        change_type=event.change_type,
        caused_by_id=event.caused_by_id,
        timestamp=event.timestamp,
    )


def model_to_confidence_event(model: ConfidenceHistoryModel) -> ConfidenceChangeEvent:
    """Convert an ORM model to a domain ConfidenceChangeEvent."""
    return ConfidenceChangeEvent(
        claim_id=model.claim_id,
        old_confidence=model.old_confidence,
        new_confidence=model.new_confidence,
        reason=model.reason,
        change_type=cast(Literal["contradiction", "support", "manual"], model.change_type),
        caused_by_id=model.caused_by_id,
        timestamp=model.timestamp,
    )
