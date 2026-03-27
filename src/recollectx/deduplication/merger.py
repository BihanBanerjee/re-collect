"""LLM-based claim merging for deduplication.

Uses the LLM to decide what to do with duplicate claims:
ADD, UPDATE, DELETE, or NONE — following the Memora pattern.
"""

from dataclasses import dataclass

from recollectx.claims import (
    Claim,
    EpisodicClaim,
    SemanticClaim,
)
from recollectx.deduplication.similarity import claim_to_text
from recollectx.llm.base import LLMProvider
from recollectx.llm.prompts import UPDATE_SCHEMA


@dataclass
class MergeDecision:
    """Result of an LLM merge decision for a single claim."""

    action: str  # ADD, UPDATE, DELETE, NONE
    target_id: str | None  # existing claim ID if UPDATE/DELETE
    merged_content: str | None  # new content if UPDATE
    reason: str


class LLMMerger:
    """Uses an LLM to decide how to handle duplicate claims.

    Given a new claim and its similar existing claims, asks the LLM
    whether to ADD, UPDATE, DELETE, or skip (NONE).
    """

    def __init__(self, llm_provider: LLMProvider, temperature: float = 0.2):
        self.llm = llm_provider
        self.temperature = temperature

    def decide(
        self, new_claim: Claim, existing_claims: list[Claim]
    ) -> MergeDecision:
        """Ask the LLM what to do with the new claim given existing ones.

        Args:
            new_claim: The incoming claim
            existing_claims: Similar existing claims found by similarity search

        Returns:
            MergeDecision with the LLM's recommended action
        """
        existing_text = "\n".join(
            f'- id: "{c.id}", content: "{claim_to_text(c)}"'
            for c in existing_claims
        )
        new_text = claim_to_text(new_claim)

        prompt = (
            "Compare this new memory with existing ones and decide: ADD, UPDATE, DELETE, or NONE.\n\n"
            f"Existing memories:\n{existing_text}\n\n"
            f"New memory: {new_text}\n\n"
            "Return JSON with a 'decisions' array containing one object with: "
            "new_memory, action (ADD/UPDATE/DELETE/NONE), target_id (existing ID or null), "
            "merged_content (merged text if UPDATE, null otherwise), reason."
        )

        try:
            response = self.llm.generate_structured(
                prompt=prompt,
                schema=UPDATE_SCHEMA,
                temperature=self.temperature,
            )

            decisions = response.get("decisions", [])
            if not decisions:
                return MergeDecision(
                    action="ADD", target_id=None, merged_content=None, reason="No decision returned"
                )

            d = decisions[0]
            return MergeDecision(
                action=d.get("action", "ADD"),
                target_id=d.get("target_id"),
                merged_content=d.get("merged_content"),
                reason=d.get("reason", ""),
            )
        except Exception:
            # If LLM fails, default to ADD
            return MergeDecision(
                action="ADD", target_id=None, merged_content=None, reason="LLM unavailable, defaulting to ADD"
            )

    def apply_decision(
        self, decision: MergeDecision, new_claim: Claim, existing_claims: list[Claim]
    ) -> Claim | None:
        """Apply a merge decision to produce the resulting claim.

        Args:
            decision: The LLM's decision
            new_claim: The incoming claim
            existing_claims: The existing similar claims

        Returns:
            The claim to store, or None if NONE/DELETE with no replacement
        """
        if decision.action == "NONE":
            return None

        if decision.action == "ADD":
            return new_claim

        if decision.action == "DELETE" and decision.target_id:
            # Caller should delete the target; return the new claim as replacement
            return new_claim

        if decision.action == "UPDATE" and decision.merged_content:
            # Build an updated claim with the LLM's merged content
            return _rebuild_claim(new_claim, decision.merged_content)

        return new_claim


def _rebuild_claim(original: Claim, merged_content: str) -> Claim:
    """Rebuild a claim with updated content from the LLM."""
    if isinstance(original, SemanticClaim):
        return SemanticClaim(
            id=original.id,
            subject=original.subject,
            predicate=original.predicate,
            object=merged_content,
            confidence=original.confidence,
            importance=original.importance,
            evidence=original.evidence,
            support_count=original.support_count + 1,
            created_at=original.created_at,
            last_reinforced_at=original.last_reinforced_at,
        )
    elif isinstance(original, EpisodicClaim):
        return EpisodicClaim(
            id=original.id,
            summary=merged_content,
            confidence=original.confidence,
            importance=original.importance,
            evidence=original.evidence,
            support_count=original.support_count + 1,
            created_at=original.created_at,
            last_reinforced_at=original.last_reinforced_at,
        )
    return original
