"""Main deduplication logic for claims.

Simple two-step pipeline:
1. Find similar claims using embedding cosine similarity
2. Ask the LLM what to do (ADD, UPDATE, DELETE, NONE)
3. Execute the action against storage
"""

import logging
from dataclasses import dataclass, field

from re_collect.claims import Claim
from re_collect.deduplication.merger import LLMMerger, MergeDecision
from re_collect.deduplication.similarity import EmbeddingSimilarity
from re_collect.llm.base import EmbeddingProvider, LLMProvider
from re_collect.storage.memory_store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of processing a claim through deduplication."""

    action: str
    claim: Claim | None
    reason: str
    similar_ids: list[str] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)


class ClaimDeduplicator:
    """Detect and handle duplicate claims.

    Uses embeddings to find similar claims, then asks an LLM
    what to do with them, then executes the action against storage.
    """

    def __init__(
        self,
        storage: MemoryStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        similarity_threshold: float = 0.8,
    ):
        self.storage = storage
        self.similarity = EmbeddingSimilarity(embedding_provider)
        self.merger = LLMMerger(llm_provider)
        self.threshold = similarity_threshold

    def process(self, new_claim: Claim) -> DeduplicationResult:
        """Process a new claim: find duplicates, decide, and execute."""
        existing_claims = self.storage.query(type=new_claim.type)

        similar = self._find_similar(new_claim, existing_claims)
        similar_ids = [c.id for c in similar]

        if not similar:
            self.storage.put(new_claim)
            logger.debug(f"ADD claim {new_claim.id}: no similar claims found")
            return DeduplicationResult(
                action="ADD",
                claim=new_claim,
                reason="No similar claims found",
                similar_ids=[],
            )

        decision = self.merger.decide(new_claim, similar)

        return self._execute(decision, new_claim, similar, similar_ids)

    def _execute(
        self,
        decision: MergeDecision,
        new_claim: Claim,
        similar: list[Claim],
        similar_ids: list[str],
    ) -> DeduplicationResult:
        """Execute the LLM's decision against storage."""

        if decision.action == "ADD":
            self.storage.put(new_claim)
            logger.debug(f"ADD claim {new_claim.id}: {decision.reason}")
            return DeduplicationResult(
                action="ADD",
                claim=new_claim,
                reason=decision.reason,
                similar_ids=similar_ids,
            )

        if decision.action == "UPDATE":
            result_claim = self.merger.apply_decision(decision, new_claim, similar)
            if result_claim and decision.target_id:
                target = next((c for c in similar if c.id == decision.target_id), None)
                if target:
                    updated = self.merger.apply_decision(decision, target, similar)
                    if updated:
                        self.storage.update(updated)
                        result_claim = updated
                        logger.debug(f"UPDATE claim {decision.target_id}: {decision.reason}")
                else:
                    self.storage.put(result_claim)
                    logger.debug(
                        f"ADD (target not found) claim {result_claim.id}: {decision.reason}"
                    )
            elif result_claim:
                self.storage.put(result_claim)
                logger.debug(f"ADD (from UPDATE) claim {result_claim.id}: {decision.reason}")
            return DeduplicationResult(
                action="UPDATE",
                claim=result_claim,
                reason=decision.reason,
                similar_ids=similar_ids,
            )

        if decision.action == "DELETE":
            deleted_ids = []
            if decision.target_id:
                deleted = self.storage.delete(decision.target_id)
                if deleted:
                    deleted_ids.append(decision.target_id)
                    logger.debug(f"DELETE claim {decision.target_id}: {decision.reason}")
            self.storage.put(new_claim)
            return DeduplicationResult(
                action="DELETE",
                claim=new_claim,
                reason=decision.reason,
                similar_ids=similar_ids,
                deleted_ids=deleted_ids,
            )

        # NONE — duplicate, skip it
        logger.debug(f"NONE for claim {new_claim.id}: {decision.reason}")
        return DeduplicationResult(
            action="NONE",
            claim=None,
            reason=decision.reason,
            similar_ids=similar_ids,
        )

    def _find_similar(
        self, new_claim: Claim, existing_claims: list[Claim]
    ) -> list[Claim]:
        """Find existing claims similar to the new one."""
        similar: list[tuple[float, Claim]] = []

        for existing in existing_claims:
            if existing.id == new_claim.id:
                continue
            if type(existing) is not type(new_claim):
                continue

            score = self.similarity.calculate(new_claim, existing)
            if score >= self.threshold:
                similar.append((score, existing))

        similar.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in similar]
