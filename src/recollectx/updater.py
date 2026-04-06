"""MemoryUpdater — LLM-powered write path for memory management.

Inspired by Memora's MemoryUpdater pattern:
1. Cheap policy pre-filter (reject obvious junk)
2. Vector search for similar existing claims
3. LLM decides: ADD, UPDATE, DELETE, or NONE
4. Execute against storage
"""

import json
import logging
import re
from dataclasses import dataclass, field

from recollectx.claims import (
    Claim,
    EpisodicClaim,
    SemanticClaim,
)
from recollectx.graph.edges import BeliefEdge
from recollectx.llm.base import LLMProvider
from recollectx.storage.memory_store import MemoryStore, belief_to_text

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Result of processing a claim through the updater."""

    action: str
    claim: Claim | None
    reason: str
    similar_ids: list[str] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)
    edges: list[BeliefEdge] = field(default_factory=list)


class MemoryUpdater:
    """LLM-powered memory update logic.

    Uses vector search to find similar existing claims, then asks an LLM
    whether to ADD, UPDATE, DELETE, or skip (NONE).
    """

    def __init__(
        self,
        store: MemoryStore,
        llm: LLMProvider,
        similarity_k: int = 5,
    ):
        self.store = store
        self.llm = llm
        self.similarity_k = similarity_k

    def process(self, claim: Claim) -> UpdateResult:
        """Process a claim through the LLM update pipeline."""
        query_text = belief_to_text(claim)
        if not query_text:
            self.store.put(claim)
            return UpdateResult(
                action="ADD",
                claim=claim,
                reason="Not embeddable, stored directly",
            )

        similar = self.store.semantic_query(
            query_text,
            type=claim.type,
            k=self.similarity_k,
        )

        similar = [c for c in similar if c.id != claim.id]

        if not similar:
            self.store.put(claim)
            logger.debug(f"ADD claim {claim.id}: no similar claims found")
            return UpdateResult(
                action="ADD",
                claim=claim,
                reason="No similar claims found",
            )

        similar_ids = [c.id for c in similar]

        decision = self._ask_llm(claim, similar)

        return self._execute(decision, claim, similar, similar_ids)

    def _ask_llm(
        self, new_claim: Claim, existing: list[Claim]
    ) -> dict:
        """Ask the LLM to decide what to do with the new claim."""
        existing_text = json.dumps(
            [
                {"id": c.id, "content": belief_to_text(c)}
                for c in existing
            ],
            indent=2,
        )
        new_text = belief_to_text(new_claim)

        prompt = (
            "You are a smart memory manager. Compare new memories with existing ones.\n\n"
            "STEP 1 — Decide the action for the new memory:\n"
            "  ADD: Store as a new memory. Use when the new memory contains genuinely new information.\n"
            "  UPDATE: Merge into an existing memory. Use ONLY when the new memory adds detail to the same fact "
            "(e.g. 'likes pizza' → 'likes pepperoni pizza'). Provide target_id and merged_content.\n"
            "  DELETE: Remove an outdated existing memory and store the new one. Use ONLY when the old memory "
            "is completely obsolete and should not be kept at all.\n"
            "  NONE: Skip. The new memory is an exact duplicate of an existing one.\n\n"
            "STEP 2 — Detect relationships between the new memory and each existing memory:\n"
            "  supports: new memory reinforces, confirms, or coexists with an existing memory.\n"
            "  contradicts: new memory REPLACES or INVALIDATES an existing memory. "
            "The key test: can both be true at the same time? If NO → contradicts.\n"
            "  derives: new memory is a logical consequence of an existing memory.\n"
            "  Only include relationships that clearly exist. Skip if none.\n\n"
            "IMPORTANT — When a user's belief, preference, or status CHANGES:\n"
            "  Action = ADD the new fact. Relationship = contradicts the old fact.\n"
            "  Do NOT create intermediate 'Previously...' claims. The contradiction edge preserves history.\n\n"
            "Examples:\n"
            '  Old: "user likes chess" → New: "user likes rock climbing" → ADD + contradicts old (replaced hobby)\n'
            '  Old: "user is a student" → New: "user is an engineer" → ADD + contradicts old (changed role)\n'
            '  Old: "user prefers Python" → New: "user prefers TypeScript" → ADD + contradicts old (changed preference)\n'
            '  Old: "user likes pizza" → New: "user likes sushi" → ADD + supports old (both can be true)\n'
            '  Old: "user is an engineer" → New: "user is an engineer at Google" → UPDATE (more detail, same fact)\n\n'
            "Return JSON:\n"
            '{"decisions": [{"new_memory": "...", "action": "ADD|UPDATE|DELETE|NONE", '
            '"target_id": "existing ID or null", "merged_content": "merged text if UPDATE, null otherwise", '
            '"reason": "brief explanation"}], '
            '"relationships": [{"existing_id": "ID of existing memory", '
            '"relation": "supports|contradicts|derives"}]}\n\n'
            f"Existing memories:\n{existing_text}\n\n"
            f"New memory: {json.dumps([new_text])}\n\n"
            "Provide your decisions and relationships:"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=500,
            )
            parsed = self._parse_response(response.content)
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"LLM update decision failed: {e}")

        return {
            "action": "ADD",
            "target_id": None,
            "merged_content": None,
            "reason": "LLM unavailable, defaulting to ADD",
        }

    def _parse_response(self, text: str) -> dict | None:
        """Parse the LLM's JSON response."""
        md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if md_match:
            text = md_match.group(1)
        else:
            raw_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if raw_match:
                text = raw_match.group(1)

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

        decisions = parsed.get("decisions", [])
        if not decisions:
            return None

        d = decisions[0]

        relationships = []
        for rel_item in parsed.get("relationships", []):
            existing_id = rel_item.get("existing_id")
            relation = rel_item.get("relation")
            if existing_id and relation in ("supports", "contradicts", "derives"):
                relationships.append({"existing_id": existing_id, "relation": relation})

        return {
            "action": d.get("action", "ADD"),
            "target_id": d.get("target_id"),
            "merged_content": d.get("merged_content"),
            "reason": d.get("reason", ""),
            "relationships": relationships,
        }

    def _execute(
        self,
        decision: dict,
        new_claim: Claim,
        similar: list[Claim],
        similar_ids: list[str],
    ) -> UpdateResult:
        """Execute the LLM's decision against storage."""
        action = decision["action"]
        valid_similar_ids = {c.id for c in similar}

        if action == "ADD":
            self.store.put(new_claim)
            edges = self._create_edges(
                new_claim.id, decision.get("relationships", []), valid_similar_ids
            )
            logger.debug(f"ADD claim {new_claim.id}: {decision['reason']}")
            return UpdateResult(
                action="ADD",
                claim=new_claim,
                reason=decision["reason"],
                similar_ids=similar_ids,
                edges=edges,
            )

        if action == "UPDATE":
            target_id = decision.get("target_id")
            merged_content = decision.get("merged_content")

            if target_id and merged_content:
                target = next((c for c in similar if c.id == target_id), None)
                if target:
                    updated = _rebuild_claim(target, merged_content)
                    self.store.update(updated)
                    edges = self._create_edges(
                        updated.id, decision.get("relationships", []), valid_similar_ids
                    )
                    logger.debug(f"UPDATE claim {target_id}: {decision['reason']}")
                    return UpdateResult(
                        action="UPDATE",
                        claim=updated,
                        reason=decision["reason"],
                        similar_ids=similar_ids,
                        edges=edges,
                    )

            self.store.put(new_claim)
            edges = self._create_edges(
                new_claim.id, decision.get("relationships", []), valid_similar_ids
            )
            logger.debug(f"ADD (update fallback) {new_claim.id}: {decision['reason']}")
            return UpdateResult(
                action="ADD",
                claim=new_claim,
                reason=decision["reason"],
                similar_ids=similar_ids,
                edges=edges,
            )

        if action == "DELETE":
            target_id = decision.get("target_id")
            deleted_ids = []
            if target_id:
                deleted = self.store.delete(target_id)
                if deleted:
                    deleted_ids.append(target_id)
                    logger.debug(f"DELETE claim {target_id}: {decision['reason']}")

            self.store.put(new_claim)
            remaining_ids = valid_similar_ids - set(deleted_ids)
            edges = self._create_edges(
                new_claim.id, decision.get("relationships", []), remaining_ids
            )
            return UpdateResult(
                action="DELETE",
                claim=new_claim,
                reason=decision["reason"],
                similar_ids=similar_ids,
                deleted_ids=deleted_ids,
                edges=edges,
            )

        # NONE — duplicate, skip it
        logger.debug(f"NONE for claim {new_claim.id}: {decision['reason']}")
        return UpdateResult(
            action="NONE",
            claim=None,
            reason=decision["reason"],
            similar_ids=similar_ids,
        )

    def _create_edges(
        self,
        claim_id: str,
        relationships: list[dict],
        valid_ids: set[str],
    ) -> list[BeliefEdge]:
        """Create relationship edges from LLM-detected relationships."""
        edges: list[BeliefEdge] = []

        for rel_item in relationships:
            existing_id = rel_item.get("existing_id")
            relation = rel_item.get("relation")

            if not existing_id or not relation:
                continue
            if existing_id == claim_id:
                continue
            if existing_id not in valid_ids:
                continue
            if relation not in ("supports", "contradicts", "derives"):
                continue

            edge = BeliefEdge(claim_id, existing_id, relation)
            try:
                self.store.put_edge(edge)
                edges.append(edge)
                logger.debug(f"Edge: {claim_id} --{relation}--> {existing_id}")
            except Exception:
                pass

        return edges


def _rebuild_claim(original: Claim, merged_content: str) -> Claim:
    """Rebuild a claim with updated content from the LLM.

    For SemanticClaims, merged_content is expected to be the new object value only
    (the update prompt instructs the LLM to return just the object, not the full triple).
    """
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
