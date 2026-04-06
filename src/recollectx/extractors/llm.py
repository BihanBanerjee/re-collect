"""LLM-based claim extraction from text.

This module implements automatic claim extraction using a Large Language Model.
It requires an LLMProvider implementation (OpenAI, Anthropic, Google, Ollama, etc.).

The extractor classifies memories into two types:
- SEMANTIC: Permanent facts about the user
- EPISODIC: Time-bound events
"""

import json
import time
from typing import Any

from recollectx.claims import Claim, EpisodicClaim, SemanticClaim
from recollectx.llm.base import LLMProvider
from recollectx.llm.prompts import get_extraction_prompt, EXTRACTION_SCHEMA


class LLMExtractor:
    """Extract claims from text using a Large Language Model.

    This extractor uses sophisticated prompts to automatically identify
    episodic and semantic claims in natural language text.

    Features:
    - Automatic claim type classification (semantic/episodic)
    - Confidence scoring based on clarity of statements
    - Context-aware extraction
    Example:
        from recollectx.llm.providers import OpenAIProvider
        from recollectx.extractors import LLMExtractor

        llm = OpenAIProvider(api_key="...")
        extractor = LLMExtractor(llm, min_confidence=0.6)

        claims = extractor.extract("I love pizza on Fridays")
        for claim in claims:
            print(f"{claim.__class__.__name__}: confidence={claim.confidence}")
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        min_confidence: float = 0.5,
        max_claims_per_text: int = 20,
        temperature: float = 0.1,
    ):
        """Initialize the LLM extractor.

        Args:
            llm_provider: LLM provider implementation (OpenAI, Anthropic, etc.)
            min_confidence: Minimum confidence threshold (0.0-1.0). Claims below this
                are filtered out. Lower = more claims, higher = fewer but more certain.
            max_claims_per_text: Maximum claims to extract per text (prevents
                over-extraction from very long texts)
            temperature: LLM temperature (0.0=deterministic, 1.0=creative).
                Lower is better for factual extraction (default: 0.1)
        """
        self.llm = llm_provider
        self.min_confidence = min_confidence
        self.max_claims = max_claims_per_text
        self.temperature = temperature

    def extract(
        self, text: str, context: dict[str, Any] | None = None
    ) -> list[Claim]:
        """Extract claims from text using LLM.

        Args:
            text: Natural language text to analyze
            context: Optional context dict (user_id, timestamp, conversation_id, etc.)

        Returns:
            List of extracted claims (filtered by min_confidence)

        Example:
            claims = extractor.extract(
                text="I love pizza. Last Friday I ordered from Dominos.",
                context={"user_id": "user_123"}
            )
        """
        # Build extraction prompt
        prompt = get_extraction_prompt(text)

        # Get response from LLM
        try:
            # Try structured output first if available
            if hasattr(self.llm, "generate_structured"):
                response = self.llm.generate_structured(
                    prompt=prompt,
                    schema=EXTRACTION_SCHEMA,
                    temperature=self.temperature,
                )
            else:
                # Fallback to regular generation and parse JSON
                response_obj = self.llm.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=1000,
                )
                response = self._parse_json_response(
                    response_obj.content if hasattr(response_obj, "content") else str(response_obj)
                )
        except Exception as e:
            print(f"Extraction failed: {e}")
            return []

        # Convert response to Claim objects
        claims = self._parse_claims(response, context)

        # Filter by confidence and limit
        filtered = [c for c in claims if c.confidence >= self.min_confidence]
        return filtered[: self.max_claims]

    def extract_batch(
        self, texts: list[str], context: dict[str, Any] | None = None
    ) -> list[list[Claim]]:
        """Extract claims from multiple texts.

        Note: This currently processes texts sequentially. For better performance
        with batch-capable LLMs, consider implementing parallel processing.

        Args:
            texts: List of texts to analyze
            context: Optional shared context

        Returns:
            List of claim lists (one per input text)
        """
        results = []
        for text in texts:
            claims = self.extract(text, context)
            results.append(claims)
        return results

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response.

        Args:
            text: Text that may contain JSON

        Returns:
            Parsed dict or empty dict
        """
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        return {"memories": []}

    def _parse_claims(
        self, response: dict[str, Any], context: dict[str, Any] | None
    ) -> list[Claim]:
        """Convert LLM response to Claim objects.

        Semantic memories: {"subject": "user", "predicate": "works_as", "object": "engineer", "type": "semantic", "confidence": 0.9}
        Episodic memories: {"content": "Had coffee this morning", "type": "episodic", "confidence": 0.8}
        """
        claims: list[Claim] = []
        default_subject = self._get_subject(context)

        for memory_data in response.get("memories", []):
            memory_type = memory_data.get("type", "")
            confidence = float(memory_data.get("confidence", 0.5))

            if not memory_type:
                continue

            try:
                if memory_type == "episodic":
                    content = memory_data.get("content", "").strip()
                    if not content:
                        continue
                    claim = self._create_episodic_claim(content, confidence, context)

                elif memory_type == "semantic":
                    subject = memory_data.get("subject", default_subject).strip() or default_subject
                    predicate = memory_data.get("predicate", "").strip()
                    obj = memory_data.get("object", "").strip()
                    if not predicate or not obj:
                        continue
                    claim = SemanticClaim(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=confidence,
                        evidence=(f"{subject} {predicate} {obj}",),
                    )

                else:
                    continue

                claims.append(claim)

            except (ValueError, KeyError) as e:
                print(f"Failed to create claim: {e}")
                continue

        return claims

    def _get_subject(self, context: dict[str, Any] | None) -> str:
        """Get the subject identifier from context or use default."""
        if context and "user_id" in context:
            return str(context["user_id"])
        return "user"

    def _create_episodic_claim(
        self,
        content: str,
        confidence: float,
        context: dict[str, Any] | None,
    ) -> EpisodicClaim:
        """Create an EpisodicClaim from extracted content."""
        # Use context timestamp if available
        if context and "timestamp" in context:
            created_at = context["timestamp"]
        else:
            created_at = time.time()

        return EpisodicClaim(
            summary=content,
            confidence=confidence,
            created_at=created_at,
            last_reinforced_at=created_at,
            evidence=(content,),
        )


