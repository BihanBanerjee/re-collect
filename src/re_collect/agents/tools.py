"""LangChain-based memory retrieval tools using MemoryStore.

Tools:
- search_memories: Semantic vector search (cosine similarity)
- get_recent_memories: Time-based retrieval
- get_facts_about: Get semantic facts about a subject
- get_preferences: Get user preferences
- get_all_context: Get all available context
"""

import time
from typing import Any
from langchain.tools import tool
from pydantic import BaseModel, Field


# Global memory reference (set by MemoryAgent)
_memory_instance = None


def set_memory_instance(memory):
    """Set the global memory instance for tools to use."""
    global _memory_instance
    _memory_instance = memory


def get_memory_instance():
    """Get the global memory instance."""
    if _memory_instance is None:
        raise RuntimeError("Memory instance not set. Call set_memory_instance() first.")
    return _memory_instance


# =============================================================================
# Tool Input Schemas
# =============================================================================

class SearchMemoriesInput(BaseModel):
    query: str = Field(description="The search query to find relevant memories")
    limit: int = Field(default=10, description="Maximum number of results to return")


class RecentMemoriesInput(BaseModel):
    days: int = Field(default=7, description="Number of days to look back")
    limit: int = Field(default=50, description="Maximum number of results to return")


class FactsAboutInput(BaseModel):
    subject: str = Field(default="user", description="Subject to get facts about")


class CombineFactsInput(BaseModel):
    topic: str = Field(description="The topic to gather all related facts about (e.g. 'food', 'allergy', 'hobby')")


class ContextInput(BaseModel):
    limit_per_type: int = Field(default=10, description="Maximum memories per type")


# =============================================================================
# Tool Functions (Sync implementations)
# =============================================================================

def _search_memories_impl(query: str, limit: int = 10) -> str:
    """Search memories using semantic similarity via MemoryStore."""
    memory = get_memory_instance()
    results = []

    if hasattr(memory.storage, "semantic_query"):
        try:
            results = memory.storage.semantic_query(query, k=limit)
        except Exception as e:
            print(f"Semantic search failed: {e}")

    if not results:
        all_claims = memory.retrieve()
        query_lower = query.lower()
        for claim in all_claims:
            claim_text = str(claim).lower()
            if query_lower in claim_text:
                results.append(claim)
                if len(results) >= limit:
                    break

    if not results:
        return "No memories found matching the query."

    return _format_memories(results)


def _get_recent_memories_impl(days: int = 7, limit: int = 50) -> str:
    """Get memories from the last N days."""
    memory = get_memory_instance()
    cutoff = time.time() - (days * 86400)

    all_claims = memory.retrieve()
    results = [
        c for c in all_claims
        if hasattr(c, "created_at") and c.created_at and c.created_at >= cutoff
    ]
    results = sorted(results, key=lambda c: c.created_at or 0, reverse=True)[:limit]

    if not results:
        return f"No memories found from the last {days} days."

    return _format_memories(results)


def _get_facts_about_impl(subject: str = "user") -> str:
    """Get semantic facts about a subject."""
    memory = get_memory_instance()
    semantic_claims = memory.retrieve(type="semantic")

    subject_lower = subject.lower()
    results = []
    for claim in semantic_claims:
        if hasattr(claim, "subject"):
            if subject_lower in claim.subject.lower():
                results.append(claim)
        elif subject_lower in str(claim).lower():
            results.append(claim)

    if not results:
        return f"No facts found about '{subject}'."

    return _format_memories(results)


def _get_preferences_impl() -> str:
    """Get user preferences and likes/dislikes."""
    memory = get_memory_instance()
    semantic_claims = memory.retrieve(type="semantic")

    preference_predicates = [
        "likes", "loves", "enjoys", "prefers", "wants",
        "dislikes", "hates", "avoids", "favorite",
    ]

    results = []
    for claim in semantic_claims:
        if hasattr(claim, "predicate"):
            pred_lower = claim.predicate.lower()
            if any(p in pred_lower for p in preference_predicates):
                results.append(claim)
        elif hasattr(claim, "content"):
            content_lower = str(claim.content).lower()
            if any(p in content_lower for p in preference_predicates):
                results.append(claim)

    if not results:
        return "No preferences found."

    return _format_memories(results)


def _get_all_context_impl(limit_per_type: int = 10) -> str:
    """Get all available context from memory."""
    memory = get_memory_instance()

    episodic = memory.retrieve(type="episodic")
    semantic = memory.retrieve(type="semantic")

    all_memories = (
        episodic[:limit_per_type] +
        semantic[:limit_per_type]
    )

    if not all_memories:
        return "No memories found."

    return _format_memories(all_memories)


def _combine_facts_impl(topic: str) -> str:
    """Get ALL facts, preferences, and related memories about a topic."""
    memory = get_memory_instance()
    results = []
    seen_ids = set()

    def _add(claim):
        cid = getattr(claim, "id", id(claim))
        if cid not in seen_ids:
            seen_ids.add(cid)
            results.append(claim)

    # Semantic vector search if available
    if hasattr(memory.storage, "semantic_query"):
        try:
            for claim in memory.storage.semantic_query(topic, k=10):
                _add(claim)
        except Exception:
            pass

    # Also scan all semantic claims for subject/object/predicate matches
    topic_lower = topic.lower()
    for claim in memory.retrieve(type="semantic"):
        for field in ("subject", "predicate", "object"):
            val = getattr(claim, field, "")
            if topic_lower in val.lower():
                _add(claim)
                break

    if not results:
        return f"No information found about '{topic}'."

    return _format_memories(results)


def _format_memories(memories: list[Any]) -> str:
    """Format memories for LLM context."""
    if not memories:
        return "No memories found."

    lines = []
    for i, mem in enumerate(memories[:30], 1):
        if hasattr(mem, "summary"):
            lines.append(f"{i}. [Event] {mem.summary}")
        elif hasattr(mem, "subject") and hasattr(mem, "predicate"):
            lines.append(f"{i}. [Fact] {mem.subject} {mem.predicate} {mem.object}")
        elif hasattr(mem, "content"):
            lines.append(f"{i}. {mem.content}")
        else:
            lines.append(f"{i}. {str(mem)}")

    return "\n".join(lines)


# =============================================================================
# LangChain Tools
# =============================================================================


@tool("search_memories", args_schema=SearchMemoriesInput)
def search_memories_tool(query: str, limit: int = 10) -> str:
    """Search memories using semantic similarity (cosine). Best for general questions about the user."""
    return _search_memories_impl(query, limit)


@tool("get_recent_memories", args_schema=RecentMemoriesInput)
def get_recent_memories_tool(days: int = 7, limit: int = 50) -> str:
    """Get memories from the last N days. Best for temporal questions like 'what happened recently'."""
    return _get_recent_memories_impl(days, limit)


@tool("get_facts_about", args_schema=FactsAboutInput)
def get_facts_about_tool(subject: str = "user") -> str:
    """Get semantic facts about a specific subject. Best for factual questions like 'what is the user's name'."""
    return _get_facts_about_impl(subject)


@tool("get_preferences")
def get_preferences_tool() -> str:
    """Get user preferences, likes, and dislikes. Best for preference questions like 'what food does the user like'."""
    return _get_preferences_impl()


@tool("get_all_context", args_schema=ContextInput)
def get_all_context_tool(limit_per_type: int = 10) -> str:
    """Get all available context from memory. Use when unsure which specific tool to use."""
    return _get_all_context_impl(limit_per_type)


@tool("combine_facts", args_schema=CombineFactsInput)
def combine_facts_tool(topic: str) -> str:
    """Get ALL related facts, preferences, and connected memories about a topic.
    Best for inference questions that require combining multiple pieces of information
    (e.g., 'Can the user eat X?' needs both food preferences AND allergy info)."""
    return _combine_facts_impl(topic)


def get_memory_tools() -> list:
    """Get all memory retrieval tools as LangChain tools."""
    return [
        search_memories_tool,
        get_recent_memories_tool,
        get_facts_about_tool,
        get_preferences_tool,
        get_all_context_tool,
        combine_facts_tool,
    ]
