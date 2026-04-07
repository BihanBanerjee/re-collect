"""LLM integration layer for recollect.

This module provides abstract interfaces (protocols) for LLM and embedding providers,
along with reusable prompts for memory operations.

All LLM functionality is OPTIONAL. The core recollect library works perfectly without
any LLM integration. LLM features enhance the library with:
- Automatic claim extraction and classification
- Memory update decisions (ADD/UPDATE/DELETE)
- Preference analysis and aggregation
- Tool-based memory retrieval

To use LLM features, you'll need to:
1. Install optional dependencies: pip install re-collect[llm]
2. Implement or use a provider (OpenAI, Anthropic, Google, Ollama, etc.)
3. Pass the provider to extractors or agents

Example:
    # Without LLM (core functionality)
    from recollect import Memory, SemanticClaim
    claim = SemanticClaim(subject="user", predicate="likes", object="pizza")

    # With LLM extraction
    from recollectx.llm.providers import OpenAIProvider
    from recollectx.extractors import LLMExtractor

    llm = OpenAIProvider(api_key="...")
    extractor = LLMExtractor(llm)
    claims = extractor.extract("I love pizza")

    # With LLM agent
    from recollectx.agents import MemoryAgent
    agent = MemoryAgent(memory=memory, llm=llm)
    response = agent.answer("What food does the user like?")
"""

from recollectx.llm.base import EmbeddingProvider, LLMProvider, LLMResponse
from recollectx.llm.prompts import (
    # Legacy aliases
    CLAIM_EXTRACTION_SCHEMA,
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    CONVERSATION_STYLE_PROMPT,
    DEDUPLICATION_SCHEMA,
    DEDUPLICATION_SYSTEM_PROMPT,
    # JSON schemas
    EXTRACTION_SCHEMA,
    MEMORY_ANSWER_PROMPT,
    # Main prompts
    MEMORY_EXTRACTION_PROMPT,
    MEMORY_UPDATE_PROMPT,
    PREFERENCE_ANALYSIS_PROMPT,
    PREFERENCE_EXTRACTION_SCHEMA,
    PREFERENCE_EXTRACTION_SYSTEM_PROMPT,
    TOOL_SELECTION_PROMPT,
    TOOL_SELECTION_SCHEMA,
    UPDATE_SCHEMA,
    build_deduplication_prompt,
    # Prompt builders
    build_extraction_prompt,
    build_preference_extraction_prompt,
    get_answer_prompt,
    get_extraction_prompt,
    get_preference_prompt,
    get_style_prompt,
    get_tool_selection_prompt,
    get_update_prompt,
)

__all__ = [
    # Base protocols
    "LLMProvider",
    "EmbeddingProvider",
    "LLMResponse",
    # Main prompts
    "MEMORY_EXTRACTION_PROMPT",
    "MEMORY_UPDATE_PROMPT",
    "MEMORY_ANSWER_PROMPT",
    "TOOL_SELECTION_PROMPT",
    "PREFERENCE_ANALYSIS_PROMPT",
    "CONVERSATION_STYLE_PROMPT",
    # JSON schemas
    "EXTRACTION_SCHEMA",
    "UPDATE_SCHEMA",
    "TOOL_SELECTION_SCHEMA",
    # Legacy aliases
    "CLAIM_EXTRACTION_SYSTEM_PROMPT",
    "CLAIM_EXTRACTION_SCHEMA",
    "DEDUPLICATION_SYSTEM_PROMPT",
    "DEDUPLICATION_SCHEMA",
    "PREFERENCE_EXTRACTION_SYSTEM_PROMPT",
    "PREFERENCE_EXTRACTION_SCHEMA",
    # Prompt builders
    "build_extraction_prompt",
    "build_deduplication_prompt",
    "build_preference_extraction_prompt",
    "get_extraction_prompt",
    "get_answer_prompt",
    "get_tool_selection_prompt",
    "get_update_prompt",
    "get_preference_prompt",
    "get_style_prompt",
]
