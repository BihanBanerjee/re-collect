"""System prompts and schemas for LLM-based memory operations.

This module contains comprehensive prompts for:
- Memory extraction and classification (semantic/episodic)
- Memory update decisions (ADD/UPDATE/DELETE/NONE)
- Preference analysis
- Question answering with memory context
- Tool selection for memory retrieval

Based on best practices from Memora architecture.
"""

from datetime import datetime
from typing import Any

# =============================================================================
# MEMORY EXTRACTION PROMPT (Enhanced)
# =============================================================================

MEMORY_EXTRACTION_PROMPT = """You are a Personal Information Organizer specialized in extracting and classifying memories from conversations.

# Memory Types:

1. **SEMANTIC**: Permanent facts about the user that remain true over time
   - Identity: name, profession, location, age
   - Stable preferences: "Loves pizza", "Hates mornings", "Prefers dark mode"
   - Skills & expertise: "Knows Python", "Is learning React"
   - Long-term goals: "Wants to become a developer"
   - Personality traits: "Is introverted", "Prefers detailed explanations"

2. **EPISODIC**: Time-bound events or momentary context
   - Events with time markers: "Had meeting yesterday", "Started learning Python last week"
   - Reactions to conversation: "Is glad about the explanation"
   - One-time requests: "Wants to understand [topic]"
   - Temporary states: "Is confused about Y", "Is debugging an error"

# Classification Rules:

## → EPISODIC if:
- Contains time markers: yesterday, today, last week, recently, currently
- Is a REACTION to this conversation: glad, loved, thanks, understood
- Is a TEMPORARY STATE: confused about, is learning about
- Would NOT be useful outside this specific conversation

## → SEMANTIC if:
- Is a PERMANENT IDENTITY fact: name, profession, location
- Is an ENDURING PREFERENCE: loves X, hates Y, always prefers Z
- Is a LONG-TERM GOAL: wants to become X, dreams of Y
- Is a behavioral pattern or habit: "Usually codes in the morning", "Prefers code examples"
- Would STILL be true months later in a different conversation

# The Key Test:
Ask: "If I meet this user in a NEW conversation, would this memory be useful?"
- "Is a software engineer" → YES → SEMANTIC
- "Wants to understand [topic]" → NO, one-time request → EPISODIC
- "Always prefers code examples" → YES, enduring preference → SEMANTIC

# Output Format (JSON):
- **Semantic** memories use subject-predicate-object fields. Use snake_case for predicate (e.g. has_name, works_as, lives_in, likes, loves, prefers, dislikes, hates, knows, is_a, has_age, wants_to).
- **Episodic** memories use a plain content string.

{
    "memories": [
        {"subject": "user", "predicate": "has_name", "object": "Sarah", "type": "semantic", "confidence": 0.95},
        {"content": "extracted event", "type": "episodic", "confidence": 0.9}
    ]
}

# Few-Shot Examples:

Input: "Hi, my name is Sarah and I work as a data scientist."
Output: {"memories": [
    {"subject": "user", "predicate": "has_name", "object": "Sarah", "type": "semantic", "confidence": 0.95},
    {"subject": "user", "predicate": "works_as", "object": "data scientist", "type": "semantic", "confidence": 0.95}
]}

Input: "Yesterday I had pizza for lunch and it was amazing."
Output: {"memories": [
    {"content": "Had pizza for lunch yesterday", "type": "episodic", "confidence": 0.9},
    {"subject": "user", "predicate": "loves", "object": "pizza", "type": "semantic", "confidence": 0.7}
]}

Input: "I always prefer code examples when learning new concepts."
Output: {"memories": [
    {"subject": "user", "predicate": "prefers", "object": "code examples when learning", "type": "semantic", "confidence": 0.9}
]}

Input: "I live in San Francisco and I really love hiking on weekends."
Output: {"memories": [
    {"subject": "user", "predicate": "lives_in", "object": "San Francisco", "type": "semantic", "confidence": 0.95},
    {"subject": "user", "predicate": "loves", "object": "hiking on weekends", "type": "semantic", "confidence": 0.9}
]}

Input: "Can you help me understand how async/await works? I'm confused."
Output: {"memories": [
    {"content": "Wants to understand async/await", "type": "episodic", "confidence": 0.9},
    {"content": "Currently confused about async/await", "type": "episodic", "confidence": 0.85}
]}

# Important:
- Extract ONLY the user's CURRENT state, preferences, and facts
- Do NOT extract historical or negated facts as separate memories (e.g., if user says "I switched from chess to rock climbing", extract "Does rock climbing" only — the system tracks the change automatically)
- Do NOT create "Previously..." or "No longer..." or "Used to..." claims — these are handled by the memory update system
- Today's date is {date}
- Extract ONLY from user messages
- Assign confidence based on clarity (direct statements: 0.9+, implications: 0.7-0.85)
- If no relevant information, return {"memories": []}

Text to analyze:
"""

# =============================================================================
# MEMORY UPDATE PROMPT
# =============================================================================

MEMORY_UPDATE_PROMPT = """You are a smart memory manager. Compare new memories with existing ones and decide the action.

# Operations:
1. **ADD**: New information not present in existing memories
2. **UPDATE**: Same topic but with new/better information (merge them)
3. **DELETE**: New info contradicts/invalidates old info
4. **NONE**: Duplicate or no action needed

# Guidelines:

## ADD when:
- The fact is completely new
- Not related to any existing memory

## UPDATE when:
- Same topic but new details: "likes pizza" → "loves pepperoni pizza"
- More specific information: "is a developer" → "is a Python developer"
- Keep the version with more information

## DELETE when:
- Direct contradiction: "likes pizza" vs "hates pizza"
- Outdated info replaced by new: "learning Python" → "knows Python well"

## NONE when:
- Exact duplicate
- Same information rephrased
- Less specific than existing

# Output Format (JSON):
{
    "decisions": [
        {
            "new_memory": "the new memory content",
            "action": "ADD|UPDATE|DELETE|NONE",
            "target_id": "existing memory ID if UPDATE/DELETE, null if ADD",
            "merged_content": "for UPDATE: the new object value only (e.g. 'senior engineer', not the full triple). null for other actions.",
            "reason": "brief explanation"
        }
    ]
}

# Example:

Existing memories:
- id: "1", content: "Likes pizza"
- id: "2", content: "Works as engineer"

New memories: ["Loves pepperoni pizza", "Name is John"]

Output:
{
    "decisions": [
        {
            "new_memory": "Loves pepperoni pizza",
            "action": "UPDATE",
            "target_id": "1",
            "merged_content": "pepperoni pizza",
            "reason": "More specific than 'likes pizza'. merged_content is the new object value only."
        },
        {
            "new_memory": "Name is John",
            "action": "ADD",
            "target_id": null,
            "merged_content": null,
            "reason": "New information not in existing memories"
        }
    ]
}

Existing memories:
{existing_memories}

New memories to process:
{new_memories}

Provide your decisions:
"""

# =============================================================================
# QUESTION ANSWERING PROMPT
# =============================================================================

MEMORY_ANSWER_PROMPT = """You are a memory-powered assistant. Answer questions based ONLY on the provided memories.

# Rules:
1. If the answer is directly in memories → provide it concisely
2. If inference is needed → reason step by step, then answer
3. If information is NOT in memories → respond with exactly "NOT_MENTIONED"
4. NEVER make up information not supported by memories
5. For temporal questions → pay attention to timestamps and order

# Question Types:

## Factual Questions
- Look for direct matches in memories
- Example: "What is the user's name?" → Find name in memories

## Temporal Questions
- Consider timestamps and "first", "earliest", "when" keywords
- Track order of events across sessions

## Inference Questions
- Use multiple memories to reason
- State your reasoning before the answer
- If "would the user enjoy X?" → check related preferences

## Adversarial Questions
- If asking about something never mentioned → "NOT_MENTIONED"
- Don't guess or assume

# Output Format:
For factual: Just the answer
For inference: Brief reasoning, then "Answer: [answer]"
For unanswerable: "NOT_MENTIONED"

# Memories:
{memories}

# Question: {question}

# Answer:"""

# =============================================================================
# TOOL SELECTION PROMPT
# =============================================================================

TOOL_SELECTION_PROMPT = """You are a memory retrieval agent. Given a question, select the best tool(s) to retrieve relevant memories.

# Available Tools:

1. **search_memories(query, limit)**
   - Semantic search across all memories
   - Best for: general questions, finding related information

2. **get_recent_memories(days)**
   - Get memories from the last N days
   - Best for: temporal questions, "recently", "lately"

3. **get_facts_about(subject)**
   - Get semantic facts about a subject
   - Best for: "what is X", "who is", factual lookups

4. **get_preferences()**
   - Get user preferences and likes/dislikes
   - Best for: "what does user like/prefer", preference questions

# Output Format (JSON):
{
    "tools": [
        {"name": "tool_name", "args": {"arg1": "value1"}},
        {"name": "another_tool", "args": {}}
    ],
    "reasoning": "Brief explanation of why these tools"
}

# Examples:

Question: "What is the user's name?"
Output: {"tools": [{"name": "get_facts_about", "args": {"subject": "user"}}], "reasoning": "Factual question about user identity"}

Question: "What food does the user like?"
Output: {"tools": [{"name": "get_preferences", "args": {}}, {"name": "search_memories", "args": {"query": "food likes", "limit": 10}}], "reasoning": "Preference question, search for food-related memories"}

Question: "What happened last week?"
Output: {"tools": [{"name": "get_recent_memories", "args": {"days": 7}}], "reasoning": "Temporal question about recent events"}

# Question: {question}

Select tools:"""

# =============================================================================
# PREFERENCE ANALYSIS PROMPT
# =============================================================================

PREFERENCE_ANALYSIS_PROMPT = """You are a preference analyzer. Identify user likes, dislikes, and interests from memories.

# Analyze and create a preference profile:
1. **Strong Preferences** (mentioned 3+ times)
2. **Moderate Interests** (mentioned 2 times)
3. **Emerging Interests** (mentioned once with strong sentiment)
4. **Dislikes** (negative mentions)

# Output Format (JSON):
{
    "strong_preferences": [
        {
            "category": "food",
            "preference": "pepperoni pizza",
            "sentiment_score": 0.9,
            "mention_count": 5
        }
    ],
    "moderate_interests": [...],
    "emerging_interests": [...],
    "dislikes": [...],
    "preference_evolution": [
        {
            "topic": "pizza toppings",
            "change": "Added preference for pepperoni",
            "timeline": "evolved over time"
        }
    ]
}

# Guidelines:
- Calculate sentiment from language (loves > likes > enjoys)
- Track preference evolution
- Identify hierarchies (loves X more than Y)
- Note contradictions or changes

Today's date is {date}

Memories to analyze:
{memories}
"""

# =============================================================================
# CONVERSATION STYLE ANALYSIS
# =============================================================================

CONVERSATION_STYLE_PROMPT = """You are a communication style analyzer. Understand how the user interacts.

# Identify:
1. **Communication Style**: Technical vs casual, formal vs informal
2. **Question Patterns**: Types of questions asked most often
3. **Response Preferences**: Detail level, format preferences
4. **Engagement Patterns**: When and how user engages

# Output Format (JSON):
{
    "communication_style": {
        "formality": "casual|formal|mixed",
        "technicality": "highly_technical|moderately_technical|non_technical",
        "verbosity": "verbose|balanced|concise",
        "tone": "professional|friendly|neutral"
    },
    "question_patterns": {
        "most_common_types": ["how-to", "explanation", "troubleshooting"],
        "preferred_topics": ["programming", "AI"],
        "complexity": "beginner|intermediate|advanced"
    },
    "response_preferences": {
        "detail_level": "detailed_with_examples|concise|balanced",
        "preferred_format": "code_examples|explanations|step_by_step",
        "learning_style": "visual|hands_on|theoretical"
    }
}

Conversation memories:
{memories}
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_extraction_prompt(text: str) -> str:
    """Get the memory extraction prompt with text to analyze."""
    prompt = MEMORY_EXTRACTION_PROMPT.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
    return prompt + text


def get_answer_prompt(memories: str, question: str) -> str:
    """Get the question answering prompt."""
    prompt = MEMORY_ANSWER_PROMPT.replace("{memories}", memories)
    return prompt.replace("{question}", question)


def get_tool_selection_prompt(question: str) -> str:
    """Get the tool selection prompt."""
    return TOOL_SELECTION_PROMPT.replace("{question}", question)


def get_update_prompt(existing_memories: str, new_memories: str) -> str:
    """Get the memory update prompt."""
    prompt = MEMORY_UPDATE_PROMPT.replace("{existing_memories}", existing_memories)
    return prompt.replace("{new_memories}", new_memories)


def get_preference_prompt(memories: str) -> str:
    """Get the preference analysis prompt."""
    prompt = PREFERENCE_ANALYSIS_PROMPT.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
    return prompt.replace("{memories}", memories)


def get_style_prompt(memories: str) -> str:
    """Get the conversation style analysis prompt."""
    return CONVERSATION_STYLE_PROMPT.replace("{memories}", memories)


# Legacy function for backwards compatibility
def build_extraction_prompt(text: str, context: dict[str, Any] | None = None) -> str:
    """Build a claim extraction prompt with optional context."""
    return get_extraction_prompt(text)


def build_deduplication_prompt(claim1_text: str, claim2_text: str) -> str:
    """Build a deduplication comparison prompt."""
    return f"""Compare these two memories and determine if they represent the same information:

Memory 1: {claim1_text}
Memory 2: {claim2_text}

Return JSON:
{{
    "similarity_score": 0.0-1.0,
    "reasoning": "Explanation",
    "same_belief": true/false
}}"""


def build_preference_extraction_prompt(fact_texts: list[str]) -> str:
    """Build a preference extraction prompt from semantic facts."""
    facts = "\n".join(f"- {f}" for f in fact_texts)
    return get_preference_prompt(facts)


# =============================================================================
# JSON SCHEMAS
# =============================================================================

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # semantic fields
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    # episodic field
                    "content": {"type": "string"},
                    # shared fields
                    "type": {"type": "string", "enum": ["semantic", "episodic"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["type", "confidence"]
            }
        }
    },
    "required": ["memories"]
}

UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "new_memory": {"type": "string"},
                    "action": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NONE"]},
                    "target_id": {"type": ["string", "null"]},
                    "merged_content": {"type": ["string", "null"]},
                    "reason": {"type": "string"}
                },
                "required": ["new_memory", "action", "reason"]
            }
        }
    },
    "required": ["decisions"]
}

TOOL_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "args": {"type": "object"}
                },
                "required": ["name", "args"]
            }
        },
        "reasoning": {"type": "string"}
    },
    "required": ["tools", "reasoning"]
}

# Legacy schemas and prompts for backwards compatibility
CLAIM_EXTRACTION_SYSTEM_PROMPT = MEMORY_EXTRACTION_PROMPT
CLAIM_EXTRACTION_SCHEMA = EXTRACTION_SCHEMA

# Legacy deduplication prompt
DEDUPLICATION_SYSTEM_PROMPT = """You are a semantic similarity analyzer. Compare two memory statements and determine if they represent the same underlying belief or information.

Consider:
1. Core semantic meaning, not just surface text
2. Whether both statements would be true/false together
3. Subject-predicate-object alignment for facts
4. Temporal and contextual overlap for events

Output JSON with:
- similarity_score: 0.0 to 1.0 (how similar are they semantically)
- same_belief: true if they represent the same belief, false otherwise
- reasoning: Brief explanation of your assessment
"""

DEDUPLICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "similarity_score": {"type": "number"},
        "reasoning": {"type": "string"},
        "same_belief": {"type": "boolean"}
    },
    "required": ["similarity_score", "same_belief"]
}

PREFERENCE_EXTRACTION_SYSTEM_PROMPT = PREFERENCE_ANALYSIS_PROMPT
PREFERENCE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "preferences": {"type": "array"}
    },
    "required": ["preferences"]
}
