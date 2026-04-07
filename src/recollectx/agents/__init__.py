"""LangChain-based memory agents with tool-based retrieval.

This module provides LLM-powered agents for intelligent memory operations:
- MemoryAgent: Full agent with tool selection and execution using LangChain
- Tool definitions for memory retrieval

Example:
    from langchain_ollama import ChatOllama
    from recollectx.agents import MemoryAgent

    llm = ChatOllama(model="llama3")
    agent = MemoryAgent(memory=memory, llm=llm)
    response = agent.answer("What is the user's name?")
"""

from .memory_agent import AgentResponse, MemoryAgent
from .tools import (
    combine_facts_tool,
    get_all_context_tool,
    get_facts_about_tool,
    get_memory_tools,
    get_preferences_tool,
    get_recent_memories_tool,
    search_memories_tool,
    set_memory_instance,
)

__all__ = [
    "MemoryAgent",
    "AgentResponse",
    "get_memory_tools",
    "set_memory_instance",
    "search_memories_tool",
    "get_recent_memories_tool",
    "get_facts_about_tool",
    "get_preferences_tool",
    "get_all_context_tool",
    "combine_facts_tool",
]
