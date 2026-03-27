"""LangGraph-based Memory Agent with MemoryStore for vector search.

The MemoryAgent uses LangGraph's prebuilt ReAct agent to:
1. Select appropriate tools based on the question
2. Execute tools to retrieve relevant memories (using cosine similarity)
3. Answer questions using the retrieved context

Architecture:
    Question -> LangGraph ReAct Agent -> Tools -> MemoryStore.semantic_query() -> Answer
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent

from .tools import get_memory_tools, set_memory_instance


@dataclass
class AgentResponse:
    """Response from the memory agent."""

    answer: str
    memories_used: list[Any]
    tools_used: list[str]
    reasoning: str | None = None


class MemoryAgent:
    """LangChain-powered agent for intelligent memory retrieval and question answering.

    Example:
        from langchain_ollama import ChatOllama
        from recollect import Memory
        from re_collect.db import SessionLocal, create_tables
        from re_collect.storage import MemoryStore
        from re_collect.storage.vector import QdrantBackend
        from re_collect.agents import MemoryAgent

        create_tables()
        db = SessionLocal()
        qdrant = QdrantBackend(
            url="http://localhost:6333",
            collection_name="memories",
            embedding_fn=embedder.embed,
            distance="cosine",
        )
        store = MemoryStore(db, qdrant)
        memory = Memory(storage=store)

        llm = ChatOllama(model="llama3")
        agent = MemoryAgent(memory=memory, llm=llm)
        response = agent.answer("What is the user's name?")
        print(response.answer)
    """

    def __init__(
        self,
        memory,
        llm: BaseChatModel,
        verbose: bool = False,
    ):
        self.memory = memory
        self.llm = llm
        self.verbose = verbose

        set_memory_instance(memory)
        self.tools = get_memory_tools()
        self._create_agent()

    def _create_agent(self):
        """Create the LangChain agent with tools."""
        system_prompt = (
            "You answer questions about a user based on their stored memories. "
            "ALWAYS use the provided tools to retrieve memories before answering. "
            "Base your answer ONLY on information from the tools. "
            "If information is not found, say 'NOT_MENTIONED'. "
            "For inference questions (e.g., 'Would the user enjoy X?', 'Can the user eat Y?'), "
            "use the combine_facts tool or call MULTIPLE tools to gather all relevant facts "
            "(preferences, allergies, habits, location) before reasoning to an answer. "
            "Start inference answers with yes or no, then briefly explain why."
        )
        self.graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )

    def answer(self, question: str) -> AgentResponse:
        """Answer a question using memory retrieval."""
        try:
            result = self.graph.invoke(
                {"messages": [{"role": "user", "content": question}]},
            )

            # Extract the final AI message
            messages = result.get("messages", [])
            answer = "NOT_MENTIONED"
            tools_used = []

            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(tc.get("name", "unknown"))
                if hasattr(msg, "content") and msg.content:
                    # Last AI message with content is the answer
                    if msg.type == "ai" and not getattr(msg, "tool_calls", None):
                        answer = msg.content

            if self._is_not_mentioned(answer):
                answer = "NOT_MENTIONED"

            return AgentResponse(
                answer=answer,
                memories_used=[],
                tools_used=tools_used,
                reasoning=None,
            )

        except Exception as e:
            print(f"Agent error: {e}")
            return AgentResponse(
                answer="NOT_MENTIONED",
                memories_used=[],
                tools_used=[],
                reasoning=f"Error: {str(e)}",
            )

    def _is_not_mentioned(self, text: str) -> bool:
        """Check if the answer indicates information is not available."""
        not_mentioned_phrases = [
            "not mentioned",
            "no information",
            "not available",
            "don't have",
            "cannot find",
            "not found",
            "unknown",
            "cannot determine",
            "insufficient",
            "no memory",
            "no record",
            "couldn't find",
            "no relevant",
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in not_mentioned_phrases)

    def search(self, query: str, limit: int = 10) -> list[Any]:
        """Direct semantic search without LLM (for simple queries)."""
        if hasattr(self.memory.storage, "semantic_query"):
            try:
                return self.memory.storage.semantic_query(query, k=limit)
            except Exception:
                pass

        all_claims = self.memory.retrieve()
        query_lower = query.lower()
        results = []
        for claim in all_claims:
            if query_lower in str(claim).lower():
                results.append(claim)
                if len(results) >= limit:
                    break
        return results

    def get_context(self, limit_per_type: int = 10) -> dict[str, list[Any]]:
        """Get all context from memory."""
        episodic = self.memory.retrieve(type="episodic")
        semantic = self.memory.retrieve(type="semantic")

        return {
            "episodic": episodic[:limit_per_type],
            "semantic": semantic[:limit_per_type],
        }
