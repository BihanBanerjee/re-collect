"""recollect - Belief-centric memory layer for agentic AI systems.

This package provides a structured memory system for AI agents based on
the concept of beliefs rather than simple logs or storage. Key features:

- Structured beliefs (claims) with confidence scores and evidence
- Two claim types: episodic and semantic
- Automatic contradiction detection for semantic claims
- Composable write policies for filtering claims
- Belief graph for tracking relationships between beliefs

Basic Example:
    from recollect import Memory, SemanticClaim
    from re_collect.db import SessionLocal, create_tables
    from re_collect.storage import MemoryStore
    from re_collect.storage.vector import FAISSBackend

    create_tables()
    db = SessionLocal()
    vectors = FAISSBackend(embed_fn=my_embed, dimension=384)
    store = MemoryStore(db, vectors)
    memory = Memory(storage=store)

    claim = SemanticClaim(
        subject="sky",
        predicate="has_color",
        object="blue",
        confidence=0.9,
    )
    memory.store(claim)
    results = memory.retrieve(type="semantic")

Optional Modules:
    - recollect.agents: LangChain-based agents with tool-based retrieval
    - recollect.extractors: Automatic claim extraction from text
    - recollect.deduplication: Duplicate claim detection and merging
    - recollect.storage.vector: Vector backends (Qdrant, Pinecone)
    - recollect.storage: MemoryStore (SQLite + vectors)
    - recollect.llm.providers: LLM providers (OpenAI, Anthropic, Ollama)

LangChain Agent Example:
    from re_collect.agents import MemoryAgent
    from re_collect.storage import MemoryStore
    from langchain_ollama import ChatOllama

    # Install with: pip install re-collect[langchain,qdrant]
    llm = ChatOllama(model="llama3")
    agent = MemoryAgent(memory=memory, llm=llm)
    response = agent.answer("What does the user like?")
"""

from .claims import Claim, EpisodicClaim, SemanticClaim
from .graph.explanation import ExplanationNode, ExplanationResult
from .memory import Memory
from .propagation import ConfidenceChangeEvent, ConfidencePropagator, PropagationConfig
from .state import AgentState
from .updater import MemoryUpdater, UpdateResult

__all__ = [
    "Memory",
    "MemoryUpdater",
    "UpdateResult",
    "AgentState",
    "Claim",
    "EpisodicClaim",
    "SemanticClaim",
    "PropagationConfig",
    "ConfidencePropagator",
    "ConfidenceChangeEvent",
    "ExplanationNode",
    "ExplanationResult",
]

__version__ = "0.2.0"
