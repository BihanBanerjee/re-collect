# re-collect

A belief-centric memory layer for agentic AI systems.

re-collect structures memory as **beliefs** — not logs. Every piece of information is stored as a typed claim with a confidence score, supporting evidence, and explicit relationships to other beliefs. This makes memory inspectable, revisable, and suitable for agents that need to reason about what they know and why.

---

## Features

- **Structured beliefs** — two claim types: `SemanticClaim` (facts as subject-predicate-object triples) and `EpisodicClaim` (events with a summary)
- **Confidence scoring** — every claim has a confidence value in `[0.0, 1.0]`; contradictions decay it, supports boost it
- **Belief graph** — tracks `supports`, `contradicts`, `derives`, and `similar` relationships between beliefs
- **Write policies** — composable filters that accept or reject claims before storage
- **LLM-powered updates** — intelligent ADD / UPDATE / DELETE decisions using any LLM provider
- **Pluggable storage** — SQLite (SQLAlchemy) + optional vector backends (FAISS, Qdrant, Pinecone)
- **LangGraph agent** — answer questions by retrieving from memory using tool-based reasoning
- **Automatic extraction** — extract structured claims from raw text via LLM
- **Deduplication** — detect and merge duplicate claims using embedding similarity + LLM merge decisions
- **Deep explanations** — trace why a belief exists with recursive justification chains

---

## Installation

### Core (SQLite storage only)

```bash
pip install re-collect
```

### With optional extras

```bash
# OpenAI LLM + embeddings
pip install "re-collect[openai]"

# Anthropic Claude
pip install "re-collect[anthropic]"

# Local Ollama models
pip install "re-collect[ollama]"

# Qdrant vector backend
pip install "re-collect[qdrant]"

# Pinecone vector backend
pip install "re-collect[pinecone]"

# Local FAISS + sentence-transformers (no API needed)
pip install "re-collect[local]"

# All LLM providers
pip install "re-collect[llm]"

# All vector backends
pip install "re-collect[vector]"

# Everything
pip install "re-collect[all]"
```

---

## Quick Start

```python
from recollectx import Memory, SemanticClaim, EpisodicClaim
from recollectx.db import SessionLocal, create_tables
from recollectx.storage import MemoryStore
from recollectx.storage.vector import FAISSBackend

# 1. Set up the database
create_tables()
db = SessionLocal()

# 2. Set up a vector backend (requires re-collect[local])
vectors = FAISSBackend(embed_fn=my_embed_fn, dimension=384)
store = MemoryStore(db, vectors)

# 3. Create a Memory instance
memory = Memory(storage=store)

# 4. Store beliefs
fact = SemanticClaim(
    subject="sky",
    predicate="has_color",
    object="blue",
    confidence=0.9,
    evidence=("direct observation",),
)
memory.store(fact)

event = EpisodicClaim(
    summary="User mentioned they love pizza on Fridays",
    confidence=0.85,
    importance=0.7,
)
memory.store(event)

# 5. Retrieve beliefs
facts = memory.retrieve(type="semantic")
episodes = memory.retrieve(type="episodic")

# 6. Explain a belief
explanation = memory.explain(fact.id)
print(explanation["supported_by"])    # belief IDs that support this fact
print(explanation["contradicted_by"]) # belief IDs that contradict it
```

---

## Core Concepts

### Claims

Claims are immutable dataclasses representing beliefs:

```python
from recollectx import SemanticClaim, EpisodicClaim

# A fact: subject → predicate → object
fact = SemanticClaim(
    subject="user",
    predicate="prefers",
    object="dark mode",
    confidence=0.95,
    importance=0.8,        # 0.0 = trivial, 1.0 = critical
    evidence=("user said so in chat",),
)

# An event
event = EpisodicClaim(
    summary="User completed onboarding on March 28",
    confidence=0.99,
    importance=0.6,
)
```

**Common fields on all claims:**

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Auto-generated UUID |
| `confidence` | `float` | How certain is this belief? `[0.0, 1.0]` |
| `importance` | `float` | How useful for future conversations? `[0.0, 1.0]` |
| `evidence` | `tuple[str, ...]` | Supporting evidence strings |
| `support_count` | `int` | Times this belief has been reinforced |

---

### Memory

`Memory` is the central interface. It wires together storage, write policies, the belief graph, optional LLM updater, and confidence propagation:

```python
from recollectx import Memory
from recollectx.policies import MinConfidence, MinEvidence

memory = Memory(
    storage=store,
    write_policy=MinConfidence(0.5) & MinEvidence(1),  # composable policies
)
```

Key methods:

| Method | Description |
|---|---|
| `store(claim)` | Store a claim (applies policy first) |
| `retrieve(**kwargs)` | Query claims by type, confidence, etc. |
| `explain(belief_id)` | Get direct supports/contradictions for a belief |
| `explain_deep(belief_id, max_depth)` | Recursive justification chain |
| `explain_confidence_history(belief_id)` | Full confidence change history |
| `add_support(src_id, dst_id)` | Manually add a support relationship |

---

### Belief Graph

The belief graph tracks relationships between claims:

```python
# Relationships are created automatically by the LLM updater,
# or you can add them manually:
memory.add_support(src_id=claim_a.id, dst_id=claim_b.id)

# Deep explanation traverses the graph
result = memory.explain_deep(claim_b.id, max_depth=3)
print(result.root.belief)        # the target belief
print(result.total_nodes)        # how many nodes in the explanation
print(result.cycle_detected)     # True if circular reasoning was found
```

---

### Write Policies

Policies filter claims before they reach storage. Combine with `&`:

```python
from recollectx.policies import MinConfidence, MinEvidence

# Only store claims with confidence >= 0.6 and at least 1 piece of evidence
policy = MinConfidence(0.6) & MinEvidence(1)

memory = Memory(storage=store, write_policy=policy)
```

Custom policies implement a simple callable protocol:

```python
from recollectx.policies.base import Decision

class MyPolicy:
    def __call__(self, claim, memory) -> Decision:
        if claim.importance < 0.3:
            return Decision.REJECT
        return Decision.ACCEPT
```

---

### Confidence Propagation

When beliefs are related, confidence flows between them automatically:

```python
from recollectx import Memory, PropagationConfig

config = PropagationConfig(
    support_boost=0.10,        # supports increase confidence by this amount
    contradiction_decay=0.15,  # contradictions reduce confidence by this amount
)

memory = Memory(storage=store, propagation_config=config)
# Now adding support/contradiction edges will adjust confidence scores
```

---

## LLM Integration

### LLM Providers

```python
from recollectx.llm.providers import OpenAIProvider, AnthropicProvider, OllamaProvider

# OpenAI (requires re-collect[openai])
llm = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")

# Anthropic (requires re-collect[anthropic])
llm = AnthropicProvider(api_key="sk-ant-...", model="claude-3-5-haiku-latest")

# Ollama — local, free (requires re-collect[ollama])
llm = OllamaProvider(model="llama3", base_url="http://localhost:11434")
```

### LLM-Powered Memory Updates

The `MemoryUpdater` uses an LLM to make intelligent write decisions:

```python
from recollectx import Memory, MemoryUpdater

updater = MemoryUpdater(llm=llm, storage=store)
memory = Memory(storage=store, updater=updater)

# Now memory.store() will search for similar existing beliefs
# and decide: ADD, UPDATE, DELETE, or skip (NONE)
memory.store(new_claim)
```

### Claim Extraction from Text

```python
from recollectx.extractors import LLMExtractor

extractor = LLMExtractor(llm_provider=llm, min_confidence=0.5, max_claims_per_text=10)

claims = await extractor.extract("Alice loves hiking on weekends and prefers trail mix as a snack.")
for claim in claims:
    memory.store(claim)
```

---

## Vector Backends

Vector backends enable semantic (similarity) search over beliefs.

### FAISS (local, no server needed)

```python
from recollectx.storage.vector import FAISSBackend

vectors = FAISSBackend(embed_fn=my_embed_fn, dimension=384)
```

### Qdrant

```python
from recollectx.storage.vector import QdrantBackend

vectors = QdrantBackend(
    url="http://localhost:6333",
    collection_name="beliefs",
    embedding_fn=my_embed_fn,
    distance="cosine",
)
```

### Pinecone

```python
from recollectx.storage.vector import PineconeBackend

vectors = PineconeBackend(
    api_key="your-api-key",
    index_name="beliefs",
    embedding_fn=my_embed_fn,
)
```

---

## LangGraph Memory Agent

Answer questions by letting an agent retrieve from memory using tools:

```python
from recollectx.agents import MemoryAgent
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")
agent = MemoryAgent(memory=memory, llm=llm)

response = agent.answer("What food does the user like?")
print(response.answer)       # "The user loves pizza"
print(response.tools_used)   # ["search_memories", "get_facts_about"]
```

Available retrieval tools: `search_memories`, `get_recent_memories`, `get_facts_about`, `combine_facts`.

---

## Deduplication

Detect and merge duplicate or near-duplicate beliefs:

```python
from recollectx.deduplication import ClaimDeduplicator

deduplicator = ClaimDeduplicator(
    storage=store,
    llm=llm,
    similarity_threshold=0.85,
)

await deduplicator.run()  # scans storage and merges duplicates
```

---

## Architecture

```
re_collect/
├── claims.py          # EpisodicClaim, SemanticClaim dataclasses
├── memory.py          # Memory — main interface
├── updater.py         # LLM-powered write decisions
├── propagation.py     # Confidence propagation on relationships
├── state.py           # AgentState for dynamic state management
│
├── db/                # SQLAlchemy ORM (ClaimModel, BeliefEdgeModel, ConfidenceHistoryModel)
├── storage/           # MemoryStore (SQLite + vector), vector backends (FAISS, Qdrant, Pinecone)
├── graph/             # BeliefGraph, edges, deep explanation
├── policies/          # Composable write policies
├── llm/               # LLMProvider protocol + OpenAI/Anthropic/Ollama/OpenRouter/Local providers
├── extractors/        # LLM-powered claim extraction from text
├── agents/            # LangGraph-based Q&A agent + tools
└── deduplication/     # Embedding similarity + LLM merge decisions
```

---

## Requirements

- Python >= 3.12
- SQLAlchemy >= 2.0
- LangChain ecosystem (langchain-core, langchain-openai, langchain-anthropic, langgraph)

Optional dependencies vary by feature — see [Installation](#installation).

---

## License

MIT
