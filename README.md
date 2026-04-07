# recollectx

[![PyPI version](https://img.shields.io/pypi/v/recollectx.svg)](https://pypi.org/project/recollectx/)
[![Python](https://img.shields.io/pypi/pyversions/recollectx.svg)](https://pypi.org/project/recollectx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-208%20passing-brightgreen.svg)](#)

**Belief-centric memory layer for agentic AI systems.**

Most memory libraries store what a user said. recollectx stores what the user *believes* — and tracks *why*.

Every piece of information is a typed claim with a confidence score, supporting evidence, and explicit relationships to other beliefs. Contradictions reduce confidence automatically. Supports boost it. The entire justification chain is inspectable at any time.

```bash
pip install recollectx
```

---

## Why recollectx?

| Feature | recollectx | mem0 | Zep | LangMem |
|---|---|---|---|---|
| Typed claims (episodic / semantic) | ✅ | ❌¹ | ❌ | ❌¹ |
| Epistemic belief graph (supports / contradicts / derives) | ✅ | ❌² | ❌³ | ❌ |
| Confidence propagation on contradictions | ✅ | ❌ | ❌ | ❌ |
| Importance-modulated temporal decay | ✅ | ❌ | ❌ | ❌ |
| Pure Python library (no service required) | ✅ | ✅ | ❌ | ✅ |
| Pluggable LLM + vector backends | ✅ | ✅ | ✅ | ✅ |
| Deep justification chains | ✅ | ❌ | ❌ | ❌ |

¹ mem0 and LangMem label memory types (episodic/semantic) but store unstructured text — no enforced schema or typed claim objects.
² mem0 has an optional entity-relation graph (Mem0g) but edges are domain labels (`works_at`, `lives_in`), not epistemic edges (`supports`, `contradicts`, `derives`).
³ Zep/Graphiti has a temporal knowledge graph but requires Neo4j or FalkorDB as a separate service — no in-process pure Python mode.

**recollectx is the only Python memory library that combines typed claim objects, an epistemic belief graph, and confidence propagation — as a pure library with no server required.** mem0's graph tracks entity relationships, not belief justifications. Zep's graph needs a database server. Neither propagates confidence when beliefs contradict.

---

## Installation

```bash
# Core (SQLite storage only, no LLM required)
pip install recollectx

# With local embeddings + FAISS (no API keys needed)
pip install "recollectx[local]"

# OpenAI
pip install "recollectx[openai]"

# Anthropic Claude
pip install "recollectx[anthropic]"

# Local Ollama models
pip install "recollectx[ollama]"

# Qdrant vector backend
pip install "recollectx[qdrant]"

# Pinecone vector backend
pip install "recollectx[pinecone]"

# Everything
pip install "recollectx[all]"
```

---

## Quick Start

```python
from recollectx import Memory, SemanticClaim, EpisodicClaim
from recollectx.db import SessionLocal, create_tables
from recollectx.storage import MemoryStore
from recollectx.storage.vector import FAISSBackend
from sentence_transformers import SentenceTransformer

# Setup
create_tables()
db = SessionLocal()
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = FAISSBackend(embed_fn=model.encode, dimension=384)
store = MemoryStore(db, vectors)
memory = Memory(storage=store)

# Store beliefs
memory.store(SemanticClaim(
    subject="user", predicate="prefers", object="dark mode",
    confidence=0.95, importance=0.8,
    evidence=("user said so in chat",),
))

memory.store(EpisodicClaim(
    summary="User mentioned they hate early meetings",
    confidence=0.85, importance=0.6,
))

# Retrieve
facts = memory.retrieve(type="semantic")
episodes = memory.retrieve(type="episodic")

# Explain a belief
explanation = memory.explain(facts[0].id)
print(explanation["supported_by"])    # IDs of supporting beliefs
print(explanation["contradicted_by"]) # IDs of contradicting beliefs
```

---

## Core Concepts

### Claims

Claims are immutable typed beliefs. Two types:

```python
from recollectx import SemanticClaim, EpisodicClaim

# Semantic: stable facts as subject → predicate → object triples
# "The user prefers Python over JavaScript"
fact = SemanticClaim(
    subject="user",
    predicate="prefers",
    object="Python",
    confidence=0.9,     # How certain is this? [0.0, 1.0]
    importance=0.8,     # How useful for future context? [0.0, 1.0]
    evidence=("user said so explicitly",),
)

# Episodic: time-bound events
# "User was debugging a FastAPI issue last Tuesday"
event = EpisodicClaim(
    summary="User was debugging a FastAPI issue",
    confidence=0.85,
    importance=0.5,
)
```

**All claims share these fields:**

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Auto-generated UUID |
| `confidence` | `float` | Certainty: `[0.0, 1.0]` |
| `importance` | `float` | Relevance to future conversations: `[0.0, 1.0]` |
| `evidence` | `tuple[str, ...]` | Supporting evidence strings |
| `created_at` | `float` | Unix timestamp |
| `support_count` | `int` | Times this belief has been reinforced |

---

### Memory

`Memory` is the main interface. It wires together storage, write policies, the belief graph, LLM updater, and confidence propagation:

```python
from recollectx import Memory, MemoryUpdater, PropagationConfig
from recollectx.policies import MinConfidence, MinEvidence

memory = Memory(
    storage=store,
    write_policy=MinConfidence(0.6) & MinEvidence(1),
    updater=MemoryUpdater(store=store, llm=llm),
    propagation_config=PropagationConfig(
        support_boost=0.10,
        contradiction_decay=0.15,
    ),
)
```

| Method | Description |
|---|---|
| `store(claim)` | Store a claim (applies policy, updater, propagation) |
| `retrieve(**kwargs)` | Query claims by `type`, `min_confidence` |
| `explain(belief_id)` | Direct supports and contradictions |
| `explain_deep(belief_id, max_depth)` | Recursive justification chain |
| `explain_confidence_history(belief_id)` | Full audit log of confidence changes |
| `add_support(src_id, dst_id)` | Manually add a support relationship |

---

### Belief Graph

The belief graph tracks typed relationships between claims. When you add a contradiction, confidence decays automatically. When you add support, confidence increases.

```python
# Contradiction: old belief conflicts with new evidence
old = SemanticClaim(subject="user", predicate="works_as", object="student", confidence=0.9)
new = SemanticClaim(subject="user", predicate="works_as", object="engineer", confidence=0.9)
memory.store(old)
memory.store(new)
# If the LLM updater detects the contradiction, old.confidence drops by 0.15 automatically

# Support: manually link two beliefs
memory.add_support(src_id=evidence_claim.id, dst_id=fact_claim.id)
# fact_claim.confidence increases by 0.10

# Deep explanation: trace why a belief exists
result = memory.explain_deep(fact_claim.id, max_depth=3)
print(result.root.belief)    # the belief
print(result.total_nodes)    # how many beliefs in the chain
print(result.cycle_detected) # True if circular reasoning detected
```

Edge types: `supports`, `contradicts`, `derives`, `similar`

---

### Temporal Decay

recollectx uses **importance-modulated exponential decay** at retrieval time. Episodic memories fade faster than semantic ones. High-importance claims decay near-zero regardless of type.

```
score = exp(-λ × hours_elapsed × (1 - importance))

Episodic  λ = 0.001  (faster — events are time-bound)
Semantic  λ = 0.0001 (slower  — facts are durable)
```

Enable with `recency_bias > 0` in semantic search:

```python
# Recency-weighted retrieval
results = store.semantic_query(
    "user work preferences",
    recency_bias=0.1,          # enables decay re-ranking
    episodic_ttl_days=90,      # optional: drop episodic claims older than 90 * (1 + importance) days
    k=10,
)
```

A claim with `importance=0.9` after 6 weeks retains ~99% of its retrieval score. A low-importance episode (`importance=0.1`) retains ~50%. Semantic facts are nearly unaffected.

---

### Write Policies

Policies filter claims before storage. Compose with `&`:

```python
from recollectx.policies import MinConfidence, MinEvidence

# Reject claims with confidence < 0.6 or fewer than 1 evidence string
policy = MinConfidence(0.6) & MinEvidence(1)

memory = Memory(storage=store, write_policy=policy)
```

Custom policies:

```python
from recollectx.policies.base import Decision

class ImportanceGate:
    def __call__(self, claim, memory) -> Decision:
        return Decision.ACCEPT if claim.importance >= 0.4 else Decision.REJECT
```

---

### Confidence Propagation

```python
from recollectx import PropagationConfig

config = PropagationConfig(
    support_boost=0.10,           # +0.10 when supported
    contradiction_decay=0.15,     # -0.15 when contradicted
    min_confidence=0.01,          # floor
    max_confidence=0.99,          # ceiling
    symmetric_contradiction=True, # both claims decay, not just the target
)
```

Every confidence change is recorded as a `ConfidenceChangeEvent` with `claim_id`, `old_confidence`, `new_confidence`, `change_type`, and `caused_by_id`. Full audit trail available via `memory.explain_confidence_history(claim_id)`.

---

## LLM Integration

### LLM Providers

```python
from recollectx.llm.providers import OpenAIProvider, AnthropicProvider, OllamaProvider

# OpenAI  (pip install "recollectx[openai]")
llm = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")

# Anthropic  (pip install "recollectx[anthropic]")
llm = AnthropicProvider(api_key="sk-ant-...", model="claude-3-5-haiku-latest")

# Local Ollama — free, no API key  (pip install "recollectx[ollama]")
llm = OllamaProvider(model="llama3", base_url="http://localhost:11434")
```

All providers implement `LLMProvider` — a simple two-method protocol. Bring your own by implementing `generate()` and optionally `generate_structured()`.

---

### Claim Extraction from Text

```python
from recollectx.extractors import LLMExtractor

extractor = LLMExtractor(llm_provider=llm, min_confidence=0.5, max_claims_per_text=10)

# Single text
claims = extractor.extract("I love hiking on weekends and prefer trail mix as a snack.")
for claim in claims:
    memory.store(claim)

# Batch
all_claims = extractor.extract_batch([
    "I just started a new job as an ML engineer.",
    "I prefer Python over JavaScript for backend work.",
])
```

The extractor classifies each claim as `SemanticClaim` (stable fact) or `EpisodicClaim` (time-bound event) automatically.

---

### LLM-Powered Memory Updates

The `MemoryUpdater` makes intelligent write decisions by searching for similar existing claims first:

```python
from recollectx import MemoryUpdater

updater = MemoryUpdater(store=store, llm=llm, similarity_k=5)
memory = Memory(storage=store, updater=updater)

# memory.store() now runs an LLM decision pipeline:
# ADD    — genuinely new information, store as new claim
# UPDATE — adds detail to an existing fact, merge into target
# DELETE — old fact is obsolete, remove old and store new
# NONE   — exact duplicate, skip
memory.store(new_claim)
```

The updater also auto-detects relationships (`supports`, `contradicts`, `derives`) between the new claim and existing ones, and creates belief graph edges.

---

## Vector Backends

### FAISS (local, no server)

```python
from recollectx.storage.vector import FAISSBackend
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = FAISSBackend(embed_fn=model.encode, dimension=384)
```

### Qdrant (production)

```python
from recollectx.storage.vector import QdrantBackend

vectors = QdrantBackend(
    url="http://localhost:6333",
    collection_name="beliefs",
    embedding_fn=model.encode,
    distance="cosine",
)
```

### Pinecone (managed)

```python
from recollectx.storage.vector import PineconeBackend

vectors = PineconeBackend(
    api_key="your-api-key",
    index_name="beliefs",
    embedding_fn=model.encode,
)
```

---

## Memory Agent (LangGraph)

Answer questions by letting an agent retrieve from memory using tool-based reasoning:

```python
from recollectx.agents import MemoryAgent
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-haiku-latest")
agent = MemoryAgent(memory=memory, llm=llm)

response = agent.answer("What programming language does the user prefer?")
print(response.answer)      # "The user prefers Python"
print(response.tools_used)  # ["get_facts_about", "search_memories"]
```

Available tools: `search_memories`, `get_recent_memories`, `get_facts_about`, `combine_facts`

---

## Deduplication

Detect and merge near-duplicate beliefs using embedding similarity + LLM merge decisions:

```python
from recollectx.deduplication import ClaimDeduplicator

deduplicator = ClaimDeduplicator(
    storage=store,
    embedding_provider=embedding_provider,
    llm_provider=llm,
    similarity_threshold=0.85,
)

result = deduplicator.process(new_claim)
print(result.action)  # "ADD", "UPDATE", "DELETE", or "NONE"
print(result.reason)  # LLM's explanation
```

---

## Full Example

```python
from recollectx import (
    Memory, MemoryUpdater, PropagationConfig,
    SemanticClaim, EpisodicClaim,
)
from recollectx.agents import MemoryAgent
from recollectx.db import SessionLocal, create_tables
from recollectx.extractors import LLMExtractor
from recollectx.llm.providers import OllamaProvider
from recollectx.policies import MinConfidence
from recollectx.storage import MemoryStore
from recollectx.storage.vector import FAISSBackend
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

# 1. Setup
create_tables()
db = SessionLocal()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = FAISSBackend(embed_fn=embed_model.encode, dimension=384)
store = MemoryStore(db, vectors)
llm = OllamaProvider(model="llama3")

# 2. Memory with all features
memory = Memory(
    storage=store,
    write_policy=MinConfidence(0.5),
    updater=MemoryUpdater(store=store, llm=llm),
    propagation_config=PropagationConfig(support_boost=0.10, contradiction_decay=0.15),
)

# 3. Extract claims from conversation
extractor = LLMExtractor(llm, min_confidence=0.5)
claims = extractor.extract("I started a new job as an ML engineer. I love Python.")
for claim in claims:
    memory.store(claim)

# 4. Query with temporal decay
results = store.semantic_query(
    "user career and skills",
    recency_bias=0.1,
    episodic_ttl_days=90,
    k=5,
)

# 5. Explain a belief
explanation = memory.explain_deep(results[0].id, max_depth=3)
print(f"Belief: {results[0]}")
print(f"Supported by {len(explanation.root.supported_by)} claims")

# 6. Q&A agent
agent = MemoryAgent(memory=memory, llm=ChatOllama(model="llama3"))
response = agent.answer("What does the user do for work?")
print(response.answer)  # "The user works as an ML engineer"
```

---

## Architecture

```
recollectx/
├── claims.py          # EpisodicClaim, SemanticClaim — typed immutable beliefs
├── memory.py          # Memory — main interface (store, retrieve, explain)
├── updater.py         # LLM-powered ADD / UPDATE / DELETE decisions
├── propagation.py     # Confidence propagation on supports / contradictions
├── state.py           # AgentState for dynamic state management
│
├── db/                # SQLAlchemy ORM models + session management
├── storage/           # MemoryStore (SQLite + vector) + FAISS / Qdrant / Pinecone
├── graph/             # BeliefGraph, edges, deep explanation traversal
├── policies/          # Composable write policies (MinConfidence, MinEvidence)
├── llm/               # LLMProvider protocol + OpenAI / Anthropic / Ollama / OpenRouter
├── extractors/        # LLM-powered claim extraction from raw text
├── agents/            # LangGraph ReAct agent + retrieval tools
└── deduplication/     # Embedding similarity + LLM merge decisions
```

---

## Requirements

- Python ≥ 3.12
- SQLAlchemy ≥ 2.0
- LangChain ecosystem (langchain-core, langgraph)

All other dependencies are optional — install only what you need via extras.

---

## License

MIT — [Bihan Banerjee](https://github.com/BihanBanerjee)
