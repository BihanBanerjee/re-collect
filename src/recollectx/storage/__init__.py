"""Storage layer for the recollect memory system.

- MemoryStore: Combines SQLite (via SQLAlchemy) with vector search
- VectorBackend: Protocol for vector storage (in .vector submodule)
"""

from .memory_store import MemoryStore, belief_to_text

__all__ = [
    "MemoryStore",
    "belief_to_text",
]
