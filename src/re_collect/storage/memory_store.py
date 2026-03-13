"""MemoryStore — the main storage layer for re-collect.

Combines SQLite (authoritative storage via SQLAlchemy) with vector search
(semantic recall) into a single interface. All CRUD goes through SQLite;
vectors accelerate semantic queries.

Architecture:
    User query
        |
    Vector search (candidate belief IDs)
        |
    Storage fetch (authoritative beliefs)
        |
    Graph filtering (contradictions)
        |
    Confidence + decay filtering
        |
    Final belief set
"""

import logging
import time
from typing import Any

from sqlalchemy.orm import Session

from ..claims import Claim, EpisodicClaim, SemanticClaim
# from ..db.
