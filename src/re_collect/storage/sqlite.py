"""SQLite-based storage backend for persistent claim storage.

This module provides the SQLiteBackend class which stores claims in a
local SQLite database using aiosqlite for async support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from ..claims import Claim, EpisodicClaim, ProceduralClaim, SemanticClaim


# Valid columns for order_by to prevent SQL injection
VALID_ORDER_COLUMNS = frozenset({
    "created_at", "confidence", "last_reinforced_at", "support_count"
})

# Maps claim type string → the right class constructor
_CLAIM_CONSTRUCTORS: dict[str, type[Claim]] = {
    "": Claim,
    "episodic": EpisodicClaim,
    "semantic": SemanticClaim,
    "procedural": ProceduralClaim,
}

# SQL to create the claims table
_CREATE_CLAIMS_TABLE = """
CREATE TABLE IF NOT EXISTS claims (
    id                  TEXT PRIMARY KEY,
    type                TEXT NOT NULL,
    confidence          REAL NOT NULL,
    importance          REAL NOT NULL DEFAULT 0.5,
    evidence            TEXT NOT NULL,
    created_at          REAL NOT NULL,
    last_reinforced_at  REAL NOT NULL,
    support_count       INTEGER NOT NULL DEFAULT 1,
    summary             TEXT,
    subject             TEXT,
    predicate           TEXT,
    object              TEXT,
    trigger             TEXT,
    action              TEXT
)
"""


def _claim_to_row(claim: Claim) -> dict[str, Any]:
    """Convert a Claim object to a dict for database insertion."""
    row: dict[str, Any] = {
        "id": claim.id,
        "type": claim.type,
        "confidence": claim.confidence,
        "importance": claim.importance,
        "evidence": json.dumps(list(claim.evidence)),
        "created_at": claim.created_at,
        "last_reinforced_at": claim.last_reinforced_at,
        "support_count": claim.support_count,
        # type-specific fields default to None
        "summary": None,
        "subject": None,
        "predicate": None,
        "object": None,
        "trigger": None,
        "action": None,
    }

    if isinstance(claim, EpisodicClaim):
        row["summary"] = claim.summary
    elif isinstance(claim, SemanticClaim):
        row["subject"] = claim.subject
        row["predicate"] = claim.predicate
        row["object"] = claim.object
    elif isinstance(claim, ProceduralClaim):
        row["trigger"] = claim.trigger
        row["action"] = claim.action

    return row


def _row_to_claim(row: aiosqlite.Row) -> Claim:
    """Convert a database row back into the appropriate Claim subclass."""
    claim_type = row["type"]

    if claim_type not in _CLAIM_CONSTRUCTORS:
        raise ValueError(f"Unknown claim type: {claim_type!r}")

    common: dict[str, Any] = {
        "id": row["id"],
        "confidence": row["confidence"],
        "importance": row["importance"] if row["importance"] is not None else 0.5,
        "evidence": tuple(json.loads(row["evidence"])),
        "created_at": row["created_at"],
        "last_reinforced_at": row["last_reinforced_at"],
        "support_count": row["support_count"],
    }

    if claim_type == "episodic":
        return EpisodicClaim(**common, summary=row["summary"] or "")
    elif claim_type == "semantic":
        return SemanticClaim(
            **common,
            subject=row["subject"] or "",
            predicate=row["predicate"] or "",
            object=row["object"] or "",
        )
    elif claim_type == "procedural":
        return ProceduralClaim(
            **common,
            trigger=row["trigger"] or "",
            action=row["action"] or "",
        )
    else:
        return Claim(**common)


class SQLiteBackend:
    """SQLite storage backend for claims.

    Uses aiosqlite for fully async operation. Supports in-memory databases
    for testing via db_path=":memory:".

    Example:
        async with SQLiteBackend(":memory:") as storage:
            await storage.put(claim)
            result = await storage.get(claim.id)
    """

    def __init__(self, db_path: str | Path = "recollect.db") -> None:
        self._db_path = str(db_path)
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open connection and create tables."""
        if self._connection is not None:
            return
        self._connection = await aiosqlite.connect(self._db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA foreign_keys = ON")
        await self._connection.execute(_CREATE_CLAIMS_TABLE)
        await self._connection.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def __aenter__(self) -> SQLiteBackend:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _ensure_connected(self) -> aiosqlite.Connection:
        """Return connection or raise if not connected."""
        if self._connection is None:
            raise RuntimeError(
                "Not connected. Use 'await connect()' or the async context manager."
            )
        return self._connection

    async def put(self, claim: Claim) -> None:
        """Insert or replace a claim."""
        conn = self._ensure_connected()
        row = _claim_to_row(claim)
        await conn.execute(
            """
            INSERT OR REPLACE INTO claims (
                id, type, confidence, importance, evidence,
                created_at, last_reinforced_at, support_count,
                summary, subject, predicate, object, trigger, action
            ) VALUES (
                :id, :type, :confidence, :importance, :evidence,
                :created_at, :last_reinforced_at, :support_count,
                :summary, :subject, :predicate, :object, :trigger, :action
            )
            """,
            row,
        )
        await conn.commit()

    async def get(self, id: str) -> Claim | None:
        """Retrieve a claim by ID. Returns None if not found."""
        conn = self._ensure_connected()
        cursor = await conn.execute("SELECT * FROM claims WHERE id = ?", (id,))
        row = await cursor.fetchone()
        return _row_to_claim(row) if row else None

    async def query(
        self,
        type: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[Claim]:
        """Query claims with optional type and confidence filters."""
        conn = self._ensure_connected()

        conditions = ["confidence >= ?"]
        params: list[Any] = [min_confidence]

        if type is not None:
            conditions.append("type = ?")
            params.append(type)

        where = " AND ".join(conditions)
        cursor = await conn.execute(f"SELECT * FROM claims WHERE {where}", params)
        rows = await cursor.fetchall()
        return [_row_to_claim(row) for row in rows]

    async def delete(self, id: str) -> bool:
        """Delete a claim by ID. Returns True if deleted, False if not found."""
        conn = self._ensure_connected()
        cursor = await conn.execute("DELETE FROM claims WHERE id = ?", (id,))
        await conn.commit()
        return cursor.rowcount > 0

    async def count(self, type: str | None = None) -> int:
        """Count stored claims, optionally filtered by type."""
        conn = self._ensure_connected()

        if type is not None:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM claims WHERE type = ?", (type,)
            )
        else:
            cursor = await conn.execute("SELECT COUNT(*) FROM claims")

        row = await cursor.fetchone()
        return row[0] if row else 0
