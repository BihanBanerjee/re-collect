"""Database connection and session management.

Uses lazy initialization — engine is created only when first accessed.
SQLite only (no PostgreSQL).
"""

import os
from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# Global instances (lazy initialized)
_engine = None
_SessionLocal = None


def get_engine(db_path: str | None = None) -> Engine:
    """Get or create the database engine.

    Args:
        db_path: Path to SQLite database file. If None, uses
                 ~/.recollect/memory.db as default.
    """
    global _engine
    if _engine is None:
        if db_path is None:
            db_dir = os.path.expanduser("~/.recollect")
            os.makedirs(db_dir, exist_ok=True)
            db_path = f"{db_dir}/memory.db"

        database_url = f"sqlite:///{db_path}" if db_path != ":memory:" else "sqlite://"
        _engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )
    return _engine


def get_session_local(db_path: str | None = None) -> Any:
    """Get or create the SessionLocal factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autoflush=False,
            autocommit=False,
            bind=get_engine(db_path),
        )
    return _SessionLocal


def SessionLocal(db_path: str | None = None) -> Session:
    """Create a new database session."""
    session: Session = get_session_local(db_path)()
    return session


def get_db(db_path: str | None = None) -> Generator[Session, None, None]:
    """Database session generator. Yields a session and ensures it's closed."""
    db = SessionLocal(db_path)
    try:
        yield db
    finally:
        db.close()


def create_tables(db_path: str | None = None) -> None:
    """Create all database tables. Safe to call multiple times."""
    import recollectx.db.models  # noqa: F401 — ensure models are registered
    Base.metadata.create_all(bind=get_engine(db_path))


def reset_engine() -> None:
    """Reset engine and session factory. Useful for testing."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
