"""ORM model for the claims table."""

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from re_collect.db.database import Base


class ClaimModel(Base):
    """SQLAlchemy model for claims.

    All claim types (episodic, semantic, procedural) share one table.
    Type-specific columns are nullable
    """

    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    last_reinforced_at: Mapped[float] = mapped_column(Float, nullable=False)
    support_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # EpisodicClaim
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # SemanticClaim
    subject: Mapped[str | None] = mapped_column(String, nullable=True)
    predicate: Mapped[str | None] = mapped_column(String, nullable=True)
    object: Mapped[str | None] = mapped_column(String, nullable=True)

    # ProceduralClaim
    trigger: Mapped[str | None] = mapped_column(Text, nullable=True)
    action: Mapped[str | None] = mapped_column(Text, nullable=True)

