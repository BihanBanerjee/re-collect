"""ORM model for the confidence_history table."""

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from re_collect.db.database import Base


class ConfidenceHistoryModel(Base):
    """SQLAlchemy model for tracking confidence changes over time."""

    __tablename__ = "confidence_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(
        String, ForeignKey("claims.id", ondelete="CASCADE"), nullable=False
    )
    old_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    new_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    change_type: Mapped[str] = mapped_column(String, nullable=False)
    caused_by_id: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
