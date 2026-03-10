"""ORM model for the belief_edges table."""

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from re_collect.db.database import Base


class BeliefEdgeModel(Base):
    """SQLAlchemy model for belief graph edges.

    Tracks relationships between claims: supports, contradicts, derives, similar.
    """

    __tablename__ = "belief_edges"
    __table_args__ = (
        UniqueConstraint("src_id", "dst_id", "relation", name="uq_edge"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    src_id: Mapped[str] = mapped_column(
        String, ForeignKey("claims.id", ondelete="CASCADE"), nullable=False
    )
    dst_id: Mapped[str] = mapped_column(
        String, ForeignKey("claims.id", ondelete="CASCADE"), nullable=False
    )
    relation: Mapped[str] = mapped_column(String, nullable=False)
