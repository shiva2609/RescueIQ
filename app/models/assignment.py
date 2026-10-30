"""ResponderAssignment ORM model – Python 3.9 compatible."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class ResponderAssignment(Base):
    __tablename__ = "responder_assignments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    incident_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False
    )
    team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    assigned_by: Mapped[str] = mapped_column(String(200), nullable=False, default="Coordinator")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="Active")
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    incident: Mapped["Incident"] = relationship(  # noqa: F821
        "Incident", back_populates="assignments"
    )

    __table_args__ = (
        Index("ix_assignments_incident_id", "incident_id"),
        Index("ix_assignments_status", "status"),
    )
