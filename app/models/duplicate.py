"""DuplicateLink and DuplicateGroup ORM models – Python 3.9 compatible."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


def _new_id():
    return str(uuid.uuid4())


class DuplicateGroup(Base):
    """
    Cluster grouping of multiple incidents determined to be the same real-world event.
    """
    __tablename__ = "duplicate_groups"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    canonical_incident_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False
    )
    group_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="Active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("ix_dup_groups_canonical", "canonical_incident_id"),
        Index("ix_dup_groups_status", "status"),
    )


class DuplicateLink(Base):
    __tablename__ = "duplicate_links"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_id
    )
    incident_a: Mapped[str] = mapped_column(
        String(36), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False
    )
    incident_b: Mapped[str] = mapped_column(
        String(36), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False
    )
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    hash_sim: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    image_sim: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    text_sim: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    geo_dist_m: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    time_sim: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recommendation: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="Pending")
    decided_by: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    detected_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_dup_links_status", "status"),
        Index("ix_dup_links_incident_a", "incident_a"),
        Index("ix_dup_links_incident_b", "incident_b"),
    )
