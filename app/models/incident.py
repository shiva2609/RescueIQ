"""
SQLAlchemy ORM models for Incident and IncidentImage.
Python 3.9 compatible – uses Optional[T] instead of T | None.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    Boolean, Float, Index, Integer, String, Text,
    DateTime, ForeignKey,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)

    description: Mapped[str] = mapped_column(Text, nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, default="Other")
    reporter: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    damage_class: Mapped[str] = mapped_column(String(50), nullable=False)
    model_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_abstained: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    probabilities_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)

    severity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    urgency_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    impact_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    access_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    factor_breakdown_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    duplicate_of: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    image_hash: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text_embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    verification_status: Mapped[str] = mapped_column(String(20), nullable=False, default="Pending")
    incident_status: Mapped[str] = mapped_column(String(30), nullable=False, default="Reported")
    assigned_to: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    images: Mapped[List["IncidentImage"]] = relationship(
        "IncidentImage", back_populates="incident", cascade="all, delete-orphan"
    )
    audit_history: Mapped[List["StatusHistory"]] = relationship(
        "StatusHistory",
        foreign_keys="StatusHistory.incident_id",
        back_populates="incident",
        cascade="all, delete-orphan",
        order_by="StatusHistory.created_at.desc()",
    )
    assignments: Mapped[List["ResponderAssignment"]] = relationship(
        "ResponderAssignment",
        back_populates="incident",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_incidents_priority_score", "priority_score"),
        Index("ix_incidents_verification_status", "verification_status"),
        Index("ix_incidents_damage_class", "damage_class"),
        Index("ix_incidents_category", "category"),
        Index("ix_incidents_created_at", "created_at"),
    )


class IncidentImage(Base):
    __tablename__ = "incident_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    incident_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False
    )
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    image_hash: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    incident: Mapped["Incident"] = relationship("Incident", back_populates="images")
