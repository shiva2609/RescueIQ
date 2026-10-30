"""
SQLAlchemy database engine + session factory.

Supports SQLite (local dev) and PostgreSQL (production) transparently.
"""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from app.config import settings


# ── Engine ────────────────────────────────────────────────────────────────────
_connect_args: dict = {}
if settings.is_sqlite:
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    connect_args=_connect_args,
)

# Enable WAL mode and foreign-key enforcement for SQLite
if settings.is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ── Declarative base ──────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_db():
    """FastAPI dependency – yields a DB session and closes it on completion."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables that are not yet present (idempotent).
    In production, prefer Alembic migrations instead."""
    # Import all models so their metadata is registered before create_all
    from app.models import incident, duplicate, audit, assignment  # noqa: F401
    Base.metadata.create_all(bind=engine)
