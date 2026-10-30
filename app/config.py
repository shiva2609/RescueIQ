"""
RescueIQ – Application settings.

Reads DATABASE_URL from the environment; falls back to a local SQLite file
so the app works out-of-the-box without PostgreSQL installed.

PostgreSQL (production):
    export DATABASE_URL="postgresql+psycopg2://user:pass@host/dbname"

SQLite (local dev / CI):
    DATABASE_URL is not set → uses damage_mapper.db in the project root.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = ROOT / "uploads"
STATIC_DIR = ROOT / "static"
DB_DEFAULT = f"sqlite:///{ROOT / 'damage_mapper.db'}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Paths ─────────────────────────────────────────────────────────────
    ROOT_DIR: Path = ROOT
    STATIC_DIR: Path = STATIC_DIR
    UPLOAD_DIR: Path = UPLOADS_DIR

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = DB_DEFAULT
    # For SQLite we need check_same_thread=False; ignored for PostgreSQL
    DB_ECHO: bool = False                  # set True to log all SQL

    # ── Image uploads ────────────────────────────────────────────────────
    MAX_IMAGE_BYTES: int = 10 * 1024 * 1024   # 10 MB
    ALLOWED_MIME: frozenset[str] = frozenset({
        "image/jpeg", "image/png", "image/webp"
    })
    MAX_IMAGE_DIMENSION: int = 1600           # pixels on longest side

    # ── Classifier ───────────────────────────────────────────────────────
    ABSTENTION_THRESHOLD: float = 0.50

    # ── Priority engine ──────────────────────────────────────────────────
    WEIGHT_SEVERITY: float = 0.40
    WEIGHT_URGENCY: float = 0.25
    WEIGHT_IMPACT: float = 0.20
    WEIGHT_ACCESS: float = 0.15

    # ── Pagination ───────────────────────────────────────────────────────
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    # ── App meta ─────────────────────────────────────────────────────────
    APP_TITLE: str = "RescueIQ"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = (
        "AI-assisted disaster damage assessment and response triage. "
        "Powered by EfficientNet-B0 vision classifier and MiniLM text embeddings."
    )

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")


settings = Settings()

# Ensure upload directory exists at import time
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
