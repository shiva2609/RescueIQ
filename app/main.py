"""
RescueIQ – FastAPI application entry point (v2.0)

Architecture:
  app/config.py        – Settings (env vars / .env)
  app/database.py      – SQLAlchemy engine + session
  app/models/          – ORM models
  app/schemas/         – Pydantic request/response schemas
  app/services/        – Business logic (pure Python)
  app/routers/         – HTTP layer (thin, delegates to services)
  app/ai/              – EfficientNet-B0 + MiniLM wrappers

Backward compatibility:
  All existing endpoints (/api/incidents, /api/stats, /api/health,
  /api/duplicates, /api/audit-logs, /api/analyze/image, /api/demo-seed)
  are preserved with identical request / response shapes.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import analysis, assess, audit, demo, duplicates, health, incidents, users


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB tables on startup (Alembic handles migrations in prod)."""
    init_db()
    yield


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a JSON error for any unhandled exception (prevents stack traces leaking)."""
    import traceback, logging
    logging.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(incidents.router)
app.include_router(duplicates.router)
app.include_router(analysis.router)
app.include_router(audit.router)
app.include_router(demo.router)
app.include_router(users.router)
app.include_router(assess.router)


# ── Static files + frontend ───────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")


@app.get("/", include_in_schema=False)
def root():
    """Serve the single-page frontend application."""
    return FileResponse(settings.STATIC_DIR / "index.html")
