"""User management router."""
from __future__ import annotations

import hashlib
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dependencies import DbSession
from app.models.user import User
from app.schemas.user import UserCreate, UserRead, UserUpdate

router = APIRouter(prefix="/api/users", tags=["Users"])


def hash_password(password: str) -> str:
    """Deterministic salted SHA-256 for password hashing."""
    salt = "rescueiq_salt_secure_2026"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


@router.post(
    "",
    response_model=UserRead,
    status_code=201,
    summary="Register a new user (citizen, responder, or coordinator)",
)
def create_user(payload: UserCreate, db: DbSession):
    # Check if email exists
    existing = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="User with this email already exists")

    user = User(
        id=str(uuid.uuid4()),
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        phone_number=payload.phone_number,
        agency_affiliation=payload.agency_affiliation,
        is_active=True,
        is_verified=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get(
    "",
    response_model=List[UserRead],
    summary="List all users with optional role filtering",
)
def list_users(
    db: DbSession,
    role: str = Query(default=None, description="Filter by user role"),
    limit: int = Query(default=50, ge=1, le=100),
):
    q = select(User).where(User.is_active == True)  # noqa: E712
    if role and role != "all":
        q = q.where(User.role == role)
    q = q.order_by(User.created_at.desc()).limit(limit)
    return db.execute(q).scalars().all()


@router.get(
    "/{user_id}",
    response_model=UserRead,
    summary="Get user profile by ID",
)
def get_user(user_id: str, db: DbSession):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
