"""User Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=200)
    role: str = Field(default="citizen", description="citizen | responder | coordinator | admin")
    phone_number: Optional[str] = None
    agency_affiliation: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="Plaintext password for registration")


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    agency_affiliation: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserRead(UserBase):
    id: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
