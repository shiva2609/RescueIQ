"""Common Pydantic response models used across schemas."""
from __future__ import annotations

from typing import Generic, TypeVar, Any, Optional, List, Dict
from pydantic import BaseModel

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard paginated list response."""
    items: List[T]
    total: int
    page: int
    limit: int
    pages: int

    @classmethod
    def build(cls, items: List[T], total: int, page: int, limit: int) -> "PaginatedResponse[T]":
        import math
        return cls(
            items=items,
            total=total,
            page=page,
            limit=limit,
            pages=max(1, math.ceil(total / limit)) if total else 1,
        )


class ErrorDetail(BaseModel):
    detail: str
    field: Optional[str] = None


class SuccessResponse(BaseModel):
    status: str = "ok"
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
