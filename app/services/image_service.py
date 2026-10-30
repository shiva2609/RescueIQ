"""
Image upload service: validate, sanitise, resize, hash.

Validates:
  - Content-Type header
  - Magic-byte signature (prevents MIME spoofing)
  - File size limit
  - PIL image integrity
Optionally resizes to MAX_IMAGE_DIMENSION on longest side.
Returns SHA-256 digest and dHash.
"""
from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.config import settings
from app.ai.duplicates import compute_dhash


# Magic-byte signatures (first N bytes) ──────────────────────────────────────
_MAGIC: dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",      # RIFF????WEBP
}

_EXT_MAP = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _detect_mime(data: bytes) -> str | None:
    for magic, mime in _MAGIC.items():
        if data[: len(magic)] == magic:
            if mime == "image/webp":
                # Confirm WEBP marker at byte 8
                if len(data) >= 12 and data[8:12] == b"WEBP":
                    return mime
                return None
            return mime
    return None


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def validate_and_save_upload(
    upload: UploadFile,
    dest_dir: Path,
    stem: str,
) -> dict:
    """
    Validate uploaded image, optionally resize, save to dest_dir/stem+ext.

    Returns:
        image_path (str): relative path for storage  e.g. "uploads/abc.jpg"
        mime_type (str)
        file_size_bytes (int)
        sha256 (str)
        dhash (int | None)
    """
    # 1. Content-Type header check
    ct = (upload.content_type or "").split(";")[0].strip()
    if ct not in settings.ALLOWED_MIME:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported image type '{ct}'. Allowed: {sorted(settings.ALLOWED_MIME)}",
        )

    # 2. Read raw bytes
    raw = await upload.read()
    if len(raw) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {settings.MAX_IMAGE_BYTES // 1_048_576} MB limit",
        )

    # 3. Magic-byte validation (prevents MIME spoofing)
    detected = _detect_mime(raw)
    if detected is None:
        raise HTTPException(
            status_code=422,
            detail="File does not match a recognised JPEG / PNG / WEBP signature",
        )

    # 4. PIL integrity check + optional resize
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()                       # raises on corrupt data
        img = Image.open(io.BytesIO(raw)) # re-open after verify (verify() closes it)

        # Resize if needed
        w, h = img.size
        max_dim = settings.MAX_IMAGE_DIMENSION
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            fmt = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[detected]
            img.save(buf, format=fmt, quality=85)
            raw = buf.getvalue()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded file is not a valid image: {exc}",
        )

    # 5. Save to disk
    ext = _EXT_MAP[detected]
    filename = f"{stem}{ext}"
    dest_path = dest_dir / filename
    dest_path.write_bytes(raw)

    # 6. Compute hashes
    sha = sha256_of(raw)
    try:
        dhash = compute_dhash(dest_path)
    except Exception:
        dhash = None

    return {
        "image_path": f"uploads/{filename}",
        "mime_type": detected,
        "file_size_bytes": len(raw),
        "sha256": sha,
        "dhash": dhash,
    }
