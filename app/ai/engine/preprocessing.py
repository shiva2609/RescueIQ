"""
Image preprocessing and validation pipeline for the assessment engine.

Handles:
- File existence and readability check
- Magic-byte MIME validation (no spoofing via extension tricks)
- Image size / dimension limits
- Colour space normalization (always returns RGB PIL.Image)
- Resize strategies for CLIP (square centre-crop) and YOLO (letterbox)

All functions are pure – they do not modify the original file on disk.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageOps

# ── Limits (can be overridden by calling code) ────────────────────────────────
MAX_FILE_BYTES: int = 20 * 1024 * 1024       # 20 MB
MIN_DIMENSION:  int = 32                       # pixels; below this is useless
MAX_DIMENSION:  int = 4096                     # pixels on longest side

# ── Magic bytes ───────────────────────────────────────────────────────────────
_MAGIC: Dict[bytes, str] = {
    b"\xff\xd8\xff":        "image/jpeg",
    b"\x89PNG\r\n\x1a\n":  "image/png",
    b"RIFF":                "image/webp",   # confirmed only if bytes[8:12]==WEBP
    b"GIF87a":              "image/gif",
    b"GIF89a":              "image/gif",
    b"II*\x00":             "image/tiff",
    b"MM\x00*":             "image/tiff",
}

ALLOWED_MIME = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/tiff"
})


class PreprocessingError(ValueError):
    """Raised when image preprocessing fails a validation gate."""


def detect_mime(header: bytes) -> Optional[str]:
    for magic, mime in _MAGIC.items():
        if header[: len(magic)] == magic:
            if mime == "image/webp":
                return mime if header[8:12] == b"WEBP" else None
            return mime
    return None


def validate_raw_bytes(data: bytes) -> str:
    """Validate raw file bytes; return detected MIME type or raise PreprocessingError."""
    if len(data) == 0:
        raise PreprocessingError("Image file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise PreprocessingError(
            f"Image exceeds maximum allowed size of {MAX_FILE_BYTES // 1_048_576} MB."
        )
    mime = detect_mime(data[:12])
    if mime is None or mime not in ALLOWED_MIME:
        raise PreprocessingError(
            f"File does not match a recognised image format (JPEG/PNG/WebP/GIF/TIFF). "
            f"Detected: {mime!r}."
        )
    return mime


def load_and_validate_image(image_path: Path) -> Image.Image:
    """
    Load and validate an image from disk.

    Returns:
        PIL Image in RGB mode, already orientation-corrected.
    Raises:
        PreprocessingError on any validation failure.
    """
    if not image_path.exists():
        raise PreprocessingError(f"Image file not found: {image_path}")

    raw = image_path.read_bytes()
    validate_raw_bytes(raw)

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)   # correct EXIF orientation
        img = img.convert("RGB")
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"Cannot decode image: {exc}") from exc

    w, h = img.size
    if min(w, h) < MIN_DIMENSION:
        raise PreprocessingError(
            f"Image dimensions {w}×{h} px are too small (minimum {MIN_DIMENSION} px)."
        )

    return img


def resize_for_clip(img: Image.Image, size: int = 224) -> Image.Image:
    """
    Resize image for CLIP inference using centred-crop strategy.
    Matches the preprocessing expected by openai/clip ViT models.
    """
    # Scale so shorter side = size, then centre-crop
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Centre crop
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def resize_for_yolo(img: Image.Image, target: int = 640) -> Image.Image:
    """
    Resize image for YOLO inference using letterboxing (preserve aspect ratio).
    Fills with grey (#114, #114, #114) to maintain pixel statistics.
    """
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target, target), (114, 114, 114))
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    return canvas


def safe_thumbnail(img: Image.Image, max_dim: int = MAX_DIMENSION) -> Image.Image:
    """Downscale oversized images without distortion before any further processing."""
    if max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img


def get_image_metadata(img: Image.Image) -> Dict:
    """Return human-readable metadata about a loaded PIL image."""
    return {
        "width": img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "aspect_ratio": round(img.size[0] / img.size[1], 3),
    }
