"""Strict upload validation: MIME type, size cap, and image integrity."""

from __future__ import annotations

import io
from typing import Final

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES: Final[int] = 10 * 1024 * 1024
ALLOWED_MIME_TYPES: Final[set[str]] = {"image/jpeg", "image/png", "image/webp"}
_READ_CHUNK_BYTES: Final[int] = 64 * 1024

_EXTENSION_TO_MIME: Final[dict[str, str]] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class ImageValidationError(Exception):
    """Raised when an uploaded file fails validation."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def sniff_mime_type(data: bytes) -> str | None:
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _normalize_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower()


def _declared_mime(upload: UploadFile) -> str | None:
    declared = _normalize_content_type(upload.content_type)
    if declared in ALLOWED_MIME_TYPES:
        return declared
    if declared in {"application/octet-stream", "binary/octet-stream", None, ""}:
        name = (upload.filename or "").lower()
        for suffix, mime in _EXTENSION_TO_MIME.items():
            if name.endswith(suffix):
                return mime
        return None
    return declared


async def read_upload_capped(upload: UploadFile, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Read an upload in chunks and abort before exceeding max_bytes."""
    reported_size = getattr(upload, "size", None)
    if isinstance(reported_size, int) and reported_size > max_bytes:
        raise ImageValidationError(
            400,
            "FILE_TOO_LARGE",
            f"File exceeds the {max_bytes} byte limit.",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ImageValidationError(
                400,
                "FILE_TOO_LARGE",
                f"File exceeds the {max_bytes} byte limit.",
            )
        chunks.append(chunk)

    if total == 0:
        raise ImageValidationError(400, "INVALID_IMAGE", "Uploaded file is empty.")
    return b"".join(chunks)


def validate_image_bytes(data: bytes, declared_mime: str | None = None) -> str:
    """Validate MIME type and reject corrupted images without crashing the worker."""
    if not data:
        raise ImageValidationError(400, "INVALID_IMAGE", "Uploaded file is empty.")

    sniffed = sniff_mime_type(data)
    if sniffed is None:
        raise ImageValidationError(
            400,
            "INVALID_IMAGE",
            "Only JPEG, PNG, and WEBP images are allowed.",
        )

    if declared_mime and declared_mime not in ALLOWED_MIME_TYPES:
        raise ImageValidationError(
            400,
            "INVALID_IMAGE",
            f"MIME type {declared_mime!r} is not allowed. Use image/jpeg, image/png, or image/webp.",
        )

    if declared_mime in ALLOWED_MIME_TYPES and declared_mime != sniffed:
        raise ImageValidationError(
            400,
            "INVALID_IMAGE",
            f"Declared MIME type {declared_mime!r} does not match file contents ({sniffed}).",
        )

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError(
            422,
            "UNREADABLE_IMAGE",
            "The uploaded image is corrupted or unreadable.",
        ) from exc

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError(
            422,
            "UNREADABLE_IMAGE",
            "The uploaded image is corrupted or unreadable.",
        ) from exc

    return sniffed


async def validate_upload(upload: UploadFile) -> tuple[bytes, str]:
    data = await read_upload_capped(upload)
    mime = validate_image_bytes(data, declared_mime=_declared_mime(upload))
    return data, mime
