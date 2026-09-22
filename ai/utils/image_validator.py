import io
from fastapi import HTTPException, UploadFile
from PIL import Image

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB limit


async def read_upload_capped(upload: UploadFile, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Reads raw file bytes and raises HTTP 413 if payload exceeds max_bytes."""
    content = await upload.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File '{upload.filename}' exceeds maximum allowed size of {max_bytes // (1024 * 1024)}MB."
        )
    return content


async def validate_upload(upload: UploadFile) -> tuple[bytes, str]:
    """Validates file size, content-type/magic bytes, and PIL readability."""
    data = await read_upload_capped(upload)

    # 1. Basic Content-Type check
    content_type = upload.content_type or ""
    if content_type.lower() not in ALLOWED_MIME_TYPES:
        # Fallback verification via PIL format
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
            fmt = (img.format or "").lower()
            if fmt == "jpeg":
                content_type = "image/jpeg"
            elif fmt == "png":
                content_type = "image/png"
            elif fmt == "webp":
                content_type = "image/webp"
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported image format: {fmt}")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{content_type}'. Must be JPEG, PNG, or WebP."
            )
    else:
        # Verify valid image buffer
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
        except Exception:
            raise HTTPException(status_code=422, detail=f"Corrupted or unreadable image file: {upload.filename}")

    return data, content_type