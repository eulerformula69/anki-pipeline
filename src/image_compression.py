from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image

MAX_CARD_IMAGE_BYTES = 10 * 1024
MAX_CARD_IMAGE_SIZE = (710, 400)


def compress_card_image(
    image: Image.Image,
    destination: Path,
    *,
    max_bytes: int = MAX_CARD_IMAGE_BYTES,
    max_size: tuple[int, int] = MAX_CARD_IMAGE_SIZE,
) -> None:
    """Save a JPEG that fits the card dimensions and hard byte limit."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    working = image.convert("RGB")
    working.thumbnail(max_size, Image.Resampling.LANCZOS)
    encoded = _encode_within_limit(working, max_bytes)

    while encoded is None and min(working.size) > 1:
        width, height = working.size
        scale = 0.85
        working = working.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        encoded = _encode_within_limit(working, max_bytes)

    if encoded is None:
        raise OSError(f"Не удалось сжать изображение до {max_bytes} байт.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)


def _encode_within_limit(image: Image.Image, max_bytes: int) -> bytes | None:
    best: bytes | None = None
    low, high = 5, 90
    while low <= high:
        quality = (low + high) // 2
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True, progressive=True)
        data = output.getvalue()
        if len(data) <= max_bytes:
            best = data
            low = quality + 1
        else:
            high = quality - 1
    return best
