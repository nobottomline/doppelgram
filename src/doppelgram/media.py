"""Image normalization for the vision model and for forwarding cards to the owner."""

from __future__ import annotations

import base64
import io

from PIL import Image


def to_jpeg(raw: bytes, max_side: int = 1536, quality: int = 90) -> bytes:
    """Any Telegram image (jpeg/png/webp, with or without alpha) -> bounded RGB JPEG."""
    with Image.open(io.BytesIO(raw)) as img:
        img.load()
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            out = Image.new("RGB", rgba.size, (255, 255, 255))
            out.paste(rgba, mask=rgba.split()[-1])
        else:
            out = img.convert("RGB")
    out.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def image_data_url(raw: bytes, max_side: int) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(to_jpeg(raw, max_side)).decode()
