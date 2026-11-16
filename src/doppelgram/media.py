"""Image normalization for the vision model and for forwarding cards to the owner."""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageOps


def to_jpeg(raw: bytes, max_side: int = 1536, quality: int = 90) -> bytes:
    """Any Telegram image (jpeg/png/webp, with or without alpha) -> bounded RGB JPEG."""
    with Image.open(io.BytesIO(raw)) as original:
        original.load()
        img = ImageOps.exif_transpose(original)  # phone photos: apply the EXIF rotation before it is dropped
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


MAX_DECODED_FRAMES = 120  # per target after a seek: a long gap between keyframes is not decoded to the end


def video_frames(raw: bytes, count: int = 2) -> list[bytes]:
    """A few still frames (JPEG) spread over a video or a video note, so the model sees the person, not only the
    tiny thumbnail Telegram attaches. Empty if the file cannot be decoded. Bounded work: after each seek at most
    MAX_DECODED_FRAMES frames are decoded; the nearest one so far is taken."""
    import av  # imported lazily, like for voice messages

    frames: list[bytes] = []
    try:
        with av.open(io.BytesIO(raw)) as container:
            if not container.streams.video:
                return []
            stream = container.streams.video[0]
            seconds = (float(stream.duration * stream.time_base) if stream.duration and stream.time_base
                       else (container.duration or 0) / 1_000_000)
            targets = [seconds * (k + 1) / (count + 1) for k in range(count)] if seconds > 0 else [0.0]
            for target in targets:
                if stream.time_base:
                    container.seek(int(target / stream.time_base), stream=stream)
                taken = None
                for n, frame in enumerate(container.decode(video=0)):
                    taken = frame
                    if frame.time is None or frame.time >= target - 0.05 or n + 1 >= MAX_DECODED_FRAMES:
                        break
                if taken is not None:
                    buf = io.BytesIO()
                    taken.to_image().convert("RGB").save(buf, format="JPEG", quality=88)
                    frames.append(buf.getvalue())
    except Exception:  # a corrupt or unsupported file: the caller falls back to the thumbnail
        return frames
    return frames
