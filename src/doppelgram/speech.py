"""Speech to text for voice messages and video notes, through an audio-capable chat model.

The audio is decoded locally (any container Telegram uses: OGG/Opus voice, MP4 video notes), resampled to
16 kHz mono WAV and sent to the configured model as `input_audio` on the same OpenAI-compatible endpoint
(OpenRouter by default). Nothing is stored: the caller keeps transcripts in memory only.
"""

from __future__ import annotations

import base64
import io
import logging
import wave
from dataclasses import dataclass, replace

from .config import LLMConfig
from .llm import LLMClient, LLMError

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

PROMPT = ("Transcribe this voice message word for word, in the language it is spoken in. Do not translate, "
          "summarise or add anything. If there is no speech, return an empty text. "
          'Answer with JSON only: {"text": "<transcript>"}')


class SpeechError(RuntimeError):
    pass


@dataclass(frozen=True)
class Transcript:
    text: str
    seconds: float
    cost_usd: float


def to_wav(raw: bytes, max_seconds: float) -> tuple[bytes, float]:
    """Decode the first audio stream of any media file to 16 kHz mono 16-bit WAV, cut at max_seconds."""
    import av  # imported lazily: only needed when speech is used

    out = io.BytesIO()
    seconds = 0.0
    try:
        with av.open(io.BytesIO(raw)) as container:
            if not container.streams.audio:
                raise SpeechError("no audio track")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            with wave.open(out, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                limit = int(max_seconds * SAMPLE_RATE)
                written = 0
                for frame in container.decode(audio=0):
                    for chunk in resampler.resample(frame):
                        pcm = bytes(chunk.planes[0])[: chunk.samples * 2]
                        take = min(chunk.samples, limit - written)
                        wav.writeframes(pcm[: take * 2])
                        written += take
                        if written >= limit:
                            break
                    if written >= limit:
                        break
                seconds = written / SAMPLE_RATE
    except SpeechError:
        raise
    except Exception as exc:  # corrupt or unsupported media
        raise SpeechError(f"cannot decode audio: {exc}") from exc
    if seconds <= 0:
        raise SpeechError("empty audio")
    return out.getvalue(), seconds


class Transcriber:
    """Transcribes audio with an audio-capable model on the main LLM endpoint."""

    def __init__(self, llm_cfg: LLMConfig, model: str, fallback_models: list[str] | None = None,
                 max_seconds: float = 180.0, client: LLMClient | None = None):
        extra = dict(llm_cfg.extra_body)
        if "openrouter.ai" in llm_cfg.base_url:
            extra.setdefault("reasoning", {"enabled": False})  # thinking tokens cost and add nothing here
        cfg = replace(llm_cfg, model=model, fallback_models=list(fallback_models or []), max_tokens=1000,
                      temperature=0.0, json_mode=True, extra_body=extra)
        self.client = client or LLMClient(cfg)
        self.max_seconds = max_seconds

    async def transcribe(self, raw: bytes, failures: list | None = None) -> Transcript:
        wav, seconds = to_wav(raw, self.max_seconds)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
        ]}]
        try:
            result = await self.client.complete_json(messages, failures=failures)
        except LLMError as exc:
            raise SpeechError(str(exc)) from exc
        text = result.data.get("text") if isinstance(result.data, dict) else None
        if not isinstance(text, str):
            raise SpeechError("the model returned no transcript")
        return Transcript(text.strip(), seconds, result.cost_usd)

    async def aclose(self) -> None:
        await self.client.aclose()
