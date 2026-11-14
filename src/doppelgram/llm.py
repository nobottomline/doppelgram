"""OpenAI-compatible chat client (OpenRouter, Z.ai, DeepSeek, ...) with model fallback and cost tracking."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from .config import LLMConfig

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    def __init__(self, message: str, failures: list | None = None):
        super().__init__(message)
        self.failures = failures or []  # every failed attempt of this call, possibly billed


@dataclass
class LLMResult:
    data: dict
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    failures: list[FailedCall] = field(default_factory=list)  # attempts before the successful one


@dataclass
class FailedCall:
    model: str
    error: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0  # a reply can be billed even if it was unusable


def _fenced(text: str) -> str | None:
    """Body of the first ``` block (str.find only: linear time on any model output)."""
    start = text.find("```")
    if start == -1:
        return None
    end = text.find("```", start + 3)
    if end == -1:
        return None
    body = text[start + 3:end]
    if body[:4].lower() == "json":
        body = body[4:]
    return body.strip()


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating code fences and surrounding prose."""
    text = (text or "").strip()
    candidates = [text]
    fenced = _fenced(text)
    if fenced is not None:
        candidates.insert(0, fenced)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            value = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    # The reply itself is not included: it may describe a person and would end up in the logs.
    raise LLMError(f"model reply is not a JSON object ({len(text)} chars)")


def _error_summary(status: int, resp=None, payload=None) -> str:
    """Status and the provider's error code only: error messages can echo the request (a profile), and
    this text reaches the logs and the owner."""
    if payload is None:
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
    err = payload.get("error") if isinstance(payload, dict) else None
    code = (err.get("code") or err.get("type")) if isinstance(err, dict) else None
    code = str(code)[:60] if code is not None and re.fullmatch(r"[\w.\-: ]{1,60}", str(code)) else None
    return f"HTTP {status}" + (f" ({code})" if code else "")


def _strip_reasoning(text: str) -> str:
    """Drop a leading <think>...</think> block (local reasoning models); find-based, linear time."""
    end = text.rfind("</think>")
    return text[end + len("</think>"):] if end != -1 else text


def _content_text(content) -> str:
    """`message.content` is a string for most providers, a list of parts for some."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


# Parameters a provider may reject with HTTP 400/422, and how to adapt: drop it or rename it.
_ADAPTABLE = (
    ("response_format", None),
    ("max_tokens", "max_completion_tokens"),
    ("temperature", None),
)


class LLMClient:
    """Chat Completions client for any OpenAI-compatible endpoint (OpenAI, OpenRouter, Z.ai, DeepSeek,
    Groq, Together, vLLM, Ollama, LM Studio, ...). Parameters a provider rejects are adapted per model."""

    def __init__(self, cfg: LLMConfig, http: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.http = http or httpx.AsyncClient(timeout=cfg.timeout_s)
        self._adapted: dict[str, dict[str, str | None]] = {}  # model -> {param: replacement or None}

    @property
    def _is_openrouter(self) -> bool:
        return "openrouter.ai" in self.cfg.base_url

    @property
    def _url(self) -> str:
        base = self.cfg.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    def _body(self, model: str, messages: list[dict]) -> dict:
        body: dict = {"model": model, "messages": messages, "max_tokens": self.cfg.max_tokens}
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        if self.cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self._is_openrouter:
            body["usage"] = {"include": True}
        body.update(self.cfg.extra_body)
        for param, replacement in self._adapted.get(model, {}).items():
            if param in body:
                value = body.pop(param)
                if replacement:
                    body[replacement] = value
        return body

    def _adapt(self, model: str, status: int, error_text: str) -> bool:
        """Learn from a 400/422 that names an unsupported parameter; True if a retry makes sense."""
        if status not in (400, 422):
            return False
        low = error_text.lower()
        known = self._adapted.setdefault(model, {})
        for param, replacement in _ADAPTABLE:
            if param in low and param not in known:
                known[param] = replacement
                log.info("model %s rejected %r; %s", model, param,
                         f"using {replacement!r}" if replacement else "sending without it")
                return True
        return False

    def _cost(self, usage: dict) -> float:
        if isinstance(usage.get("cost"), (int, float)):
            return float(usage["cost"])
        return (usage.get("prompt_tokens", 0) * self.cfg.price_in_per_mtok
                + usage.get("completion_tokens", 0) * self.cfg.price_out_per_mtok) / 1e6

    async def complete_json(self, messages: list[dict], failures: list | None = None) -> LLMResult:
        """Try the primary model, then fallbacks. Raises LLMError if all fail.

        Failures belong to the call (never shared state): several agents use one client concurrently.
        Pass `failures` to own the list, so billed attempts survive cancellation.
        """
        failures = failures if failures is not None else []
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:  # local servers (Ollama, LM Studio, vLLM) may need no key
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        if self._is_openrouter:
            headers["X-Title"] = "doppelgram"
        for model in [self.cfg.model, *self.cfg.fallback_models]:
            for _attempt in range(len(_ADAPTABLE) + 1):
                usage: dict = {}
                try:
                    resp = await self.http.post(self._url, headers=headers, json=self._body(model, messages))
                    if resp.status_code >= 400:
                        if self._adapt(model, resp.status_code, resp.text):
                            continue  # same model, adapted request
                        raise LLMError(_error_summary(resp.status_code, resp))
                    payload = resp.json()
                    usage = payload.get("usage") or {}
                    if "error" in payload:
                        raise LLMError(_error_summary(resp.status_code, payload=payload))
                    choice = payload["choices"][0]["message"]
                    data = extract_json(_strip_reasoning(_content_text(choice.get("content"))))
                    return LLMResult(
                        data=data,
                        model=payload.get("model", model),
                        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                        cost_usd=self._cost(usage),
                        failures=failures,
                    )
                except (httpx.HTTPError, LLMError, KeyError, IndexError, TypeError, ValueError) as exc:
                    log.warning("LLM call to %s failed: %s", model, exc)
                    failures.append(FailedCall(
                        model, str(exc)[:300], int(usage.get("prompt_tokens", 0) or 0),
                        int(usage.get("completion_tokens", 0) or 0), self._cost(usage) if usage else 0.0))
                    break  # next model
        raise LLMError("all models failed: " + "; ".join(f"{f.model}: {f.error}" for f in failures),
                       failures)

    async def aclose(self) -> None:
        await self.http.aclose()
