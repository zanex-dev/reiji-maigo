"""Zero-dependency headless single-turn Flash model runner for Reiji-Maigo swarm.

All swarm calls are strictly stateless, single-turn shadow invocations:
  - System message is locked to the worker's immutable L0 prefix (hashed via SHA-256).
  - No multi-turn conversation history is accumulated across calls.
  - Strict stdlib-only top-level imports (urllib.request, json, hashlib, etc.).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

DEFAULT_FLASH_MODEL = "gemini-2.5-flash"
ENV_FLASH_BASE_URL = "REIJI_FLASH_BASE_URL"
ENV_FLASH_API_KEY = "REIJI_FLASH_API_KEY"
ENV_FLASH_MODEL = "REIJI_FLASH_MODEL"

_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.IGNORECASE | re.DOTALL,
)


class SwarmRunnerError(RuntimeError):
    """Raised when a headless Flash model invocation fails."""


class SwarmJSONError(ValueError):
    """Raised when a model response cannot be parsed into a JSON object (dict)."""


@dataclass(frozen=True, slots=True)
class SwarmCallResult:
    """Immutable telemetry-bearing result of a single headless swarm model call."""

    data: dict[str, Any]
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: int
    prefix_sha256: str


@runtime_checkable
class SwarmModelRunner(Protocol):
    """Protocol for stateless, single-turn JSON shadow invocations."""

    def run_shadow_json(
        self,
        *,
        role: str,
        locked_system_prefix: str,
        user_payload: str,
    ) -> SwarmCallResult:
        """Execute an isolated Turn-1 shadow call and return a parsed JSON object."""
        ...


def compute_prefix_sha256(locked_system_prefix: str) -> str:
    """Compute a stable SHA-256 hex digest for a worker's locked L0 system prefix."""
    if not isinstance(locked_system_prefix, str):
        raise TypeError("locked_system_prefix must be a str")
    return hashlib.sha256(locked_system_prefix.encode("utf-8")).hexdigest()


def _validate_shadow_args(
    role: str,
    locked_system_prefix: str,
    user_payload: str,
) -> tuple[str, str, str]:
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be a non-empty string")
    if not isinstance(locked_system_prefix, str) or not locked_system_prefix.strip():
        raise ValueError("locked_system_prefix must be a non-empty string")
    if not isinstance(user_payload, str):
        raise TypeError("user_payload must be a str")
    return role.strip(), locked_system_prefix, user_payload


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Robustly extract and validate a top-level JSON object (dict) from model output.

    Supports:
      1. Pure JSON object text.
      2. Markdown fenced blocks (```json ... ``` or ``` ... ```), even with prose around them.
      3. Prose-wrapped JSON objects located via incremental ``JSONDecoder.raw_decode`` scan.
    """
    if not isinstance(raw_text, str):
        raise SwarmJSONError(f"Expected str response for JSON extraction, got {type(raw_text).__name__}")

    stripped = raw_text.strip()
    if not stripped:
        raise SwarmJSONError("Empty model response; expected a JSON object")

    # 1. Direct parse
    parsed_direct = _try_parse_json_dict(stripped)
    if parsed_direct is not None:
        return parsed_direct

    # 2. Markdown code fences (```json ... ``` or ``` ... ```)
    for match in _FENCED_JSON_RE.finditer(stripped):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        parsed_fenced = _try_parse_json_dict(candidate)
        if parsed_fenced is not None:
            return parsed_fenced

    # 3. Balanced top-level JSON object scan via raw_decode
    decoder = json.JSONDecoder()
    idx = 0
    length = len(stripped)
    while idx < length:
        brace_pos = stripped.find("{", idx)
        if brace_pos == -1:
            break
        try:
            obj, _end = decoder.raw_decode(stripped, brace_pos)
        except json.JSONDecodeError:
            idx = brace_pos + 1
            continue
        if isinstance(obj, dict):
            return obj
        idx = brace_pos + 1

    # Check if the response was valid JSON of a non-dict type (e.g. list / primitive)
    try:
        any_val = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SwarmJSONError(
            f"Failed to extract a valid JSON object from model output: {exc}"
        ) from exc
    raise SwarmJSONError(
        f"Model output must be a JSON object (dict), got {type(any_val).__name__}"
    )


def _try_parse_json_dict(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _resolve_chat_completions_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise ValueError(
            f"Flash base URL is not configured; pass base_url or set {ENV_FLASH_BASE_URL}"
        )
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def _extract_message_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SwarmRunnerError("OpenAI-compatible response missing non-empty 'choices'")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise SwarmRunnerError("Invalid choice entry in OpenAI-compatible response")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise SwarmRunnerError("Response choice missing 'message' object")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    raise SwarmRunnerError("Response message missing textual 'content'")


def _extract_usage_tokens(payload: Mapping[str, Any]) -> tuple[int, int]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return 0, 0
    prompt_raw = usage.get("prompt_tokens", 0)
    completion_raw = usage.get("completion_tokens", 0)
    prompt_tokens = int(prompt_raw) if isinstance(prompt_raw, int) and not isinstance(prompt_raw, bool) else 0
    completion_tokens = (
        int(completion_raw)
        if isinstance(completion_raw, int) and not isinstance(completion_raw, bool)
        else 0
    )
    return max(0, prompt_tokens), max(0, completion_tokens)


class HttpFlashRunner:
    """Stdlib-only OpenAI-compatible ``/v1/chat/completions`` headless Flash runner.

    Configuration resolution order: explicit constructor arg > injected ``env`` > ``os.environ``.
      - ``REIJI_FLASH_BASE_URL``
      - ``REIJI_FLASH_API_KEY``
      - ``REIJI_FLASH_MODEL`` (defaults to ``gemini-2.5-flash``)
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format_json: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        source_env: Mapping[str, str] = os.environ if env is None else env
        self.base_url: str = (
            base_url if base_url is not None else source_env.get(ENV_FLASH_BASE_URL, "")
        ).strip()
        self.api_key: str = (
            api_key if api_key is not None else source_env.get(ENV_FLASH_API_KEY, "")
        ).strip()
        resolved_model = (
            model if model is not None else source_env.get(ENV_FLASH_MODEL, DEFAULT_FLASH_MODEL)
        ).strip()
        self.model: str = resolved_model or DEFAULT_FLASH_MODEL
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        self.timeout: float = float(timeout)
        self.temperature: float = float(temperature)
        self.max_tokens: int | None = max_tokens
        self.response_format_json: bool = bool(response_format_json)
        self.extra_headers: dict[str, str] = dict(extra_headers) if extra_headers else {}
        self._opener: Callable[..., Any] = urllib.request.urlopen if opener is None else opener

    def run_shadow_json(
        self,
        *,
        role: str,
        locked_system_prefix: str,
        user_payload: str,
    ) -> SwarmCallResult:
        _role, prefix, payload_text = _validate_shadow_args(
            role, locked_system_prefix, user_payload
        )
        endpoint = _resolve_chat_completions_url(self.base_url)
        prefix_sha256 = compute_prefix_sha256(prefix)

        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": prefix},
                {"role": "user", "content": payload_text},
            ],
        }
        if self.max_tokens is not None:
            body["max_tokens"] = int(self.max_tokens)
        if self.response_format_json:
            body["response_format"] = {"type": "json_object"}

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Reiji-Swarm-Role": _role,
            "X-Reiji-Prefix-SHA256": prefix_sha256,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        started = time.perf_counter()
        try:
            with self._opener(req, timeout=self.timeout) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                err_bytes = exc.read()
                if err_bytes:
                    detail = err_bytes.decode("utf-8", errors="replace")[:300]
            except Exception:
                detail = ""
            raise SwarmRunnerError(
                f"Flash HTTP {exc.code} for role={_role!r} at {endpoint}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SwarmRunnerError(
                f"Flash network error for role={_role!r} at {endpoint}: {exc.reason}"
            ) from exc
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))

        try:
            response_obj = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SwarmRunnerError(
                f"Invalid JSON envelope from Flash endpoint {endpoint}: {exc}"
            ) from exc
        if not isinstance(response_obj, Mapping):
            raise SwarmRunnerError("Flash endpoint returned non-object JSON envelope")

        raw_text = _extract_message_content(response_obj)
        data = extract_json_object(raw_text)
        prompt_tokens, completion_tokens = _extract_usage_tokens(response_obj)

        return SwarmCallResult(
            data=data,
            raw_text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_ms=elapsed_ms,
            prefix_sha256=prefix_sha256,
        )


def _invoke_callable_target(
    target: Callable[..., Any],
    *,
    role: str,
    locked_system_prefix: str,
    user_payload: str,
) -> Any:
    """Invoke either ``fn(system, prompt)`` or a keyword-compatible adapter method."""
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return target(locked_system_prefix, user_payload)

    params = list(sig.parameters.values())
    param_names = {p.name for p in params}
    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    if {"locked_system_prefix", "user_payload"}.issubset(param_names):
        kwargs: dict[str, Any] = {
            "locked_system_prefix": locked_system_prefix,
            "user_payload": user_payload,
        }
        if "role" in param_names or has_var_keyword:
            kwargs["role"] = role
        return target(**kwargs)

    positional_slots = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if has_var_positional or len(positional_slots) >= 2:
        return target(locked_system_prefix, user_payload)

    if {"system", "prompt"}.issubset(param_names):
        return target(system=locked_system_prefix, prompt=user_payload)
    if {"system_prompt", "user_prompt"}.issubset(param_names):
        return target(system_prompt=locked_system_prefix, user_prompt=user_payload)
    if {"system_prompt", "user_payload"}.issubset(param_names):
        return target(system_prompt=locked_system_prefix, user_payload=user_payload)

    return target(locked_system_prefix, user_payload)


class CallableFlashRunner:
    """Adapter wrapping any ``callable(system, prompt) -> dict | str`` or duck-typed
    ``EngineAdapter.complete_json`` for unit tests and host-app integration."""

    def __init__(self, target: Any) -> None:
        complete_json = getattr(target, "complete_json", None)
        if callable(complete_json):
            self._callable: Callable[..., Any] = complete_json
        elif callable(target):
            self._callable = target
        else:
            raise TypeError(
                "CallableFlashRunner requires a callable(system, prompt) or an object with .complete_json()"
            )

    def run_shadow_json(
        self,
        *,
        role: str,
        locked_system_prefix: str,
        user_payload: str,
    ) -> SwarmCallResult:
        _role, prefix, payload_text = _validate_shadow_args(
            role, locked_system_prefix, user_payload
        )
        prefix_sha256 = compute_prefix_sha256(prefix)

        started = time.perf_counter()
        result = _invoke_callable_target(
            self._callable,
            role=_role,
            locked_system_prefix=prefix,
            user_payload=payload_text,
        )
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))

        if isinstance(result, SwarmCallResult):
            if result.prefix_sha256 == prefix_sha256:
                return result
            return SwarmCallResult(
                data=result.data,
                raw_text=result.raw_text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                elapsed_ms=result.elapsed_ms,
                prefix_sha256=prefix_sha256,
            )

        if isinstance(result, dict):
            data = dict(result)
            raw_text = json.dumps(data, ensure_ascii=False)
            return SwarmCallResult(
                data=data,
                raw_text=raw_text,
                prompt_tokens=0,
                completion_tokens=0,
                elapsed_ms=elapsed_ms,
                prefix_sha256=prefix_sha256,
            )

        if isinstance(result, str):
            data = extract_json_object(result)
            return SwarmCallResult(
                data=data,
                raw_text=result,
                prompt_tokens=0,
                completion_tokens=0,
                elapsed_ms=elapsed_ms,
                prefix_sha256=prefix_sha256,
            )

        raise SwarmJSONError(
            f"CallableFlashRunner target must return dict, str, or SwarmCallResult; got {type(result).__name__}"
        )


__all__ = [
    "CallableFlashRunner",
    "DEFAULT_FLASH_MODEL",
    "ENV_FLASH_API_KEY",
    "ENV_FLASH_BASE_URL",
    "ENV_FLASH_MODEL",
    "HttpFlashRunner",
    "SwarmCallResult",
    "SwarmJSONError",
    "SwarmModelRunner",
    "SwarmRunnerError",
    "compute_prefix_sha256",
    "extract_json_object",
]
