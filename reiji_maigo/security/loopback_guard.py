"""Zero-trust localhost loopback HTTP server, anti-DNS-rebinding, and replay guard.

Stdlib-only clean-room security module for ``reiji_maigo``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import secrets
import socket
import sys
import threading
from typing import Any

ADMIN_TOKEN_HEADER: str = "X-Reiji-Admin-Token"
REQUEST_ID_HEADER: str = "X-Reiji-Request-Id"
DEFAULT_REPLAY_CACHE_SIZE: int = 4096

_ALLOWED_BIND_HOST: str = "127.0.0.1"
_ALLOWED_HOST_RE: re.Pattern[str] = re.compile(
    r"^(?:127\.0\.0\.1|localhost)(?::\d{1,5})?$",
    re.IGNORECASE,
)
_ALLOWED_ORIGIN_RE: re.Pattern[str] = re.compile(
    r"^https?://(?:127\.0\.0\.1|localhost)(?::\d{1,5})?$",
    re.IGNORECASE,
)
_ALLOWED_SEC_FETCH_SITES: frozenset[str] = frozenset(
    {"none", "same-origin", "same-site"}
)

__all__ = [
    "ADMIN_TOKEN_HEADER",
    "REQUEST_ID_HEADER",
    "DEFAULT_REPLAY_CACHE_SIZE",
    "ReplayGuard",
    "ExclusiveLoopbackHTTPServer",
    "LoopbackGuardedRequestHandler",
    "is_allowed_host",
    "is_allowed_origin",
    "is_allowed_sec_fetch_site",
    "verify_admin_token",
    "validate_loopback_request",
]


class ReplayGuard:
    """Thread-safe bounded request-ID anti-replay filter (deque + set)."""

    def __init__(self, max_size: int = DEFAULT_REPLAY_CACHE_SIZE) -> None:
        if not isinstance(max_size, int) or max_size < 1:
            raise ValueError("max_size must be a positive integer")
        self._lock = threading.Lock()
        self._queue: deque[str] = deque()
        self._seen: set[str] = set()
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def check_and_record(self, req_id: str) -> bool:
        """Accept a fresh request ID once (length 8..64) and reject replays."""
        if not isinstance(req_id, str) or len(req_id) < 8 or len(req_id) > 64:
            return False
        if any(ch.isspace() or ord(ch) < 32 for ch in req_id):
            return False
        with self._lock:
            if req_id in self._seen:
                return False
            if len(self._queue) >= self._max_size:
                oldest = self._queue.popleft()
                self._seen.discard(oldest)
            self._queue.append(req_id)
            self._seen.add(req_id)
            return True


def _get_header(headers: Mapping[str, str] | Any, name: str) -> str | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        direct = getter(name)
        if direct is not None:
            return str(direct)
        target_lower = name.lower()
        items = getattr(headers, "items", None)
        if callable(items):
            for key, value in items():
                if str(key).lower() == target_lower:
                    return str(value)
    return None


def is_allowed_host(host: str | None) -> bool:
    """Validate ``Host`` header to mitigate DNS rebinding attacks."""
    if not isinstance(host, str):
        return False
    candidate = host.strip()
    if not candidate:
        return False
    return _ALLOWED_HOST_RE.fullmatch(candidate) is not None


def is_allowed_origin(origin: str | None) -> bool:
    """Allow absent ``Origin`` (non-browser) or strict localhost HTTP(S) origins."""
    if origin is None:
        return True
    if not isinstance(origin, str):
        return False
    candidate = origin.strip()
    if not candidate:
        return True
    return _ALLOWED_ORIGIN_RE.fullmatch(candidate) is not None


def is_allowed_sec_fetch_site(sec_fetch_site: str | None) -> bool:
    """Allow absent ``Sec-Fetch-Site`` or non-cross-site browser fetch metadata."""
    if sec_fetch_site is None:
        return True
    if not isinstance(sec_fetch_site, str):
        return False
    candidate = sec_fetch_site.strip().lower()
    if not candidate:
        return True
    return candidate in _ALLOWED_SEC_FETCH_SITES


def verify_admin_token(provided: str | None, expected: str | None) -> bool:
    """Constant-time verification of ``X-Reiji-Admin-Token``."""
    if not isinstance(provided, str) or not isinstance(expected, str):
        return False
    p_clean = provided.strip()
    e_clean = expected.strip()
    if not p_clean or not e_clean:
        return False
    return hmac.compare_digest(p_clean.encode("utf-8"), e_clean.encode("utf-8"))


def validate_loopback_request(
    headers: Mapping[str, str] | Any,
    *,
    admin_token: str | None = None,
    require_admin_token: bool = True,
    replay_guard: ReplayGuard | None = None,
    require_request_id: bool = False,
) -> tuple[bool, int, str]:
    """Run full zero-trust loopback header, token, and replay checks.

    Returns ``(ok, http_status, reason)``.
    """
    host = _get_header(headers, "Host")
    if not is_allowed_host(host):
        return False, 403, "forbidden host"

    origin = _get_header(headers, "Origin")
    if not is_allowed_origin(origin):
        return False, 403, "forbidden origin"

    sec_fetch_site = _get_header(headers, "Sec-Fetch-Site")
    if not is_allowed_sec_fetch_site(sec_fetch_site):
        return False, 403, "forbidden sec-fetch-site"

    if require_admin_token:
        provided_token = _get_header(headers, ADMIN_TOKEN_HEADER)
        if not verify_admin_token(provided_token, admin_token):
            return False, 401, "unauthorized admin token"

    if replay_guard is not None:
        req_id = _get_header(headers, REQUEST_ID_HEADER)
        if req_id is None or not req_id.strip():
            if require_request_id:
                return False, 400, "missing request id"
        elif not replay_guard.check_and_record(req_id.strip()):
            return False, 409, "replayed or invalid request id"

    return True, 200, "ok"


class LoopbackGuardedRequestHandler(BaseHTTPRequestHandler):
    """Request handler enforcing loopback zero-trust checks and ``nosniff`` headers."""

    auto_guard: bool = True

    def end_headers(self) -> None:
        # Ensure every response (including errors) injects X-Content-Type-Options: nosniff.
        has_nosniff = False
        buffer = getattr(self, "_headers_buffer", None)
        if isinstance(buffer, list):
            for entry in buffer:
                if isinstance(entry, bytes) and entry.lower().startswith(
                    b"x-content-type-options:"
                ):
                    has_nosniff = True
                    break
        if not has_nosniff:
            self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(int(status_code))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_loopback_guard(
        self,
        *,
        require_admin_token: bool | None = None,
        require_request_id: bool | None = None,
    ) -> tuple[bool, int, str]:
        server = self.server
        validator = getattr(server, "validate_request", None)
        if callable(validator):
            return validator(
                self.headers,
                require_admin_token=require_admin_token,
                require_request_id=require_request_id,
            )
        return validate_loopback_request(
            self.headers,
            admin_token=getattr(server, "admin_token", None),
            require_admin_token=(
                bool(getattr(server, "require_admin_token", True))
                if require_admin_token is None
                else bool(require_admin_token)
            ),
            replay_guard=getattr(server, "replay_guard", None),
            require_request_id=(
                bool(getattr(server, "require_request_id", False))
                if require_request_id is None
                else bool(require_request_id)
            ),
        )

    def enforce_loopback_guard(
        self,
        *,
        require_admin_token: bool | None = None,
        require_request_id: bool | None = None,
    ) -> bool:
        ok, status_code, reason = self._check_loopback_guard(
            require_admin_token=require_admin_token,
            require_request_id=require_request_id,
        )
        if not ok:
            self._send_guard_error(status_code, reason)
            return False
        return True

    def _send_guard_error(self, status_code: int, reason: str) -> None:
        self.close_connection = True
        self.send_json(status_code, {"ok": False, "error": reason})

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        if self.auto_guard:
            ok, status_code, reason = self._check_loopback_guard()
            if not ok:
                self._send_guard_error(status_code, reason)
                return False
        return True

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if getattr(self.server, "quiet", True):
            return
        super().log_message(format, *args)


def _wrap_handler_class(
    handler_cls: type[BaseHTTPRequestHandler],
) -> type[BaseHTTPRequestHandler]:
    if issubclass(handler_cls, LoopbackGuardedRequestHandler):
        return handler_cls

    class _AutoGuardedHandler(LoopbackGuardedRequestHandler, handler_cls):  # type: ignore[misc,valid-type]
        pass

    _AutoGuardedHandler.__name__ = f"Guarded{handler_cls.__name__}"
    return _AutoGuardedHandler


class ExclusiveLoopbackHTTPServer(ThreadingHTTPServer):
    """Exclusive 127.0.0.1 HTTP server with SO_EXCLUSIVEADDRUSE and zero-trust guards.

    Rejects non-loopback bind hosts, disables ``SO_REUSEADDR``, enables Win32
    ``SO_EXCLUSIVEADDRUSE``, enforces Host/Origin/Sec-Fetch-Site/Admin-Token/
    ReplayGuard validation, and injects ``X-Content-Type-Options: nosniff`` on
    every response.
    """

    allow_reuse_address = False

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        bind_and_activate: bool = True,
        *,
        admin_token: str | None = None,
        require_admin_token: bool = True,
        replay_guard: ReplayGuard | None = None,
        require_request_id: bool = False,
        quiet: bool = True,
    ) -> None:
        if not isinstance(server_address, tuple) or len(server_address) != 2:
            raise ValueError("server_address must be a (host, port) 2-tuple")
        host, _port = server_address
        if host != _ALLOWED_BIND_HOST:
            raise ValueError(
                f"ExclusiveLoopbackHTTPServer only allows binding to {_ALLOWED_BIND_HOST!r}, "
                f"got {host!r}"
            )

        if require_admin_token:
            if admin_token is None:
                resolved_token: str | None = secrets.token_urlsafe(32)
            elif isinstance(admin_token, str) and admin_token.strip():
                resolved_token = admin_token.strip()
            else:
                raise ValueError("admin_token must be a non-empty string when required")
        else:
            resolved_token = (
                admin_token.strip()
                if isinstance(admin_token, str) and admin_token.strip()
                else None
            )

        self.admin_token: str | None = resolved_token
        self.require_admin_token: bool = bool(require_admin_token)
        self.replay_guard: ReplayGuard = (
            replay_guard if replay_guard is not None else ReplayGuard()
        )
        self.require_request_id: bool = bool(require_request_id)
        self.quiet: bool = bool(quiet)
        self.exclusive_addr_enabled: bool = False

        wrapped_handler = _wrap_handler_class(RequestHandlerClass)
        super().__init__(
            server_address,
            wrapped_handler,
            bind_and_activate=bind_and_activate,
        )

    def server_bind(self) -> None:
        host = self.server_address[0]
        if host != _ALLOWED_BIND_HOST:
            raise ValueError(
                f"ExclusiveLoopbackHTTPServer only allows binding to {_ALLOWED_BIND_HOST!r}, "
                f"got {host!r}"
            )
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
                )
                self.exclusive_addr_enabled = True
            except OSError:
                self.exclusive_addr_enabled = False
        super().server_bind()

    def validate_request(
        self,
        headers: Mapping[str, str] | Any,
        *,
        require_admin_token: bool | None = None,
        require_request_id: bool | None = None,
    ) -> tuple[bool, int, str]:
        return validate_loopback_request(
            headers,
            admin_token=self.admin_token,
            require_admin_token=(
                self.require_admin_token
                if require_admin_token is None
                else bool(require_admin_token)
            ),
            replay_guard=self.replay_guard,
            require_request_id=(
                self.require_request_id
                if require_request_id is None
                else bool(require_request_id)
            ),
        )
