"""Zero-dependency native MCP (Model Context Protocol) server over Stdio.

Implements JSON-RPC 2.0 over standard input/output using only the Python
standard library. Automatically detects and echoes both newline-delimited JSON
(NDJSON) and LSP-style ``Content-Length`` framed messages, while keeping all
diagnostic logs strictly on ``sys.stderr``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any, BinaryIO, TextIO

from reiji_maigo.bridges.shinobu import ReijiShinobuPlugin
from reiji_maigo.core.contracts import LOCKED_DOMAINS
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.swarm.coordinator import SwarmCoordinator

JSONRPC_VERSION: str = "2.0"
SUPPORTED_MCP_VERSIONS: tuple[str, ...] = ("2025-03-26", "2024-11-05")
DEFAULT_MCP_VERSION: str = "2024-11-05"
VALID_FRAMINGS: frozenset[str] = frozenset({"auto", "ndjson", "lsp"})

# Standard JSON-RPC 2.0 & MCP error codes
ERR_PARSE_ERROR: int = -32700
ERR_INVALID_REQUEST: int = -32600
ERR_METHOD_NOT_FOUND: int = -32601
ERR_INVALID_PARAMS: int = -32602
ERR_INTERNAL_ERROR: int = -32603
ERR_RESOURCE_NOT_FOUND: int = -32002
ERR_FORBIDDEN_READ_ONLY: int = -32003

MUTATING_TOOL_NAMES: frozenset[str] = frozenset({"reiji_update_pool"})

_TOKEN_RE = re.compile(r"[0-9A-Za-z_\u4e00-\u9fff]{2,}")


def _resolve_package_version() -> str:
    try:
        import reiji_maigo

        ver = getattr(reiji_maigo, "__version__", None)
        if isinstance(ver, str) and ver.strip():
            return ver.strip()
    except Exception:
        pass
    return "0.1.0"


def _log_stderr(message: str) -> None:
    """Write diagnostic output exclusively to stderr (never stdout)."""
    try:
        sys.stderr.write(f"[reiji-mcp] {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _jsonrpc_result(req_id: Any, result: Any) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": req_id,
        "result": result,
    }


def _jsonrpc_error(
    req_id: Any,
    code: int,
    message: str,
    *,
    data: Any = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {"code": int(code), "message": str(message)}
    if data is not None:
        err["data"] = data
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": req_id,
        "error": err,
    }


def _normalize_input_schema(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    schema = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _serialize_item(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _serialize_item(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_item(v) for v in value]
    return value


def _extract_query_tokens(query: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(query):
        tok = match.strip().lower()
        if tok and tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return tokens


@dataclass(slots=True)
class _McpToolEntry:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]

    def to_mcp_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": copy.deepcopy(self.input_schema),
        }


@dataclass(slots=True)
class _McpResourceEntry:
    uri: str
    name: str
    description: str
    mime_type: str
    reader: Callable[[], str]

    def to_mcp_schema(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


# ---------------------------------------------------------------------------
# Stdio Stream Framing Helpers (NDJSON + LSP Content-Length)
# ---------------------------------------------------------------------------


def _is_binary_stream(stream: Any) -> bool:
    if hasattr(stream, "encoding") and getattr(stream, "encoding", None) is not None:
        return False
    mode = getattr(stream, "mode", None)
    if isinstance(mode, str):
        return "b" in mode
    # Fall back to probing io.TextIOBase
    import io

    return not isinstance(stream, io.TextIOBase)


def _read_line_bytes(stream: Any, binary: bool) -> bytes:
    chunk = stream.readline()
    if not chunk:
        return b""
    if binary:
        return bytes(chunk)
    return str(chunk).encode("utf-8")


def _read_exact_bytes(stream: Any, length: int, binary: bool) -> bytes:
    if length <= 0:
        return b""
    if binary:
        buf = bytearray()
        while len(buf) < length:
            piece = stream.read(length - len(buf))
            if not piece:
                break
            buf.extend(piece)
        return bytes(buf)

    # Text stream fallback: accumulate characters until byte count reaches length.
    buf = bytearray()
    while len(buf) < length:
        ch = stream.read(1)
        if not ch:
            break
        buf.extend(str(ch).encode("utf-8"))
    return bytes(buf)


def _write_frame(
    stream: Any,
    payload_bytes: bytes,
    *,
    style: str,
    binary: bool,
) -> None:
    if style == "lsp":
        header = f"Content-Length: {len(payload_bytes)}\r\n\r\n".encode("ascii")
        wire = header + payload_bytes
    else:
        wire = payload_bytes + b"\n"

    if binary:
        stream.write(wire)
    else:
        stream.write(wire.decode("utf-8"))
    stream.flush()


def _looks_like_header_line(line: bytes) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith((b"{", b"[")):
        return False
    return b":" in stripped


def _read_next_frame(
    stream: Any,
    *,
    framing: str,
    binary: bool,
) -> tuple[str, bytes | None, str | None] | None:
    """Read the next framed JSON payload from ``stream``.

    Returns:
        - ``None`` on clean EOF.
        - ``(detected_style, payload_bytes, None)`` on valid frame read.
        - ``(detected_style, None, error_message)`` on framing error.
    """
    while True:
        first_line = _read_line_bytes(stream, binary)
        if not first_line:
            return None
        if first_line.strip():
            break

    use_lsp = framing == "lsp" or (framing == "auto" and _looks_like_header_line(first_line))
    if not use_lsp:
        return ("ndjson", first_line.strip(), None)

    headers: dict[str, str] = {}
    current_line = first_line
    while True:
        stripped = current_line.strip()
        if not stripped:
            break
        if b":" not in current_line:
            return ("lsp", None, f"Malformed header line: {stripped!r}")
        raw_key, raw_val = current_line.split(b":", 1)
        key = raw_key.decode("ascii", errors="replace").strip().lower()
        val = raw_val.decode("ascii", errors="replace").strip()
        headers[key] = val

        current_line = _read_line_bytes(stream, binary)
        if not current_line:
            return ("lsp", None, "Unexpected EOF while reading LSP headers")

    length_raw = headers.get("content-length")
    if length_raw is None:
        return ("lsp", None, "Missing Content-Length header")
    try:
        content_length = int(length_raw)
    except ValueError:
        return ("lsp", None, f"Invalid Content-Length header: {length_raw!r}")
    if content_length < 0:
        return ("lsp", None, f"Negative Content-Length header: {content_length}")

    payload = _read_exact_bytes(stream, content_length, binary)
    if len(payload) < content_length:
        return (
            "lsp",
            None,
            f"Truncated payload: expected {content_length} bytes, got {len(payload)}",
        )
    return ("lsp", payload, None)


# ---------------------------------------------------------------------------
# Native MCP Server
# ---------------------------------------------------------------------------


class ReijiMcpServer:
    """Zero-dependency native JSON-RPC 2.0 MCP server for Reiji-Maigo."""

    def __init__(
        self,
        engine: ReijiMemoryEngine | None = None,
        *,
        home: Path | str | None = None,
        db_path: Path | str | None = None,
        coordinator: SwarmCoordinator | None = None,
        read_only: bool = False,
    ) -> None:
        self._plugin = ReijiShinobuPlugin(
            home=home,
            db_path=db_path,
            engine=engine,
            coordinator=coordinator,
        )
        self._read_only: bool = bool(read_only)
        self._initialized: bool = False
        self._negotiated_version: str = DEFAULT_MCP_VERSION
        self._tools: dict[str, _McpToolEntry] | None = None
        self._resources: dict[str, _McpResourceEntry] = self._build_resources()

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def engine(self) -> ReijiMemoryEngine:
        return self._plugin.get_engine()

    @property
    def coordinator(self) -> SwarmCoordinator:
        return self._plugin.get_coordinator()

    def close(self) -> None:
        self._plugin.close()

    def __enter__(self) -> ReijiMcpServer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Tool & Resource Registries
    # ------------------------------------------------------------------

    def _get_tools(self) -> dict[str, _McpToolEntry]:
        if self._tools is not None:
            return self._tools

        entries: dict[str, _McpToolEntry] = {}
        for spec in self._plugin.build_tools():
            name = str(spec.name)
            entries[name] = _McpToolEntry(
                name=name,
                description=str(spec.description),
                input_schema=_normalize_input_schema(spec.parameters),
                handler=spec.handler,
            )

        entries["reiji_hybrid_recall"] = _McpToolEntry(
            name="reiji_hybrid_recall",
            description=(
                "Hybrid memory recall delegating to an installed 'reranker' plugin "
                "provider when available, with automatic fallback to Level-1 card "
                "recall and Level-3 FTS deep slice."
            ),
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "domain": {"type": "string", "enum": list(LOCKED_DOMAINS)},
                    "query_tags": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 5},
                    "min_tag_hits": {"type": "integer", "default": 1},
                    "pad": {"type": "integer", "default": 3},
                    "include_deep_slice": {"type": "boolean", "default": True},
                },
            },
            handler=self._handle_hybrid_recall,
        )

        self._tools = entries
        return entries

    def _handle_hybrid_recall(
        self,
        *,
        query: str,
        domain: str | None = None,
        query_tags: list[str] | None = None,
        top_k: int = 5,
        min_tag_hits: int = 1,
        pad: int = 3,
        include_deep_slice: bool = True,
    ) -> str:
        query_text = str(query or "").strip()
        if not query_text:
            raise ValueError("query must be a non-empty string")
        safe_top_k = max(1, int(top_k))
        safe_min_hits = max(1, int(min_tag_hits))
        safe_pad = max(0, int(pad))

        engine = self.engine
        kernel = getattr(engine, "kernel", None)
        reranker = (
            kernel.get_provider("reranker")
            if kernel is not None and callable(getattr(kernel, "get_provider", None))
            else None
        )

        if reranker is not None and callable(getattr(reranker, "hybrid_recall", None)):
            raw_result = self._invoke_reranker_hybrid_recall(
                reranker.hybrid_recall,
                engine=engine,
                query=query_text,
                domain=domain,
                query_tags=query_tags,
                top_k=safe_top_k,
                min_tag_hits=safe_min_hits,
                pad=safe_pad,
                include_deep_slice=bool(include_deep_slice),
            )
            if isinstance(raw_result, str):
                try:
                    parsed = json.loads(raw_result)
                    if isinstance(parsed, Mapping):
                        payload = dict(parsed)
                        payload.setdefault("ok", True)
                        payload.setdefault("mode", "reranker_plugin")
                        return json.dumps(payload, ensure_ascii=False)
                except ValueError:
                    pass
                return json.dumps(
                    {
                        "ok": True,
                        "mode": "reranker_plugin",
                        "query": query_text,
                        "text": raw_result,
                    },
                    ensure_ascii=False,
                )
            if isinstance(raw_result, Mapping):
                payload = _serialize_item(raw_result)
                payload.setdefault("ok", True)
                payload.setdefault("mode", "reranker_plugin")
                payload.setdefault("query", query_text)
                return json.dumps(payload, ensure_ascii=False)
            return json.dumps(
                {
                    "ok": True,
                    "mode": "reranker_plugin",
                    "query": query_text,
                    "results": _serialize_item(raw_result),
                },
                ensure_ascii=False,
            )

        # Fallback: Level-1 recall_cards + Level-3 deep_slice
        effective_tags = list(query_tags) if query_tags else _extract_query_tokens(query_text)
        cards = (
            engine.recall_cards(
                domain,
                effective_tags,
                min_tag_hits=safe_min_hits,
                top_k=safe_top_k,
            )
            if effective_tags
            else []
        )
        deep_hits: list[dict[str, Any]] = []
        if include_deep_slice:
            deep_hits = engine.deep_slice(query_text, limit=safe_top_k, pad=safe_pad)

        return json.dumps(
            {
                "ok": True,
                "mode": "fallback_lexical",
                "query": query_text,
                "domain": domain,
                "cards": [c.to_dict() for c in cards],
                "deep_hits": deep_hits,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _invoke_reranker_hybrid_recall(
        fn: Callable[..., Any],
        *,
        engine: ReijiMemoryEngine,
        query: str,
        domain: str | None,
        query_tags: list[str] | None,
        top_k: int,
        min_tag_hits: int,
        pad: int,
        include_deep_slice: bool,
    ) -> Any:
        candidates: dict[str, Any] = {
            "engine": engine,
            "query": query,
            "domain": domain,
            "query_tags": list(query_tags) if query_tags is not None else None,
            "top_k": top_k,
            "min_tag_hits": min_tag_hits,
            "pad": pad,
            "include_deep_slice": include_deep_slice,
        }
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(query=query, domain=domain, query_tags=query_tags, top_k=top_k)

        params = sig.parameters
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        kwargs: dict[str, Any] = {}
        pos_args: list[Any] = []
        for param_name, param in params.items():
            if param_name not in candidates:
                continue
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                pos_args.append(candidates[param_name])
            else:
                kwargs[param_name] = candidates[param_name]
        if has_var_kw:
            for k, v in candidates.items():
                if k != "engine" and k not in kwargs:
                    kwargs[k] = v
        return fn(*pos_args, **kwargs)

    def _build_resources(self) -> dict[str, _McpResourceEntry]:
        def read_pool_snapshot() -> str:
            return self.engine.pool_snapshot(include_undigested=True)

        def read_swarm_pulse() -> str:
            coord = self.coordinator
            return json.dumps(
                {
                    "ok": True,
                    "summary": coord.pulse_summary(),
                    "banner": coord.render_pulse_banner(),
                },
                ensure_ascii=False,
            )

        def read_security_summary() -> str:
            eng = self.engine
            summary = eng.security_summary()
            seals = eng.verify_card_seals()
            return json.dumps(
                {
                    "ok": bool(summary["audit_chain"]["ok"] and seals["ok"]),
                    "security": summary,
                    "card_seals": seals,
                },
                ensure_ascii=False,
            )

        resources = [
            _McpResourceEntry(
                uri="reiji://pool/snapshot",
                name="ActivePool Snapshot",
                description="Current Reiji-Maigo ActivePool working-memory snapshot.",
                mime_type="text/plain",
                reader=read_pool_snapshot,
            ),
            _McpResourceEntry(
                uri="reiji://swarm/pulse",
                name="Swarm Pulse Telemetry",
                description="Current Reiji-Maigo memory swarm pulse summary and banner.",
                mime_type="application/json",
                reader=read_swarm_pulse,
            ),
            _McpResourceEntry(
                uri="reiji://security/summary",
                name="Security Summary",
                description="Read-only security audit summary including hash-chain and card HMAC seals.",
                mime_type="application/json",
                reader=read_security_summary,
            ),
        ]
        return {r.uri: r for r in resources}

    # ------------------------------------------------------------------
    # JSON-RPC 2.0 Dispatch
    # ------------------------------------------------------------------

    def handle_rpc_message(self, msg: Mapping[str, Any]) -> dict[str, Any] | None:
        """Process a single parsed JSON-RPC 2.0 message.

        Returns a JSON-RPC 2.0 response dictionary for requests, or ``None`` for
        notification messages (messages without an ``id`` field).
        """
        if not isinstance(msg, Mapping):
            return _jsonrpc_error(None, ERR_INVALID_REQUEST, "Invalid Request: message must be a JSON object")

        is_notification = "id" not in msg
        req_id = msg.get("id")
        if not is_notification and req_id is not None:
            if isinstance(req_id, bool) or not isinstance(req_id, (str, int)):
                return _jsonrpc_error(None, ERR_INVALID_REQUEST, "Invalid Request: 'id' must be string, integer, or null")

        if msg.get("jsonrpc") != JSONRPC_VERSION:
            return _jsonrpc_error(
                None if is_notification else req_id,
                ERR_INVALID_REQUEST,
                "Invalid Request: 'jsonrpc' must be '2.0'",
            )

        method = msg.get("method")
        if not isinstance(method, str) or not method.strip():
            return _jsonrpc_error(
                None if is_notification else req_id,
                ERR_INVALID_REQUEST,
                "Invalid Request: 'method' must be a non-empty string",
            )

        raw_params = msg.get("params")
        if raw_params is None:
            params: Mapping[str, Any] = {}
        elif isinstance(raw_params, Mapping):
            params = raw_params
        else:
            if is_notification:
                return None
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: 'params' must be an object")

        # Notifications never receive a response per JSON-RPC 2.0.
        if method == "notifications/initialized":
            self._initialized = True
            return None if is_notification else _jsonrpc_result(req_id, {})

        if is_notification:
            return None

        try:
            if method == "initialize":
                return _jsonrpc_result(req_id, self._rpc_initialize(params))
            if method == "ping":
                return _jsonrpc_result(req_id, {})
            if method == "tools/list":
                return _jsonrpc_result(req_id, self._rpc_tools_list())
            if method == "tools/call":
                return self._rpc_tools_call(req_id, params)
            if method == "resources/list":
                return _jsonrpc_result(req_id, self._rpc_resources_list())
            if method == "resources/read":
                return self._rpc_resources_read(req_id, params)
            return _jsonrpc_error(req_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method}")
        except Exception as exc:
            _log_stderr(f"Internal error handling {method!r}: {exc!r}")
            return _jsonrpc_error(req_id, ERR_INTERNAL_ERROR, f"Internal error: {exc}")

    def _rpc_initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        if isinstance(requested, str) and requested in SUPPORTED_MCP_VERSIONS:
            negotiated = requested
        else:
            negotiated = DEFAULT_MCP_VERSION
        self._negotiated_version = negotiated

        instructions = (
            "Reiji-Maigo native stdio MCP server exposing ActivePool working memory, "
            "long-term memory cards, verbatim evidence, hybrid recall, swarm pulse, "
            "and read-only security audit resources."
        )
        if self._read_only:
            instructions += " Running in read-only mode (mutating tools disabled)."

        return {
            "protocolVersion": negotiated,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": {
                "name": "reiji-maigo",
                "version": _resolve_package_version(),
            },
            "instructions": instructions,
        }

    def _rpc_tools_list(self) -> dict[str, Any]:
        tools_map = self._get_tools()
        tools = [
            entry.to_mcp_schema()
            for name, entry in tools_map.items()
            if not (self._read_only and name in MUTATING_TOOL_NAMES)
        ]
        return {"tools": tools}

    def _rpc_tools_call(self, req_id: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name.strip():
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: 'name' must be a non-empty string")
        name = name.strip()

        raw_args = params.get("arguments")
        if raw_args is None:
            arguments: dict[str, Any] = {}
        elif isinstance(raw_args, Mapping):
            arguments = dict(raw_args)
        else:
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: 'arguments' must be an object")

        if self._read_only and name in MUTATING_TOOL_NAMES:
            return _jsonrpc_error(
                req_id,
                ERR_FORBIDDEN_READ_ONLY,
                f"Tool {name!r} is disabled in read-only mode",
                data={"tool": name, "read_only": True},
            )

        tools_map = self._get_tools()
        entry = tools_map.get(name)
        if entry is None:
            return _jsonrpc_error(
                req_id,
                ERR_INVALID_PARAMS,
                f"Unknown tool: {name!r}",
                data={"tool": name},
            )

        try:
            raw_output = entry.handler(**arguments)
        except Exception as exc:
            err_text = json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                ensure_ascii=False,
            )
            return _jsonrpc_result(
                req_id,
                {
                    "content": [{"type": "text", "text": err_text}],
                    "isError": True,
                },
            )

        text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False)
        return _jsonrpc_result(
            req_id,
            {
                "content": [{"type": "text", "text": text}],
                "isError": False,
            },
        )

    def _rpc_resources_list(self) -> dict[str, Any]:
        return {
            "resources": [entry.to_mcp_schema() for entry in self._resources.values()]
        }

    def _rpc_resources_read(self, req_id: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: 'uri' must be a non-empty string")
        uri = uri.strip()

        entry = self._resources.get(uri)
        if entry is None:
            return _jsonrpc_error(
                req_id,
                ERR_RESOURCE_NOT_FOUND,
                f"Resource not found: {uri!r}",
                data={"uri": uri},
            )

        text = entry.reader()
        return _jsonrpc_result(
            req_id,
            {
                "contents": [
                    {
                        "uri": entry.uri,
                        "mimeType": entry.mime_type,
                        "text": text,
                    }
                ]
            },
        )

    # ------------------------------------------------------------------
    # Stdio Serving Loop
    # ------------------------------------------------------------------

    def serve_stdio(
        self,
        in_stream: BinaryIO | TextIO | None = None,
        out_stream: BinaryIO | TextIO | None = None,
        *,
        framing: str = "auto",
    ) -> int:
        """Serve JSON-RPC 2.0 MCP messages from ``in_stream`` to ``out_stream``.

        Args:
            in_stream: Input stream (defaults to ``sys.stdin.buffer`` or ``sys.stdin``).
            out_stream: Output stream (defaults to ``sys.stdout.buffer`` or ``sys.stdout``).
            framing: One of ``"auto"``, ``"ndjson"``, or ``"lsp"``.

        Returns:
            Process exit code (``0`` on clean EOF).
        """
        framing_norm = str(framing).strip().lower()
        if framing_norm not in VALID_FRAMINGS:
            raise ValueError(
                f"Unsupported framing {framing!r}; expected one of {sorted(VALID_FRAMINGS)}"
            )

        reader = in_stream if in_stream is not None else getattr(sys.stdin, "buffer", sys.stdin)
        writer = out_stream if out_stream is not None else getattr(sys.stdout, "buffer", sys.stdout)
        in_binary = _is_binary_stream(reader)
        out_binary = _is_binary_stream(writer)

        while True:
            frame = _read_next_frame(reader, framing=framing_norm, binary=in_binary)
            if frame is None:
                return 0

            detected_style, payload_bytes, frame_error = frame
            reply_style = detected_style if framing_norm == "auto" else framing_norm

            if frame_error is not None or payload_bytes is None:
                _log_stderr(f"Framing error: {frame_error}")
                err_resp = _jsonrpc_error(None, ERR_PARSE_ERROR, f"Parse error: {frame_error}")
                self._emit_response(writer, err_resp, style=reply_style, binary=out_binary)
                continue

            try:
                decoded = json.loads(payload_bytes.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                _log_stderr(f"JSON decode error: {exc}")
                err_resp = _jsonrpc_error(None, ERR_PARSE_ERROR, f"Parse error: {exc}")
                self._emit_response(writer, err_resp, style=reply_style, binary=out_binary)
                continue

            if isinstance(decoded, list):
                if not decoded:
                    err_resp = _jsonrpc_error(None, ERR_INVALID_REQUEST, "Invalid Request: empty batch")
                    self._emit_response(writer, err_resp, style=reply_style, binary=out_binary)
                    continue
                batch_responses = []
                for item in decoded:
                    resp = self.handle_rpc_message(item)
                    if resp is not None:
                        batch_responses.append(resp)
                if batch_responses:
                    self._emit_response(writer, batch_responses, style=reply_style, binary=out_binary)
                continue

            response = self.handle_rpc_message(decoded)
            if response is not None:
                self._emit_response(writer, response, style=reply_style, binary=out_binary)

    @staticmethod
    def _emit_response(
        writer: Any,
        payload: Any,
        *,
        style: str,
        binary: bool,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        _write_frame(writer, encoded, style=style, binary=binary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reiji-mcp-stdio",
        description="Run Reiji-Maigo native zero-dependency MCP server over stdio.",
    )
    parser.add_argument("--home", default=None, help="Reiji-Maigo home directory")
    parser.add_argument("--db", dest="db_path", default=None, help="SQLite database path")
    parser.add_argument(
        "--framing",
        default="auto",
        choices=sorted(VALID_FRAMINGS),
        help="Stdio framing mode (auto, ndjson, or lsp)",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Disable mutating tools such as reiji_update_pool",
    )
    args = parser.parse_args(argv)

    with ReijiMcpServer(
        home=args.home,
        db_path=args.db_path,
        read_only=bool(args.read_only),
    ) as server:
        return server.serve_stdio(framing=args.framing)


if __name__ == "__main__":
    raise SystemExit(main())
