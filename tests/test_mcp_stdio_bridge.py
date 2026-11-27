"""Unit tests for the zero-dependency native Stdio MCP server bridge."""

from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reiji_maigo.bridges.mcp_stdio import (
    DEFAULT_MCP_VERSION,
    ERR_FORBIDDEN_READ_ONLY,
    ERR_INVALID_PARAMS,
    ERR_INVALID_REQUEST,
    ERR_METHOD_NOT_FOUND,
    ERR_PARSE_ERROR,
    ERR_RESOURCE_NOT_FOUND,
    SUPPORTED_MCP_VERSIONS,
    ReijiMcpServer,
)
from reiji_maigo.engine import ReijiMemoryEngine


def _encode_lsp(payload: dict | list | bytes) -> bytes:
    body = (
        payload
        if isinstance(payload, bytes)
        else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def _decode_lsp_stream(raw: bytes) -> list[dict]:
    messages: list[dict] = []
    idx = 0
    total = len(raw)
    while idx < total:
        sep = raw.find(b"\r\n\r\n", idx)
        if sep == -1:
            raise AssertionError(f"Missing LSP header separator in remaining bytes: {raw[idx:]!r}")
        header_block = raw[idx:sep].decode("ascii")
        length = None
        for line in header_block.split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1].strip())
        if length is None:
            raise AssertionError(f"Missing Content-Length in header: {header_block!r}")
        body_start = sep + 4
        body_end = body_start + length
        body = raw[body_start:body_end]
        if len(body) != length:
            raise AssertionError(f"Expected {length} bytes, got {len(body)}")
        messages.append(json.loads(body.decode("utf-8")))
        idx = body_end
    return messages


class TestReijiMcpStdioBridge(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.home = Path(self._tmpdir.name) / "reiji_home"
        self.engine = ReijiMemoryEngine(home=self.home, load_plugins=False)
        self.addCleanup(self.engine.close)
        # Seed raw turns so card archiving, Level-2 evidence, and Level-3 FTS work end-to-end.
        self.engine.append_turn(
            "ep_mcp_01",
            "user",
            "We need sqlite wal tuning and mcp stdio framing with Content-Length support.",
            origin_trust="user_verified",
        )
        self.engine.append_turn(
            "ep_mcp_01",
            "assistant",
            "Agreed: enable WAL mode and dual NDJSON plus LSP Content-Length stdio framing.",
            origin_trust="assistant",
        )

    # ------------------------------------------------------------------
    # Handshake, Version Negotiation, Ping & Notifications
    # ------------------------------------------------------------------

    def test_initialize_version_negotiation_and_notifications(self) -> None:
        server = ReijiMcpServer(engine=self.engine)

        for supported in SUPPORTED_MCP_VERSIONS:
            resp = server.handle_rpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": f"init-{supported}",
                    "method": "initialize",
                    "params": {"protocolVersion": supported},
                }
            )
            assert resp is not None
            self.assertEqual(resp["id"], f"init-{supported}")
            self.assertEqual(resp["result"]["protocolVersion"], supported)
            self.assertIn("tools", resp["result"]["capabilities"])
            self.assertIn("resources", resp["result"]["capabilities"])
            self.assertEqual(resp["result"]["serverInfo"]["name"], "reiji-maigo")

        fallback = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {"protocolVersion": "1999-01-01"},
            }
        )
        assert fallback is not None
        self.assertEqual(fallback["id"], 0)
        self.assertEqual(fallback["result"]["protocolVersion"], DEFAULT_MCP_VERSION)

        # Notification (no id) must return None.
        notif = server.handle_rpc_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        self.assertIsNone(notif)

        ping = server.handle_rpc_message({"jsonrpc": "2.0", "id": 99, "method": "ping"})
        self.assertEqual(ping, {"jsonrpc": "2.0", "id": 99, "result": {}})

    # ------------------------------------------------------------------
    # Stdio Framing (NDJSON, LSP Content-Length, Auto & Stderr Isolation)
    # ------------------------------------------------------------------

    def test_serve_stdio_auto_ndjson_roundtrip(self) -> None:
        server = ReijiMcpServer(engine=self.engine)
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        ]
        raw_in = ("\n".join(json.dumps(r) for r in requests) + "\n").encode("utf-8")
        in_buf = io.BytesIO(raw_in)
        out_buf = io.BytesIO()
        err_buf = io.StringIO()

        with contextlib.redirect_stderr(err_buf):
            rc = server.serve_stdio(in_buf, out_buf, framing="auto")

        self.assertEqual(rc, 0)
        raw_out = out_buf.getvalue()
        self.assertNotIn(b"Content-Length:", raw_out)
        lines = [line for line in raw_out.decode("utf-8").splitlines() if line.strip()]
        # 3 requests + 1 silent notification -> 3 responses
        self.assertEqual(len(lines), 3)
        parsed = [json.loads(line) for line in lines]
        self.assertEqual([m["id"] for m in parsed], [1, 2, 3])
        self.assertEqual(parsed[0]["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(parsed[1]["result"], {})
        self.assertEqual(len(parsed[2]["result"]["tools"]), 8)

    def test_serve_stdio_auto_lsp_content_length_utf8_roundtrip(self) -> None:
        server = ReijiMcpServer(engine=self.engine)
        create_req = {
            "jsonrpc": "2.0",
            "id": "cn-1",
            "method": "tools/call",
            "params": {
                "name": "reiji_update_pool",
                "arguments": {
                    "action": "create",
                    "title": "零依赖协议验证",
                    "step": "中文多字节 Content-Length 长度校验",
                    "span": {"episode_id": "ep_mcp_01", "turn_start": 1, "turn_end": 2},
                },
            },
        }
        raw_in = (
            _encode_lsp({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
            + _encode_lsp({"jsonrpc": "2.0", "method": "notifications/initialized"})
            + _encode_lsp(create_req)
        )
        in_buf = io.BytesIO(raw_in)
        out_buf = io.BytesIO()

        rc = server.serve_stdio(in_buf, out_buf, framing="auto")
        self.assertEqual(rc, 0)
        messages = _decode_lsp_stream(out_buf.getvalue())
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(messages[1]["id"], "cn-1")
        self.assertFalse(messages[1]["result"]["isError"])
        tool_payload = json.loads(messages[1]["result"]["content"][0]["text"])
        self.assertTrue(tool_payload["ok"])
        self.assertEqual(tool_payload["event"]["title"], "零依赖协议验证")

    def test_serve_stdio_explicit_framing_and_text_streams(self) -> None:
        server = ReijiMcpServer(engine=self.engine)

        # Text stream with explicit ndjson
        text_in = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": "t1", "method": "ping"}) + "\n")
        text_out = io.StringIO()
        self.assertEqual(server.serve_stdio(text_in, text_out, framing="ndjson"), 0)
        self.assertEqual(json.loads(text_out.getvalue().strip())["id"], "t1")

        # Explicit lsp forcing Content-Length reply
        lsp_in = io.BytesIO(_encode_lsp({"jsonrpc": "2.0", "id": "l1", "method": "ping"}))
        lsp_out = io.BytesIO()
        self.assertEqual(server.serve_stdio(lsp_in, lsp_out, framing="lsp"), 0)
        decoded = _decode_lsp_stream(lsp_out.getvalue())
        self.assertEqual(decoded[0]["id"], "l1")

        with self.assertRaises(ValueError):
            server.serve_stdio(io.BytesIO(b""), io.BytesIO(), framing="invalid")

    def test_serve_stdio_parse_errors_isolated_to_stderr(self) -> None:
        server = ReijiMcpServer(engine=self.engine)
        raw_in = (
            b"{not valid json\n"
            + json.dumps({"jsonrpc": "2.0", "id": "ok-after-err", "method": "ping"}).encode("utf-8")
            + b"\n"
        )
        in_buf = io.BytesIO(raw_in)
        out_buf = io.BytesIO()
        err_buf = io.StringIO()

        with contextlib.redirect_stderr(err_buf):
            rc = server.serve_stdio(in_buf, out_buf, framing="auto")

        self.assertEqual(rc, 0)
        self.assertIn("[reiji-mcp]", err_buf.getvalue())
        lines = [ln for ln in out_buf.getvalue().decode("utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        err_resp = json.loads(lines[0])
        self.assertIsNone(err_resp["id"])
        self.assertEqual(err_resp["error"]["code"], ERR_PARSE_ERROR)
        ok_resp = json.loads(lines[1])
        self.assertEqual(ok_resp["id"], "ok-after-err")
        self.assertEqual(ok_resp["result"], {})

    def test_serve_stdio_malformed_lsp_headers_return_parse_error(self) -> None:
        server = ReijiMcpServer(engine=self.engine)
        raw_in = b"Content-Length: notanint\r\n\r\n"
        in_buf = io.BytesIO(raw_in)
        out_buf = io.BytesIO()
        err_buf = io.StringIO()

        with contextlib.redirect_stderr(err_buf):
            rc = server.serve_stdio(in_buf, out_buf, framing="lsp")

        self.assertEqual(rc, 0)
        decoded = _decode_lsp_stream(out_buf.getvalue())
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["error"]["code"], ERR_PARSE_ERROR)
        self.assertIn("Invalid Content-Length", err_buf.getvalue())

    # ------------------------------------------------------------------
    # Tools List, Tool Calls & Hybrid Recall (Fallback + Reranker Plugin)
    # ------------------------------------------------------------------

    def test_tools_list_schema_and_end_to_end_memory_tools(self) -> None:
        server = ReijiMcpServer(engine=self.engine)
        listed = server.handle_rpc_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert listed is not None
        tools = listed["result"]["tools"]
        names = [t["name"] for t in tools]
        expected_names = {
            "reiji_pool_snapshot",
            "reiji_update_pool",
            "reiji_recall_cards",
            "reiji_read_evidence",
            "reiji_scout_deep",
            "reiji_swarm_pulse",
            "reiji_security_audit",
            "reiji_hybrid_recall",
        }
        self.assertEqual(set(names), expected_names)
        for tool in tools:
            self.assertIn("inputSchema", tool)
            self.assertNotIn("parameters", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")

        # 1. Create event via reiji_update_pool
        create_resp = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "reiji_update_pool",
                    "arguments": {
                        "action": "create",
                        "title": "MCP Stdio Bridge",
                        "step": "Implement native stdio transport",
                        "span": {"episode_id": "ep_mcp_01", "turn_start": 1, "turn_end": 2},
                    },
                },
            }
        )
        assert create_resp is not None
        self.assertFalse(create_resp["result"]["isError"])
        event_id = json.loads(create_resp["result"]["content"][0]["text"])["event"]["event_id"]

        # 2. Verify snapshot includes created event
        snap_resp = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "reiji_pool_snapshot", "arguments": {}},
            }
        )
        assert snap_resp is not None
        self.assertFalse(snap_resp["result"]["isError"])
        snap_data = json.loads(snap_resp["result"]["content"][0]["text"])
        self.assertIn("MCP Stdio Bridge", snap_data["snapshot"])

        # 3. Archive event to long-term card
        arch_resp = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "reiji_update_pool",
                    "arguments": {
                        "action": "archive",
                        "event_id": event_id,
                        "domain": "proj",
                        "tags": ["sqlite", "mcp", "stdio"],
                        "state": {"framing": "dual"},
                    },
                },
            }
        )
        assert arch_resp is not None
        self.assertFalse(arch_resp["result"]["isError"])
        card_id = json.loads(arch_resp["result"]["content"][0]["text"])["card"]["card_id"]

        # 4. Recall cards & read Level-2 evidence
        recall_resp = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "reiji_recall_cards",
                    "arguments": {"domain": "proj", "query_tags": ["mcp"]},
                },
            }
        )
        assert recall_resp is not None
        recall_data = json.loads(recall_resp["result"]["content"][0]["text"])
        self.assertEqual(len(recall_data["cards"]), 1)
        self.assertEqual(recall_data["cards"][0]["card_id"], card_id)

        ev_resp = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "reiji_read_evidence",
                    "arguments": {"card_id": card_id, "pad": 0},
                },
            }
        )
        assert ev_resp is not None
        ev_data = json.loads(ev_resp["result"]["content"][0]["text"])
        self.assertGreaterEqual(len(ev_data["turns"]), 2)

        # 5. Hybrid recall fallback (no reranker plugin -> recall_cards + deep_slice)
        hybrid_fb = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "reiji_hybrid_recall",
                    "arguments": {
                        "query": "sqlite wal",
                        "domain": "proj",
                        "query_tags": ["sqlite"],
                        "top_k": 3,
                    },
                },
            }
        )
        assert hybrid_fb is not None
        self.assertFalse(hybrid_fb["result"]["isError"])
        fb_data = json.loads(hybrid_fb["result"]["content"][0]["text"])
        self.assertEqual(fb_data["mode"], "fallback_lexical")
        self.assertEqual(len(fb_data["cards"]), 1)
        self.assertGreaterEqual(len(fb_data["deep_hits"]), 1)

        # 6. Hybrid recall delegating to reranker plugin provider
        class _DummyReranker:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def hybrid_recall(self, *, query: str, domain: str | None = None, top_k: int = 5) -> dict:
                self.calls.append({"query": query, "domain": domain, "top_k": top_k})
                return {"items": [{"card_id": card_id, "score": 0.99}]}

        dummy = _DummyReranker()
        with mock.patch.object(
            self.engine.kernel,
            "get_provider",
            side_effect=lambda role: dummy if role == "reranker" else None,
        ):
            hybrid_plugin = server.handle_rpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "tools/call",
                    "params": {
                        "name": "reiji_hybrid_recall",
                        "arguments": {"query": "mcp framing", "domain": "proj", "top_k": 2},
                    },
                }
            )
        assert hybrid_plugin is not None
        self.assertFalse(hybrid_plugin["result"]["isError"])
        plugin_data = json.loads(hybrid_plugin["result"]["content"][0]["text"])
        self.assertEqual(plugin_data["mode"], "reranker_plugin")
        self.assertEqual(plugin_data["items"][0]["card_id"], card_id)
        self.assertEqual(dummy.calls, [{"query": "mcp framing", "domain": "proj", "top_k": 2}])

        # 7. Swarm pulse & security audit tools
        pulse_resp = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "reiji_swarm_pulse", "arguments": {"format": "both"}},
            }
        )
        assert pulse_resp is not None
        self.assertFalse(pulse_resp["result"]["isError"])
        pulse_data = json.loads(pulse_resp["result"]["content"][0]["text"])
        self.assertIn("summary", pulse_data)
        self.assertIn("banner", pulse_data)

        sec_resp = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "reiji_security_audit", "arguments": {"include_entries": True}},
            }
        )
        assert sec_resp is not None
        self.assertFalse(sec_resp["result"]["isError"])
        sec_data = json.loads(sec_resp["result"]["content"][0]["text"])
        self.assertTrue(sec_data["ok"])
        self.assertTrue(sec_data["audit_chain"]["ok"])
        self.assertTrue(sec_data["card_seals"]["ok"])

    # ------------------------------------------------------------------
    # Read-Only Gate
    # ------------------------------------------------------------------

    def test_read_only_hides_and_blocks_reiji_update_pool(self) -> None:
        server = ReijiMcpServer(engine=self.engine, read_only=True)
        self.assertTrue(server.read_only)

        listed = server.handle_rpc_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert listed is not None
        names = [t["name"] for t in listed["result"]["tools"]]
        self.assertNotIn("reiji_update_pool", names)
        self.assertIn("reiji_pool_snapshot", names)
        self.assertIn("reiji_hybrid_recall", names)
        self.assertEqual(len(names), 7)

        blocked = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "reiji_update_pool",
                    "arguments": {
                        "action": "create",
                        "title": "Should Fail",
                        "step": "Blocked",
                        "span": {"episode_id": "ep_mcp_01", "turn_start": 1, "turn_end": 1},
                    },
                },
            }
        )
        assert blocked is not None
        self.assertIn("error", blocked)
        self.assertEqual(blocked["error"]["code"], ERR_FORBIDDEN_READ_ONLY)
        self.assertEqual(self.engine.status()["counts"]["pool_total"], 0)

        # Read-only tools still work cleanly.
        snap = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "reiji_pool_snapshot", "arguments": {}},
            }
        )
        assert snap is not None
        self.assertFalse(snap["result"]["isError"])

    # ------------------------------------------------------------------
    # Read-Only Resources
    # ------------------------------------------------------------------

    def test_resources_list_and_read(self) -> None:
        server = ReijiMcpServer(engine=self.engine, read_only=True)
        listed = server.handle_rpc_message({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})
        assert listed is not None
        uris = {r["uri"] for r in listed["result"]["resources"]}
        self.assertEqual(
            uris,
            {
                "reiji://pool/snapshot",
                "reiji://swarm/pulse",
                "reiji://security/summary",
            },
        )

        for req_id, uri, expected_mime in (
            (2, "reiji://pool/snapshot", "text/plain"),
            (3, "reiji://swarm/pulse", "application/json"),
            (4, "reiji://security/summary", "application/json"),
        ):
            resp = server.handle_rpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "resources/read",
                    "params": {"uri": uri},
                }
            )
            assert resp is not None
            contents = resp["result"]["contents"]
            self.assertEqual(len(contents), 1)
            self.assertEqual(contents[0]["uri"], uri)
            self.assertEqual(contents[0]["mimeType"], expected_mime)
            self.assertIsInstance(contents[0]["text"], str)
            if expected_mime == "application/json":
                parsed = json.loads(contents[0]["text"])
                self.assertTrue(parsed["ok"])

        missing = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "resources/read",
                "params": {"uri": "reiji://does/not/exist"},
            }
        )
        assert missing is not None
        self.assertEqual(missing["error"]["code"], ERR_RESOURCE_NOT_FOUND)

    # ------------------------------------------------------------------
    # Error Codes & Tool Execution Error Handling
    # ------------------------------------------------------------------

    def test_jsonrpc_error_codes_and_tool_execution_is_error(self) -> None:
        server = ReijiMcpServer(engine=self.engine)

        # Invalid Request: wrong jsonrpc version
        bad_ver = server.handle_rpc_message({"jsonrpc": "1.0", "id": 1, "method": "ping"})
        assert bad_ver is not None
        self.assertEqual(bad_ver["error"]["code"], ERR_INVALID_REQUEST)

        # Invalid Request: bool id
        bad_id = server.handle_rpc_message({"jsonrpc": "2.0", "id": True, "method": "ping"})
        assert bad_id is not None
        self.assertEqual(bad_id["error"]["code"], ERR_INVALID_REQUEST)

        # Method not found
        unknown_method = server.handle_rpc_message({"jsonrpc": "2.0", "id": 2, "method": "no/such_method"})
        assert unknown_method is not None
        self.assertEqual(unknown_method["error"]["code"], ERR_METHOD_NOT_FOUND)

        # Invalid params: non-object params
        bad_params = server.handle_rpc_message({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": []})
        assert bad_params is not None
        self.assertEqual(bad_params["error"]["code"], ERR_INVALID_PARAMS)

        # Invalid params: unknown tool
        unknown_tool = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "reiji_nonexistent", "arguments": {}},
            }
        )
        assert unknown_tool is not None
        self.assertEqual(unknown_tool["error"]["code"], ERR_INVALID_PARAMS)

        # Invalid params: arguments is not object
        bad_args = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "reiji_pool_snapshot", "arguments": "not-an-object"},
            }
        )
        assert bad_args is not None
        self.assertEqual(bad_args["error"]["code"], ERR_INVALID_PARAMS)

        # Invalid params: resources/read missing uri
        missing_uri = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "resources/read",
                "params": {},
            }
        )
        assert missing_uri is not None
        self.assertEqual(missing_uri["error"]["code"], ERR_INVALID_PARAMS)

        # Tool runtime exception -> result with isError=True
        tool_err = server.handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "reiji_update_pool",
                    "arguments": {"action": "unsupported_action"},
                },
            }
        )
        assert tool_err is not None
        self.assertIn("result", tool_err)
        self.assertTrue(tool_err["result"]["isError"])
        err_payload = json.loads(tool_err["result"]["content"][0]["text"])
        self.assertFalse(err_payload["ok"])
        self.assertIn("Unsupported action", err_payload["error"])

    # ------------------------------------------------------------------
    # White-Hat Zero-Dependency Hygiene Check
    # ------------------------------------------------------------------

    def test_mcp_stdio_module_is_zero_dependency_and_clean(self) -> None:
        source_path = (
            Path(__file__).resolve().parent.parent / "reiji_maigo" / "bridges" / "mcp_stdio.py"
        )
        text = source_path.read_text(encoding="utf-8")

        # Verify no forbidden literals or machine paths
        forbidden_tokens = (
            "App" + "Data",
            "C:\\" + "Users\\",
            "/" + "Users/",
            "acerola" + "_orion",
            "orion" + "_private",
            "shinobu" + "_private",
        )
        for forbidden in forbidden_tokens:
            self.assertNotIn(forbidden, text)

        # Verify all imports are standard library or reiji_maigo
        tree = ast.parse(text)
        import sys as _sys

        stdlib_names = set(getattr(_sys, "stdlib_module_names", ()))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    self.assertTrue(
                        top == "reiji_maigo" or top in stdlib_names,
                        f"Unexpected non-stdlib import: {alias.name}",
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".", 1)[0]
                self.assertTrue(
                    top == "reiji_maigo" or top in stdlib_names,
                    f"Unexpected non-stdlib import: {node.module}",
                )


if __name__ == "__main__":
    unittest.main()
