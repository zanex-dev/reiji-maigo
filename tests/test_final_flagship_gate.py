"""End-to-end flagship gate tests for Reiji-Maigo v0.2.0."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]

from reiji_maigo.core.contracts import SourceSpan
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.plugins import BUILTIN_PLUGIN_FACTORIES
from reiji_maigo.plugins.bench_harness import ReijiBenchHarness
from reiji_maigo.plugins.bootstrap_importer import BootstrapImporter
from reiji_maigo.plugins.companion import render_mcp_server_config
from reiji_maigo.plugins.hybrid_reranker import MemoRrfReranker

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_BUILTIN_PLUGINS = {
    "roi_ledger",
    "epoch_vault",
    "thermal_sync",
    "companion",
    "hybrid_reranker",
    "bootstrap_importer",
    "bench_harness",
}


def _run_cli(
    args: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
    stdin_text: str | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "reiji_maigo", *args],
        cwd=str(REPO_ROOT),
        env=env,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )


class TestFinalFlagshipGate(unittest.TestCase):
    """Flagship v0.2.0 integration, CLI, MCP, metadata, and cleanroom verification."""

    def test_mcp_cli_subprocess_initialize_tools_list_and_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rpc_messages = [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "flagship-gate", "version": "0.2.0"},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "reiji_hybrid_recall",
                        "arguments": {"query": "flagship", "top_k": 3},
                    },
                },
            ]
            stdin_payload = "\n".join(json.dumps(m, ensure_ascii=False) for m in rpc_messages) + "\n"
            proc = _run_cli(
                ["--home", tmp, "mcp", "--framing", "ndjson"],
                env_overrides={"REIJI_MAIGO_DISABLE_PLUGINS": "0"},
                stdin_text=stdin_payload,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")

            responses = [
                json.loads(line)
                for line in proc.stdout.splitlines()
                if line.strip()
            ]
            by_id = {resp.get("id"): resp for resp in responses}

            self.assertIn(1, by_id)
            self.assertIn("result", by_id[1])
            self.assertIn("protocolVersion", by_id[1]["result"])
            self.assertIn("serverInfo", by_id[1]["result"])

            self.assertIn(2, by_id)
            tools = by_id[2]["result"]["tools"]
            tool_names = {t["name"] for t in tools}
            self.assertIn("reiji_hybrid_recall", tool_names)
            self.assertIn("reiji_update_pool", tool_names)
            self.assertEqual(len(tools), 8)

            self.assertIn(3, by_id)
            self.assertIn("result", by_id[3])
            self.assertFalse(by_id[3]["result"].get("isError", False))

    def test_mcp_cli_read_only_hides_mutating_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rpc_messages = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            ]
            stdin_payload = "\n".join(json.dumps(m) for m in rpc_messages) + "\n"
            proc = _run_cli(
                ["--home", tmp, "mcp", "--framing", "ndjson", "--read-only"],
                stdin_text=stdin_payload,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
            responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
            by_id = {resp.get("id"): resp for resp in responses}
            tool_names = {t["name"] for t in by_id[2]["result"]["tools"]}
            self.assertNotIn("reiji_update_pool", tool_names)

    def test_import_rerank_memdream_bench_and_hybrid_recall_dual_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home_dir = Path(tmp) / "home"
            ckpt_dir = Path(tmp) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "2026-11-27_handoff.md").write_text(
                "# Release Handoff\n\n"
                "- Primary cache architecture uses MemoRRF and SQLite WAL.\n"
                "- Pending task: verify flagship v0.2.0 MCP bridge.\n",
                encoding="utf-8",
            )

            # Full-plugin mode: import -> hybrid_recall (provider) -> MemDream -> bench run
            with mock.patch.dict(os.environ, {"REIJI_MAIGO_DISABLE_PLUGINS": "0"}, clear=False):
                with ReijiMemoryEngine(home=home_dir, load_plugins=True) as engine:
                    self.assertEqual(set(BUILTIN_PLUGIN_FACTORIES.keys()), EXPECTED_BUILTIN_PLUGINS)
                    self.assertTrue(
                        EXPECTED_BUILTIN_PLUGINS.issubset(set(engine.status()["loaded_plugins"]))
                    )

                    importer = BootstrapImporter(engine)
                    imp_res = importer.import_checkpoints_dir(ckpt_dir)
                    self.assertTrue(imp_res.get("ok", True))

                    # Inspect imported seed cards via storage + MemoryCard contract
                    with engine.db.read_connection() as conn:
                        card_rows = [
                            dict(r)
                            for r in conn.execute("SELECT card_id, domain, title FROM memory_cards").fetchall()
                        ]
                    self.assertGreaterEqual(len(card_rows), 1)
                    seed_card_id = str(card_rows[0]["card_id"])
                    seed_card = engine.cards.get_card(seed_card_id)
                    self.assertIsNotNone(seed_card)
                    seed_domain = str(seed_card.domain)
                    seed_title = str(seed_card.title)
                    seed_tags = tuple(seed_card.tags)
                    self.assertGreaterEqual(len(seed_tags), 1)

                    # Engine facade delegates to active reranker provider
                    reranked = engine.hybrid_recall(
                        seed_title,
                        domain=seed_domain,
                        query_tags=seed_tags[:2],
                        top_k=3,
                    )
                    self.assertIsNotNone(reranked)
                    if isinstance(reranked, dict):
                        hits = reranked.get("results") or reranked.get("cards") or []
                        self.assertGreaterEqual(len(hits), 1)
                    else:
                        self.assertGreaterEqual(len(reranked), 1)

                    # Run MemDream self-probe pass
                    reranker = MemoRrfReranker()
                    dream_res = reranker.run_mem_dream(engine)
                    self.assertIsInstance(dream_res, dict)

                    # Run 6-dimension deterministic bench harness
                    harness = ReijiBenchHarness()
                    suite = harness.run_suite(mode="deterministic")
                    self.assertEqual(suite.get("total_cases"), 6, msg=f"bench={suite}")
                    self.assertEqual(suite.get("passed_cases"), 6, msg=f"bench={suite}")
                    self.assertEqual(suite.get("failed_cases"), 0, msg=f"bench={suite}")
                    self.assertEqual(suite.get("pass_rate"), 1.0, msg=f"bench={suite}")
                    md = harness.render_markdown_scorecard(suite)
                    self.assertIn("|", md)

            # Bare-kernel mode: engine.hybrid_recall seamlessly falls back to recall_cards
            with mock.patch.dict(os.environ, {"REIJI_MAIGO_DISABLE_PLUGINS": "1"}, clear=False):
                with ReijiMemoryEngine(home=home_dir, load_plugins=False) as bare_engine:
                    self.assertEqual(bare_engine.status()["loaded_plugins"], [])
                    fallback_cards = bare_engine.hybrid_recall(
                        seed_title,
                        domain=seed_domain,
                        query_tags=seed_tags[:2],
                        top_k=3,
                    )
                    self.assertIsInstance(fallback_cards, list)
                    self.assertGreaterEqual(len(fallback_cards), 1)
                    self.assertEqual(fallback_cards[0].card_id, seed_card_id)

            # Verify CLI bench run --markdown end-to-end
            bench_proc = _run_cli(
                ["--home", str(home_dir), "bench", "run", "--markdown"],
                env_overrides={"REIJI_MAIGO_DISABLE_PLUGINS": "0"},
            )
            self.assertEqual(bench_proc.returncode, 0, msg=f"stderr={bench_proc.stderr!r}")
            self.assertIn("|", bench_proc.stdout)

    def test_companion_mcp_config_helper_and_cli_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = render_mcp_server_config(python_exe="/usr/bin/python3", home=tmp)
            self.assertIn("mcpServers", cfg)
            self.assertIn("reiji-maigo", cfg["mcpServers"])
            entry = cfg["mcpServers"]["reiji-maigo"]
            self.assertEqual(entry["command"], "/usr/bin/python3")
            self.assertEqual(entry["args"][-1], "mcp")
            self.assertIn("--home", entry["args"])
            self.assertEqual(entry["env"]["REIJI_MAIGO_HOME"], str(Path(tmp)))

            proc = _run_cli(
                ["--home", tmp, "companion", "mcp-config", "--json"],
                env_overrides={"REIJI_MAIGO_DISABLE_PLUGINS": "0"},
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
            parsed = json.loads(proc.stdout)
            self.assertIn("mcpServers", parsed)
            self.assertIn("reiji-maigo", parsed["mcpServers"])
            cli_entry = parsed["mcpServers"]["reiji-maigo"]
            self.assertTrue(cli_entry["command"])
            self.assertIn("mcp", cli_entry["args"])

    def test_entry_points_direct_load_invokes_all_seven_plugin_slots(self) -> None:
        """Simulate `pip install -e .` where entry_points loads all 7 plugins before builtin fallback."""
        import importlib
        from reiji_maigo.core.kernel import ReijiKernel

        class _RealEp:
            def __init__(self, name: str, target: str) -> None:
                self.name = name
                self._target = target

            def load(self):
                mod_name, attr_name = self._target.split(":", 1)
                return getattr(importlib.import_module(mod_name), attr_name)

        pyproject_raw = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        eps: list[_RealEp] = []
        in_group = False
        for line in pyproject_raw.splitlines():
            s = line.strip()
            if s == '[project.entry-points."reiji_maigo.plugins"]':
                in_group = True
                continue
            if in_group and s.startswith("["):
                break
            if in_group and "=" in s:
                k, v = [p.strip() for p in s.split("=", 1)]
                eps.append(_RealEp(k, v.strip('"')))

        self.assertEqual({ep.name for ep in eps}, EXPECTED_BUILTIN_PLUGINS)
        with tempfile.TemporaryDirectory() as tmp:
            kernel = ReijiKernel(home=tmp, disable_plugins=False)
            records = kernel.load_plugins(entry_points_provider=lambda _grp: eps)
            self.assertEqual(len(records), 7)
            self.assertTrue(all(r.status == "loaded" and len(r.slots) >= 1 for r in records))
            for cmd in ("roi", "epoch", "radar", "companion", "rerank", "import", "bench"):
                self.assertIn(cmd, kernel.cli_commands)
            self.assertIsNotNone(kernel.get_provider("reranker"))

    def test_vault_portable_bundle_cross_home_export_import_and_audit_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home_src = Path(tmp) / "home_src"
            home_dst = Path(tmp) / "home_dst"
            bundle_zip = Path(tmp) / "portable_memory.zip"

            with ReijiMemoryEngine(home=home_src, load_plugins=False) as src_engine:
                t1 = src_engine.append_turn("ep_port", "user", "测试跨机安全归档导出与导入")
                src_engine.cards.create_card(
                    domain="proj",
                    card_type="decision",
                    title="跨机便携包验签",
                    state={"summary": "导出签名密钥并在新目录无损恢复"},
                    tags=["vault", "portable"],
                    spans=[SourceSpan("ep_port", t1, t1)],
                )

            key_proc = _run_cli(["--home", str(home_src), "vault", "export-key", "--json"])
            self.assertEqual(key_proc.returncode, 0, msg=key_proc.stderr)
            signing_key_hex = json.loads(key_proc.stdout)["signing_key_hex"]
            self.assertEqual(len(signing_key_hex), 64)

            exp_proc = _run_cli(["--home", str(home_src), "vault", "export", str(bundle_zip), "--json"])
            self.assertEqual(exp_proc.returncode, 0, msg=exp_proc.stderr)

            # Import into a brand-new --home directory WITHOUT needing --overwrite
            imp_proc = _run_cli(
                [
                    "--home",
                    str(home_dst),
                    "vault",
                    "import",
                    str(bundle_zip),
                    "--signing-key",
                    signing_key_hex,
                    "--json",
                ]
            )
            self.assertEqual(imp_proc.returncode, 0, msg=imp_proc.stderr)
            imp_data = json.loads(imp_proc.stdout)
            self.assertIn(imp_data["status"], {"imported", "overwritten"})

            # Adopted signing key in home_dst must verify both audit_chain and card_seals
            aud_proc = _run_cli(["--home", str(home_dst), "audit", "verify", "--json"])
            self.assertEqual(aud_proc.returncode, 0, msg=aud_proc.stderr)
            aud_data = json.loads(aud_proc.stdout)
            self.assertTrue(aud_data["ok"])
            self.assertTrue(aud_data["audit_chain"]["ok"])
            self.assertTrue(aud_data["card_seals"]["ok"])

            # Once home_dst has non-identical data, re-importing without --overwrite must fail safely
            _run_cli(
                [
                    "--home",
                    str(home_dst),
                    "turn",
                    "--episode",
                    "ep_local",
                    "--role",
                    "user",
                    "--content",
                    "本地新增对话防误覆盖",
                ]
            )
            refuse_proc = _run_cli(
                [
                    "--home",
                    str(home_dst),
                    "vault",
                    "import",
                    str(bundle_zip),
                    "--signing-key",
                    signing_key_hex,
                ]
            )
            self.assertEqual(refuse_proc.returncode, 2)

    def test_license_readme_and_pyproject_v020_metadata(self) -> None:
        license_path = REPO_ROOT / "LICENSE"
        readme_path = REPO_ROOT / "README.md"
        pyproject_path = REPO_ROOT / "pyproject.toml"

        self.assertTrue(license_path.is_file())
        self.assertTrue(readme_path.is_file())
        self.assertTrue(pyproject_path.is_file())

        license_text = license_path.read_text(encoding="utf-8")
        self.assertIn("PolyForm Noncommercial License 1.0.0", license_text)

        readme_text = readme_path.read_text(encoding="utf-8")
        self.assertIn("```mermaid", readme_text)
        self.assertIn("reiji mcp", readme_text)
        self.assertIn("reiji bench", readme_text)

        pyproject_raw = pyproject_path.read_text(encoding="utf-8")
        self.assertIn('license = { text = "PolyForm-Noncommercial-1.0.0" }', pyproject_raw)
        self.assertIn('version = "0.2.1"', pyproject_raw)
        self.assertIn('readme = "README.md"', pyproject_raw)

        if tomllib is not None:
            data = tomllib.loads(pyproject_raw)
            project = data["project"]
            self.assertEqual(project["version"], "0.2.1")
            self.assertEqual(project["readme"], "README.md")
            self.assertEqual(project["dependencies"], [])
            self.assertEqual(project["license"], {"text": "PolyForm-Noncommercial-1.0.0"})
            reiji_eps = project["entry-points"]["reiji_maigo.plugins"]
            self.assertEqual(set(reiji_eps.keys()), EXPECTED_BUILTIN_PLUGINS)

    def test_cleanroom_scan_zero_leaks_across_pkg_readme_and_agents(self) -> None:
        targets: list[Path] = sorted((REPO_ROOT / "reiji_maigo").rglob("*.py"))
        targets.append(REPO_ROOT / "README.md")
        targets.append(REPO_ROOT / "AGENTS.md")

        # Assemble forbidden patterns dynamically so this test file itself stays cleanroom-safe.
        forbidden_patterns: list[tuple[str, re.Pattern[str]]] = [
            ("win_user_path", re.compile(r"[A-Za-z]:[\\/]" + "Users" + r"[\\/]", re.IGNORECASE)),
            ("appdata", re.compile("App" + "Data", re.IGNORECASE)),
            ("private_orion", re.compile("orion" + "_private", re.IGNORECASE)),
            ("private_shinobu", re.compile("shinobu" + "_private", re.IGNORECASE)),
        ]

        violations: list[str] = []
        for path in targets:
            text = path.read_text(encoding="utf-8")
            rel = path.relative_to(REPO_ROOT)
            for label, pattern in forbidden_patterns:
                match = pattern.search(text)
                if match:
                    violations.append(f"{rel}: matched {label} ({match.group(0)!r})")

        self.assertEqual(violations, [], msg=f"Cleanroom violations found: {violations}")


if __name__ == "__main__":
    unittest.main()
