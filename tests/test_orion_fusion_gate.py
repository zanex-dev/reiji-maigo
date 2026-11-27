"""End-to-end fusion wiring, CLI, pulse telemetry, sibling host & clean-room gate tests."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

from reiji_maigo.bridges.star_panel import (
    RADAR_COOLDOWN_FRACTION,
    StarPanelBridgePlugin,
    create_plugin as create_star_panel_plugin,
)
from reiji_maigo.cli import build_parser, main as cli_main
from reiji_maigo.core.kernel import ReijiKernel
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.plugins import BUILTIN_PLUGIN_FACTORIES, register_builtin_plugins
from reiji_maigo.plugins.roi_ledger import RoiLedger
from reiji_maigo.swarm.coordinator import SwarmCoordinator

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "reiji_maigo"


def _load_pyproject() -> dict:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    raw = pyproject_path.read_bytes()
    try:
        import tomllib

        return tomllib.loads(raw.decode("utf-8"))
    except ModuleNotFoundError:
        # Python 3.10 fallback: parse needed sections minimally.
        text = raw.decode("utf-8")
        result: dict = {"project": {"dependencies": [], "entry-points": {}}}
        current_ep: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m_ep = re.match(r'^\[project\.entry-points\."([^"]+)"\]$', stripped)
            if m_ep:
                current_ep = m_ep.group(1)
                result["project"]["entry-points"][current_ep] = {}
                continue
            if stripped.startswith("["):
                current_ep = None
                continue
            if current_ep and "=" in stripped:
                k, v = [part.strip() for part in stripped.split("=", 1)]
                result["project"]["entry-points"][current_ep][k] = v.strip('"')
        return result


class _DuckPanelHost:
    """Headless duck-typed star-panel host used to verify StarPanelBridgePlugin."""

    def __init__(self) -> None:
        self.tabs: list[dict] = []
        self.badge_formatters: list = []
        self.radar_marker_providers: list = []
        self.radar_listeners: list = []

    def add_tab(self, tab_id: str, title: str, build_tab, *, on_select=None, on_refresh=None) -> None:
        self.tabs.append(
            {
                "tab_id": tab_id,
                "title": title,
                "build_tab": build_tab,
                "on_select": on_select,
                "on_refresh": on_refresh,
            }
        )

    def format_header_badge(self, fn) -> None:
        self.badge_formatters.append(fn)

    def add_radar_markers(self, fn) -> None:
        self.radar_marker_providers.append(fn)

    def on_radar_state_changed(self, fn) -> None:
        self.radar_listeners.append(fn)


class TestOrionFusionWiringAndGate(unittest.TestCase):
    """Verify engine bootstrap, CLI plugin parsers, pulse telemetry, and clean-room gates."""

    def test_engine_builtin_plugin_bootstrap_and_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"REIJI_MAIGO_DISABLE_PLUGINS": "0"}, clear=False):
                with ReijiMemoryEngine(home=tmp, load_plugins=True) as engine:
                    expected = set(BUILTIN_PLUGIN_FACTORIES.keys())
                    self.assertEqual(
                        expected,
                        {
                            "roi_ledger",
                            "epoch_vault",
                            "thermal_sync",
                            "companion",
                            "hybrid_reranker",
                            "bootstrap_importer",
                            "bench_harness",
                        },
                    )
                    loaded = engine.status()["loaded_plugins"]
                    self.assertEqual(len(loaded), len(set(loaded)))
                    self.assertTrue(expected.issubset(set(loaded)))
                    for cmd in ("roi", "epoch", "radar", "companion", "rerank", "import", "bench"):
                        self.assertIn(cmd, engine.kernel.cli_commands)

                    # Idempotent re-registration must skip already-loaded plugins.
                    second_pass = register_builtin_plugins(engine.kernel)
                    self.assertEqual(second_pass, ())
                    self.assertEqual(engine.status()["loaded_plugins"], loaded)

            with mock.patch.dict(os.environ, {"REIJI_MAIGO_DISABLE_PLUGINS": "1"}, clear=False):
                with ReijiMemoryEngine(home=tmp, load_plugins=True) as bare_engine:
                    status = bare_engine.status()
                    self.assertTrue(status["plugins_disabled"])
                    self.assertEqual(status["loaded_plugins"], [])
                    self.assertEqual(bare_engine.kernel.cli_commands, {})

    def test_cli_plugin_configure_parser_and_error_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp) / "companion_cfg"
            with mock.patch.dict(os.environ, {"REIJI_MAIGO_DISABLE_PLUGINS": "0"}, clear=False):
                # 1. Native subparser + legacy extra_args coexistence
                kernel = ReijiKernel(disable_plugins=False)
                register_builtin_plugins(kernel)
                kernel.register_cli("legacy_ping", lambda _e, _a: 0, help_text="Legacy command")
                parser = build_parser(plugin_commands=kernel.cli_commands)

                parsed_roi = parser.parse_args(["roi", "status", "--json"])
                self.assertEqual(parsed_roi.command, "roi")
                self.assertEqual(parsed_roi._plugin_spec.name, "roi")

                parsed_legacy = parser.parse_args(["legacy_ping", "sub", "--raw", "abc"])
                self.assertEqual(parsed_legacy.command, "legacy_ping")
                self.assertEqual(parsed_legacy.extra_args, ["sub", "--raw", "abc"])

                # 2. Execute all 4 built-in plugin CLI commands through cli_main
                for argv in (
                    ["--home", tmp, "roi", "status", "--json"],
                    ["--home", tmp, "epoch", "scan", "--json"],
                    ["--home", tmp, "radar", "status", "--json"],
                    ["--home", tmp, "companion", "status", "--config-dir", str(cfg_dir), "--json"],
                ):
                    out_buf = io.StringIO()
                    with contextlib.redirect_stdout(out_buf):
                        code = cli_main(argv)
                    self.assertEqual(code, 0, f"CLI failed for {argv}: {out_buf.getvalue()}")
                    payload = json.loads(out_buf.getvalue())
                    self.assertIsInstance(payload, dict)

                # 3. Verify unified exit code 2 on plugin handler ValueError
                def _boom_handler(_engine, _args):
                    raise ValueError("simulated plugin validation failure")

                orig_register = register_builtin_plugins

                def _register_with_boom(k: ReijiKernel):
                    recs = orig_register(k)
                    k.register_cli("boom", _boom_handler, help_text="Boom test")
                    return recs

                err_buf = io.StringIO()
                with mock.patch("reiji_maigo.plugins.register_builtin_plugins", side_effect=_register_with_boom):
                    with contextlib.redirect_stderr(err_buf):
                        exit_code = cli_main(["--home", tmp, "boom"])
                self.assertEqual(exit_code, 2)
                self.assertIn("simulated plugin validation failure", err_buf.getvalue())

    def test_swarm_pulse_economics_and_cold_vault_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"REIJI_MAIGO_DISABLE_PLUGINS": "0"}, clear=False):
                with ReijiMemoryEngine(home=tmp, load_plugins=True) as engine:
                    ledger = RoiLedger(engine.db)
                    ledger.record_window_saving(
                        "ep_fusion_test",
                        saved_tokens=12_500,
                        cliff_200k_avoided=True,
                    )

                    coord = SwarmCoordinator(engine)
                    summary = coord.pulse_summary()
                    self.assertIn("economics", summary)
                    self.assertIn("cold_vault", summary)

                    econ = summary["economics"]
                    self.assertRegex(str(econ["flash_cost_usd"]), r"^-?\d+\.\d{7}$")
                    self.assertRegex(str(econ["gross_saved_usd"]), r"^-?\d+\.\d{7}$")
                    self.assertRegex(str(econ["net_roi_usd"]), r"^-?\d+\.\d{7}$")
                    self.assertGreaterEqual(int(econ["entries_count"]), 1)

                    banner = coord.render_pulse_banner()
                    self.assertIn("  Economics   :", banner)
                    self.assertIn("  Cold Vault  :", banner)
                    self.assertIn(str(econ["gross_saved_usd"]), banner)

                out_buf = io.StringIO()
                with contextlib.redirect_stdout(out_buf):
                    code = cli_main(["--home", tmp, "swarm", "status"])
                self.assertEqual(code, 0)
                cli_output = out_buf.getvalue()
                self.assertIn("  Economics   :", cli_output)
                self.assertIn("  Cold Vault  :", cli_output)

    def test_pyproject_entry_points_and_zero_dependencies(self) -> None:
        data = _load_pyproject()
        project = data["project"]
        self.assertEqual(project.get("dependencies", []), [])

        entry_points = project.get("entry-points", {})
        reiji_eps = entry_points.get("reiji_maigo.plugins", {})
        self.assertEqual(
            set(reiji_eps.keys()),
            {
                "roi_ledger",
                "epoch_vault",
                "thermal_sync",
                "companion",
                "hybrid_reranker",
                "bootstrap_importer",
                "bench_harness",
            },
        )
        for ep_name, target in reiji_eps.items():
            mod_name, attr_name = target.split(":", 1)
            mod = importlib.import_module(mod_name)
            factory = getattr(mod, attr_name)
            plugin = factory()
            self.assertEqual(getattr(plugin, "name", None), ep_name)

        star_panel_group = "acerola" + "_orion.plugins"
        star_eps = entry_points.get(star_panel_group, {})
        self.assertEqual(
            star_eps.get("reiji_maigo"),
            "reiji_maigo.bridges.star_panel:create_plugin",
        )
        mod_name, attr_name = star_eps["reiji_maigo"].split(":", 1)
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, attr_name)
        self.assertIs(factory, create_star_panel_plugin)

    def test_star_panel_bridge_duck_typed_host_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = create_star_panel_plugin(home=tmp)
            try:
                self.assertIsInstance(plugin, StarPanelBridgePlugin)
                host = _DuckPanelHost()
                plugin.register_ui(host)

                self.assertEqual(len(host.tabs), 1)
                self.assertEqual(host.tabs[0]["tab_id"], "reiji")
                self.assertEqual(len(host.badge_formatters), 1)
                self.assertEqual(len(host.radar_marker_providers), 1)
                self.assertEqual(len(host.radar_listeners), 1)

                badge = host.badge_formatters[0]()
                self.assertIn("text", badge)
                markers = host.radar_marker_providers[0]()
                self.assertTrue(
                    any(
                        abs(float(m.get("fraction", -1.0)) - RADAR_COOLDOWN_FRACTION) < 1e-6
                        for m in markers
                    )
                )
                state = plugin.collect_deck_state()
                self.assertIn("pool", state)
                self.assertIn("roi", state)
                self.assertIn("vault", state)
            finally:
                plugin.close()

    def test_sibling_star_panel_host_integration_if_present(self) -> None:
        sibling_dir = REPO_ROOT.parent / ("acerola" + "-orion-public")
        if not sibling_dir.is_dir():
            self.skipTest("Sibling star-panel repository not present")

        pkg_name = "acerola" + "_orion"
        inserted = False
        if str(sibling_dir) not in sys.path:
            sys.path.insert(0, str(sibling_dir))
            inserted = True

        host_cls = None
        candidate_modules = (
            f"{pkg_name}.ui.plugin_host",
            f"{pkg_name}.panel_host",
            f"{pkg_name}.core.panel_host",
            f"{pkg_name}.plugins.panel_host",
            f"{pkg_name}.ui.panel_host",
        )
        try:
            for mod_name in candidate_modules:
                try:
                    mod = importlib.import_module(mod_name)
                except Exception:  # noqa: BLE001
                    continue
                cls = getattr(mod, "PanelPluginHost", None)
                if cls is not None:
                    host_cls = cls
                    break

            if host_cls is None:
                self.skipTest("Sibling PanelPluginHost symbol not exported in known modules")

            with tempfile.TemporaryDirectory() as tmp:
                plugin = create_star_panel_plugin(home=tmp)
                try:
                    try:
                        host = host_cls()
                    except TypeError:
                        self.skipTest("Sibling PanelPluginHost requires host-specific constructor args")
                    plugin.register_ui(host)
                    # Host must remain healthy after registration
                    self.assertIsNotNone(plugin.collect_deck_state())
                finally:
                    plugin.close()
        finally:
            if inserted and str(sibling_dir) in sys.path:
                sys.path.remove(str(sibling_dir))

    def test_agents_md_and_package_clean_room_gate(self) -> None:
        forbidden_path_patterns = (
            re.compile(r"[A-Za-z]:\\\\Users\\\\", re.IGNORECASE),
            re.compile(r"[A-Za-z]:/Users/", re.IGNORECASE),
            re.compile(r"/Users/[A-Za-z0-9._-]+"),
            re.compile(r"/home/[A-Za-z0-9._-]+"),
            re.compile(r"AppData", re.IGNORECASE),
        )
        forbidden_private_tokens = (
            "shinobu" + "_private",
            "orion" + "_private",
            "acerola" + "_orion",
        )

        targets = [REPO_ROOT / "AGENTS.md", *sorted(PKG_ROOT.rglob("*.py"))]
        for target in targets:
            source = target.read_text(encoding="utf-8")
            rel = target.relative_to(REPO_ROOT)
            for pattern in forbidden_path_patterns:
                self.assertIsNone(
                    pattern.search(source),
                    f"Host path leak detected in {rel}: pattern={pattern.pattern}",
                )
            lower_src = source.lower()
            for token in forbidden_private_tokens:
                self.assertNotIn(
                    token,
                    lower_src,
                    f"Forbidden token {token!r} leaked in {rel}",
                )


if __name__ == "__main__":
    unittest.main()
