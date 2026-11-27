"""Unit tests for reiji_maigo.plugins registry and StarPanelBridgePlugin."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reiji_maigo.bridges.star_panel import (
    PLUGIN_DESCRIPTION,
    PLUGIN_NAME,
    RADAR_COOLDOWN_FRACTION,
    StarPanelBridgePlugin,
    TAB_ID,
    TAB_TITLE,
    create_plugin,
)
from reiji_maigo.core.contracts import MemoryCard, SourceSpan
from reiji_maigo.core.kernel import ReijiKernel
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.plugins import BUILTIN_PLUGIN_FACTORIES, register_builtin_plugins
from reiji_maigo.plugins.roi_ledger import RoiLedger


class _MockUiHost:
    """Headless mock desktop host implementing all four duck-typed UI slots."""

    def __init__(self) -> None:
        self.tabs: list[dict[str, object]] = []
        self.badge_callbacks: list[object] = []
        self.radar_marker_callbacks: list[object] = []
        self.radar_state_callbacks: list[object] = []

    def add_tab(
        self,
        tab_id: str,
        title: str,
        builder: object,
        *,
        on_select: object = None,
        on_refresh: object = None,
    ) -> None:
        self.tabs.append(
            {
                "tab_id": tab_id,
                "title": title,
                "builder": builder,
                "on_select": on_select,
                "on_refresh": on_refresh,
            }
        )

    def format_header_badge(self, fn: object) -> None:
        self.badge_callbacks.append(fn)

    def add_radar_markers(self, fn: object) -> None:
        self.radar_marker_callbacks.append(fn)

    def on_radar_state_changed(self, fn: object) -> None:
        self.radar_state_callbacks.append(fn)


class _PartialUiHost:
    """Host implementing only badge and radar marker slots."""

    def __init__(self) -> None:
        self.badge_callbacks: list[object] = []
        self.radar_marker_callbacks: list[object] = []

    def format_header_badge(self, fn: object) -> None:
        self.badge_callbacks.append(fn)

    def add_radar_markers(self, fn: object) -> None:
        self.radar_marker_callbacks.append(fn)


class _EmptyUiHost:
    """Host exposing no UI slots at all."""


def _make_protected_domain(_engine: ReijiMemoryEngine) -> str:
    return "companion"


def _seed_pending_l1_review(
    engine: ReijiMemoryEngine,
    *,
    title: str = "Core Identity Rule",
    op_kind: str = "revoke",
    actor: str = "scribe_bot",
    reason: str = "non-admin requested protected change",
) -> tuple[MemoryCard, dict[str, object]]:
    domain = _make_protected_domain(engine)
    span = SourceSpan(episode_id="ep_panel", turn_start=1, turn_end=2)
    card = engine.cards.create_card(
        domain=domain,
        card_type="rule",
        title=title,
        state={"rule": "keep memory truthful"},
        tags=["protected"],
        spans=[span],
    )
    replacement = None
    if op_kind == "supersede":
        replacement = MemoryCard(
            card_id=f"{card.card_id}_v2",
            domain=domain,
            card_type="rule",
            title=f"{title} v2",
            state={"rule": "updated truthful rule"},
            lifecycle_status="active",
            valid_from="2026-11-27T00:00:00+00:00",
            valid_to=None,
            superseded_by=None,
            tags=("protected",),
            spans=(span,),
        )
    decision = engine.approval_gate.guard_supersede_or_revoke(
        engine.cards,
        actor=actor,
        op_kind=op_kind,
        target_card_id=card.card_id,
        replacement_card=replacement,
        origin_trust="assistant",
        reason=reason,
    )
    return card, decision


class TestBuiltinPluginRegistry(unittest.TestCase):
    def test_builtin_factories_expose_all_seven_plugins(self) -> None:
        self.assertEqual(
            set(BUILTIN_PLUGIN_FACTORIES.keys()),
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
        for key, factory in BUILTIN_PLUGIN_FACTORIES.items():
            plugin = factory()
            self.assertEqual(getattr(plugin, "name", None), key)

    def test_register_builtin_plugins_loads_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kernel = ReijiKernel(home=Path(tmp), disable_plugins=False)
            first = register_builtin_plugins(kernel)
            self.assertEqual(len(first), 7)
            self.assertTrue(all(r.status == "loaded" for r in first))
            self.assertTrue(all(r.source == "builtin" for r in first))

            for cmd_name in ("roi", "epoch", "radar", "companion", "rerank", "import", "bench"):
                self.assertIn(cmd_name, kernel.cli_commands)

            # Second call skips already-registered plugins
            second = register_builtin_plugins(kernel)
            self.assertEqual(second, ())
            self.assertEqual(len(kernel.plugin_records), 7)

    def test_register_builtin_plugins_respects_disable_switches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            disabled_all = ReijiKernel(home=Path(tmp), disable_plugins=True)
            self.assertEqual(register_builtin_plugins(disabled_all), ())
            self.assertEqual(disabled_all.plugin_records, {})

            disabled_one = ReijiKernel(
                home=Path(tmp),
                disable_plugins=False,
                disabled_plugins=("companion",),
            )
            records = register_builtin_plugins(disabled_one)
            by_name = {r.name: r for r in records}
            self.assertEqual(by_name["companion"].status, "skipped")
            self.assertEqual(by_name["roi_ledger"].status, "loaded")
            self.assertNotIn("companion", disabled_one.cli_commands)

    def test_register_builtin_plugins_isolates_factory_failure(self) -> None:
        def _broken_factory() -> object:
            raise RuntimeError("simulated factory crash")

        with tempfile.TemporaryDirectory() as tmp:
            kernel = ReijiKernel(home=Path(tmp), disable_plugins=False)
            records = register_builtin_plugins(
                kernel,
                factories={
                    "broken_one": _broken_factory,
                    "roi_ledger": BUILTIN_PLUGIN_FACTORIES["roi_ledger"],
                },
            )
            by_name = {r.name: r for r in records}
            self.assertEqual(by_name["broken_one"].status, "failed")
            self.assertIn("simulated factory crash", by_name["broken_one"].error or "")
            self.assertEqual(by_name["roi_ledger"].status, "loaded")
            self.assertEqual(len(kernel.load_errors), 1)
            self.assertEqual(kernel.load_errors[0].name, "broken_one")


class TestStarPanelBridgePlugin(unittest.TestCase):
    def test_zero_top_level_tkinter_import(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; "
            "import reiji_maigo.bridges.star_panel as sp; "
            "assert 'tkinter' not in sys.modules, 'tkinter imported at module level'; "
            "assert sp.PLUGIN_NAME == 'reiji_maigo'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_deck_state_and_l1_approve_reject_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "reiji_home"
            plugin = create_plugin(home=home)
            try:
                self.assertEqual(plugin.name, PLUGIN_NAME)
                self.assertEqual(plugin.description, PLUGIN_DESCRIPTION)

                engine = plugin.engine
                # Seed an active pool event so pool metrics and badge are non-zero
                engine.pool.create_event(
                    "Build Star Panel Deck",
                    "Mount duck-typed bridge",
                    SourceSpan(episode_id="ep_panel", turn_start=1, turn_end=1),
                )
                # Seed ROI savings so 7-decimal metrics are non-zero
                ledger = RoiLedger(engine.db)
                ledger.record_window_saving(
                    "ep_panel",
                    10_000,
                    cliff_200k_avoided=True,
                )

                # Seed two pending L1 reviews: one to approve (revoke), one to reject
                card_a, dec_a = _seed_pending_l1_review(
                    engine,
                    title="Protected Policy A",
                    op_kind="revoke",
                )
                card_b, dec_b = _seed_pending_l1_review(
                    engine,
                    title="Protected Policy B",
                    op_kind="supersede",
                )
                self.assertEqual(dec_a["status"], "pending_approval")
                self.assertEqual(dec_b["status"], "pending_approval")
                rev_a = str(dec_a["review_id"])
                rev_b = str(dec_b["review_id"])

                state = plugin.collect_deck_state()
                self.assertEqual(state["pool"]["active_count"], 1)
                self.assertEqual(state["l1_reviews"]["pending_count"], 2)
                self.assertEqual(state["cards"]["active"], 2)
                self.assertEqual(state["vault"]["epochs_count"], 0)
                self.assertTrue(state["audit"]["ok"])
                self.assertEqual(len(state["roi"]["net_roi_usd"].split(".")[1]), 7)
                self.assertGreater(float(state["roi"]["net_roi_usd"]), 0.0)

                # Approve rev_a -> card_a revoked
                approved = plugin.approve_l1(rev_a, approver="panel_admin")
                self.assertEqual(approved["status"], "approved")
                self.assertTrue(approved["revoked"])
                refreshed_a = engine.cards.get_card(card_a.card_id)
                self.assertIsNotNone(refreshed_a)
                self.assertEqual(refreshed_a.lifecycle_status, "revoked")

                # Reject rev_b -> card_b untouched
                rejected = plugin.reject_l1(
                    rev_b,
                    reason="not verified",
                    reviewer="panel_admin",
                )
                self.assertEqual(rejected["status"], "rejected")
                self.assertEqual(rejected["reason"], "not verified")
                refreshed_b = engine.cards.get_card(card_b.card_id)
                self.assertIsNotNone(refreshed_b)
                self.assertEqual(refreshed_b.lifecycle_status, "active")

                after_state = plugin.collect_deck_state()
                self.assertEqual(after_state["l1_reviews"]["pending_count"], 0)
                self.assertTrue(after_state["audit"]["ok"])
            finally:
                plugin.close()

    def test_headless_mock_ui_host_registration_and_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = create_plugin(home=Path(tmp) / "reiji_home")
            try:
                host = _MockUiHost()
                mounted = plugin.register_ui(host)
                self.assertEqual(
                    mounted,
                    {
                        "add_tab": True,
                        "format_header_badge": True,
                        "add_radar_markers": True,
                        "on_radar_state_changed": True,
                    },
                )
                self.assertEqual(len(host.tabs), 1)
                tab = host.tabs[0]
                self.assertEqual(tab["tab_id"], TAB_ID)
                self.assertEqual(tab["title"], TAB_TITLE)
                self.assertTrue(callable(tab["builder"]))
                self.assertTrue(callable(tab["on_select"]))
                self.assertTrue(callable(tab["on_refresh"]))

                badge_fn = host.badge_callbacks[0]
                badge = badge_fn()
                self.assertEqual(badge["fg"], "#f59e0b")
                self.assertEqual(badge["bg"], "#1e1b18")
                self.assertTrue(badge["text"].startswith("🕰️ Pool "))
                self.assertIn("· L1:0 · ROI ", badge["text"])

                markers_fn = host.radar_marker_callbacks[0]
                markers = markers_fn()
                self.assertEqual(
                    markers,
                    [
                        {
                            "fraction": RADAR_COOLDOWN_FRACTION,
                            "color": "#f59e0b",
                            "width": 1.5,
                            "dash": (3, 2),
                            "tag": "reiji_cooldown",
                        }
                    ],
                )

                radar_cb = host.radar_state_callbacks[0]
                radar_cb({"phase": "warning", "remaining_ttl_sec": 42.0})
                state = plugin.collect_deck_state()
                self.assertEqual(state["radar"]["last_state"]["phase"], "warning")
                self.assertEqual(state["radar"]["last_state"]["remaining_ttl_sec"], 42.0)

                # Partial and empty hosts degrade gracefully without raising
                partial = _PartialUiHost()
                self.assertEqual(
                    plugin.register_ui(partial),
                    {
                        "add_tab": False,
                        "format_header_badge": True,
                        "add_radar_markers": True,
                        "on_radar_state_changed": False,
                    },
                )
                empty = _EmptyUiHost()
                self.assertEqual(
                    plugin.register_ui(empty),
                    {
                        "add_tab": False,
                        "format_header_badge": False,
                        "add_radar_markers": False,
                        "on_radar_state_changed": False,
                    },
                )
            finally:
                plugin.close()

    def test_cli_and_hooks_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = create_plugin(home=Path(tmp) / "reiji_home")
            try:
                # Unrelated args return None
                unrelated = argparse.Namespace(command="other")
                self.assertIsNone(plugin.handle_cli(unrelated))

                parser = argparse.ArgumentParser(prog="host")
                subparsers = parser.add_subparsers(dest="command")
                plugin.register_cli(subparsers)

                _, dec = _seed_pending_l1_review(plugin.engine, op_kind="revoke")
                rev_id = str(dec["review_id"])

                # status --json
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = plugin.handle_cli(parser.parse_args(["reiji", "status", "--json"]))
                self.assertEqual(rc, 0)
                status_payload = json.loads(buf.getvalue())
                self.assertEqual(status_payload["l1_reviews"]["pending_count"], 1)

                # l1-approve --json
                buf2 = io.StringIO()
                with redirect_stdout(buf2):
                    rc2 = plugin.handle_cli(
                        parser.parse_args(
                            ["reiji", "l1-approve", rev_id, "--approver", "cli_admin", "--json"]
                        )
                    )
                self.assertEqual(rc2, 0)
                approved_payload = json.loads(buf2.getvalue())
                self.assertEqual(approved_payload["status"], "approved")
                self.assertEqual(approved_payload["approved_by"], "cli_admin")

                # Hooks integration with ReijiKernel
                kernel = ReijiKernel(home=Path(tmp) / "hook_home")
                plugin.register_hooks(kernel)
                kernel.hooks.emit(
                    "thermal.handoff_ready",
                    {"phase": "cold", "remaining_ttl_sec": 0.0},
                )
                self.assertEqual(
                    plugin.collect_deck_state()["radar"]["last_state"]["phase"],
                    "cold",
                )
            finally:
                plugin.close()

    def test_real_tk_tab_when_display_available(self) -> None:
        # Run real Tk UI verification in an isolated subprocess so Tcl/Tk interpreter
        # teardown never collides with subsequent multi-threaded unittest modules.
        repo_root = Path(__file__).resolve().parents[1]
        script = (
            "import sys, tempfile\n"
            "from pathlib import Path\n"
            "try:\n"
            "    import tkinter as tk\n"
            "except Exception as exc:\n"
            "    print(f'SKIP:tkinter unavailable: {exc}')\n"
            "    sys.exit(77)\n"
            "try:\n"
            "    root = tk.Tk()\n"
            "    root.withdraw()\n"
            "except Exception as exc:\n"
            "    print(f'SKIP:Tk display unavailable in this environment: {exc}')\n"
            "    sys.exit(77)\n"
            "from reiji_maigo.bridges.star_panel import create_plugin\n"
            "from tests.test_star_panel_bridge import _MockUiHost, _seed_pending_l1_review\n"
            "with tempfile.TemporaryDirectory() as tmp:\n"
            "    plugin = create_plugin(home=Path(tmp) / 'reiji_tk_home')\n"
            "    try:\n"
            "        _seed_pending_l1_review(plugin.engine, title='Tk Policy 1', op_kind='revoke')\n"
            "        _seed_pending_l1_review(plugin.engine, title='Tk Policy 2', op_kind='revoke')\n"
            "        host = _MockUiHost()\n"
            "        plugin.register_ui(host)\n"
            "        builder = host.tabs[0]['builder']\n"
            "        frame = builder(root)\n"
            "        root.update_idletasks()\n"
            "        l1_listbox = frame._reiji_l1_listbox\n"
            "        assert l1_listbox.size() == 2, f'expected 2, got {l1_listbox.size()}'\n"
            "        l1_listbox.selection_set(0)\n"
            "        res_approve = frame._reiji_approve_selected()\n"
            "        assert res_approve is not None and res_approve['status'] == 'approved'\n"
            "        assert l1_listbox.size() == 1\n"
            "        frame._reiji_reason_entry.insert(0, 'rejected from tk test')\n"
            "        l1_listbox.selection_set(0)\n"
            "        res_reject = frame._reiji_reject_selected()\n"
            "        assert res_reject is not None and res_reject['status'] == 'rejected'\n"
            "        assert l1_listbox.size() == 0\n"
            "    finally:\n"
            "        plugin.close()\n"
            "        try:\n"
            "            root.destroy()\n"
            "        except Exception:\n"
            "            pass\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 77:
            msg = (proc.stdout or proc.stderr or "Tk unavailable").strip()
            self.skipTest(msg.removeprefix("SKIP:"))
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"Real Tk subprocess test failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )

    def test_clean_room_source_scan(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        forbidden = (
            "acerola" + "_orion",
            "orion" + "_private",
            "shinobu" + "_private",
            "App" + "Data",
        )
        targets = (
            repo_root / "reiji_maigo" / "plugins" / "__init__.py",
            repo_root / "reiji_maigo" / "bridges" / "star_panel.py",
        )
        for path in targets:
            content = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(
                    token.lower(),
                    content.lower(),
                    msg=f"Forbidden clean-room token found in {path.name}",
                )


if __name__ == "__main__":
    unittest.main()
