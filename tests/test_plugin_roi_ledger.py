"""Unit tests for reiji_maigo.plugins.roi_ledger (7-decimal anti-MemoryTax ledger)."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from decimal import Decimal
import io
import json
from pathlib import Path
import re
import unittest

from reiji_maigo.storage import ReijiDatabase
from reiji_maigo.core.kernel import ReijiKernel
from reiji_maigo.plugins.roi_ledger import (
    DEFAULT_PRICING,
    RoiLedger,
    RoiLedgerPlugin,
    USD_QUANT,
    create_plugin,
    ensure_roi_tables,
)
from reiji_maigo.swarm.guards import record_swarm_telemetry

_SEVEN_DECIMAL_RE = re.compile(r"^-?\d+\.\d{7}$")


class TestRoiLedgerPlugin(unittest.TestCase):
    def setUp(self) -> None:
        self.db = ReijiDatabase(":memory:")

    def tearDown(self) -> None:
        self.db.close()

    def test_ensure_roi_tables_is_idempotent(self) -> None:
        ensure_roi_tables(self.db)
        ensure_roi_tables(self.db)

        with self.db.read_connection() as conn:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            self.assertIn("roi_ledger_entries", tables)
            self.assertIn("roi_pricing_config", tables)

            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(roi_ledger_entries)").fetchall()
            }
            expected_cols = {
                "source_telemetry_id",
                "role",
                "prompt_tokens",
                "cached_l0_tokens",
                "completion_tokens",
                "saved_flagship_tokens",
                "cliff_200k_avoided",
                "flash_cost_nano",
                "gross_saved_nano",
                "created_at",
            }
            self.assertTrue(expected_cols.issubset(cols))

    def test_sync_from_swarm_telemetry_precision_and_l0_cache_discount(self) -> None:
        ledger = RoiLedger(self.db)
        # Before swarm_telemetry exists, sync is a safe no-op.
        self.assertEqual(ledger.sync_from_swarm_telemetry(), 0)

        # Scribe: 10,000 prompt tokens with 8,000 L0 cached + 1,000 completion.
        # uncached = 2,000 * 0.15 / 1e6 = 0.0003000
        # cached   = 8,000 * 0.0375 / 1e6 = 0.0003000
        # output   = 1,000 * 0.60 / 1e6 = 0.0006000
        # scribe total = 0.0012000
        record_swarm_telemetry(
            self.db,
            role="scribe",
            action="digest",
            metrics={
                "prompt_tokens": 10_000,
                "cached_l0_tokens": 8_000,
                "completion_tokens": 1_000,
            },
        )
        # Scout: 4,000 uncached prompt + 500 output; scanned 12,000 -> distilled 2,000 => saved 10,000.
        # flash = 4,000 * 0.15 / 1e6 + 500 * 0.60 / 1e6 = 0.0006000 + 0.0003000 = 0.0009000
        # saved = 10,000 * 2.50 / 1e6 = 0.0250000
        record_swarm_telemetry(
            self.db,
            role="scout",
            action="deep_recall",
            metrics={
                "prompt_tokens": 4_000,
                "completion_tokens": 500,
                "scanned_raw_tokens": 12_000,
                "distilled_tokens": 2_000,
            },
        )
        # Archivist: 1 token prompt to verify sub-micro precision handling.
        record_swarm_telemetry(
            self.db,
            role="archivist",
            action="compact",
            metrics={
                "prompt_tokens": 1,
                "completion_tokens": 0,
            },
        )

        self.assertEqual(ledger.sync_from_swarm_telemetry(), 3)
        # Idempotent on repeated sync (UNIQUE source_telemetry_id).
        self.assertEqual(ledger.sync_from_swarm_telemetry(), 0)

        summary = ledger.summary()
        for key in ("flash_cost_usd", "gross_saved_usd", "net_roi_usd", "roi_multiple"):
            self.assertRegex(summary[key], _SEVEN_DECIMAL_RE)

        # Total flash = 0.0012000 + 0.0009000 + 0.00000015 -> rounded to 0.0021002
        self.assertEqual(summary["flash_cost_usd"], "0.0021002")
        self.assertEqual(summary["gross_saved_usd"], "0.0250000")
        self.assertEqual(summary["net_roi_usd"], "0.0228999")
        self.assertFalse(summary["tax_alert"])
        self.assertEqual(summary["entries_count"], 3)

    def test_window_saving_cliff_200k_and_memory_tax_alert(self) -> None:
        ledger = RoiLedger(self.db)
        # Incur Flash cost with zero savings -> MemoryTax alert triggers.
        record_swarm_telemetry(
            self.db,
            role="scribe",
            action="digest",
            metrics={"prompt_tokens": 100_000, "completion_tokens": 20_000},
        )
        ledger.sync_from_swarm_telemetry()
        # Flash cost = 100k*0.15/1e6 + 20k*0.60/1e6 = 0.015 + 0.012 = 0.0270000
        before = ledger.summary()
        self.assertEqual(before["flash_cost_usd"], "0.0270000")
        self.assertEqual(before["gross_saved_usd"], "0.0000000")
        self.assertEqual(before["net_roi_usd"], "-0.0270000")
        self.assertTrue(before["tax_alert"])

        # Record normal main-window saving: 5,000 tokens * 2.50 / 1e6 = 0.0125000 (still < 0.027).
        rec1 = ledger.record_window_saving("ep-001", 5_000, cliff_200k_avoided=False)
        self.assertEqual(rec1["gross_saved_usd"], "0.0125000")
        mid = ledger.summary()
        self.assertTrue(mid["tax_alert"])

        # Record 200K pricing-cliff avoided saving: 5,000 * 2.50 * 2.0 / 1e6 = 0.0250000.
        rec2 = ledger.record_window_saving(
            "ep-002",
            5_000,
            cliff_200k_avoided=True,
            now_iso="2026-11-27T08:00:00Z",
        )
        self.assertEqual(rec2["gross_saved_usd"], "0.0250000")
        self.assertTrue(rec2["cliff_200k_avoided"])
        self.assertEqual(rec2["created_at"], "2026-11-27T08:00:00Z")

        after = ledger.summary()
        # Total saved = 0.0375000 > 0.0270000 => net = 0.0105000, alert cleared.
        self.assertEqual(after["gross_saved_usd"], "0.0375000")
        self.assertEqual(after["net_roi_usd"], "0.0105000")
        self.assertFalse(after["tax_alert"])
        self.assertGreater(Decimal(after["roi_multiple"]), Decimal("1.0000000"))

        with self.assertRaises(ValueError):
            ledger.record_window_saving("", 100)
        with self.assertRaises(ValueError):
            ledger.record_window_saving("ep-bad", -1)
        with self.assertRaises(ValueError):
            ledger.record_window_saving("ep-bool", True)  # type: ignore[arg-type]

    def test_reconcile_detects_drift_and_missing_rows(self) -> None:
        ledger = RoiLedger(self.db)
        tid1 = record_swarm_telemetry(
            self.db,
            role="scribe",
            action="digest",
            metrics={"prompt_tokens": 50_000, "cached_l0_tokens": 20_000, "completion_tokens": 2_000},
        )
        # Before sync, reconcile flags missing row.
        pre = ledger.reconcile()
        self.assertFalse(pre["ok"])
        self.assertEqual(pre["missing_in_ledger"], [tid1])

        ledger.sync_from_swarm_telemetry()
        ok_report = ledger.reconcile()
        self.assertTrue(ok_report["ok"])
        self.assertEqual(ok_report["drift_usd"], "0.0000000")
        self.assertLessEqual(Decimal(ok_report["drift_usd"]), USD_QUANT)

        # Tamper within tolerance (50 nano-USD = 0.00000005 USD <= 0.0000001 USD).
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE roi_ledger_entries SET flash_cost_nano = flash_cost_nano + 50 WHERE source_telemetry_id = ?",
                (tid1,),
            )
        within_tol = ledger.reconcile()
        self.assertTrue(within_tol["ok"])

        # Tamper beyond tolerance (+500 nano-USD total = 0.00000055 USD > 0.0000001 USD).
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE roi_ledger_entries SET flash_cost_nano = flash_cost_nano + 500 WHERE source_telemetry_id = ?",
                (tid1,),
            )
        drifted = ledger.reconcile()
        self.assertFalse(drifted["ok"])
        self.assertGreater(Decimal(drifted["drift_usd"]), USD_QUANT)

    def test_set_pricing_persists_and_validates(self) -> None:
        ledger = RoiLedger(self.db)
        updated = ledger.set_pricing(
            flash_input="0.20",
            flash_cached_l0="0.05",
            flagship_input=Decimal("3.00"),
            cliff_200k_multiplier="2.5",
        )
        self.assertEqual(updated["flash_input"], "0.20")
        self.assertEqual(updated["flash_cached_l0"], "0.05")
        self.assertEqual(updated["flagship_input"], "3.00")
        self.assertEqual(updated["cliff_200k_multiplier"], "2.5")
        self.assertEqual(updated["flash_output"], str(DEFAULT_PRICING["flash_output"]))

        # Reload with fresh RoiLedger instance to confirm DB persistence.
        reloaded = RoiLedger(self.db)
        rec = reloaded.record_window_saving("ep-price", 10_000, cliff_200k_avoided=True)
        # 10,000 * 3.00 * 2.5 / 1e6 = 0.0750000
        self.assertEqual(rec["gross_saved_usd"], "0.0750000")

        with self.assertRaises(ValueError):
            ledger.set_pricing(unknown_rate="1.0")
        with self.assertRaises(ValueError):
            ledger.set_pricing(flash_input="-0.1")
        with self.assertRaises(ValueError):
            ledger.set_pricing(cliff_200k_multiplier="0.5")

    def test_plugin_registration_cli_parser_extra_args_and_hooks(self) -> None:
        kernel = ReijiKernel()
        plugin = create_plugin()
        self.assertIsInstance(plugin, RoiLedgerPlugin)
        self.assertEqual(plugin.name, "roi_ledger")

        plugin.register_cli(kernel)
        plugin.register_hooks(kernel)

        spec = kernel.get_cli_command("roi")
        self.assertIsNotNone(spec)
        assert spec is not None
        handler = spec.handler
        self.assertTrue(callable(getattr(handler, "configure_parser", None)))

        # 1) configure_parser mode across all 4 subcommands.
        root = argparse.ArgumentParser(prog="reiji-maigo")
        subs = root.add_subparsers(dest="cmd")
        roi_sub = subs.add_parser("roi")
        handler.configure_parser(roi_sub)

        buf = io.StringIO()
        args_price = root.parse_args(
            ["roi", "set-pricing", "--flagship-input", "4.00", "--json"]
        )
        with redirect_stdout(buf):
            rc = handler(args_price, db=self.db)
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["pricing"]["flagship_input"], "4.00")

        buf = io.StringIO()
        args_rec = root.parse_args(
            [
                "roi",
                "record-saving",
                "--episode-id",
                "ep-cli",
                "--saved-tokens",
                "25000",
                "--cliff-200k-avoided",
                "--json",
            ]
        )
        with redirect_stdout(buf):
            rc = handler(args_rec, db=self.db)
        self.assertEqual(rc, 0)
        # 25,000 * 4.00 * 2.0 / 1e6 = 0.2000000
        self.assertEqual(json.loads(buf.getvalue())["gross_saved_usd"], "0.2000000")

        # 2) extra_args fallback mode.
        buf = io.StringIO()
        ns_fallback = argparse.Namespace(
            extra_args=["status", "--json"],
            json=False,
        )
        with redirect_stdout(buf):
            rc = handler(self.db, ns_fallback)
        self.assertEqual(rc, 0)
        status_data = json.loads(buf.getvalue())
        self.assertEqual(status_data["gross_saved_usd"], "0.2000000")
        self.assertFalse(status_data["tax_alert"])

        buf = io.StringIO()
        ns_reconcile = argparse.Namespace(extra_args=["reconcile", "--json"], json=False)
        with redirect_stdout(buf):
            rc = handler(self.db, ns_reconcile)
        self.assertEqual(rc, 0)
        self.assertTrue(json.loads(buf.getvalue())["ok"])

        # 3) HookBus integration.
        record_swarm_telemetry(
            self.db,
            role="scout",
            action="deep_recall",
            metrics={"prompt_tokens": 2_000, "completion_tokens": 400},
        )
        emit_res = kernel.hooks.emit("swarm.telemetry.recorded", {"db": self.db})
        self.assertTrue(emit_res.ok)
        self.assertEqual(emit_res.results[0]["synced"], 1)

        save_res = kernel.hooks.emit(
            "roi.window_saved",
            {"db": self.db, "episode_id": "ep-hook", "saved_tokens": 1_000},
        )
        self.assertTrue(save_res.ok)
        self.assertEqual(save_res.results[0]["episode_id"], "ep-hook")

    def test_clean_room_boundaries(self) -> None:
        plugins_dir = Path(__file__).resolve().parent.parent / "reiji_maigo" / "plugins"
        source_text = (plugins_dir / "roi_ledger.py").read_text(encoding="utf-8")
        forbidden_tokens = (
            "C:\\Users\\",
            "App" + "Data",
            "acerola" + "_orion",
            "orion" + "_private",
            "shinobu" + "_private",
        )
        for token in forbidden_tokens:
            self.assertNotIn(token, source_text)


if __name__ == "__main__":
    unittest.main()
