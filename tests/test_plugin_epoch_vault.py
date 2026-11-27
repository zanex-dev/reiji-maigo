"""Unit tests for the append-only SHA-256 verified epoch cold vault plugin."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

from reiji_maigo.core.contracts import MemoryCard, SourceSpan
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.plugins.epoch_vault import (
    COLD_EPOCH_ID_KEY,
    COLD_PAYLOAD_SHA256_KEY,
    COLD_STUB_KEY,
    ENV_EPOCH_VAULT_DIR,
    EpochColdVault,
    EpochVaultPlugin,
    EpochVerificationError,
    create_plugin,
    resolve_epoch_vault_dir,
)


def _make_card(
    card_id: str,
    *,
    domain: str = "proj",
    card_type: str = "note",
    title: str = "Sample Card",
    state: dict | None = None,
    lifecycle_status: str = "active",
    valid_from: str = "2026-01-01T00:00:00+00:00",
    spans: tuple[SourceSpan, ...] = (),
    tags: tuple[str, ...] = ("alpha",),
) -> MemoryCard:
    return MemoryCard(
        card_id=card_id,
        domain=domain,
        card_type=card_type,
        title=title,
        state=dict(state if state is not None else {"detail": f"payload-{card_id}"}),
        lifecycle_status=lifecycle_status,
        valid_from=valid_from,
        valid_to=None,
        superseded_by=None,
        tags=tags,
        spans=spans,
    )


class TestEpochColdVaultPlugin(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.home = self.root / "reiji_home"
        self.vault_dir = self.root / "external_epoch_vault"
        self.engine = ReijiMemoryEngine(home=self.home, load_plugins=False)
        self.vault = EpochColdVault(self.vault_dir)

    def tearDown(self) -> None:
        self.engine.close()
        self._tmp.cleanup()

    def _seed_hot_dataset(self) -> dict[str, object]:
        old_ts = "2026-01-10T00:00:00+00:00"
        recent_ts = "2026-03-01T00:00:00+00:00"
        now_iso = "2026-03-10T00:00:00+00:00"

        # Turns: 1 & 2 stale unreferenced; 3 protected by active card; 4 protected by pool; 5 above watermark
        t1 = self.engine.db.append_turn("ep-old", "user", "obsolete turn alpha", created_at=old_ts)
        t2 = self.engine.db.append_turn("ep-old", "assistant", "obsolete turn beta", created_at=old_ts)
        t3 = self.engine.db.append_turn("ep-active", "user", "referenced by active card", created_at=old_ts)
        t4 = self.engine.db.append_turn("ep-pool", "user", "referenced by active pool", created_at=old_ts)
        t5 = self.engine.db.append_turn("ep-undigested", "user", "beyond watermark", created_at=old_ts)
        self.engine.db.set_watermark("pool_digested", t4)

        # Active pool event referencing t4
        with self.engine.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO pool_events (
                    event_id, title, scope, status, steps_json, checkpoint_summary,
                    spans_json, hit_count, created_at, last_touched_at, dormant_nights
                )
                VALUES (?, ?, ?, ?, '[]', '', ?, 1, ?, ?, 0)
                """,
                (
                    "ev-active-1",
                    "Active Workstream",
                    "proj",
                    "active",
                    json.dumps([{"episode_id": "ep-pool", "turn_start": t4, "turn_end": t4}]),
                    old_ts,
                    recent_ts,
                ),
            )

        # Card 1: old superseded card (eligible)
        old_sup_state = {
            "summary": "Legacy architecture draft v1",
            "metrics": {"latency_ms": 420, "nested": ["a", "b"]},
        }
        self.engine.cards.create_card(
            "proj",
            "note",
            "Old Superseded Card",
            old_sup_state,
            ("alpha",),
            (SourceSpan("ep-old", t1, t2),),
            card_id="card-old-sup",
            valid_from=old_ts,
            created_at=old_ts,
            updated_at=old_ts,
        )

        c_new_active = _make_card(
            "card-active-keep",
            title="Current Active Card",
            state={"summary": "Current v2 spec"},
            valid_from=old_ts,
            spans=(SourceSpan("ep-active", t3, t3),),
        )
        self.engine.cards.supersede_card("card-old-sup", c_new_active, now_iso=old_ts)

        # Card 2: old revoked card (eligible)
        old_rev_state = {"reason": "deprecated setting", "value": 99}
        self.engine.cards.create_card(
            "proj",
            "note",
            "Old Revoked Card",
            old_rev_state,
            ("alpha",),
            (SourceSpan("ep-old", t1, t1),),
            card_id="card-old-rev",
            valid_from=old_ts,
            created_at=old_ts,
            updated_at=old_ts,
        )
        self.engine.cards.revoke_card("card-old-rev", now_iso=old_ts)

        # Card 3: recently revoked card (too young -> not eligible for 30d window)
        self.engine.cards.create_card(
            "proj",
            "note",
            "Recent Revoked Card",
            {"note": "revoked 9 days ago"},
            ("alpha",),
            (SourceSpan("ep-old", t2, t2),),
            card_id="card-recent-rev",
            valid_from=recent_ts,
            created_at=recent_ts,
            updated_at=recent_ts,
        )
        self.engine.cards.revoke_card("card-recent-rev", now_iso=recent_ts)

        return {
            "now_iso": now_iso,
            "old_ts": old_ts,
            "recent_ts": recent_ts,
            "old_sup_state": old_sup_state,
            "old_rev_state": old_rev_state,
            "turns": {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5},
        }

    def test_append_only_surface_exposes_no_delete_api(self) -> None:
        forbidden_tokens = (
            "delete",
            "remove",
            "purge",
            "drop",
            "prune",
            "erase",
            "truncate",
            "destroy",
            "unlink",
        )
        public_attrs = [name for name in dir(EpochColdVault) if not name.startswith("_")]
        for attr in public_attrs:
            lowered = attr.lower()
            for token in forbidden_tokens:
                self.assertNotIn(
                    token,
                    lowered,
                    f"EpochColdVault must not expose destructive method {attr!r}",
                )

    def test_default_vault_dir_resolution(self) -> None:
        default_dir = resolve_epoch_vault_dir(self.engine, env={})
        self.assertEqual(default_dir, (self.home / "epoch_vault").resolve())

        env_target = self.root / "env_vault"
        from_env = resolve_epoch_vault_dir(
            self.engine,
            env={ENV_EPOCH_VAULT_DIR: str(env_target)},
        )
        self.assertEqual(from_env, env_target.resolve())

        explicit = self.root / "explicit_vault"
        from_arg = resolve_epoch_vault_dir(
            self.engine,
            vault_dir=explicit,
            env={ENV_EPOCH_VAULT_DIR: str(env_target)},
        )
        self.assertEqual(from_arg, explicit.resolve())

    def test_scan_preview_seal_verify_and_restore_end_to_end(self) -> None:
        seeded = self._seed_hot_dataset()
        now_iso = str(seeded["now_iso"])
        turns = seeded["turns"]

        # 1. Scan candidates
        scan = self.vault.scan_candidates(self.engine, older_than_days=30, now_iso=now_iso)
        self.assertEqual(scan["card_ids"], ["card-old-rev", "card-old-sup"])
        self.assertEqual(scan["turn_ids"], [turns["t1"], turns["t2"]])
        self.assertEqual(scan["protected_turns_skipped"], 2)

        # 2. Dry-run seal does not mutate disk epochs or hot DB
        dry = self.vault.seal_epoch(self.engine, older_than_days=30, now_iso=now_iso, confirm=False)
        self.assertTrue(dry["ok"])
        self.assertTrue(dry["dry_run"])
        self.assertFalse(dry["sealed"])
        self.assertEqual(list(self.vault.epochs_dir.iterdir()), [])
        self.assertNotIn(COLD_STUB_KEY, self.engine.cards.get_card("card-old-sup").state)

        # 3. Confirmed atomic seal
        res = self.vault.seal_epoch(self.engine, older_than_days=30, now_iso=now_iso, confirm=True)
        self.assertTrue(res["ok"])
        self.assertTrue(res["sealed"])
        epoch_id = res["epoch_id"]
        self.assertIsNotNone(epoch_id)
        self.assertEqual(res["card_count"], 2)
        self.assertEqual(res["turn_count"], 2)

        epoch_dir = self.vault.epochs_dir / epoch_id
        self.assertTrue((epoch_dir / "cards.jsonl").is_file())
        self.assertTrue((epoch_dir / "turns.jsonl").is_file())
        self.assertTrue((epoch_dir / "manifest.json").is_file())

        # Index DB uses WAL mode
        conn = sqlite3.connect(str(self.vault.index_db_path))
        try:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            self.assertEqual(str(mode).lower(), "wal")
        finally:
            conn.close()

        # Hot cards tombstoned, seals valid, audit chain valid
        stub_card = self.engine.cards.get_card("card-old-sup")
        self.assertIsNotNone(stub_card)
        self.assertIs(stub_card.state.get(COLD_STUB_KEY), True)
        self.assertEqual(stub_card.state.get(COLD_EPOCH_ID_KEY), epoch_id)
        self.assertTrue(bool(stub_card.state.get(COLD_PAYLOAD_SHA256_KEY)))

        # Stale turns removed from raw_turns and FTS; protected turns preserved
        with self.engine.db.read_connection() as hconn:
            remaining_turn_ids = [
                int(r["turn_id"])
                for r in hconn.execute("SELECT turn_id FROM raw_turns ORDER BY turn_id ASC").fetchall()
            ]
        self.assertEqual(remaining_turn_ids, [turns["t3"], turns["t4"], turns["t5"]])
        self.assertEqual(self.engine.db.search_turns_fts("obsolete turn alpha"), [])

        seals_check = self.engine.verify_card_seals()
        self.assertTrue(seals_check["ok"], f"Card seals broken after seal_epoch: {seals_check}")
        chain_check = self.engine.audit.verify_chain()
        self.assertTrue(chain_check["ok"], f"Audit chain broken after seal_epoch: {chain_check}")

        with self.engine.db.read_connection() as hconn:
            sealed_audit = hconn.execute(
                "SELECT action, target_id FROM audit_chain WHERE action = 'epoch_sealed'"
            ).fetchall()
        self.assertEqual(len(sealed_audit), 1)
        self.assertEqual(sealed_audit[0]["target_id"], epoch_id)

        # Re-scanning skips already-stubbed cards
        rescan = self.vault.scan_candidates(self.engine, older_than_days=30, now_iso=now_iso)
        self.assertEqual(rescan["card_count"], 0)
        self.assertEqual(rescan["turn_count"], 0)
        self.assertEqual(rescan["already_stubbed_cards"], 2)

        # Verify cold vault & summary
        vcheck = self.vault.verify_vault()
        self.assertTrue(vcheck["ok"], f"Vault verification failed: {vcheck}")
        self.assertEqual(vcheck["epochs_checked"], 1)
        self.assertEqual(vcheck["cards_verified"], 2)
        self.assertEqual(vcheck["turns_verified"], 2)

        summary = self.vault.vault_summary(self.engine)
        self.assertEqual(summary["epoch_count"], 1)
        self.assertEqual(summary["sealed_cards"], 2)
        self.assertEqual(summary["sealed_turns"], 2)
        self.assertEqual(summary["hot_stub_cards"], 2)
        self.assertEqual(summary["latest_epoch_id"], epoch_id)

        # 4. Lossless restore_card with re-seal & audit
        restored = self.vault.restore_card(self.engine, "card-old-sup", now_iso="2026-03-11T00:00:00+00:00")
        self.assertEqual(restored.state, seeded["old_sup_state"])
        self.assertNotIn(COLD_STUB_KEY, restored.state)
        self.assertTrue(self.engine.verify_card_seals()["ok"])
        self.assertTrue(self.engine.audit.verify_chain()["ok"])

        summary_after = self.vault.vault_summary(self.engine)
        self.assertEqual(summary_after["hot_stub_cards"], 1)

    def test_failed_disk_verification_aborts_before_mutating_hot_store(self) -> None:
        seeded = self._seed_hot_dataset()
        now_iso = str(seeded["now_iso"])
        turns = seeded["turns"]

        with mock.patch.object(
            self.vault,
            "_verify_epoch_dir_from_disk",
            side_effect=EpochVerificationError("simulated disk read-back corruption"),
        ):
            with self.assertRaises(EpochVerificationError):
                self.vault.seal_epoch(
                    self.engine,
                    older_than_days=30,
                    now_iso=now_iso,
                    confirm=True,
                )

        # Hot cards and turns must remain completely untouched
        card = self.engine.cards.get_card("card-old-sup")
        self.assertEqual(card.state, seeded["old_sup_state"])
        self.assertNotIn(COLD_STUB_KEY, card.state)
        with self.engine.db.read_connection() as hconn:
            tids = [int(r["turn_id"]) for r in hconn.execute("SELECT turn_id FROM raw_turns").fetchall()]
        self.assertIn(turns["t1"], tids)
        self.assertIn(turns["t2"], tids)
        self.assertTrue(self.engine.verify_card_seals()["ok"])
        self.assertTrue(self.engine.audit.verify_chain()["ok"])

    def test_tamper_detection_blocks_verify_and_restore(self) -> None:
        seeded = self._seed_hot_dataset()
        now_iso = str(seeded["now_iso"])
        res = self.vault.seal_epoch(self.engine, older_than_days=30, now_iso=now_iso, confirm=True)
        epoch_id = res["epoch_id"]
        epoch_dir = self.vault.epochs_dir / epoch_id

        # Tamper with cards.jsonl
        cards_file = epoch_dir / "cards.jsonl"
        original_bytes = cards_file.read_bytes()
        cards_file.write_bytes(original_bytes.replace(b"Legacy architecture", b"Tampered payload!!"))

        vcheck = self.vault.verify_vault()
        self.assertFalse(vcheck["ok"])
        self.assertTrue(len(vcheck["issues"]) >= 1)

        # Restore must refuse tampered cold data and preserve hot tombstone
        with self.assertRaises(EpochVerificationError):
            self.vault.restore_card(self.engine, "card-old-sup")
        self.assertIs(self.engine.cards.get_card("card-old-sup").state.get(COLD_STUB_KEY), True)

        # Deleting an epoch directory externally is also detected by verify_vault
        shutil.rmtree(epoch_dir)
        vcheck_missing = self.vault.verify_vault()
        self.assertFalse(vcheck_missing["ok"])
        self.assertIn(epoch_id, vcheck_missing["missing_epochs"])

    def test_plugin_registration_and_cli_subcommands(self) -> None:
        seeded = self._seed_hot_dataset()
        now_iso = str(seeded["now_iso"])

        plugin = create_plugin(vault_dir=self.vault_dir)
        self.assertIsInstance(plugin, EpochVaultPlugin)
        self.assertEqual(plugin.name, "epoch_vault")

        rec = self.engine.kernel.register_plugin(plugin)
        self.assertEqual(rec.status, "loaded")
        spec = self.engine.kernel.get_cli_command("epoch")
        self.assertIsNotNone(spec)
        self.assertTrue(callable(getattr(spec.handler, "configure_parser", None)))

        def _run_configured(argv: list[str]) -> tuple[int, dict]:
            parser = argparse.ArgumentParser(prog="reiji epoch")
            spec.handler.configure_parser(parser)
            ns = parser.parse_args(argv)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = spec.handler(self.engine, ns)
            return rc, json.loads(buf.getvalue())

        rc, scan_out = _run_configured(["scan", "--older-than-days", "30", "--now-iso", now_iso, "--json"])
        self.assertEqual(rc, 0)
        self.assertEqual(scan_out["card_count"], 2)

        rc, seal_out = _run_configured(
            ["seal", "--older-than-days", "30", "--now-iso", now_iso, "--confirm", "--json"]
        )
        self.assertEqual(rc, 0)
        self.assertTrue(seal_out["sealed"])

        rc, verify_out = _run_configured(["verify", "--json"])
        self.assertEqual(rc, 0)
        self.assertTrue(verify_out["ok"])

        rc, restore_out = _run_configured(["restore", "card-old-rev", "--json"])
        self.assertEqual(rc, 0)
        self.assertTrue(restore_out["ok"])
        self.assertEqual(restore_out["card"]["state"], seeded["old_rev_state"])

        # Also verify compatibility with REMAINDER-style extra_args fallback
        extra_ns = argparse.Namespace(extra_args=["verify", "--json"], json=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc_extra = spec.handler(self.engine, extra_ns)
        self.assertEqual(rc_extra, 0)
        self.assertTrue(json.loads(buf.getvalue())["ok"])

        self.assertTrue(self.engine.verify_card_seals()["ok"])
        self.assertTrue(self.engine.audit.verify_chain()["ok"])

    def test_source_hygiene_and_no_plugins_init(self) -> None:
        plugin_path = Path(__file__).resolve().parents[1] / "reiji_maigo" / "plugins" / "epoch_vault.py"
        source = plugin_path.read_text(encoding="utf-8")
        self.assertNotIn("AppData", source)
        self.assertNotIn("C:\\Users", source)
        self.assertNotIn("C:/Users", source)
        self.assertNotIn("acerola" + "_orion", source.lower())


if __name__ == "__main__":
    unittest.main()
