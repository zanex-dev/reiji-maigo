"""Unit tests for SQLite user_version schema migrations and BootstrapImporter plugin."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from reiji_maigo.cli import build_parser
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.plugins.bootstrap_importer import (
    BootstrapImporter,
    BootstrapImporterPlugin,
    create_plugin,
)
from reiji_maigo.storage.migrations import (
    CURRENT_SCHEMA_VERSION,
    SchemaMigrationError,
    SchemaMigrationStep,
    apply_schema_migrations,
    get_schema_version,
)
from reiji_maigo.storage.sqlite_engine import ReijiDatabase


class TestSchemaMigrations(unittest.TestCase):
    def test_fresh_database_reaches_current_version_without_spurious_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "fresh.db"
            with ReijiDatabase(db_path) as db:
                with db.read_connection() as conn:
                    self.assertEqual(get_schema_version(conn), CURRENT_SCHEMA_VERSION)
                    rows = conn.execute(
                        "SELECT version, name FROM schema_migrations ORDER BY version ASC"
                    ).fetchall()
                    self.assertEqual([int(r["version"]) for r in rows], [1, 2])
            self.assertFalse(Path(f"{db_path}.bak-v0").exists())
            self.assertFalse(Path(f"{db_path}.bak-v1").exists())

    def test_v1_to_v2_upgrade_creates_verified_backup_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "legacy_v1.db"
            raw_conn = sqlite3.connect(str(db_path))
            try:
                raw_conn.execute(
                    """
                    CREATE TABLE raw_turns (
                        turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        episode_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                raw_conn.execute(
                    "INSERT INTO raw_turns (episode_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    ("ep_v1", "user", "legacy turn payload", "2026-01-01T00:00:00+00:00"),
                )
                raw_conn.execute("PRAGMA user_version = 1;")
                raw_conn.commit()
            finally:
                raw_conn.close()

            with ReijiDatabase(db_path) as db:
                with db.read_connection() as conn:
                    self.assertEqual(get_schema_version(conn), CURRENT_SCHEMA_VERSION)
                    # Second call must be a strict no-op.
                    second = apply_schema_migrations(conn, db_path=str(db_path))
                    self.assertFalse(second["migrated"])
                    self.assertEqual(second["applied_versions"], [])
                    self.assertIsNone(second["backup_path"])

            backup_path = Path(f"{db_path}.bak-v1")
            self.assertTrue(backup_path.exists())
            bak_conn = sqlite3.connect(str(backup_path))
            try:
                self.assertEqual(get_schema_version(bak_conn), 1)
                row = bak_conn.execute("SELECT content FROM raw_turns").fetchone()
                self.assertEqual(row[0], "legacy turn payload")
            finally:
                bak_conn.close()

    def test_failed_migration_rolls_back_and_preserves_version_and_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rollback.db"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "CREATE TABLE raw_turns (turn_id INTEGER PRIMARY KEY, episode_id TEXT, role TEXT, content TEXT, created_at TEXT)"
                )
                conn.execute(
                    "INSERT INTO raw_turns VALUES (1, 'ep1', 'user', 'keep me', '2026-01-01T00:00:00+00:00')"
                )
                conn.execute("PRAGMA user_version = 1;")
                conn.commit()

                def _broken_v2(c: sqlite3.Connection) -> None:
                    c.execute("CREATE TABLE partial_should_rollback (id INTEGER PRIMARY KEY)")
                    c.execute("INSERT INTO partial_should_rollback VALUES (42)")
                    raise RuntimeError("simulated migration crash")

                with self.assertRaises(SchemaMigrationError):
                    apply_schema_migrations(
                        conn,
                        db_path=str(db_path),
                        target_version=2,
                        migrations=[SchemaMigrationStep(2, "broken_v2", _broken_v2)],
                    )

                self.assertEqual(get_schema_version(conn), 1)
                partial_tbl = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='partial_should_rollback'"
                ).fetchone()
                self.assertIsNone(partial_tbl)
                self.assertTrue(Path(f"{db_path}.bak-v1").exists())
            finally:
                conn.close()

    def test_downgrade_is_rejected(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("PRAGMA user_version = 99;")
            with self.assertRaises(SchemaMigrationError):
                apply_schema_migrations(conn, target_version=CURRENT_SCHEMA_VERSION)
        finally:
            conn.close()


class TestBootstrapImporter(unittest.TestCase):
    def test_import_transcript_jsonl_sanitizes_dedups_and_skips_media_and_reparse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            transcript = tmp_dir / "session_alpha.jsonl"
            lines = [
                json.dumps({"role": "user", "content": "My key is sk-1234567890abcdefghijklmnop please note."}),
                json.dumps(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Noted safely without leaking credentials."},
                            {"type": "image_url", "image_url": {"url": "file:///fake/photo.png"}},
                        ],
                    }
                ),
                "{bad json line",
                json.dumps({"role": "user", "content": ""}),
            ]
            transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")

            fake_png = tmp_dir / "diagram.png"
            fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")

            with ReijiMemoryEngine(home=tmp_dir / "home", db_path=tmp_dir / "reiji.db", disable_plugins=True) as engine:
                importer = BootstrapImporter(engine)

                first = importer.import_transcript_jsonl(transcript, episode_id="ep_jsonl")
                self.assertEqual(first["imported"], 2)
                self.assertEqual(first["skipped_duplicates"], 0)
                self.assertEqual(first["skipped_invalid"], 2)

                # Re-running the same transcript must yield 0 duplicate insertions.
                second = importer.import_transcript_jsonl(transcript, episode_id="ep_jsonl")
                self.assertEqual(second["imported"], 0)
                self.assertEqual(second["skipped_duplicates"], 2)

                # Verify sanitization actually redacted the secret token.
                with engine.db.read_connection() as conn:
                    rows = conn.execute(
                        "SELECT role, content FROM raw_turns WHERE episode_id = ? ORDER BY turn_id ASC",
                        ("ep_jsonl",),
                    ).fetchall()
                self.assertEqual(len(rows), 2)
                self.assertNotIn("sk-1234567890abcdefghijklmnop", rows[0]["content"])
                self.assertEqual(rows[1]["content"], "Noted safely without leaking credentials.")

                # Binary media extension must be skipped.
                media_res = importer.import_transcript_jsonl(fake_png)
                self.assertEqual(media_res["imported"], 0)
                self.assertEqual(media_res["skipped_unsafe"], 1)
                self.assertEqual(media_res["skipped_reason"], "binary_media_extension")

                # Reparse point / symlink detection must skip without reading.
                with mock.patch(
                    "reiji_maigo.plugins.bootstrap_importer._is_reparse_or_symlink",
                    return_value=True,
                ):
                    reparse_res = importer.import_transcript_jsonl(transcript)
                    self.assertEqual(reparse_res["imported"], 0)
                    self.assertEqual(reparse_res["skipped_unsafe"], 1)
                    self.assertEqual(reparse_res["skipped_reason"], "reparse_point_or_symlink")

    def test_import_legacy_sqlite_readonly_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            legacy_db = tmp_dir / "old_memory.db"
            conn = sqlite3.connect(str(legacy_db))
            try:
                conn.execute(
                    """
                    CREATE TABLE turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        text TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO turns (session_id, role, text, timestamp) VALUES (?, ?, ?, ?)",
                    ("old_ep_1", "user", "Decided to adopt SQLite WAL mode.", "2026-02-01T10:00:00+00:00"),
                )
                conn.execute(
                    "INSERT INTO turns (session_id, role, text, timestamp) VALUES (?, ?, ?, ?)",
                    ("old_ep_1", "assistant", "Confirmed WAL + FTS5 design.", "2026-02-01T10:01:00+00:00"),
                )
                conn.commit()
            finally:
                conn.close()

            mtime_before = legacy_db.stat().st_mtime_ns

            with ReijiMemoryEngine(home=tmp_dir / "home", db_path=tmp_dir / "reiji.db", disable_plugins=True) as engine:
                importer = BootstrapImporter(engine)

                first = importer.import_legacy_sqlite(legacy_db)
                self.assertEqual(first["tables_scanned"], ["turns"])
                self.assertEqual(first["imported"], 2)
                self.assertEqual(first["skipped_duplicates"], 0)

                # Legacy DB must remain untouched (read-only import).
                self.assertEqual(legacy_db.stat().st_mtime_ns, mtime_before)

                second = importer.import_legacy_sqlite(legacy_db)
                self.assertEqual(second["imported"], 0)
                self.assertEqual(second["skipped_duplicates"], 2)

    def test_import_checkpoints_dir_mints_sealed_seed_cards_and_dedups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ckpt_dir = tmp_dir / "checkpoints"
            ckpt_dir.mkdir()

            (ckpt_dir / "handoff_01.md").write_text(
                "# Release Readiness Handoff\n\n## Summary\nSQLite migration gate complete.\n\n## Next\nRun final gate.\n",
                encoding="utf-8",
            )
            (ckpt_dir / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            with ReijiMemoryEngine(home=tmp_dir / "home", db_path=tmp_dir / "reiji.db", disable_plugins=True) as engine:
                importer = BootstrapImporter(engine)

                first = importer.import_checkpoints_dir(ckpt_dir, domain="proj", mint_seed_cards=True)
                self.assertEqual(first["files_scanned"], 1)
                self.assertEqual(first["skipped_unsafe"], 1)
                self.assertGreaterEqual(first["imported"], 2)
                self.assertEqual(len(first["seed_cards_minted"]), 1)
                self.assertEqual(len(first["spans"]), 1)

                # Verify minted card exists, has SourceSpan, and passes HMAC seal check.
                card_id = first["seed_cards_minted"][0]
                card = engine.cards.get_card(card_id)
                self.assertIsNotNone(card)
                assert card is not None
                self.assertEqual(card.title, "Release Readiness Handoff")
                self.assertEqual(len(card.spans), 1)
                seals = engine.verify_card_seals()
                self.assertTrue(seals["ok"])

                # Second run must import 0 new turns and mint 0 new cards.
                second = importer.import_checkpoints_dir(ckpt_dir, domain="proj", mint_seed_cards=True)
                self.assertEqual(second["imported"], 0)
                self.assertEqual(second["skipped_duplicates"], first["imported"])
                self.assertEqual(second["seed_cards_minted"], [])

                st = importer.status()
                self.assertEqual(st["schema_version"], CURRENT_SCHEMA_VERSION)
                self.assertEqual(st["total_imported"], first["imported"])
                self.assertEqual(st["by_source_kind"]["checkpoints_dir"], first["imported"])

    def test_plugin_registration_and_cli_configure_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            transcript = tmp_dir / "cli_sample.jsonl"
            transcript.write_text(
                json.dumps({"role": "user", "content": "Hello via CLI import"}) + "\n",
                encoding="utf-8",
            )

            with ReijiMemoryEngine(home=tmp_dir / "home", db_path=tmp_dir / "reiji.db", disable_plugins=True) as engine:
                plugin = create_plugin()
                self.assertIsInstance(plugin, BootstrapImporterPlugin)
                self.assertEqual(plugin.name, "bootstrap_importer")

                rec = engine.kernel.register_plugin(plugin)
                self.assertEqual(rec.status, "loaded")
                self.assertIn("import", engine.kernel.cli_commands)

                parser = build_parser(plugin_commands=engine.kernel.cli_commands)

                # Run `reiji import transcript <path> --json`
                args_tr = parser.parse_args(["import", "transcript", str(transcript), "--json"])
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = args_tr._plugin_spec.handler(engine, args_tr)
                self.assertEqual(rc, 0)
                payload_tr = json.loads(buf.getvalue())
                self.assertEqual(payload_tr["imported"], 1)

                # Run `reiji import status --json`
                args_st = parser.parse_args(["import", "status", "--json"])
                buf2 = io.StringIO()
                with redirect_stdout(buf2):
                    rc2 = args_st._plugin_spec.handler(engine, args_st)
                self.assertEqual(rc2, 0)
                payload_st = json.loads(buf2.getvalue())
                self.assertEqual(payload_st["total_imported"], 1)
                self.assertEqual(payload_st["schema_version"], CURRENT_SCHEMA_VERSION)

                # Confirm parser supports all 4 subcommands cleanly.
                for subcmd in (
                    ["import", "status"],
                    ["import", "transcript", str(transcript)],
                    ["import", "sqlite", str(tmp_dir / "dummy.db")],
                    ["import", "checkpoints", str(tmp_dir), "--domain", "infra", "--no-seed-cards"],
                ):
                    parsed = parser.parse_args(subcmd)
                    self.assertIsInstance(parsed, argparse.Namespace)
                    self.assertEqual(parsed.command, "import")


if __name__ == "__main__":
    unittest.main()
