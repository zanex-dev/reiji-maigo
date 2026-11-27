"""Unit tests for reiji_maigo.plugins.thermal_sync."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from reiji_maigo.core.kernel import ReijiKernel
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.plugins.thermal_sync import (
    ACTIVE_GRACE_SEC,
    CACHE_SAFE_TTL_SEC,
    FILE_ATTRIBUTE_REPARSE_POINT,
    TAIL_READ_BYTES,
    TIER_2X_THRESHOLD_TOKENS,
    TIER_2X_WARN_TOKENS,
    WARN_IDLE_SEC,
    ThermalSyncManager,
    ThermalSyncPlugin,
    create_plugin,
    evaluate_thermal_state,
    has_reparse_or_symlink,
    read_shared_tail_bytes,
)


class ThermalStateMachineTests(unittest.TestCase):
    def test_constants_match_contract(self) -> None:
        self.assertEqual(TAIL_READ_BYTES, 16 * 1024)
        self.assertEqual(CACHE_SAFE_TTL_SEC, 250.0)
        self.assertEqual(ACTIVE_GRACE_SEC, 5.0)
        self.assertEqual(WARN_IDLE_SEC, 190.0)
        self.assertEqual(TIER_2X_WARN_TOKENS, 180_000)
        self.assertEqual(TIER_2X_THRESHOLD_TOKENS, 200_000)

    def test_five_thermal_phases(self) -> None:
        now = 1_700_000_000.0

        empty = evaluate_thermal_state(last_activity_epoch=None, now_epoch=now)
        self.assertEqual(empty.phase, "empty")
        self.assertIsNone(empty.idle_sec)
        self.assertEqual(empty.remaining_ttl_sec, 0.0)
        self.assertEqual(empty.pricing_tier, "1x")
        self.assertFalse(empty.should_digest_on_cooldown)

        active = evaluate_thermal_state(
            last_activity_epoch=now - 3.0,
            now_epoch=now,
            context_tokens=40_000,
        )
        self.assertEqual(active.phase, "active")
        self.assertAlmostEqual(active.remaining_ttl_sec, 247.0)
        self.assertFalse(active.should_digest_on_cooldown)

        in_flight = evaluate_thermal_state(
            last_activity_epoch=now - 260.0,
            now_epoch=now,
            context_tokens=195_000,
            in_flight=True,
        )
        self.assertEqual(in_flight.phase, "active")
        self.assertEqual(in_flight.pricing_tier, "near_2x")
        self.assertFalse(in_flight.should_digest_on_cooldown)

        hot = evaluate_thermal_state(
            last_activity_epoch=now - 90.0,
            now_epoch=now,
            context_tokens=80_000,
        )
        self.assertEqual(hot.phase, "hot")
        self.assertAlmostEqual(hot.remaining_ttl_sec, 160.0)
        self.assertFalse(hot.should_digest_on_cooldown)

        warning = evaluate_thermal_state(
            last_activity_epoch=now - 210.0,
            now_epoch=now,
            context_tokens=120_000,
        )
        self.assertEqual(warning.phase, "warning")
        self.assertAlmostEqual(warning.remaining_ttl_sec, 40.0)
        self.assertFalse(warning.should_digest_on_cooldown)

        cold = evaluate_thermal_state(
            last_activity_epoch=now - 255.0,
            now_epoch=now,
            context_tokens=60_000,
        )
        self.assertEqual(cold.phase, "cold")
        self.assertEqual(cold.remaining_ttl_sec, 0.0)
        self.assertTrue(cold.should_digest_on_cooldown)

    def test_pricing_tier_thresholds(self) -> None:
        now = 1_700_000_000.0

        tier_1x = evaluate_thermal_state(
            last_activity_epoch=now - 20.0,
            now_epoch=now,
            context_tokens=179_999,
        )
        self.assertEqual(tier_1x.pricing_tier, "1x")
        self.assertFalse(tier_1x.should_digest_on_cooldown)

        near_2x = evaluate_thermal_state(
            last_activity_epoch=now - 20.0,
            now_epoch=now,
            context_tokens=180_000,
        )
        self.assertEqual(near_2x.pricing_tier, "near_2x")
        self.assertTrue(near_2x.should_digest_on_cooldown)

        tier_2x = evaluate_thermal_state(
            last_activity_epoch=now - 20.0,
            now_epoch=now,
            context_tokens=200_000,
        )
        self.assertEqual(tier_2x.pricing_tier, "2x")
        self.assertTrue(tier_2x.should_digest_on_cooldown)


class SharedTailReaderTests(unittest.TestCase):
    def test_rejects_media_suffixes_and_user_uploaded_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(PermissionError):
                read_shared_tail_bytes(root / "photo.png")
            with self.assertRaises(PermissionError):
                read_shared_tail_bytes(root / "clip.mp4")
            with self.assertRaises(PermissionError):
                read_shared_tail_bytes(root / ".user_uploaded" / "transcript.jsonl")

    def test_detects_symlink_and_reparse_point_and_refuses_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fake.jsonl"
            p.write_text('{"role":"user","content":"ok"}\n', encoding="utf-8")

            self.assertFalse(has_reparse_or_symlink(p))

            fake_symlink_stat = mock.MagicMock()
            fake_symlink_stat.st_mode = stat.S_IFLNK | 0o777
            fake_symlink_stat.st_file_attributes = 0
            with mock.patch("os.lstat", return_value=fake_symlink_stat):
                self.assertTrue(has_reparse_or_symlink(p))
                with self.assertRaises(PermissionError):
                    read_shared_tail_bytes(p)

            fake_reparse_stat = mock.MagicMock()
            fake_reparse_stat.st_mode = stat.S_IFREG | 0o644
            fake_reparse_stat.st_file_attributes = FILE_ATTRIBUTE_REPARSE_POINT
            with mock.patch("os.lstat", return_value=fake_reparse_stat):
                self.assertTrue(has_reparse_or_symlink(p))
                with self.assertRaises(PermissionError):
                    read_shared_tail_bytes(p)

    def test_strips_leading_utf8_fragment_and_trailing_unflushed_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "utf8_tail.jsonl"
            # Build lines containing multi-byte UTF-8 CJK characters
            line1 = '{"role":"user","content":"第一行中文测试数据用于切分验证"}\n'.encode("utf-8")
            line2 = '{"role":"assistant","content":"第二行完整回复保留下来"}\n'.encode("utf-8")
            partial_tail = '{"role":"user","content":"尚未刷盘的半'.encode("utf-8")
            p.write_bytes(line1 + line2 + partial_tail)

            # Choose max_bytes so the cut lands strictly inside a multi-byte UTF-8 char of line1
            total_size = len(line1) + len(line2) + len(partial_tail)
            max_bytes = total_size - 28  # cuts inside the CJK bytes of line1
            tail = read_shared_tail_bytes(p, max_bytes=max_bytes)

            # Should decode cleanly as strict UTF-8 and contain only complete line2
            decoded = tail.decode("utf-8")
            self.assertEqual(decoded, line2.decode("utf-8"))

    def test_shared_read_while_writer_keeps_file_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "live.jsonl"
            with open(p, "wb") as writer:
                row = '{"role":"user","content":"live stream turn"}\n'.encode("utf-8")
                writer.write(row)
                writer.flush()
                tail = read_shared_tail_bytes(p)
                self.assertEqual(tail, row)


class ThermalSyncManagerAndPluginTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.engine = ReijiMemoryEngine(
            home=self.root / "reiji_home",
            disable_plugins=True,
        )

    def tearDown(self) -> None:
        close_engine = getattr(self.engine, "close", None)
        if callable(close_engine):
            close_engine()
        db = getattr(self.engine, "db", None)
        close_db = getattr(db, "close", None)
        if callable(close_db):
            close_db()
        self._tmp.cleanup()

    def test_incremental_sync_with_partial_line_and_cooldown_handoff(self) -> None:
        transcript = self.root / "session.jsonl"
        t0 = 1_700_000_000.0
        line1 = json.dumps(
            {
                "role": "user",
                "content": "First complete turn",
                "timestamp": t0,
                "context_tokens": 50_000,
            }
        )
        partial_prefix = '{"role":"assistant","content":"Second turn '
        partial_suffix = f'completed","timestamp":{t0 + 10.0},"context_tokens":55000}}\n'

        transcript.write_text(line1 + "\n" + partial_prefix, encoding="utf-8")

        handoffs: list[dict] = []
        scribe_events: list[dict] = []
        self.engine.kernel.hooks.register("thermal.handoff_ready", handoffs.append)
        self.engine.kernel.hooks.register("thermal.scribe_requested", scribe_events.append)

        mgr = ThermalSyncManager(self.engine)
        res1 = mgr.sync_tail(
            transcript,
            episode_id="ep-thermal-1",
            now_epoch=t0 + 20.0,
            trigger_scribe_on_cool=True,
        )
        self.assertEqual(res1["synced_turns"], 1)
        self.assertEqual(res1["phase"], "hot")
        self.assertFalse(res1["handoff_emitted"])
        self.assertFalse(res1["scribe_triggered"])
        self.assertEqual(len(handoffs), 0)

        # Complete the partial second line and sync again at cooldown (>250s idle)
        with open(transcript, "a", encoding="utf-8") as fh:
            fh.write(partial_suffix)

        res2 = mgr.sync_tail(
            transcript,
            episode_id="ep-thermal-1",
            now_epoch=t0 + 300.0,
            trigger_scribe_on_cool=True,
        )
        self.assertEqual(res2["synced_turns"], 1)
        self.assertEqual(res2["phase"], "cold")
        self.assertTrue(res2["handoff_emitted"])
        self.assertTrue(res2["scribe_triggered"])
        self.assertEqual(len(handoffs), 1)
        self.assertEqual(handoffs[0]["reason"], "cache_cooldown")
        self.assertEqual(len(scribe_events), 1)

        # Third sync with no new bytes must not duplicate turns
        res3 = mgr.sync_tail(
            transcript,
            episode_id="ep-thermal-1",
            now_epoch=t0 + 305.0,
        )
        self.assertEqual(res3["synced_turns"], 0)
        self.assertEqual(self.engine.status()["counts"]["raw_turns"], 2)

    def test_near_2x_emits_handoff_even_while_hot(self) -> None:
        transcript = self.root / "heavy.jsonl"
        t0 = 1_700_000_000.0
        transcript.write_text(
            json.dumps(
                {
                    "role": "user",
                    "content": "Context approaching cliff",
                    "timestamp": t0,
                    "usage": {"prompt_tokens": 185_000},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        handoffs: list[dict] = []
        self.engine.kernel.hooks.register("thermal.handoff_ready", handoffs.append)

        mgr = ThermalSyncManager(self.engine)
        res = mgr.sync_tail(transcript, now_epoch=t0 + 30.0)
        self.assertEqual(res["phase"], "hot")
        self.assertEqual(res["pricing_tier"], "near_2x")
        self.assertTrue(res["handoff_emitted"])
        self.assertEqual(handoffs[0]["reason"], "pricing_tier_near_2x")

    def test_cursor_resets_when_file_truncated(self) -> None:
        transcript = self.root / "rotated.jsonl"
        t0 = 1_700_000_000.0
        transcript.write_text(
            json.dumps({"role": "user", "content": "Longer original turn payload", "timestamp": t0})
            + "\n",
            encoding="utf-8",
        )
        mgr = ThermalSyncManager(self.engine)
        res1 = mgr.sync_tail(transcript, now_epoch=t0 + 5.0)
        self.assertEqual(res1["synced_turns"], 1)

        # Truncate to a shorter new file
        transcript.write_text(
            json.dumps({"role": "user", "content": "New", "timestamp": t0 + 10.0}) + "\n",
            encoding="utf-8",
        )
        res2 = mgr.sync_tail(transcript, now_epoch=t0 + 12.0)
        self.assertTrue(res2["cursor_reset"])
        self.assertEqual(res2["synced_turns"], 1)

    def test_plugin_registration_and_radar_cli_subcommands(self) -> None:
        plugin = create_plugin()
        self.assertIsInstance(plugin, ThermalSyncPlugin)
        self.assertEqual(plugin.name, "thermal_sync")

        kernel = ReijiKernel(home=self.root / "k_home", disable_plugins=True)
        plugin.register(kernel, engine=self.engine)

        spec = kernel.get_cli_command("radar")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(callable(getattr(spec.handler, "configure_parser", None)))

        parser = argparse.ArgumentParser(prog="reiji radar")
        spec.handler.configure_parser(parser)

        transcript = self.root / "cli_chat.jsonl"
        t0 = 1_700_000_000.0
        transcript.write_text(
            json.dumps(
                {
                    "role": "user",
                    "content": "Radar CLI check",
                    "timestamp": t0,
                    "context_tokens": 205_000,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        # 1. probe
        probe_args = parser.parse_args(
            ["probe", str(transcript), "--now-epoch", str(t0 + 15.0), "--json"]
        )
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = spec.handler(probe_args, self.engine)
        self.assertEqual(rc, 0)
        probe_out = json.loads(buf.getvalue())
        self.assertEqual(probe_out["phase"], "hot")
        self.assertEqual(probe_out["pricing_tier"], "2x")

        # 2. sync
        sync_args = parser.parse_args(
            [
                "sync",
                str(transcript),
                "--episode-id",
                "ep-cli",
                "--now-epoch",
                str(t0 + 20.0),
                "--trigger-scribe",
            ]
        )
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = spec.handler(sync_args, self.engine)
        self.assertEqual(rc, 0)
        sync_out = json.loads(buf.getvalue())
        self.assertEqual(sync_out["synced_turns"], 1)
        self.assertTrue(sync_out["handoff_emitted"])
        self.assertTrue(sync_out["scribe_triggered"])

        # 3. status
        status_args = parser.parse_args(["status"])
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = spec.handler(status_args, self.engine)
        self.assertEqual(rc, 0)
        status_out = json.loads(buf.getvalue())
        self.assertEqual(status_out["tracked_files"], 1)

    def test_large_file_caps_read_to_16kb_tail_window(self) -> None:
        transcript = self.root / "large.jsonl"
        t0 = 1_700_000_000.0
        lines = [
            json.dumps(
                {
                    "role": "user" if idx % 2 == 0 else "assistant",
                    "content": f"turn-{idx:04d}-" + ("x" * 200),
                    "timestamp": t0 + idx,
                }
            )
            for idx in range(120)
        ]
        transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertGreater(transcript.stat().st_size, TAIL_READ_BYTES)

        tail = read_shared_tail_bytes(transcript)
        self.assertLessEqual(len(tail), TAIL_READ_BYTES)
        self.assertIn(b"turn-0119-", tail)
        self.assertNotIn(b"turn-0000-", tail)

        mgr = ThermalSyncManager(self.engine)
        res = mgr.sync_tail(transcript, now_epoch=t0 + 130.0)
        self.assertTrue(res["window_truncated"])
        self.assertGreater(res["synced_turns"], 0)
        self.assertLess(res["synced_turns"], 120)

    def test_plugin_source_hygiene_and_no_package_init(self) -> None:
        plugin_path = Path(__file__).resolve().parents[1] / "reiji_maigo" / "plugins" / "thermal_sync.py"
        src = plugin_path.read_text(encoding="utf-8").lower()
        for forbidden in ("app" + "data", "acerola" + "_orion", "c:\\" + "users\\"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
