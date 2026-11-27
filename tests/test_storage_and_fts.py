"""Unit tests for SQLite WAL storage engine and mixed CJK/English Bigram FTS5."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import unittest

from reiji_maigo.core import SourceSpan
from reiji_maigo.storage import (
    MAX_SPAN_TURNS,
    ReijiDatabase,
    build_fts_query,
    tokenize_for_fts,
)

EXPECTED_TABLES = {
    "raw_turns",
    "raw_turns_fts",
    "pool_events",
    "memory_tags",
    "memory_cards",
    "card_tags",
    "card_links",
    "meta_watermarks",
    "pending_backfills",
}


class TokenizerTests(unittest.TestCase):
    def test_tokenize_cjk_bigrams_and_single_char_fallback(self) -> None:
        self.assertEqual(tokenize_for_fts("配额问题"), "配额 额问 问题")
        self.assertEqual(tokenize_for_fts("配额"), "配额")
        self.assertEqual(tokenize_for_fts("A 猫 B"), "a 猫 b")
        self.assertEqual(
            tokenize_for_fts("零时迷子 SQLite_WAL 429 配额"),
            "零时 时迷 迷子 sqlite_wal 429 配额",
        )
        self.assertEqual(tokenize_for_fts("ＦＴＳ５配额"), "fts5 配额")
        self.assertEqual(tokenize_for_fts("   !!! ---   "), "")

    def test_build_fts_query_safe_phrases_and_escaping(self) -> None:
        self.assertEqual(build_fts_query("配额"), '"配额"')
        self.assertEqual(build_fts_query("配额问题"), '"配额 额问 问题"')
        self.assertEqual(build_fts_query("SQLite 配额"), '"sqlite" AND "配额"')
        self.assertEqual(build_fts_query("配额 配额"), '"配额"')
        self.assertEqual(build_fts_query("   !!!   "), "")
        # Neutralize FTS5 operators, quotes, wildcards, column filters, and parens.
        safe = build_fts_query('foo"bar OR NOT (baz*) ^near:1')
        self.assertEqual(safe, '"foo" AND "bar" AND "or" AND "not" AND "baz" AND "near" AND "1"')


class ReijiDatabaseTests(unittest.TestCase):
    def test_wal_pragmas_and_full_schema_on_file_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "nested" / "reiji.sqlite3"
            with ReijiDatabase(db_file) as db:
                with db.read_connection() as conn:
                    journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
                    foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
                    busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
                    tables = {
                        row[0]
                        for row in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'table'"
                        ).fetchall()
                    }

                self.assertEqual(str(journal_mode).lower(), "wal")
                self.assertEqual(int(foreign_keys), 1)
                self.assertEqual(int(busy_timeout), 5000)
                self.assertTrue(EXPECTED_TABLES.issubset(tables))

                # Verify memory_tags constraints: locked domain CHECK and canonical_name UNIQUE.
                with db.transaction() as conn:
                    conn.execute(
                        """
                        INSERT INTO memory_tags (domain, canonical_name, aliases_json, usage_count)
                        VALUES ('infra', 'sqlite-wal', '["wal"]', 1)
                        """
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    with db.transaction() as conn:
                        conn.execute(
                            """
                            INSERT INTO memory_tags (domain, canonical_name)
                            VALUES ('infra', 'sqlite-wal')
                            """
                        )
                with self.assertRaises(sqlite3.IntegrityError):
                    with db.transaction() as conn:
                        conn.execute(
                            """
                            INSERT INTO memory_tags (domain, canonical_name)
                            VALUES ('unknown_domain', 'bad-tag')
                            """
                        )

    def test_cjk_two_char_bigram_fts_uses_index_and_matches(self) -> None:
        with ReijiDatabase(":memory:") as db:
            t1 = db.append_turn("ep-1", "user", "今天 API 配额触发了 429 限流告警")
            t2 = db.append_turn("ep-2", "assistant", "已把 sqlite_engine 切到 WAL 并检查配额超限日志")
            t3 = db.append_turn("ep-3", "user", "这是一段关于午饭和猫的闲聊 A 猫 B")
            db.append_turn("ep-4", "user", "配额在这里，但超限在另一句里分开出现")

            # Verify EXPLAIN QUERY PLAN uses FTS5 virtual table index (no raw_turns full scan).
            fts_expr = build_fts_query("配额")
            with db.read_connection() as conn:
                plan_rows = conn.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT t.turn_id
                    FROM raw_turns_fts
                    JOIN raw_turns AS t ON t.turn_id = raw_turns_fts.rowid
                    WHERE raw_turns_fts MATCH ?
                    """,
                    (fts_expr,),
                ).fetchall()
            plan_text = " | ".join(str(row["detail"]) for row in plan_rows)
            self.assertIn("VIRTUAL TABLE INDEX", plan_text.upper())
            self.assertIn("SEARCH T USING INTEGER PRIMARY KEY", plan_text.upper())
            self.assertNotIn("SCAN T", plan_text.upper())

            # 2-char Chinese query hits all turns containing "配额".
            hits = db.search_turns_fts("配额", limit=10)
            hit_ids = {h["turn_id"] for h in hits}
            self.assertIn(t1, hit_ids)
            self.assertIn(t2, hit_ids)
            self.assertNotIn(t3, hit_ids)

            # exclude_episode filters out ep-1.
            filtered = db.search_turns_fts("配额", limit=10, exclude_episode="ep-1")
            self.assertTrue(all(h["episode_id"] != "ep-1" for h in filtered))
            self.assertIn(t2, {h["turn_id"] for h in filtered})

            # 4-char exact phrase "配额超限" matches ep-2, not ep-4 where terms are separated.
            phrase_hits = db.search_turns_fts("配额超限")
            self.assertEqual([h["turn_id"] for h in phrase_hits], [t2])

            # Mixed English + Chinese identifier search.
            mixed_hits = db.search_turns_fts("sqlite_engine WAL 配额")
            self.assertEqual([h["turn_id"] for h in mixed_hits], [t2])

            # Single CJK character fallback ("A 猫 B").
            single_hits = db.search_turns_fts("猫")
            self.assertEqual([h["turn_id"] for h in single_hits], [t3])

            # Malicious / operator-heavy query strings never raise OperationalError.
            self.assertEqual(db.search_turns_fts('OR NOT "" () * ^ :'), [])
            self.assertEqual(db.search_turns_fts("   "), [])

    def test_read_span_padding_isolation_and_hard_cap(self) -> None:
        with ReijiDatabase(":memory:") as db:
            ep1_ids: list[int] = []
            # Interleave turns across ep-1 and ep-2.
            for idx in range(1, 6):
                ep1_ids.append(db.append_turn("ep-1", "user", f"ep1-msg-{idx}"))
                db.append_turn("ep-2", "user", f"ep2-noise-{idx}")

            # Core span covering ep1 #3 only, with pad=1 should pull ep1 #2, #3, #4 (no ep2).
            span = SourceSpan(episode_id="ep-1", turn_start=ep1_ids[2], turn_end=ep1_ids[2])
            padded = db.read_span(span, pad=1)
            self.assertEqual([r["turn_id"] for r in padded], [ep1_ids[1], ep1_ids[2], ep1_ids[3]])
            self.assertTrue(all(r["episode_id"] == "ep-1" for r in padded))

            # Hard cap <= 40 even when caller requests max_turns=100 over 70 turns.
            bulk_ids = [db.append_turn("ep-bulk", "user", f"turn-{i}") for i in range(70)]
            big_span = SourceSpan(episode_id="ep-bulk", turn_start=bulk_ids[0], turn_end=bulk_ids[-1])
            capped = db.read_span(big_span, max_turns=100)
            self.assertEqual(len(capped), MAX_SPAN_TURNS)
            self.assertEqual(capped[0]["turn_id"], bulk_ids[0])
            self.assertEqual(capped[-1]["turn_id"], bulk_ids[MAX_SPAN_TURNS - 1])

            self.assertEqual(db.read_span(big_span, max_turns=0), [])
            with self.assertRaises(ValueError):
                db.read_span(big_span, pad=-1)
            with self.assertRaises(ValueError):
                db.read_span(big_span, max_turns=-5)

    def test_watermarks_and_undigested_turns(self) -> None:
        with ReijiDatabase(":memory:") as db:
            self.assertEqual(db.get_watermark("pool_digested"), 0)

            ids = [db.append_turn("ep-w", "user", f"msg-{i}") for i in range(15)]
            batch1 = db.get_undigested_turns(watermark_key="pool_digested", limit=5)
            self.assertEqual([r["turn_id"] for r in batch1], ids[:5])

            db.set_watermark("pool_digested", ids[4])
            self.assertEqual(db.get_watermark("pool_digested"), ids[4])

            batch2 = db.get_undigested_turns(watermark_key="pool_digested", limit=12)
            self.assertEqual([r["turn_id"] for r in batch2], ids[5:])

            # Independent watermark key still starts from 0.
            self.assertEqual(db.get_watermark("card_digested"), 0)
            other_batch = db.get_undigested_turns(watermark_key="card_digested", limit=3)
            self.assertEqual([r["turn_id"] for r in other_batch], ids[:3])

    def test_thread_safe_concurrent_writes_and_searches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "concurrent.sqlite3"
            with ReijiDatabase(db_file) as db:
                def worker(worker_id: int) -> list[int]:
                    created: list[int] = []
                    for i in range(15):
                        tid = db.append_turn(
                            f"ep-t{worker_id}",
                            "user",
                            f"线程{worker_id}写入第{i}条 配额 监控 worker_{worker_id}",
                        )
                        created.append(tid)
                        if i % 5 == 0:
                            db.search_turns_fts("配额", limit=5)
                            db.set_watermark(f"wm_{worker_id}", tid)
                    return created

                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(worker, range(4)))

                all_ids = [tid for group in results for tid in group]
                self.assertEqual(len(all_ids), 60)
                self.assertEqual(len(set(all_ids)), 60)
                hits = db.search_turns_fts("配额", limit=100)
                self.assertEqual(len(hits), 60)


if __name__ == "__main__":
    unittest.main()
