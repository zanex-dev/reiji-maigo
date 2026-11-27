"""Thread-safe single-file SQLite WAL storage engine with Bigram FTS5 index."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Any

from reiji_maigo.core.contracts import SourceSpan
from reiji_maigo.storage.migrations import apply_schema_migrations
from reiji_maigo.storage.tokenizer import build_fts_query, tokenize_for_fts

MAX_SPAN_TURNS: int = 40

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_turns_episode_turn
    ON raw_turns(episode_id, turn_id);

CREATE VIRTUAL TABLE IF NOT EXISTS raw_turns_fts USING fts5(
    tokenized_content,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS pool_events (
    event_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    steps_json TEXT NOT NULL DEFAULT '[]',
    checkpoint_summary TEXT NOT NULL DEFAULT '',
    spans_json TEXT NOT NULL DEFAULT '[]',
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_touched_at TEXT NOT NULL,
    dormant_nights INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_pool_events_status_touched
    ON pool_events(status, last_touched_at DESC);

CREATE INDEX IF NOT EXISTS idx_pool_events_scope
    ON pool_events(scope);

CREATE TABLE IF NOT EXISTS memory_tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL CHECK(domain IN ('proj', 'infra', 'research', 'companion', 'life')),
    canonical_name TEXT NOT NULL UNIQUE,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    usage_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_memory_tags_domain
    ON memory_tags(domain);

CREATE TABLE IF NOT EXISTS memory_cards (
    card_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL CHECK(domain IN ('proj', 'infra', 'research', 'companion', 'life')),
    card_type TEXT NOT NULL,
    title TEXT NOT NULL,
    state_json TEXT NOT NULL DEFAULT '{}',
    lifecycle_status TEXT NOT NULL DEFAULT 'active',
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    superseded_by TEXT REFERENCES memory_cards(card_id),
    spans_json TEXT NOT NULL DEFAULT '[]',
    foresight_note TEXT NOT NULL DEFAULT '',
    foresight_until TEXT,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_memory_cards_domain_status
    ON memory_cards(domain, lifecycle_status);

CREATE INDEX IF NOT EXISTS idx_memory_cards_valid_window
    ON memory_cards(valid_from, valid_to);

CREATE INDEX IF NOT EXISTS idx_memory_cards_foresight
    ON memory_cards(foresight_until);

CREATE TABLE IF NOT EXISTS card_tags (
    card_id TEXT NOT NULL REFERENCES memory_cards(card_id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES memory_tags(tag_id) ON DELETE CASCADE,
    PRIMARY KEY (card_id, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_card_tags_tag_id
    ON card_tags(tag_id);

CREATE TABLE IF NOT EXISTS card_links (
    src_card_id TEXT NOT NULL REFERENCES memory_cards(card_id) ON DELETE CASCADE,
    dst_card_id TEXT NOT NULL REFERENCES memory_cards(card_id) ON DELETE CASCADE,
    relation TEXT NOT NULL DEFAULT 'related',
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (src_card_id, dst_card_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_card_links_src
    ON card_links(src_card_id);

CREATE INDEX IF NOT EXISTS idx_card_links_dst
    ON card_links(dst_card_id);

CREATE TABLE IF NOT EXISTS meta_watermarks (
    key TEXT PRIMARY KEY,
    turn_id INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS pending_backfills (
    backfill_id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    turn_start INTEGER NOT NULL,
    turn_end INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_pending_backfills_status
    ON pending_backfills(status, backfill_id);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_turn_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "turn_id": int(row["turn_id"]),
        "episode_id": str(row["episode_id"]),
        "role": str(row["role"]),
        "content": str(row["content"]),
        "created_at": str(row["created_at"]),
    }


class ReijiDatabase:
    """Thread-safe SQLite WAL database for Reiji-Maigo memory storage and Bigram FTS5."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        self._closed = False

        is_memory = self._db_path == ":memory:" or self._db_path.startswith("file:")
        if not is_memory:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self._db_path,
            timeout=5.0,
            check_same_thread=False,
            uri=self._db_path.startswith("file:"),
        )
        self._conn.row_factory = sqlite3.Row
        self._configure_and_init()

    def _configure_and_init(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA busy_timeout = 5000;")
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
            self._migration_report = apply_schema_migrations(
                self._conn,
                db_path=self._db_path,
            )

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> ReijiDatabase:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Acquire the database RLock and execute a transactional block."""
        with self._lock:
            with self._conn:
                yield self._conn

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """Acquire the database RLock for consistent read operations."""
        with self._lock:
            yield self._conn

    def append_turn(
        self,
        episode_id: str,
        role: str,
        content: str,
        created_at: str | None = None,
    ) -> int:
        """Append a raw conversation turn and synchronize its Bigram FTS5 entry."""
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        timestamp = _utc_now_iso() if created_at is None else str(created_at)
        tokenized = tokenize_for_fts(content)

        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO raw_turns (episode_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (episode_id, role, content, timestamp),
            )
            turn_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO raw_turns_fts (rowid, tokenized_content)
                VALUES (?, ?)
                """,
                (turn_id, tokenized),
            )
            return turn_id

    def read_span(
        self,
        span: SourceSpan,
        *,
        pad: int = 0,
        max_turns: int = MAX_SPAN_TURNS,
    ) -> list[dict[str, Any]]:
        """Read turns for a SourceSpan with optional episode-local padding and hard <=40 cap."""
        if not isinstance(span, SourceSpan):
            raise TypeError("span must be a SourceSpan instance")
        if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
            raise ValueError("pad must be a non-negative int")
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 0:
            raise ValueError("max_turns must be a non-negative int")

        cap = min(max_turns, MAX_SPAN_TURNS)
        if cap == 0:
            return []

        with self.read_connection() as conn:
            before_rows: list[sqlite3.Row] = []
            if pad > 0:
                before_rows = list(
                    reversed(
                        conn.execute(
                            """
                            SELECT turn_id, episode_id, role, content, created_at
                            FROM raw_turns
                            WHERE episode_id = ? AND turn_id < ?
                            ORDER BY turn_id DESC
                            LIMIT ?
                            """,
                            (span.episode_id, span.turn_start, min(pad, cap)),
                        ).fetchall()
                    )
                )

            core_rows = conn.execute(
                """
                SELECT turn_id, episode_id, role, content, created_at
                FROM raw_turns
                WHERE episode_id = ? AND turn_id BETWEEN ? AND ?
                ORDER BY turn_id ASC
                LIMIT ?
                """,
                (span.episode_id, span.turn_start, span.turn_end, cap),
            ).fetchall()

            after_rows: list[sqlite3.Row] = []
            if pad > 0:
                after_rows = conn.execute(
                    """
                    SELECT turn_id, episode_id, role, content, created_at
                    FROM raw_turns
                    WHERE episode_id = ? AND turn_id > ?
                    ORDER BY turn_id ASC
                    LIMIT ?
                    """,
                    (span.episode_id, span.turn_end, min(pad, cap)),
                ).fetchall()

        combined = (before_rows + list(core_rows) + list(after_rows))[:cap]
        return [_row_to_turn_dict(row) for row in combined]

    def search_turns_fts(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_episode: str = "",
    ) -> list[dict[str, Any]]:
        """Search raw turns via the Bigram FTS5 index ranked by BM25."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative int")
        if limit == 0:
            return []

        fts_query = build_fts_query(query)
        if not fts_query:
            return []

        with self.read_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    t.turn_id,
                    t.episode_id,
                    t.role,
                    t.content,
                    t.created_at,
                    bm25(raw_turns_fts) AS bm25_score
                FROM raw_turns_fts
                JOIN raw_turns AS t ON t.turn_id = raw_turns_fts.rowid
                WHERE raw_turns_fts MATCH ?
                  AND (? = '' OR t.episode_id != ?)
                ORDER BY bm25(raw_turns_fts) ASC, t.turn_id DESC
                LIMIT ?
                """,
                (fts_query, exclude_episode, exclude_episode, limit),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            item = _row_to_turn_dict(row)
            bm25_val = float(row["bm25_score"])
            item["bm25"] = bm25_val
            item["score"] = -bm25_val
            results.append(item)
        return results

    def get_watermark(self, key: str) -> int:
        """Read the persisted turn_id watermark for a given consumer key (defaults to 0)."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("watermark key must be a non-empty string")
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT turn_id FROM meta_watermarks WHERE key = ?",
                (key,),
            ).fetchone()
            return int(row["turn_id"]) if row is not None else 0

    def set_watermark(self, key: str, turn_id: int) -> None:
        """Persist the turn_id watermark for a given consumer key."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("watermark key must be a non-empty string")
        if isinstance(turn_id, bool) or not isinstance(turn_id, int) or turn_id < 0:
            raise ValueError("turn_id must be a non-negative int")

        now = _utc_now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO meta_watermarks (key, turn_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    turn_id = excluded.turn_id,
                    updated_at = excluded.updated_at
                """,
                (key, turn_id, now),
            )

    def get_undigested_turns(
        self,
        watermark_key: str = "pool_digested",
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Fetch raw turns strictly newer than the stored watermark, ordered by turn_id ASC."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative int")
        if limit == 0:
            return []
        if not isinstance(watermark_key, str) or not watermark_key.strip():
            raise ValueError("watermark_key must be a non-empty string")

        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT turn_id FROM meta_watermarks WHERE key = ?",
                (watermark_key,),
            ).fetchone()
            last_turn_id = int(row["turn_id"]) if row is not None else 0

            rows = conn.execute(
                """
                SELECT turn_id, episode_id, role, content, created_at
                FROM raw_turns
                WHERE turn_id > ?
                ORDER BY turn_id ASC
                LIMIT ?
                """,
                (last_turn_id, limit),
            ).fetchall()

        return [_row_to_turn_dict(r) for r in rows]
