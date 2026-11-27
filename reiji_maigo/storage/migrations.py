"""Idempotent SQLite PRAGMA user_version schema migration gate with backup & rollback protection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import unquote, urlparse

CURRENT_SCHEMA_VERSION: int = 2

_BUSINESS_TABLE_NAMES: frozenset[str] = frozenset(
    {
        "raw_turns",
        "turns",
        "pool_events",
        "memory_tags",
        "memory_cards",
        "card_tags",
        "card_links",
        "meta_watermarks",
        "pending_backfills",
        "audit_chain",
        "card_seals",
    }
)


class SchemaMigrationError(RuntimeError):
    """Raised when schema version validation or forward migration fails."""


@dataclass(frozen=True, slots=True)
class SchemaMigrationStep:
    """Single forward schema migration step targeting a specific user_version."""

    version: int
    name: str
    upgrade: Callable[[sqlite3.Connection], None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Read the current SQLite ``PRAGMA user_version`` integer."""
    row = conn.execute("PRAGMA user_version;").fetchone()
    return int(row[0]) if row is not None else 0


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );
        """
    )


def _list_user_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
          AND name != 'schema_migrations'
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def _has_existing_business_rows(conn: sqlite3.Connection, tables: set[str]) -> bool:
    for table in sorted(tables & _BUSINESS_TABLE_NAMES):
        try:
            row = conn.execute(f'SELECT 1 FROM "{table}" LIMIT 1;').fetchone()
        except sqlite3.DatabaseError:
            continue
        if row is not None:
            return True
    return False


def _resolve_file_db_path(db_path: str | Path) -> Path | None:
    raw = str(db_path or "").strip()
    if not raw or raw == ":memory:":
        return None
    if raw.startswith("file:"):
        parsed = urlparse(raw)
        if parsed.path == ":memory:" or "mode=memory" in (parsed.query or "").lower():
            return None
        fs_path = unquote(parsed.path)
        if os.name == "nt" and fs_path.startswith("/") and len(fs_path) >= 3 and fs_path[2] == ":":
            fs_path = fs_path[1:]
        if not fs_path:
            return None
        return Path(fs_path)
    return Path(raw)


def _create_sqlite_backup(
    conn: sqlite3.Connection,
    db_file: Path,
    old_ver: int,
) -> Path:
    """Create a consistent SQLite snapshot backup via ``Connection.backup()``."""
    base_backup = Path(f"{db_file}.bak-v{old_ver}")
    backup_path = base_backup
    if backup_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = 0
        while True:
            tag = stamp if suffix == 0 else f"{stamp}_{suffix}"
            candidate = Path(f"{db_file}.bak-v{old_ver}.{tag}")
            if not candidate.exists():
                backup_path = candidate
                break
            suffix += 1

    tmp_path = Path(f"{backup_path}.tmp")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    dst_conn: sqlite3.Connection | None = None
    try:
        dst_conn = sqlite3.connect(str(tmp_path))
        conn.backup(dst_conn)
        dst_conn.commit()
    except Exception:
        if dst_conn is not None:
            try:
                dst_conn.close()
            except Exception:
                pass
            dst_conn = None
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    finally:
        if dst_conn is not None:
            dst_conn.close()

    os.replace(tmp_path, backup_path)
    return backup_path


def _migrate_v1_baseline(conn: sqlite3.Connection) -> None:
    """Record baseline v1 schema state without mutating existing core tables."""
    _ensure_migrations_table(conn)


def _migrate_v2_importer_and_indexes(conn: sqlite3.Connection) -> None:
    """Add idempotent import deduplication ledger and timestamp lookup indexes."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS import_fingerprints (
            fingerprint TEXT PRIMARY KEY,
            source_kind TEXT NOT NULL,
            episode_id TEXT NOT NULL,
            turn_id INTEGER NOT NULL,
            imported_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_import_fingerprints_source
            ON import_fingerprints(source_kind, episode_id);
        """
    )
    tables = _list_user_tables(conn)
    if "raw_turns" in tables:
        cols = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(raw_turns);").fetchall()
        }
        if "created_at" in cols:
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_raw_turns_created_at
                    ON raw_turns(created_at);
                """
            )
    if "memory_cards" in tables:
        cols = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(memory_cards);").fetchall()
        }
        if "updated_at" in cols:
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_cards_updated_at
                    ON memory_cards(updated_at);
                """
            )


DEFAULT_MIGRATIONS: tuple[SchemaMigrationStep, ...] = (
    SchemaMigrationStep(
        version=1,
        name="v1_baseline_core_schema",
        upgrade=_migrate_v1_baseline,
    ),
    SchemaMigrationStep(
        version=2,
        name="v2_import_fingerprints_and_indexes",
        upgrade=_migrate_v2_importer_and_indexes,
    ),
)


def apply_schema_migrations(
    conn: sqlite3.Connection,
    *,
    db_path: str = ":memory:",
    target_version: int = CURRENT_SCHEMA_VERSION,
    migrations: Sequence[SchemaMigrationStep] | None = None,
) -> dict[str, Any]:
    """Apply idempotent forward schema migrations gated by ``PRAGMA user_version``.

    - Refuses downgrades when ``old_ver > target_version``.
    - For file-backed databases with existing business tables upgrading across
      versions (``0 < old_ver < target_version``, or legacy unversioned ``v0``
      containing business rows), writes a consistent SQLite ``backup()`` snapshot
      at ``<db_path>.bak-v<old_ver>`` before executing any forward migration DDL.
    - Rolls back failed migration steps atomically without advancing
      ``PRAGMA user_version``.
    """
    if isinstance(target_version, bool) or not isinstance(target_version, int) or target_version < 0:
        raise ValueError("target_version must be a non-negative integer")

    steps = sorted(
        DEFAULT_MIGRATIONS if migrations is None else migrations,
        key=lambda step: step.version,
    )
    old_ver = get_schema_version(conn)

    if old_ver > target_version:
        raise SchemaMigrationError(
            f"Database schema version v{old_ver} is newer than target v{target_version}; "
            "downgrade is not supported."
        )

    if old_ver == target_version:
        _ensure_migrations_table(conn)
        conn.commit()
        return {
            "previous_version": old_ver,
            "current_version": old_ver,
            "target_version": target_version,
            "migrated": False,
            "applied_versions": [],
            "backup_path": None,
        }

    pending = [step for step in steps if old_ver < step.version <= target_version]
    if not pending:
        raise SchemaMigrationError(
            f"No forward migration path registered from v{old_ver} to v{target_version}"
        )
    if pending[-1].version != target_version:
        raise SchemaMigrationError(
            f"Registered migrations end at v{pending[-1].version}, cannot reach v{target_version}"
        )

    # Ensure no uncommitted dirty transaction interferes with backup or savepoints.
    if conn.in_transaction:
        conn.commit()

    backup_path_str: str | None = None
    db_file = _resolve_file_db_path(db_path)
    if db_file is not None and db_file.exists():
        tables = _list_user_tables(conn)
        has_business_tables = bool(tables & _BUSINESS_TABLE_NAMES) or bool(tables)
        needs_backup = False
        if has_business_tables and 0 < old_ver < target_version:
            needs_backup = True
        elif has_business_tables and old_ver == 0 and _has_existing_business_rows(conn, tables):
            needs_backup = True
        if needs_backup:
            created_backup = _create_sqlite_backup(conn, db_file, old_ver)
            backup_path_str = str(created_backup)

    applied_versions: list[int] = []
    current_ver = old_ver

    for step in pending:
        sp_name = f"reiji_mig_v{step.version}"
        conn.execute(f"SAVEPOINT {sp_name};")
        try:
            _ensure_migrations_table(conn)
            step.upgrade(conn)
            conn.execute(
                """
                INSERT INTO schema_migrations (version, name, applied_at)
                VALUES (?, ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                    name = excluded.name,
                    applied_at = excluded.applied_at
                """,
                (step.version, step.name, _utc_now_iso()),
            )
            conn.execute(f"PRAGMA user_version = {int(step.version)};")
            conn.execute(f"RELEASE SAVEPOINT {sp_name};")
            conn.commit()
            current_ver = step.version
            applied_versions.append(step.version)
        except Exception as exc:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {sp_name};")
                conn.execute(f"RELEASE SAVEPOINT {sp_name};")
            except sqlite3.DatabaseError:
                try:
                    conn.rollback()
                except sqlite3.DatabaseError:
                    pass
            # Defensive reset of user_version in case SQLite did not revert PRAGMA.
            try:
                conn.execute(f"PRAGMA user_version = {int(current_ver)};")
                conn.commit()
            except sqlite3.DatabaseError:
                pass
            raise SchemaMigrationError(
                f"Migration step v{step.version} ({step.name}) failed and was rolled back "
                f"(user_version remains v{current_ver}): {exc}"
            ) from exc

    return {
        "previous_version": old_ver,
        "current_version": get_schema_version(conn),
        "target_version": target_version,
        "migrated": bool(applied_versions),
        "applied_versions": applied_versions,
        "backup_path": backup_path_str,
    }


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "DEFAULT_MIGRATIONS",
    "SchemaMigrationError",
    "SchemaMigrationStep",
    "apply_schema_migrations",
    "get_schema_version",
]
