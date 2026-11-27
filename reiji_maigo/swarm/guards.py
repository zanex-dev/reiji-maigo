"""Physical role-based capability guards, event notes staging table, and swarm telemetry.

Three mutually isolated worker capability proxies:
  - ``ScribeCapabilities``: undigested turns, watermarks, ActivePool read/write, event notes staging.
    Forbidden from touching CardStore write APIs or midnight archival/compaction.
  - ``ScoutCapabilities``: strictly read-only across ActivePool snapshots, event notes, tags,
    Level 1/2/3 card recall, and raw span slices. Any write attempt raises ``PermissionError``.
  - ``ArchivistCapabilities``: midnight triage, chain compaction, pool archival, event note
    consumption, CardStore lifecycle (create/supersede/revoke/link), and backfill resolution.
    Forbidden from advancing Scribe watermarks or mutating raw conversation turns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reiji_maigo import SourceSpan
    from reiji_maigo.engine import ReijiMemoryEngine
    from reiji_maigo.storage.sqlite_engine import ReijiDatabase

_SWARM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pool_event_notes (
    event_id TEXT PRIMARY KEY,
    detail_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS swarm_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    action TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_swarm_telemetry_role_created
    ON swarm_telemetry (role, created_at DESC);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_swarm_tables(db: ReijiDatabase) -> None:
    """Idempotently create ``pool_event_notes`` and ``swarm_telemetry`` tables."""
    with db.transaction() as conn:
        conn.executescript(_SWARM_SCHEMA_SQL)


def save_pool_event_note(
    db: ReijiDatabase,
    event_id: str,
    detail: Mapping[str, Any],
    *,
    updated_at: str | None = None,
) -> None:
    """Upsert a Level-2 staging detail JSON payload for a pool event."""
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")
    if not isinstance(detail, Mapping):
        raise TypeError("detail must be a mapping (dict[str, Any])")
    ts = _utc_now_iso() if updated_at is None else str(updated_at)
    serialized = json.dumps(dict(detail), ensure_ascii=False)
    ensure_swarm_tables(db)
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO pool_event_notes (event_id, detail_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                detail_json = excluded.detail_json,
                updated_at = excluded.updated_at
            """,
            (event_id.strip(), serialized, ts),
        )


def get_pool_event_note(db: ReijiDatabase, event_id: str) -> dict[str, Any] | None:
    """Read a staged Level-2 detail note for a pool event, or None if absent."""
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")
    ensure_swarm_tables(db)
    with db.read_connection() as conn:
        row = conn.execute(
            "SELECT detail_json FROM pool_event_notes WHERE event_id = ?",
            (event_id.strip(),),
        ).fetchone()
    if row is None:
        return None
    parsed = json.loads(row["detail_json"])
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def pop_pool_event_note(db: ReijiDatabase, event_id: str) -> dict[str, Any] | None:
    """Atomically read and delete a staged Level-2 detail note for a pool event."""
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")
    cleaned_id = event_id.strip()
    ensure_swarm_tables(db)
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT detail_json FROM pool_event_notes WHERE event_id = ?",
            (cleaned_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "DELETE FROM pool_event_notes WHERE event_id = ?",
            (cleaned_id,),
        )
    parsed = json.loads(row["detail_json"])
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def record_swarm_telemetry(
    db: ReijiDatabase,
    *,
    role: str,
    action: str,
    metrics: Mapping[str, Any],
    created_at: str | None = None,
) -> int:
    """Append an immutable telemetry ledger record for a swarm worker action."""
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be a non-empty string")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("action must be a non-empty string")
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    ts = _utc_now_iso() if created_at is None else str(created_at)
    serialized = json.dumps(dict(metrics), ensure_ascii=False)
    ensure_swarm_tables(db)
    with db.transaction() as conn:
        cur = conn.execute(
            """
            INSERT INTO swarm_telemetry (role, action, metrics_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (role.strip(), action.strip(), serialized, ts),
        )
        return int(cur.lastrowid)


def list_swarm_telemetry(
    db: ReijiDatabase,
    *,
    role: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query recent swarm telemetry rows ordered newest-first."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative int")
    if limit == 0:
        return []
    ensure_swarm_tables(db)
    with db.read_connection() as conn:
        if role is not None:
            rows = conn.execute(
                """
                SELECT id, role, action, metrics_json, created_at
                FROM swarm_telemetry
                WHERE role = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (role.strip(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, role, action, metrics_json, created_at
                FROM swarm_telemetry
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "role": str(r["role"]),
            "action": str(r["action"]),
            "metrics": json.loads(r["metrics_json"]),
            "created_at": str(r["created_at"]),
        }
        for r in rows
    ]


class _BaseCapabilities:
    """Base sandbox proxy enforcing physical deny-by-default capability boundaries."""

    ROLE: str = ""
    __slots__ = ("_engine",)

    def __init__(self, engine: ReijiMemoryEngine) -> None:
        if engine is None or not hasattr(engine, "db"):
            raise TypeError("engine must be a ReijiMemoryEngine instance")
        ensure_swarm_tables(engine.db)
        object.__setattr__(self, "_engine", engine)

    def __setattr__(self, name: str, value: Any) -> None:
        raise PermissionError(
            f"Role {self.ROLE!r} capability sandbox is immutable; cannot set {name!r}"
        )

    def __delattr__(self, name: str) -> None:
        raise PermissionError(
            f"Role {self.ROLE!r} capability sandbox is immutable; cannot delete {name!r}"
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise PermissionError(
            f"Role {self.ROLE!r} is physically forbidden from accessing {name!r}"
        )

    def record_telemetry(
        self,
        action: str,
        metrics: Mapping[str, Any],
        *,
        created_at: str | None = None,
    ) -> int:
        """Record a telemetry entry locked to this worker's role."""
        return record_swarm_telemetry(
            self._engine.db,
            role=self.ROLE,
            action=action,
            metrics=metrics,
            created_at=created_at,
        )

    def _call_first_attr(
        self,
        target: Any,
        candidates: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        for attr in candidates:
            fn = getattr(target, attr, None)
            if callable(fn):
                return fn(*args, **kwargs)
        raise AttributeError(
            f"{type(target).__name__} implements none of {candidates!r}"
        )


class ScribeCapabilities(_BaseCapabilities):
    """Capability sandbox for Scribe (速记蜂).

    Allowed:
      - Read undigested turns & read/update Scribe watermarks
      - ActivePool read/write (snapshot, get/list/open/create/append_step/update_status)
      - Stage & inspect Level-2 event detail notes (``pool_event_notes``)
      - Record Scribe telemetry

    Physically blocked (raises ``PermissionError``):
      - All ``CardStore`` write interfaces (create/supersede/revoke/link/resolve_backfill)
      - Midnight archival / compaction / event note pop
      - Direct access to ``engine`` / ``db`` / ``cards`` / ``pool`` / ``kernel``
    """

    ROLE = "scribe"
    __slots__ = ()

    # --- Turns & watermarks ---

    def get_undigested_turns(
        self,
        watermark_key: str = "pool_digested",
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        return self._engine.db.get_undigested_turns(
            watermark_key=watermark_key,
            limit=limit,
        )

    def get_watermark(self, key: str = "pool_digested") -> int:
        return self._engine.db.get_watermark(key)

    def set_watermark(self, key: str, turn_id: int) -> None:
        self._engine.db.set_watermark(key, turn_id)

    # --- ActivePool read/write ---

    def pool_snapshot(
        self,
        *,
        now_iso: str | None = None,
        include_undigested: bool = True,
    ) -> str:
        return self._engine.pool_snapshot(
            now_iso=now_iso,
            include_undigested=include_undigested,
        )

    def get_event(self, event_id: str) -> Any:
        return self._engine.pool.get_event(event_id)

    def list_events(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_first_attr(
            self._engine.pool,
            ("list_events", "list_active_events", "get_active_events"),
            *args,
            **kwargs,
        )

    def create_event(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_first_attr(
            self._engine.pool,
            ("create_event", "open_event", "add_event"),
            *args,
            **kwargs,
        )

    def open_event(self, *args: Any, **kwargs: Any) -> Any:
        return self.create_event(*args, **kwargs)

    def append_step(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.pool.append_step(*args, **kwargs)

    def update_status(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.pool.update_status(*args, **kwargs)

    def touch_event(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.pool.touch_event(*args, **kwargs)

    # --- Event detail staging notes ---

    def save_event_note(
        self,
        event_id: str,
        detail: Mapping[str, Any],
        *,
        updated_at: str | None = None,
    ) -> None:
        save_pool_event_note(
            self._engine.db,
            event_id,
            detail,
            updated_at=updated_at,
        )

    def get_event_note(self, event_id: str) -> dict[str, Any] | None:
        return get_pool_event_note(self._engine.db, event_id)


class ScoutCapabilities(_BaseCapabilities):
    """Strictly read-only capability sandbox for Scout (侦察蜂).

    Allowed (read-only):
      - ``pool_snapshot``, ``get_event``, ``get_event_note``
      - ``list_tags``, ``get_card``
      - ``recall_level1``, ``read_evidence_level2``, ``deep_slice_level3``
      - ``read_span``, ``search_turns_fts``
      - ``record_telemetry`` (locked to role ``scout``)

    Any write action or access to internal handles raises ``PermissionError``.
    """

    ROLE = "scout"
    __slots__ = ()

    def pool_snapshot(
        self,
        *,
        now_iso: str | None = None,
        include_undigested: bool = True,
    ) -> str:
        return self._engine.pool_snapshot(
            now_iso=now_iso,
            include_undigested=include_undigested,
        )

    def get_event(self, event_id: str) -> Any:
        return self._engine.pool.get_event(event_id)

    def get_event_note(self, event_id: str) -> dict[str, Any] | None:
        return get_pool_event_note(self._engine.db, event_id)

    def list_tags(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.cards.list_tags(*args, **kwargs)

    def get_card(self, card_id: str) -> Any:
        return self._engine.cards.get_card(card_id)

    def recall_level1(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.cards.recall_level1(*args, **kwargs)

    def read_evidence_level2(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.cards.read_evidence_level2(*args, **kwargs)

    def deep_slice_level3(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.cards.deep_slice_level3(*args, **kwargs)

    def read_span(
        self,
        span: SourceSpan,
        *,
        pad: int = 0,
        max_turns: int = 40,
    ) -> list[dict[str, Any]]:
        return self._engine.db.read_span(span, pad=pad, max_turns=max_turns)

    def search_turns_fts(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_episode: str = "",
    ) -> list[dict[str, Any]]:
        return self._engine.db.search_turns_fts(
            query,
            limit=limit,
            exclude_episode=exclude_episode,
        )


class ArchivistCapabilities(_BaseCapabilities):
    """Capability sandbox for Archivist (馆长蜂).

    Allowed:
      - ``plan_midnight_triage``, ``compact_chain``, ``archive_pool_event``
      - ``pool_snapshot``, ``get_event``, ``list_events``, ``update_status``
      - ``get_or_pop_event_note`` (plus ``get_event_note`` / ``pop_event_note``)
      - ``CardStore`` curation: ``create_card``, ``supersede_card``, ``revoke_card``,
        ``link_cards``, ``get_card``, ``list_tags``, ``recall_level1``,
        ``read_evidence_level2``, ``read_span``
      - ``resolve_backfill`` (plus ``list_pending_backfills``)
      - ``record_telemetry`` (locked to role ``archivist``)

    Physically blocked (raises ``PermissionError``):
      - Scribe watermark mutation (``set_watermark``), raw turn appends (``append_turn``),
        event note staging writes (``save_event_note``), and direct internal handles.
    """

    ROLE = "archivist"
    __slots__ = ()

    # --- Midnight triage, compaction & pool archival ---

    def plan_midnight_triage(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.pool.plan_midnight_triage(*args, **kwargs)

    def compact_chain(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.pool.compact_chain(*args, **kwargs)

    def increment_dormant_nights(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.pool.increment_dormant_nights(*args, **kwargs)

    def archive_pool_event(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.archive_pool_event(*args, **kwargs)

    def pool_snapshot(
        self,
        *,
        now_iso: str | None = None,
        include_undigested: bool = True,
    ) -> str:
        return self._engine.pool_snapshot(
            now_iso=now_iso,
            include_undigested=include_undigested,
        )

    def get_event(self, event_id: str) -> Any:
        return self._engine.pool.get_event(event_id)

    def list_events(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_first_attr(
            self._engine.pool,
            ("list_events", "list_active_events", "get_active_events"),
            *args,
            **kwargs,
        )

    def update_status(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.pool.update_status(*args, **kwargs)

    # --- Event note consumption ---

    def get_event_note(self, event_id: str) -> dict[str, Any] | None:
        return get_pool_event_note(self._engine.db, event_id)

    def pop_event_note(self, event_id: str) -> dict[str, Any] | None:
        return pop_pool_event_note(self._engine.db, event_id)

    def get_or_pop_event_note(
        self,
        event_id: str,
        *,
        pop: bool = False,
    ) -> dict[str, Any] | None:
        if pop:
            return pop_pool_event_note(self._engine.db, event_id)
        return get_pool_event_note(self._engine.db, event_id)

    # --- CardStore curation & readback ---

    def create_card(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.cards.create_card(*args, **kwargs)

    def supersede_card(
        self,
        old_card_id: str,
        new_card: Any,
        *,
        now_iso: str | None = None,
        actor: str = "swarm:archivist",
        origin_trust: str = "assistant",
        poison_detected: bool = False,
        reason: str = "",
    ) -> Any:
        gate = getattr(self._engine, "approval_gate", None)
        if gate is None:
            return self._engine.cards.supersede_card(
                old_card_id,
                new_card,
                now_iso=now_iso,
            )
        outcome = gate.guard_supersede_or_revoke(
            self._engine.cards,
            actor=actor,
            op_kind="supersede",
            target_card_id=old_card_id,
            replacement_card=new_card,
            origin_trust=origin_trust,
            poison_detected=poison_detected,
            reason=reason,
            now_iso=now_iso,
        )
        if outcome.get("allowed"):
            return outcome.get("result_card")
        return outcome

    def revoke_card(
        self,
        card_id: str,
        *,
        now_iso: str | None = None,
        actor: str = "swarm:archivist",
        origin_trust: str = "assistant",
        poison_detected: bool = False,
        reason: str = "",
    ) -> bool:
        gate = getattr(self._engine, "approval_gate", None)
        if gate is None:
            return bool(self._engine.cards.revoke_card(card_id, now_iso=now_iso))
        outcome = gate.guard_supersede_or_revoke(
            self._engine.cards,
            actor=actor,
            op_kind="revoke",
            target_card_id=card_id,
            replacement_card=None,
            origin_trust=origin_trust,
            poison_detected=poison_detected,
            reason=reason,
            now_iso=now_iso,
        )
        if outcome.get("allowed"):
            return bool(outcome.get("revoked", False))
        return False

    def link_cards(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_first_attr(
            self._engine.cards,
            ("link_cards", "add_link", "create_link"),
            *args,
            **kwargs,
        )

    def get_card(self, card_id: str) -> Any:
        return self._engine.cards.get_card(card_id)

    def list_tags(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.cards.list_tags(*args, **kwargs)

    def register_or_resolve_tag(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.cards.register_or_resolve_tag(*args, **kwargs)

    def recall_level1(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.cards.recall_level1(*args, **kwargs)

    def read_evidence_level2(self, *args: Any, **kwargs: Any) -> Any:
        return self._engine.cards.read_evidence_level2(*args, **kwargs)

    def read_span(
        self,
        span: SourceSpan,
        *,
        pad: int = 0,
        max_turns: int = 40,
    ) -> list[dict[str, Any]]:
        return self._engine.db.read_span(span, pad=pad, max_turns=max_turns)

    # --- Pending backfills ---

    def resolve_backfill(
        self,
        backfill_id: int,
        *,
        status: str = "resolved",
        increment_attempts: bool = False,
        now_iso: str | None = None,
    ) -> bool:
        """Resolve (or update status of) a pending_backfills record by backfill_id."""
        fn = getattr(self._engine.cards, "resolve_backfill", None)
        if callable(fn):
            return bool(
                fn(
                    backfill_id,
                    status=status,
                    increment_attempts=increment_attempts,
                    now_iso=now_iso,
                )
            )
        if isinstance(backfill_id, bool) or not isinstance(backfill_id, int) or backfill_id <= 0:
            raise ValueError("backfill_id must be a positive int")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be a non-empty string")
        ts = _utc_now_iso() if now_iso is None else str(now_iso)
        delta = 1 if increment_attempts else 0
        with self._engine.db.transaction() as conn:
            cur = conn.execute(
                """
                UPDATE pending_backfills
                SET status = ?,
                    attempts = attempts + ?,
                    updated_at = ?
                WHERE backfill_id = ?
                """,
                (status.strip(), delta, ts, backfill_id),
            )
            return cur.rowcount > 0

    def list_pending_backfills(self, *, status: str | None = "pending") -> list[dict[str, Any]]:
        return self._engine.cards.list_pending_backfills(status=status)


def build_role_capabilities(
    engine: ReijiMemoryEngine,
    role: str,
) -> ScribeCapabilities | ScoutCapabilities | ArchivistCapabilities:
    """Factory returning the physically isolated capability proxy for ``role``."""
    normalized = role.strip().lower() if isinstance(role, str) else ""
    if normalized == "scribe":
        return ScribeCapabilities(engine)
    if normalized == "scout":
        return ScoutCapabilities(engine)
    if normalized == "archivist":
        return ArchivistCapabilities(engine)
    raise ValueError(f"Unknown swarm role: {role!r}")


__all__ = [
    "ArchivistCapabilities",
    "ScoutCapabilities",
    "ScribeCapabilities",
    "build_role_capabilities",
    "ensure_swarm_tables",
    "get_pool_event_note",
    "list_swarm_telemetry",
    "pop_pool_event_note",
    "record_swarm_telemetry",
    "save_pool_event_note",
]
