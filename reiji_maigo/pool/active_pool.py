"""Active short-term working memory pool with 4D Zeigarnik decay and lossless compaction."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
import math
import sqlite3
import uuid

from reiji_maigo.core.contracts import PoolEvent, SourceSpan
from reiji_maigo.storage.sqlite_engine import ReijiDatabase

VALID_SCOPES: tuple[str, ...] = ("quest", "micro")
_VALID_SCOPE_SET: frozenset[str] = frozenset(VALID_SCOPES)

POOL_VISIBLE_STATUSES: tuple[str, ...] = ("active", "blocked", "done", "aborted")
OFF_POOL_STATUSES: tuple[str, ...] = ("archived", "hibernated", "dropped")
VALID_STATUSES: tuple[str, ...] = POOL_VISIBLE_STATUSES + OFF_POOL_STATUSES
_VALID_STATUS_SET: frozenset[str] = frozenset(VALID_STATUSES)

_STATUS_WEIGHTS: dict[str, float] = {
    "active": 1.5,
    "blocked": 1.5,
    "done": 0.35,
    "aborted": 0.35,
    "hibernated": 0.2,
    "archived": 0.1,
    "dropped": 0.1,
}

_SCOPE_WEIGHTS: dict[str, float] = {
    "quest": 2.0,
    "micro": 0.8,
}

_SCOPE_HALF_LIFE_HOURS: dict[str, float] = {
    "quest": 168.0,
    "micro": 12.0,
}

_LN2: float = math.log(2.0)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_utc(iso_str: str) -> datetime:
    text = iso_str.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _elapsed_hours(start_iso: str, end_iso: str) -> float:
    start_dt = _parse_iso_utc(start_iso)
    end_dt = _parse_iso_utc(end_iso)
    delta_seconds = (end_dt - start_dt).total_seconds()
    return max(0.0, delta_seconds / 3600.0)


def estimate_tokens(text: str) -> int:
    """Estimate LLM token count for mixed CJK/ASCII text without external dependencies."""
    if not text:
        return 0
    cjk_or_wide = 0
    ascii_chars = 0
    for ch in text:
        if ord(ch) >= 0x2E80:
            cjk_or_wide += 1
        else:
            ascii_chars += 1
    est = math.ceil(cjk_or_wide * 1.0 + ascii_chars / 4.0)
    return max(1, est)


def estimate_event_tokens(ev: PoolEvent) -> int:
    """Estimate token footprint of a PoolEvent."""
    parts = [ev.title]
    if ev.checkpoint_summary:
        parts.append(ev.checkpoint_summary)
    if ev.steps:
        parts.extend(ev.steps)
    combined = "\n".join(parts)
    # Include lightweight metadata overhead for status/scope/span markers.
    return estimate_tokens(combined) + len(ev.spans) * 2


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text so estimate_tokens(result) <= max_tokens."""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    ellipsis = "..."
    if max_tokens <= estimate_tokens(ellipsis):
        # Very tight budget fallback: binary search raw prefix.
        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid]
            if estimate_tokens(candidate) <= max_tokens:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    lo, hi = 0, len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if estimate_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def compute_retention_score(ev: PoolEvent, now_iso: str) -> float:
    """Compute 4D Zeigarnik retention score for a pool event.

    Formula:
        W_status(active/blocked=1.5, done/aborted=0.35)
        * P_scope(quest=2.0, micro=0.8)
        * exp(-ln(2) * dt_hours / half_life(quest=168h, micro=12h))
        * (1 + 0.3 * ln(1 + hit_count))
        / max(20, est_tokens)
    """
    w_status = _STATUS_WEIGHTS.get(ev.status, 0.35)
    p_scope = _SCOPE_WEIGHTS.get(ev.scope, 1.0)
    half_life = _SCOPE_HALF_LIFE_HOURS.get(ev.scope, 48.0)

    dt_hours = _elapsed_hours(ev.last_touched_at, now_iso)
    decay = math.exp(-_LN2 * dt_hours / half_life)
    hit_boost = 1.0 + 0.3 * math.log(1.0 + ev.hit_count)
    denom = float(max(20, estimate_event_tokens(ev)))
    return (w_status * p_scope * decay * hit_boost) / denom


def _row_to_event(row: sqlite3.Row) -> PoolEvent:
    steps_raw = json.loads(str(row["steps_json"]))
    spans_raw = json.loads(str(row["spans_json"]))
    return PoolEvent(
        event_id=str(row["event_id"]),
        title=str(row["title"]),
        scope=str(row["scope"]),
        status=str(row["status"]),
        steps=tuple(str(s) for s in steps_raw),
        checkpoint_summary=str(row["checkpoint_summary"]),
        spans=tuple(SourceSpan.from_dict(item) for item in spans_raw),
        hit_count=int(row["hit_count"]),
        created_at=str(row["created_at"]),
        last_touched_at=str(row["last_touched_at"]),
        dormant_nights=int(row["dormant_nights"]),
    )


def _format_spans_compact(spans: tuple[SourceSpan, ...]) -> str:
    return ", ".join(f"{s.episode_id}:{s.turn_start}-{s.turn_end}" for s in spans)


def _render_event_block(ev: PoolEvent, score: float) -> str:
    lines = [
        f"- [{ev.status}|{ev.scope}] {ev.title} (id={ev.event_id}, hits={ev.hit_count}, score={score:.4f})"
    ]
    if ev.checkpoint_summary:
        lines.append(f"  summary: {ev.checkpoint_summary}")
    if ev.steps:
        lines.append(f"  steps: {' -> '.join(ev.steps)}")
    if ev.spans:
        lines.append(f"  spans: {_format_spans_compact(ev.spans)}")
    return "\n".join(lines)


class ActivePool:
    """Linear hot working-memory event pool backed by ReijiDatabase."""

    def __init__(
        self,
        db: ReijiDatabase,
        *,
        max_events: int = 8,
        max_tokens: int = 900,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        if max_tokens < 20:
            raise ValueError("max_tokens must be >= 20")
        self._db = db
        self._max_events = max_events
        self._max_tokens = max_tokens

    @property
    def max_events(self) -> int:
        return self._max_events

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _save_event(self, conn: sqlite3.Connection, ev: PoolEvent) -> None:
        conn.execute(
            """
            INSERT INTO pool_events (
                event_id, title, scope, status,
                steps_json, checkpoint_summary, spans_json,
                hit_count, created_at, last_touched_at, dormant_nights
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                title = excluded.title,
                scope = excluded.scope,
                status = excluded.status,
                steps_json = excluded.steps_json,
                checkpoint_summary = excluded.checkpoint_summary,
                spans_json = excluded.spans_json,
                hit_count = excluded.hit_count,
                created_at = excluded.created_at,
                last_touched_at = excluded.last_touched_at,
                dormant_nights = excluded.dormant_nights
            """,
            (
                ev.event_id,
                ev.title,
                ev.scope,
                ev.status,
                json.dumps(list(ev.steps), ensure_ascii=False),
                ev.checkpoint_summary,
                json.dumps([s.to_dict() for s in ev.spans], ensure_ascii=False),
                ev.hit_count,
                ev.created_at,
                ev.last_touched_at,
                ev.dormant_nights,
            ),
        )

    def _fetch_event_in_conn(
        self, conn: sqlite3.Connection, event_id: str
    ) -> PoolEvent:
        row = conn.execute(
            "SELECT * FROM pool_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"PoolEvent not found: {event_id!r}")
        return _row_to_event(row)

    def get_event(self, event_id: str) -> PoolEvent | None:
        with self._db.read_connection() as conn:
            row = conn.execute(
                "SELECT * FROM pool_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            return _row_to_event(row) if row is not None else None

    def list_events(
        self,
        *,
        statuses: tuple[str, ...] | None = None,
        include_inactive: bool = False,
    ) -> list[PoolEvent]:
        with self._db.read_connection() as conn:
            if statuses is not None:
                if not statuses:
                    return []
                placeholders = ",".join("?" for _ in statuses)
                rows = conn.execute(
                    f"""
                    SELECT * FROM pool_events
                    WHERE status IN ({placeholders})
                    ORDER BY last_touched_at DESC, event_id ASC
                    """,
                    tuple(statuses),
                ).fetchall()
            elif include_inactive:
                rows = conn.execute(
                    """
                    SELECT * FROM pool_events
                    ORDER BY last_touched_at DESC, event_id ASC
                    """
                ).fetchall()
            else:
                placeholders = ",".join("?" for _ in POOL_VISIBLE_STATUSES)
                rows = conn.execute(
                    f"""
                    SELECT * FROM pool_events
                    WHERE status IN ({placeholders})
                    ORDER BY last_touched_at DESC, event_id ASC
                    """,
                    POOL_VISIBLE_STATUSES,
                ).fetchall()
            return [_row_to_event(row) for row in rows]

    def create_event(
        self,
        title: str,
        step: str,
        span: SourceSpan,
        *,
        scope: str = "quest",
        now_iso: str | None = None,
    ) -> PoolEvent:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(step, str) or not step.strip():
            raise ValueError("step must be a non-empty string")
        if not isinstance(span, SourceSpan):
            raise TypeError("span must be a SourceSpan")
        if scope not in _VALID_SCOPE_SET:
            raise ValueError(f"Invalid scope {scope!r}; expected one of: {VALID_SCOPES}")

        ts = now_iso if now_iso is not None else _utc_now_iso()
        event = PoolEvent(
            event_id=f"ev_{uuid.uuid4().hex[:12]}",
            title=title.strip(),
            scope=scope,
            status="active",
            steps=(step.strip(),),
            checkpoint_summary="",
            spans=(span,),
            hit_count=0,
            created_at=ts,
            last_touched_at=ts,
            dormant_nights=0,
        )
        with self._db.transaction() as conn:
            self._save_event(conn, event)
        return event

    def append_step(
        self,
        event_id: str,
        step: str,
        span: SourceSpan,
        *,
        status: str | None = None,
        now_iso: str | None = None,
    ) -> PoolEvent:
        if not isinstance(step, str) or not step.strip():
            raise ValueError("step must be a non-empty string")
        if not isinstance(span, SourceSpan):
            raise TypeError("span must be a SourceSpan")
        if status is not None and status not in _VALID_STATUS_SET:
            raise ValueError(
                f"Invalid status {status!r}; expected one of: {VALID_STATUSES}"
            )

        ts = now_iso if now_iso is not None else _utc_now_iso()
        with self._db.transaction() as conn:
            existing = self._fetch_event_in_conn(conn, event_id)
            next_status = status
            if next_status is None:
                next_status = (
                    "active" if existing.status == "hibernated" else existing.status
                )
            next_spans = (
                existing.spans
                if span in existing.spans
                else existing.spans + (span,)
            )
            updated = PoolEvent(
                event_id=existing.event_id,
                title=existing.title,
                scope=existing.scope,
                status=next_status,
                steps=existing.steps + (step.strip(),),
                checkpoint_summary=existing.checkpoint_summary,
                spans=next_spans,
                hit_count=existing.hit_count + 1,
                created_at=existing.created_at,
                last_touched_at=ts,
                dormant_nights=0,
            )
            self._save_event(conn, updated)
            return updated

    def touch_event(
        self,
        event_id: str,
        *,
        now_iso: str | None = None,
    ) -> PoolEvent:
        """Refresh recency, increment hit_count, and reset dormant_nights."""
        ts = now_iso if now_iso is not None else _utc_now_iso()
        with self._db.transaction() as conn:
            existing = self._fetch_event_in_conn(conn, event_id)
            updated = PoolEvent(
                event_id=existing.event_id,
                title=existing.title,
                scope=existing.scope,
                status=existing.status,
                steps=existing.steps,
                checkpoint_summary=existing.checkpoint_summary,
                spans=existing.spans,
                hit_count=existing.hit_count + 1,
                created_at=existing.created_at,
                last_touched_at=ts,
                dormant_nights=0,
            )
            self._save_event(conn, updated)
            return updated

    def update_status(
        self,
        event_id: str,
        status: str,
        *,
        now_iso: str | None = None,
    ) -> PoolEvent:
        if status not in _VALID_STATUS_SET:
            raise ValueError(
                f"Invalid status {status!r}; expected one of: {VALID_STATUSES}"
            )
        with self._db.transaction() as conn:
            existing = self._fetch_event_in_conn(conn, event_id)
            updated = PoolEvent(
                event_id=existing.event_id,
                title=existing.title,
                scope=existing.scope,
                status=status,
                steps=existing.steps,
                checkpoint_summary=existing.checkpoint_summary,
                spans=existing.spans,
                hit_count=existing.hit_count,
                created_at=existing.created_at,
                last_touched_at=now_iso if now_iso is not None else existing.last_touched_at,
                dormant_nights=existing.dormant_nights,
            )
            self._save_event(conn, updated)
            return updated

    def increment_dormant_nights(
        self,
        event_ids: Iterable[str] | None = None,
        *,
        delta: int = 1,
    ) -> int:
        """Increment dormant_nights for specified events (or all active/blocked pool events)."""
        if delta < 1:
            raise ValueError("delta must be >= 1")
        with self._db.transaction() as conn:
            if event_ids is None:
                cur = conn.execute(
                    """
                    UPDATE pool_events
                    SET dormant_nights = dormant_nights + ?
                    WHERE status IN ('active', 'blocked')
                    """,
                    (delta,),
                )
                return cur.rowcount

            ids = list(event_ids)
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            cur = conn.execute(
                f"""
                UPDATE pool_events
                SET dormant_nights = dormant_nights + ?
                WHERE event_id IN ({placeholders})
                """,
                (delta, *ids),
            )
            return cur.rowcount

    def compact_chain(
        self,
        event_id: str,
        checkpoint_summary: str,
        *,
        keep_last: int = 2,
    ) -> PoolEvent:
        """Compress old step texts into checkpoint_summary while preserving 100% of spans."""
        if not isinstance(checkpoint_summary, str) or not checkpoint_summary.strip():
            raise ValueError("checkpoint_summary must be a non-empty string")
        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 0:
            raise ValueError("keep_last must be a non-negative int")

        with self._db.transaction() as conn:
            existing = self._fetch_event_in_conn(conn, event_id)
            kept_steps = existing.steps[-keep_last:] if keep_last > 0 else ()
            updated = PoolEvent(
                event_id=existing.event_id,
                title=existing.title,
                scope=existing.scope,
                status=existing.status,
                steps=kept_steps,
                checkpoint_summary=checkpoint_summary.strip(),
                spans=existing.spans,  # 100% lossless span preservation
                hit_count=existing.hit_count,
                created_at=existing.created_at,
                last_touched_at=existing.last_touched_at,
                dormant_nights=existing.dormant_nights,
            )
            self._save_event(conn, updated)
            return updated

    def plan_midnight_triage(
        self,
        now_iso: str | None = None,
        *,
        max_steps: int = 5,
        max_dormant_nights: int = 7,
    ) -> dict[str, list[PoolEvent]]:
        """Plan midnight pool triage without destructive mutation.

        Returns buckets:
            - to_archive_done: completed/aborted quests ready for long-term archival
            - to_compact: surviving hot events whose step chain exceeds max_steps
            - to_hibernate: dormant quests or over-capacity lower-score quests
            - to_drop_micro: expired, finished, or over-capacity micro tasks
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if max_dormant_nights < 1:
            raise ValueError("max_dormant_nights must be >= 1")

        ts = now_iso if now_iso is not None else _utc_now_iso()
        events = self.list_events(statuses=POOL_VISIBLE_STATUSES)

        to_archive_done: list[PoolEvent] = []
        to_compact: list[PoolEvent] = []
        to_hibernate: list[PoolEvent] = []
        to_drop_micro: list[PoolEvent] = []
        surviving: list[tuple[float, PoolEvent]] = []

        for ev in events:
            dt_hours = _elapsed_hours(ev.last_touched_at, ts)
            effective_dormant = max(ev.dormant_nights, int(dt_hours // 24.0))
            score = compute_retention_score(ev, ts)

            if ev.scope == "micro":
                if (
                    ev.status in ("done", "aborted")
                    or dt_hours >= 24.0
                    or effective_dormant >= 2
                ):
                    to_drop_micro.append(ev)
                    continue
                surviving.append((score, ev))
                continue

            if ev.status in ("done", "aborted"):
                to_archive_done.append(ev)
                continue

            if effective_dormant >= max_dormant_nights:
                to_hibernate.append(ev)
                continue

            surviving.append((score, ev))

        # Enforce pool capacity ceiling by evicting lowest retention-score tail events.
        surviving.sort(key=lambda item: (item[0], item[1].last_touched_at), reverse=True)
        kept = surviving[: self._max_events]
        overflow = surviving[self._max_events :]

        for _, ev in overflow:
            if ev.scope == "micro":
                to_drop_micro.append(ev)
            else:
                to_hibernate.append(ev)

        for _, ev in kept:
            if len(ev.steps) > max_steps:
                to_compact.append(ev)

        to_archive_done.sort(key=lambda ev: ev.last_touched_at)
        to_compact.sort(key=lambda ev: len(ev.steps), reverse=True)
        to_hibernate.sort(key=lambda ev: compute_retention_score(ev, ts))
        to_drop_micro.sort(key=lambda ev: compute_retention_score(ev, ts))

        return {
            "to_archive_done": to_archive_done,
            "to_compact": to_compact,
            "to_hibernate": to_hibernate,
            "to_drop_micro": to_drop_micro,
        }

    def render_snapshot(
        self,
        *,
        now_iso: str | None = None,
        include_undigested: bool = True,
    ) -> str:
        """Render a token-budgeted working-memory snapshot with undigested tail fallback."""
        ts = now_iso if now_iso is not None else _utc_now_iso()
        events = self.list_events(statuses=POOL_VISIBLE_STATUSES)
        scored = sorted(
            ((compute_retention_score(ev, ts), ev) for ev in events),
            key=lambda item: (item[0], item[1].last_touched_at),
            reverse=True,
        )[: self._max_events]

        undigested = (
            self._db.get_undigested_turns(watermark_key="pool_digested", limit=12)
            if include_undigested
            else []
        )

        header = "[ActivePool]"
        budget = self._max_tokens
        header_tokens = estimate_tokens(header)
        remaining = max(0, budget - header_tokens)

        # Reserve guaranteed floor budget for undigested tail if present.
        tail_reserve = 0
        if undigested and remaining > 12:
            tail_reserve = min(max(24, budget // 4), remaining // 2)

        events_budget = max(0, remaining - tail_reserve)
        rendered_events: list[str] = []
        used_event_tokens = 0

        for score, ev in scored:
            block = _render_event_block(ev, score)
            block_tokens = estimate_tokens(block) + 1  # newline separator
            if used_event_tokens + block_tokens <= events_budget:
                rendered_events.append(block)
                used_event_tokens += block_tokens
            elif not rendered_events and events_budget >= 8:
                clipped = _truncate_to_tokens(block, events_budget)
                if clipped:
                    rendered_events.append(clipped)
                    used_event_tokens += estimate_tokens(clipped) + 1
                break
            else:
                break

        sections: list[str] = [header]
        if rendered_events:
            sections.append("\n".join(rendered_events))
        elif not undigested:
            empty_line = "(empty)"
            if estimate_tokens(header + "\n" + empty_line) <= budget:
                sections.append(empty_line)

        if undigested:
            current_text = "\n".join(sections)
            tail_budget = budget - estimate_tokens(current_text) - 1
            tail_block = self._render_undigested_tail(undigested, tail_budget)
            if tail_block:
                sections.append(tail_block)

        snapshot = "\n".join(sections)
        if estimate_tokens(snapshot) > budget:
            snapshot = _truncate_to_tokens(snapshot, budget)
        return snapshot

    @staticmethod
    def _render_undigested_tail(
        turns: list[dict[str, object]], max_tokens: int
    ) -> str:
        tail_header = "[Undigested Tail]"
        header_cost = estimate_tokens(tail_header)
        if max_tokens <= header_cost + 2:
            return ""

        avail = max_tokens - header_cost
        # Select most recent turns that fit, then present in chronological order.
        chosen_reversed: list[str] = []
        used = 0
        for turn in reversed(turns):
            turn_id = turn["turn_id"]
            episode_id = turn["episode_id"]
            role = turn["role"]
            content = " ".join(str(turn["content"]).split())
            prefix = f"- #{turn_id} ({episode_id}/{role}): "
            line = prefix + content
            line_cost = estimate_tokens(line) + 1
            if used + line_cost <= avail:
                chosen_reversed.append(line)
                used += line_cost
            else:
                leftover = avail - used - 1
                if leftover > estimate_tokens(prefix) + 3:
                    clipped_line = _truncate_to_tokens(line, leftover)
                    if clipped_line:
                        chosen_reversed.append(clipped_line)
                break

        if not chosen_reversed:
            return ""
        chosen_reversed.reverse()
        return "\n".join([tail_header, *chosen_reversed])
