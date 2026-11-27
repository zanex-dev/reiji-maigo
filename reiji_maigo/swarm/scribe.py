"""Parallel shadow Scribe worker (ScribeAgent) and non-blocking debounced queue (AsyncScribeWorker).

ScribeAgent digests newly appended conversation turns inside a stateless Turn-1
Flash shadow window with a byte-locked L0 system prefix:
  - Filters out pure banter, greetings, memes, and uncommitted brainstorming
    (A-MAC / MemReader gate) while still advancing the ``pool_digested`` watermark.
  - Resolves ambiguous pronouns into concrete entities before mutating ActivePool.
  - Creates or appends linear progress steps on ActivePool events and stages
    structured Level-2 drawer details in ``pool_event_notes``.
  - Strictly clamps model-emitted ``SourceSpan`` coordinates to the real
    ``(episode_id, turn_id)`` bounds of the current batch before persistence.

AsyncScribeWorker runs a single daemon thread backed by a Condition-driven
debounced batch queue so ``notify(episode_id, turn_id)`` returns immediately
(<0.1 ms) with zero race conditions across concurrent callers.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reiji_maigo import SourceSpan
from reiji_maigo.pool.active_pool import VALID_SCOPES, VALID_STATUSES
from reiji_maigo.swarm.guards import ScribeCapabilities
from reiji_maigo.swarm.runner import SwarmCallResult, SwarmModelRunner

DEFAULT_WATERMARK_KEY = "pool_digested"

# Byte-locked immutable L0 system prefix for KV-cache prefix reuse.
# Never interpolate timestamps, turn counts, or dynamic state into this constant.
SCRIBE_L0_PREFIX: str = (
    "You are Reiji-Maigo Scribe, a stateless shadow memory clerk operating on a "
    "working-memory ActivePool and its Level-2 staging drawer.\n"
    "Your ONLY task is to inspect the provided ActivePool snapshot and the batch of "
    "undigested conversation turns, then return a SINGLE valid JSON object.\n"
    "\n"
    "MANDATORY GATES (A-MAC / MemReader discipline):\n"
    "1. Banter & Noise Gate: Pure greetings, small talk, jokes, memes, filler "
    "acknowledgements, emotional exclamations, or uncommitted brainstorms with no "
    "concrete decision, progress, or durable fact MUST NOT create or mutate any "
    "ActivePool event. Count such filtered turns in integer field `skipped_banter` "
    "and emit no event operation for them.\n"
    "2. Pronoun Resolution Gate: Never leave unresolved pronouns or vague deictics "
    "(\"it\", \"that\", \"this\", \"the script\", \"he/she\", \"它\", \"那个\") in "
    "any title, step, or detail note. Resolve every reference to its explicit concrete "
    "entity name using the ActivePool snapshot and batch turns.\n"
    "3. Continuation First: When turns advance an existing ActivePool event, reuse "
    "its exact `event_id` with `append_step`, `update_status`, or `touch_event` "
    "rather than opening a duplicate event.\n"
    "4. Grounded Spans Only: Every operation's `span` MUST use an exact `episode_id` "
    "and integer `turn_start`..`turn_end` present in the provided undigested batch. "
    "Never invent or extrapolate turn numbers.\n"
    "5. Level-2 Drawer Detail: Whenever an event carries structured parameters, file "
    "paths, decisions, constraints, metrics, or open questions that would bloat a "
    "one-line step, record them in the operation's `detail` object.\n"
    "\n"
    "OUTPUT SCHEMA (strict JSON object, no markdown commentary required):\n"
    "{\n"
    '  "skipped_banter": <non-negative int>,\n'
    '  "operations": [\n'
    "    {\n"
    '      "op": "create_event" | "append_step" | "update_status" | "touch_event",\n'
    '      "event_id": "<existing ev_... id; required except for create_event>",\n'
    '      "title": "<concise entity-resolved title; required for create_event>",\n'
    '      "scope": "quest" | "micro",\n'
    '      "step": "<single concise factual step with resolved entities>",\n'
    '      "status": "active" | "blocked" | "done" | "aborted" | "hibernated" | "dropped",\n'
    '      "span": {"episode_id": "<str>", "turn_start": <int>, "turn_end": <int>},\n'
    '      "detail": {\n'
    '        "summary": "<concise Level-2 drawer summary>",\n'
    '        "facts": ["<concrete durable fact>"],\n'
    '        "entities": ["<resolved entity>"],\n'
    '        "open_questions": ["<unresolved question>"]\n'
    "      }\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "If all turns in the batch are pure banter/noise, return "
    '{"skipped_banter": <count>, "operations": []}.'
)

_VALID_SCOPE_SET = frozenset(VALID_SCOPES)
_VALID_STATUS_SET = frozenset(VALID_STATUSES)
_DEFAULT_SCOPE = "quest" if "quest" in _VALID_SCOPE_SET else next(iter(VALID_SCOPES))
_STATUS_ALIASES: dict[str, str] = {
    "resolved": "done",
    "completed": "done",
    "finished": "done",
    "cancelled": "aborted",
    "canceled": "aborted",
    "paused": "hibernated",
    "stalled": "blocked",
}


def _normalize_status(raw_status: Any) -> str | None:
    if not isinstance(raw_status, str):
        return None
    cleaned = raw_status.strip().lower()
    if not cleaned:
        return None
    mapped = _STATUS_ALIASES.get(cleaned, cleaned)
    return mapped if mapped in _VALID_STATUS_SET else None


@dataclass(frozen=True, slots=True)
class ScribeDigestReport:
    """Immutable report describing a single ScribeAgent digestion cycle."""

    turns_digested: int
    watermark_after: int
    created_ids: tuple[str, ...]
    updated_ids: tuple[str, ...]
    skipped_banter: int
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class _BatchBounds:
    """Real (episode_id, turn_id) boundaries extracted from a turn batch."""

    max_turn_id: int
    ordered_episodes: tuple[str, ...]
    episode_ranges: dict[str, tuple[int, int]]
    turn_to_episode: dict[int, str]


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
            return int(stripped)
    return None


def _extract_turn_id(turn: Mapping[str, Any]) -> int:
    for key in ("turn_id", "id", "seq"):
        parsed = _coerce_int(turn.get(key))
        if parsed is not None:
            return parsed
    raise ValueError(f"Undigested turn missing integer turn_id: {turn!r}")


def _extract_episode_id(turn: Mapping[str, Any]) -> str:
    raw = turn.get("episode_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "ep_default"


def _compute_batch_bounds(turns: Sequence[Mapping[str, Any]]) -> _BatchBounds:
    ordered_episodes: list[str] = []
    seen_episodes: set[str] = set()
    episode_turns: dict[str, list[int]] = {}
    turn_to_episode: dict[int, str] = {}
    max_turn_id = 0

    for turn in turns:
        tid = _extract_turn_id(turn)
        ep = _extract_episode_id(turn)
        if tid > max_turn_id:
            max_turn_id = tid
        if ep not in seen_episodes:
            seen_episodes.add(ep)
            ordered_episodes.append(ep)
        episode_turns.setdefault(ep, []).append(tid)
        turn_to_episode[tid] = ep

    episode_ranges = {
        ep: (min(ids), max(ids))
        for ep, ids in episode_turns.items()
    }
    return _BatchBounds(
        max_turn_id=max_turn_id,
        ordered_episodes=tuple(ordered_episodes),
        episode_ranges=episode_ranges,
        turn_to_episode=turn_to_episode,
    )


def _clamp_span(raw_span: Any, bounds: _BatchBounds) -> tuple[SourceSpan, bool]:
    """Clamp a model-emitted span strictly within the current batch's real bounds.

    Returns ``(span, was_clamped)`` where ``was_clamped`` is True whenever the
    model output was missing, malformed, or fell outside the real batch range.
    """
    was_clamped = False
    raw_ep: str | None = None
    raw_start: int | None = None
    raw_end: int | None = None

    if isinstance(raw_span, SourceSpan):
        raw_ep = raw_span.episode_id
        raw_start = raw_span.turn_start
        raw_end = raw_span.turn_end
    elif isinstance(raw_span, Mapping):
        ep_val = raw_span.get("episode_id")
        if isinstance(ep_val, str) and ep_val.strip():
            raw_ep = ep_val.strip()
        raw_start = _coerce_int(raw_span.get("turn_start"))
        if raw_start is None:
            raw_start = _coerce_int(raw_span.get("start_turn"))
        raw_end = _coerce_int(raw_span.get("turn_end"))
        if raw_end is None:
            raw_end = _coerce_int(raw_span.get("end_turn"))
        if raw_start is None and raw_end is None:
            single_tid = _coerce_int(raw_span.get("turn_id"))
            if single_tid is not None:
                raw_start = single_tid
                raw_end = single_tid
            else:
                turn_ids = raw_span.get("turn_ids")
                if isinstance(turn_ids, Sequence) and not isinstance(turn_ids, (str, bytes)):
                    parsed_ids = [t for t in (_coerce_int(x) for x in turn_ids) if t is not None]
                    if parsed_ids:
                        raw_start = min(parsed_ids)
                        raw_end = max(parsed_ids)
    else:
        was_clamped = True

    # Resolve episode_id strictly against real batch episodes
    if raw_ep in bounds.episode_ranges:
        resolved_ep = raw_ep
    else:
        was_clamped = True
        resolved_ep = None
        for candidate_tid in (raw_start, raw_end):
            if candidate_tid is not None and candidate_tid in bounds.turn_to_episode:
                resolved_ep = bounds.turn_to_episode[candidate_tid]
                break
        if resolved_ep is None:
            resolved_ep = bounds.ordered_episodes[-1]

    lo, hi = bounds.episode_ranges[resolved_ep]

    start = raw_start
    end = raw_end
    if start is None and end is None:
        was_clamped = True
        start, end = lo, hi
    elif start is None:
        was_clamped = True
        start = end
    elif end is None:
        was_clamped = True
        end = start

    assert start is not None and end is not None
    if start > end:
        was_clamped = True
        start, end = end, start

    clamped_start = max(lo, min(hi, start))
    clamped_end = max(lo, min(hi, end))
    if clamped_start != start or clamped_end != end:
        was_clamped = True
    if clamped_start > clamped_end:
        was_clamped = True
        clamped_start, clamped_end = clamped_end, clamped_start

    return (
        SourceSpan(
            episode_id=resolved_ep,
            turn_start=clamped_start,
            turn_end=clamped_end,
        ),
        was_clamped,
    )


def _normalize_detail_payload(raw_detail: Any) -> dict[str, Any] | None:
    if raw_detail is None:
        return None
    if isinstance(raw_detail, Mapping):
        cleaned = {str(k): v for k, v in raw_detail.items() if v is not None}
        return cleaned or None
    if isinstance(raw_detail, str):
        text = raw_detail.strip()
        return {"summary": text} if text else None
    if isinstance(raw_detail, Sequence) and not isinstance(raw_detail, (bytes, bytearray)):
        items = [x for x in raw_detail if x is not None and str(x).strip()]
        return {"facts": items} if items else None
    return None


def _merge_event_detail(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Incrementally merge Level-2 drawer detail notes without losing prior facts."""
    if not existing:
        return dict(incoming)
    merged: dict[str, Any] = dict(existing)
    for key, val in incoming.items():
        prev = merged.get(key)
        if isinstance(prev, list) and isinstance(val, list):
            combined = list(prev)
            for item in val:
                if item not in combined:
                    combined.append(item)
            merged[key] = combined
        elif isinstance(prev, Mapping) and isinstance(val, Mapping):
            sub = dict(prev)
            sub.update(val)
            merged[key] = sub
        elif isinstance(val, str):
            if val.strip():
                merged[key] = val.strip()
        else:
            merged[key] = val
    return merged


def _format_user_payload(
    *,
    pool_snapshot: str,
    turns: Sequence[Mapping[str, Any]],
    bounds: _BatchBounds,
) -> str:
    compact_turns = []
    for t in turns:
        entry: dict[str, Any] = {
            "episode_id": _extract_episode_id(t),
            "turn_id": _extract_turn_id(t),
            "role": str(t.get("role", "user")),
            "content": str(t.get("content", "")),
        }
        if t.get("created_at"):
            entry["created_at"] = str(t["created_at"])
        compact_turns.append(entry)

    payload = {
        "active_pool_snapshot": pool_snapshot,
        "batch_bounds": {
            ep: {"min_turn": rng[0], "max_turn": rng[1]}
            for ep, rng in bounds.episode_ranges.items()
        },
        "undigested_turns": compact_turns,
    }
    return json.dumps(payload, ensure_ascii=False)


def _append_unique(target: list[str], item: str) -> None:
    if item not in target:
        target.append(item)


class ScribeAgent:
    """Stateless Turn-1 Flash shadow worker that maintains ActivePool & Level-2 notes."""

    def __init__(
        self,
        caps: ScribeCapabilities,
        runner: SwarmModelRunner,
        *,
        batch_limit: int = 12,
        watermark_key: str = DEFAULT_WATERMARK_KEY,
    ) -> None:
        if caps is None:
            raise TypeError("caps must be a ScribeCapabilities instance")
        if runner is None or not hasattr(runner, "run_shadow_json"):
            raise TypeError("runner must implement SwarmModelRunner.run_shadow_json")
        if isinstance(batch_limit, bool) or not isinstance(batch_limit, int) or batch_limit < 1:
            raise ValueError("batch_limit must be an integer >= 1")
        if not isinstance(watermark_key, str) or not watermark_key.strip():
            raise ValueError("watermark_key must be a non-empty string")

        self._caps = caps
        self._runner = runner
        self._batch_limit = batch_limit
        self._watermark_key = watermark_key.strip()
        self._digest_lock = threading.Lock()

    @property
    def batch_limit(self) -> int:
        return self._batch_limit

    @property
    def watermark_key(self) -> str:
        return self._watermark_key

    def digest_once(self, *, now_iso: str | None = None) -> ScribeDigestReport:
        """Digest up to ``batch_limit`` undigested turns in one Turn-1 shadow call.

        Zero model calls are made when there are no undigested turns.
        """
        started = time.monotonic()
        with self._digest_lock:
            wm_before = int(self._caps.get_watermark(self._watermark_key))
            turns = self._caps.get_undigested_turns(
                watermark_key=self._watermark_key,
                limit=self._batch_limit,
            )
            if not turns:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                return ScribeDigestReport(
                    turns_digested=0,
                    watermark_after=wm_before,
                    created_ids=(),
                    updated_ids=(),
                    skipped_banter=0,
                    elapsed_ms=elapsed_ms,
                )

            bounds = _compute_batch_bounds(turns)
            pool_snap = self._caps.pool_snapshot(
                now_iso=now_iso,
                include_undigested=False,
            )
            user_payload = _format_user_payload(
                pool_snapshot=pool_snap,
                turns=turns,
                bounds=bounds,
            )

            call_result: SwarmCallResult = self._runner.run_shadow_json(
                role="scribe",
                locked_system_prefix=SCRIBE_L0_PREFIX,
                user_payload=user_payload,
            )

            created_ids, updated_ids, skipped_banter, clamped_spans = (
                self._apply_operations(
                    call_result.data,
                    bounds=bounds,
                    now_iso=now_iso,
                )
            )

            wm_after = max(wm_before, bounds.max_turn_id)
            self._caps.set_watermark(self._watermark_key, wm_after)

            elapsed_ms = int((time.monotonic() - started) * 1000)
            self._caps.record_telemetry(
                "digest_once",
                {
                    "watermark_key": self._watermark_key,
                    "turns_digested": len(turns),
                    "watermark_before": wm_before,
                    "watermark_after": wm_after,
                    "created_count": len(created_ids),
                    "updated_count": len(updated_ids),
                    "skipped_banter": skipped_banter,
                    "clamped_spans": clamped_spans,
                    "prompt_tokens": call_result.prompt_tokens,
                    "completion_tokens": call_result.completion_tokens,
                    "model_elapsed_ms": call_result.elapsed_ms,
                    "elapsed_ms": elapsed_ms,
                    "prefix_sha256": call_result.prefix_sha256,
                },
                created_at=now_iso,
            )

            return ScribeDigestReport(
                turns_digested=len(turns),
                watermark_after=wm_after,
                created_ids=tuple(created_ids),
                updated_ids=tuple(updated_ids),
                skipped_banter=skipped_banter,
                elapsed_ms=elapsed_ms,
            )

    def _event_exists(self, event_id: str) -> bool:
        try:
            ev = self._caps.get_event(event_id)
        except (KeyError, LookupError, ValueError):
            return False
        return ev is not None

    def _save_merged_detail(
        self,
        event_id: str,
        raw_detail: Any,
        *,
        now_iso: str | None,
    ) -> None:
        incoming = _normalize_detail_payload(raw_detail)
        if not incoming:
            return
        existing = self._caps.get_event_note(event_id)
        merged = _merge_event_detail(existing, incoming)
        self._caps.save_event_note(event_id, merged, updated_at=now_iso)

    def _apply_operations(
        self,
        data: Mapping[str, Any],
        *,
        bounds: _BatchBounds,
        now_iso: str | None,
    ) -> tuple[list[str], list[str], int, int]:
        created_ids: list[str] = []
        updated_ids: list[str] = []
        clamped_spans = 0

        raw_skipped = _coerce_int(data.get("skipped_banter"))
        if raw_skipped is not None and raw_skipped >= 0:
            skipped_banter = raw_skipped
        else:
            banter_turns = data.get("banter_turns")
            if isinstance(banter_turns, Sequence) and not isinstance(banter_turns, (str, bytes)):
                skipped_banter = len(banter_turns)
            else:
                skipped_banter = 0

        raw_ops = data.get("operations")
        if not isinstance(raw_ops, list):
            raw_ops = []

        for item in raw_ops:
            if not isinstance(item, Mapping):
                continue
            op = str(item.get("op") or item.get("action") or "").strip().lower()
            if not op:
                continue

            if op in ("skip_banter", "banter", "skip", "noop", "ignore"):
                skipped_banter += 1
                continue

            if op in ("create_event", "create", "open_event"):
                title = str(item.get("title") or "").strip()
                step = str(item.get("step") or "").strip()
                if not title and step:
                    title = step[:80]
                if not step and title:
                    step = title
                if not title or not step:
                    continue

                raw_scope = str(item.get("scope") or _DEFAULT_SCOPE).strip().lower()
                scope = raw_scope if raw_scope in _VALID_SCOPE_SET else _DEFAULT_SCOPE
                span, clamped = _clamp_span(item.get("span"), bounds)
                if clamped:
                    clamped_spans += 1

                ev = self._caps.create_event(
                    title,
                    step,
                    span,
                    scope=scope,
                    now_iso=now_iso,
                )
                event_id = str(ev.event_id)
                _append_unique(created_ids, event_id)

                st = _normalize_status(item.get("status"))
                if st is not None and st != "active":
                    self._caps.update_status(event_id, st, now_iso=now_iso)

                self._save_merged_detail(event_id, item.get("detail"), now_iso=now_iso)
                continue

            if op in ("append_step", "append", "update_event"):
                event_id = str(item.get("event_id") or "").strip()
                step = str(item.get("step") or "").strip()
                if not step:
                    continue

                span, clamped = _clamp_span(item.get("span"), bounds)
                if clamped:
                    clamped_spans += 1

                valid_status = _normalize_status(item.get("status"))

                if event_id and self._event_exists(event_id):
                    self._caps.append_step(
                        event_id,
                        step,
                        span,
                        status=valid_status,
                        now_iso=now_iso,
                    )
                    if event_id not in created_ids:
                        _append_unique(updated_ids, event_id)
                    self._save_merged_detail(event_id, item.get("detail"), now_iso=now_iso)
                else:
                    # Graceful fallback if model referenced a non-existent event_id
                    fallback_title = str(item.get("title") or step[:80]).strip()
                    raw_scope = str(item.get("scope") or _DEFAULT_SCOPE).strip().lower()
                    scope = raw_scope if raw_scope in _VALID_SCOPE_SET else _DEFAULT_SCOPE
                    ev = self._caps.create_event(
                        fallback_title,
                        step,
                        span,
                        scope=scope,
                        now_iso=now_iso,
                    )
                    new_id = str(ev.event_id)
                    _append_unique(created_ids, new_id)
                    if valid_status is not None and valid_status != "active":
                        self._caps.update_status(new_id, valid_status, now_iso=now_iso)
                    self._save_merged_detail(new_id, item.get("detail"), now_iso=now_iso)
                continue

            if op in ("update_status", "status"):
                event_id = str(item.get("event_id") or "").strip()
                st = _normalize_status(item.get("status"))
                if not event_id or st is None or not self._event_exists(event_id):
                    continue
                self._caps.update_status(event_id, st, now_iso=now_iso)
                if event_id not in created_ids:
                    _append_unique(updated_ids, event_id)
                self._save_merged_detail(event_id, item.get("detail"), now_iso=now_iso)
                continue

            if op in ("touch_event", "touch"):
                event_id = str(item.get("event_id") or "").strip()
                if not event_id or not self._event_exists(event_id):
                    continue
                self._caps.touch_event(event_id, now_iso=now_iso)
                if event_id not in created_ids:
                    _append_unique(updated_ids, event_id)
                self._save_merged_detail(event_id, item.get("detail"), now_iso=now_iso)
                continue

        return created_ids, updated_ids, skipped_banter, clamped_spans


class AsyncScribeWorker:
    """Single daemon-thread debounced queue for non-blocking Scribe digestion.

    - ``notify(episode_id, turn_id)`` performs only an O(1) in-memory state update
      under a condition lock and returns immediately (<0.1 ms).
    - Coalesces burst notifications via ``min_turns`` and ``debounce_sec``.
    - ``flush(timeout)`` synchronously drains all queued/undigested turns in serial.
    - ``close()`` flushes (optionally) and cleanly joins the background daemon thread.
    """

    def __init__(
        self,
        scribe: ScribeAgent,
        *,
        min_turns: int = 1,
        debounce_sec: float = 0.05,
        max_wait_sec: float | None = None,
        on_report: Callable[[ScribeDigestReport], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        thread_name: str = "reiji-scribe-worker",
    ) -> None:
        if scribe is None or not hasattr(scribe, "digest_once"):
            raise TypeError("scribe must be a ScribeAgent instance")
        if isinstance(min_turns, bool) or not isinstance(min_turns, int) or min_turns < 1:
            raise ValueError("min_turns must be an integer >= 1")
        if isinstance(debounce_sec, bool) or not isinstance(debounce_sec, (int, float)) or debounce_sec < 0:
            raise ValueError("debounce_sec must be a non-negative number")
        if max_wait_sec is not None and (
            isinstance(max_wait_sec, bool)
            or not isinstance(max_wait_sec, (int, float))
            or max_wait_sec <= 0
        ):
            raise ValueError("max_wait_sec must be None or a positive number")

        self._scribe = scribe
        self._min_turns = min_turns
        self._debounce_sec = float(debounce_sec)
        self._max_wait_sec = float(max_wait_sec) if max_wait_sec is not None else None
        self._on_report = on_report
        self._on_error = on_error

        self._cond = threading.Condition()
        self._pending_count = 0
        self._last_episode_id: str | None = None
        self._last_turn_id: int = 0
        self._first_pending_at: float | None = None
        self._last_notify_at: float | None = None

        self._notify_seq = 0
        self._processed_seq = 0
        self._flush_request_gen = 0
        self._flush_done_gen = 0
        self._closed = False

        self._reports: list[ScribeDigestReport] = []
        self._last_error: Exception | None = None

        self._thread = threading.Thread(
            target=self._run_loop,
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    @property
    def reports(self) -> tuple[ScribeDigestReport, ...]:
        with self._cond:
            return tuple(self._reports)

    @property
    def last_error(self) -> Exception | None:
        with self._cond:
            return self._last_error

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def notify(self, episode_id: str, turn_id: int) -> bool:
        """Enqueue a lightweight turn notification without blocking on I/O or LLM."""
        now = time.monotonic()
        with self._cond:
            if self._closed:
                return False
            self._pending_count += 1
            self._notify_seq += 1
            self._last_episode_id = episode_id
            if turn_id > self._last_turn_id:
                self._last_turn_id = turn_id
            if self._first_pending_at is None:
                self._first_pending_at = now
            self._last_notify_at = now
            self._cond.notify_all()
            return True

    def flush(self, timeout: float | None = 5.0) -> bool:
        """Synchronously drain all pending and currently undigested turns.

        Returns True if drained before ``timeout``, or False if timed out.
        """
        deadline = None if timeout is None else (time.monotonic() + float(timeout))
        with self._cond:
            if self._closed and not self._thread.is_alive():
                return True
            self._flush_request_gen += 1
            target_gen = self._flush_request_gen
            target_seq = self._notify_seq
            self._cond.notify_all()

            while (
                self._flush_done_gen < target_gen
                or self._processed_seq < target_seq
            ):
                if not self._thread.is_alive():
                    return self._flush_done_gen >= target_gen and self._processed_seq >= target_seq
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()
            return True

    def close(self, *, flush: bool = True, timeout: float | None = 5.0) -> None:
        """Shut down the daemon worker cleanly, optionally draining pending work first."""
        if flush:
            self.flush(timeout=timeout)
        with self._cond:
            if not self._closed:
                self._closed = True
                self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def __enter__(self) -> AsyncScribeWorker:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _run_loop(self) -> None:
        while True:
            with self._cond:
                while True:
                    flush_pending = self._flush_request_gen > self._flush_done_gen
                    has_pending = self._pending_count > 0

                    if not flush_pending and not has_pending:
                        if self._closed:
                            return
                        self._cond.wait()
                        continue

                    if flush_pending or self._closed:
                        break

                    # Not an explicit flush: apply min_turns / debounce_sec / max_wait_sec
                    now = time.monotonic()
                    max_wait_expired = (
                        self._max_wait_sec is not None
                        and self._first_pending_at is not None
                        and (now - self._first_pending_at) >= self._max_wait_sec
                    )
                    if max_wait_expired:
                        break

                    if self._pending_count < self._min_turns:
                        if self._max_wait_sec is not None and self._first_pending_at is not None:
                            wait_s = max(0.0, self._max_wait_sec - (now - self._first_pending_at))
                            self._cond.wait(timeout=wait_s)
                        else:
                            self._cond.wait()
                        continue

                    if self._debounce_sec > 0.0 and self._last_notify_at is not None:
                        since_last = now - self._last_notify_at
                        if since_last < self._debounce_sec:
                            wait_s = self._debounce_sec - since_last
                            if self._max_wait_sec is not None and self._first_pending_at is not None:
                                max_rem = max(0.0, self._max_wait_sec - (now - self._first_pending_at))
                                wait_s = min(wait_s, max_rem)
                            self._cond.wait(timeout=wait_s)
                            continue
                    break

                batch_seq = self._notify_seq
                batch_flush_gen = self._flush_request_gen
                self._pending_count = 0
                self._first_pending_at = None
                self._last_notify_at = None

            # Execute digestion serially outside the condition lock so notify() stays <0.1ms
            batch_reports: list[ScribeDigestReport] = []
            batch_error: Exception | None = None
            try:
                prev_watermark: int | None = None
                while True:
                    report = self._scribe.digest_once()
                    if report.turns_digested == 0:
                        break
                    batch_reports.append(report)
                    if self._on_report is not None:
                        self._on_report(report)
                    if prev_watermark is not None and report.watermark_after <= prev_watermark:
                        # Guard against infinite loops if a mock/failing store stalls watermark
                        break
                    prev_watermark = report.watermark_after
            except Exception as exc:  # noqa: BLE001
                batch_error = exc
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except Exception:  # noqa: BLE001
                        pass

            with self._cond:
                if batch_reports:
                    self._reports.extend(batch_reports)
                self._last_error = batch_error
                if batch_seq > self._processed_seq:
                    self._processed_seq = batch_seq
                if batch_flush_gen > self._flush_done_gen:
                    self._flush_done_gen = batch_flush_gen
                self._cond.notify_all()
