"""Midnight Archivist (子夜馆长蜂) worker for Reiji-Maigo memory swarm.

Executes isolated single-turn Flash shadow invocations under a byte-locked L0
system prefix to perform midnight memory curation:
  1. Drop expired / completed / overflow ``micro`` events deterministically.
  2. Losslessly compact long-chain hot ``quest`` events (preserving 100% spans).
  3. Archive completed/aborted events and hibernate dormant/overflow quests into
     three-layer drawer ``MemoryCard`` entries, absorbing ``pool_event_notes``.
  4. Resolve same-domain active card conflicts via TEPA bitemporal
     ``supersede_card`` and ``revoke_card``.
  5. Distill Level-3 page-fault ``pending_backfills`` queue entries into formal
     ``MemoryCard`` entries and mark them ``done``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import time
from typing import Any

from reiji_maigo.core.contracts import (
    LOCKED_DOMAINS,
    MemoryCard,
    PoolEvent,
    SourceSpan,
    validate_domain,
)
from reiji_maigo.swarm.guards import ArchivistCapabilities
from reiji_maigo.swarm.runner import SwarmCallResult, SwarmModelRunner

_WHITESPACE_RE = re.compile(r"\s+")
_DEFAULT_FALLBACK_DOMAIN: str = LOCKED_DOMAINS[0]  # "proj"

ARCHIVIST_L0_PREFIX: str = (
    "You are Reiji-Maigo's Midnight Archivist (子夜馆长蜂), a stateless single-turn "
    "memory curator. Always respond with ONE valid JSON object and no extra prose.\n"
    "\n"
    "CORE RULES:\n"
    "1. Locked L1 domains: strictly choose one of [\"proj\", \"infra\", \"research\", \"companion\", \"life\"].\n"
    "2. Controlled L2 tags (anti-entropy): ALWAYS reuse existing canonical tags from "
    "`controlled_tags` (or map synonyms to their canonical_name) before inventing a new tag. "
    "Keep 1-5 concise lowercase tags per card.\n"
    "3. Three-layer drawer `state` schema (mandatory for archive/backfill):\n"
    "   {\"summary\": \"<1-2 sentence distilled truth>\", "
    "\"detail\": [\"<concrete fact/decision/file>\"] or {\"<key>\": \"<value>\"}, "
    "\"open_questions\": [\"<unresolved item>\"]}\n"
    "   Always preserve concrete details from `event_note` inside `state.detail`.\n"
    "4. TEPA bitemporal conflict resolution: inspect `existing_active_cards` in the same domain. "
    "If the new card replaces/updates an older active card's state, output "
    "\"supersede_card_id\": \"<old_card_id>\". If older active cards are invalidated or retracted, "
    "output \"revoke_card_ids\": [\"<old_card_id>\", ...]. Link strongly related non-conflicting "
    "cards via \"related_card_ids\": [\"<card_id>\", ...].\n"
    "\n"
    "TASK OUTPUT SCHEMAS (selected by input `task`):\n"
    "- task=\"compact\": {\"checkpoint_summary\": \"<lossless concise checkpoint covering older steps>\"}\n"
    "- task=\"archive\" | \"backfill\": {\"domain\": \"...\", \"card_type\": \"...\", \"title\": \"...\", "
    "\"tags\": [...], \"state\": {\"summary\": \"...\", \"detail\": [...] | {...}, \"open_questions\": [...]}, "
    "\"foresight_note\": \"\", \"foresight_until\": null, \"supersede_card_id\": null, "
    "\"revoke_card_ids\": [], \"related_card_ids\": [], \"skip\": false}\n"
    "- task=\"conflict_check\": {\"supersede_card_id\": null, \"revoke_card_ids\": [], \"related_card_ids\": []}"
)


@dataclass(frozen=True, slots=True)
class ArchivistReport:
    """Immutable summary of a midnight curation run."""

    dropped_micro: tuple[str, ...]
    compacted_events: tuple[str, ...]
    archived_cards: tuple[str, ...]
    hibernated_cards: tuple[str, ...]
    superseded_cards: tuple[str, ...]
    revoked_cards: tuple[str, ...]
    backfilled_cards: tuple[str, ...]
    elapsed_ms: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_tag_text(raw: str) -> str:
    return _WHITESPACE_RE.sub(" ", raw.strip()).lower()


def _validate_positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def _extract_str_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        cleaned = raw.strip()
        return [cleaned] if cleaned else []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
    return out


def _normalize_domain(raw_domain: Any, *, fallback: str = _DEFAULT_FALLBACK_DOMAIN) -> str:
    if isinstance(raw_domain, str):
        candidate = raw_domain.strip().lower()
        try:
            return validate_domain(candidate)
        except ValueError:
            pass
    return validate_domain(fallback)


def _group_tags_by_domain(
    tag_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {d: [] for d in LOCKED_DOMAINS}
    for row in tag_rows:
        domain = str(row.get("domain", "")).strip()
        if domain not in grouped:
            continue
        grouped[domain].append(
            {
                "canonical_name": str(row.get("canonical_name", "")),
                "aliases": list(row.get("aliases") or ()),
                "usage_count": int(row.get("usage_count", 0)),
            }
        )
    return grouped


def _sanitize_tags_for_domain(
    domain: str,
    raw_tags: Any,
    existing_tags: list[dict[str, Any]],
    *,
    fallback_tag: str,
) -> list[str]:
    """Resolve aliases within domain and guard against cross-domain canonical collisions."""
    alias_to_canonical: dict[str, str] = {}
    owner_by_canonical: dict[str, str] = {}

    for row in existing_tags:
        row_domain = str(row.get("domain", "")).strip()
        canonical = _normalize_tag_text(str(row.get("canonical_name", "")))
        if not canonical:
            continue
        owner_by_canonical[canonical] = row_domain
        if row_domain == domain:
            alias_to_canonical[canonical] = canonical
            for alias in row.get("aliases") or ():
                if isinstance(alias, str) and alias.strip():
                    alias_to_canonical[_normalize_tag_text(alias)] = canonical

    candidates = _extract_str_list(raw_tags)
    resolved: list[str] = []
    seen: set[str] = set()

    for raw in candidates:
        norm = _normalize_tag_text(raw)
        if not norm:
            continue
        if norm in alias_to_canonical:
            chosen = alias_to_canonical[norm]
        elif norm in owner_by_canonical and owner_by_canonical[norm] != domain:
            scoped = f"{domain}:{norm}"
            if owner_by_canonical.get(scoped, domain) != domain:
                continue
            chosen = scoped
        else:
            chosen = norm

        if chosen not in seen:
            seen.add(chosen)
            resolved.append(chosen)

    if not resolved:
        fb = _normalize_tag_text(fallback_tag)
        if fb in owner_by_canonical and owner_by_canonical[fb] != domain:
            fb = f"{domain}:{fb}"
        resolved.append(fb)

    return resolved


def _merge_list_unique(base: Iterable[Any], extra: Iterable[Any]) -> list[Any]:
    merged: list[Any] = []
    seen_repr: set[str] = set()
    for item in (*base, *extra):
        key = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
        if key not in seen_repr:
            seen_repr.add(key)
            merged.append(item)
    return merged


def _build_three_layer_state(
    llm_data: Mapping[str, Any],
    *,
    event_note: dict[str, Any] | None,
    fallback_summary: str,
    fallback_detail: list[str],
) -> dict[str, Any]:
    """Assemble mandatory three-layer drawer state while losslessly absorbing event_note."""
    raw_state = llm_data.get("state")
    state_map: Mapping[str, Any] = raw_state if isinstance(raw_state, Mapping) else {}

    raw_summary = state_map.get("summary") or llm_data.get("summary")
    summary = (
        raw_summary.strip()
        if isinstance(raw_summary, str) and raw_summary.strip()
        else fallback_summary.strip()
    )

    raw_detail = state_map.get("detail")
    if isinstance(raw_detail, str):
        llm_detail: list[Any] | dict[str, Any] | None = (
            [raw_detail.strip()] if raw_detail.strip() else None
        )
    elif isinstance(raw_detail, list):
        llm_detail = [item for item in raw_detail if item is not None and item != ""]
    elif isinstance(raw_detail, Mapping):
        llm_detail = dict(raw_detail)
    else:
        llm_detail = None

    detail: list[Any] | dict[str, Any]
    if event_note:
        note_copy = dict(event_note)
        if set(note_copy.keys()) == {"detail"} and isinstance(note_copy["detail"], list):
            note_list = [item for item in note_copy["detail"] if item is not None and item != ""]
            if isinstance(llm_detail, list):
                detail = _merge_list_unique(note_list, llm_detail)
            elif isinstance(llm_detail, dict):
                merged_dict = dict(llm_detail)
                existing_items = merged_dict.get("event_note_detail")
                if isinstance(existing_items, list):
                    merged_dict["event_note_detail"] = _merge_list_unique(existing_items, note_list)
                else:
                    merged_dict["event_note_detail"] = note_list
                detail = merged_dict
            else:
                detail = note_list
        elif set(note_copy.keys()) == {"detail"} and isinstance(note_copy["detail"], Mapping):
            note_inner = dict(note_copy["detail"])
            if isinstance(llm_detail, dict):
                merged_dict = dict(llm_detail)
                for k, v in note_inner.items():
                    if k in merged_dict and isinstance(merged_dict[k], list) and isinstance(v, list):
                        merged_dict[k] = _merge_list_unique(v, merged_dict[k])
                    else:
                        merged_dict[k] = v
                detail = merged_dict
            elif isinstance(llm_detail, list) and llm_detail:
                merged_dict = dict(note_inner)
                if "notes" not in merged_dict:
                    merged_dict["notes"] = llm_detail
                elif isinstance(merged_dict["notes"], list):
                    merged_dict["notes"] = _merge_list_unique(merged_dict["notes"], llm_detail)
                else:
                    merged_dict["archivist_notes"] = llm_detail
                detail = merged_dict
            else:
                detail = note_inner
        else:
            if isinstance(llm_detail, dict):
                merged_dict = dict(llm_detail)
                for k, v in note_copy.items():
                    if k in merged_dict and isinstance(merged_dict[k], list) and isinstance(v, list):
                        merged_dict[k] = _merge_list_unique(v, merged_dict[k])
                    else:
                        merged_dict[k] = v
                detail = merged_dict
            elif isinstance(llm_detail, list) and llm_detail:
                merged_dict = dict(note_copy)
                if "notes" not in merged_dict:
                    merged_dict["notes"] = llm_detail
                elif isinstance(merged_dict["notes"], list):
                    merged_dict["notes"] = _merge_list_unique(merged_dict["notes"], llm_detail)
                else:
                    merged_dict["archivist_notes"] = llm_detail
                detail = merged_dict
            else:
                detail = note_copy
    else:
        if isinstance(llm_detail, list) and llm_detail:
            detail = llm_detail
        elif isinstance(llm_detail, dict) and llm_detail:
            detail = llm_detail
        else:
            detail = list(fallback_detail)

    open_questions = _extract_str_list(
        state_map.get("open_questions", llm_data.get("open_questions"))
    )

    state: dict[str, Any] = {
        "summary": summary,
        "detail": detail,
        "open_questions": open_questions,
    }
    # Preserve extra structured fields from llm state without overwriting core drawer keys.
    for key, val in state_map.items():
        if key not in state:
            state[key] = val
    return state


def _serialize_card_brief(card: MemoryCard) -> dict[str, Any]:
    summary = card.state.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = json.dumps(card.state, ensure_ascii=False)[:240]
    return {
        "card_id": card.card_id,
        "domain": card.domain,
        "card_type": card.card_type,
        "title": card.title,
        "tags": list(card.tags),
        "summary": summary,
        "valid_from": card.valid_from,
    }


class ArchivistAgent:
    """Midnight sleep-distillation curator operating inside ``ArchivistCapabilities``."""

    def __init__(
        self,
        caps: ArchivistCapabilities,
        runner: SwarmModelRunner,
        *,
        max_steps: int = 5,
        max_dormant_nights: int = 7,
        backfill_limit: int = 5,
    ) -> None:
        if not isinstance(caps, ArchivistCapabilities):
            raise TypeError("caps must be an ArchivistCapabilities instance")
        if not isinstance(runner, SwarmModelRunner):
            raise TypeError("runner must satisfy the SwarmModelRunner protocol")
        self._caps = caps
        self._runner = runner
        self._max_steps = _validate_positive_int("max_steps", max_steps, minimum=1)
        self._max_dormant_nights = _validate_positive_int(
            "max_dormant_nights", max_dormant_nights, minimum=1
        )
        self._backfill_limit = _validate_positive_int(
            "backfill_limit", backfill_limit, minimum=0
        )

    def curate_midnight(self, *, now_iso: str | None = None) -> ArchivistReport:
        """Run a full midnight curation pass across ActivePool, CardStore, and pending_backfills."""
        t0 = time.monotonic()
        ts = _utc_now_iso() if now_iso is None else str(now_iso)

        triage = self._caps.plan_midnight_triage(
            now_iso=ts,
            max_steps=self._max_steps,
            max_dormant_nights=self._max_dormant_nights,
        )

        dropped_micro: list[str] = []
        compacted_events: list[str] = []
        archived_cards: list[str] = []
        hibernated_cards: list[str] = []
        superseded_cards: list[str] = []
        revoked_cards: list[str] = []
        backfilled_cards: list[str] = []

        superseded_seen: set[str] = set()
        revoked_seen: set[str] = set()

        metrics: dict[str, Any] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prefix_sha256": None,
        }

        # 1. Deterministic drop of expired / done / overflow micro tasks (0 tokens)
        for ev in triage.get("to_drop_micro", ()):
            self._caps.update_status(ev.event_id, "dropped", now_iso=ts)
            self._caps.pop_event_note(ev.event_id)
            dropped_micro.append(ev.event_id)

        # 2. Lossless long-chain compaction (shortens steps while keeping 100% spans)
        for ev in triage.get("to_compact", ()):
            self._compact_single_event(ev, metrics=metrics)
            compacted_events.append(ev.event_id)

        # 3. Archive completed/aborted events into three-layer MemoryCards + TEPA
        for ev in triage.get("to_archive_done", ()):
            card = self._archive_single_event(
                ev,
                hibernate=False,
                ts=ts,
                metrics=metrics,
                superseded_cards=superseded_cards,
                revoked_cards=revoked_cards,
                superseded_seen=superseded_seen,
                revoked_seen=revoked_seen,
            )
            archived_cards.append(card.card_id)

        # 4. Hibernate dormant / over-capacity quests into hibernated MemoryCards + TEPA
        for ev in triage.get("to_hibernate", ()):
            card = self._archive_single_event(
                ev,
                hibernate=True,
                ts=ts,
                metrics=metrics,
                superseded_cards=superseded_cards,
                revoked_cards=revoked_cards,
                superseded_seen=superseded_seen,
                revoked_seen=revoked_seen,
            )
            hibernated_cards.append(card.card_id)

        # 5. Distill Level-3 page-fault pending_backfills into formal MemoryCards
        if self._backfill_limit > 0:
            pending = self._caps.list_pending_backfills(status="pending")[: self._backfill_limit]
            for item in pending:
                created_id = self._process_single_backfill(
                    item,
                    ts=ts,
                    metrics=metrics,
                    superseded_cards=superseded_cards,
                    revoked_cards=revoked_cards,
                    superseded_seen=superseded_seen,
                    revoked_seen=revoked_seen,
                )
                if created_id is not None:
                    backfilled_cards.append(created_id)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self._caps.record_telemetry(
            "curate_midnight",
            {
                "dropped_micro": len(dropped_micro),
                "compacted_events": len(compacted_events),
                "archived_cards": len(archived_cards),
                "hibernated_cards": len(hibernated_cards),
                "superseded_cards": len(superseded_cards),
                "revoked_cards": len(revoked_cards),
                "backfilled_cards": len(backfilled_cards),
                "llm_calls": metrics["llm_calls"],
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["completion_tokens"],
                "prefix_sha256": metrics["prefix_sha256"],
                "elapsed_ms": elapsed_ms,
            },
            created_at=ts,
        )

        return ArchivistReport(
            dropped_micro=tuple(dropped_micro),
            compacted_events=tuple(compacted_events),
            archived_cards=tuple(archived_cards),
            hibernated_cards=tuple(hibernated_cards),
            superseded_cards=tuple(superseded_cards),
            revoked_cards=tuple(revoked_cards),
            backfilled_cards=tuple(backfilled_cards),
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Internal single-turn shadow helpers
    # ------------------------------------------------------------------

    def _call_shadow(
        self,
        payload: Mapping[str, Any],
        *,
        metrics: dict[str, Any],
    ) -> SwarmCallResult:
        result = self._runner.run_shadow_json(
            role="archivist",
            locked_system_prefix=ARCHIVIST_L0_PREFIX,
            user_payload=json.dumps(dict(payload), ensure_ascii=False),
        )
        metrics["llm_calls"] += 1
        metrics["prompt_tokens"] += result.prompt_tokens
        metrics["completion_tokens"] += result.completion_tokens
        metrics["prefix_sha256"] = result.prefix_sha256
        return result

    def _compact_single_event(
        self,
        ev: PoolEvent,
        *,
        metrics: dict[str, Any],
    ) -> PoolEvent:
        keep_last = min(2, max(1, self._max_steps))
        if keep_last >= len(ev.steps):
            keep_last = max(0, len(ev.steps) - 1)

        event_note = self._caps.get_event_note(ev.event_id)
        payload: dict[str, Any] = {
            "task": "compact",
            "event": ev.to_dict(),
            "keep_last": keep_last,
            "event_note": event_note,
        }
        res = self._call_shadow(payload, metrics=metrics)
        raw_summary = res.data.get("checkpoint_summary") or res.data.get("summary")
        if isinstance(raw_summary, str) and raw_summary.strip():
            checkpoint_summary = raw_summary.strip()
        else:
            older = ev.steps[: len(ev.steps) - keep_last] if keep_last > 0 else ev.steps
            parts = [p for p in (ev.checkpoint_summary.strip(), " -> ".join(older)) if p]
            checkpoint_summary = " | ".join(parts) if parts else ev.title

        return self._caps.compact_chain(
            ev.event_id,
            checkpoint_summary,
            keep_last=keep_last,
        )

    def _collect_candidate_active_cards(
        self,
        search_text: str,
        existing_tags: list[dict[str, Any]],
        *,
        top_k: int = 8,
    ) -> list[MemoryCard]:
        if not existing_tags:
            return []

        lowered_text = search_text.lower()
        matched_tags: list[str] = []
        all_canonical: list[str] = []

        for row in existing_tags:
            canonical = str(row.get("canonical_name", "")).strip()
            if not canonical:
                continue
            all_canonical.append(canonical)
            vocab = [canonical] + [
                str(a).strip() for a in (row.get("aliases") or ()) if isinstance(a, str) and a.strip()
            ]
            if any(term.lower() in lowered_text for term in vocab):
                matched_tags.append(canonical)

        candidates: list[MemoryCard] = []
        seen_ids: set[str] = set()

        if matched_tags:
            for card in self._caps.recall_level1(None, matched_tags, min_tag_hits=1, top_k=top_k):
                if card.card_id not in seen_ids:
                    seen_ids.add(card.card_id)
                    candidates.append(card)

        if len(candidates) < top_k and all_canonical:
            for card in self._caps.recall_level1(None, all_canonical, min_tag_hits=1, top_k=top_k):
                if card.card_id not in seen_ids:
                    seen_ids.add(card.card_id)
                    candidates.append(card)
                    if len(candidates) >= top_k:
                        break

        return candidates

    def _archive_single_event(
        self,
        ev: PoolEvent,
        *,
        hibernate: bool,
        ts: str,
        metrics: dict[str, Any],
        superseded_cards: list[str],
        revoked_cards: list[str],
        superseded_seen: set[str],
        revoked_seen: set[str],
    ) -> MemoryCard:
        existing_tags = self._caps.list_tags()
        event_note = self._caps.get_event_note(ev.event_id)

        search_blob = "\n".join(
            [
                ev.title,
                ev.checkpoint_summary,
                *ev.steps,
                json.dumps(event_note, ensure_ascii=False) if event_note else "",
            ]
        )
        initial_candidates = self._collect_candidate_active_cards(search_blob, existing_tags)
        initial_candidate_ids = {c.card_id for c in initial_candidates}

        payload: dict[str, Any] = {
            "task": "archive",
            "hibernate": hibernate,
            "event": ev.to_dict(),
            "event_note": event_note,
            "controlled_tags": _group_tags_by_domain(existing_tags),
            "existing_active_cards": [_serialize_card_brief(c) for c in initial_candidates],
        }
        res = self._call_shadow(payload, metrics=metrics)
        data = res.data

        domain = _normalize_domain(data.get("domain"))
        tags = _sanitize_tags_for_domain(
            domain,
            data.get("tags"),
            existing_tags,
            fallback_tag="hibernated" if hibernate else "archived",
        )
        raw_title = data.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else ev.title
        raw_card_type = data.get("card_type")
        card_type = (
            raw_card_type.strip()
            if isinstance(raw_card_type, str) and raw_card_type.strip()
            else None
        )
        foresight_note = (
            str(data.get("foresight_note", "")).strip()
            if data.get("foresight_note") is not None
            else ""
        )
        raw_until = data.get("foresight_until")
        foresight_until = (
            raw_until.strip() if isinstance(raw_until, str) and raw_until.strip() else None
        )

        fallback_summary = ev.checkpoint_summary.strip() or ev.title
        state = _build_three_layer_state(
            data,
            event_note=event_note,
            fallback_summary=fallback_summary,
            fallback_detail=list(ev.steps),
        )

        # Discover same-domain active cards matching final tags before inserting new_card
        post_candidates = self._caps.recall_level1(domain, tags, min_tag_hits=1, top_k=6)
        unseen_domain_candidates = [
            c for c in post_candidates if c.card_id not in initial_candidate_ids
        ]

        new_card = self._caps.archive_pool_event(
            ev.event_id,
            domain=domain,
            tags=tags,
            state=state,
            hibernate=hibernate,
            card_type=card_type,
            title=title,
            foresight_note=foresight_note,
            foresight_until=foresight_until,
            now_iso=ts,
        )

        # Consume staged note only after archival succeeds
        if event_note is not None:
            self._caps.pop_event_note(ev.event_id)

        self._apply_tepa_and_links(
            new_card,
            llm_data=data,
            unseen_domain_candidates=unseen_domain_candidates,
            ts=ts,
            metrics=metrics,
            superseded_cards=superseded_cards,
            revoked_cards=revoked_cards,
            superseded_seen=superseded_seen,
            revoked_seen=revoked_seen,
        )
        return new_card

    def _process_single_backfill(
        self,
        item: Mapping[str, Any],
        *,
        ts: str,
        metrics: dict[str, Any],
        superseded_cards: list[str],
        revoked_cards: list[str],
        superseded_seen: set[str],
        revoked_seen: set[str],
    ) -> str | None:
        backfill_id = int(item["backfill_id"])
        span = SourceSpan(
            episode_id=str(item["episode_id"]),
            turn_start=int(item["turn_start"]),
            turn_end=int(item["turn_end"]),
        )
        reason = str(item.get("reason", ""))
        turns = self._caps.read_span(span, pad=0)
        if not turns:
            self._caps.resolve_backfill(
                backfill_id,
                status="done",
                increment_attempts=True,
                now_iso=ts,
            )
            return None

        existing_tags = self._caps.list_tags()
        turn_texts = [
            f"[{t.get('turn_id')}] {t.get('role', '')}: {t.get('content', '')}"
            for t in turns
        ]
        search_blob = "\n".join([reason, *turn_texts])
        initial_candidates = self._collect_candidate_active_cards(search_blob, existing_tags)
        initial_candidate_ids = {c.card_id for c in initial_candidates}

        payload: dict[str, Any] = {
            "task": "backfill",
            "backfill_id": backfill_id,
            "reason": reason,
            "span": span.to_dict(),
            "turns": turns,
            "controlled_tags": _group_tags_by_domain(existing_tags),
            "existing_active_cards": [_serialize_card_brief(c) for c in initial_candidates],
        }
        res = self._call_shadow(payload, metrics=metrics)
        data = res.data

        if data.get("skip") is True:
            self._caps.resolve_backfill(
                backfill_id,
                status="done",
                increment_attempts=True,
                now_iso=ts,
            )
            return None

        domain = _normalize_domain(data.get("domain"))
        tags = _sanitize_tags_for_domain(
            domain,
            data.get("tags"),
            existing_tags,
            fallback_tag="backfill",
        )
        raw_title = data.get("title")
        title = (
            raw_title.strip()
            if isinstance(raw_title, str) and raw_title.strip()
            else f"Backfill {span.episode_id}:{span.turn_start}-{span.turn_end}"
        )
        raw_card_type = data.get("card_type")
        card_type = (
            raw_card_type.strip()
            if isinstance(raw_card_type, str) and raw_card_type.strip()
            else "fact"
        )
        foresight_note = (
            str(data.get("foresight_note", "")).strip()
            if data.get("foresight_note") is not None
            else ""
        )
        raw_until = data.get("foresight_until")
        foresight_until = (
            raw_until.strip() if isinstance(raw_until, str) and raw_until.strip() else None
        )

        fallback_summary = reason.strip() or title
        state = _build_three_layer_state(
            data,
            event_note=None,
            fallback_summary=fallback_summary,
            fallback_detail=turn_texts,
        )
        state.setdefault("backfill_id", backfill_id)
        if reason:
            state.setdefault("backfill_reason", reason)

        post_candidates = self._caps.recall_level1(domain, tags, min_tag_hits=1, top_k=6)
        unseen_domain_candidates = [
            c for c in post_candidates if c.card_id not in initial_candidate_ids
        ]

        new_card = self._caps.create_card(
            domain=domain,
            card_type=card_type,
            title=title,
            state=state,
            tags=tags,
            spans=(span,),
            foresight_note=foresight_note,
            foresight_until=foresight_until,
            valid_from=ts,
            created_at=ts,
            updated_at=ts,
        )

        self._apply_tepa_and_links(
            new_card,
            llm_data=data,
            unseen_domain_candidates=unseen_domain_candidates,
            ts=ts,
            metrics=metrics,
            superseded_cards=superseded_cards,
            revoked_cards=revoked_cards,
            superseded_seen=superseded_seen,
            revoked_seen=revoked_seen,
        )

        self._caps.resolve_backfill(
            backfill_id,
            status="done",
            increment_attempts=True,
            now_iso=ts,
        )
        return new_card.card_id

    def _apply_tepa_and_links(
        self,
        new_card: MemoryCard,
        *,
        llm_data: Mapping[str, Any],
        unseen_domain_candidates: list[MemoryCard],
        ts: str,
        metrics: dict[str, Any],
        superseded_cards: list[str],
        revoked_cards: list[str],
        superseded_seen: set[str],
        revoked_seen: set[str],
    ) -> None:
        supersede_ids: list[str] = []
        single_sup = llm_data.get("supersede_card_id")
        if isinstance(single_sup, str) and single_sup.strip():
            supersede_ids.append(single_sup.strip())
        for sid in _extract_str_list(llm_data.get("supersede_card_ids")):
            if sid not in supersede_ids:
                supersede_ids.append(sid)

        revoke_ids = _extract_str_list(llm_data.get("revoke_card_ids"))
        related_ids = _extract_str_list(llm_data.get("related_card_ids"))

        # If final domain/tags revealed active cards that weren't shown in the first pass,
        # run one focused conflict-check turn under the exact same locked L0 prefix.
        if unseen_domain_candidates and not supersede_ids and not revoke_ids:
            conflict_payload = {
                "task": "conflict_check",
                "new_card": _serialize_card_brief(new_card),
                "existing_active_cards": [
                    _serialize_card_brief(c)
                    for c in unseen_domain_candidates
                    if c.card_id != new_card.card_id
                ],
            }
            if conflict_payload["existing_active_cards"]:
                check_res = self._call_shadow(conflict_payload, metrics=metrics)
                check_data = check_res.data
                chk_sup = check_data.get("supersede_card_id")
                if isinstance(chk_sup, str) and chk_sup.strip():
                    supersede_ids.append(chk_sup.strip())
                for sid in _extract_str_list(check_data.get("supersede_card_ids")):
                    if sid not in supersede_ids:
                        supersede_ids.append(sid)
                for rid in _extract_str_list(check_data.get("revoke_card_ids")):
                    if rid not in revoke_ids:
                        revoke_ids.append(rid)
                for rel in _extract_str_list(check_data.get("related_card_ids")):
                    if rel not in related_ids:
                        related_ids.append(rel)

        # Execute bitemporal supersedes
        for old_id in supersede_ids:
            if old_id == new_card.card_id or old_id in superseded_seen:
                continue
            existing = self._caps.get_card(old_id)
            if existing is None or existing.lifecycle_status != "active":
                continue
            self._caps.supersede_card(old_id, new_card, now_iso=ts)
            superseded_seen.add(old_id)
            superseded_cards.append(old_id)

        # Execute TEPA revocations
        for old_id in revoke_ids:
            if (
                old_id == new_card.card_id
                or old_id in superseded_seen
                or old_id in revoked_seen
            ):
                continue
            existing = self._caps.get_card(old_id)
            if existing is None or existing.lifecycle_status != "active":
                continue
            if self._caps.revoke_card(old_id, now_iso=ts):
                revoked_seen.add(old_id)
                revoked_cards.append(old_id)

        # Link non-conflicting related cards
        for rel_id in related_ids:
            if (
                rel_id == new_card.card_id
                or rel_id in superseded_seen
                or rel_id in revoked_seen
            ):
                continue
            existing = self._caps.get_card(rel_id)
            if existing is None or existing.lifecycle_status != "active":
                continue
            self._caps.link_cards(new_card.card_id, rel_id, relation="related", weight=1.0)


__all__ = [
    "ARCHIVIST_L0_PREFIX",
    "ArchivistAgent",
    "ArchivistReport",
]
