"""Immutable core data contracts for reiji-maigo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LOCKED_DOMAINS: tuple[str, ...] = ("proj", "infra", "research", "companion", "life")
_LOCKED_DOMAIN_SET: frozenset[str] = frozenset(LOCKED_DOMAINS)


def validate_domain(domain: str) -> str:
    """Validate that a memory domain belongs to LOCKED_DOMAINS."""
    if not isinstance(domain, str) or domain not in _LOCKED_DOMAIN_SET:
        allowed = ", ".join(LOCKED_DOMAINS)
        raise ValueError(f"Invalid domain {domain!r}; expected one of: {allowed}")
    return domain


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Immutable provenance span pointing back to an episode turn range."""

    episode_id: str
    turn_start: int
    turn_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("SourceSpan.episode_id must be a non-empty string")
        if isinstance(self.turn_start, bool) or not isinstance(self.turn_start, int):
            raise TypeError("SourceSpan.turn_start must be an int")
        if isinstance(self.turn_end, bool) or not isinstance(self.turn_end, int):
            raise TypeError("SourceSpan.turn_end must be an int")
        if self.turn_start < 0:
            raise ValueError("SourceSpan.turn_start must be >= 0")
        if self.turn_end < self.turn_start:
            raise ValueError("SourceSpan.turn_end must be >= turn_start")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "turn_start": self.turn_start,
            "turn_end": self.turn_end,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceSpan:
        return cls(
            episode_id=str(data["episode_id"]),
            turn_start=int(data["turn_start"]),
            turn_end=int(data["turn_end"]),
        )


@dataclass(frozen=True, slots=True)
class PoolEvent:
    """Immutable event record stored in the active short-term memory pool."""

    event_id: str
    title: str
    scope: str
    status: str
    steps: tuple[str, ...]
    checkpoint_summary: str
    spans: tuple[SourceSpan, ...]
    hit_count: int
    created_at: str
    last_touched_at: str
    dormant_nights: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("PoolEvent.event_id must be a non-empty string")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("PoolEvent.title must be a non-empty string")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("PoolEvent.scope must be a non-empty string")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("PoolEvent.status must be a non-empty string")
        if not isinstance(self.steps, tuple) or not all(isinstance(s, str) for s in self.steps):
            raise TypeError("PoolEvent.steps must be a tuple[str, ...]")
        if not isinstance(self.spans, tuple) or not all(isinstance(s, SourceSpan) for s in self.spans):
            raise TypeError("PoolEvent.spans must be a tuple[SourceSpan, ...]")
        if isinstance(self.hit_count, bool) or not isinstance(self.hit_count, int) or self.hit_count < 0:
            raise ValueError("PoolEvent.hit_count must be a non-negative int")
        if isinstance(self.dormant_nights, bool) or not isinstance(self.dormant_nights, int) or self.dormant_nights < 0:
            raise ValueError("PoolEvent.dormant_nights must be a non-negative int")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "scope": self.scope,
            "status": self.status,
            "steps": list(self.steps),
            "checkpoint_summary": self.checkpoint_summary,
            "spans": [span.to_dict() for span in self.spans],
            "hit_count": self.hit_count,
            "created_at": self.created_at,
            "last_touched_at": self.last_touched_at,
            "dormant_nights": self.dormant_nights,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PoolEvent:
        raw_spans = data.get("spans", ())
        spans = tuple(
            s if isinstance(s, SourceSpan) else SourceSpan.from_dict(s)
            for s in raw_spans
        )
        return cls(
            event_id=str(data["event_id"]),
            title=str(data["title"]),
            scope=str(data["scope"]),
            status=str(data["status"]),
            steps=tuple(str(step) for step in data.get("steps", ())),
            checkpoint_summary=str(data.get("checkpoint_summary", "")),
            spans=spans,
            hit_count=int(data.get("hit_count", 0)),
            created_at=str(data["created_at"]),
            last_touched_at=str(data["last_touched_at"]),
            dormant_nights=int(data.get("dormant_nights", 0)),
        )


@dataclass(frozen=True, slots=True)
class MemoryCard:
    """Immutable bitemporal long-term memory card bound to a locked domain."""

    card_id: str
    domain: str
    card_type: str
    title: str
    state: dict[str, Any]
    lifecycle_status: str
    valid_from: str
    valid_to: str | None
    superseded_by: str | None
    tags: tuple[str, ...]
    spans: tuple[SourceSpan, ...]
    foresight_note: str = ""
    foresight_until: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.card_id, str) or not self.card_id.strip():
            raise ValueError("MemoryCard.card_id must be a non-empty string")
        validate_domain(self.domain)
        if not isinstance(self.card_type, str) or not self.card_type.strip():
            raise ValueError("MemoryCard.card_type must be a non-empty string")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("MemoryCard.title must be a non-empty string")
        if not isinstance(self.state, dict):
            raise TypeError("MemoryCard.state must be a dict[str, Any]")
        if not isinstance(self.lifecycle_status, str) or not self.lifecycle_status.strip():
            raise ValueError("MemoryCard.lifecycle_status must be a non-empty string")
        if not isinstance(self.valid_from, str) or not self.valid_from.strip():
            raise ValueError("MemoryCard.valid_from must be a non-empty string")
        if not isinstance(self.tags, tuple) or not all(isinstance(t, str) for t in self.tags):
            raise TypeError("MemoryCard.tags must be a tuple[str, ...]")
        if not isinstance(self.spans, tuple) or not all(isinstance(s, SourceSpan) for s in self.spans):
            raise TypeError("MemoryCard.spans must be a tuple[SourceSpan, ...]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "domain": self.domain,
            "card_type": self.card_type,
            "title": self.title,
            "state": dict(self.state),
            "lifecycle_status": self.lifecycle_status,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "superseded_by": self.superseded_by,
            "tags": list(self.tags),
            "spans": [span.to_dict() for span in self.spans],
            "foresight_note": self.foresight_note,
            "foresight_until": self.foresight_until,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryCard:
        raw_spans = data.get("spans", ())
        spans = tuple(
            s if isinstance(s, SourceSpan) else SourceSpan.from_dict(s)
            for s in raw_spans
        )
        return cls(
            card_id=str(data["card_id"]),
            domain=str(data["domain"]),
            card_type=str(data["card_type"]),
            title=str(data["title"]),
            state=dict(data.get("state") or {}),
            lifecycle_status=str(data["lifecycle_status"]),
            valid_from=str(data["valid_from"]),
            valid_to=None if data.get("valid_to") is None else str(data["valid_to"]),
            superseded_by=None if data.get("superseded_by") is None else str(data["superseded_by"]),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            spans=spans,
            foresight_note=str(data.get("foresight_note", "")),
            foresight_until=None if data.get("foresight_until") is None else str(data["foresight_until"]),
        )
