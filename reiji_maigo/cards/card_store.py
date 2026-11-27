"""Bitemporal, revocable long-term memory card store with controlled L2 tags and PPR recall."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
import re
import sqlite3
from typing import Any
from uuid import uuid4

from reiji_maigo.cards.ppr import personalized_pagerank
from reiji_maigo.core.contracts import MemoryCard, SourceSpan, validate_domain
from reiji_maigo.storage.sqlite_engine import ReijiDatabase

VALID_LIFECYCLE_STATUSES: frozenset[str] = frozenset({"active", "superseded", "revoked"})
_SHARED_TAG_EDGE_WEIGHT: float = 0.25
_WHITESPACE_RE = re.compile(r"\s+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_tag(raw_tag: str) -> str:
    if not isinstance(raw_tag, str):
        raise TypeError("tag must be a string")
    normalized = _WHITESPACE_RE.sub(" ", raw_tag.strip()).lower()
    if not normalized:
        raise ValueError("tag must be a non-empty string")
    return normalized


def _coerce_spans(spans: Iterable[SourceSpan | Mapping[str, Any]]) -> tuple[SourceSpan, ...]:
    coerced: list[SourceSpan] = []
    for item in spans:
        if isinstance(item, SourceSpan):
            coerced.append(item)
        elif isinstance(item, Mapping):
            coerced.append(SourceSpan.from_dict(item))
        else:
            raise TypeError("each span must be a SourceSpan or mapping")
    if not coerced:
        raise ValueError("MemoryCard must bind at least one SourceSpan")
    return tuple(coerced)


class CardStore:
    """Back-row long-term memory store with locked L1 domains, controlled L2 tags,
    TEPA/Zep bitemporal lifecycle, PPR multi-hop recall, and 3-level evidence slicing.
    """

    def __init__(self, db: ReijiDatabase) -> None:
        if not isinstance(db, ReijiDatabase):
            raise TypeError("db must be a ReijiDatabase instance")
        self._db = db

    # ------------------------------------------------------------------
    # Controlled L2 tag vocabulary
    # ------------------------------------------------------------------

    def register_or_resolve_tag(
        self,
        domain: str,
        raw_tag: str,
        *,
        aliases: Iterable[str] = (),
    ) -> str:
        """Register or normalize a controlled L2 tag under a locked L1 domain.

        If ``raw_tag`` matches an existing canonical tag or any of its aliases within
        ``domain``, returns the existing canonical name (merging any new aliases).
        """
        validate_domain(domain)
        norm = _normalize_tag(raw_tag)
        norm_aliases = self._clean_aliases(aliases, exclude=norm)

        with self._db.transaction() as conn:
            _, canonical = self._register_or_resolve_tag_in_tx(
                conn,
                domain,
                norm,
                aliases=norm_aliases,
            )
            return canonical

    def list_tags(self, domain: str | None = None) -> list[dict[str, Any]]:
        """List registered controlled L2 tags, optionally filtered by domain."""
        if domain is not None:
            validate_domain(domain)

        with self._db.read_connection() as conn:
            if domain is None:
                rows = conn.execute(
                    """
                    SELECT tag_id, domain, canonical_name, aliases_json, usage_count, created_at, updated_at
                    FROM memory_tags
                    ORDER BY domain ASC, canonical_name ASC
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT tag_id, domain, canonical_name, aliases_json, usage_count, created_at, updated_at
                    FROM memory_tags
                    WHERE domain = ?
                    ORDER BY canonical_name ASC
                    """,
                    (domain,),
                ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "tag_id": int(row["tag_id"]),
                    "domain": str(row["domain"]),
                    "canonical_name": str(row["canonical_name"]),
                    "aliases": list(json.loads(row["aliases_json"] or "[]")),
                    "usage_count": int(row["usage_count"]),
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"]),
                }
            )
        return result

    @staticmethod
    def _clean_aliases(aliases: Iterable[str], *, exclude: str) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = {exclude}
        for alias in aliases:
            norm_alias = _normalize_tag(alias)
            if norm_alias not in seen:
                seen.add(norm_alias)
                cleaned.append(norm_alias)
        return cleaned

    def _register_or_resolve_tag_in_tx(
        self,
        conn: sqlite3.Connection,
        domain: str,
        norm_tag: str,
        *,
        aliases: Iterable[str] = (),
    ) -> tuple[int, str]:
        now = _utc_now_iso()
        rows = conn.execute(
            """
            SELECT tag_id, canonical_name, aliases_json
            FROM memory_tags
            WHERE domain = ?
            ORDER BY tag_id ASC
            """,
            (domain,),
        ).fetchall()

        # Pass 1: exact canonical match in target domain.
        matched_row: sqlite3.Row | None = None
        for row in rows:
            if str(row["canonical_name"]) == norm_tag:
                matched_row = row
                break

        # Pass 2: alias match in target domain.
        if matched_row is None:
            for row in rows:
                existing_aliases = [
                    _normalize_tag(a)
                    for a in json.loads(row["aliases_json"] or "[]")
                    if isinstance(a, str) and a.strip()
                ]
                if norm_tag in existing_aliases:
                    matched_row = row
                    break

        if matched_row is not None:
            tag_id = int(matched_row["tag_id"])
            canonical = str(matched_row["canonical_name"])
            existing_aliases = [
                _normalize_tag(a)
                for a in json.loads(matched_row["aliases_json"] or "[]")
                if isinstance(a, str) and a.strip()
            ]
            merged = list(existing_aliases)
            merged_set = set(merged) | {canonical}
            changed = False
            for alias in aliases:
                norm_alias = _normalize_tag(alias)
                if norm_alias not in merged_set:
                    merged.append(norm_alias)
                    merged_set.add(norm_alias)
                    changed = True
            if changed:
                conn.execute(
                    """
                    UPDATE memory_tags
                    SET aliases_json = ?, updated_at = ?
                    WHERE tag_id = ?
                    """,
                    (json.dumps(merged, ensure_ascii=False), now, tag_id),
                )
            return tag_id, canonical

        # Guard against cross-domain canonical collision (schema enforces UNIQUE canonical_name).
        collision = conn.execute(
            "SELECT domain FROM memory_tags WHERE canonical_name = ?",
            (norm_tag,),
        ).fetchone()
        if collision is not None:
            raise ValueError(
                f"Tag {norm_tag!r} is already registered under domain {collision['domain']!r}"
            )

        clean_aliases = self._clean_aliases(aliases, exclude=norm_tag)
        cur = conn.execute(
            """
            INSERT INTO memory_tags (
                domain, canonical_name, aliases_json, usage_count, created_at, updated_at
            )
            VALUES (?, ?, ?, 0, ?, ?)
            """,
            (domain, norm_tag, json.dumps(clean_aliases, ensure_ascii=False), now, now),
        )
        return int(cur.lastrowid), norm_tag

    def _resolve_query_tag_ids(
        self,
        conn: sqlite3.Connection,
        domain: str | None,
        query_tags: Iterable[str],
    ) -> list[int]:
        raw_norms: list[str] = []
        seen_raw: set[str] = set()
        for raw in query_tags:
            if not isinstance(raw, str) or not raw.strip():
                continue
            norm = _normalize_tag(raw)
            if norm not in seen_raw:
                seen_raw.add(norm)
                raw_norms.append(norm)

        if not raw_norms:
            return []

        if domain is None:
            rows = conn.execute(
                "SELECT tag_id, canonical_name, aliases_json FROM memory_tags"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT tag_id, canonical_name, aliases_json FROM memory_tags WHERE domain = ?",
                (domain,),
            ).fetchall()

        matched_ids: list[int] = []
        seen_ids: set[int] = set()
        for row in rows:
            tag_id = int(row["tag_id"])
            canonical = str(row["canonical_name"])
            aliases = {
                _normalize_tag(a)
                for a in json.loads(row["aliases_json"] or "[]")
                if isinstance(a, str) and a.strip()
            }
            vocab = aliases | {canonical}
            if any(q in vocab for q in raw_norms) and tag_id not in seen_ids:
                seen_ids.add(tag_id)
                matched_ids.append(tag_id)
        return matched_ids

    # ------------------------------------------------------------------
    # Card CRUD, bitemporal superseding, revocation, and graph links
    # ------------------------------------------------------------------

    def create_card(
        self,
        domain: str,
        card_type: str,
        title: str,
        state: dict[str, Any],
        tags: Iterable[str],
        spans: Iterable[SourceSpan],
        **kw: Any,
    ) -> MemoryCard:
        """Create and persist a bitemporal memory card with mandatory SourceSpan binding."""
        validate_domain(domain)
        if not isinstance(state, dict):
            raise TypeError("state must be a dict[str, Any]")

        allowed_kw = {
            "card_id",
            "lifecycle_status",
            "valid_from",
            "valid_to",
            "superseded_by",
            "foresight_note",
            "foresight_until",
            "created_at",
            "updated_at",
        }
        unknown = set(kw.keys()) - allowed_kw
        if unknown:
            unknown_str = ", ".join(sorted(unknown))
            raise TypeError(f"Unexpected keyword argument(s): {unknown_str}")

        now = _utc_now_iso()
        card_id = str(kw["card_id"]) if kw.get("card_id") is not None else f"card_{uuid4().hex[:12]}"
        lifecycle_status = str(kw.get("lifecycle_status", "active"))
        if lifecycle_status not in VALID_LIFECYCLE_STATUSES:
            raise ValueError(
                f"Invalid lifecycle_status {lifecycle_status!r}; expected one of {sorted(VALID_LIFECYCLE_STATUSES)}"
            )
        valid_from = str(kw["valid_from"]) if kw.get("valid_from") is not None else now
        valid_to = None if kw.get("valid_to") is None else str(kw["valid_to"])
        superseded_by = None if kw.get("superseded_by") is None else str(kw["superseded_by"])
        foresight_note = str(kw.get("foresight_note", ""))
        foresight_until = None if kw.get("foresight_until") is None else str(kw["foresight_until"])
        created_at = str(kw["created_at"]) if kw.get("created_at") is not None else now
        updated_at = str(kw["updated_at"]) if kw.get("updated_at") is not None else now

        span_tuple = _coerce_spans(spans)

        with self._db.transaction() as conn:
            resolved_tags = self._resolve_and_dedup_tags_in_tx(conn, domain, tags)
            canonical_names = tuple(name for _, name in resolved_tags)

            # Validate full contract before persistence.
            card = MemoryCard(
                card_id=card_id,
                domain=domain,
                card_type=card_type,
                title=title,
                state=dict(state),
                lifecycle_status=lifecycle_status,
                valid_from=valid_from,
                valid_to=valid_to,
                superseded_by=superseded_by,
                tags=canonical_names,
                spans=span_tuple,
                foresight_note=foresight_note,
                foresight_until=foresight_until,
            )

            self._insert_card_in_tx(
                conn,
                card,
                resolved_tags=resolved_tags,
                created_at=created_at,
                updated_at=updated_at,
            )
            return self._load_card_in_tx(conn, card.card_id)

    def _resolve_and_dedup_tags_in_tx(
        self,
        conn: sqlite3.Connection,
        domain: str,
        tags: Iterable[str],
    ) -> list[tuple[int, str]]:
        resolved: list[tuple[int, str]] = []
        seen_ids: set[int] = set()
        for raw_tag in tags:
            norm = _normalize_tag(raw_tag)
            tag_id, canonical = self._register_or_resolve_tag_in_tx(conn, domain, norm)
            if tag_id not in seen_ids:
                seen_ids.add(tag_id)
                resolved.append((tag_id, canonical))
        return resolved

    def _insert_card_in_tx(
        self,
        conn: sqlite3.Connection,
        card: MemoryCard,
        *,
        resolved_tags: list[tuple[int, str]],
        created_at: str,
        updated_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO memory_cards (
                card_id, domain, card_type, title, state_json,
                lifecycle_status, valid_from, valid_to, superseded_by,
                spans_json, foresight_note, foresight_until,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card.card_id,
                card.domain,
                card.card_type,
                card.title,
                json.dumps(card.state, ensure_ascii=False),
                card.lifecycle_status,
                card.valid_from,
                card.valid_to,
                card.superseded_by,
                json.dumps([span.to_dict() for span in card.spans], ensure_ascii=False),
                card.foresight_note,
                card.foresight_until,
                created_at,
                updated_at,
            ),
        )
        for tag_id, _ in resolved_tags:
            conn.execute(
                "INSERT OR IGNORE INTO card_tags (card_id, tag_id) VALUES (?, ?)",
                (card.card_id, tag_id),
            )
            conn.execute(
                "UPDATE memory_tags SET usage_count = usage_count + 1, updated_at = ? WHERE tag_id = ?",
                (updated_at, tag_id),
            )

    def get_card(self, card_id: str) -> MemoryCard | None:
        """Retrieve a card by ID regardless of lifecycle status (supports audit trail)."""
        if not isinstance(card_id, str) or not card_id.strip():
            raise ValueError("card_id must be a non-empty string")
        with self._db.read_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memory_cards WHERE card_id = ?",
                (card_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_card(conn, row)

    def _load_card_in_tx(self, conn: sqlite3.Connection, card_id: str) -> MemoryCard:
        row = conn.execute(
            "SELECT * FROM memory_cards WHERE card_id = ?",
            (card_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Card not found: {card_id}")
        return self._row_to_card(conn, row)

    @staticmethod
    def _row_to_card(conn: sqlite3.Connection, row: sqlite3.Row) -> MemoryCard:
        tag_rows = conn.execute(
            """
            SELECT mt.canonical_name
            FROM card_tags AS ct
            JOIN memory_tags AS mt ON mt.tag_id = ct.tag_id
            WHERE ct.card_id = ?
            ORDER BY mt.canonical_name ASC
            """,
            (row["card_id"],),
        ).fetchall()
        tags = tuple(str(r["canonical_name"]) for r in tag_rows)
        spans = tuple(
            SourceSpan.from_dict(item)
            for item in json.loads(row["spans_json"] or "[]")
        )
        return MemoryCard(
            card_id=str(row["card_id"]),
            domain=str(row["domain"]),
            card_type=str(row["card_type"]),
            title=str(row["title"]),
            state=dict(json.loads(row["state_json"] or "{}")),
            lifecycle_status=str(row["lifecycle_status"]),
            valid_from=str(row["valid_from"]),
            valid_to=None if row["valid_to"] is None else str(row["valid_to"]),
            superseded_by=None if row["superseded_by"] is None else str(row["superseded_by"]),
            tags=tags,
            spans=spans,
            foresight_note=str(row["foresight_note"] or ""),
            foresight_until=None if row["foresight_until"] is None else str(row["foresight_until"]),
        )

    def supersede_card(
        self,
        old_card_id: str,
        new_card: MemoryCard,
        *,
        now_iso: str | None = None,
    ) -> MemoryCard:
        """Bitemporally supersede an existing card with a newer replacement card.

        Marks ``old_card_id`` as ``superseded``, sets ``valid_to`` and ``superseded_by``,
        and persists ``new_card`` if it is not already stored.
        """
        if not isinstance(old_card_id, str) or not old_card_id.strip():
            raise ValueError("old_card_id must be a non-empty string")
        if not isinstance(new_card, MemoryCard):
            raise TypeError("new_card must be a MemoryCard instance")
        if old_card_id == new_card.card_id:
            raise ValueError("A card cannot supersede itself")
        if not new_card.spans:
            raise ValueError("Replacement MemoryCard must bind at least one SourceSpan")
        if new_card.lifecycle_status not in VALID_LIFECYCLE_STATUSES:
            raise ValueError(f"Invalid lifecycle_status {new_card.lifecycle_status!r}")

        ts = _utc_now_iso() if now_iso is None else str(now_iso)

        with self._db.transaction() as conn:
            old_row = conn.execute(
                "SELECT card_id FROM memory_cards WHERE card_id = ?",
                (old_card_id,),
            ).fetchone()
            if old_row is None:
                raise KeyError(f"Card not found: {old_card_id}")

            existing_new = conn.execute(
                "SELECT card_id FROM memory_cards WHERE card_id = ?",
                (new_card.card_id,),
            ).fetchone()
            if existing_new is None:
                resolved_tags = self._resolve_and_dedup_tags_in_tx(
                    conn, new_card.domain, new_card.tags
                )
                self._insert_card_in_tx(
                    conn,
                    new_card,
                    resolved_tags=resolved_tags,
                    created_at=ts,
                    updated_at=ts,
                )

            conn.execute(
                """
                UPDATE memory_cards
                SET lifecycle_status = 'superseded',
                    valid_to = ?,
                    superseded_by = ?,
                    updated_at = ?
                WHERE card_id = ?
                """,
                (ts, new_card.card_id, ts, old_card_id),
            )
            return self._load_card_in_tx(conn, new_card.card_id)

    def revoke_card(self, card_id: str, *, now_iso: str | None = None) -> bool:
        """Explicitly revoke a card (TEPA): remove from active recall while preserving audit history."""
        if not isinstance(card_id, str) or not card_id.strip():
            raise ValueError("card_id must be a non-empty string")

        ts = _utc_now_iso() if now_iso is None else str(now_iso)

        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT card_id, valid_to FROM memory_cards WHERE card_id = ?",
                (card_id,),
            ).fetchone()
            if row is None:
                return False

            effective_valid_to = ts if now_iso is not None else (row["valid_to"] or ts)
            conn.execute(
                """
                UPDATE memory_cards
                SET lifecycle_status = 'revoked',
                    valid_to = ?,
                    updated_at = ?
                WHERE card_id = ?
                """,
                (effective_valid_to, ts, card_id),
            )
            return True

    def link_cards(
        self,
        src_card_id: str,
        dst_card_id: str,
        *,
        relation: str = "related",
        weight: float = 1.0,
        bidirectional: bool = True,
    ) -> None:
        """Create or update a weighted relation edge between two memory cards."""
        if not isinstance(src_card_id, str) or not src_card_id.strip():
            raise ValueError("src_card_id must be a non-empty string")
        if not isinstance(dst_card_id, str) or not dst_card_id.strip():
            raise ValueError("dst_card_id must be a non-empty string")
        if src_card_id == dst_card_id:
            raise ValueError("Self-loop card links are not allowed")
        if not isinstance(relation, str) or not relation.strip():
            raise ValueError("relation must be a non-empty string")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or float(weight) <= 0.0:
            raise ValueError("weight must be a positive number")

        now = _utc_now_iso()
        w = float(weight)

        with self._db.transaction() as conn:
            for cid in (src_card_id, dst_card_id):
                exists = conn.execute(
                    "SELECT 1 FROM memory_cards WHERE card_id = ?",
                    (cid,),
                ).fetchone()
                if exists is None:
                    raise KeyError(f"Card not found: {cid}")

            pairs = [(src_card_id, dst_card_id)]
            if bidirectional:
                pairs.append((dst_card_id, src_card_id))

            for src, dst in pairs:
                conn.execute(
                    """
                    INSERT INTO card_links (src_card_id, dst_card_id, relation, weight, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(src_card_id, dst_card_id, relation) DO UPDATE SET
                        weight = excluded.weight,
                        created_at = excluded.created_at
                    """,
                    (src, dst, relation.strip(), w, now),
                )

    # ------------------------------------------------------------------
    # Three-level progressive recall
    # ------------------------------------------------------------------

    def recall_level1(
        self,
        domain: str | None,
        query_tags: Iterable[str],
        *,
        min_tag_hits: int = 1,
        top_k: int = 3,
    ) -> list[MemoryCard]:
        """Level-1 index recall: active cards only, fusing controlled tag hits with PPR diffusion."""
        if domain is not None:
            validate_domain(domain)
        if isinstance(min_tag_hits, bool) or not isinstance(min_tag_hits, int) or min_tag_hits < 1:
            raise ValueError("min_tag_hits must be an int >= 1")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative int")
        if top_k == 0:
            return []

        with self._db.read_connection() as conn:
            query_tag_ids = self._resolve_query_tag_ids(conn, domain, query_tags)
            if not query_tag_ids:
                return []

            query_tag_id_set = set(query_tag_ids)

            # Strictly restrict candidate universe to active cards (FAMA/TEPA zero-leakage).
            if domain is None:
                active_rows = conn.execute(
                    "SELECT * FROM memory_cards WHERE lifecycle_status = 'active'"
                ).fetchall()
            else:
                active_rows = conn.execute(
                    """
                    SELECT * FROM memory_cards
                    WHERE lifecycle_status = 'active' AND domain = ?
                    """,
                    (domain,),
                ).fetchall()

            if not active_rows:
                return []

            active_by_id: dict[str, sqlite3.Row] = {
                str(row["card_id"]): row for row in active_rows
            }
            active_ids = set(active_by_id.keys())

            # Map active cards to their tag_ids and tag_ids to active cards.
            placeholders = ",".join("?" for _ in active_ids)
            active_id_list = list(active_ids)
            tag_link_rows = conn.execute(
                f"SELECT card_id, tag_id FROM card_tags WHERE card_id IN ({placeholders})",
                active_id_list,
            ).fetchall()

            card_to_tags: dict[str, set[int]] = {cid: set() for cid in active_ids}
            tag_to_cards: dict[int, list[str]] = {}
            for r in tag_link_rows:
                cid = str(r["card_id"])
                tid = int(r["tag_id"])
                card_to_tags[cid].add(tid)
                tag_to_cards.setdefault(tid, []).append(cid)

            # Compute direct tag hits and seed distribution.
            direct_hits: dict[str, int] = {}
            seeds: dict[str, float] = {}
            for cid, tids in card_to_tags.items():
                hits = len(tids & query_tag_id_set)
                if hits > 0:
                    direct_hits[cid] = hits
                if hits >= min_tag_hits:
                    seeds[cid] = float(hits)

            if not seeds:
                return []

            # Build active-only adjacency graph from explicit card_links + light shared-tag edges.
            adj: dict[str, dict[str, float]] = {cid: {} for cid in active_ids}

            link_rows = conn.execute(
                f"""
                SELECT src_card_id, dst_card_id, weight
                FROM card_links
                WHERE src_card_id IN ({placeholders})
                  AND dst_card_id IN ({placeholders})
                """,
                active_id_list + active_id_list,
            ).fetchall()
            for r in link_rows:
                src = str(r["src_card_id"])
                dst = str(r["dst_card_id"])
                w = float(r["weight"])
                if src != dst and w > 0.0:
                    adj[src][dst] = adj[src].get(dst, 0.0) + w

            for cids in tag_to_cards.values():
                if len(cids) < 2:
                    continue
                for i, src in enumerate(cids):
                    for dst in cids[i + 1 :]:
                        adj[src][dst] = adj[src].get(dst, 0.0) + _SHARED_TAG_EDGE_WEIGHT
                        adj[dst][src] = adj[dst].get(src, 0.0) + _SHARED_TAG_EDGE_WEIGHT

            ppr_scores = personalized_pagerank(adj, seeds, damping=0.85, iters=12)

            # Rank active cards reachable by direct hits or PPR multi-hop diffusion.
            denom = float(len(query_tag_id_set))
            scored: list[tuple[float, int, float, str, str]] = []
            for cid in active_ids:
                hits = direct_hits.get(cid, 0)
                ppr_val = ppr_scores.get(cid, 0.0)
                if cid not in seeds and ppr_val <= 0.0:
                    continue
                direct_norm = hits / denom
                combined = direct_norm + ppr_val
                updated_at = str(active_by_id[cid]["updated_at"] or "")
                scored.append((combined, hits, ppr_val, updated_at, cid))

            scored.sort(key=lambda item: item[4])
            scored.sort(key=lambda item: item[3], reverse=True)
            scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            top_ids = [item[4] for item in scored[:top_k]]
            return [self._row_to_card(conn, active_by_id[cid]) for cid in top_ids]

    def read_evidence_level2(self, card_id: str, *, pad: int = 1) -> list[dict[str, Any]]:
        """Level-2 evidence recall: slice raw turns bound to the card's SourceSpans."""
        if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
            raise ValueError("pad must be a non-negative int")

        card = self.get_card(card_id)
        if card is None:
            raise KeyError(f"Card not found: {card_id}")

        seen_turns: dict[int, dict[str, Any]] = {}
        for span in card.spans:
            for turn in self._db.read_span(span, pad=pad):
                seen_turns[int(turn["turn_id"])] = turn

        return [seen_turns[tid] for tid in sorted(seen_turns.keys())]

    def deep_slice_level3(
        self,
        query: str,
        *,
        limit: int = 2,
        pad: int = 3,
    ) -> list[dict[str, Any]]:
        """Level-3 page-fault recall: search old raw turns via FTS5 and enqueue self-healing backfills."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative int")
        if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
            raise ValueError("pad must be a non-negative int")
        if limit == 0 or not isinstance(query, str) or not query.strip():
            return []

        hits = self._db.search_turns_fts(query, limit=limit)
        if not hits:
            return []

        now = _utc_now_iso()
        reason = f"level3_miss:{query.strip()[:120]}"
        results: list[dict[str, Any]] = []

        for hit in hits:
            episode_id = str(hit["episode_id"])
            turn_id = int(hit["turn_id"])
            anchor_span = SourceSpan(episode_id=episode_id, turn_start=turn_id, turn_end=turn_id)
            turns = self._db.read_span(anchor_span, pad=pad)

            if turns:
                turn_start = int(turns[0]["turn_id"])
                turn_end = int(turns[-1]["turn_id"])
            else:
                turn_start = turn_id
                turn_end = turn_id

            bound_span = SourceSpan(
                episode_id=episode_id,
                turn_start=turn_start,
                turn_end=turn_end,
            )

            with self._db.transaction() as conn:
                existing = conn.execute(
                    """
                    SELECT backfill_id FROM pending_backfills
                    WHERE episode_id = ? AND turn_start = ? AND turn_end = ? AND status = 'pending'
                    ORDER BY backfill_id ASC
                    LIMIT 1
                    """,
                    (episode_id, turn_start, turn_end),
                ).fetchone()
                if existing is not None:
                    backfill_id = int(existing["backfill_id"])
                    conn.execute(
                        "UPDATE pending_backfills SET reason = ?, updated_at = ? WHERE backfill_id = ?",
                        (reason, now, backfill_id),
                    )
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO pending_backfills (
                            episode_id, turn_start, turn_end, reason, status, attempts, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, 'pending', 0, ?, ?)
                        """,
                        (episode_id, turn_start, turn_end, reason, now, now),
                    )
                    backfill_id = int(cur.lastrowid)

            results.append(
                {
                    "hit": hit,
                    "span": bound_span.to_dict(),
                    "turns": turns,
                    "backfill_id": backfill_id,
                }
            )

        return results

    def list_pending_backfills(self, *, status: str | None = "pending") -> list[dict[str, Any]]:
        """List self-healing backfill queue entries (defaults to pending items)."""
        with self._db.read_connection() as conn:
            if status is None:
                rows = conn.execute(
                    """
                    SELECT backfill_id, episode_id, turn_start, turn_end, reason, status, attempts, created_at, updated_at
                    FROM pending_backfills
                    ORDER BY backfill_id ASC
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT backfill_id, episode_id, turn_start, turn_end, reason, status, attempts, created_at, updated_at
                    FROM pending_backfills
                    WHERE status = ?
                    ORDER BY backfill_id ASC
                    """,
                    (status,),
                ).fetchall()

        return [
            {
                "backfill_id": int(r["backfill_id"]),
                "episode_id": str(r["episode_id"]),
                "turn_start": int(r["turn_start"]),
                "turn_end": int(r["turn_end"]),
                "reason": str(r["reason"]),
                "status": str(r["status"]),
                "attempts": int(r["attempts"]),
                "created_at": str(r["created_at"]),
                "updated_at": str(r["updated_at"]),
            }
            for r in rows
        ]
