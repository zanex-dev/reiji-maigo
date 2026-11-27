"""Unified two-tier memory engine facade bridging ActivePool and CardStore."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
from typing import Any
import uuid

from reiji_maigo.cards.card_store import CardStore
from reiji_maigo.core.contracts import MemoryCard, PoolEvent, SourceSpan
from reiji_maigo.core.kernel import ReijiKernel
from reiji_maigo.pool.active_pool import ActivePool, VALID_SCOPES
from reiji_maigo.security import (
    AAD_CARD,
    DEFAULT_VAULT_FILENAME,
    AuditChainLedger,
    MutationApprovalGate,
    MutationTier,
    SecretVault,
    can_mutate_authoritative_memory,
    ensure_security_tables,
    sanitize_for_memory,
    sign_canonical,
    verify_canonical,
)
from reiji_maigo.storage.sqlite_engine import ReijiDatabase

HIBERNATED_CARD_TYPE: str = "hibernated_event"
ARCHIVED_CARD_TYPE: str = "archived_event"
DEFAULT_DB_FILENAME: str = "reiji_maigo.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_spans_lossless(
    base_spans: Iterable[SourceSpan],
    extra_spans: Iterable[SourceSpan],
) -> tuple[SourceSpan, ...]:
    merged: list[SourceSpan] = []
    seen: set[SourceSpan] = set()
    for span in (*base_spans, *extra_spans):
        if span not in seen:
            seen.add(span)
            merged.append(span)
    return tuple(merged)


class _EphemeralSecretVault(SecretVault):
    """In-memory SecretVault for ``:memory:`` databases without a custom home."""

    def __init__(self) -> None:
        super().__init__(Path(":memory:"))
        self._data: dict[str, Any] = {}

    def load(
        self, *, strict: bool = False, for_update: bool = False
    ) -> dict[str, Any]:
        return dict(self._data)

    def save(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("vault data must be a Mapping")
        self._data = dict(data)

    def get_or_create_signing_key(self, *, key_name: str = "signing_key_hex") -> bytes:
        existing = self._data.get(key_name)
        if existing is not None:
            if not isinstance(existing, str) or len(existing) != 64:
                raise ValueError(f"Corrupted signing key entry {key_name!r} in memory vault")
            return bytes.fromhex(existing)
        key_bytes = secrets.token_bytes(32)
        self._data[key_name] = key_bytes.hex()
        return key_bytes


class _SealedCardStore(CardStore):
    """CardStore subclass that refreshes ``card_seals`` HMAC on legal mutations."""

    def __init__(
        self,
        db: ReijiDatabase,
        *,
        on_card_mutated: Callable[[MemoryCard], Any],
    ) -> None:
        super().__init__(db)
        self._on_card_mutated = on_card_mutated

    def create_card(self, *args: Any, **kwargs: Any) -> MemoryCard:
        card = super().create_card(*args, **kwargs)
        self._on_card_mutated(card)
        return card

    def supersede_card(
        self,
        old_card_id: str,
        new_card: MemoryCard,
        *,
        now_iso: str | None = None,
    ) -> MemoryCard:
        result = super().supersede_card(old_card_id, new_card, now_iso=now_iso)
        old_card = self.get_card(old_card_id)
        if old_card is not None:
            self._on_card_mutated(old_card)
        self._on_card_mutated(result)
        return result

    def revoke_card(
        self,
        card_id: str,
        *,
        now_iso: str | None = None,
    ) -> bool:
        revoked = super().revoke_card(card_id, now_iso=now_iso)
        if revoked:
            updated = self.get_card(card_id)
            if updated is not None:
                self._on_card_mutated(updated)
        return revoked


class ReijiMemoryEngine:
    """Facade aggregating ReijiKernel, ReijiDatabase, ActivePool, and CardStore.

    Coordinates two-tier memory transitions:
      - ActivePool -> CardStore archival (completed events or hibernated swap-out)
      - CardStore -> ActivePool hot wake-up (resuming hibernated events losslessly)
    """

    def __init__(
        self,
        *,
        home: Path | str | None = None,
        db_path: Path | str | None = None,
        kernel: ReijiKernel | None = None,
        max_events: int = 8,
        max_tokens: int = 900,
        load_plugins: bool = True,
        disable_plugins: bool | None = None,
        disabled_plugins: Iterable[str] | None = None,
        env: Mapping[str, str] | None = None,
        vault_path: Path | str | None = None,
        signing_key: bytes | None = None,
    ) -> None:
        self.kernel: ReijiKernel = (
            kernel
            if kernel is not None
            else ReijiKernel(
                home=home,
                env=env,
                disable_plugins=disable_plugins,
                disabled_plugins=disabled_plugins,
            )
        )
        self.home: Path = self.kernel.home
        resolved_db_path: Path | str = (
            self.home / DEFAULT_DB_FILENAME if db_path is None else db_path
        )
        self.db: ReijiDatabase = ReijiDatabase(resolved_db_path)

        # Resolve SecretVault without ever leaking writes to the user's real home in :memory: tests.
        is_memory_db = str(resolved_db_path) == ":memory:"
        default_user_home = Path.home() / ".reiji_maigo"
        if vault_path is not None:
            self.vault: SecretVault = SecretVault(vault_path)
        elif is_memory_db and home is None and (
            kernel is None or self.home == default_user_home
        ):
            self.vault = _EphemeralSecretVault()
        elif not is_memory_db and home is None and kernel is None:
            self.vault = SecretVault(
                Path(resolved_db_path).resolve().parent / DEFAULT_VAULT_FILENAME
            )
        else:
            self.vault = SecretVault(self.home / DEFAULT_VAULT_FILENAME)

        if signing_key is not None:
            if not isinstance(signing_key, (bytes, bytearray)) or not signing_key:
                raise ValueError("signing_key must be non-empty bytes or None")
            self._signing_key: bytes = bytes(signing_key)
        else:
            self._signing_key = self.vault.get_or_create_signing_key()

        ensure_security_tables(self.db)
        self.audit: AuditChainLedger = AuditChainLedger(
            self.db,
            signing_key=self._signing_key,
        )
        self.approval_gate: MutationApprovalGate = MutationApprovalGate(
            self.db,
            ledger=self.audit,
            signing_key=self._signing_key,
        )

        self.pool: ActivePool = ActivePool(
            self.db,
            max_events=max_events,
            max_tokens=max_tokens,
        )
        self.cards: CardStore = _SealedCardStore(
            self.db,
            on_card_mutated=self.seal_card,
        )
        if load_plugins:
            self.kernel.load_plugins()
            from reiji_maigo.plugins import register_builtin_plugins

            register_builtin_plugins(self.kernel)

    def close(self) -> None:
        """Close the underlying SQLite database connection."""
        self.db.close()

    def __enter__(self) -> ReijiMemoryEngine:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Two-tier transitions: Pool <-> CardStore
    # ------------------------------------------------------------------

    def archive_pool_event(
        self,
        event_id: str,
        *,
        domain: str,
        tags: Iterable[str],
        state: dict[str, Any],
        hibernate: bool = False,
        card_type: str | None = None,
        title: str | None = None,
        foresight_note: str = "",
        foresight_until: str | None = None,
        now_iso: str | None = None,
    ) -> MemoryCard:
        """Transfer a PoolEvent into CardStore preserving all SourceSpans, then remove it from ActivePool.

        When ``hibernate=True``, stores explicit hibernation metadata in the card's
        ``state`` payload so it can later be recalled and hot-woken via
        :meth:`wake_hibernated_card`.
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a dict[str, Any]")

        ev = self.pool.get_event(event_id)
        if ev is None:
            raise KeyError(f"PoolEvent not found: {event_id!r}")
        if not ev.spans:
            raise ValueError(f"PoolEvent {event_id!r} has no bound SourceSpan")

        ts = _utc_now_iso() if now_iso is None else str(now_iso)
        merged_state: dict[str, Any] = dict(state)

        if hibernate:
            resolved_card_type = card_type or HIBERNATED_CARD_TYPE
            merged_state["pool_state"] = "hibernated"
            merged_state["hibernated"] = True
        else:
            resolved_card_type = card_type or ARCHIVED_CARD_TYPE
            merged_state.setdefault("pool_state", "archived")
            merged_state.setdefault("hibernated", False)

        merged_state.setdefault("source_event_id", ev.event_id)
        merged_state.setdefault("source_scope", ev.scope)
        merged_state.setdefault("source_status", ev.status)
        merged_state.setdefault("checkpoint_summary", ev.checkpoint_summary)
        merged_state.setdefault("steps", list(ev.steps))
        merged_state.setdefault("hit_count", ev.hit_count)

        card = self.cards.create_card(
            domain=domain,
            card_type=resolved_card_type,
            title=title.strip() if isinstance(title, str) and title.strip() else ev.title,
            state=merged_state,
            tags=tags,
            spans=ev.spans,
            foresight_note=foresight_note,
            foresight_until=foresight_until,
            valid_from=ts,
            created_at=ts,
            updated_at=ts,
        )

        target_pool_status = "hibernated" if hibernate else "archived"
        self.pool.update_status(event_id, target_pool_status, now_iso=ts)

        self.kernel.hooks.emit(
            "pool.archived",
            {
                "event_id": event_id,
                "card_id": card.card_id,
                "hibernated": hibernate,
                "domain": card.domain,
            },
        )
        return card

    def wake_hibernated_card(
        self,
        card_id: str,
        new_step: str,
        span: SourceSpan,
        *,
        scope: str | None = None,
        now_iso: str | None = None,
        deactivate_card: bool = True,
    ) -> PoolEvent:
        """Hot-wake a hibernated MemoryCard back into ActivePool and append ``new_step``.

        Preserves 100% of historical spans from both the card and any underlying
        hibernated pool event record, and appends ``span`` without duplicates.
        """
        if not isinstance(new_step, str) or not new_step.strip():
            raise ValueError("new_step must be a non-empty string")
        if not isinstance(span, SourceSpan):
            raise TypeError("span must be a SourceSpan")

        card = self.cards.get_card(card_id)
        if card is None:
            raise KeyError(f"Card not found: {card_id!r}")
        if card.lifecycle_status != "active":
            raise ValueError(
                f"Cannot wake non-active card {card_id!r} (status={card.lifecycle_status!r})"
            )

        is_hibernated = (
            bool(card.state.get("hibernated"))
            or card.state.get("pool_state") == "hibernated"
            or card.card_type == HIBERNATED_CARD_TYPE
        )
        if not is_hibernated:
            raise ValueError(f"Card {card_id!r} is not marked as hibernated")

        ts = _utc_now_iso() if now_iso is None else str(now_iso)
        source_event_id = card.state.get("source_event_id")
        existing: PoolEvent | None = (
            self.pool.get_event(str(source_event_id))
            if isinstance(source_event_id, str) and source_event_id.strip()
            else None
        )

        if existing is not None:
            woken = self.pool.append_step(
                existing.event_id,
                new_step.strip(),
                span,
                status="active",
                now_iso=ts,
            )
            merged_spans = _merge_spans_lossless(woken.spans, card.spans)
            if merged_spans != woken.spans:
                woken = self._persist_pool_event(
                    PoolEvent(
                        event_id=woken.event_id,
                        title=woken.title,
                        scope=woken.scope,
                        status=woken.status,
                        steps=woken.steps,
                        checkpoint_summary=woken.checkpoint_summary,
                        spans=merged_spans,
                        hit_count=woken.hit_count,
                        created_at=woken.created_at,
                        last_touched_at=woken.last_touched_at,
                        dormant_nights=woken.dormant_nights,
                    )
                )
        else:
            raw_scope = scope or str(card.state.get("source_scope") or "quest")
            resolved_scope = raw_scope if raw_scope in VALID_SCOPES else "quest"
            prior_steps_raw = card.state.get("steps")
            prior_steps = (
                tuple(str(s) for s in prior_steps_raw if isinstance(s, str) and s.strip())
                if isinstance(prior_steps_raw, list)
                else ()
            )
            prior_summary = str(card.state.get("checkpoint_summary") or "")
            raw_hits = card.state.get("hit_count", 0)
            prior_hits = int(raw_hits) if isinstance(raw_hits, int) and raw_hits >= 0 else 0
            merged_spans = _merge_spans_lossless(card.spans, (span,))
            woken = self._persist_pool_event(
                PoolEvent(
                    event_id=f"ev_{uuid.uuid4().hex[:12]}",
                    title=card.title,
                    scope=resolved_scope,
                    status="active",
                    steps=prior_steps + (new_step.strip(),),
                    checkpoint_summary=prior_summary,
                    spans=merged_spans,
                    hit_count=prior_hits + 1,
                    created_at=card.valid_from or ts,
                    last_touched_at=ts,
                    dormant_nights=0,
                )
            )

        if deactivate_card:
            self.cards.revoke_card(card_id, now_iso=ts)

        self.kernel.hooks.emit(
            "card.woken",
            {
                "card_id": card_id,
                "event_id": woken.event_id,
                "title": woken.title,
            },
        )
        return woken

    def _persist_pool_event(self, ev: PoolEvent) -> PoolEvent:
        with self.db.transaction() as conn:
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
        return ev

    # ------------------------------------------------------------------
    # High-level convenience methods
    # ------------------------------------------------------------------

    def append_turn(
        self,
        episode_id: str,
        role: str,
        content: str,
        created_at: str | None = None,
        *,
        origin_trust: str = "user_verified",
        sanitize: bool = True,
    ) -> int:
        stored_content = content
        redaction_count = 0
        blindfold_count = 0
        poison_detected = False
        san_res = None

        if sanitize:
            san_res = sanitize_for_memory(content, origin_trust=origin_trust)
            stored_content = san_res.clean_text
            redaction_count = san_res.redaction_count
            blindfold_count = san_res.blindfold_count
            poison_detected = san_res.poison_detected

        turn_id = self.db.append_turn(
            episode_id, role, stored_content, created_at=created_at
        )

        if san_res is not None:
            if redaction_count > 0 or blindfold_count > 0:
                self.audit.append(
                    f"ingress:{role}",
                    "turn_sanitized",
                    target_id=f"{episode_id}:{turn_id}",
                    tier=MutationTier.L0_AUTO.value,
                    detail={
                        "turn_id": turn_id,
                        "episode_id": episode_id,
                        "role": role,
                        "origin_trust": san_res.origin_trust,
                        "redaction_count": redaction_count,
                        "redaction_kinds": list(san_res.redaction_kinds),
                        "blindfold_count": blindfold_count,
                    },
                    now_iso=created_at,
                )
            if poison_detected:
                self.audit.append(
                    f"ingress:{role}",
                    "poison_detected",
                    target_id=f"{episode_id}:{turn_id}",
                    tier=MutationTier.L2_FORBIDDEN.value,
                    detail={
                        "turn_id": turn_id,
                        "episode_id": episode_id,
                        "role": role,
                        "origin_trust": san_res.origin_trust,
                        "poison_reasons": list(san_res.poison_reasons),
                        "can_mutate": can_mutate_authoritative_memory(san_res),
                    },
                    now_iso=created_at,
                )

        self.kernel.hooks.emit(
            "turn.appended",
            {
                "turn_id": turn_id,
                "episode_id": episode_id,
                "role": role,
                "origin_trust": origin_trust,
                "redaction_count": redaction_count,
                "blindfold_count": blindfold_count,
                "poison_detected": poison_detected,
            },
        )
        return turn_id

    # ------------------------------------------------------------------
    # Card cryptographic sealing & security telemetry
    # ------------------------------------------------------------------

    def seal_card(
        self,
        card_or_id: MemoryCard | str,
        *,
        now_iso: str | None = None,
    ) -> str:
        """Compute and persist a domain-separated HMAC-SHA256 seal for a MemoryCard."""
        if isinstance(card_or_id, MemoryCard):
            card = card_or_id
        elif isinstance(card_or_id, str) and card_or_id.strip():
            fetched = self.cards.get_card(card_or_id.strip())
            if fetched is None:
                raise KeyError(f"Card not found: {card_or_id!r}")
            card = fetched
        else:
            raise TypeError("card_or_id must be a MemoryCard or non-empty card_id str")

        ts = _utc_now_iso() if now_iso is None else str(now_iso)
        seal_hmac = sign_canonical(
            card.to_dict(),
            self._signing_key,
            domain_tag=AAD_CARD,
        )
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO card_seals (card_id, seal_hmac, algo, sealed_at, updated_at)
                VALUES (?, ?, 'hmac-sha256', ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    seal_hmac = excluded.seal_hmac,
                    algo = excluded.algo,
                    updated_at = excluded.updated_at
                """,
                (card.card_id, seal_hmac, ts, ts),
            )
        return seal_hmac

    def verify_card_seals(
        self,
        *,
        card_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Verify HMAC-SHA256 seals for all (or selected) cards against ``card_seals``."""
        with self.db.read_connection() as conn:
            if card_ids is None:
                card_rows = conn.execute(
                    "SELECT card_id FROM memory_cards ORDER BY card_id ASC"
                ).fetchall()
                target_ids = [str(row["card_id"]) for row in card_rows]
            else:
                target_ids = [str(cid) for cid in card_ids]

            seal_rows = conn.execute(
                "SELECT card_id, seal_hmac, algo FROM card_seals"
            ).fetchall()

        seals_map: dict[str, tuple[str, str]] = {
            str(row["card_id"]): (str(row["seal_hmac"]), str(row["algo"]))
            for row in seal_rows
        }

        tampered: list[dict[str, str]] = []
        unsealed: list[str] = []
        sealed_count = 0

        for cid in target_ids:
            card = self.cards.get_card(cid)
            if card is None:
                tampered.append({"card_id": cid, "reason": "card_missing"})
                continue
            seal_entry = seals_map.get(cid)
            if seal_entry is None:
                unsealed.append(cid)
                continue
            sealed_count += 1
            seal_hmac, algo = seal_entry
            if algo != "hmac-sha256":
                tampered.append({"card_id": cid, "reason": f"unsupported_algo:{algo}"})
                continue
            if not verify_canonical(
                card.to_dict(),
                seal_hmac,
                self._signing_key,
                domain_tag=AAD_CARD,
            ):
                tampered.append({"card_id": cid, "reason": "seal_hmac_mismatch"})

        orphan_seals: list[str] = []
        if card_ids is None:
            known = set(target_ids)
            orphan_seals = sorted(cid for cid in seals_map if cid not in known)

        ok = not tampered and not unsealed and not orphan_seals
        return {
            "ok": ok,
            "checked": len(target_ids),
            "sealed": sealed_count,
            "tampered": tampered,
            "unsealed": unsealed,
            "orphan_seals": orphan_seals,
        }

    def security_summary(self) -> dict[str, Any]:
        """Aggregate live security telemetry across sanitizer, L1/L2 gates, chain, and seals."""
        ensure_security_tables(self.db)
        redacted_secrets = 0
        blindfolded_paths = 0
        sanitized_turns = 0
        poison_detected_count = 0
        l2_blocked_count = 0

        with self.db.read_connection() as conn:
            san_rows = conn.execute(
                "SELECT detail_json FROM audit_chain WHERE action = 'turn_sanitized'"
            ).fetchall()
            for row in san_rows:
                sanitized_turns += 1
                try:
                    detail = json.loads(row["detail_json"] or "{}")
                except json.JSONDecodeError:
                    detail = {}
                redacted_secrets += int(detail.get("redaction_count") or 0)
                blindfolded_paths += int(detail.get("blindfold_count") or 0)

            poison_detected_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM audit_chain WHERE action = 'poison_detected'"
                ).fetchone()[0]
            )
            l2_blocked_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM audit_chain WHERE action LIKE 'mutation_blocked:%'"
                ).fetchone()[0]
            )
            pending_l1 = int(
                conn.execute(
                    "SELECT COUNT(*) FROM mutation_reviews WHERE status = 'pending'"
                ).fetchone()[0]
            )

        chain_report = self.audit.verify_chain()
        seals_report = self.verify_card_seals()
        redactions_total = redacted_secrets + blindfolded_paths
        poison_blocked_total = poison_detected_count + l2_blocked_count

        return {
            "redactions": redactions_total,
            "redacted_secrets": redacted_secrets,
            "blindfolded_paths": blindfolded_paths,
            "sanitized_turns": sanitized_turns,
            "poison_detected": poison_detected_count,
            "l2_blocked": l2_blocked_count,
            "poison_blocked": poison_blocked_total,
            "pending_l1": pending_l1,
            "audit_chain": chain_report,
            "card_seals": {
                "ok": seals_report["ok"],
                "checked": seals_report["checked"],
                "sealed": seals_report["sealed"],
                "tampered_count": len(seals_report["tampered"]),
                "unsealed_count": len(seals_report["unsealed"]),
                "orphan_count": len(seals_report["orphan_seals"]),
            },
        }

    def pool_snapshot(
        self,
        *,
        now_iso: str | None = None,
        include_undigested: bool = True,
    ) -> str:
        return self.pool.render_snapshot(
            now_iso=now_iso,
            include_undigested=include_undigested,
        )

    def recall_cards(
        self,
        domain: str | None,
        query_tags: Iterable[str],
        *,
        min_tag_hits: int = 1,
        top_k: int = 3,
    ) -> list[MemoryCard]:
        return self.cards.recall_level1(
            domain,
            query_tags,
            min_tag_hits=min_tag_hits,
            top_k=top_k,
        )

    def hybrid_recall(
        self,
        query: str,
        *,
        domain: str | None = None,
        query_tags: Iterable[str] = (),
        top_k: int = 3,
        now_iso: str | None = None,
    ) -> Any:
        """Delegate hybrid recall to the active ``reranker`` provider, falling back to ``recall_cards``."""
        reranker = None
        get_provider = getattr(self.kernel, "get_provider", None)
        if callable(get_provider):
            try:
                reranker = get_provider("reranker")
            except (KeyError, LookupError, TypeError):
                reranker = None
        if reranker is None:
            providers_map = getattr(self.kernel, "providers", None)
            if isinstance(providers_map, dict):
                slot = providers_map.get("reranker")
                if isinstance(slot, dict) and slot:
                    reranker = next(iter(slot.values()))
                elif slot is not None:
                    reranker = slot

        tag_tuple = tuple(query_tags)
        if reranker is not None:
            fn = getattr(reranker, "hybrid_recall", None)
            if callable(fn):
                return fn(
                    self,
                    query,
                    domain=domain,
                    query_tags=tag_tuple,
                    top_k=top_k,
                    now_iso=now_iso,
                )
            if callable(reranker):
                return reranker(
                    self,
                    query,
                    domain=domain,
                    query_tags=tag_tuple,
                    top_k=top_k,
                    now_iso=now_iso,
                )

        fallback_tags = [str(tag).strip() for tag in tag_tuple if str(tag).strip()]
        if not fallback_tags and query and str(query).strip():
            fallback_tags = [tok for tok in str(query).strip().split() if tok]
        if not fallback_tags:
            return []
        return self.recall_cards(
            domain,
            fallback_tags,
            min_tag_hits=1,
            top_k=top_k,
        )

    def read_evidence(self, card_id: str, *, pad: int = 1) -> list[dict[str, Any]]:
        return self.cards.read_evidence_level2(card_id, pad=pad)

    def deep_slice(
        self,
        query: str,
        *,
        limit: int = 2,
        pad: int = 3,
    ) -> list[dict[str, Any]]:
        return self.cards.deep_slice_level3(query, limit=limit, pad=pad)

    def status(self) -> dict[str, Any]:
        """Return a structured status overview of kernel, storage, pool, and card counts."""
        with self.db.read_connection() as conn:
            raw_turns_count = int(conn.execute("SELECT COUNT(*) FROM raw_turns").fetchone()[0])
            pool_active_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM pool_events WHERE status IN ('active', 'blocked', 'done', 'aborted')"
                ).fetchone()[0]
            )
            pool_total_count = int(conn.execute("SELECT COUNT(*) FROM pool_events").fetchone()[0])
            cards_active_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memory_cards WHERE lifecycle_status = 'active'"
                ).fetchone()[0]
            )
            cards_total_count = int(conn.execute("SELECT COUNT(*) FROM memory_cards").fetchone()[0])
            tags_count = int(conn.execute("SELECT COUNT(*) FROM memory_tags").fetchone()[0])
            backfills_pending = int(
                conn.execute(
                    "SELECT COUNT(*) FROM pending_backfills WHERE status = 'pending'"
                ).fetchone()[0]
            )

        kernel_status = self.kernel.status()
        return {
            "home": str(self.home),
            "db_path": self.db.db_path,
            "kernel": kernel_status,
            "plugins_disabled": kernel_status["plugins_disabled"],
            "loaded_plugins": kernel_status["loaded_plugins"],
            "counts": {
                "raw_turns": raw_turns_count,
                "pool_visible": pool_active_count,
                "pool_total": pool_total_count,
                "cards_active": cards_active_count,
                "cards_total": cards_total_count,
                "tags": tags_count,
                "pending_backfills": backfills_pending,
            },
            "security": self.security_summary(),
        }


__all__ = [
    "ARCHIVED_CARD_TYPE",
    "DEFAULT_DB_FILENAME",
    "HIBERNATED_CARD_TYPE",
    "ReijiMemoryEngine",
]
