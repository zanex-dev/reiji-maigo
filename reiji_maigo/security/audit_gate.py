"""Tamper-evident hash-chained audit ledger and L0/L1/L2 mutation approval gate.

Provides:
- ``ensure_security_tables``: idempotent SQLite schema bootstrap for
  ``audit_chain``, ``mutation_reviews``, and ``card_seals``.
- ``AuditChainLedger``: append-only hash/HMAC chained audit log with
  tamper and deletion detection via ``verify_chain``.
- ``MutationApprovalGate``: L0/L1/L2 governance gate protecting critical
  memory cards from unapproved or poisoned supersede/revoke mutations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
import json
import sqlite3
from typing import Any
from uuid import uuid4

from reiji_maigo.cards.card_store import CardStore
from reiji_maigo.core.contracts import MemoryCard
from reiji_maigo.storage.sqlite_engine import ReijiDatabase

PROTECTED_CARD_TYPES: frozenset[str] = frozenset(
    {"core_rule", "persona_anchor", "decision", "companion_bond"}
)
GENESIS_PREV_HASH: str = "0" * 64
VALID_OP_KINDS: frozenset[str] = frozenset({"supersede", "revoke"})
UNTRUSTED_ORIGINS: frozenset[str] = frozenset({"untrusted_external"})


class MutationTier(str, Enum):
    """Three-tier memory mutation security classification aligned with L0/L1/L2."""

    L0_AUTO = "L0"
    L1_APPROVAL = "L1"
    L2_FORBIDDEN = "L2"


VALID_TIERS: frozenset[str] = frozenset(tier.value for tier in MutationTier)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_tier(tier: str | MutationTier) -> str:
    value = tier.value if isinstance(tier, MutationTier) else str(tier).strip().upper()
    if value not in VALID_TIERS:
        raise ValueError(f"Invalid tier {tier!r}; expected one of {sorted(VALID_TIERS)}")
    return value


def _normalize_op_kind(op_kind: str) -> str:
    if not isinstance(op_kind, str) or not op_kind.strip():
        raise ValueError("op_kind must be a non-empty string")
    value = op_kind.strip().lower()
    if value not in VALID_OP_KINDS:
        raise ValueError(
            f"Invalid op_kind {op_kind!r}; expected one of {sorted(VALID_OP_KINDS)}"
        )
    return value


def _canonical_json(data: Mapping[str, Any] | None) -> str:
    if data is None:
        return "{}"
    if not isinstance(data, Mapping):
        raise TypeError("detail must be a Mapping[str, Any] or None")
    return json.dumps(dict(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compute_entry_hash(
    *,
    seq: int,
    ts: str,
    actor: str,
    action: str,
    target_id: str,
    tier: str,
    detail_json: str,
    prev_hash: str,
    signing_key: bytes | None,
) -> tuple[str, str]:
    payload = json.dumps(
        {
            "seq": int(seq),
            "ts": ts,
            "actor": actor,
            "action": action,
            "target_id": target_id,
            "tier": tier,
            "detail_json": detail_json,
            "prev_hash": prev_hash,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    if signing_key is not None:
        digest = hmac.new(signing_key, payload, hashlib.sha256).hexdigest()
        return digest, "hmac-sha256"
    digest = hashlib.sha256(payload).hexdigest()
    return digest, "sha256"


def ensure_security_tables(db: ReijiDatabase) -> None:
    """Idempotently create security tables: audit_chain, mutation_reviews, card_seals."""
    if not isinstance(db, ReijiDatabase):
        raise TypeError("db must be a ReijiDatabase instance")

    with db.transaction() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_chain (
                seq INTEGER PRIMARY KEY,
                ts TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                target_id TEXT NOT NULL DEFAULT '',
                tier TEXT NOT NULL DEFAULT 'L0',
                detail_json TEXT NOT NULL DEFAULT '{}',
                prev_hash TEXT NOT NULL,
                entry_hash TEXT NOT NULL,
                sig_mode TEXT NOT NULL DEFAULT 'sha256'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_chain_ts ON audit_chain(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_chain_target ON audit_chain(target_id)"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mutation_reviews (
                review_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                tier TEXT NOT NULL DEFAULT 'L1',
                actor TEXT NOT NULL,
                op_kind TEXT NOT NULL,
                target_card_id TEXT NOT NULL,
                replacement_card_json TEXT,
                origin_trust TEXT NOT NULL DEFAULT 'assistant',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                decided_at TEXT,
                decided_by TEXT,
                decision_reason TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mutation_reviews_status
            ON mutation_reviews(status, created_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mutation_reviews_target
            ON mutation_reviews(target_card_id)
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS card_seals (
                card_id TEXT PRIMARY KEY,
                seal_hmac TEXT NOT NULL,
                algo TEXT NOT NULL DEFAULT 'hmac-sha256',
                sealed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


class AuditChainLedger:
    """Hash-chained SQLite audit ledger detecting row tampering and deletions."""

    def __init__(self, db: ReijiDatabase, *, signing_key: bytes | None = None) -> None:
        if not isinstance(db, ReijiDatabase):
            raise TypeError("db must be a ReijiDatabase instance")
        if signing_key is not None:
            if not isinstance(signing_key, (bytes, bytearray)) or len(signing_key) == 0:
                raise ValueError("signing_key must be non-empty bytes or None")
            self._signing_key: bytes | None = bytes(signing_key)
        else:
            self._signing_key = None

        self._db = db
        ensure_security_tables(db)

    def append(
        self,
        actor: str,
        action: str,
        *,
        target_id: str = "",
        tier: str = "L0",
        detail: Mapping[str, Any] | None = None,
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        """Append a chained audit entry binding ``seq`` and ``prev_hash`` into ``entry_hash``."""
        with self._db.transaction() as conn:
            return self._append_in_tx(
                conn,
                actor=actor,
                action=action,
                target_id=target_id,
                tier=tier,
                detail=detail,
                now_iso=now_iso,
            )

    def _append_in_tx(
        self,
        conn: sqlite3.Connection,
        *,
        actor: str,
        action: str,
        target_id: str = "",
        tier: str = "L0",
        detail: Mapping[str, Any] | None = None,
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("actor must be a non-empty string")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")

        actor_clean = actor.strip()
        action_clean = action.strip()
        target_clean = "" if target_id is None else str(target_id)
        tier_clean = _normalize_tier(tier)
        detail_json = _canonical_json(detail)
        ts = _utc_now_iso() if now_iso is None else str(now_iso)

        last_row = conn.execute(
            "SELECT seq, entry_hash FROM audit_chain ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if last_row is None:
            next_seq = 1
            prev_hash = GENESIS_PREV_HASH
        else:
            next_seq = int(last_row["seq"]) + 1
            prev_hash = str(last_row["entry_hash"])

        entry_hash, sig_mode = _compute_entry_hash(
            seq=next_seq,
            ts=ts,
            actor=actor_clean,
            action=action_clean,
            target_id=target_clean,
            tier=tier_clean,
            detail_json=detail_json,
            prev_hash=prev_hash,
            signing_key=self._signing_key,
        )

        conn.execute(
            """
            INSERT INTO audit_chain (
                seq, ts, actor, action, target_id, tier,
                detail_json, prev_hash, entry_hash, sig_mode
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                next_seq,
                ts,
                actor_clean,
                action_clean,
                target_clean,
                tier_clean,
                detail_json,
                prev_hash,
                entry_hash,
                sig_mode,
            ),
        )

        return {
            "seq": next_seq,
            "ts": ts,
            "actor": actor_clean,
            "action": action_clean,
            "target_id": target_clean,
            "tier": tier_clean,
            "detail": json.loads(detail_json),
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
            "sig_mode": sig_mode,
        }

    def verify_chain(self) -> dict[str, Any]:
        """Verify continuity and cryptographic integrity across all ``audit_chain`` rows."""
        with self._db.read_connection() as conn:
            rows = conn.execute(
                """
                SELECT seq, ts, actor, action, target_id, tier,
                       detail_json, prev_hash, entry_hash, sig_mode
                FROM audit_chain
                ORDER BY seq ASC
                """
            ).fetchall()

        if not rows:
            return {"ok": True, "checked": 0, "broken_seq": None, "reason": "empty chain"}

        expected_seq = 1
        expected_prev = GENESIS_PREV_HASH
        expected_mode = "hmac-sha256" if self._signing_key is not None else "sha256"
        checked = 0

        for row in rows:
            actual_seq = int(row["seq"])
            if actual_seq != expected_seq:
                return {
                    "ok": False,
                    "checked": checked,
                    "broken_seq": expected_seq,
                    "reason": f"sequence gap: expected seq {expected_seq}, found seq {actual_seq}",
                }

            actual_prev = str(row["prev_hash"])
            if not hmac.compare_digest(actual_prev, expected_prev):
                return {
                    "ok": False,
                    "checked": checked,
                    "broken_seq": actual_seq,
                    "reason": f"prev_hash mismatch at seq {actual_seq}",
                }

            actual_mode = str(row["sig_mode"])
            if actual_mode != expected_mode:
                return {
                    "ok": False,
                    "checked": checked,
                    "broken_seq": actual_seq,
                    "reason": (
                        f"sig_mode mismatch at seq {actual_seq}: "
                        f"expected {expected_mode}, found {actual_mode}"
                    ),
                }

            recomputed_hash, _ = _compute_entry_hash(
                seq=actual_seq,
                ts=str(row["ts"]),
                actor=str(row["actor"]),
                action=str(row["action"]),
                target_id=str(row["target_id"]),
                tier=str(row["tier"]),
                detail_json=str(row["detail_json"]),
                prev_hash=actual_prev,
                signing_key=self._signing_key,
            )
            actual_hash = str(row["entry_hash"])
            if not hmac.compare_digest(actual_hash, recomputed_hash):
                return {
                    "ok": False,
                    "checked": checked,
                    "broken_seq": actual_seq,
                    "reason": f"entry_hash mismatch at seq {actual_seq}",
                }

            checked += 1
            expected_prev = actual_hash
            expected_seq += 1

        return {"ok": True, "checked": checked, "broken_seq": None, "reason": "ok"}

    def list_entries(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List audit entries ordered by ``seq`` ascending."""
        with self._db.read_connection() as conn:
            rows = conn.execute(
                """
                SELECT seq, ts, actor, action, target_id, tier,
                       detail_json, prev_hash, entry_hash, sig_mode
                FROM audit_chain
                ORDER BY seq ASC
                LIMIT ? OFFSET ?
                """,
                (max(0, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> dict[str, Any]:
        try:
            detail = json.loads(row["detail_json"] or "{}")
        except json.JSONDecodeError:
            detail = {"_raw": str(row["detail_json"])}
        return {
            "seq": int(row["seq"]),
            "ts": str(row["ts"]),
            "actor": str(row["actor"]),
            "action": str(row["action"]),
            "target_id": str(row["target_id"]),
            "tier": str(row["tier"]),
            "detail": detail,
            "prev_hash": str(row["prev_hash"]),
            "entry_hash": str(row["entry_hash"]),
            "sig_mode": str(row["sig_mode"]),
        }


class MutationApprovalGate:
    """L0/L1/L2 approval gate safeguarding protected memory cards on supersede/revoke."""

    def __init__(
        self,
        db: ReijiDatabase,
        *,
        ledger: AuditChainLedger | None = None,
        signing_key: bytes | None = None,
        admin_actors: Iterable[str] = ("human_admin",),
    ) -> None:
        if not isinstance(db, ReijiDatabase):
            raise TypeError("db must be a ReijiDatabase instance")
        self._db = db
        ensure_security_tables(db)
        self._ledger = (
            ledger
            if ledger is not None
            else AuditChainLedger(db, signing_key=signing_key)
        )
        self._admin_actors: frozenset[str] = frozenset(
            str(a).strip() for a in admin_actors if isinstance(a, str) and a.strip()
        )

    @property
    def ledger(self) -> AuditChainLedger:
        return self._ledger

    def is_protected_card(self, card: MemoryCard) -> bool:
        """Return True if the card requires human approval before non-admin mutation."""
        if not isinstance(card, MemoryCard):
            raise TypeError("card must be a MemoryCard instance")
        return (
            card.domain == "companion"
            or card.card_type in PROTECTED_CARD_TYPES
            or bool(card.state.get("protected"))
        )

    def guard_supersede_or_revoke(
        self,
        cards: CardStore,
        *,
        actor: str,
        op_kind: str,
        target_card_id: str,
        replacement_card: MemoryCard | None = None,
        origin_trust: str = "assistant",
        poison_detected: bool = False,
        reason: str = "",
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        """Enforce L2 hard-reject, L1 human-approval queue, or L0 direct execution."""
        if not isinstance(cards, CardStore):
            raise TypeError("cards must be a CardStore instance")
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("actor must be a non-empty string")
        if not isinstance(target_card_id, str) or not target_card_id.strip():
            raise ValueError("target_card_id must be a non-empty string")

        actor_clean = actor.strip()
        op_clean = _normalize_op_kind(op_kind)
        trust_clean = str(origin_trust or "assistant").strip().lower()
        ts = _utc_now_iso() if now_iso is None else str(now_iso)

        if op_clean == "supersede":
            if not isinstance(replacement_card, MemoryCard):
                raise ValueError(
                    "replacement_card (MemoryCard) is required when op_kind='supersede'"
                )
            if replacement_card.card_id == target_card_id:
                raise ValueError("A card cannot supersede itself")
        elif replacement_card is not None:
            raise ValueError("replacement_card must be None when op_kind='revoke'")

        # L2: hard reject poisoned or untrusted external mutation attempts.
        if poison_detected or trust_clean in UNTRUSTED_ORIGINS:
            reasons: list[str] = []
            if poison_detected:
                reasons.append("poison_detected")
            if trust_clean in UNTRUSTED_ORIGINS:
                reasons.append(f"origin_trust={trust_clean}")
            block_reason = f"L2 forbidden mutation blocked ({', '.join(reasons)})"
            audit_entry = self._ledger.append(
                actor_clean,
                f"mutation_blocked:{op_clean}",
                target_id=target_card_id,
                tier=MutationTier.L2_FORBIDDEN.value,
                detail={
                    "op_kind": op_clean,
                    "origin_trust": trust_clean,
                    "poison_detected": bool(poison_detected),
                    "reason": block_reason,
                    "replacement_card_id": (
                        replacement_card.card_id if replacement_card is not None else None
                    ),
                },
                now_iso=ts,
            )
            return {
                "status": "rejected",
                "allowed": False,
                "tier": MutationTier.L2_FORBIDDEN.value,
                "op_kind": op_clean,
                "target_card_id": target_card_id,
                "review_id": None,
                "reason": block_reason,
                "audit_seq": audit_entry["seq"],
                "result_card": None,
            }

        target_card = cards.get_card(target_card_id)
        if target_card is None:
            raise KeyError(f"Card not found: {target_card_id}")

        # L1: protected card mutated by non-admin -> stage in mutation_reviews without mutating card.
        if self.is_protected_card(target_card) and actor_clean not in self._admin_actors:
            review_id = f"rev_{uuid4().hex[:12]}"
            replacement_json = (
                json.dumps(replacement_card.to_dict(), ensure_ascii=False, sort_keys=True)
                if replacement_card is not None
                else None
            )
            queue_reason = (
                reason.strip()
                if isinstance(reason, str) and reason.strip()
                else (
                    f"protected card ({target_card.domain}/{target_card.card_type}) "
                    "requires human approval"
                )
            )
            with self._db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO mutation_reviews (
                        review_id, status, tier, actor, op_kind, target_card_id,
                        replacement_card_json, origin_trust, reason, created_at
                    )
                    VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_id,
                        MutationTier.L1_APPROVAL.value,
                        actor_clean,
                        op_clean,
                        target_card_id,
                        replacement_json,
                        trust_clean,
                        queue_reason,
                        ts,
                    ),
                )
                audit_entry = self._ledger._append_in_tx(
                    conn,
                    actor=actor_clean,
                    action=f"mutation_queued:{op_clean}",
                    target_id=target_card_id,
                    tier=MutationTier.L1_APPROVAL.value,
                    detail={
                        "review_id": review_id,
                        "op_kind": op_clean,
                        "origin_trust": trust_clean,
                        "card_domain": target_card.domain,
                        "card_type": target_card.card_type,
                        "replacement_card_id": (
                            replacement_card.card_id
                            if replacement_card is not None
                            else None
                        ),
                        "reason": queue_reason,
                    },
                    now_iso=ts,
                )

            return {
                "status": "pending_approval",
                "allowed": False,
                "tier": MutationTier.L1_APPROVAL.value,
                "op_kind": op_clean,
                "target_card_id": target_card_id,
                "review_id": review_id,
                "reason": queue_reason,
                "audit_seq": audit_entry["seq"],
                "result_card": None,
            }

        # L0: auto-execute and record in audit chain.
        result_card: MemoryCard | None = None
        revoked = False
        if op_clean == "supersede":
            assert replacement_card is not None
            result_card = cards.supersede_card(
                target_card_id,
                replacement_card,
                now_iso=ts,
            )
        else:
            revoked = cards.revoke_card(target_card_id, now_iso=ts)

        audit_entry = self._ledger.append(
            actor_clean,
            f"mutation_executed:{op_clean}",
            target_id=target_card_id,
            tier=MutationTier.L0_AUTO.value,
            detail={
                "op_kind": op_clean,
                "origin_trust": trust_clean,
                "card_domain": target_card.domain,
                "card_type": target_card.card_type,
                "replacement_card_id": (
                    replacement_card.card_id if replacement_card is not None else None
                ),
                "revoked": revoked,
            },
            now_iso=ts,
        )
        return {
            "status": "executed",
            "allowed": True,
            "tier": MutationTier.L0_AUTO.value,
            "op_kind": op_clean,
            "target_card_id": target_card_id,
            "review_id": None,
            "reason": "auto-approved",
            "audit_seq": audit_entry["seq"],
            "result_card": result_card,
            "revoked": revoked,
        }

    def approve_review(
        self,
        review_id: str,
        cards: CardStore,
        *,
        approver: str = "human_admin",
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        """Atomically claim and execute a pending L1 mutation review and log to audit chain."""
        if not isinstance(review_id, str) or not review_id.strip():
            raise ValueError("review_id must be a non-empty string")
        if not isinstance(cards, CardStore):
            raise TypeError("cards must be a CardStore instance")
        if not isinstance(approver, str) or not approver.strip():
            raise ValueError("approver must be a non-empty string")

        approver_clean = approver.strip()
        ts = _utc_now_iso() if now_iso is None else str(now_iso)

        # Phase 1: Atomically claim pending review to prevent double execution.
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM mutation_reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Review not found: {review_id}")
            current_status = str(row["status"])
            if current_status != "pending":
                raise ValueError(
                    f"Review {review_id!r} cannot be approved from status {current_status!r}"
                )
            cur = conn.execute(
                """
                UPDATE mutation_reviews
                SET status = 'approving'
                WHERE review_id = ? AND status = 'pending'
                """,
                (review_id,),
            )
            if cur.rowcount != 1:
                raise ValueError(f"Review {review_id!r} is no longer pending")

            op_kind = _normalize_op_kind(str(row["op_kind"]))
            target_card_id = str(row["target_card_id"])
            requester = str(row["actor"])
            replacement_json = row["replacement_card_json"]

        replacement_card: MemoryCard | None = None
        if op_kind == "supersede":
            if not replacement_json:
                self._restore_pending_review(review_id)
                raise ValueError(f"Review {review_id!r} missing replacement_card_json")
            replacement_card = MemoryCard.from_dict(json.loads(str(replacement_json)))

        # Phase 2: Execute card mutation; restore pending status on failure.
        result_card: MemoryCard | None = None
        revoked = False
        try:
            if op_kind == "supersede":
                assert replacement_card is not None
                result_card = cards.supersede_card(
                    target_card_id,
                    replacement_card,
                    now_iso=ts,
                )
            else:
                revoked = cards.revoke_card(target_card_id, now_iso=ts)
                if not revoked:
                    raise KeyError(f"Card not found: {target_card_id}")
        except Exception:
            self._restore_pending_review(review_id)
            raise

        # Phase 3: Atomically finalize review status and append audit chain record.
        with self._db.transaction() as conn:
            conn.execute(
                """
                UPDATE mutation_reviews
                SET status = 'approved',
                    decided_at = ?,
                    decided_by = ?
                WHERE review_id = ? AND status = 'approving'
                """,
                (ts, approver_clean, review_id),
            )
            audit_entry = self._ledger._append_in_tx(
                conn,
                actor=approver_clean,
                action=f"mutation_approved:{op_kind}",
                target_id=target_card_id,
                tier=MutationTier.L1_APPROVAL.value,
                detail={
                    "review_id": review_id,
                    "requested_by": requester,
                    "op_kind": op_kind,
                    "replacement_card_id": (
                        replacement_card.card_id if replacement_card is not None else None
                    ),
                    "revoked": revoked,
                },
                now_iso=ts,
            )

        return {
            "status": "approved",
            "review_id": review_id,
            "tier": MutationTier.L1_APPROVAL.value,
            "op_kind": op_kind,
            "target_card_id": target_card_id,
            "approved_by": approver_clean,
            "decided_at": ts,
            "audit_seq": audit_entry["seq"],
            "result_card": result_card,
            "revoked": revoked,
        }

    def reject_review(
        self,
        review_id: str,
        *,
        reason: str = "",
        reviewer: str = "human_admin",
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        """Reject a pending L1 mutation review without mutating the target card."""
        if not isinstance(review_id, str) or not review_id.strip():
            raise ValueError("review_id must be a non-empty string")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("reviewer must be a non-empty string")

        reviewer_clean = reviewer.strip()
        reason_clean = str(reason or "").strip()
        ts = _utc_now_iso() if now_iso is None else str(now_iso)

        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM mutation_reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Review not found: {review_id}")
            current_status = str(row["status"])
            if current_status != "pending":
                raise ValueError(
                    f"Review {review_id!r} cannot be rejected from status {current_status!r}"
                )

            op_kind = str(row["op_kind"])
            target_card_id = str(row["target_card_id"])
            requester = str(row["actor"])

            conn.execute(
                """
                UPDATE mutation_reviews
                SET status = 'rejected',
                    decided_at = ?,
                    decided_by = ?,
                    decision_reason = ?
                WHERE review_id = ? AND status = 'pending'
                """,
                (ts, reviewer_clean, reason_clean, review_id),
            )
            audit_entry = self._ledger._append_in_tx(
                conn,
                actor=reviewer_clean,
                action=f"mutation_rejected:{op_kind}",
                target_id=target_card_id,
                tier=MutationTier.L1_APPROVAL.value,
                detail={
                    "review_id": review_id,
                    "requested_by": requester,
                    "op_kind": op_kind,
                    "reason": reason_clean,
                },
                now_iso=ts,
            )

        return {
            "status": "rejected",
            "review_id": review_id,
            "tier": MutationTier.L1_APPROVAL.value,
            "op_kind": op_kind,
            "target_card_id": target_card_id,
            "rejected_by": reviewer_clean,
            "reason": reason_clean,
            "decided_at": ts,
            "audit_seq": audit_entry["seq"],
        }

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        """Fetch a single review record by ID."""
        if not isinstance(review_id, str) or not review_id.strip():
            raise ValueError("review_id must be a non-empty string")
        with self._db.read_connection() as conn:
            row = conn.execute(
                "SELECT * FROM mutation_reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_review(row)

    def list_reviews(
        self,
        *,
        status: str | None = "pending",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List review records optionally filtered by status."""
        with self._db.read_connection() as conn:
            if status is None:
                rows = conn.execute(
                    """
                    SELECT * FROM mutation_reviews
                    ORDER BY created_at ASC, review_id ASC
                    LIMIT ?
                    """,
                    (max(0, int(limit)),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM mutation_reviews
                    WHERE status = ?
                    ORDER BY created_at ASC, review_id ASC
                    LIMIT ?
                    """,
                    (str(status), max(0, int(limit))),
                ).fetchall()
        return [self._row_to_review(row) for row in rows]

    def _restore_pending_review(self, review_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                """
                UPDATE mutation_reviews
                SET status = 'pending'
                WHERE review_id = ? AND status = 'approving'
                """,
                (review_id,),
            )

    @staticmethod
    def _row_to_review(row: sqlite3.Row) -> dict[str, Any]:
        replacement_raw = row["replacement_card_json"]
        return {
            "review_id": str(row["review_id"]),
            "status": str(row["status"]),
            "tier": str(row["tier"]),
            "actor": str(row["actor"]),
            "op_kind": str(row["op_kind"]),
            "target_card_id": str(row["target_card_id"]),
            "replacement_card": (
                json.loads(str(replacement_raw)) if replacement_raw else None
            ),
            "origin_trust": str(row["origin_trust"]),
            "reason": str(row["reason"] or ""),
            "created_at": str(row["created_at"]),
            "decided_at": None if row["decided_at"] is None else str(row["decided_at"]),
            "decided_by": None if row["decided_by"] is None else str(row["decided_by"]),
            "decision_reason": str(row["decision_reason"] or ""),
        }
